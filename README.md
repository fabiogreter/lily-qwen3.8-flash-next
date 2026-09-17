# lily

lily is a Metal inference server for Apple Silicon that serves one model,
Qwen3.8-Flash-Next, and is built around that one model and one machine. It
started as a fork of Perplexity's [lily](https://github.com/perplexityai/pplx-garden/tree/main/lily),
a compact Metal engine for Qwen3.6-35B-A3B that already decoded about 30 %
faster than mlx-lm ([their write-up](https://www.perplexity.ai/hub/blog/optimizing-on-device-inference-for-apple-silicon)).
This fork ported it to a new architecture and then kept going: every part of
the model runs as hand-written Metal kernels, the GPU never waits for the
host between tokens, the model's own draft head speculates, and the caches
remember every conversation across requests, forks and restarts.

Against the other engine that runs this model on a Mac, Unsloth's llama.cpp
fork, with the same model, prompts and machine, lily prefills 1.5 to 2.5
times faster and decodes 2 to 3.6 times faster, and the gap widens with
context.

## Performance

M5 Max, 40-core GPU, 128 GB. Same prompts on both engines, cut from real
documentation and code, a fresh prompt for every run, 256 greedy tokens of
new text, medians of three interleaved repeats, each engine's own timings
over HTTP.

| tokens per second | 4K context | 16K | 32K | 64K |
|---|---:|---:|---:|---:|
| **prefill** lily | 1 300 | 1 546 | 1 499 | 1 382 |
| prefill llama.cpp | 887 | 882 | 713 | 550 |
| **decode** lily, 2 drafts | 102 | 102 | 102 | 98 |
| decode llama.cpp, MTP 2 drafts | 52 | 44 | 37 | 27 |
| decode lily, no drafts | 86 | 86 | 84 | 82 |
| decode llama.cpp, no drafts | 39 | 31 | 25 | 17 |

lily's decode is nearly flat from 1K to 64K because the architecture allows
it and the sparse-attention kernels keep the cost of context at a few percent
of a step. Two drafts per step is the right setting on both engines: at three
the acceptance rate drops to about 50 % and the extra verify row costs more
than it returns.

Three things to know when reading the table. The quantizations differ
slightly, lily's affine 4-bit with group 64 against llama.cpp's UD-IQ4_XS.
The llama.cpp MTP figures come from a build with a one-line fix, because the
shipped Unsloth Studio build fails to load the draft head and silently runs
without it. And the 1K column is left out: at that length a prefill finishes
inside the GPU's clock ramp and measures the machine, not the engine. Method,
noise band, history and the remaining levers: [docs/performance.md](docs/performance.md).

## The model

Qwen3.8-Flash-Next is Qwen's `qwen4_exp` preview architecture: 48 layers, 36
of them Gated DeltaNet with a fixed-size recurrent state and 12 sparse
attention with an indexer that picks 512 blocks per query, a 512-expert MoE
with 10 active, a four-stream gated residual, a 32 GB hashed n-gram
embedding, a multi-token-prediction head and a vision tower.

lily runs a 4-bit conversion of it, 98 GiB on disk, produced by
`tools/convert/convert_qwen38_flash_next.py` from the Hugging Face BF16
weights and published as
[fabiogreter/Qwen3.8-Flash-Next-lily-q4](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4).
The vision tower is kept in bf16, the draft head is quantized like the
trunk, and the n-gram table stays on disk and is read through the page cache
rather than uploaded. The layout is documented in
[docs/qwen38-flash-next-checkpoint-format.md](docs/qwen38-flash-next-checkpoint-format.md).

## Running it

You need an Apple GPU of family 10 or later (M5 and newer), macOS 26 for
Metal 4 and tensor operations, 128 GB of unified memory, and the Rust
toolchain pinned by `rust-toolchain.toml`.

```sh
cargo build --release --locked

./target/release/lily --model ~/models/Qwen3.8-Flash-Next-lily-q4 \
  --bind 127.0.0.1:8000 --max-seq 131072
```

The server binds at once, answers `/health` with `503 loading` while the
model loads, and serves after about 30 seconds. `tools/service/lily-service.sh install`
turns it into a launchd agent that starts at login and unloads the model
after 30 idle minutes. `--help` lists every flag.

The API is OpenAI's: `POST /v1/chat/completions` (streaming or not) and
`POST /v1/completions`, with sampling parameters, stop strings, tools and
tool calls, `reasoning_content`, `prompt_cache_key`, usage blocks and
`GET /v1/models`. Any OpenAI client and agent works unchanged; opencode does.
The quirks:

- **Images are base64 data URIs**, PNG or JPEG, up to eight per request, in
  user messages. The server never fetches URLs. An image is scaled to at
  most 2 megapixels and costs one prompt token per 32 x 32 pixels, so a
  1920 x 1080 screenshot is 2 040 tokens and answers in about three seconds.
  No video.
- **One request runs at a time.** Others wait in a queue (503 when it is
  full). There is no batching across requests.
- **Thinking is on by default**, as the model's template intends. Turn it off
  per request with `reasoning_effort: "none"` or
  `chat_template_kwargs: {"enable_thinking": false}`.
- **Every response carries a `timings` object**: prompt tokens, cached
  tokens, prefill and decode rates, draft acceptance. `GET /v1/timings` keeps
  the last 32. `tools/opencode-plugin-timings/` shows them in opencode.
- `max_tokens` beyond the context is clamped rather than refused, and
  `/health` reports `loading`, `ready`, `idle`, `reloading` or `recovering`,
  the last after a GPU fault the server recovers from on its own.

## Caching

Three tiers, and a client sees them only as `cached_tokens`:

1. **Resident sessions.** Every conversation's state stays in GPU memory
   under a byte budget. Continuing it costs only the new tokens; editing,
   regenerating or branching forks a copy, so parallel conversations never
   destroy each other's context.
2. **The disk tier.** Sessions evicted from GPU memory go to disk and come
   back in about a second per few gigabytes when their prefix returns. It
   survives restarts.
3. **Durable prefixes.** Agent runs share a long preamble, the system
   prompt, tool schemas and repository instructions, and differ only from the
   user's message on. When a prompt agrees with a cached one for at least
   1 024 tokens beyond where it could resume, that shared prefix is written
   once as a durable entry, and every later run with the same preamble starts
   from it. Running many tasks in the same repository pays the preamble once.

Images are identified by their content, not by their placeholder tokens, so
two screenshots behind the same preamble never share cached state past the
image. The rules are in [docs/architecture.md](docs/architecture.md), "The
session cache".

## How the speed was achieved

The frame is simple: a decode step has to read about 4.4 GB of weights and
state for one token, so it is bandwidth-bound, and everything is about not
wasting that bandwidth and not waiting between steps. Prefill reads the
weights once per 4 096-token chunk and is compute-bound instead.

**What came from Perplexity.** The Metal kernel foundations and the shape of
the engine: 4-bit weight streaming kernels adapted from MLX, tensor-op GEMMs,
the MoE, Gated DeltaNet and attention kernels, a runtime shader compiler, and
a small greedy server with a token-prefix cache for one checkpoint. That
engine was the 30 % over mlx-lm.

**What this fork added**, in the two weeks since the import:

- The Qwen3.8-Flash-Next graph and its converter: sparse attention with the
  indexer, hyper-connections, the paged n-gram table, the draft head, the
  vision tower.
- A Metal 4 transport with one command buffer per step, level barriers
  instead of serial ones, decode steps parked on GPU events so the host is
  never on the critical path, and control flow such as the accepted draft
  count decided on the GPU.
- Speculative decoding through the model's own head, with the verify pass on
  purpose-built small-row GEMMs.
- Kernel fusion where the profile said so: the hyper-connection read from six
  dispatches to two, the sparse-attention selection with no serial steps,
  tiled sparse attention for prefill.
- The session cache with recurrent-state checkpoints, forks, the disk tier
  and durable prefixes, and the full OpenAI request surface around it.

Most of this transfers. The transport, parking, GPU-side control flow,
fusion, speculative decoding and the caching are engine properties and would
serve any model, in any framework including MLX. The sparse-attention and
recurrent-state work belongs to this architecture family, and the paged
n-gram table to this model alone. The detailed account, with every
measurement, is [docs/architecture.md](docs/architecture.md); what was
tried and what remains is [docs/optimization-potential.md](docs/optimization-potential.md)
and the dated reports in `docs/`.

## Converting a checkpoint

The converter needs a Python environment with `mlx`, `safetensors`, `numpy`,
`torch`, `torchvision`, `pillow` and `transformers`, and a Metal device.

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python mlx safetensors numpy torch torchvision pillow \
    "transformers @ git+https://github.com/huggingface/transformers"

.venv/bin/python tools/convert/convert_qwen38_flash_next.py \
    --src ~/models/Qwen3.8-Flash-Next --dst ~/models/Qwen3.8-Flash-Next-lily-q4
```

The full conversion takes about a minute on an M5 Max. `--layers 4` writes
the small checkpoint the tests use. [tools/README.md](tools/README.md) has the
other flags and the reference harness that checks lily against Hugging Face
transformers on the same weights.

## Tests

```sh
cargo test --locked
```

runs the kernel tests against CPU references, the shader compilation test and
the API tests; they need a Metal device but no checkpoint. Tests that need a
checkpoint are ignored by default and take `LILY_MODEL_DIR_FLASH`; the
four-layer conversion is enough for all of them. [CONTRIBUTING.md](CONTRIBUTING.md)
has the gates and the two things to know before trusting a green run.

## License

Apache-2.0. `NOTICE` records the upstream copyright and the MLX and MLX-LM
code the kernels were adapted from. The model and its conversion are under
Qwen's licence, stated on the checkpoint's card.
