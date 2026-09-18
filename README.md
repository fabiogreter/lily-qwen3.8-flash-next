# lily-qwen3.8-flash-next

lily-qwen3.8-flash-next is a Metal inference server for Apple Silicon that
serves one model, Qwen3.8-Flash-Next. It is a fork of Perplexity's [lily](https://github.com/perplexityai/pplx-garden/tree/main/lily),
a compact Metal engine for Qwen3.6-35B-A3B that decoded about 30 % faster
than mlx-lm ([their write-up](https://www.perplexity.ai/hub/blog/optimizing-on-device-inference-for-apple-silicon)).
The fork ports that engine to Qwen3.8-Flash-Next's architecture and tunes it
for this one model on this class of machine: the model runs as hand-written
Metal kernels, decode steps are pipelined so the GPU does not wait for the
host, speculative decoding uses the model's own draft head, conversations
are cached across requests, forks and restarts, and on a machine with half
the memory the checkpoint needs it keeps the busiest experts resident and
reads the rest on demand.

Measured against Unsloth's llama.cpp fork with the same model, prompts and
machine, prefill is 1.6 to 3 times faster and decode 1.9 to 3.4 times
faster; the difference grows with context.

## Performance

M5 Max, 40-core GPU, 128 GB. Same prompts on both engines, cut from real
documentation and code, a fresh prompt for every run, 256 greedy tokens of
new text, medians of three interleaved repeats, each engine's own timings
over HTTP. This fork at commit `38d2642` (2026-09-18), llama.cpp on
2026-09-17.

| tokens per second | 4K context | 16K | 32K | 64K |
|---|---:|---:|---:|---:|
| **prefill** this fork | 1 435 | 1 756 | 1 864 | 1 651 |
| prefill llama.cpp | 887 | 882 | 713 | 550 |
| **decode** this fork, 2 drafts | 98 | 97 | 102 | 92 |
| decode llama.cpp, MTP 2 drafts | 52 | 44 | 37 | 27 |
| decode this fork, no drafts | 87 | 85 | 85 | 82 |
| decode llama.cpp, no drafts | 39 | 31 | 25 | 17 |

The fork's decode is nearly flat from 1K to 64K because the architecture allows
it and the sparse-attention kernels keep the cost of context at a few percent
of a step. Both engines were also run with three drafts per step; acceptance
fell to about 50 % and decode was slower than with two, so those rows are
left out. The speculative and prefill rows are medians over a seven-minute
series, and the first repeat of it, at the GPU's full 1 620 MHz, decoded
105 / 109 / 109 / 90 tok/s with two drafts: under sustained load the M5
Max lets the GPU clock sag to about 1 500 to 1 550 MHz, which costs
compute-bound passes a few percent and leaves memory-bound plain decode
untouched. It used to be worse: a windowless process loses its performance
envelope after about 90 seconds and ran at 1 240 MHz and 24 W, so the
server now holds a user-initiated, latency-critical activity assertion for
each request. The trace and the numbers are in the noise section of the
performance document.

The quantizations differ slightly: the fork's affine 4-bit with group 64 against
llama.cpp's UD-IQ4_XS. The llama.cpp MTP rows come from a build with a
one-line fix that the shipped one lacks. Method, noise band, the fixed
`lily-bench` matrix and the full record: [docs/performance.md](docs/performance.md).

### Smaller machines

The checkpoint is 104.6 GB, most of it the 68 GB of routed experts, and
the fork runs it on machines that cannot hold all of that. On a 64 GB
machine the engine keeps a usage-ranked two thirds of the experts on the
GPU and reads the rest from the checkpoint files as they are routed to,
sized automatically from physical memory; nothing changes on a machine
that fits it. Measured with the reads cold, as on a 64 GB machine, on an
8K real-text prompt: about 930 tok/s prefill and 55 to 65 tok/s plain
decode, against 2 250 and 87 with everything resident, with the same
tokens produced. Speculative decoding is off there because its extra trunk
passes cost more than they return. Nothing to configure: the server sizes
it from the machine's memory and keeps 12 GB or a sixth of it free for
everything else; `--memory-gb 64` plans for that much instead (also the
way to try the mode on a bigger machine), and an `expert-usage.json` next
to the checkpoint (the one measured for this model is
`tools/bench/expert-usage-qwen38-flash-next.json`) tells it which experts
to keep. Details, measurements and knobs:
[docs/low-ram-experts.md](docs/low-ram-experts.md); `lily-experts`
measures the expert usage that places them.

### MLX engines

No released mlx-lm runs this model. Several MLX-based engines ship their own
implementation of the architecture and publish numbers for an M5 Max:
[MTPLX](https://mtplx.com/benchmarks/) reports 79 tok/s at 9K and 61 tok/s
at 109K of context with its speculative path, 44 without;
[oMLX](https://github.com/jundot/omlx/releases) reports 58 to 70 tok/s with
its speculative path in its 0.7.0 development builds. We have not re-verified
either with our harness, and their quantizations and sampling settings differ
from the table above.

## The model

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is
Qwen's `qwen4_exp` preview architecture: 48 layers, 36
of them Gated DeltaNet with a fixed-size recurrent state and 12 sparse
attention with an indexer that picks 512 blocks per query, a 512-expert MoE
with 10 active, a four-stream gated residual, a 32 GB hashed n-gram
embedding, a multi-token-prediction head and a vision tower.

The server runs a 4-bit conversion of it, 98 GiB on disk, produced by
`tools/convert/convert_qwen38_flash_next.py` from the Hugging Face BF16
weights and published as
[fabiogreter/Qwen3.8-Flash-Next-lily-q4](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4).
The vision tower is kept in bf16, the draft head is quantized like the
trunk, and the n-gram table stays on disk and is read through the page cache
rather than uploaded. The layout is documented in
[docs/qwen38-flash-next-checkpoint-format.md](docs/qwen38-flash-next-checkpoint-format.md).

## Running it

You need an Apple GPU of family 10 or later (M5 and newer), macOS 26 for
Metal 4 and tensor operations, the Rust toolchain pinned by
`rust-toolchain.toml`, and 128 GB of unified memory to hold the whole
checkpoint, or 64 GB with the expert cache above (the n-gram table lives in
the page cache either way).

```sh
cargo build --release --locked

./target/release/lily --model ~/models/Qwen3.8-Flash-Next-lily-q4 \
  --bind 127.0.0.1:8000 --max-seq 131072
```

The server binds at once, answers `/health` with `503 loading` while the
model loads, and serves after about 30 seconds. `tools/service/lily-service.sh install`
turns it into a launchd agent that starts at login and unloads the model
after 30 idle minutes. `--help` lists every flag.

The API is OpenAI-compatible: `POST /v1/chat/completions` (streaming or
not) and `POST /v1/completions`, with sampling parameters, stop strings,
tools and tool calls, `reasoning_content`, `prompt_cache_key`, usage blocks
and `GET /v1/models`. OpenAI clients and agents work unchanged; opencode is
what the server is developed against. The quirks:

- **Images are base64 data URIs**, PNG or JPEG, up to eight per request, in
  user messages. The server never fetches URLs. An image is scaled to at
  most 2 megapixels and costs one prompt token per 32 x 32 pixels, so a
  1920 x 1080 screenshot is 2 040 tokens and answers in about three seconds.
  No video.
- **One request runs at a time.** Others wait in a queue (503 when it is
  full). There is no batching across requests.
- **Thinking is on by default**, as the model's template intends. Per
  request, `reasoning_effort` takes `none` to turn it off and `low`,
  `medium` or `high` to set the level (the template's default is high);
  `chat_template_kwargs` with `enable_thinking` and `reasoning_effort` works
  too. `--thinking` and `--reasoning-effort` set the server's defaults.
- **Every response carries a `timings` object**: prompt tokens, cached
  tokens, prefill and decode rates, draft acceptance. `GET /v1/timings` keeps
  the last 32. `tools/opencode-plugin-timings/` shows them in opencode.
- `max_tokens` beyond the context is clamped rather than refused, and
  `/health` reports `loading`, `ready`, `idle`, `reloading` or `recovering`,
  the last after a GPU fault the server recovers from on its own.

## Caching

Prompt state is kept in three places. A client notices them only through
`cached_tokens` in the usage block.

1. **Resident sessions.** Every conversation's state stays in GPU memory
   under a byte budget. Continuing it costs only the new tokens; editing,
   regenerating or branching forks a copy, so parallel conversations do not
   destroy each other's context.
2. **The disk tier.** Sessions evicted from GPU memory go to disk and come
   back in about a second per few gigabytes when their prefix returns. It
   survives restarts.
3. **Durable prefixes.** Two runs of the same agent share their preamble,
   the system prompt, tool schemas and repository instructions, and differ
   only from the user's message on. The second run cannot resume from the
   first, because the first run's checkpoint sits at the end of its whole
   prompt, past the point where the two diverge. When the shared part is at
   least 1 024 tokens long, the server writes it to disk as a durable entry, and
   every later run with the same preamble starts from there. Many tasks
   against the same repository pay for the preamble once.

Images are identified by their content, not by their placeholder tokens, so
two screenshots behind the same preamble never share cached state past the
image. The rules are in [docs/architecture.md](docs/architecture.md), "The
session cache".

## How the speed was achieved

A decode step reads about 4.4 GB of weights and state for one token, so it
is bandwidth-bound, and the work is about not wasting that bandwidth and not
waiting between steps. Prefill reads the weights once per 4 096-token chunk
and is compute-bound instead.

**What came from Perplexity.** The Metal kernel foundations and the shape of
the engine: 4-bit weight streaming kernels adapted from MLX, tensor-op GEMMs,
the MoE, Gated DeltaNet and attention kernels, a runtime shader compiler, and
a small greedy server with a token-prefix cache for one checkpoint. That
engine measured about 30 % over mlx-lm on its model.

**What this fork added:**

- The Qwen3.8-Flash-Next graph and its converter: sparse attention with the
  indexer, hyper-connections, the paged n-gram table, the draft head, the
  vision tower.
- A Metal 4 transport with one command buffer per step, level barriers
  instead of serial ones, decode steps parked on GPU events so the host is
  not on the critical path, and control flow such as the accepted draft
  count decided on the GPU.
- Speculative decoding through the model's own head, with the verify pass on
  small-row GEMMs written for it and exact speculative sampling when the
  request samples.
- Decode kernels taken to the ceiling their read sizes allow: the
  hyper-connection read from six dispatches to two, the sparse-attention
  selection with no serial steps, the split attention kernel with its
  latency chains cut, the expert gathers and the recurrent-state step tuned
  in a chained harness. A decode step is a chain of about 700 dependency
  levels, and the measurement of what such a chain can stream is what says
  the step is within 10 to 15 % of its floor.
- Prefill on the tensor ops where it was not: sparse attention over the
  gathered union of neighbouring queries' selections (from 40 % of a chunk to
  10 %), the Gated DeltaNet scan in chunked form instead of a token-serial
  recurrence, and the expert GEMM's last tiles at half and quarter height.
- The expert cache that runs the model on half the memory.
- A process activity assertion held while a request runs, so a server
  with no window keeps the GPU's performance envelope under sustained load
  and the machine can still sleep when idle.
- The session cache with recurrent-state checkpoints, forks, the disk tier
  and durable prefixes, and the full OpenAI request surface around it.

Part of this is specific to the model and part is not. The transport,
parking, GPU-side control flow, speculative decoding, the caching and the
expert cache are properties of the engine and apply to any model served from
any framework, MLX included. The sparse-attention and recurrent-state work
is specific to this architecture family, and the paged n-gram table to this
model alone. The detailed account, with every measurement, is
[docs/architecture.md](docs/architecture.md); what was done, what was tried
and dropped and why, and what is left but outside this project's target is
the second half of [docs/performance.md](docs/performance.md).

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
other flags and the reference harness that checks the engine against Hugging Face
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
