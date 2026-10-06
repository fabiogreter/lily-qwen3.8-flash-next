# lily-qwen3.8-flash-next

lily-qwen3.8-flash-next is a Metal inference server for Apple Silicon that
serves one model, Qwen3.8-Flash-Next. It is a fork of Perplexity's [lily](https://github.com/perplexityai/pplx-garden/tree/main/lily),
a compact Metal engine for Qwen3.6-35B-A3B that decoded about 30 % faster
than mlx-lm ([their write-up](https://www.perplexity.ai/hub/blog/optimizing-on-device-inference-for-apple-silicon)).

This fork ports that engine to Qwen3.8-Flash-Next and tunes it for this
model: hand-written Metal kernels, pipelined decode steps, speculative
decoding with the model's own draft head, conversations cached across
requests and restarts, and an expert cache that runs the model on half the
memory it needs.

Against Unsloth's llama.cpp fork on the same machine and prompts, prefill is
2.7 to 4.2 times faster and decode 2.1 to 3.6 times faster, more so at longer
context.

## Running it

You need an M5 or newer (Apple GPU family 10), macOS 26, the Rust toolchain
pinned by `rust-toolchain.toml`, and 128 GB of unified memory, or 64 GB with
the expert cache (see [Smaller machines](#smaller-machines)).

```sh
cargo build --release --locked

./target/release/lily --model ~/models/Qwen3.8-Flash-Next-lily-q4 \
  --bind 127.0.0.1:8000 --max-seq 131072
```

The checkpoint is on Hugging Face as
[fabiogreter/Qwen3.8-Flash-Next-lily-q4](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4).
The server answers `/health` with `503 loading` while the model loads and
serves after about 25 seconds, while the n-gram table keeps loading in the
background. `--help` lists every flag. The server can also be run as a
launchd service, see here: [tools/service/README.md](tools/service/README.md).

The API is OpenAI-compatible: `POST /v1/chat/completions` (streaming or
not), `POST /v1/completions` and `GET /v1/models`, with tools,
`reasoning_content` and `prompt_cache_key`. It is developed against
opencode. The differences:

- **Images** are base64 data URIs (PNG or JPEG, no count limit beyond the
  context); the server never fetches URLs. A 1920 x 1080 screenshot costs
  about 2 000 prompt tokens. No video.
- **Message text is always text.** A special token spelled in a message, a
  tool result or a tool definition (`<|im_end|>`, `<|image_pad|>`) is
  tokenized as the characters it is, never as conversation structure or an
  image placeholder; only the chat template's own markup is special.
  `/v1/completions` takes its raw prompt as written, special tokens
  included.
- **Up to four requests decode together** (`--max-batch`, continuous
  batching): a request that arrives while another decodes gets its prefill
  between the other's steps instead of waiting for its answer, then the two
  share each decode step. A request decoding alone keeps speculative
  decoding. Prefills run one at a time; more requests queue, with 503 when
  the queue is full. `--max-batch 1` serves one request at a time.
- **Thinking is on by default.** `reasoning_effort` (`none`, `low`,
  `medium`, `high`) sets it per request; `--thinking` and
  `--reasoning-effort` set the server's default.
- **Thinking controls**, all off by default (experimental; the inserted
  texts are untested as prompts). Per request, at the top level or in
  `chat_template_kwargs` (which wins):
  - `thinking_budget` (tokens; negative turns a server default off): once
    the reasoning block holds that many of the model's tokens it is closed
    at the next line end (after a grace window also a sentence end, after
    a second one anywhere), never inside a code fence or a tool call, with
    a short preface and `</think>`.
  - `thinking_nudges` (bool, with a budget): sentences of increasingly firm
    wording inserted into the reasoning at 50, 75 and 90 % of the budget.
  - `tool_call_ends_thinking` (bool, chat requests with tools): a
    `<tool_call>` at a line start inside the reasoning block ends it, with
    `</think>` inserted in front of it.

  Inserted tokens are fed to the model like generated ones, appear in the
  stream (the texts in `reasoning_content`) and count as completion
  tokens. The server's defaults for chat: `--thinking-budget
  low=4000,medium=8000,xhigh=16000` (by the template's reasoning effort,
  `xhigh` when unset; or one number for all),
  `--thinking-budget-tool-turn-factor` (scales it after a tool result),
  `--thinking-nudges`, `--tool-call-ends-thinking`,
  `--thinking-budget-grace` and `--thinking-texts` (a JSON file with the
  texts). `/v1/completions` takes only the request fields, and only for a
  prompt that ends with `<think>\n`.
- **Every response carries a `timings` object** with prefill and decode
  rates, cached tokens and draft acceptance. `GET /v1/timings` keeps the
  last 32; `tools/opencode-plugin-timings/` shows them in opencode.
- `max_tokens` beyond the context is clamped, not refused.

## Performance

M5 Max, 40-core GPU, 128 GB. Both engines over HTTP with the same real-text
prompts, 256 greedy tokens, medians of three interleaved repeats. This fork
at commit `c096b75` (2026-10-01); the llama.cpp rows are from 2026-09-17 and
were not re-measured.

| tokens per second | 4K context | 16K | 32K | 64K |
|---|---:|---:|---:|---:|
| **prefill** this fork | 2 387 | 2 534 | 2 495 | 2 335 |
| prefill llama.cpp | 887 | 882 | 713 | 550 |
| **decode** this fork, 2 drafts | 110 | 105 | 105 | 97 |
| decode llama.cpp, MTP 2 drafts | 52 | 44 | 37 | 27 |
| decode this fork, no drafts | 85 | 84 | 83 | 77 |
| decode llama.cpp, no drafts | 39 | 31 | 25 | 17 |

Decode loses about a tenth from 4K to 64K, where llama.cpp loses half,
because the sparse-attention kernels keep the cost of context to a small
part of a step. Three drafts per step were slower than two on both
engines. Under sustained load the GPU clock sags a few percent, which slows
the speculative and prefill rows slightly.

During actual use with opencode, prefill is often quite a bit slower, due to
most turns being short, so that fixed per-request costs dominate.

Decode in practice is typically faster, because the speculative decoding gets
more hits than in the synthetic tests: usually still >100 tok/s at 220-230K
context. One day of opencode sessions on a single project (339 requests,
2026-09-30, server log) accepted 74 to 82% of the drafts at every context
length, against 59 to 64% on the synthetic prompts:

| context | requests | decode tok/s | drafts accepted | ms per step |
|---|---:|---:|---:|---:|
| 32-64K | 72 | 113 | 82% | 23.6 |
| 64-128K | 148 | 101 | 74% | 24.8 |
| 128-192K | 89 | 103 | 80% | 25.4 |
| 192-270K | 29 | 106 | 80% | 24.7 |

Acceptance does not grow with context; decode stays flat because a step is
only about 5% slower at 250K than at 50K. Agent output (tool calls, paths,
code repeated from the context) is easy for the draft head to predict. Other
projects and tasks will accept more or fewer drafts.

The quantizations for the tests differ slightly (affine 4-bit, group 64, against
UD-IQ4_XS), and the llama.cpp MTP rows use a one-line fix the shipped build
lacks. Method, noise and the full record: [docs/performance.md](docs/performance.md).

### Smaller machines

On smaller machines, lily automatically employs an expert cache that
keeps only the most used experts in GPU memory and streams the rest from
disk. Expert usage is tracked during runtime, so the cache adapts to your
usage. The cached configuration then gets saved to disk and reused on the
next run.

Moving experts around between RAM and disk does of course cost time. On a
simulated 64 GB machine, prefill was measured at about 930 tok/s, and decode
at around 50-65 tok/s. Speculative decoding is off in this mode, as it requires
additional expert reads and thus slows down the process. Note that in practice,
performance may be worse, as there is additional RAM required to hold the session
KV cache. If anyone wants to test with a real 64 GB machine, your feedback is welcome.
Less than 64 GB is probably impractical.

The cache sizes itself from physical memory; `--memory-gb 64` plans for
64 GB instead, which is also how to try the mode on a bigger machine.
Details: [docs/low-ram-experts.md](docs/low-ram-experts.md).

### MLX engines

At the time of testing, no released official mlx-lm ran this model. MLX-based
engines with their own implementation publish M5 Max numbers:

- [MTPLX](https://mtplx.com/benchmarks/) 79 tok/s at 9K and 61 at 109K with
its speculative path (44 without)
- [oMLX 0.7.0](https://github.com/jundot/omlx/releases/tag/v0.7.0) (released
2026-09-30), Qwen3.8-Flash-Next oQ4e on an M5 Max 128 GB, from the chart in
its release notes:

| tokens per second | 4K context | 16K | 64K |
|---|---:|---:|---:|
| prefill oMLX 0.7.0, published | 2 768 | 2 844 | 2 366 |
| prefill this fork | 2 387 | 2 534 | 2 335 |
| generation oMLX 0.7.0, published | 93.0 | 87.1 | 75.9 |
| decode this fork, 2 drafts | 110 | 105 | 97 |
| decode this fork, no drafts | 85 | 84 | 77 |

The oMLX rows are its own published numbers, not measured by us, with a
different quantization (oQ4e) and sampling the chart does not state (the
benchmarks in its pull requests used temperature 1.0; ours are greedy).
This fork's rows are the table above, prefill with the draft head loaded as
shipped; without it (`--mtp-drafts 0`) prefill measured 2 582 / 2 647 /
2 448. We have not run MTPLX or oMLX ourselves.

## The model

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is
Qwen's `qwen4_exp` preview architecture: 48 layers, 36 of them Gated
DeltaNet with a fixed-size recurrent state and 12 sparse attention with an
indexer that picks 512 blocks per query, a 512-expert MoE with 10 active, a
four-stream gated residual, a 32 GB hashed n-gram embedding, a
multi-token-prediction head and a vision tower.

The server runs a 4-bit conversion of it (98 GiB), made by
`tools/convert/convert_qwen38_flash_next.py` from the BF16 weights, with the
vision tower kept in bf16. The layout is documented in
[docs/qwen38-flash-next-checkpoint-format.md](docs/qwen38-flash-next-checkpoint-format.md).

## Exactness

The fork is not logit-identical to other engines, and for this model no two
bf16 engines are: small rounding differences flip near-tie choices in its
expert router and sparse attention. On the same 4-bit weights as the MLX
port, the fork's output differs from MLX's by a mean KL of about 2e-3 from
1K to 32K of context, as much as MLX differs from itself when only its
prefill chunk size changes. Method and numbers:
[tools/README.md](tools/README.md#referencemlx_paritysh).

## Caching

Prompt state is cached in three tiers; clients see them only as
`cached_tokens` in the usage block.

1. **Resident sessions.** Each conversation's state stays in GPU memory
   within a budget, so continuing it costs only the new tokens. A re-sent
   last turn (a regenerated or re-tokenized answer) rewinds the session in
   place; edits further back and other conversations fork a copy instead
   of overwriting it. Long answers carry checkpoints of their own, so a
   prompt that diverges inside one resumes near the divergence.
2. **Disk.** Sessions evicted from GPU memory move to disk and come back in
   about a second per few gigabytes. They survive restarts.
3. **Durable prefixes.** Runs of the same agent share a preamble (system
   prompt, tool schemas, repository instructions) and diverge at the user's
   message, so no run can resume another's checkpoint. When the shared part
   runs at least 1 024 tokens past where a run could resume, the server
   stores it separately and every later
   run starts from there.

Images are identified by their content, so two different screenshots never
share cached state. The rules: [docs/architecture.md](docs/architecture.md),
"The session cache".

## How the speed was achieved

A decode step reads about 4.4 GB of weights and state per token, so decode
is bandwidth-bound. Prefill reads the weights once per 4 096-token chunk and
is compute-bound.

**From Perplexity:** the Metal kernel foundations (4-bit weight streaming
adapted from MLX, tensor-op GEMMs, the MoE, Gated DeltaNet and attention
kernels), the runtime shader compiler, and a small server with a
token-prefix cache.

**Added in this fork:**

- The Qwen3.8-Flash-Next graph and its converter: sparse attention with the
  indexer, hyper-connections, the paged n-gram table, the draft head, the
  vision tower.
- A Metal 4 transport: one command buffer per step, level barriers, decode
  steps parked on GPU events so the host is off the critical path, and
  control flow such as the accepted draft count decided on the GPU.
- Speculative decoding with the model's own head, exact when the request
  samples.
- Decode kernels tuned until a step runs within 10 to 15 % of what its
  chain of reads can stream (the bandwidth limit, accounting for ramp times).
- Prefill on the tensor ops: sparse attention over the union of neighbouring
  queries' selections, a chunked Gated DeltaNet scan, and better tiling for
  the expert GEMM.
- The expert cache for half-memory machines.
- An activity assertion per request, so a server with no window keeps its
  GPU performance under sustained load.
- The session cache and the full OpenAI request surface.

The transport, parking, speculative decoding, caching and expert cache
apply to any model and framework, MLX included; the sparse-attention and
recurrent-state work is specific to this architecture family. The detailed
account is [docs/architecture.md](docs/architecture.md); what was tried,
dropped and why is in [docs/performance.md](docs/performance.md).

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

It takes about a minute on an M5 Max. `--layers 4` writes the small
checkpoint the tests use. [tools/README.md](tools/README.md) has the other
flags and the reference harnesses.

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
