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

There are two checkpoints, both mostly 4-bit: **q4-xl**, recommended,
keeps the dense tensors every token passes through at 8 bits and is
noticeably more reliable on long agent turns; **q4** is faster, by a few
percent for a request decoding alone and about 15 % for batched ones.
[The model](#the-model) has the details.

## Running it

You need an M5 or newer (Apple GPU family 10), macOS 26, the Rust toolchain
pinned by `rust-toolchain.toml`, and 128 GB of unified memory, or 64 GB with
the expert cache (see [Smaller machines](#smaller-machines)).

```sh
hf download fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl \
  --local-dir ~/models/Qwen3.8-Flash-Next-lily-q4-xl

cargo build --release --locked

./target/release/lily --model ~/models/Qwen3.8-Flash-Next-lily-q4-xl \
  --bind 127.0.0.1:8000 --max-seq 131072
```

The server answers `/health` with `503 loading` while the model loads and
serves after about 25 seconds, while the n-gram table keeps loading in the
background. `--help` lists every flag. The server can also be run as a
launchd service, see here: [tools/service/README.md](tools/service/README.md).

The API is OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`,
`/v1/models`, with tools and `reasoning_content`) and developed against
opencode. Up to `--max-batch` requests (4 by default) decode together.
Thinking is on by default, with thinking controls that keep the model from
reasoning without end: a thinking budget (8 000 tokens at `low` and `medium`
effort, 16 000 at `xhigh`) with nudges before it, and a tool call drawn
inside the reasoning ends it. Every response carries a `timings` object with
prefill and decode rates, cached tokens and draft acceptance;
`tools/opencode-plugin-timings/` shows them in opencode. The API's details,
batching and the thinking controls: [docs/server.md](docs/server.md).

## Performance

M5 Max, 40-core GPU, 128 GB, over HTTP with real-text prompts (this
repository's documentation and code), 256 greedy tokens of new text,
medians of three repeats, 2026-10-07. oMLX's rows are its own published
numbers for its oQ4e conversion on an M5 Max 128 GB, from the chart in the
[0.7.0 release notes](https://github.com/jundot/omlx/releases/tag/v0.7.0),
not measured by us; the chart does not state its sampling or whether its
generation used MTP.

| tokens per second | 4K context | 16K | 32K | 64K |
|---|---:|---:|---:|---:|
| **prefill** q4-xl | 2 219 | 2 370 | 2 373 | 2 180 |
| prefill q4 | 2 158 | 2 208 | 2 240 | 2 205 |
| prefill oMLX 0.7.0, published | 2 768 | 2 844 | | 2 366 |
| **decode** q4-xl, 2 drafts | 101 | 101 | 94 | 94 |
| decode q4, 2 drafts | 103 | 100 | 99 | 99 |
| decode q4-xl, no drafts | 73 | 73 | 69 | 68 |
| decode q4, no drafts | 87 | 82 | 85 | 79 |
| generation oMLX 0.7.0, published | 93.0 | 87.1 | | 75.9 |

A request decoding alone uses two drafts per step from the model's own
draft head; batched requests decode without drafts. Decode loses little from
4K to 64K because the sparse-attention kernels keep the cost of context to a
small part of a step.

In actual use with opencode, prefill is often slower than this, because most
turns are short and fixed per-request costs dominate. Decode is typically
faster: agent output (tool calls, paths, code repeated from the context) is
easy for the draft head to predict, so more drafts are accepted than on
these prompts, and decode has stayed above 100 tok/s beyond 200K tokens
of context, because a step grows only slowly with context.
Method, noise and the full record: [docs/performance.md](docs/performance.md).

## The model

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is
Qwen's `qwen4_exp` preview architecture: 48 layers, 36 of them Gated
DeltaNet with a fixed-size recurrent state and 12 sparse attention with an
indexer that picks 512 blocks per query, a 512-expert MoE with 10 active, a
four-stream gated residual, a 32 GB hashed n-gram embedding, a
multi-token-prediction head and a vision tower.

The server runs either of two conversions of it, made by
`tools/convert/convert_qwen38_flash_next.py` from the BF16 weights with the
vision tower kept in bf16. Both keep the routed experts and the n-gram
tables, about 95 % of the bytes, at 4 bits:

- **[q4-xl](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl)**
  (recommended, 106.9 GB) keeps attention, the shared expert, the LM head
  and the embedding at 8 bits. On long agent turns the 4-bit model more
  often announces its next step and ends the turn without making the tool
  call; on a captured opencode turn, sampled 30 times each with the thinking
  controls on, 23 of 30 samples went on to a tool call on q4-xl against 18
  on q4. Of the 8-bit sets tried, the attention tensors carried the
  difference. The draft head keeps a 4-bit path, so drafting costs what it
  does on q4 while the trunk, which verifies every draft, decides the output.
- **[q4](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4)**
  (105.5 GB) is 4-bit apart from a small set of routing and mixing
  tensors, and is faster when requests decode without drafts, for example
  several at once (see [Performance](#performance)).

This is one prompt, not a task-level evaluation. The layout and the 8-bit
groups are documented in
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

## Smaller machines

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

Add `--q4-xl` for the q4-xl checkpoint (`--dst
~/models/Qwen3.8-Flash-Next-lily-q4-xl`). It takes about a minute on an M5
Max. `--layers 4` writes the small checkpoint the tests use. [tools/README.md](tools/README.md) has the other
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
