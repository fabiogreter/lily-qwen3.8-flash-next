# lily

lily is a Metal inference server for Apple Silicon. It loads one quantized
checkpoint, runs the whole model as hand-written Metal kernels, and serves it
over an OpenAI-compatible HTTP API. The kernels compile from source at
runtime; there is no offline shader build step.

It is a fork of Perplexity's
[pplx-garden/lily](https://github.com/perplexityai/pplx-garden/tree/main/lily),
imported at commit `1ed972e` (2026-09-02). The fork adds a second model
architecture and the server around it:

| `model_type`  | checkpoint                                                  | status                   |
|---------------|-------------------------------------------------------------|--------------------------|
| `qwen3_5_moe` | Qwen3.6-35B-A3B, MLX affine 4-bit (upstream's target)       | unchanged graph, new API |
| `qwen4_exp`   | Qwen3.8-Flash-Next, lily's own `qwen4_exp-affine-v1` layout | this fork                |

Upstream's own account of the engine this fork started from: Perplexity,
[Optimizing On-Device Inference for Apple Silicon](https://www.perplexity.ai/hub/blog/optimizing-on-device-inference-for-apple-silicon).

How the engine works: [docs/architecture.md](docs/architecture.md). What it
measures: [docs/performance.md](docs/performance.md).

## The model it serves

Qwen3.8-Flash-Next is Qwen's `qwen4_exp` preview architecture: 48 layers of
Gated DeltaNet / Qwen Sparse Attention (3:1) with a 512-expert MoE, a 4-stream
gated residual (hyper-connections), a hashed n-gram embedding at layer 2 with
a 51B-parameter table, and a QSA indexer that selects 512 blocks of 4 tokens
per query once the context exceeds 2 051 tokens. A separate
multi-token-prediction head drives speculative decoding.

The checkpoint is produced by `tools/convert/convert_qwen38_flash_next.py`
from the raw Hugging Face BF16 weights: affine 4-bit / group 64 for experts,
attention, GDN, shared expert and embeddings, 4-bit / group 32 for the n-gram
table, 8-bit for routers, gates and the hyper-connection mixers. The full
checkpoint is 103 GB on disk, plus 1.5 GB for the draft head. Of that, 71 GB
is uploaded to the GPU and the 32 GB n-gram table stays in the page cache.
The layout is written out in
[docs/qwen38-flash-next-checkpoint-format.md](docs/qwen38-flash-next-checkpoint-format.md).

### What is not supported

- **Image input.** The converter now keeps the vision tower and the engine
  loads it (`--vision`), but nothing runs it yet: the API still rejects image
  content. Vision support is in progress (`docs/vision-support-plan.md`).
- **Batch size 1.** One generation runs at a time; further requests queue.
  There is no batching across requests.
- **Greedy drafts only.** The draft head proposes with argmax, so a request
  that samples with temperature accepts fewer drafts and gains less from
  speculation than a greedy one.
- **macOS 26 and Metal 4.** The engine submits through the Metal 4 command
  queue and the GEMMs use Metal tensor operations. Older systems and Apple
  GPUs before family 10 cannot run it.
- **Other checkpoints.** lily validates the exact architecture and
  quantization layout at load time. Dense Qwen checkpoints, smaller Qwen
  checkpoints, BF16 checkpoints, GGUF, AWQ, GPTQ, int8 and fp8 are rejected.
- **Constrained decoding.** `response_format` other than `text`, `n > 1` and
  `logprobs` are rejected.

## Performance

On an M5 Max (40-core GPU, 128 GB), greedy, synthetic prompts, median of
three repeats:

| tok/s                  | 1K prompt | 8K prompt | 32K prompt |
|------------------------|-----------|-----------|------------|
| prefill                | 1 825     | 1 396     | 1 195      |
| decode, no drafts      | 86.8      | 79.7      | 75.6       |
| decode, 2 drafts       | 116.9     | 98.1      | 70.9       |

Draft acceptance on this prompt is 73% at 1K and 64% at 8K; on real code and
prose it has measured 66 to 98%. The 1K prefill column and the 32K
speculative cell both carry caveats. Method, noise band, the comparison
against mlx-lm and the remaining levers are in
[docs/performance.md](docs/performance.md).

## Requirements

- Apple GPU family 10 or later (M5 and newer)
- macOS 26 or later: the engine submits through the Metal 4 command queue
  (`MTL4CommandQueue`, argument tables, residency sets) and the GEMMs use
  Metal tensor operations
- Rust 1.92, pinned by `rust-toolchain.toml` (this repo drives it through
  asdf's `rustup`; `.tool-versions` names the asdf shim version)
- A converted checkpoint (see below), or upstream's
  `mlx-community/Qwen3.6-35B-A3B-4bit` at revision
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`
- About 103 GB of memory for the full Qwen3.8-Flash-Next checkpoint, plus the
  session cache budget: 71 GB of GPU-resident weights and 32 GB of page cache
  for the n-gram table

## Converting a checkpoint

The converter needs a Python environment with `mlx`, `safetensors`, `numpy`,
`torch` and `transformers`, and a Metal device (`mlx` needs one even for CPU
arrays). `--dry-run` never imports `mlx`.

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python mlx safetensors numpy torch \
    "transformers @ git+https://github.com/huggingface/transformers"

.venv/bin/python tools/convert/convert_qwen38_flash_next.py \
    --src ~/models/Qwen3.8-Flash-Next \
    --dst ~/models/Qwen3.8-Flash-Next-lily-q4
```

The full 48-layer conversion takes about a minute on an M5 Max and writes
103.1 GB. `--layers N` truncates the model, which is how the test checkpoints
are made (`--layers 4` writes 38.6 GB). `--mtp-only` appends the 1.5 GB
draft head to an existing conversion; `--no-mtp` leaves it out. See
[tools/README.md](tools/README.md) for the other flags and
[docs/qwen38-flash-next-checkpoint-format.md](docs/qwen38-flash-next-checkpoint-format.md)
for what it writes.

## Building

```sh
cargo build --release --locked
```

## Running the server

```sh
./target/release/lily \
  --model ~/models/Qwen3.8-Flash-Next-lily-q4 \
  --bind 127.0.0.1:8000 \
  --max-seq 131072
```

The server binds immediately (`/health` answers `503 loading` until the model
is up), loads the checkpoint, preloads the n-gram table into the page cache,
compiles its shaders during a warm-up, and then serves:

- `POST /v1/chat/completions` (streaming and not)
- `POST /v1/completions`
- `GET /v1/models`
- `GET /health`

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-Flash-Next",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

### Request surface

Chat requests take text messages (string content or `type: text` parts) with
roles `system`, `user`, `assistant` (with `tool_calls` and `reasoning_content`)
and `tool`; `tools` and `tool_choice` (`none` hides the tools, anything else
renders them); `temperature`, `top_p`, `top_k`, `min_p`, `seed`,
`presence_penalty`, `frequency_penalty`, `repetition_penalty` (penalties
apply to generated tokens); `stop` strings; `max_tokens` /
`max_completion_tokens` (default: the rest of the context; a larger value is
clamped to it, as OpenAI-compatible servers do, and the response then ends
with `finish_reason: "length"`); `stream_options.include_usage`;
`reasoning_effort` (`none` disables thinking, `low`, `medium`, `high`);
`chat_template_kwargs` with `enable_thinking`, `reasoning_effort`,
`preserve_thinking`; and `prompt_cache_key` as a cache hint. Sampling
defaults come from the checkpoint's `generation_config.json` (temperature
1.0, top_k 20, top_p 0.95 for Qwen3.8) and can be overridden on the command
line. `top_k` above 1 024 or unset is capped at 1 024 candidates.

Reasoning comes back as `reasoning_content` (in `message` or in stream
deltas). Tool calls are parsed from the model's `<tool_call><function=…>`
XML, typed by the tool's JSON schema, and returned as OpenAI `tool_calls`
with `finish_reason: "tool_calls"`. The final message must be `user` or
`tool`.

Rejected with 400: images, `n > 1`, `logprobs`, `response_format` other than
`text`, `echo`, and a prompt that fills the whole context (`prompt exceeds
the server context: N prompt tokens, M tokens of context`). Every rejected
request leaves a `rejected POST /v1/chat/completions with 400: ...` line in
the log, and a clamped `max_tokens` a `warning:` line. Requests wait in a
bounded queue (`--queue`, 503 when full) and run one at a time; a client
that disconnects cancels its generation at the next token.

Usage blocks report `prompt_tokens_details.cached_tokens` for session-cache
reuse and `completion_tokens_details` with accepted and rejected drafts.

### Timings

Every response carries a `timings` object next to `usage`: the same numbers
the per-request log line prints, as JSON. It is lily's own extension, so a
client that ignores the field sees exactly the response it saw before.

| field | meaning |
|-------|---------|
| `prompt_tokens` | every token of the rendered prompt |
| `cached_tokens` | prompt tokens the session cache supplied (resident prefix, fork, or a checkpoint restored from disk) |
| `prefill_tokens` | prompt tokens actually run through the model: `prompt_tokens - cached_tokens` |
| `prefill_ms` | the `prefix` figure of the log line: the cache lookup, a disk restore if there was one, the prefill and the checkpoint |
| `prefill_per_second` | `prefill_tokens` per second, `null` when nothing was prefilled (never over the whole prompt, which a cache hit would inflate) |
| `generated_tokens`, `decode_ms`, `decode_per_second` | the decode loop, rate `null` when nothing was generated |
| `drafted_tokens`, `accepted_tokens`, `acceptance_ratio` | speculative decoding, all three `null` when the draft head is off for the request, so they never read as a 0 % acceptance; the ratio is `null` when nothing was proposed |

Durations are milliseconds, rates tokens per second, and the final prompt
token is fed by the first decode step rather than the prefill, so
`prefill_ms` covers one token fewer than `prefill_tokens` counts.

In a stream the object rides the last chunk before `data: [DONE]`: the usage
chunk when the request asked for `stream_options.include_usage`, the
finish-reason chunk otherwise. No client sees a chunk shape it did not
already get.

`GET /v1/timings` returns the same objects for the last 32 completed
requests, newest first, each with its response `id`, `model` and `created`:

```json
{"object": "list", "data": [{"id": "chatcmpl-1789465213-2", "model": "Qwen3.8-Flash-Next",
  "created": 1789465213, "timings": {"prompt_tokens": 21934, "cached_tokens": 0,
  "prefill_tokens": 21934, "prefill_ms": 60098.238, "prefill_per_second": 364.97,
  "generated_tokens": 16, "decode_ms": 415.843, "decode_per_second": 38.48,
  "drafted_tokens": 10, "accepted_tokens": 10, "acceptance_ratio": 1.0}}]}
```

The endpoint exists because client libraries drop response fields they do not
know: `tools/opencode-plugin-timings/` reads it to show lily's real prefill,
decode and acceptance numbers in the opencode terminal UI.

### Flags

| flag | default | meaning |
|------|---------|---------|
| `--model` | required | checkpoint directory |
| `--bind` | `127.0.0.1:8000` | HTTP listen address |
| `--max-seq` | 131072 | prompt plus completion capacity per request (kernel limit 262 144) |
| `--cache-bytes` | derived | GPU bytes for cached sessions, e.g. `24G` (default: working set - weights - paged table - 8 GiB, at least 8 GiB) |
| `--max-sessions` | 16 | most cached sessions |
| `--ngram-table` | `paged` | `paged` (memory-mapped) or `resident` (uploaded) n-gram table |
| `--ngram-preload` | true | read the table at startup so no request pays cold reads |
| `--ngram-lock` | false | `mlock` the table (32 GB other apps cannot reclaim) |
| `--mtp-drafts` | 2 | draft tokens per speculative step (0 turns the draft head off) |
| `--vision` | `auto` | load the vision tower when the checkpoint carries it (0.9 GB); `off` leaves it on disk |
| `--disk-cache-dir` | `~/Library/Caches/lily/sessions` | where evicted sessions are kept |
| `--disk-cache-bytes` | 100G | disk tier budget, LRU; `0` disables the tier |
| `--disk-cache-ttl` | 3d | delete disk entries unused this long (`0`: never) |
| `--durable-min-tokens` | 1024 | a shared prefix at least this long that nothing could resume from becomes a durable disk entry (`0`: off) |
| `--idle-unload` | 0 | unload the model after this long without a request (`30m`, `2h`; `0`: never) |
| `--thinking` | true | open a reasoning block unless the request says otherwise |
| `--reasoning-effort` | template default | `low`, `medium`, `xhigh` |
| `--temperature` … `--repetition-penalty` | generation_config | sampling defaults |
| `--queue` | 32 | requests waiting before 503 |

Two more binaries share the engine: `lily-probe` runs one prompt step by step
and records the top logits (used by `tools/reference/compare.py`), and
`lily-bench` measures prefill and pipelined decode throughput (`--drafts N`
measures speculative decoding and its acceptance rate instead).

### Session cache and disk tier

Every request leaves its state in a cache of sessions under a byte budget
(`--cache-bytes`; by default what the device's recommended working set leaves
after the weights, the paged n-gram table and 8 GiB of headroom for the
applications sharing the machine, but at least 8 GiB; the log states the
derivation as `session cache budget: ... = working set - allocated - paged
weights - headroom`). A session holds the per-token caches for its tokens and
up to three checkpoints of the recurrent state (Gated DeltaNet states and
conv windows) taken at `prompt_len - 1` of recent requests. A new prompt
resumes from the furthest position that is both checkpointed (or the live
end) and a common prefix. Extending a conversation reuses its session in
place; regenerating, editing, or branching forks a copy of the shared prefix,
so parallel conversations never destroy each other's context. Per-token
caches grow in 8 192-token steps.

Sessions evicted from GPU memory go to a disk tier
(`--disk-cache-dir`, default `~/Library/Caches/lily/sessions`;
`--disk-cache-bytes`, default 100 GB, least recently used first, `0` turns it
off; `--disk-cache-ttl`, default 3 days unused, after which an entry is
deleted even when the budget has room). A later prompt that shares their
prefix reads it back in about a second per few gigabytes instead of
recomputing it; the tier survives restarts and is keyed by the model's cache
layout, so other models never read it. Only sessions of 256 tokens or more
are kept, and an 8 GB free-space margin is respected.

The disk tier also learns where agent runs diverge. Two runs of the same
client share their preamble (system prompt, tool schemas, instructions
files) and differ from the user's message on, which is before any
checkpoint, so the second run could not resume from the first. When a prompt
agrees with a cached lineage for at least `--durable-min-tokens` (default
1 024; `0` turns it off) beyond where it resumed, the server writes that
prefix as a **durable prefix entry** to the disk tier, and every later run
with the same preamble resumes from it. Durable entries never occupy GPU
memory; at most 16 exist, least recently used first, and they age out like
any entry once the preamble stops matching. The `timings` object reports
`agreement_tokens` on every request and `durable_prefix_tokens` on the one
that wrote the entry; the log prints a `divergence at N` line with the text
either side of the seam whenever a long shared prefix could not be resumed
from, which is how a client that renders its preamble differently between
runs is found. See
`docs/architecture.md` for the rule.

### Idle unloading, health states and stopping

With `--idle-unload 30m` the engine drops the model once no request has run
for half an hour: resident sessions go to the disk tier first, then the
weights, scratch, caches, Metal context and the n-gram mapping are released
(the process falls from tens of gigabytes to about 120 MB of footprint). The
next request reloads the model and waits for it instead of failing; requests
that arrive during the reload queue as usual. `/health` reports where the
engine is:

```json
{"status": "ok", "state": "idle", "model": "Qwen3.8-Flash-Next", "idle_unload_secs": 1800}
```

`state` is `loading` (first load, 503), `ready`, `idle`, `reloading`,
`recovering` (503, see below) or `stopping` (503); `status` stays `ok`
whenever a request would be served, so clients that only read the status
code behave as before.

**GPU faults.** A Metal 4 command-queue error reported for a command buffer
(typically `MTL4CommandQueueErrorDomain error 1`, a GPU timeout, which the
system raises when a workload runs longer than it allows; severe memory
pressure, with the GPU stalling on paging, is the likely trigger) leaves the
queue in a state nothing can trust, so the server treats it as a transport
failure: the running request gets a 503 whose message names the error (or a
stream error event and `[DONE]` when the response had started), the engine is
dropped **without spilling** its sessions (their GPU state is suspect;
clients re-prefill, and entries the disk tier already holds stay valid),
loaded again on the same thread while `/health` answers `503` with
`"state": "recovering"`, and then serves again as `ready`. Requests that
arrive meanwhile queue for the reload. More than 3 faults within 10 minutes
exit the process with status 1, so a supervisor restarts it with its
throttle interval. The log names the error and its meaning
(`GPU fault: MTL4CommandQueueErrorDomain error 1 (Timeout: ...)`), the
recovery count, and the reload time.

**Stopping.** SIGTERM or SIGINT stop the server cleanly: the listener closes,
a running request gets 10 s to finish (then it is cancelled and told so with
a 503 or a stream error), queued requests get 503, resident sessions are
spilled to the disk tier so the prefix caches survive the restart, and the
process exits 0. A second signal exits at once with status 130. A load that
fails exits 1 so a supervisor restarts the server with its backoff instead of
leaving a server up that can never answer.

### Running as a service

`tools/service/lily-service.sh install` renders a launchd agent
(`~/Library/LaunchAgents/com.lily.server.plist`) that starts the server at
login on `127.0.0.1:8000` with `--max-seq 131072 --idle-unload 30m`, restarts
it after a crash or a failed load (a clean stop stays down), and logs to
`~/Library/Logs/lily/server.log`. `status`, `logs`, `stop`, `start`, `restart`
and `uninstall` do what they say; see
[tools/service/README.md](tools/service/README.md) for the plist keys and the
environment variables that change the flags.

## Tests

```sh
cargo test --locked
```

This runs the CPU-reference kernel tests (including the hyper-connection, PLE,
QSA and sampler kernels), the n-gram hasher and paged-table tests on a
synthetic checkpoint, the shader compilation test, the output parser and
tool-call tests, and the API schema tests. They need a Metal device but no
checkpoint. Tests that need a checkpoint are ignored by default:

```sh
LILY_MODEL_DIR_35B=~/models/Qwen3.6-35B-A3B-4bit \
  cargo test --test test_tokenizer -- --ignored --test-threads=1

LILY_MODEL_DIR_35B=~/models/Qwen3.6-35B-A3B-4bit \
  cargo test --test test_e2e_35b -- --ignored --test-threads=1

LILY_MODEL_DIR_FLASH=~/models/Qwen3.8-Flash-Next-lily-q4 \
  cargo test --release --lib paged_gather_timing -- --ignored --nocapture

# Speculation invariance and the disk-tier round trip (a 4-layer conversion
# with the draft head is enough):
LILY_MODEL_DIR_FLASH=~/models/Qwen3.8-Flash-Next-lily-q4-l4 \
  cargo test --release --test test_speculative_flash -- --ignored --test-threads=1

# The vision tower's tensors against the source checkpoint's values (a
# conversion with the tower appended):
LILY_MODEL_DIR_FLASH=~/models/Qwen3.8-Flash-Next-lily-q4-l4 \
  cargo test --release --lib the_four_layer_checkpoint_tower -- --ignored --test-threads=1

# The tower's output on the 333 x 777 image against the reference golden's
# sample (needs tools/reference/goldens/large/*.pixel_values.npy, untracked;
# tools/README.md says how to regenerate it):
LILY_MODEL_DIR_FLASH=~/models/Qwen3.8-Flash-Next-lily-q4-l4 \
  cargo test --release --lib the_tower_reproduces -- --ignored --nocapture --test-threads=1
```

Two things to know before trusting a green run are in
[CONTRIBUTING.md](CONTRIBUTING.md): a skipped test reports `ok`, and the
tests share one GPU.

End-to-end scripts against a running server live in `tools/e2e/` (`e2e.sh`,
`longctx.py`, `disk.py`). Correctness against Hugging Face transformers on
the same dequantized weights is checked with `tools/reference/`; see
[tools/README.md](tools/README.md).

## Source layout

```text
src/config.rs         strict 35B-A3B checkpoint validation
src/weights.rs        MLX affine Q4/Q8 weight loading (shared loader)
src/metal.rs          Metal 4 transport: queue, argument tables, residency,
                      passes, events, the per-kernel profile mode
src/model.rs          Qwen3.5 prefill and decode graph
src/moe_ffn.rs        sparse-MoE FFN graph shared by both models
src/qwen4exp/         Qwen3.8-Flash-Next config, weights, n-gram table, model graph,
                      speculative decoding (spec.rs)
src/engine.rs         the model-agnostic trait the server drives
src/generate.rs       tokenizer wrapper and the pipelined decode loop
src/serve.rs          engine thread, request flow, OpenAI response shapes
src/serve/http.rs     minimal HTTP/1.1 on std::net (chunked streaming, disconnects)
src/serve/api.rs      request schemas and validation
src/serve/stream.rs   detokenizer, reasoning split, tool-call blocks, stop strings
src/serve/tools.rs    tool schemas and the <tool_call> XML parser
src/serve/session.rs  session cache with checkpoints, forks and a byte budget
src/serve/timings.rs  the `timings` response object and the /v1/timings ring buffer
src/serve/disk.rs     the disk tier below it (LRU, budget, format-tagged files)
src/kernels/          Rust dispatch and Metal shader sources
  hc.*                hyper-connection (gated residual) kernels
  ple.*               n-gram embedding kernels
  qsa.*               Qwen Sparse Attention kernels
  skinny.*            small-row GEMMs for the verify pass
  spec.*              speculative accept and rollback kernels
  sample.*            GPU sampler
tests/                kernel, API, tokenizer, shader, and 35B golden tests
tools/                converter, Hugging Face reference harness, bench and service scripts,
                      the opencode timings plugin
benchmarks/           Lily/MLX harnesses and the fail-closed matrix runner
```

## License

Apache-2.0. See `LICENSE` and `NOTICE`. `NOTICE` records the upstream
copyright and the MLX and MLX-LM code the Metal kernels were adapted from.
