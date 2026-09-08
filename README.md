# lily

A Metal inference server for Apple Silicon, forked from Perplexity's
[pplx-garden/lily](https://github.com/perplexityai/pplx-garden/tree/main/lily)
(imported at commit `1ed972e`, 2026-09-02) and extended to a second model and
to a full OpenAI-compatible API:

| `model_type`  | checkpoint                                                   | status              |
|---------------|--------------------------------------------------------------|---------------------|
| `qwen3_5_moe` | Qwen3.6-35B-A3B, MLX affine 4-bit (upstream's target)        | unchanged graph, new API |
| `qwen4_exp`   | Qwen3.8-Flash-Next, lily's own `qwen4_exp-affine-v1` layout  | this fork           |

The Metal kernels compile from source at runtime; there is no offline shader
build step.

Reports, with measurement contract:

- [2026-09-08: phase 2, the server and its memory](docs/2026-09-08-phase2-server-report.md)
  (71 GB resident, 32 GB table in the page cache, decode 78 tok/s greedy / 62 to 75 sampling)
- [2026-09-07: Qwen3.8-Flash-Next engine on the M5 Max](docs/2026-09-07-performance-qwen38-flash-next.md)
- upstream, Qwen3.6-35B-A3B: [2026-09-01: MLX 0.31.2](docs/2026-09-01-performance-mlx-0.31.2.md),
  [2026-09-02: MLX 0.32.2](docs/2026-09-02-performance-mlx-0.32.2.md)

Design notes: [phase 2 design](docs/phase2-server-design.md),
[checkpoint format](docs/qwen38-flash-next-checkpoint-format.md).

## Qwen3.8-Flash-Next

Qwen3.8-Flash-Next is Qwen's `qwen4_exp` preview architecture: 48 layers of
Gated DeltaNet / Qwen Sparse Attention (3:1) with 512-expert MoE, a 4-stream
gated residual (hyper-connections), a hashed n-gram embedding at layer 2 with
a 51B-parameter table, and a QSA indexer that selects 512 blocks of 4 tokens
per query once the context exceeds 2 051 tokens. Everything the model needs
is implemented; the vision tower and the MTP head are dropped.

The checkpoint is produced by `tools/convert/convert_qwen38_flash_next.py`
from the raw Hugging Face BF16 weights: affine 4-bit / group 64 for experts,
attention, GDN, shared expert and embeddings, 4-bit / group 32 for the n-gram
table, 8-bit for routers, gates and the hyper-connection mixers. The full
checkpoint is 103 GB on disk.

**Memory.** Only 71 GB of it is uploaded to the GPU. The 32 GB n-gram table
is a pure row gather (16 rows of 100 bytes per token), so the server memory
maps the checkpoint files and copies the rows into a small staging buffer
each step; the rows live in the page cache, evictable, or pinned with
`--ngram-lock`. On a 128 GB machine this leaves room for tens of gigabytes of
cached sessions without raising the GPU wired limit. `--ngram-table resident`
restores the fully resident layout.

Correctness is checked against Hugging Face transformers on the same
dequantized weights with a 4-layer truncation (`tools/reference`): argmax
agreement at every compared position at short context and through the sparse
attention path, logit gaps within bf16 rounding. The paged table reproduces
the resident table's logits exactly.

## Requirements

- Apple GPU family 10 or later (M5 and newer)
- macOS 26 or later for Metal tensor operations
- Rust 1.92, pinned by `rust-toolchain.toml` (this repo drives it through
  asdf's `rustup`; `.tool-versions` names the asdf shim version)
- A converted checkpoint (see above), or upstream's
  `mlx-community/Qwen3.6-35B-A3B-4bit` at revision
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`

Lily validates the exact architecture and quantization layout at load time.
Dense Qwen checkpoints, smaller Qwen checkpoints, BF16 checkpoints, GGUF,
AWQ, GPTQ, int8 and fp8 are not supported.

## Run

```sh
cargo build --release --locked

./target/release/lily \
  --model /path/to/Qwen3.8-Flash-Next-lily-q4 \
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
`max_completion_tokens` (default: the rest of the context);
`stream_options.include_usage`; `reasoning_effort` (`none` disables
thinking, `low`, `medium`, `high`); `chat_template_kwargs` with
`enable_thinking`, `reasoning_effort`, `preserve_thinking`; and
`prompt_cache_key` as a cache hint. Sampling defaults come from the
checkpoint's `generation_config.json` (temperature 1.0, top_k 20, top_p 0.95
for Qwen3.8) and can be overridden on the command line. `top_k` above 1 024
or unset is capped at 1 024 candidates.

Reasoning comes back as `reasoning_content` (in `message` or in stream
deltas). Tool calls are parsed from the model's
`<tool_call><function=…>` XML, typed by the tool's JSON schema, and returned
as OpenAI `tool_calls` with `finish_reason: "tool_calls"`. The final message
must be `user` or `tool`.

Rejected with 400: images, `n > 1`, `logprobs`, `response_format` other than
`text`, `echo`. Requests wait in a bounded queue (`--queue`, 503 when full)
and run one at a time; a client that disconnects cancels its generation at
the next token.

### Session cache

Every request leaves its state in a cache of sessions under a byte budget
(`--cache-bytes`; by default what the device's recommended working set leaves
after the weights, minus 2 GiB). A session holds the per-token caches for its
tokens and up to three checkpoints of the recurrent state (Gated DeltaNet
states and conv windows) taken at `prompt_len - 1` of recent requests. A new
prompt resumes from the furthest position that is both checkpointed (or the
live end) and a common prefix. Extending a conversation reuses its session in
place; regenerating, editing, or branching forks a copy of the shared prefix,
so parallel conversations never destroy each other's context. Per-token caches
grow in 8 192-token steps. `usage.prompt_tokens_details.cached_tokens`
reports the reuse.

### Flags

| flag | default | meaning |
|------|---------|---------|
| `--max-seq` | 131072 | prompt plus completion capacity per request (kernel limit 262 144) |
| `--cache-bytes` | derived | GPU bytes for cached sessions, e.g. `24G` |
| `--max-sessions` | 16 | most cached sessions |
| `--ngram-table` | `paged` | `paged` (memory-mapped) or `resident` (uploaded) n-gram table |
| `--ngram-preload` | true | read the table at startup so no request pays cold reads |
| `--ngram-lock` | false | `mlock` the table (32 GB other apps cannot reclaim) |
| `--thinking` | true | open a reasoning block unless the request says otherwise |
| `--reasoning-effort` | template default | `low`, `medium`, `xhigh` |
| `--temperature` … `--repetition-penalty` | generation_config | sampling defaults |
| `--queue` | 32 | requests waiting before 503 |

Two more binaries share the engine: `lily-probe` runs one prompt step by step
and records the top logits (used by `tools/reference/compare.py`), and
`lily-bench` measures prefill and pipelined decode throughput.

## Tests

```sh
cargo test --locked
```

This runs the CPU-reference kernel tests (including the hyper-connection, PLE,
QSA and sampler kernels), the n-gram hasher and paged-table tests on a
synthetic checkpoint, the shader compilation test, the output parser and
tool-call tests, and the API schema tests. Tests that need a checkpoint are
ignored by default:

```sh
LILY_MODEL_DIR_35B=/path/to/Qwen3.6-35B-A3B-4bit \
  cargo test --test test_tokenizer -- --ignored --test-threads=1

LILY_MODEL_DIR_35B=/path/to/Qwen3.6-35B-A3B-4bit \
  cargo test --test test_e2e_35b -- --ignored --test-threads=1

LILY_MODEL_DIR_FLASH=/path/to/Qwen3.8-Flash-Next-lily-q4 \
  cargo test --release --lib paged_gather_timing -- --ignored --nocapture
```

## Source layout

```text
src/config.rs         strict 35B-A3B checkpoint validation
src/weights.rs        MLX affine Q4/Q8 weight loading (shared loader)
src/model.rs          Qwen3.5 prefill and decode graph
src/moe_ffn.rs        sparse-MoE FFN graph shared by both models
src/qwen4exp/         Qwen3.8-Flash-Next config, weights, n-gram table, model graph
src/engine.rs         the model-agnostic trait the server drives
src/generate.rs       tokenizer wrapper and the pipelined decode loop
src/serve.rs          engine thread, request flow, OpenAI response shapes
src/serve/http.rs     minimal HTTP/1.1 on std::net (chunked streaming, disconnects)
src/serve/api.rs      request schemas and validation
src/serve/stream.rs   detokenizer, reasoning split, tool-call blocks, stop strings
src/serve/tools.rs    tool schemas and the <tool_call> XML parser
src/serve/session.rs  session cache with checkpoints, forks and a byte budget
src/kernels/          Rust dispatch and Metal shader sources
  hc.*                hyper-connection (gated residual) kernels
  ple.*               n-gram embedding kernels
  qsa.*               Qwen Sparse Attention kernels
  sample.*            GPU sampler
tests/                kernel, API, tokenizer, shader, and 35B golden tests
tools/                converter and Hugging Face reference harness
benchmarks/           Lily/MLX harnesses and the fail-closed matrix runner
```

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
