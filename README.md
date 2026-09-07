# lily

A small Metal inference server for Apple Silicon, forked from Perplexity's
[pplx-garden/lily](https://github.com/perplexityai/pplx-garden/tree/main/lily)
(imported at commit `1ed972e`, 2026-09-02) and extended to a second model:

| `model_type`  | checkpoint                                                   | status              |
|---------------|--------------------------------------------------------------|---------------------|
| `qwen3_5_moe` | Qwen3.6-35B-A3B, MLX affine 4-bit (upstream's target)        | unchanged upstream  |
| `qwen4_exp`   | Qwen3.8-Flash-Next, lily's own `qwen4_exp-affine-v1` layout  | this fork           |

Lily exposes a minimal subset of the OpenAI chat completions API and always
decodes greedily. The Metal kernels compile from source at runtime; there is
no offline shader build step.

Upstream's performance reports for the 35B model, with measurement contract:

- [2026-09-01: MLX 0.31.2](docs/2026-09-01-performance-mlx-0.31.2.md)
- [2026-09-02: MLX 0.32.2](docs/2026-09-02-performance-mlx-0.32.2.md)

## Qwen3.8-Flash-Next

Qwen3.8-Flash-Next is Qwen's `qwen4_exp` preview architecture: 48 layers of
Gated DeltaNet / Qwen Sparse Attention (3:1) with 512-expert MoE, a 4-stream
gated residual (hyper-connections), a hashed n-gram embedding at layer 2 with
a 51B-parameter table, and a QSA indexer that selects 512 blocks of 4 tokens
per query once the context exceeds 2 051 tokens. Everything the model needs
is implemented; the vision tower and the MTP head are dropped.

The checkpoint is produced by `tools/convert/convert_qwen38_flash_next.py`
from the raw Hugging Face BF16 weights (format in
`docs/qwen38-flash-next-checkpoint-format.md`): affine 4-bit / group 64 for
experts, attention, GDN, shared expert and embeddings, 4-bit / group 32 for the
n-gram table, 8-bit for routers, gates and the hyper-connection mixers. The
full model is 103 GB and needs the 128 GB M5 Max with the GPU wired limit
raised (`sudo sysctl iogpu.wired_limit_mb=114688`).

Correctness is checked against Hugging Face transformers on the same
dequantized weights with a 4-layer truncation (`tools/reference`): argmax
agreement at every compared position at short context and through the sparse
attention path at 2 300 tokens, logit gaps within bf16 rounding.

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
  --max-seq 32768
```

The server picks the engine from the checkpoint's `model_type`, compiles its
shaders during a startup warm-up, and provides:

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-Flash-Next",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "prompt_cache_key": "conversation-1"
  }'
```

The request surface is intentionally strict. It accepts text-only
system/user/assistant messages, `max_tokens`, `stream: false`, and the optional
`prompt_cache_key`. The final message must have role `user`, and the server
always renders the checkpoint template with thinking disabled. Sampling
parameters, streaming responses, tools, response formats, multimodal content,
and speculative decoding are rejected.

`--max-seq` is the combined prompt-plus-completion capacity. It cannot exceed
the kernel limit of 262,144 tokens and is clamped to the checkpoint's declared
`max_position_embeddings` when that value is smaller.

Two more binaries share the engine: `lily-probe` runs one prompt step by step
and records the top logits (used by `tools/reference/compare.py`), and
`lily-bench` measures prefill and pipelined decode throughput.

## Session prefix cache

The server keeps a fixed two-entry LRU cache of decode states. A state is reused
only when its token sequence is a strict prefix of the new prompt. A matching
`prompt_cache_key` prefers the corresponding entry but never bypasses token
equality. The response reports reused tokens in
`usage.prompt_tokens_details.cached_tokens`.

## Tests

```sh
cargo test --locked
```

This runs the CPU-reference kernel tests (including the hyper-connection, PLE
and QSA kernels), the shader compilation test, API surface tests, and
prefix-cache tests. Tests that need the 35B checkpoint are explicitly ignored
by default:

```sh
LILY_MODEL_DIR_35B=/path/to/Qwen3.6-35B-A3B-4bit \
  cargo test --test test_tokenizer -- --ignored --test-threads=1

LILY_MODEL_DIR_35B=/path/to/Qwen3.6-35B-A3B-4bit \
  cargo test --test test_e2e_35b -- --ignored --test-threads=1
```

## Source layout

```text
src/config.rs         strict 35B-A3B checkpoint validation
src/weights.rs        MLX affine Q4/Q8 weight loading (shared loader)
src/model.rs          Qwen3.5 prefill and single-token greedy model graph
src/moe_ffn.rs        sparse-MoE FFN graph shared by both models
src/qwen4exp/         Qwen3.8-Flash-Next config, weights, and model graph
src/engine.rs         the model-agnostic trait the server drives
src/generate.rs       tokenizer-backed greedy decode loop
src/serve.rs          minimal OpenAI-compatible HTTP server
src/serve/session.rs  strict token-prefix session cache
src/kernels/          Rust dispatch and Metal shader sources
  hc.*                hyper-connection (gated residual) kernels
  ple.*               n-gram embedding kernels
  qsa.*               Qwen Sparse Attention kernels
tests/                kernel, API, tokenizer, shader, and 35B golden tests
tools/                converter and Hugging Face reference harness
benchmarks/           Lily/MLX harnesses and the fail-closed matrix runner
```

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
