# tools

Python helpers around the Qwen3.8-Flash-Next port. They run from the project
venv (`uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python
mlx safetensors numpy torch "transformers @ git+https://github.com/huggingface/transformers"`).

## convert/convert_qwen38_flash_next.py

Converts the raw Hugging Face BF16 checkpoint into lily's
`qwen4_exp-affine-v1` layout (`docs/qwen38-flash-next-checkpoint-format.md`).

```sh
.venv/bin/python tools/convert/convert_qwen38_flash_next.py \
    --src ~/projects/personal/local-llms/models/Qwen3.8-Flash-Next \
    --dst ~/projects/personal/local-llms/models/Qwen3.8-Flash-Next-lily-q4 \
    [--layers 4] [--dry-run] [--ngram-bits 4 --ngram-group 32]
```

`mlx` needs a Metal device even for CPU arrays, so a real conversion cannot run
in a GPU-less sandbox; `--dry-run` never imports it. Measured on the M5 Max:

| run                    | output    | time |
|------------------------|-----------|------|
| full (48 layers)       | 103.1 GB  | 57 s |
| `--layers 4` (tests)   | 38.6 GB   | 20 s |
| `--layers 1` (smoke)   | 2.2 GB    | 2 s  |

Output categories for the full model: experts 67.95 GB, n-gram table 32.00 GB,
dense 2.45 GB, embeddings + LM head 0.72 GB.

## reference/hf_reference.py

Runs the model truncated to N layers on the CPU with transformers, either on
the raw BF16 weights (`--src`) or on lily's checkpoint dequantized back to bf16
(`--lily`), and writes a JSON golden: argmax at every position, top-8 logits
for the last positions. The n-gram table is gathered lazily from the shards.

```sh
.venv/bin/python tools/reference/hf_reference.py --lily <dir>-l4 --layers 4 \
    --tokens tools/reference/goldens/l4_capital_tokens.json --out golden.json
```

## reference/compare.py

Compares a `lily-probe` record with a golden on the same token sequence
(teacher forcing): argmax agreement, top-8 overlap, logit gap on shared ids.

```sh
cargo run --release --bin lily-probe -- --model <dir>-l4 \
    --prompt "The capital of Switzerland is" --out lily.json
.venv/bin/python tools/reference/compare.py lily.json golden.json
```

Result on the 4-layer model, prompt "The capital of Switzerland is", 8 greedy
steps: against the dequantized weights lily agrees at 9/9 positions with a
worst shared-id logit gap of 0.26; against the raw bf16 weights 7/9, the rest
being quantization error.
