# tools

Python helpers around the Qwen3.8-Flash-Next port. They run from the project
venv (`uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python
mlx safetensors numpy torch torchvision pillow "transformers @ git+https://github.com/huggingface/transformers"`;
`torchvision` and `pillow` are for the vision reference: the checkpoint's image
processor is the torchvision-backed `Qwen2VLImageProcessorFast`).

## convert/convert_qwen38_flash_next.py

Converts the raw Hugging Face BF16 checkpoint into lily's
`qwen4_exp-affine-v1` layout (`docs/qwen38-flash-next-checkpoint-format.md`).

```sh
.venv/bin/python tools/convert/convert_qwen38_flash_next.py \
    --src ~/models/Qwen3.8-Flash-Next \
    --dst ~/models/Qwen3.8-Flash-Next-lily-q4 \
    [--layers 4] [--dry-run] [--ngram-bits 4 --ngram-group 32] [--no-mtp] [--no-vision]
```

`mlx` needs a Metal device even for CPU arrays, so a real conversion cannot run
in a GPU-less sandbox; `--dry-run` never imports it. Measured on the M5 Max:

| run                    | output    | time |
|------------------------|-----------|------|
| full (48 layers)       | 103.1 GB  | 57 s |
| `--layers 4` (tests)   | 38.6 GB   | 20 s |
| `--layers 1` (smoke)   | 2.2 GB    | 2 s  |

Output categories for the full model: experts 67.95 GB, n-gram table 32.00 GB,
dense 2.45 GB, embeddings + LM head 0.72 GB, plus the draft head 1.48 GB and
the vision tower 0.90 GB (105.49 GB with both; the table above predates them).

Two optional parts can be appended to a finished conversion instead of being
written with it, each as its own shard family merged into the index and
`config.json`: `--mtp-only` for the draft head (`mtp-*.safetensors`, quantized
like the trunk) and `--vision-only` for the vision tower (`vision-*.safetensors`,
333 bf16 tensors copied byte for byte, 0.90 GB, in under a second). Both refuse
to run twice. `--no-mtp` and `--no-vision` drop the part from a full conversion,
and `config.json` then loses `vision_config` and the vision token ids so that it
matches the weights (`docs/qwen38-flash-next-checkpoint-format.md`).

```sh
.venv/bin/python tools/convert/convert_qwen38_flash_next.py \
    --src ~/models/Qwen3.8-Flash-Next --dst ~/models/Qwen3.8-Flash-Next-lily-q4 --vision-only
```

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

## reference/hf_vision_reference.py, compare_vision.py, make_images.py

The vision counterpart (docs/architecture.md, "How the tower was verified"). `make_images.py`
writes the four synthetic test images in `reference/images/` (333 x 777,
640 x 480, 1920 x 1080, 3840 x 2160; `images.json` has their sha256).
`hf_vision_reference.py` produces one golden per comparison in the plan from
`Qwen4ExpForConditionalGeneration` and its parts: `preprocess` (pixel_values,
image_grid_thw), `tower` (`Qwen4ExpVisionModel` in float32 on the CPU),
`positions` (expanded input_ids, 3-axis position_ids, rope_deltas) and
`forward` (4-layer logits with the image, or the text-only control), plus
`measure-preprocess` and `measure-tower`, which write the tolerance floors to
`goldens/vision_tolerance_floors.json`. Every golden records the pixel cap it
was made with; the default is lily's server-side cap, `--image-max-pixels
2097152` (2 048 tokens) and `--image-min-pixels 65536`. Full tensors go to
`goldens/large/` (not tracked); the JSON keeps shape, sha256, statistics and a
seeded 4 096-element sample.

`lily-vision-probe` is the Rust side of comparisons 1 and 2. With `--model`
it loads only the tower from a converted checkpoint, runs it over a golden's
`pixel_values` (the matching
`goldens/large/preprocess_<img>_cap<max>.pixel_values.npy`, or `--pixels`),
and writes the merged and pre-merger outputs as `.npy` plus a candidate JSON in
the golden format, carrying the golden's own sample indices. `--repeat` times
the runs, `--kernel-profile` prints GPU ms per kernel, `--blocks N` stops after
N transformer blocks for block-by-block comparisons.

With `--image <png or jpeg>` it runs lily's own preprocessing
(`src/qwen4exp/image.rs`) under `--max-pixels` (default 2 097 152) and
`--min-pixels` (default 65 536), writes a `preprocess` record plus
`<out>.pixel_values.npy` for comparison 1, and reports the time it took. The
record takes its sample indices from `--preprocess-golden`, or from the
`hf_vision_preprocess_<img>_cap<max>.json` next to `--golden` when that is
given; without either the sample is an even stride and the comparison needs
the `.npy` files. With `--model` as well, the tower then runs on lily's own
pixel rows (the grid must match the tower golden's, so use the golden's cap)
and the preprocess record goes to `<out stem>.preprocess.json` next to the
tower record (`--preprocess-out` overrides). No GPU is touched without
`--model`.

```sh
# comparison 2 on the reference's pixel_values
cargo run --release --bin lily-vision-probe -- --model <dir>-l4 \
    --golden tools/reference/goldens/hf_vision_tower_333x777_cap2097152.json --out lily_tower.json
.venv/bin/python tools/reference/compare_vision.py lily_tower.json tools/reference/goldens/hf_vision_tower_333x777_cap2097152.json

# comparison 1 alone, no GPU
cargo run --release --bin lily-vision-probe -- --image tools/reference/images/333x777.png \
    --preprocess-golden tools/reference/goldens/hf_vision_preprocess_333x777_cap2097152.json --out lily_pre.json
.venv/bin/python tools/reference/compare_vision.py lily_pre.json tools/reference/goldens/hf_vision_preprocess_333x777_cap2097152.json

# comparisons 1 and 2 end to end: the tower on lily's own pixel rows
cargo run --release --bin lily-vision-probe -- --model <dir>-l4 --image tools/reference/images/333x777.png \
    --golden tools/reference/goldens/hf_vision_tower_333x777_cap2097152.json --out lily_tower.json
.venv/bin/python tools/reference/compare_vision.py lily_tower.preprocess.json tools/reference/goldens/hf_vision_preprocess_333x777_cap2097152.json
.venv/bin/python tools/reference/compare_vision.py lily_tower.json tools/reference/goldens/hf_vision_tower_333x777_cap2097152.json
```

The Rust unit tests in `tests/unit/qwen4exp/image.rs` run comparison 1 on
the five committed preprocess goldens from `cargo test` without Python; the
exact check against Pillow's own `Image.resize` output reads raw dumps from
`LILY_PIL_RESIZE_DIR` when set (`pil_<stem>_<w>x<h>.rgb`, with `<stem>.png`
beside them or a test image of that name).

```sh
.venv/bin/python tools/reference/hf_vision_reference.py preprocess --src ~/models/Qwen3.8-Flash-Next \
    --image tools/reference/images/333x777.png
.venv/bin/python tools/reference/hf_vision_reference.py tower      --src ... --image ...
.venv/bin/python tools/reference/hf_vision_reference.py positions  --src ... --image ...
.venv/bin/python tools/reference/hf_vision_reference.py forward    --src ... --lily <dir>-l4 --image ... --greedy 4
.venv/bin/python tools/reference/compare_vision.py lily_record.json tools/reference/goldens/hf_vision_tower_333x777_cap2097152.json
```

`compare_vision.py` judges a candidate in the same JSON format: comparisons 1
and 2 within tolerance (max abs and rel error, cosine, relative L2, fraction
within tolerance), comparison 3 exactly, comparison 4 through `compare.py`.
The facts lily's implementation has to reproduce, with the tolerances and the
numbers behind them, are in `reference/VISION.md`.

## bench/timeline.sh and bench/summarize.py

The performance timeline. `timeline.sh` runs
the fixed `lily-bench` matrix (1K / 8K / 32K prompts, 0 and 2 drafts, 96
tokens, 3 repeats) at one or more commits (`--commit <rev>`, older commits are
built in worktrees under `target/timeline/`, `HEAD` is the working tree),
writes every run's JSON, a `.env.txt` snapshot (swap, paging, thermal level)
and a `run.json` into `docs/bench/<date>-<sha>/`, and calls `summarize.py` to
regenerate the tables in `docs/performance-timeline.md`. Both the records and
that document are local measurement output: they are not tracked, and
`summarize.py` creates the document when it is missing. The published
numbers and the method behind them are in `docs/performance.md`.
Several commits are interleaved per repeat so drift
affects them equally; `--cooldown SEC` idles between runs. Run it by hand on
mains power; it refuses to start on battery unless told `--allow-battery`, and
nothing runs it automatically. About 10 minutes per commit. `summarize.py`
needs only the system `python3`; both work with macOS bash 3.2.

```sh
tools/bench/timeline.sh --note "what changed"                        # working tree
tools/bench/timeline.sh --commit f5b3317 --commit HEAD --cooldown 20  # compare two commits
tools/bench/timeline.sh --dry-run --commit HEAD                       # print the commands only
```

## service/lily-service.sh

Not Python: a bash script and a launchd plist template that run the server as
a per-user agent (start at login, restart after a crash or a failed load,
`--idle-unload 30m`). `install`, `uninstall`, `start`, `stop`, `restart`,
`status`, `logs`; see `service/README.md`.
