#!/usr/bin/env bash
# Full-model logit parity between lily and the MLX runtime on the same weights
# (models/Qwen3.8-Flash-Next-mlx-4bit-g3264, byte-identical to lily's
# checkpoint except the norms, which MLX stores as bf16(1 + w)).
#
#   tools/reference/mlx_parity.sh prompts   # write the prompt texts (CPU only, once)
#   tools/reference/mlx_parity.sh lily      # lily-probe on every prompt (loads lily, ~100 GB)
#   tools/reference/mlx_parity.sh mlx       # MLX replay of each lily record (loads MLX, ~107 GB peak)
#   tools/reference/mlx_parity.sh compare   # compare.py on every pair, summary to summary.txt
#   tools/reference/mlx_parity.sh runs      # compare_runs.py: lily vs MLX against both engines' floors
#
# Variants of a phase (a noise floor, an intervention) replay the base lily
# record instead of decoding their own path and write tagged records:
#   FOLLOW=1 LILY_TAG=_c2048 LILY_PREFILL_CHUNK=2048 tools/reference/mlx_parity.sh lily
#   MLX_TAG=_p1024 MLX_PREFILL=1024 tools/reference/mlx_parity.sh mlx
#
# The two model phases are separate invocations on purpose: the machine holds
# only one of the two models at a time. Each refuses to start while another
# engine is running and records swap before and after. Output lands in
# docs/bench/mlx-parity/ (gitignored, like the other raw benchmark records).
set -euo pipefail

LILY_DIR=$(cd "$(dirname "$0")/../.." && pwd)
MODELS=${MODELS:-$HOME/projects/personal/local-llms/models}
LILY_MODEL=${LILY_MODEL:-$MODELS/Qwen3.8-Flash-Next-lily-q4}
MLX_MODEL=${MLX_MODEL:-$MODELS/Qwen3.8-Flash-Next-mlx-4bit-g3264}
MLX_PROJECT=${MLX_PROJECT:-$MODELS/qwen38-flash-next-mlx}
OUT=${OUT:-$LILY_DIR/docs/bench/mlx-parity}
STEPS=${STEPS:-32}  # greedy decode steps per prompt
TOP=${TOP:-64}      # logits recorded per step
LILY_TAG=${LILY_TAG:-}        # suffix of a lily variant's records
FOLLOW=${FOLLOW:-}            # 1: the lily variant replays the base record
MLX_TAG=${MLX_TAG:-}          # suffix of an MLX variant's records
MLX_PREFILL=${MLX_PREFILL:-2048}
MLX_INTERNALS=${MLX_INTERNALS:-}  # 1: also record per-layer block selections and router picks
# name:target tokens; 0 means the short fixed question below.
PROMPTS=${PROMPTS:-"capital:0 sparse3k:3500 ctx16k:16384 ctx32k:32768"}

guard() {
    local running rc=0
    running=$(pgrep -fl "lily-bench|lily-probe|lily serve|target/release/lily|mlx_lm|mlx_reference" 2>&1) || rc=$?
    if (( rc > 1 )); then  # 1 is "no match"; above it pgrep could not look (e.g. in a sandbox)
        echo "cannot list processes to check for a running engine: $running" >&2
        exit 1
    fi
    if (( rc == 0 )); then
        echo "another engine is running; stop it first:" >&2
        echo "$running" >&2
        exit 1
    fi
}

swap() { sysctl -n vm.swapusage; }

case ${1:-} in
prompts)
    mkdir -p "$OUT/prompts"
    "$LILY_DIR/.venv/bin/python" - "$OUT/prompts" "$PROMPTS" "$LILY_DIR/tools/bench" <<'EOF'
import sys
sys.path.insert(0, sys.argv[3])
from http_bench import DEFAULT_TOKENIZER, Corpus, prompt_of

out, spec = sys.argv[1], sys.argv[2].split()
corpus = Corpus(DEFAULT_TOKENIZER)
for item in spec:
    name, target = item.split(":")
    target = int(target)
    text = "The capital of Switzerland is" if target == 0 else prompt_of(corpus, target, 1_000_003 + target)
    open(f"{out}/{name}.txt", "w").write(text)
    print(f"{name}: {corpus.count(text)} tokens of text (the chat template adds a few)")
EOF
    ;;
lily)
    guard
    mkdir -p "$OUT"
    cargo build --release --bin lily-probe --manifest-path "$LILY_DIR/Cargo.toml"
    for item in $PROMPTS; do
        name=${item%%:*}
        echo "== lily$LILY_TAG $name; swap before: $(swap)"
        if [[ -n $FOLLOW ]]; then
            [[ -n $LILY_TAG ]] || { echo "FOLLOW needs a LILY_TAG" >&2; exit 1; }
            input=(--follow "$OUT/lily_$name.json")
        else
            input=(--prompt "$(cat "$OUT/prompts/$name.txt")" --max-tokens "$STEPS")
        fi
        "$LILY_DIR/target/release/lily-probe" --model "$LILY_MODEL" "${input[@]}" \
            --top "$TOP" --out "$OUT/lily${LILY_TAG}_$name.json" 2>&1 | tee "$OUT/lily${LILY_TAG}_$name.log"
        echo "   swap after: $(swap)"
    done
    ;;
mlx)
    guard
    for item in $PROMPTS; do
        name=${item%%:*}
        echo "== mlx$MLX_TAG $name; swap before: $(swap)"
        (cd "$MLX_PROJECT" && .venv/bin/python tools/mlx_reference.py --model "$MLX_MODEL" \
            --lily-record "$OUT/lily_$name.json" --out "$OUT/mlx${MLX_TAG}_$name.json" --top "$TOP" \
            --prefill-step "$MLX_PREFILL" ${MLX_INTERNALS:+--internals}) 2>&1 | tee "$OUT/mlx${MLX_TAG}_$name.log"
        echo "   swap after: $(swap)"
    done
    ;;
compare)
    : > "$OUT/summary.txt"
    status=0
    for item in $PROMPTS; do
        name=${item%%:*}
        echo "== $name" >> "$OUT/summary.txt"
        "$LILY_DIR/.venv/bin/python" "$LILY_DIR/tools/reference/compare.py" \
            "$OUT/lily_$name.json" "$OUT/mlx_$name.json" >> "$OUT/summary.txt" || status=1
        echo >> "$OUT/summary.txt"
    done
    cat "$OUT/summary.txt"
    echo "summary: $OUT/summary.txt"
    exit $status
    ;;
runs)
    # Every record of a prompt that followed its base, paired as given in
    # PAIRS (tags, `base` for the lily record that set the path).
    PAIRS=${PAIRS:-"base:mlx mlx:mlx_p1024 base:lily_c2048"}
    for item in $PROMPTS; do
        name=${item%%:*}
        args=()
        for pair in $PAIRS; do
            l=${pair%%:*} r=${pair##*:}
            [[ $l == base ]] || l="$OUT/${l}_$name.json"
            [[ $r == base ]] || r="$OUT/${r}_$name.json"
            args+=("$l:$r")
        done
        echo "######## $name"
        "$LILY_DIR/.venv/bin/python" "$LILY_DIR/tools/reference/compare_runs.py" "$OUT/lily_$name.json" "${args[@]}" \
            --csv "$OUT/runs_$name.csv"
    done | tee "$OUT/runs.txt"
    ;;
*)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
