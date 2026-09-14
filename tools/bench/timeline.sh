#!/usr/bin/env bash
# Performance timeline: run the fixed lily-bench matrix at one or more commits
# and record every run under docs/bench/<date>-<sha>/, then regenerate
# docs/performance-timeline.md. Both are local measurement output and are not
# tracked. Run it on demand, on mains power; nothing runs it automatically.
#
#   tools/bench/timeline.sh --note "Q8 skinny GEMMs for m <= 16"
#   tools/bench/timeline.sh --commit 1e33b4e --commit f5b3317 --commit HEAD --cooldown 20
#
# With several commits the script builds them all first and then interleaves
# them per repeat (repeat 1 of every commit, then repeat 2, ...), so thermal
# drift over a long run lands on every commit alike instead of on whichever
# ran last. A 40-minute back-fill measured sequentially showed the first cell
# 30 to 40% faster than everything after it; that is why.
#
# Per commit at the default matrix (3 prompts x 2 draft settings x 3 repeats)
# expect roughly 10 minutes: each run loads the model (~16 s) and the 32K
# prefills take ~26 s each. Requires bash 3.2 (macOS) or newer.
set -euo pipefail

usage() {
    cat <<'USAGE'
usage: tools/bench/timeline.sh [options]
  --model DIR       checkpoint directory (default ../models/Qwen3.8-Flash-Next-lily-q4)
  --commit REV      bench REV (repeatable); "HEAD" or "worktree" means the working tree,
                    anything else is built in a worktree under target/timeline/.
                    Default: the working tree only.
  --note TEXT       one line on what changed (stored in run.json; with several commits it
                    applies to all of them, the commit subject is recorded anyway)
  --prompts LIST    prompt lengths, space separated (default "1024 8192 32768")
  --drafts LIST     drafts per step, space separated (default "0 2"; non-zero values are
                    skipped when a commit's lily-bench has no --drafts flag)
  --steps N         decode steps / generated tokens per run (default 96)
  --repeats N       repeats per cell (default 3)
  --cooldown SEC    idle seconds between runs (default 0), for thermal recovery
  --out-root DIR    where record directories go (default docs/bench)
  --allow-battery   run even when the machine reports battery power
  --dry-run         print the commands without running anything
USAGE
}

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model="$root/../models/Qwen3.8-Flash-Next-lily-q4"
revs=()
note=""
prompts="1024 8192 32768"
drafts="0 2"
steps=96
repeats=3
cooldown=0
out_root="$root/docs/bench"
allow_battery=0
dry_run=0

while [ $# -gt 0 ]; do
    case "$1" in
        --model) model="$2"; shift 2 ;;
        --commit) revs+=("$2"); shift 2 ;;
        --note) note="$2"; shift 2 ;;
        --prompts) prompts="$2"; shift 2 ;;
        --drafts) drafts="$2"; shift 2 ;;
        --steps) steps="$2"; shift 2 ;;
        --repeats) repeats="$2"; shift 2 ;;
        --cooldown) cooldown="$2"; shift 2 ;;
        --out-root) out_root="$2"; shift 2 ;;
        --allow-battery) allow_battery=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[ ${#revs[@]} -gt 0 ] || revs=(worktree)

run() {
    if [ "$dry_run" = 1 ]; then
        printf '+'; printf ' %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

# Swap use, paging and thermal state right before a run: when repeats disagree
# this is the first place to look (phase 1 held the n-gram table resident and
# pushed the machine into swap after its first pass).
snapshot_env() {
    {
        date -u +%Y-%m-%dT%H:%M:%SZ
        sysctl vm.swapusage 2>/dev/null || true
        vm_stat 2>/dev/null | grep -E "Pages free|Pages wired|Pageouts|Swapins|Swapouts" || true
        pmset -g therm 2>/dev/null || true
    } > "$1"
}
swap_used() {
    sysctl -n vm.swapusage 2>/dev/null | sed -E 's/.*used = ([0-9.]+)M.*/\1/' || true
}

power=$(pmset -g batt 2>/dev/null | head -n 1 || true)
case "$power" in
    *"AC Power"*) power_source="ac" ;;
    *"Battery"*) power_source="battery" ;;
    *) power_source="unknown" ;;
esac
if [ "$power_source" != "ac" ] && [ "$allow_battery" != 1 ] && [ "$dry_run" != 1 ]; then
    echo "refusing to bench on '$power_source' power (pmset: ${power:-n/a}); pass --allow-battery to override" >&2
    exit 1
fi
[ -f "$model/config.json" ] || { echo "no checkpoint at $model" >&2; exit 1; }

date_tag=$(date +%Y-%m-%d)
worktree_dirty=0
if [ -n "$(git -C "$root" status --porcelain --untracked-files=no -- . ':!docs/bench' ':!docs/performance-timeline.md')" ]; then
    worktree_dirty=1
fi

# Per-commit tables (bash 3.2: indexed arrays only, same index everywhere).
n=0
c_sha=(); c_short=(); c_label=(); c_tree=(); c_bin=(); c_out=(); c_dirty=(); c_has_drafts=(); c_has_preload=()
for rev in "${revs[@]}"; do
    if [ "$rev" = worktree ] || [ "$rev" = HEAD ]; then
        sha=$(git -C "$root" rev-parse HEAD)
        short=$(git -C "$root" rev-parse --short HEAD)
        tree="$root"
        dirty=$worktree_dirty
    else
        sha=$(git -C "$root" rev-parse --verify "$rev^{commit}")
        short=$(git -C "$root" rev-parse --short "$sha")
        tree="$root/target/timeline/$short"
        dirty=0
        [ -d "$tree" ] || run git -C "$root" worktree add --detach "$tree" "$sha"
    fi
    label="$short"
    [ "$dirty" = 1 ] && label="$short-dirty"
    out="$out_root/$date_tag-$label"
    if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ] && [ "$dry_run" != 1 ]; then
        echo "record directory exists and is not empty: $out" >&2
        exit 1
    fi
    echo "== building lily-bench at $label in $tree"
    run cargo build --release --locked --bin lily-bench --manifest-path "$tree/Cargo.toml"
    bin="$tree/target/release/lily-bench"
    has_drafts=0; has_preload=0
    if [ -x "$bin" ]; then
        help=$("$bin" --help 2>&1 || true)
        case "$help" in *"--drafts"*) has_drafts=1 ;; esac
        case "$help" in *"--ngram-preload"*) has_preload=1 ;; esac
    fi
    c_sha[$n]=$sha; c_short[$n]=$short; c_label[$n]=$label; c_tree[$n]=$tree; c_bin[$n]=$bin
    c_out[$n]=$out; c_dirty[$n]=$dirty; c_has_drafts[$n]=$has_drafts; c_has_preload[$n]=$has_preload
    n=$((n + 1))
done
interleaved=""
[ $n -gt 1 ] && interleaved="${c_short[*]}"

# write_meta INDEX FINISHED-TIMESTAMP (empty while running)
write_meta() {
    local i=$1
    python3 - "${c_out[$i]}/run.json" <<'PY' "${c_sha[$i]}" "${c_short[$i]}" "${c_dirty[$i]}" "$note" "$model" "$prompts" "$drafts" "$steps" "$repeats" "${c_has_drafts[$i]}" "${c_has_preload[$i]}" "$power_source" "$started" "$2" "$cooldown" "$swap_start" "$(swap_used)" "$interleaved"
import json, subprocess, sys
(path, sha, short, dirty, note, model, prompts, drafts, steps, repeats, has_drafts, has_preload,
 power, started, finished, cooldown, swap_start, swap_now, interleaved) = sys.argv[1:]
def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None
meta = {
    "schema_version": 1,
    "date": started[:10],
    "started": started,
    "finished": finished or None,
    "commit": sha,
    "commit_short": short,
    "dirty": dirty == "1",
    "subject": sh("git", "log", "-1", "--format=%s", sha),
    "commit_time": sh("git", "log", "-1", "--format=%cI", sha),
    "note": note,
    "model_dir": model.rstrip("/").rsplit("/", 1)[-1],
    "matrix": {
        "prompt_lens": [int(p) for p in prompts.split()],
        "drafts": [int(d) for d in drafts.split()],
        "decode_steps": int(steps),
        "repeats": int(repeats),
        "cooldown_secs": float(cooldown),
        "interleaved_with": interleaved.split() if interleaved else [],
    },
    "bench_flags": {"drafts": has_drafts == "1", "ngram_preload": has_preload == "1"},
    "host": {
        "macos": sh("sw_vers", "-productVersion"),
        "hw_model": sh("sysctl", "-n", "hw.model"),
        "power": power,
        "swap_used_mb_start": float(swap_start) if swap_start else None,
        "swap_used_mb_end": float(swap_now) if (finished and swap_now) else None,
    },
}
with open(path, "w") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
PY
}

started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
swap_start=""
if [ "$dry_run" != 1 ]; then
    swap_start=$(swap_used)
    i=0
    while [ $i -lt $n ]; do
        mkdir -p "${c_out[$i]}"
        write_meta $i ""
        i=$((i + 1))
    done
fi

failures=0
# Repeats outermost, commits next: drift spreads evenly over commits and cells.
for r in $(seq 1 "$repeats"); do
    i=0
    while [ $i -lt $n ]; do
        out=${c_out[$i]}; bin=${c_bin[$i]}
        for p in $prompts; do
            for d in $drafts; do
                if [ "$d" != 0 ] && [ "${c_has_drafts[$i]}" != 1 ]; then
                    [ "$r" = 1 ] && echo "skipping ${c_label[$i]} drafts=$d: its lily-bench has no --drafts flag"
                    continue
                fi
                stem="$out/p$p-d$d-r$r"
                args=(--model "$model" --prompt-len "$p" --decode-steps "$steps" --json-out "$stem.json")
                [ "${c_has_preload[$i]}" = 1 ] && args+=(--ngram-preload)
                [ "$d" != 0 ] && args+=(--drafts "$d")
                echo "== ${c_label[$i]} p=$p drafts=$d repeat=$r"
                if [ "$dry_run" = 1 ]; then
                    run "$bin" "${args[@]}"
                    continue
                fi
                snapshot_env "$stem.env.txt"
                if "$bin" "${args[@]}" > "$stem.log" 2>&1; then
                    grep -E "^prefill:" "$stem.log" | tee -a "$out/log.txt"
                else
                    failures=$((failures + 1))
                    echo "FAILED ${c_label[$i]} p=$p drafts=$d repeat=$r (see $stem.log)" | tee -a "$out/log.txt"
                    tail -n 5 "$stem.log"
                fi
                [ "$cooldown" != 0 ] && sleep "$cooldown"
            done
        done
        i=$((i + 1))
    done
done

if [ "$dry_run" != 1 ]; then
    finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    i=0
    while [ $i -lt $n ]; do
        write_meta $i "$finished"
        i=$((i + 1))
    done
    echo "== summary"
    if [ "$out_root" = "$root/docs/bench" ]; then
        python3 "$root/tools/bench/summarize.py" --root "$out_root" --write "$root/docs/performance-timeline.md"
    else
        # Side-by-side runs elsewhere (an A/B pair) are not part of the
        # timeline; print their tables instead of rewriting the document.
        python3 "$root/tools/bench/summarize.py" --root "$out_root"
    fi
    i=0
    while [ $i -lt $n ]; do
        echo "records: ${c_out[$i]}"
        [ "${c_tree[$i]}" != "$root" ] && echo "worktree kept at ${c_tree[$i]} (remove with: git worktree remove ${c_tree[$i]})"
        i=$((i + 1))
    done
    if [ "$failures" != 0 ]; then
        echo "$failures run(s) failed" >&2
        exit 1
    fi
fi
