#!/usr/bin/env python3
"""Pairwise distribution distances between runs on one teacher-forced token
path, so an engine difference can be set against each engine's own noise.

    .venv/bin/python tools/reference/compare_runs.py BASE A:B [C:D ...] [--csv out.csv]

BASE is the `lily-probe` record that defined the path; every other run
followed it: `lily-probe --follow BASE` (its `at_ref_ids`) or
`models/qwen38-flash-next-mlx/tools/mlx_reference.py --lily-record BASE`
(its `at_lily_ids`). A run named `base` is BASE itself. Each run therefore
has logits on BASE's recorded ids at every step plus its own full-vocabulary
log-sum-exp, and a pair A:B is compared at every step on

* KL(A || B) over BASE's ids, the remaining mass as one lumped bucket,
* the largest log-probability gap on BASE's top-8 ids,
* whether the two runs' own argmax agrees, and BASE's top-1/top-2 margin
  where it does not.

Distances between runs are heavy-tailed (a few near-tie positions carry most
of a mean), so the summary gives median, 90th percentile and max beside the
mean, and lists the positions whose KL exceeds ten times the pair's median.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


class Run:
    """Logits on BASE's ids, log-sum-exp and argmax, per position."""

    def __init__(self, path: str, base: dict):
        self.name = Path(path).stem
        rec = base if path == "base" else json.loads(Path(path).read_text())
        self.by_pos: dict[int, tuple[list[float], float, int]] = {}
        base_ids = {s["position"]: s["ids"] for s in base["steps"]}
        if "steps" in rec:  # a lily-probe record: BASE itself or a --follow run
            for s in rec["steps"]:
                at = s["logits"] if rec is base else s["at_ref_ids"]
                if rec is not base and s.get("at_ref_ids") is None:
                    raise SystemExit(f"{path}: not a --follow run of the base")
                self.by_pos[s["position"]] = (at, s["logsumexp"], s["ids"][0])
        else:  # an MLX golden
            for e in rec["top8_last"]:
                if e["lily_ids"] != base_ids.get(e["position"]):
                    raise SystemExit(f"{path}: position {e['position']} was not replayed from this base")
                self.by_pos[e["position"]] = (e["at_lily_ids"], e["logsumexp"], rec["argmax"][e["position"]])


def kl(p_logits, p_lse, q_logits, q_lse) -> float:
    p = [math.exp(l - p_lse) for l in p_logits]
    q = [math.exp(l - q_lse) for l in q_logits]
    out = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0)
    rest_p, rest_q = 1.0 - sum(p), 1.0 - sum(q)
    if rest_p > 1e-12 and rest_q > 1e-12:
        out += rest_p * math.log(rest_p / rest_q)
    return max(out, 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("pairs", nargs="+", help="A:B, each a record path or `base`")
    ap.add_argument("--csv", help="per-position KL of every pair")
    a = ap.parse_args()

    base = json.loads(Path(a.base).read_text())
    margin = {s["position"]: s["logits"][0] - s["logits"][1] for s in base["steps"]}
    runs: dict[str, Run] = {}
    rows: dict[str, dict[int, float]] = {}
    for pair in a.pairs:
        left, right = pair.split(":")
        for path in (left, right):
            if path not in runs:
                runs[path] = Run(path, base)
        x, y = runs[left], runs[right]
        positions = sorted(set(x.by_pos) & set(y.by_pos))
        kls, gaps, flips = [], [], []
        for p in positions:
            xl, xs, xa = x.by_pos[p]
            yl, ys, ya = y.by_pos[p]
            kls.append(kl(xl, xs, yl, ys))
            gaps.append(max(abs((u - xs) - (v - ys)) for u, v in zip(xl[:8], yl[:8])))
            if xa != ya:
                flips.append(f"{p}({margin[p]:.3f})")
        label = f"{x.name} vs {y.name}"
        rows[label] = dict(zip(positions, kls))
        med = statistics.median(kls)
        p90 = sorted(kls)[int(0.9 * (len(kls) - 1))]
        print(f"== {label}: {len(positions)} positions")
        print(f"   KL mean {statistics.fmean(kls):.2e}  median {med:.2e}  p90 {p90:.2e}  max {max(kls):.2e}")
        print(f"   top-8 log-prob gap median {statistics.median(gaps):.3f}  max {max(gaps):.3f}")
        print(f"   argmax flips {len(flips)}: {' '.join(flips) or '-'}  (BASE margin in parentheses)")
        outliers = [f"{p}:{k:.1e}" for p, k in zip(positions, kls) if k > 10 * med]
        print(f"   KL > 10x median at {len(outliers)}: {' '.join(outliers[:20]) or '-'}")
    if a.csv:
        labels = list(rows)
        all_pos = sorted(set().union(*(r.keys() for r in rows.values())))
        with open(a.csv, "w") as f:
            f.write("position,base_margin," + ",".join(labels) + "\n")
            for p in all_pos:
                f.write(f"{p},{margin.get(p, float('nan')):.4f}," + ",".join(f"{rows[l].get(p, float('nan')):.4e}" for l in labels) + "\n")
        print(f"per-position KL: {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
