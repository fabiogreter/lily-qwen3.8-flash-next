#!/usr/bin/env python3
"""Compare a lily probe record against a Hugging Face reference golden produced
on the same token sequence (teacher forcing).

    .venv/bin/python tools/reference/compare.py \
        tools/reference/goldens/lily_l4_capital.json \
        tools/reference/goldens/hf_l4_capital_dequant.json

lily records logits only for positions it decoded at (the last prompt token
and every generated token; `lily-vision-probe --forward` adds the golden's
top-8 positions); the HF golden records argmax for every position and top-8
logits for the last `--last` positions. Positions present in both are compared
on argmax agreement, top-8 overlap, and the logit gap on shared ids. A golden
with a `greedy` continuation (the vision forward goldens) is also compared at
the generated positions, token k of it being the argmax at `n - 1 + k`; a
candidate carrying a `positions` block (VISION.md comparison 3) has it checked
exactly against the golden's.

When both sides carry a full-vocabulary `logsumexp` (lily-probe records it; the
MLX golden from `models/qwen38-flash-next-mlx/tools/mlx_reference.py` also
records the golden's logits at lily's ids), the distributions are compared too:
KL(lily || golden) over lily's recorded ids with the remaining mass as one
lumped bucket, the log-probability gap on lily's chosen token, and lily's
top-1/top-2 margin, which shows whether a flipped argmax was a near-tie.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def distribution(step: dict, golden: dict) -> tuple[float, float] | None:
    """(KL(lily || golden), |Δ log p(chosen)|), or None without the fields."""
    if "logsumexp" not in step or "at_lily_ids" not in golden or golden.get("lily_ids") != step["ids"]:
        return None
    p = [math.exp(l - step["logsumexp"]) for l in step["logits"]]
    q = [math.exp(l - golden["logsumexp"]) for l in golden["at_lily_ids"]]
    kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0)
    rest_p, rest_q = 1.0 - sum(p), 1.0 - sum(q)
    if rest_p > 1e-12 and rest_q > 1e-12:
        kl += rest_p * math.log(rest_p / rest_q)
    chosen = step["ids"].index(step["chosen"])
    dlogp = abs((step["logits"][chosen] - step["logsumexp"]) - (golden["at_lily_ids"][chosen] - golden["logsumexp"]))
    return kl, dlogp


def main(lily_path: str, hf_path: str) -> int:
    lily = json.loads(Path(lily_path).read_text())
    hf = json.loads(Path(hf_path).read_text())
    prompt = lily["prompt_token_ids"]
    if hf["prompt_token_ids"][: len(prompt)] != prompt:
        print("prompt token ids differ between the two records", file=sys.stderr)
        return 2
    positions_ok = True
    if lily.get("positions") is not None:
        want = hf.get("positions")
        if want is None:
            print("candidate carries positions but the golden has none", file=sys.stderr)
            positions_ok = False
        else:
            same_ids = lily["positions"]["position_ids"] == want["position_ids"]
            same_delta = lily["positions"]["rope_deltas"] == want["rope_deltas"]
            positions_ok = same_ids and same_delta
            print(f"positions: {'exact' if same_ids else 'DIFFER'}, rope_deltas {lily['positions']['rope_deltas']} vs {want['rope_deltas']} ({'same' if same_delta else 'DIFFER'})")
    hf_top = {entry["position"]: entry for entry in hf["top8_last"]}
    greedy = hf.get("greedy") or []
    agree = 0
    total = 0
    worst_gap = 0.0
    kls: list[float] = []
    worst_dlogp = 0.0
    flip_margins: list[float] = []
    print(f"{'pos':>5} {'lily':>8} {'hf':>8} {'match':>5} {'top8∩':>5} {'max|Δlogit|':>12} {'KL':>9} {'|Δlogp|':>8} {'margin':>7}  lily top-3 / hf top-3")
    for step in lily["steps"]:
        pos = step["position"]
        if pos < len(hf["argmax"]):
            hf_arg = hf["argmax"][pos]
        elif pos + 1 - len(hf["argmax"]) < len(greedy):
            hf_arg = greedy[pos + 1 - len(hf["argmax"])]
        else:
            break
        match = hf_arg == step["chosen"]
        total += 1
        agree += match
        overlap = "-"
        gap = "-"
        kl_s = dlogp_s = "-"
        margin = step["logits"][0] - step["logits"][1] if len(step["logits"]) > 1 else float("nan")
        if not match:
            flip_margins.append(margin)
        hf_tops = ""
        if pos in hf_top:
            h = hf_top[pos]
            dist = distribution(step, h)
            if dist is not None:
                kls.append(dist[0])
                worst_dlogp = max(worst_dlogp, dist[1])
                kl_s, dlogp_s = f"{dist[0]:.2e}", f"{dist[1]:.4f}"
            shared = set(step["ids"][:8]) & set(h["ids"][:8])
            overlap = str(len(shared))
            h_map = dict(zip(h["ids"], h["logits"]))
            l_map = dict(zip(step["ids"], step["logits"]))
            diffs = [abs(l_map[i] - h_map[i]) for i in shared]
            if diffs:
                gap_v = max(diffs)
                worst_gap = max(worst_gap, gap_v)
                gap = f"{gap_v:.3f}"
            hf_tops = f"{step['ids'][:3]} / {h['ids'][:3]}"
        print(f"{pos:>5} {step['chosen']:>8} {hf_arg:>8} {'yes' if match else 'NO':>5} {overlap:>5} {gap:>12} {kl_s:>9} {dlogp_s:>8} {margin:>7.3f}  {hf_tops}")
    print(f"\nargmax agreement: {agree}/{total}; worst shared-id logit gap: {worst_gap:.3f}")
    if kls:
        kls.sort()
        print(f"KL(lily || golden) over {len(kls)} positions: mean {sum(kls) / len(kls):.2e}, "
              f"median {kls[len(kls) // 2]:.2e}, max {kls[-1]:.2e}; worst |Δ log p(chosen)|: {worst_dlogp:.4f}")
    if flip_margins:
        print(f"lily top-1/top-2 margin at the {len(flip_margins)} flipped positions: {', '.join(f'{m:.3f}' for m in flip_margins)}")
    print(f"lily text: {lily['text']!r}")
    return 0 if agree == total and positions_ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
