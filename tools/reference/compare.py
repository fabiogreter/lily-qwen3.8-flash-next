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
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


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
    print(f"{'pos':>5} {'lily':>8} {'hf':>8} {'match':>5} {'top8∩':>5} {'max|Δlogit|':>12}  lily top-3 / hf top-3")
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
        hf_tops = ""
        if pos in hf_top:
            h = hf_top[pos]
            shared = set(step["ids"]) & set(h["ids"])
            overlap = str(len(shared))
            h_map = dict(zip(h["ids"], h["logits"]))
            l_map = dict(zip(step["ids"], step["logits"]))
            diffs = [abs(l_map[i] - h_map[i]) for i in shared]
            if diffs:
                gap_v = max(diffs)
                worst_gap = max(worst_gap, gap_v)
                gap = f"{gap_v:.3f}"
            hf_tops = f"{step['ids'][:3]} / {h['ids'][:3]}"
        print(f"{pos:>5} {step['chosen']:>8} {hf_arg:>8} {'yes' if match else 'NO':>5} {overlap:>5} {gap:>12}  {hf_tops}")
    print(f"\nargmax agreement: {agree}/{total}; worst shared-id logit gap: {worst_gap:.3f}")
    print(f"lily text: {lily['text']!r}")
    return 0 if agree == total and positions_ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
