#!/usr/bin/env python3
"""Compare the sparse-attention block selections a lily probe recorded on its
decode steps with the ones the HF reference recorded for the same positions.

    .venv/bin/python tools/reference/compare_selection.py lily.json hf.json [layer]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(lily_path: str, hf_path: str, layer: str | None) -> int:
    lily = json.loads(Path(lily_path).read_text())
    hf = json.loads(Path(hf_path).read_text())
    layers = hf.get("selected_blocks", {})
    if not layers:
        print("HF golden has no selected_blocks (re-run hf_reference.py)", file=sys.stderr)
        return 2
    layer = layer or sorted(layers, key=int)[-1]
    hf_sel = layers[layer]
    print(f"HF layer {layer}; {'pos':>5} {'lily':>5} {'hf':>5} {'common':>6} {'only lily':>28} {'only hf':>28}")
    for step in lily["steps"]:
        pos = str(step["position"])
        dbg = step.get("debug") or {}
        if pos not in hf_sel or not dbg or dbg.get("selected_blocks") is None:
            continue
        mine = set(dbg["selected_blocks"])
        theirs = set(hf_sel[pos])
        only_mine = sorted(mine - theirs)
        only_theirs = sorted(theirs - mine)
        print(f"{pos:>5} {len(mine):>5} {len(theirs):>5} {len(mine & theirs):>6} {str(only_mine[:6]):>28} {str(only_theirs[:6]):>28}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None))
