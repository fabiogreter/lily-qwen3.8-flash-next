#!/usr/bin/env python3
"""Compare a candidate record (lily's output in the golden's own JSON-plus-npy
format) against a vision golden from hf_vision_reference.py.

    .venv/bin/python tools/reference/compare_vision.py <candidate.json> <golden.json> \
        [--atol X] [--rtol Y] [--max-atol Z] [--min-frac F] [--min-cosine C] [--max-rel-l2 L]

The golden's `kind` selects the comparison from docs/architecture.md, "How the tower was verified":

  preprocess  (1) image_grid_thw and resized size exact; pixel_values within
              tolerance: max abs error, max rel error, cosine, fraction within atol
  tower       (2) merged embeddings within tolerance (same metrics); the
              within-fraction threshold is the bf16 reference's own fraction on
              that image and cap (goldens/vision_tolerance_floors.json) minus
              TOWER_TOLERANCE["frac_margin"], falling back to the absolute
              min_frac when no floor is recorded; the pre-merger hidden states
              are reported when both records carry them
  positions   (3) input_ids, mm_token_type_ids, position_ids and rope_deltas exact
  forward     (4) argmax agreement through compare.py; the candidate may be a
              lily-probe record (`steps`) or an hf_reference.py golden (`argmax`),
              and the positions block is checked exactly when both have one

Tolerances come from the command line, else from the golden's `tolerance`
block, else from vision_golden.py. The full tensors are compared when both
`.npy` files are found (candidate paths resolve relative to the candidate JSON,
golden paths relative to goldens/), otherwise the shared 4 096-element sample.
Exit status 0 when every check passes, 1 otherwise, 2 on a malformed input.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare  # noqa: E402
from vision_golden import PREPROCESS_TOLERANCE, TOWER_TOLERANCE, compare_records, fmt, tower_floor_within  # noqa: E402


def tolerances(args, golden: dict, defaults: dict) -> dict:
    tol = dict(defaults)
    tol.update({k: v for k, v in (golden.get("tolerance") or {}).items() if v is not None})
    for key in ("atol", "rtol", "max_atol", "min_frac", "min_cosine", "max_rel_l2"):
        value = getattr(args, key, None)
        if value is not None:
            tol[key] = value
    return tol


def judge(metrics: dict, tol: dict) -> tuple[bool, list[str]]:
    if "ok" in metrics:
        return metrics["ok"], [metrics.get("reason", metrics.get("mode", ""))]
    reasons = []
    if metrics["fraction_within"] < tol.get("min_frac", 1.0):
        reasons.append(f"fraction within tolerance {metrics['fraction_within']:.6f} < {tol['min_frac']}")
    if tol.get("max_atol") is not None and metrics["max_abs_err"] > tol["max_atol"]:
        reasons.append(f"max abs error {fmt(metrics['max_abs_err'])} > {tol['max_atol']}")
    if tol.get("min_cosine") is not None and metrics["cosine"] < tol["min_cosine"]:
        reasons.append(f"cosine {metrics['cosine']:.6f} < {tol['min_cosine']}")
    if tol.get("max_rel_l2") is not None and metrics["rel_l2_err"] > tol["max_rel_l2"]:
        reasons.append(f"relative L2 error {fmt(metrics['rel_l2_err'])} > {tol['max_rel_l2']}")
    return not reasons, reasons


def report(label: str, metrics: dict, tol: dict) -> bool:
    ok, reasons = judge(metrics, tol)
    if "ok" in metrics:
        print(f"{label}: {'identical' if ok else 'FAIL'} ({'; '.join(reasons)})")
        return ok
    print(
        f"{label} [{metrics['mode']}, {metrics['elements']} elements]: max|err| {fmt(metrics['max_abs_err'])}, "
        f"mean|err| {fmt(metrics['mean_abs_err'])}, max rel {fmt(metrics['max_rel_err'])}, rel L2 {fmt(metrics['rel_l2_err'])}, "
        f"cosine {metrics['cosine']:.6f}, within(atol {tol['atol']}, rtol {tol['rtol']}) {metrics['fraction_within']:.6f}"
    )
    print(f"  worst element {metrics['worst_index']}: candidate {fmt(metrics['worst_candidate'])} vs golden {fmt(metrics['worst_golden'])}")
    print(f"  {'ok' if ok else 'FAIL: ' + '; '.join(reasons)}")
    return ok


def exact(label: str, a, b) -> bool:
    if a == b:
        print(f"{label}: exact match")
        return True
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            print(f"{label}: FAIL, length {len(a)} vs golden {len(b)}")
            return False
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                if isinstance(x, list):
                    # nested (position axes): recurse to name the axis
                    return exact(f"{label}[{i}]", x, y)
                print(f"{label}: FAIL, first difference at index {i}: candidate {x} vs golden {y}")
                return False
    print(f"{label}: FAIL, candidate {a!r} vs golden {b!r}")
    return False


def cmp_preprocess(cand: dict, gold: dict, cand_dir: Path, gold_dir: Path, args) -> bool:
    tol = tolerances(args, gold, PREPROCESS_TOLERANCE)
    ok = exact("image_grid_thw", cand.get("image_grid_thw"), gold["image_grid_thw"])
    ok &= exact("resized_hw", cand.get("resized_hw"), gold["resized_hw"])
    if cand.get("cap") and cand["cap"] != gold["cap"]:
        print(f"note: cap differs: candidate {cand['cap']} vs golden {gold['cap']}")
    metrics = compare_records(cand["pixel_values"], gold["pixel_values"], cand_dir, gold_dir, tol["atol"], tol["rtol"])
    ok &= report("pixel_values", metrics, tol)
    return ok


def cmp_tower(cand: dict, gold: dict, cand_dir: Path, gold_dir: Path, args) -> bool:
    tol = tolerances(args, gold, TOWER_TOLERANCE)
    ok = True
    # The within-fraction gate relative to the measured bf16 floor (see
    # vision_golden.TOWER_TOLERANCE); a --min-frac on the command line wins.
    image = Path(gold["image"]["file"]).stem
    floor = tower_floor_within(image, gold["cap"])
    if getattr(args, "min_frac", None) is not None:
        print(f"within-fraction threshold {tol['min_frac']} from the command line")
    elif floor is None:
        print(f"no bf16 floor recorded for {image} at cap {gold['cap']}: using the absolute min_frac {tol['min_frac']}")
    else:
        tol["min_frac"] = floor - tol["frac_margin"]
        print(f"within-fraction floor (bf16 reference vs f32 on {image}) {floor:.6f}, margin {tol['frac_margin']}, threshold {tol['min_frac']:.6f}")
    if cand.get("input") and cand["input"].get("pixel_values_sha256") != gold["input"]["pixel_values_sha256"]:
        print("note: the candidate tower ran on different pixel_values than the golden (expected when it is fed lily's own preprocessing)")
    metrics = compare_records(cand["merged"], gold["merged"], cand_dir, gold_dir, tol["atol"], tol["rtol"])
    ok &= report("merged embeddings", metrics, tol)
    if "pre_merger" in cand and "pre_merger" in gold:
        pre = compare_records(cand["pre_merger"], gold["pre_merger"], cand_dir, gold_dir, tol["atol"], tol["rtol"])
        report("pre-merger hidden states (reported, not judged)", pre, dict(tol, min_frac=0.0, max_atol=None, min_cosine=None))
    return ok


def cmp_positions(cand: dict, gold: dict) -> bool:
    ok = exact("input_ids", cand.get("input_ids"), gold["input_ids"])
    ok &= exact("mm_token_type_ids", cand.get("mm_token_type_ids"), gold["mm_token_type_ids"])
    ok &= exact("image_grid_thw", cand.get("image_grid_thw"), gold["image_grid_thw"])
    for axis, name in enumerate(("temporal", "height", "width")):
        c = cand.get("position_ids")
        ok &= exact(f"position_ids[{name}]", c[axis] if c else None, gold["position_ids"][axis])
    ok &= exact("rope_deltas", cand.get("rope_deltas"), gold["rope_deltas"])
    return ok


def as_probe_record(record: dict) -> dict:
    """An hf_reference.py-style golden viewed as a lily-probe record for compare.py."""
    if "steps" in record:
        return record
    steps = [{"position": t["position"], "chosen": record["argmax"][t["position"]], "ids": t["ids"], "logits": t["logits"]} for t in record["top8_last"]]
    return {"prompt_token_ids": record["prompt_token_ids"], "steps": steps, "text": record.get("text", "")}


def cmp_forward(cand: dict, gold: dict, cand_path: Path, gold_path: Path) -> bool:
    ok = True
    if cand.get("positions") and gold.get("positions"):
        ok &= cmp_positions(cand["positions"], gold["positions"])
    if "steps" in cand:
        return compare.main(str(cand_path), str(gold_path)) == 0 and ok
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(as_probe_record(cand), f)
        tmp = f.name
    try:
        return compare.main(tmp, str(gold_path)) == 0 and ok
    finally:
        Path(tmp).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate")
    parser.add_argument("golden")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--max-atol", dest="max_atol", type=float)
    parser.add_argument("--min-frac", dest="min_frac", type=float)
    parser.add_argument("--min-cosine", dest="min_cosine", type=float)
    parser.add_argument("--max-rel-l2", dest="max_rel_l2", type=float)
    args = parser.parse_args(argv)

    cand_path, gold_path = Path(args.candidate), Path(args.golden)
    cand = json.loads(cand_path.read_text())
    gold = json.loads(gold_path.read_text())
    kind = gold.get("kind")
    if kind is None:
        print("golden has no `kind`; is it an hf_vision_reference.py golden?", file=sys.stderr)
        return 2
    if cand.get("kind") not in (None, kind):
        print(f"candidate kind {cand.get('kind')} does not match golden kind {kind}", file=sys.stderr)
        return 2
    print(f"comparison: {kind} ({gold_path.name})")
    if kind == "preprocess":
        ok = cmp_preprocess(cand, gold, cand_path.parent, gold_path.parent, args)
    elif kind == "tower":
        ok = cmp_tower(cand, gold, cand_path.parent, gold_path.parent, args)
    elif kind == "positions":
        ok = cmp_positions(cand, gold)
    elif kind == "forward":
        ok = cmp_forward(cand, gold, cand_path, gold_path)
    else:
        print(f"unknown golden kind {kind!r}", file=sys.stderr)
        return 2
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
