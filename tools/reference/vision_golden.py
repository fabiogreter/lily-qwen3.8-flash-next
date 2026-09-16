"""Shared pieces of the vision reference goldens: the pixel cap, the JSON-plus-npy
record for large tensors, and the tolerance metrics `compare_vision.py` reports.

A golden is one JSON file under `tools/reference/goldens/`. Tensors too large to
commit (pixel_values, tower embeddings) are written as `.npy` into
`goldens/large/` (ignored by git); the JSON keeps shape, dtype, sha256 of the
raw bytes, summary statistics and a seeded sample of 4 096 elements with their
flat indices, which is enough to compare a re-implementation within tolerance
when the `.npy` is not at hand. When it is, the full tensor is compared.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

# The server-side pixel cap (docs/vision-support-plan.md, item 1). One language token
# covers 32 x 32 pixels (patch 16, merge 2); 2 048 tokens is about 1.5 s of prefill and
# 8 192 tower patches. A native 1920 x 1080 screenshot (2 073 600 px) passes untouched;
# a Retina 3840 x 2160 capture is scaled to 1920 x 1056. The minimum is the checkpoint's
# `size.shortest_edge`.
DEFAULT_MAX_PIXELS = 2048 * 32 * 32  # 2 097 152
DEFAULT_MIN_PIXELS = 65536
CHECKPOINT_MAX_PIXELS = 16777216  # the checkpoint's own `size.longest_edge`: effectively no cap

SAMPLE_SIZE = 4096
SAMPLE_SEED = 0

# Default tolerances for compare_vision.py, in the units of the compared tensor.
# The numbers behind them are in goldens/vision_tolerance_floors.json (written by
# `hf_vision_reference.py measure-preprocess` / `measure-tower`) and in
# tools/reference/VISION.md.
#
# Preprocessing: pixel_values are normalised to [-1, 1], so one uint8 level is
# 2/255 = 0.00784. PIL.Image.BICUBIC and the reference's torchvision uint8 path
# differ by at most one level (on a few 1e-5 of elements); a bf16 copy of the
# reference moves by at most 0.00193. `atol` is one level plus that; `max_atol`
# allows three levels for isolated pixels; `min_frac` of the elements must be
# within `atol`.
PREPROCESS_TOLERANCE = {"atol": 2 / 255 + 0.002, "rtol": 0.0, "max_atol": 3 * 2 / 255 + 0.002, "min_frac": 0.999}

# Tower: the reference tower in bfloat16 (on MPS) against itself in float32 (CPU)
# on the same input is the floor for a bf16 Metal implementation. Measured on the
# merged [tokens, 2560] output (std about 0.028) over the four test images:
# relative L2 error 0.050 to 0.078, cosine 0.9969 to 0.9988, max abs error up to
# 0.127, and 99.94 % or more of the elements within 0.02 + 0.05 |golden|. The
# pre-merger hidden states have per-token norms spanning four orders of
# magnitude, so they are reported, not judged.
TOWER_TOLERANCE = {"atol": 0.02, "rtol": 0.05, "max_atol": None, "min_frac": 0.999, "min_cosine": 0.995, "max_rel_l2": 0.10}

GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"
LARGE_DIR = GOLDENS_DIR / "large"
IMAGES_DIR = Path(__file__).resolve().parent / "images"


def sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def cap_dict(min_pixels: int, max_pixels: int) -> dict:
    """The cap as the reference processor receives it (`size=` of Qwen2VLImageProcessor)."""
    return {"shortest_edge": int(min_pixels), "longest_edge": int(max_pixels)}


def tensor_record(array: np.ndarray, npy: Path | None, name: str) -> dict:
    """Summary record of a tensor; writes the full array to `npy` when given."""
    flat = np.ascontiguousarray(array).reshape(-1)
    as_float = flat.astype(np.float64)
    rng = np.random.default_rng(SAMPLE_SEED)
    n = min(SAMPLE_SIZE, flat.size)
    indices = np.sort(rng.choice(flat.size, size=n, replace=False))
    record = {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha256_bytes(np.ascontiguousarray(array).tobytes()),
        "stats": {
            "mean": float(as_float.mean()),
            "std": float(as_float.std()),
            "min": float(as_float.min()),
            "max": float(as_float.max()),
            "abs_mean": float(np.abs(as_float).mean()),
        },
        "sample": {
            "seed": SAMPLE_SEED,
            "indices": indices.tolist(),
            # 9 significant digits round-trip any float32 exactly.
            "values": [float(f"{v:.9g}") for v in flat[indices].astype(np.float64).tolist()],
        },
    }
    if npy is not None:
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, np.ascontiguousarray(array))
        record["npy"] = str(npy.relative_to(GOLDENS_DIR)) if npy.is_relative_to(GOLDENS_DIR) else str(npy)
    return record


def load_tensor(record: dict, json_dir: Path) -> np.ndarray | None:
    """The full tensor behind a record, if its .npy exists (checked against the sha256)."""
    npy = record.get("npy")
    if npy is None:
        return None
    candidates = [Path(npy), json_dir / npy, GOLDENS_DIR / npy]
    for path in candidates:
        if path.exists():
            array = np.load(path)
            if list(array.shape) != list(record["shape"]):
                raise ValueError(f"{path}: shape {array.shape} does not match record {record['shape']}")
            if sha256_bytes(np.ascontiguousarray(array).tobytes()) != record["sha256"]:
                raise ValueError(f"{path}: sha256 does not match its record")
            return array
    return None


def compare_values(candidate: np.ndarray, golden: np.ndarray, atol: float, rtol: float) -> dict:
    """Elementwise metrics of `candidate` against `golden` (both flat, same length)."""
    c = candidate.astype(np.float64).reshape(-1)
    g = golden.astype(np.float64).reshape(-1)
    if c.shape != g.shape:
        raise ValueError(f"cannot compare {c.shape} against {g.shape}")
    diff = np.abs(c - g)
    denom = np.maximum(np.abs(g), 1e-12)
    within = diff <= atol + rtol * np.abs(g)
    gn = np.linalg.norm(g)
    cn = np.linalg.norm(c)
    cosine = float(np.dot(c, g) / (cn * gn)) if cn > 0 and gn > 0 else float("nan")
    worst = int(np.argmax(diff))
    return {
        "elements": int(c.size),
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "max_rel_err": float((diff / denom).max()),
        "rel_l2_err": float(np.linalg.norm(c - g) / gn) if gn > 0 else float("nan"),
        "cosine": cosine,
        "fraction_within": float(within.mean()),
        "worst_index": worst,
        "worst_candidate": float(c[worst]),
        "worst_golden": float(g[worst]),
        "atol": atol,
        "rtol": rtol,
    }


def compare_records(cand: dict, gold: dict, cand_dir: Path, gold_dir: Path, atol: float, rtol: float) -> dict:
    """Compare two tensor records: on the full tensors if both .npy files are at
    hand, else on the shared seeded sample. Shape must match; dtype may differ."""
    if list(cand["shape"]) != list(gold["shape"]):
        return {"ok": False, "reason": f"shape {cand['shape']} != golden {gold['shape']}"}
    if cand["sha256"] == gold["sha256"]:
        return {"ok": True, "mode": "identical", "sha256": gold["sha256"]}
    c_full = load_tensor(cand, cand_dir)
    g_full = load_tensor(gold, gold_dir)
    if c_full is not None and g_full is not None:
        metrics = compare_values(c_full, g_full, atol, rtol)
        metrics["mode"] = "full"
    else:
        if cand["sample"]["indices"] != gold["sample"]["indices"]:
            return {"ok": False, "reason": "sample indices differ (seed or shape mismatch); provide the .npy files"}
        metrics = compare_values(np.array(cand["sample"]["values"]), np.array(gold["sample"]["values"]), atol, rtol)
        metrics["mode"] = f"sample of {len(gold['sample']['indices'])}"
    return metrics


def peak_rss_gb() -> float:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return rss / 2**30 if rss > 2**40 / 1024 else rss / 2**20


def run_meta(started: float, extra: dict | None = None) -> dict:
    meta = {"seconds": round(time.time() - started, 2), "peak_rss_gb": round(peak_rss_gb(), 2)}
    if extra:
        meta.update(extra)
    return meta


_NUMBER_ARRAY = re.compile(r"\[\s+(-?\d[^\[\]{}]*?)\s+\]", re.S)


def write_json(path: Path, record: dict) -> None:
    """Indented JSON with arrays of plain numbers folded onto one line each, so a
    4 096-element sample costs one line, not 4 096."""
    text = json.dumps(record, indent=1, allow_nan=False)
    text = _NUMBER_ARRAY.sub(lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]", text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.6g}"
