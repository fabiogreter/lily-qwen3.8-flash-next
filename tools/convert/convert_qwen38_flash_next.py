#!/usr/bin/env python3
"""Convert the raw Hugging Face Qwen3.8-Flash-Next checkpoint into lily's
`qwen4_exp-affine-v1` layout (see docs/qwen38-flash-next-checkpoint-format.md).

The source is streamed tensor by tensor; nothing close to the 336 GB source is
ever resident. Quantization is `mlx.core.quantize`, so the packed layout is
bit-identical to what lily's Q4/Q8 Metal kernels consume.

    .venv/bin/python tools/convert/convert_qwen38_flash_next.py \
        --src ~/models/Qwen3.8-Flash-Next \
        --dst ~/models/Qwen3.8-Flash-Next-lily-q4 \
        [--layers 4] [--dry-run]

`mlx` needs a Metal device even for CPU arrays, so a real conversion must run
outside any GPU-less sandbox; `--dry-run` never imports it.

`--mtp-only` adds the multi-token-prediction draft head (the `mtp.*` tensors:
one attention+MoE block plus its input projections and output mixer) to an
existing conversion as extra `mtp-*.safetensors` shards, merging them into the
index and config so the engine can run speculative decoding.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT_NAME = "qwen4_exp-affine-v1"
SOURCE_REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
LAYER_PREFIX = "model.language_model.layers."
DROP_PREFIXES = ("model.visual.",)
MTP_PREFIX = "mtp."
COPIED_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
)

# PLE hashing constants, mirrored from transformers' modeling_qwen4_exp.py.
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007

_SOURCE_ITEM_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8}


@dataclass(frozen=True)
class Quant:
    bits: int
    group_size: int

    def packed_bytes(self, shape: tuple[int, ...]) -> int:
        numel = math.prod(shape)
        groups = numel // self.group_size
        return numel * self.bits // 8 + 2 * groups * 2  # codes + bf16 scales/biases


@dataclass(frozen=True)
class Plan:
    """What happens to one source tensor."""

    kind: str  # "drop" | "copy" | "quant" | "expert_gate_up" | "ple_const"
    quant: Quant | None = None


@dataclass(frozen=True)
class SourceTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: Path
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass
class OutTensor:
    """One output tensor: a safetensors dtype tag plus its raw little-endian bytes."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes | memoryview

    @property
    def nbytes(self) -> int:
        return len(self.data)


@dataclass
class Totals:
    bytes_by_category: dict[str, int] = field(default_factory=dict)
    tensors: int = 0

    def add(self, category: str, nbytes: int) -> None:
        self.bytes_by_category[category] = self.bytes_by_category.get(category, 0) + nbytes
        self.tensors += 1

    def total(self) -> int:
        return sum(self.bytes_by_category.values())


# --- Planning -----------------------------------------------------------------


def read_source_headers(src: Path) -> list[SourceTensor]:
    """Parses every shard header named by the index; no tensor data is read."""
    index = json.loads((src / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    tensors: list[SourceTensor] = []
    for shard_name in shards:
        shard = src / shard_name
        with shard.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        base = 8 + header_len
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            tensors.append(
                SourceTensor(
                    name=name,
                    dtype=entry["dtype"],
                    shape=tuple(entry["shape"]),
                    shard=shard,
                    start=base + start,
                    end=base + end,
                )
            )
    tensors.sort(key=lambda t: (str(t.shard), t.start))
    return tensors


def layer_index(name: str) -> int | None:
    if not name.startswith(LAYER_PREFIX):
        return None
    return int(name[len(LAYER_PREFIX) :].split(".", 1)[0])


Q4 = Quant(4, 64)
Q8 = Quant(8, 64)

_Q4_SUFFIXES = (
    r"\.self_attn\.(q|k|v|o)_proj\.weight$",
    r"\.linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)\.weight$",
    r"\.mlp\.shared_expert\.(gate|up|down)_proj\.weight$",
    r"\.mlp\.experts\.down_proj$",
)
_Q8_SUFFIXES = (
    r"^mtp\.fc_(embedding|hidden)\.weight$",
    r"\.mlp\.gate\.weight$",
    r"\.mlp\.shared_expert_gate\.weight$",
    r"hyper_connection(_mixer)?\.input_mix_weight_(down|up)\.weight$",
    r"\.block_inject_weight\.weight$",
    r"\.ple\.(key_proj|value_proj)\.weight$",
    r"\.self_attn\.indexer\.index_qk_proj\.weight$",
)
_COPY_SUFFIXES = (
    r"^mtp\.pre_fc_norm_(embedding|hidden)\.weight$",
    r"\.hc_norm\.weight$",
    r"\.self_attn\.(q_norm|k_norm)\.weight$",
    r"\.self_attn\.indexer\.(q_layernorm|k_layernorm)\.weight$",
    r"\.linear_attn\.(A_log|dt_bias|norm\.weight|conv1d\.weight)$",
    r"\.ple\.(norm_key|norm_query|norm_conv)\.weight$",
    r"\.ple\.conv1d\.weight$",
)
_PLE_CONSTS = (
    "ple.ple_embedding.layer_multipliers",
    "ple.ple_embedding.ngram_heads_vocab_sizes",
    "ple.ple_embedding.ngram_heads_offsets",
)
_NGRAM_SHARD = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$")


def plan_tensor(t: SourceTensor, keep_layers: int, ngram: Quant, mtp: bool = True) -> Plan:
    name = t.name
    if name.startswith(DROP_PREFIXES) or (not mtp and name.startswith(MTP_PREFIX)):
        return Plan("drop")
    li = layer_index(name)
    if li is not None and li >= keep_layers:
        return Plan("drop")
    if name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
        return Plan("quant", Q4)
    if name.endswith(".mlp.experts.gate_up_proj"):
        return Plan("expert_gate_up", Q4)
    if _NGRAM_SHARD.search(name):
        return Plan("quant", ngram)
    if any(name.endswith(s) for s in _PLE_CONSTS):
        return Plan("ple_const")
    for pattern in _Q4_SUFFIXES:
        if re.search(pattern, name):
            return Plan("quant", Q4)
    for pattern in _Q8_SUFFIXES:
        if re.search(pattern, name):
            return Plan("quant", Q8)
    for pattern in _COPY_SUFFIXES:
        if re.search(pattern, name):
            return Plan("copy")
    raise ValueError(f"no conversion rule for tensor {name} {t.shape} {t.dtype}")


def category(name: str) -> str:
    if name.startswith(MTP_PREFIX):
        return "mtp"
    if "ngram_embedding.shard_" in name:
        return "ngram"
    if ".mlp.experts." in name:
        return "experts"
    if name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
        return "embed/lm_head"
    return "dense"


def gate_up_output_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    e, two_i, h = shape
    return (e, two_i // 2, h)


def estimate(tensors: list[SourceTensor], keep_layers: int, ngram: Quant, mtp: bool = True) -> Totals:
    totals = Totals()
    for t in tensors:
        plan = plan_tensor(t, keep_layers, ngram, mtp)
        cat = category(t.name)
        if plan.kind in ("drop", "ple_const"):
            continue
        if plan.kind == "copy":
            totals.add(cat, t.nbytes)
        elif plan.kind == "quant":
            assert plan.quant is not None
            totals.add(cat, plan.quant.packed_bytes(t.shape))
        elif plan.kind == "expert_gate_up":
            assert plan.quant is not None
            totals.add(cat, plan.quant.packed_bytes(gate_up_output_shape(t.shape)))
            totals.add(cat, plan.quant.packed_bytes(gate_up_output_shape(t.shape)))
    return totals


# --- PLE constant verification -------------------------------------------------


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(vocab: int, ngram_size: int, ple_layer_index: int, seed: int) -> list[int]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(vocab, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    out = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        out.append(2 * (_splitmix64(value) % half_bound) + 1)
    return out


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for d in range(3, math.isqrt(value) + 1, 2):
        if value % d == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def head_vocab_sizes(base: int, heads: int, ple_layer_index: int) -> tuple[list[int], list[int]]:
    """(sizes, offsets): head h uses the (global_h + 1)-th prime after base - 1."""
    sizes, offsets, total = [], [], 0
    for head in range(heads):
        size = _find_nth_prime_after(base - 1, ple_layer_index * heads + head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    return sizes, offsets


def verify_ple_constants(text_cfg: dict, consts: dict[str, list[int]]) -> dict:
    vocab = text_cfg["vocab_size"]
    ngram_size = text_cfg.get("ngram_size", 3)
    heads_per_ngram = text_cfg.get("heads_per_ngram", 8)
    base = text_cfg.get("ngram_vocab_size_base", 20_000_000)
    seed = text_cfg.get("seed", 1234)
    heads = (ngram_size - 1) * heads_per_ngram
    sizes, offsets = head_vocab_sizes(base, heads, 0)
    expected = {
        "layer_multipliers": build_layer_multipliers(vocab, ngram_size, 0, seed),
        "ngram_heads_vocab_sizes": sizes,
        "ngram_heads_offsets": offsets,
    }
    for key, value in expected.items():
        if list(consts[key]) != value:
            raise SystemExit(f"PLE constant {key} in checkpoint {consts[key]} != recomputed {value}")
    divisor = text_cfg.get("make_ngram_vocab_size_divisible_by", 128)
    total = sum(sizes)
    return {
        **expected,
        "ngram_size": ngram_size,
        "heads_per_ngram": heads_per_ngram,
        "seed": seed,
        "total_vocab_size": total,
        "padded_vocab_size": math.ceil(total / divisor) * divisor,
    }


# --- Data path ----------------------------------------------------------------


def read_source_bytes(t: SourceTensor) -> bytes:
    with t.shard.open("rb") as f:
        f.seek(t.start)
        return f.read(t.nbytes)


def bf16_bytes_to_mx(raw: bytes | memoryview, shape: tuple[int, ...]):
    import mlx.core as mx

    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    return mx.array(u16).view(mx.bfloat16)


def mx_bf16_to_out(a) -> OutTensor:
    import mlx.core as mx

    u16 = np.array(a.view(mx.uint16))
    return OutTensor("BF16", tuple(a.shape), u16.tobytes())


def mx_u32_to_out(a) -> OutTensor:
    return OutTensor("U32", tuple(a.shape), np.array(a).tobytes())


def quantize_mx(w, quant: Quant):
    """`(codes u32, scales bf16, biases bf16)` in MLX affine layout, evaluated."""
    import mlx.core as mx

    codes, scales, biases = mx.quantize(w, group_size=quant.group_size, bits=quant.bits)
    mx.eval(codes, scales, biases)
    return codes, scales, biases


def dequant_max_error(w, codes, scales, biases, quant: Quant) -> float:
    import mlx.core as mx

    deq = mx.dequantize(codes, scales, biases, group_size=quant.group_size, bits=quant.bits)
    return float(mx.max(mx.abs(deq.astype(mx.float32) - w.astype(mx.float32))))


def write_safetensors(path: Path, tensors: dict[str, OutTensor]) -> None:
    """Minimal safetensors writer: 8-byte LE header length, JSON header, raw data."""
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, t in tensors.items():
        header[name] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": [offset, offset + t.nbytes]}
        offset += t.nbytes
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for t in tensors.values():
            f.write(t.data)


class ShardWriter:
    """Accumulates output tensors and flushes ~shard_bytes safetensors files."""

    def __init__(self, dst: Path, shard_bytes: int, stem: str = "model"):
        self.dst = dst
        self.shard_bytes = shard_bytes
        self.stem = stem
        self.pending: dict[str, OutTensor] = {}
        self.pending_bytes = 0
        self.files: list[Path] = []
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0

    def add(self, name: str, tensor: OutTensor) -> None:
        if name in self.weight_map or name in self.pending:
            raise ValueError(f"duplicate output tensor {name}")
        if self.pending and self.pending_bytes + tensor.nbytes > self.shard_bytes:
            self.flush()
        self.pending[name] = tensor
        self.pending_bytes += tensor.nbytes
        self.total_bytes += tensor.nbytes

    def flush(self) -> None:
        if not self.pending:
            return
        path = self.dst / f"tmp-shard-{len(self.files):05d}.safetensors"
        write_safetensors(path, self.pending)
        for name in self.pending:
            self.weight_map[name] = path.name
        self.files.append(path)
        self.pending = {}
        self.pending_bytes = 0

    def finish(self, merge: bool = False) -> None:
        """Names the shards and writes the index; with `merge`, the new tensors
        are added to the directory's existing index instead."""
        self.flush()
        n = len(self.files)
        renamed: dict[str, str] = {}
        for i, path in enumerate(self.files):
            final = f"{self.stem}-{i + 1:05d}-of-{n:05d}.safetensors"
            path.rename(self.dst / final)
            renamed[path.name] = final
        weight_map = {k: renamed[v] for k, v in self.weight_map.items()}
        index_path = self.dst / "model.safetensors.index.json"
        total = self.total_bytes
        if merge:
            existing = json.loads(index_path.read_text())
            clash = set(existing["weight_map"]) & set(weight_map)
            if clash:
                raise SystemExit(f"index already holds {sorted(clash)[:3]}...; refusing to merge")
            weight_map = {**existing["weight_map"], **weight_map}
            total += existing.get("metadata", {}).get("total_size", 0)
        index = {"metadata": {"total_size": total}, "weight_map": weight_map}
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True))


def mtp_block(text_cfg: dict) -> dict:
    """What the engine needs to know about the converted draft head."""
    mtp = text_cfg.get("mtp") or {}
    return {
        "layers": text_cfg.get("mtp_num_hidden_layers", mtp.get("num_hidden_layers", 1)),
        "layer_types": mtp.get("layer_types", ["full_attention"]),
        "rope_theta": mtp.get("rope_theta", text_cfg["rope_parameters"]["rope_theta"]),
        "quantization": {"default": {"bits": Q4.bits, "group_size": Q4.group_size, "mode": "affine"}},
    }


def write_config(src: Path, dst: Path, keep_layers: int, ngram: Quant, ple: dict, revision: str | None, mtp: bool) -> None:
    cfg = json.loads((src / "config.json").read_text())
    text = cfg["text_config"]
    text["num_hidden_layers"] = keep_layers
    text["layer_types"] = text["layer_types"][:keep_layers]
    cfg["lily"] = {
        "format": FORMAT_NAME,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": revision,
        "layers": keep_layers,
        "quantization": {
            "default": {"bits": Q4.bits, "group_size": Q4.group_size, "mode": "affine"},
            "ngram_embedding": {"bits": ngram.bits, "group_size": ngram.group_size, "mode": "affine"},
            "q8_suffixes": list(_Q8_SUFFIXES),
        },
        "ple": ple,
        "mtp": mtp_block(text) if mtp else None,
        "dropped": list(DROP_PREFIXES) + ([] if mtp else [MTP_PREFIX]),
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))


def add_mtp_to_config(dst: Path) -> None:
    """Marks an existing conversion's config as carrying the draft head."""
    path = dst / "config.json"
    cfg = json.loads(path.read_text())
    lily = cfg["lily"]
    if lily.get("mtp"):
        raise SystemExit(f"{path} already declares an MTP block")
    lily["mtp"] = mtp_block(cfg["text_config"])
    lily["dropped"] = [p for p in lily.get("dropped", []) if p != MTP_PREFIX]
    path.write_text(json.dumps(cfg, indent=2))


def source_revision(src: Path) -> str | None:
    """The HF snapshot revision, if the download cache left a metadata file."""
    for meta in (src / ".cache" / "huggingface" / "download").glob("config.json.metadata"):
        lines = meta.read_text().splitlines()
        if lines:
            return lines[0].strip()
    return None


def fmt_gb(n: int) -> str:
    return f"{n / 1e9:8.2f} GB"


def print_totals(title: str, totals: Totals) -> None:
    print(f"\n{title}")
    for cat in sorted(totals.bytes_by_category):
        print(f"  {cat:14s} {fmt_gb(totals.bytes_by_category[cat])}")
    print(f"  {'total':14s} {fmt_gb(totals.total())}   ({totals.tensors} output tensors)")


def convert(args: argparse.Namespace) -> None:
    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    ngram = Quant(args.ngram_bits, args.ngram_group)
    tensors = read_source_headers(src)
    cfg = json.loads((src / "config.json").read_text())
    text_cfg = cfg["text_config"]
    keep_layers = args.layers or text_cfg["num_hidden_layers"]
    if keep_layers < 1 or keep_layers > text_cfg["num_hidden_layers"]:
        raise SystemExit(f"--layers must be in 1..{text_cfg['num_hidden_layers']}")

    mtp = not args.no_mtp
    if args.mtp_only:
        # Only the draft head, appended to a finished conversion.
        tensors = [t for t in tensors if t.name.startswith(MTP_PREFIX)]
        if not tensors:
            raise SystemExit("source has no mtp.* tensors")
        if not (dst / "model.safetensors.index.json").exists():
            raise SystemExit(f"{dst} is not a finished conversion (no index); run a full conversion first")
        if any(dst.glob("mtp-*.safetensors")):
            raise SystemExit(f"{dst} already holds mtp shards; refusing to overwrite")
        mtp = True
    totals = estimate(tensors, keep_layers, ngram, mtp)
    what = "the MTP draft head" if args.mtp_only else f"{keep_layers} layers (ngram {ngram.bits}-bit g{ngram.group_size}{', no mtp' if not mtp else ''})"
    print_totals(f"planned output for {what}", totals)
    if args.dry_run:
        return

    dst.mkdir(parents=True, exist_ok=True)
    if not args.mtp_only:
        if any(dst.glob("model-*.safetensors")):
            raise SystemExit(f"{dst} already holds shards; refusing to overwrite")
        for name in COPIED_FILES:
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)

    planned = [(t, plan_tensor(t, keep_layers, ngram, mtp)) for t in tensors]
    planned = [(t, p) for t, p in planned if p.kind != "drop"]
    src_bytes = sum(t.nbytes for t, _ in planned)
    quant_positions = [i for i, (_, p) in enumerate(planned) if p.kind == "quant"]
    check_idx = set(np.random.default_rng(0).choice(quant_positions, size=min(3, len(quant_positions)), replace=False).tolist())

    writer = ShardWriter(dst, int(args.shard_bytes), stem="mtp" if args.mtp_only else "model")
    ple_consts: dict[str, list[int]] = {}
    checked: list[tuple[str, float]] = []
    started = time.time()
    done_bytes = 0
    for i, (t, plan) in enumerate(planned):
        raw = read_source_bytes(t)
        if plan.kind == "ple_const":
            if t.dtype != "I64":
                raise SystemExit(f"{t.name} is {t.dtype}, expected I64")
            ple_consts[t.name.rsplit(".", 1)[1]] = np.frombuffer(raw, dtype=np.int64).tolist()
        elif plan.kind == "copy":
            writer.add(t.name, OutTensor(t.dtype, t.shape, raw))
        elif plan.kind == "quant":
            assert plan.quant is not None and t.dtype == "BF16"
            w = bf16_bytes_to_mx(raw, t.shape)
            codes, scales, biases = quantize_mx(w, plan.quant)
            base = t.name.removesuffix(".weight")
            writer.add(f"{base}.weight", mx_u32_to_out(codes))
            writer.add(f"{base}.scales", mx_bf16_to_out(scales))
            writer.add(f"{base}.biases", mx_bf16_to_out(biases))
            if i in check_idx:
                checked.append((t.name, dequant_max_error(w, codes, scales, biases, plan.quant)))
        elif plan.kind == "expert_gate_up":
            assert plan.quant is not None and t.dtype == "BF16"
            w = bf16_bytes_to_mx(raw, t.shape)
            half = t.shape[1] // 2
            base = t.name.removesuffix("gate_up_proj")
            for part, rows in (("gate_proj", slice(0, half)), ("up_proj", slice(half, 2 * half))):
                codes, scales, biases = quantize_mx(w[:, rows, :], plan.quant)
                writer.add(f"{base}{part}.weight", mx_u32_to_out(codes))
                writer.add(f"{base}{part}.scales", mx_bf16_to_out(scales))
                writer.add(f"{base}{part}.biases", mx_bf16_to_out(biases))
        done_bytes += t.nbytes
        if i % 25 == 0 or i + 1 == len(planned):
            elapsed = time.time() - started
            print(
                f"[{i + 1:5d}/{len(planned)}] {fmt_gb(done_bytes)} / {fmt_gb(src_bytes)} read, "
                f"{done_bytes / max(elapsed, 1e-9) / 1e9:5.2f} GB/s, {elapsed:6.0f}s  {t.name[-64:]}",
                flush=True,
            )
    writer.finish(merge=args.mtp_only)

    if args.mtp_only:
        add_mtp_to_config(dst)
    else:
        ple = verify_ple_constants(text_cfg, ple_consts) if ple_consts else {}
        write_config(src, dst, keep_layers, ngram, ple, source_revision(src), mtp)

    elapsed = time.time() - started
    print(f"\nwrote {len(writer.files)} shards, {fmt_gb(writer.total_bytes)} in {elapsed:.0f}s to {dst}")
    for name, err in checked:
        print(f"  dequant self-check {name}: max |deq - bf16| = {err:.4g}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="raw HF checkpoint directory")
    parser.add_argument("--dst", required=True, help="output directory")
    parser.add_argument("--layers", type=int, default=None, help="keep only the first N decoder layers")
    parser.add_argument("--ngram-bits", type=int, default=4, choices=(2, 3, 4, 8))
    parser.add_argument("--ngram-group", type=int, default=32, choices=(32, 64, 128))
    parser.add_argument("--shard-bytes", type=float, default=2e9)
    parser.add_argument("--no-mtp", action="store_true", help="drop the mtp.* draft head (it is converted by default)")
    parser.add_argument("--mtp-only", action="store_true", help="append only the mtp.* draft head to an existing conversion in --dst")
    parser.add_argument("--dry-run", action="store_true", help="only print the size plan")
    convert(parser.parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
