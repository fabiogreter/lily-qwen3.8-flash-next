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

`--q8 GROUPS` stores a comma-separated subset of five projection groups at Q8
group 64 instead of Q4: `attn` (attention `q/k/v/o_proj`, trunk and draft
head), `gdn` (every Gated DeltaNet projection, `in_proj_qkv/z/a/b` and
`out_proj`), `shared` (the shared expert's `gate/up/down_proj`, trunk and draft
head), `head` (`lm_head`) and `embed` (`embed_tokens`). `--q8-dense` is
`--q8 attn,gdn,shared,head` and `--q8-embed` is `--q8 embed`; the flags
combine. The routed experts stay Q4. The choice is recorded in
`lily.quantization` (the `q8_dense` / `q8_embed` flags when they can spell it,
otherwise `q8_groups`, plus the extended `q8_suffixes`), which the loader
follows; without any of them the output is byte-identical to before.

`--q4-xl` is `--q8 attn,shared,head,embed --draft-q4`, the recipe of the
published `Qwen3.8-Flash-Next-lily-q4-xl` checkpoint.

`--draft-q4` keeps the draft head's path 4-bit under such a policy: the
head's own attention and shared expert stay Q4 whatever `attn` and `shared`
say, and an 8-bit `lm_head` gets a Q4 g64 copy, `mtp.lm_head`, that only the
head's logits read (the trunk keeps its 8-bit head; the embedding stays
shared). The trunk verifies every draft, so the head only has to be roughly
right, and each draft step reads about 0.34 GB less on `--q8
attn,shared,head`. It needs one of `attn`, `shared` or `head` in the policy
and is recorded as `lily.quantization.draft_q4`.

`--mtp-only` adds the multi-token-prediction draft head (the `mtp.*` tensors:
one attention+MoE block plus its input projections and output mixer) to an
existing conversion as extra `mtp-*.safetensors` shards, merging them into the
index and config so the engine can run speculative decoding.

The vision tower (`model.visual.*`, 333 bf16 tensors, 0.90 GB) is kept
unquantized and copied byte for byte: it is 1.3 % of the checkpoint, runs once
per image, and quantization noise would eat into a tolerance floor that is
already only relative L2 0.05 (tools/reference/VISION.md). `--no-vision` drops
it; `--vision-only` appends it to an existing conversion as `vision-*.safetensors`
shards, the counterpart of `--mtp-only`. `config.json` carries `vision_config`
and the four vision token ids only when the tower is present, so the file
matches the weights in both directions.
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
VISION_PREFIX = "model.visual."
MTP_PREFIX = "mtp."
# Wrapper-level config keys that only mean something with the tower present.
VISION_TOKEN_KEYS = ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")
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

_Q4_SUFFIXES = (r"\.mlp\.experts\.down_proj$",)
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


EMBED_NAME = "model.language_model.embed_tokens.weight"
LM_HEAD_NAME = "lm_head.weight"
# The draft head's own Q4 copy of the LM head under `--draft-q4` (not in the
# source: `with_draft_head_copy` derives it from `lm_head.weight`).
MTP_LM_HEAD_NAME = "mtp.lm_head.weight"


# The projection groups `--q8` can move from Q4 to Q8, in the order the
# config records them, each with its tensor patterns. Kept apart from
# `_Q4_SUFFIXES` so the policy can move them without touching the rules of
# the default.
Q8_GROUPS: dict[str, tuple[str, ...]] = {
    "attn": (r"\.self_attn\.(q|k|v|o)_proj\.weight$",),
    "gdn": (r"\.linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)\.weight$",),
    "shared": (r"\.mlp\.shared_expert\.(gate|up|down)_proj\.weight$",),
    "head": ("^" + re.escape(LM_HEAD_NAME) + "$",),
    "embed": ("^" + re.escape(EMBED_NAME) + "$",),
}
# What the legacy flags stand for.
DENSE_GROUPS = frozenset(("attn", "gdn", "shared", "head"))
EMBED_GROUPS = frozenset(("embed",))
# The groups whose 8-bit width the draft head would read (its own block's
# attention and shared expert, the LM head it shares); `--draft-q4` needs one.
DRAFT_PATH_GROUPS = frozenset(("attn", "shared", "head"))


def q8_group_of(name: str) -> str | None:
    """The `Q8_GROUPS` group a source tensor belongs to, if any."""
    if name == MTP_LM_HEAD_NAME:
        return "head"
    for group, patterns in Q8_GROUPS.items():
        if any(re.search(p, name) for p in patterns):
            return group
    return None


@dataclass(frozen=True)
class Policy:
    """The storage choices the command line can change: which `Q8_GROUPS`
    are 8-bit, and whether the draft head's path stays 4-bit regardless
    (`draft_q4`). The default (none) is the policy every checkpoint before
    `--q8-dense` was written with."""

    q8: frozenset[str] = frozenset()
    draft_q4: bool = False

    def __post_init__(self) -> None:
        unknown = sorted(set(self.q8) - set(Q8_GROUPS))
        if unknown:
            raise ValueError(f"unknown q8 group(s) {', '.join(unknown)}; known: {', '.join(Q8_GROUPS)}")
        object.__setattr__(self, "q8", frozenset(self.q8))
        if self.draft_q4 and not self.q8 & DRAFT_PATH_GROUPS:
            raise ValueError("--draft-q4 needs an 8-bit group on the draft path: attn, shared or head")

    @classmethod
    def parse(cls, spec: str) -> "Policy":
        """`attn,head,embed` (the `--q8` argument); repeats are an error."""
        names = [n.strip() for n in spec.split(",") if n.strip()]
        if not names:
            raise ValueError("--q8 needs at least one group")
        if len(set(names)) != len(names):
            raise ValueError(f"--q8 {spec}: a group is listed twice")
        return cls(frozenset(names))

    @classmethod
    def from_config(cls, q: dict) -> "Policy":
        """The policy a `lily.quantization` block records (the inverse of
        `config_block`)."""
        draft_q4 = bool(q.get("draft_q4"))
        if "q8_groups" in q:
            if q.get("q8_dense") or q.get("q8_embed"):
                raise ValueError("lily.quantization carries both q8_groups and q8_dense/q8_embed")
            return cls(frozenset(q["q8_groups"]), draft_q4)
        groups = (DENSE_GROUPS if q.get("q8_dense") else frozenset()) | (EMBED_GROUPS if q.get("q8_embed") else frozenset())
        return cls(groups, draft_q4)

    def groups(self) -> list[str]:
        """The 8-bit groups in `Q8_GROUPS` order."""
        return [g for g in Q8_GROUPS if g in self.q8]

    def quant_of(self, group: str, draft: bool = False) -> Quant:
        """The width of `group`'s tensors, in the draft head (`mtp.*`) when
        `draft`: Q4 there under `draft_q4` (the embedding is never a draft
        tensor; the head reads the trunk's table)."""
        if draft and self.draft_q4:
            return Q4
        return Q8 if group in self.q8 else Q4

    def draft_head_copy(self) -> bool:
        """Whether the draft head gets its own Q4 `mtp.lm_head`: under
        `draft_q4` with an 8-bit trunk head."""
        return self.draft_q4 and "head" in self.q8

    def q8_suffixes(self) -> list[str]:
        """The 8-bit tensor patterns as recorded in `config.json`: the fixed
        list, then each 8-bit group's patterns in `Q8_GROUPS` order, so the
        default list is unchanged and the legacy flags' lists are what they
        were."""
        out = list(_Q8_SUFFIXES)
        for g in self.groups():
            out += Q8_GROUPS[g]
        return out

    def config_block(self) -> dict:
        """The keys `lily.quantization` gains: the legacy `q8_dense` /
        `q8_embed` flags when they spell the set exactly (so those configs are
        byte-identical to the ones written before `--q8`), otherwise
        `q8_groups`; empty for the default. `draft_q4` follows them when
        set. The `q8_suffixes` stay the trunk's: under `draft_q4` they do not
        apply to `mtp.*` tensors."""
        dense, embed = self.q8 & DENSE_GROUPS, self.q8 & EMBED_GROUPS
        draft = {"draft_q4": True} if self.draft_q4 else {}
        if dense in (frozenset(), DENSE_GROUPS):
            return {
                **({"q8_dense": True} if dense else {}),
                **({"q8_embed": True} if embed else {}),
                **draft,
            }
        return {"q8_groups": self.groups(), **draft}

    def describe(self) -> str:
        return (", q8 " + ",".join(self.groups()) if self.q8 else "") + (", draft q4" if self.draft_q4 else "")


def with_draft_head_copy(tensors: list[SourceTensor], policy: Policy) -> list[SourceTensor]:
    """The source list plus, when the policy gives the draft head its own LM
    head, `mtp.lm_head.weight`: a second view of `lm_head.weight`'s bytes,
    placed after the last `mtp.*` tensor so it lands with the head (in the
    `mtp-*` shards of an `--mtp-only` run). Unchanged otherwise, and when the
    source has no draft head."""
    if not policy.draft_head_copy():
        return tensors
    mtp_at = [i for i, t in enumerate(tensors) if t.name.startswith(MTP_PREFIX)]
    if not mtp_at:
        return tensors
    if any(t.name == MTP_LM_HEAD_NAME for t in tensors):
        raise SystemExit(f"source already has {MTP_LM_HEAD_NAME}; refusing to derive the draft head's copy")
    head = next((t for t in tensors if t.name == LM_HEAD_NAME), None)
    if head is None:
        raise SystemExit(f"source has no {LM_HEAD_NAME} to copy for the draft head")
    copy = SourceTensor(MTP_LM_HEAD_NAME, head.dtype, head.shape, head.shard, head.start, head.end)
    at = mtp_at[-1] + 1
    return tensors[:at] + [copy] + tensors[at:]


def plan_tensor(
    t: SourceTensor,
    keep_layers: int,
    ngram: Quant,
    mtp: bool = True,
    vision: bool = True,
    policy: Policy = Policy(),
) -> Plan:
    name = t.name
    if name.startswith(VISION_PREFIX):
        # The tower stays bf16, byte for byte (see the module docstring).
        return Plan("copy") if vision else Plan("drop")
    if not mtp and name.startswith(MTP_PREFIX):
        return Plan("drop")
    li = layer_index(name)
    if li is not None and li >= keep_layers:
        return Plan("drop")
    group = q8_group_of(name)
    if group is not None:
        return Plan("quant", policy.quant_of(group, draft=name.startswith(MTP_PREFIX)))
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
    if name.startswith(VISION_PREFIX):
        return "vision"
    if name.startswith(MTP_PREFIX):
        return "mtp"
    if "ngram_embedding.shard_" in name:
        return "ngram"
    if ".mlp.experts." in name:
        return "experts"
    if name in (EMBED_NAME, LM_HEAD_NAME):
        return "embed/lm_head"
    return "dense"


def gate_up_output_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    e, two_i, h = shape
    return (e, two_i // 2, h)


def estimate(
    tensors: list[SourceTensor],
    keep_layers: int,
    ngram: Quant,
    mtp: bool = True,
    vision: bool = True,
    policy: Policy = Policy(),
) -> Totals:
    totals = Totals()
    for t in tensors:
        plan = plan_tensor(t, keep_layers, ngram, mtp, vision, policy)
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


def vision_block(tensors: list[SourceTensor]) -> dict:
    """What the engine needs to know about the copied tower: its storage dtype
    and tensor count, so the loader can tell presence without scanning the index."""
    tower = [t for t in tensors if t.name.startswith(VISION_PREFIX)]
    dtypes = sorted({t.dtype for t in tower})
    if dtypes != ["BF16"]:
        raise SystemExit(f"vision tower is stored as {dtypes}, expected BF16 only")
    return {"dtype": "bf16", "tensors": len(tower), "bytes": sum(t.nbytes for t in tower)}


def strip_vision_from_config(cfg: dict) -> None:
    """Removes the wrapper keys that describe the tower, so a config without
    the tensors does not advertise them."""
    cfg.pop("vision_config", None)
    for key in VISION_TOKEN_KEYS:
        cfg.pop(key, None)


def write_config(
    src: Path,
    dst: Path,
    keep_layers: int,
    ngram: Quant,
    ple: dict,
    revision: str | None,
    mtp: bool,
    vision: list[SourceTensor] | None,
    policy: Policy,
) -> None:
    """`vision` is the list of copied tower tensors, or None when the tower was dropped."""
    cfg = json.loads((src / "config.json").read_text())
    text = cfg["text_config"]
    text["num_hidden_layers"] = keep_layers
    text["layer_types"] = text["layer_types"][:keep_layers]
    if vision is None:
        strip_vision_from_config(cfg)
    cfg["lily"] = {
        "format": FORMAT_NAME,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": revision,
        "layers": keep_layers,
        "quantization": {
            "default": {"bits": Q4.bits, "group_size": Q4.group_size, "mode": "affine"},
            "ngram_embedding": {"bits": ngram.bits, "group_size": ngram.group_size, "mode": "affine"},
            "q8_suffixes": policy.q8_suffixes(),
            **policy.config_block(),
        },
        "ple": ple,
        "mtp": mtp_block(text) if mtp else None,
        "vision": vision_block(vision) if vision is not None else None,
        "dropped": ([] if vision is not None else [VISION_PREFIX]) + ([] if mtp else [MTP_PREFIX]),
    }
    # Not read by lily (it uses `lily.quantization`): the Hugging Face Hub only
    # unpacks u32 codes into a parameter count when it finds the MLX-style
    # top-level block, and otherwise reports ~30B for this 180B model. The
    # few 8-bit tensors are counted as 4-bit, about 0.4% high (more under
    # `--q8`, whose 8-bit groups count at half their size).
    cfg["quantization"] = {"group_size": Q4.group_size, "bits": Q4.bits, "mode": "affine"}
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


def add_vision_to_config(src: Path, dst: Path, vision: list[SourceTensor]) -> None:
    """Marks an existing conversion's config as carrying the tower: restores
    `vision_config` and the vision token ids from the source config (a
    `--no-vision` conversion stripped them) and adds the `lily.vision` block."""
    path = dst / "config.json"
    cfg = json.loads(path.read_text())
    lily = cfg["lily"]
    if lily.get("vision"):
        raise SystemExit(f"{path} already declares a vision block")
    source = json.loads((src / "config.json").read_text())
    cfg["vision_config"] = source["vision_config"]
    for key in VISION_TOKEN_KEYS:
        cfg[key] = source[key]
    lily["vision"] = vision_block(vision)
    lily["dropped"] = [p for p in lily.get("dropped", []) if p != VISION_PREFIX]
    path.write_text(json.dumps(cfg, indent=2))


def policy_of_conversion(dst: Path) -> Policy:
    """The policy an existing conversion was written with, so an appended
    part (`--mtp-only`) stores its projections the way the trunk does."""
    q = json.loads((dst / "config.json").read_text())["lily"]["quantization"]
    return Policy.from_config(q)


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


# `--q4-xl`: the published 4-bit checkpoint with its most sensitive dense
# tensors at 8 bits (Qwen3.8-Flash-Next-lily-q4-xl).
Q4_XL_GROUPS = frozenset({"attn", "shared", "head", "embed"})


def q8_groups_of_args(args: argparse.Namespace) -> frozenset[str]:
    """`--q8 GROUPS` together with its aliases `--q8-dense`, `--q8-embed` and
    `--q4-xl`."""
    groups: frozenset[str] = Policy.parse(args.q8).q8 if args.q8 else frozenset()
    if args.q4_xl:
        groups |= Q4_XL_GROUPS
    if args.q8_dense:
        groups |= DENSE_GROUPS
    if args.q8_embed:
        groups |= EMBED_GROUPS
    return groups


def policy_of_args(args: argparse.Namespace) -> Policy:
    """The q8 groups plus `--draft-q4` (which `--q4-xl` implies)."""
    return Policy(q8_groups_of_args(args), args.draft_q4 or args.q4_xl)


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
    vision = not args.no_vision
    append_only = args.mtp_only or args.vision_only
    if args.mtp_only and args.vision_only:
        raise SystemExit("--mtp-only and --vision-only are separate append runs; pass one at a time")
    try:
        groups = q8_groups_of_args(args)
        # An appended part takes the conversion's policy (below), so only a
        # full run's flags must form a valid policy on their own.
        policy = Policy(groups) if append_only else policy_of_args(args)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    if append_only:
        if not (dst / "model.safetensors.index.json").exists():
            raise SystemExit(f"{dst} is not a finished conversion (no index); run a full conversion first")
        # The draft head shares the trunk's policy (its block and the shared
        # LM head must agree); the flags may restate it but not change it.
        existing = policy_of_conversion(dst)
        if not groups <= existing.q8 or ((args.draft_q4 or args.q4_xl) and not existing.draft_q4):
            raise SystemExit(
                f"{dst} was converted with q8 groups {existing.groups() or 'none'}"
                f"{' and --draft-q4' if existing.draft_q4 else ''}; an appended part cannot change the policy"
            )
        policy = existing
    # Before the append filter: the draft head's LM head copy is an `mtp.*`
    # tensor made from the trunk's `lm_head.weight`.
    tensors = with_draft_head_copy(tensors, policy)
    if append_only:
        # One optional part, appended to a finished conversion.
        prefix, stem = (MTP_PREFIX, "mtp") if args.mtp_only else (VISION_PREFIX, "vision")
        tensors = [t for t in tensors if t.name.startswith(prefix)]
        if not tensors:
            raise SystemExit(f"source has no {prefix}* tensors")
        if any(dst.glob(f"{stem}-*.safetensors")):
            raise SystemExit(f"{dst} already holds {stem} shards; refusing to overwrite")
        mtp = vision = True
    else:
        stem = "model"
    totals = estimate(tensors, keep_layers, ngram, mtp, vision, policy)
    if args.mtp_only:
        what = "the MTP draft head"
    elif args.vision_only:
        what = "the vision tower"
    else:
        dropped = (", no mtp" if not mtp else "") + (", no vision" if not vision else "")
        what = f"{keep_layers} layers (ngram {ngram.bits}-bit g{ngram.group_size}{policy.describe()}{dropped})"
    print_totals(f"planned output for {what}", totals)
    if args.dry_run:
        return

    dst.mkdir(parents=True, exist_ok=True)
    if not append_only:
        if any(dst.glob("model-*.safetensors")):
            raise SystemExit(f"{dst} already holds shards; refusing to overwrite")
        for name in COPIED_FILES:
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)

    planned = [(t, plan_tensor(t, keep_layers, ngram, mtp, vision, policy)) for t in tensors]
    planned = [(t, p) for t, p in planned if p.kind != "drop"]
    src_bytes = sum(t.nbytes for t, _ in planned)
    quant_positions = [i for i, (_, p) in enumerate(planned) if p.kind == "quant"]
    check_idx = set(np.random.default_rng(0).choice(quant_positions, size=min(3, len(quant_positions)), replace=False).tolist()) if quant_positions else set()

    writer = ShardWriter(dst, int(args.shard_bytes), stem=stem)
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
    writer.finish(merge=append_only)

    vision_tensors = [t for t, p in planned if p.kind == "copy" and t.name.startswith(VISION_PREFIX)]
    if args.mtp_only:
        add_mtp_to_config(dst)
    elif args.vision_only:
        add_vision_to_config(src, dst, vision_tensors)
    else:
        ple = verify_ple_constants(text_cfg, ple_consts) if ple_consts else {}
        write_config(
            src, dst, keep_layers, ngram, ple, source_revision(src), mtp, vision_tensors if vision else None, policy
        )

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
    parser.add_argument("--no-vision", action="store_true", help="drop the model.visual.* vision tower (it is copied in bf16 by default)")
    parser.add_argument("--vision-only", action="store_true", help="append only the model.visual.* vision tower to an existing conversion in --dst")
    parser.add_argument(
        "--q8",
        metavar="GROUPS",
        default=None,
        help="comma-separated groups to store at Q8 group 64 instead of Q4: "
        "attn (attention q/k/v/o), gdn (GDN projections), shared (shared expert), head (lm_head), embed (embed_tokens)",
    )
    parser.add_argument("--q8-dense", action="store_true", help="alias of --q8 attn,gdn,shared,head")
    parser.add_argument("--q8-embed", action="store_true", help="alias of --q8 embed")
    parser.add_argument(
        "--draft-q4",
        action="store_true",
        help="keep the draft head's path Q4 under an 8-bit attn, shared or head group: its own attention and "
        "shared expert stay Q4, and an 8-bit lm_head gets a Q4 copy (mtp.lm_head) for the head's logits",
    )
    parser.add_argument(
        "--q4-xl",
        action="store_true",
        help="the q4-xl checkpoint: alias of --q8 attn,shared,head,embed --draft-q4",
    )
    parser.add_argument("--dry-run", action="store_true", help="only print the size plan")
    convert(parser.parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
