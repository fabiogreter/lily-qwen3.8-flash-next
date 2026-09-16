#!/usr/bin/env python3
"""Reference forward pass of Qwen3.8-Flash-Next on the CPU with Hugging Face
transformers, truncated to the first N decoder layers, producing a JSON golden
lily can be compared against.

Two weight sources:

* ``--src <raw HF dir>``: the original BF16 checkpoint (the reference the model
  was trained as; lily's quantized run differs by quantization error).
* ``--lily <converted dir>``: lily's own quantized checkpoint, dequantized back
  to bf16, so both engines run the very same weights and differences isolate
  kernel numerics.

The 102 GB n-gram table is never loaded: the PLE embedding is replaced by a
module that gathers rows on demand from the memory-mapped shards.

    .venv/bin/python tools/reference/hf_reference.py --lily ~/models/...-l4 \
        --prompt "The capital of Switzerland is" --out tools/reference/goldens/l4.json

The truncated model has no meaningful language ability; the golden is a
numerical fixture, not a quality check. The QSA indexer in transformers is a
per-query Python loop, so long prompts (> 2 100 tokens, needed to exercise
sparsity) take minutes.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

LAYER_PREFIX = "model.language_model.layers."


# --- safetensors access -----------------------------------------------------------


@dataclass(frozen=True)
class TensorRef:
    dtype: str
    shape: tuple[int, ...]
    path: Path
    start: int
    end: int


def read_headers(directory: Path) -> dict[str, TensorRef]:
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    refs: dict[str, TensorRef] = {}
    for shard_name in sorted(set(index["weight_map"].values())):
        path = directory / shard_name
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        base = 8 + n
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            s, e = entry["data_offsets"]
            refs[name] = TensorRef(entry["dtype"], tuple(entry["shape"]), path, base + s, base + e)
    return refs


_TORCH_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64, "U32": torch.uint32, "F16": torch.float16}


class Store:
    """Lazy tensor reader over one checkpoint directory (mmap per shard)."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.refs = read_headers(directory)
        self._maps: dict[Path, np.memmap] = {}

    def has(self, name: str) -> bool:
        return name in self.refs

    def _map(self, path: Path) -> np.memmap:
        if path not in self._maps:
            self._maps[path] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._maps[path]

    def raw(self, name: str) -> np.ndarray:
        ref = self.refs[name]
        return self._map(ref.path)[ref.start : ref.end]

    def tensor(self, name: str) -> torch.Tensor:
        ref = self.refs[name]
        buf = np.array(self.raw(name))  # copy out of the mmap
        return torch.frombuffer(buf, dtype=_TORCH_DTYPES[ref.dtype]).reshape(ref.shape)

    def rows(self, name: str, row_ids: np.ndarray) -> torch.Tensor:
        """Selected rows of a 2-D tensor without reading the whole tensor."""
        ref = self.refs[name]
        rows, cols = ref.shape
        itemsize = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I64": 8}[ref.dtype]
        view = self._map(ref.path)[ref.start : ref.end].reshape(rows, cols * itemsize)
        picked = np.ascontiguousarray(view[row_ids])
        return torch.frombuffer(picked, dtype=_TORCH_DTYPES[ref.dtype]).reshape(len(row_ids), cols)


# --- lily checkpoint dequantization -------------------------------------------------


def dequant_affine(codes: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """MLX affine: element c of a row is code (c % per_word) of word c // per_word."""
    per_word = 32 // bits
    mask = (1 << bits) - 1
    *lead, words = codes.shape
    k = words * per_word
    c = codes.to(torch.int64).unsqueeze(-1)
    shifts = torch.arange(per_word, dtype=torch.int64) * bits
    q = ((c >> shifts) & mask).reshape(*lead, k).to(torch.float32)
    groups = k // group
    s = scales.to(torch.float32).reshape(*lead, groups, 1)
    b = biases.to(torch.float32).reshape(*lead, groups, 1)
    w = (q.reshape(*lead, groups, group) * s + b).reshape(*lead, k)
    return w.to(torch.bfloat16)


def infer_bits(codes_cols: int, in_features: int) -> int:
    bits = 32 * codes_cols // in_features
    if bits not in (4, 8):
        raise ValueError(f"cannot infer bit width from {codes_cols} words for {in_features} inputs")
    return bits


class LilyWeights:
    """Reads lily's quantized checkpoint as bf16 tensors under HF names."""

    def __init__(self, store: Store, config: dict):
        self.store = store
        self.group = config["lily"]["quantization"]["default"]["group_size"]
        self.ngram = config["lily"]["quantization"]["ngram_embedding"]

    def is_quantized(self, base: str) -> bool:
        return self.store.has(f"{base}.scales")

    def linear(self, base: str, in_features: int, group: int | None = None) -> torch.Tensor:
        codes = self.store.tensor(f"{base}.weight")
        bits = infer_bits(codes.shape[-1], in_features)
        return dequant_affine(
            codes,
            self.store.tensor(f"{base}.scales"),
            self.store.tensor(f"{base}.biases"),
            bits,
            group or self.group,
        )

    def ngram_rows(self, shard_base: str, row_ids: np.ndarray) -> torch.Tensor:
        codes = self.store.rows(f"{shard_base}.weight", row_ids)
        scales = self.store.rows(f"{shard_base}.scales", row_ids)
        biases = self.store.rows(f"{shard_base}.biases", row_ids)
        return dequant_affine(codes, scales, biases, self.ngram["bits"], self.ngram["group_size"])


# --- lazy n-gram embedding -------------------------------------------------------


class LazyNGramEmbedding(torch.nn.Module):
    """Drop-in for the 320M x 160 nn.Embedding: gathers rows from the shards."""

    def __init__(self, layer_prefix: str, store: Store, lily: LilyWeights | None, shards: int, head_dim: int):
        super().__init__()
        self.base = f"{layer_prefix}ple.ple_embedding.ngram_embedding"
        self.store = store
        self.lily = lily
        self.shards = shards
        self.rows_per_shard = store.refs[f"{self.base}.shard_0.weight"].shape[0]
        self.embedding_dim = head_dim
        # Mirror the attribute the HF forward touches for device placement.
        self.weight = torch.nn.Parameter(torch.zeros(1, head_dim, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).numpy().astype(np.int64)
        out = torch.empty(flat.shape[0], self.embedding_dim, dtype=torch.bfloat16)
        shard_of = flat // self.rows_per_shard
        local = flat % self.rows_per_shard
        for shard in np.unique(shard_of):
            sel = np.nonzero(shard_of == shard)[0]
            rows = local[sel]
            order = np.argsort(rows)
            base = f"{self.base}.shard_{int(shard)}"
            if self.lily is None:
                picked = self.store.rows(f"{base}.weight", rows[order])
            else:
                picked = self.lily.ngram_rows(base, rows[order])
            out[torch.from_numpy(sel[order])] = picked
        return out.reshape(*ids.shape, self.embedding_dim)


# --- model construction ------------------------------------------------------------


def build_model(directory: Path, layers: int, from_lily: bool, dtype: torch.dtype):
    from transformers import Qwen4ExpConfig, Qwen4ExpForCausalLM

    raw_cfg = json.loads((directory / "config.json").read_text())
    text_cfg = dict(raw_cfg["text_config"])
    total = text_cfg["num_hidden_layers"]
    if not 1 <= layers <= total:
        raise SystemExit(f"--layers must be in 1..{total}")
    text_cfg["num_hidden_layers"] = layers
    text_cfg["layer_types"] = text_cfg["layer_types"][:layers]
    if from_lily and raw_cfg.get("lily", {}).get("format") != "qwen4_exp-affine-v1":
        raise SystemExit("--lily expects a lily qwen4_exp-affine-v1 checkpoint")
    config = Qwen4ExpConfig(**raw_cfg).text_config
    config = type(config)(**text_cfg)
    config._attn_implementation = "eager"

    store = Store(directory)
    lily = LilyWeights(store, raw_cfg) if from_lily else None

    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(config)
    ple_layers = [i for i in range(layers) if i + 1 in (text_cfg.get("ple_layer_ids") or [])]
    for i in ple_layers:
        prefix = f"{LAYER_PREFIX}{i}."
        ple = model.model.layers[i].ple.ple_embedding
        ple.ngram_embedding = LazyNGramEmbedding(prefix, store, lily, text_cfg.get("split_ngram_parts", 512), ple.ngram_embedding.embedding_dim)

    state = build_state_dict(model, store, lily, text_cfg, raw_cfg.get("lily", {}).get("ple"))
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    # Non-persistent buffers and the lazy embedding's placeholder are expected.
    allowed_missing = {"model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"}
    allowed_missing |= {f"model.layers.{i}.ple.ple_embedding.ngram_embedding.weight" for i in ple_layers}
    bad = [m for m in missing if m not in allowed_missing]
    if bad or unexpected:
        raise SystemExit(f"state dict mismatch: missing={bad[:8]} unexpected={list(unexpected)[:8]}")
    rotary = model.model.rotary_emb
    inv_freq, _ = rotary.compute_default_rope_parameters(config)
    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = inv_freq.clone()
    model = model.to(dtype)
    for i in ple_layers:
        model.model.layers[i].ple.ple_embedding.ngram_embedding.weight.data = model.model.layers[i].ple.ple_embedding.ngram_embedding.weight.data.to(torch.bfloat16)
    model.eval()
    return model, config


_PLE_CONSTS = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")


def build_state_dict(
    model,
    store: Store,
    lily: LilyWeights | None,
    text_cfg: dict,
    ple_consts: dict | None,
    rename: Callable[[str], str] | None = None,
) -> dict[str, torch.Tensor]:
    """HF parameter names -> tensors, from either weight source. lily's
    checkpoint carries the PLE hash constants in config.json, not as tensors.

    `rename` maps a parameter name of `model` to its checkpoint name; the
    default is the `Qwen4ExpForCausalLM` layout (`model.` -> `model.language_model.`).
    `hf_vision_reference.py` passes its own for the submodules of
    `Qwen4ExpForConditionalGeneration`."""
    h = text_cfg["hidden_size"]
    inter = text_cfg["moe_intermediate_size"]
    state: dict[str, torch.Tensor] = {}
    wanted = dict(model.named_parameters())
    wanted.update({k: v for k, v in model.named_buffers() if k.endswith(_PLE_CONSTS)})
    for name, param in wanted.items():
        if "ngram_embedding.weight" in name:
            continue
        if rename is not None:
            src = rename(name)
        else:
            src = "model.language_model." + name[len("model."):] if name.startswith("model.") else name
        if name.endswith(_PLE_CONSTS) and not store.has(src):
            key = name.rsplit(".", 1)[1]
            if not ple_consts or key not in ple_consts:
                raise SystemExit(f"{src} missing from checkpoint and config.json lily.ple")
            state[name] = torch.tensor(ple_consts[key], dtype=torch.int64)
            continue
        if src.endswith(".mlp.experts.gate_up_proj"):
            base = src[: -len("gate_up_proj")]
            if lily is None:
                state[name] = store.tensor(src)
            else:
                gate = lily.linear(f"{base}gate_proj", h)
                up = lily.linear(f"{base}up_proj", h)
                state[name] = torch.cat([gate, up], dim=1)
            continue
        if src.endswith(".mlp.experts.down_proj"):
            state[name] = store.tensor(src) if lily is None else lily.linear(src, inter)
            continue
        if lily is not None and src.endswith(".weight") and lily.is_quantized(src[: -len(".weight")]):
            in_features = param.shape[-1]
            state[name] = lily.linear(src[: -len(".weight")], in_features)
            continue
        state[name] = store.tensor(src)
    return state


# --- golden production ----------------------------------------------------------------


def render_prompt(directory: Path, prompt: str, thinking: bool) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(directory)
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )
    return tok(text, add_special_tokens=False)["input_ids"]


def capture_selections(model, last: int, seq_len: int, ratio: int) -> tuple[list, dict]:
    """Hooks every QSA indexer to record, for the last `last` positions, the
    selected block ids (tokens of complete blocks, expressed as block index)."""
    selections: dict = {}
    hooks = []
    positions = list(range(max(0, seq_len - last), seq_len))

    def make_hook(layer_idx):
        def hook(module, args, output):
            mask = output[0, 0]  # [seq, kv]; selected where True / 0.0
            selected = mask if mask.dtype == torch.bool else mask == 0
            per_pos = {}
            for pos in positions:
                toks = torch.nonzero(selected[pos]).flatten()
                nb = (pos + 1) // ratio
                blocks = sorted(set((toks[toks < nb * ratio] // ratio).tolist()))
                per_pos[pos] = blocks
            selections[layer_idx] = per_pos
        return hook

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            hooks.append(layer.self_attn.indexer.register_forward_hook(make_hook(i)))
    return hooks, selections


@torch.no_grad()
def run(model, tokens: list[int], last: int, greedy_steps: int) -> dict:
    ids = torch.tensor([tokens], dtype=torch.long)
    ratio = model.config.indexer_compress_ratio
    hooks, selections = capture_selections(model, last, len(tokens), ratio)
    started = time.time()
    out = model(input_ids=ids, use_cache=greedy_steps > 0)
    for h in hooks:
        h.remove()
    logits = out.logits[0].float()
    elapsed = time.time() - started
    argmax = logits.argmax(dim=-1).tolist()
    top = []
    for pos in range(max(0, len(tokens) - last), len(tokens)):
        values, idx = logits[pos].topk(8)
        top.append({"position": pos, "ids": idx.tolist(), "logits": [round(v, 4) for v in values.tolist()]})
    result = {
        "prompt_token_ids": tokens,
        "argmax": argmax,
        "top8_last": top,
        "prefill_seconds": round(elapsed, 2),
        "selected_blocks": {str(layer): {str(p): b for p, b in per_pos.items()} for layer, per_pos in selections.items()},
    }
    if greedy_steps > 0:
        past = out.past_key_values
        chosen = [argmax[-1]]
        for _ in range(greedy_steps - 1):
            step = model(input_ids=torch.tensor([[chosen[-1]]]), past_key_values=past, use_cache=True)
            past = step.past_key_values
            chosen.append(int(step.logits[0, -1].argmax()))
        result["greedy"] = chosen
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--src", help="raw HF BF16 checkpoint directory")
    source.add_argument("--lily", help="lily quantized checkpoint directory (dequantized for the run)")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--prompt", help="user message rendered through the chat template")
    parser.add_argument("--tokens", help="JSON file with a list of token ids (bypasses the template)")
    parser.add_argument("--thinking", action="store_true", help="render with enable_thinking=True")
    parser.add_argument("--last", type=int, default=16, help="positions to report top-8 logits for")
    parser.add_argument("--greedy", type=int, default=0, help="greedy continuation steps to record")
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    directory = Path(args.src or args.lily).expanduser()
    if args.tokens:
        tokens = json.loads(Path(args.tokens).read_text())
    elif args.prompt is not None:
        tokens = render_prompt(directory, args.prompt, args.thinking)
    else:
        raise SystemExit("pass --prompt or --tokens")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    started = time.time()
    model, config = build_model(directory, args.layers, from_lily=args.lily is not None, dtype=dtype)
    print(f"model built in {time.time() - started:.1f}s: {args.layers} layers, {len(tokens)} prompt tokens", file=sys.stderr)
    result = run(model, tokens, args.last, args.greedy)
    result["meta"] = {
        "source": "lily-dequant" if args.lily else "hf-bf16",
        "directory": str(directory),
        "layers": args.layers,
        "dtype": args.dtype,
        "budget": config.indexer_budget,
        "dense_limit": config.indexer_budget + config.indexer_compress_ratio - 1,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}: argmax tail {result['argmax'][-8:]}", file=sys.stderr)


if __name__ == "__main__":
    main()
