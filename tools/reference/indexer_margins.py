#!/usr/bin/env python3
"""Score every visible block for the last positions of a sequence with the
reference indexer math, then report how far the blocks lily selected (from a
probe record) sit from the reference's top-k boundary.

    .venv/bin/python tools/reference/indexer_margins.py --lily <ckpt> --layers 4 \
        --tokens tf_tokens.json --probe lily_probe.json

A disputed block whose score is within bf16 noise (~0.5%) of the k-th score is
a numerics flip, not a bug.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from hf_reference import build_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lily", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--probe", required=True, help="lily-probe JSON with selected_blocks debug")
    args = parser.parse_args()

    tokens = json.loads(Path(args.tokens).read_text())
    probe = json.loads(Path(args.probe).read_text())
    lily_sel = {s["position"]: set((s.get("debug") or {}).get("selected_blocks") or []) for s in probe["steps"]}
    lily_scores = {s["position"]: (s.get("debug") or {}).get("block_scores") for s in probe["steps"]}
    positions = [p for p in lily_sel if lily_sel[p] and p < len(tokens)]

    model, config = build_model(Path(args.lily).expanduser(), args.layers, from_lily=True, dtype=torch.bfloat16)
    ratio, budget = config.indexer_compress_ratio, config.indexer_budget
    k_max = budget // ratio

    captured: dict = {}

    def hook(module, args_, kwargs):
        captured["hidden"] = args_[0] if args_ else kwargs["hidden_states"]
        captured["pos_emb"] = args_[1] if len(args_) > 1 else kwargs["position_embeddings"]

    indexers = [(i, l.self_attn.indexer) for i, l in enumerate(model.model.layers) if hasattr(l, "self_attn")]
    layer_idx, indexer = indexers[-1]
    handle = indexer.register_forward_pre_hook(hook, with_kwargs=True)
    with torch.no_grad():
        model(input_ids=torch.tensor([tokens]), use_cache=False)
    handle.remove()

    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    hidden = captured["hidden"][0]
    cos, sin = captured["pos_emb"]
    cos, sin = cos[0], sin[0]
    with torch.no_grad():
        qk = indexer.index_qk_proj(hidden)
        nh, d = indexer.index_n_heads, indexer.index_head_dim
        q, raw_keys = qk[:, : nh * d].reshape(-1, nh, d), qk[:, nh * d :]
        q = indexer.q_layernorm(q)
        q = apply_rotary_pos_emb(q.unsqueeze(0), cos=cos.unsqueeze(0), sin=sin.unsqueeze(0), unsqueeze_dim=2)[0]
        print(f"layer {layer_idx}: {'pos':>5} {'nb':>4} {'k-th':>9} {'(k+1)-th':>9} {'rel gap':>8}  disputed lily-only blocks: score, rel. distance below k-th")
        for pos in sorted(positions):
            nb = (pos + 1) // ratio
            blocks = raw_keys[: nb * ratio].reshape(nb, ratio, d).float().mean(dim=1).to(raw_keys.dtype)
            blocks = indexer.k_layernorm(blocks)
            starts = torch.arange(nb) * ratio
            blocks = apply_rotary_pos_emb(blocks.unsqueeze(1), cos=cos[starts], sin=sin[starts]).squeeze(1)
            scores = torch.relu(q[pos].float() @ blocks.float().T).sum(dim=0) / math.sqrt(d)
            order = torch.argsort(scores, descending=True)
            kth, k1 = scores[order[k_max - 1]].item(), scores[order[k_max]].item()
            hf_set = set(order[:k_max].tolist())
            disputed = sorted(lily_sel[pos] - hf_set)
            details = ", ".join(f"{b}: {scores[b].item():.4f} ({(kth - scores[b].item()) / kth * 100:.2f}%)" for b in disputed)
            print(f"{'':>9}{pos:>5} {nb:>4} {kth:>9.4f} {k1:>9.4f} {(kth - k1) / kth * 100:>7.3f}%  {details}")
            mine = lily_scores.get(pos)
            if mine and len(mine) >= nb:
                mine_t = torch.tensor(mine[:nb])
                rel = ((mine_t - scores).abs() / scores.abs().clamp_min(1e-6))
                spread = (scores.max() - scores.min()) / scores.max() * 100
                print(f"{'':>15}lily vs hf block scores: max rel diff {rel.max().item() * 100:.3f}%, "
                      f"mean {rel.mean().item() * 100:.3f}%; hf score spread (max-min)/max = {spread.item():.2f}%")


if __name__ == "__main__":
    main()
