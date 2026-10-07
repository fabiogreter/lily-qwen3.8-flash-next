"""Unit tests for the converter's storage policy (`--q8`, `--q8-dense`,
`--q8-embed`); no checkpoint and no mlx needed.

    .venv/bin/python tools/convert/test_convert_qwen38_flash_next.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import convert_qwen38_flash_next as conv  # noqa: E402

L = "model.language_model.layers.3."
# One source tensor per rule the policy touches, with its group (None: fixed).
TENSORS = {
    f"{L}self_attn.q_proj.weight": "attn",
    f"{L}self_attn.o_proj.weight": "attn",
    "mtp.layers.0.self_attn.k_proj.weight": "attn",
    f"{L}linear_attn.in_proj_qkv.weight": "gdn",
    f"{L}linear_attn.in_proj_b.weight": "gdn",
    f"{L}linear_attn.out_proj.weight": "gdn",
    f"{L}mlp.shared_expert.gate_proj.weight": "shared",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": "shared",
    "lm_head.weight": "head",
    "model.language_model.embed_tokens.weight": "embed",
    f"{L}mlp.shared_expert_gate.weight": None,
    f"{L}mlp.gate.weight": None,
    f"{L}mlp.experts.down_proj": None,
    f"{L}mlp.experts.gate_up_proj": None,
}
FIXED_BITS = {
    f"{L}mlp.shared_expert_gate.weight": 8,
    f"{L}mlp.gate.weight": 8,
    f"{L}mlp.experts.down_proj": 4,
    f"{L}mlp.experts.gate_up_proj": 4,
}

# The q8_suffixes a `--q8-dense --q8-embed` conversion recorded before `--q8`
# existed (models/Qwen3.8-Flash-Next-lily-q8dense/config.json).
LEGACY_DENSE_EMBED_SUFFIXES = [
    r"^mtp\.fc_(embedding|hidden)\.weight$",
    r"\.mlp\.gate\.weight$",
    r"\.mlp\.shared_expert_gate\.weight$",
    r"hyper_connection(_mixer)?\.input_mix_weight_(down|up)\.weight$",
    r"\.block_inject_weight\.weight$",
    r"\.ple\.(key_proj|value_proj)\.weight$",
    r"\.self_attn\.indexer\.index_qk_proj\.weight$",
    r"\.self_attn\.(q|k|v|o)_proj\.weight$",
    r"\.linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)\.weight$",
    r"\.mlp\.shared_expert\.(gate|up|down)_proj\.weight$",
    r"^lm_head\.weight$",
    r"^model\.language_model\.embed_tokens\.weight$",
]


def bits(name: str, policy: conv.Policy) -> int:
    t = conv.SourceTensor(name, "BF16", (64, 64), Path("x"), 0, 0)
    plan = conv.plan_tensor(t, 48, conv.Quant(4, 32), policy=policy)
    assert plan.quant is not None, name
    return plan.quant.bits


def args(*argv: str):
    p = conv.argparse.ArgumentParser()
    p.add_argument("--q8", default=None)
    p.add_argument("--q8-dense", action="store_true")
    p.add_argument("--q8-embed", action="store_true")
    return p.parse_args(list(argv))


class PolicyTest(unittest.TestCase):
    def test_every_subset_moves_exactly_its_groups(self):
        names = list(conv.Q8_GROUPS)
        for mask in range(1 << len(names)):
            groups = frozenset(g for i, g in enumerate(names) if mask >> i & 1)
            policy = conv.Policy(groups)
            for name, group in TENSORS.items():
                want = FIXED_BITS[name] if group is None else (8 if group in groups else 4)
                self.assertEqual(bits(name, policy), want, f"{name} under {sorted(groups)}")

    def test_the_aliases_are_their_group_sets(self):
        self.assertEqual(conv.policy_of_args(args()).q8, frozenset())
        self.assertEqual(conv.policy_of_args(args("--q8-dense")).q8, {"attn", "gdn", "shared", "head"})
        self.assertEqual(conv.policy_of_args(args("--q8-embed")).q8, {"embed"})
        both = conv.policy_of_args(args("--q8-dense", "--q8-embed"))
        self.assertEqual(both, conv.policy_of_args(args("--q8", "embed,head,shared,gdn,attn")))
        self.assertEqual(conv.policy_of_args(args("--q8", "attn,head", "--q8-embed")).q8, {"attn", "head", "embed"})

    def test_parse_rejects_unknown_repeated_and_empty_lists(self):
        for spec in ("attn,experts", "head,head", ",", ""):
            with self.assertRaises(ValueError, msg=spec):
                conv.Policy.parse(spec)

    def test_the_legacy_sets_keep_their_config_bytes(self):
        dense_embed = conv.Policy(conv.DENSE_GROUPS | conv.EMBED_GROUPS)
        self.assertEqual(dense_embed.config_block(), {"q8_dense": True, "q8_embed": True})
        self.assertEqual(dense_embed.q8_suffixes(), LEGACY_DENSE_EMBED_SUFFIXES)
        self.assertEqual(conv.Policy(conv.DENSE_GROUPS).config_block(), {"q8_dense": True})
        self.assertEqual(conv.Policy(conv.DENSE_GROUPS).q8_suffixes(), LEGACY_DENSE_EMBED_SUFFIXES[:-1])
        self.assertEqual(conv.Policy(conv.EMBED_GROUPS).config_block(), {"q8_embed": True})
        self.assertEqual(conv.Policy().config_block(), {})
        self.assertEqual(conv.Policy().q8_suffixes(), LEGACY_DENSE_EMBED_SUFFIXES[:7])

    def test_other_sets_record_q8_groups_in_canonical_order(self):
        p = conv.Policy.parse("embed,head,attn")
        self.assertEqual(p.config_block(), {"q8_groups": ["attn", "head", "embed"]})
        self.assertEqual(
            p.q8_suffixes(),
            LEGACY_DENSE_EMBED_SUFFIXES[:8] + LEGACY_DENSE_EMBED_SUFFIXES[10:],
        )

    def test_from_config_inverts_config_block(self):
        names = list(conv.Q8_GROUPS)
        for mask in range(1 << len(names)):
            p = conv.Policy(frozenset(g for i, g in enumerate(names) if mask >> i & 1))
            self.assertEqual(conv.Policy.from_config(p.config_block()), p)
        with self.assertRaises(ValueError):
            conv.Policy.from_config({"q8_groups": ["head"], "q8_dense": True})


if __name__ == "__main__":
    unittest.main()
