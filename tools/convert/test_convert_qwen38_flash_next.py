"""Unit tests for the converter's storage policy (`--q8`, `--q8-dense`,
`--q8-embed`, `--draft-q4`); no checkpoint and no mlx needed.

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
    p.add_argument("--draft-q4", action="store_true")
    p.add_argument("--q4-xl", action="store_true")
    return p.parse_args(list(argv))


def subsets():
    """Every subset of the q8 groups."""
    names = list(conv.Q8_GROUPS)
    for mask in range(1 << len(names)):
        yield frozenset(g for i, g in enumerate(names) if mask >> i & 1)


def source(name: str, shard: str = "a", start: int = 0) -> conv.SourceTensor:
    return conv.SourceTensor(name, "BF16", (64, 64), Path(shard), start, start + 8192)


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
        self.assertEqual(
            conv.policy_of_args(args("--q4-xl")),
            conv.policy_of_args(args("--q8", "attn,shared,head,embed", "--draft-q4")),
        )

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
        for groups in subsets():
            for draft_q4 in (False, True):
                if draft_q4 and not groups & conv.DRAFT_PATH_GROUPS:
                    continue
                p = conv.Policy(groups, draft_q4)
                self.assertEqual(conv.Policy.from_config(p.config_block()), p)
        with self.assertRaises(ValueError):
            conv.Policy.from_config({"q8_groups": ["head"], "q8_dense": True})


class DraftQ4Test(unittest.TestCase):
    def test_the_draft_head_stays_q4_and_the_trunk_follows_the_groups(self):
        for groups in subsets():
            if not groups & conv.DRAFT_PATH_GROUPS:
                continue
            policy = conv.Policy(groups, draft_q4=True)
            for name, group in TENSORS.items():
                if group is None:
                    want = FIXED_BITS[name]
                elif name.startswith("mtp."):
                    want = 4
                else:
                    want = 8 if group in groups else 4
                self.assertEqual(bits(name, policy), want, f"{name} under {sorted(groups)} + draft_q4")
            self.assertEqual(bits(conv.MTP_LM_HEAD_NAME, policy), 4)
            # The head's fixed 8-bit inputs are not part of the choice.
            self.assertEqual(bits("mtp.fc_embedding.weight", policy), 8)
            self.assertEqual(bits("mtp.layers.0.self_attn.indexer.index_qk_proj.weight", policy), 8)

    def test_it_needs_a_q8_group_on_the_draft_path(self):
        for groups in ((), ("gdn",), ("embed",), ("gdn", "embed")):
            with self.assertRaises(ValueError, msg=str(groups)):
                conv.Policy(frozenset(groups), draft_q4=True)
        for group in sorted(conv.DRAFT_PATH_GROUPS):
            self.assertTrue(conv.Policy(frozenset((group,)), draft_q4=True).draft_q4)
        with self.assertRaises(ValueError):
            conv.policy_of_args(args("--q8", "gdn,embed", "--draft-q4"))
        self.assertEqual(
            conv.policy_of_args(args("--q8", "attn,shared,head,embed", "--draft-q4")),
            conv.Policy(frozenset(("attn", "shared", "head", "embed")), True),
        )

    def test_the_config_gains_draft_q4_after_the_groups(self):
        q4_xl_trunk = conv.Policy.parse("attn,shared,head,embed")
        d4 = conv.Policy(q4_xl_trunk.q8, draft_q4=True)
        self.assertEqual(d4.config_block(), {"q8_groups": ["attn", "shared", "head", "embed"], "draft_q4": True})
        self.assertEqual(list(d4.config_block()), ["q8_groups", "draft_q4"])
        # The trunk's patterns are what they were.
        self.assertEqual(d4.q8_suffixes(), q4_xl_trunk.q8_suffixes())
        dense = conv.Policy(conv.DENSE_GROUPS, draft_q4=True)
        self.assertEqual(dense.config_block(), {"q8_dense": True, "draft_q4": True})
        # Without the flag, nothing changes (every policy is still draft_q4=False).
        for groups in subsets():
            self.assertNotIn("draft_q4", conv.Policy(groups).config_block())

    def test_the_head_copy_follows_the_last_mtp_tensor_and_reads_lm_head(self):
        tensors = [
            source("model.language_model.embed_tokens.weight", "s1", 0),
            source("lm_head.weight", "s1", 8192),
            source("mtp.fc_hidden.weight", "s2", 0),
            source("mtp.layers.0.self_attn.q_proj.weight", "s2", 8192),
            source("model.visual.blocks.0.attn.qkv.weight", "s3", 0),
        ]
        q4_xl_trunk = conv.Policy.parse("attn,shared,head,embed")
        self.assertIs(conv.with_draft_head_copy(tensors, q4_xl_trunk), tensors)
        # A Q4 trunk head is already what the draft head should read.
        self.assertIs(conv.with_draft_head_copy(tensors, conv.Policy(frozenset(("attn",)), True)), tensors)
        out = conv.with_draft_head_copy(tensors, conv.Policy(q4_xl_trunk.q8, True))
        self.assertEqual([t.name for t in out[:4]], [t.name for t in tensors[:4]])
        self.assertEqual(out[4].name, conv.MTP_LM_HEAD_NAME)
        self.assertEqual(out[5:], tensors[4:])
        head = tensors[1]
        self.assertEqual((out[4].shard, out[4].start, out[4].end, out[4].shape), (head.shard, head.start, head.end, head.shape))
        self.assertEqual(conv.category(out[4].name), "mtp")
        # Without a draft head in the source there is nothing to copy for.
        no_mtp = [t for t in tensors if not t.name.startswith("mtp.")]
        self.assertIs(conv.with_draft_head_copy(no_mtp, conv.Policy(q4_xl_trunk.q8, True)), no_mtp)
        # `--no-mtp` drops the copy with the rest of the head.
        t = source(conv.MTP_LM_HEAD_NAME)
        self.assertEqual(conv.plan_tensor(t, 48, conv.Quant(4, 32), mtp=False, policy=conv.Policy(q4_xl_trunk.q8, True)).kind, "drop")


if __name__ == "__main__":
    unittest.main()
