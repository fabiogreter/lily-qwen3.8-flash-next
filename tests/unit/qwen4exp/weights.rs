use super::*;

const GB: u64 = 1 << 30;
/// The checkpoint's shape as `docs/low-ram-experts.md` measures it: 24 576
/// (layer, expert) slices of about 2.57 MB, 3.1 GB of resident weights.
const SLICES: u64 = 24_576;
const NUM_EXPERTS: usize = 512;
const SLICE: u64 = 2_573_000;
const EXPERTS: u64 = SLICES * SLICE;
const OTHER: u64 = 3_100_000_000;

fn plan(ram: u64, session: u64) -> Option<usize> {
    plan_expert_slots(ram, EXPERTS, OTHER, SLICES, NUM_EXPERTS, session)
}

#[test]
fn the_storage_policy_moves_each_group_and_nothing_else() {
    use Q8Group::*;
    let l = "model.language_model.layers.3.";
    let at = |s: &str| format!("{l}{s}");
    // One base list per loader call, with the group it belongs to.
    let calls: Vec<(Vec<String>, Q8Group)> = vec![
        (
            vec![
                at("self_attn.q_proj"),
                at("self_attn.k_proj"),
                at("self_attn.v_proj"),
            ],
            Attn,
        ),
        (vec![at("self_attn.o_proj")], Attn),
        (vec!["mtp.layers.0.self_attn.o_proj".into()], Attn),
        (
            ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"]
                .iter()
                .map(|p| at(&format!("linear_attn.{p}")))
                .collect(),
            Gdn,
        ),
        (vec![at("linear_attn.out_proj")], Gdn),
        (
            vec![at("mlp.shared_expert.gate_proj"), at("mlp.shared_expert.up_proj")],
            Shared,
        ),
        (vec![at("mlp.shared_expert.down_proj")], Shared),
        (
            vec![
                "mtp.layers.0.mlp.shared_expert.gate_proj".into(),
                "mtp.layers.0.mlp.shared_expert.up_proj".into(),
            ],
            Shared,
        ),
        (vec!["lm_head".into()], Head),
        (vec!["model.language_model.embed_tokens".into()], Embed),
    ];
    let fixed_q8 = [
        at("mlp.gate"),
        at("mlp.shared_expert_gate"),
        at("self_attn.indexer.index_qk_proj"),
        "mtp.fc_hidden".to_string(),
    ];
    let experts = at("mlp.experts.gate_proj");
    // Every subset of the five groups.
    for mask in 0..32u32 {
        let groups: Vec<Q8Group> = Q8Group::ALL
            .into_iter()
            .enumerate()
            .filter(|(i, _)| mask >> i & 1 == 1)
            .map(|(_, g)| g)
            .collect();
        let policy = StoragePolicy::q8(&groups);
        for (bases, group) in &calls {
            let bases: Vec<&str> = bases.iter().map(String::as_str).collect();
            let want = if groups.contains(group) { 8 } else { 4 };
            assert_eq!(
                expected_bits(policy, &bases),
                want,
                "{bases:?} under {groups:?}"
            );
        }
        for b in &fixed_q8 {
            assert_eq!(expected_bits(policy, &[b]), 8, "{b}");
        }
        assert_eq!(expected_bits(policy, &[&experts]), 4, "experts");
    }
    // A stack that straddled two groups is no group's.
    let mixed = [at("self_attn.o_proj"), at("linear_attn.out_proj")];
    let mixed: Vec<&str> = mixed.iter().map(String::as_str).collect();
    assert_eq!(expected_bits(StoragePolicy::q8(&Q8Group::ALL), &mixed), 4);
}

#[test]
fn the_legacy_policies_are_the_group_sets_they_stand_for() {
    use Q8Group::*;
    assert_eq!(StoragePolicy::Q4, StoragePolicy::q8(&[]));
    assert_eq!(StoragePolicy::Q4.q8_groups(), []);
    assert_eq!(
        StoragePolicy::q8(&Q8Group::DENSE).q8_groups(),
        [Attn, Gdn, Shared, Head]
    );
}

#[test]
fn a_machine_that_holds_the_checkpoint_plans_nothing_whatever_the_session() {
    assert_eq!(plan(128 * GB, 0), None);
    // The session reserve does not decide whether the checkpoint fits: a
    // machine that holds it keeps its usual session budget.
    assert_eq!(plan(128 * GB, 9 * GB), None);
}

#[test]
fn the_session_reserve_comes_out_of_the_experts_slot_for_slot() {
    let without = plan(64 * GB, 0).expect("64 GB needs the cache");
    for session in [4_900_000_000u64, 9_000_000_000] {
        let with = plan(64 * GB, session).expect("still the cache");
        let cost = without - with;
        let expected = (session / SLICE) as usize;
        assert!(
            cost.abs_diff(expected) <= 1,
            "{session}: {cost} slots, expected {expected}"
        );
    }
}

#[test]
fn a_longer_context_keeps_fewer_experts_resident() {
    let at_131k = plan(64 * GB, 4_900_000_000).unwrap();
    let at_262k = plan(64 * GB, 9_000_000_000).unwrap();
    assert!(at_262k < at_131k, "{at_262k} vs {at_131k}");
}

#[test]
fn the_slab_never_drops_below_two_layers() {
    assert_eq!(plan(16 * GB, 9 * GB), Some(2 * NUM_EXPERTS));
}

/// The formula the plan reserves from, on the real checkpoint's config
/// (config.json only, no weights, so no instance lock): a token costs the
/// server's reported 30 784 B with the draft head, and one full session is
/// the per-token caches plus the recurrent state and its checkpoints.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH (reads config.json only)"]
fn a_session_costs_its_tokens_plus_the_recurrent_state() {
    let Ok(dir) = std::env::var("LILY_MODEL_DIR_FLASH") else { return };
    let cfg = Qwen4ExpConfig::from_model_dir(&dir).expect("config");
    let at = |mtp, tokens, checkpoints| {
        super::super::model::session_bytes(&cfg, mtp, tokens, checkpoints).unwrap()
    };
    // Whole capacity steps, so the rounding does not blur the difference.
    let (short, long) = (65_536usize, 131_072usize);
    let per_token = (at(true, long, 0) - at(true, short, 0)) / (long - short) as u64;
    assert_eq!(per_token, 30_784, "the server's B/token of context");
    // Without the draft head, one attention layer fewer.
    assert!(at(false, long, 0) < at(true, long, 0));
    // Checkpoints add a snapshot each.
    let snapshot = at(false, long, 1) - at(false, long, 0);
    assert_eq!(at(false, long, 3) - at(false, long, 0), 3 * snapshot);
    let one = at(false, long, 3);
    let full = at(false, 262_144, 3);
    println!(
        "one session under the expert cache: {one} B at 131 072 tokens, {full} B at 262 144"
    );
    assert!((4_000_000_000..5_000_000_000).contains(&one), "131 072 tokens: {one} B");
    assert!((7_500_000_000..9_000_000_000).contains(&full), "262 144 tokens: {full} B");
}
