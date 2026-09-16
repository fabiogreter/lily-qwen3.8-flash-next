use super::{agreement, boundary_position, common_prefix_len, resume_position};

#[test]
fn agreement_is_the_longest_prefix_over_all_lineages_capped_below_the_prompt_end() {
    let prompt = [1, 2, 3, 4, 5];
    let gpu: &[u32] = &[1, 2, 9, 9];
    let disk: &[u32] = &[1, 2, 3, 7, 7, 7];
    // The disk lineage agrees further, and its tail is what it continued with.
    assert_eq!(agreement(&prompt, [gpu, disk]), (3, vec![7, 7, 7]));
    // Order does not matter.
    assert_eq!(agreement(&prompt, [disk, gpu]), (3, vec![7, 7, 7]));
    // A lineage that contains the whole prompt: capped at prompt_len - 1 like
    // every resume position, tail starts at the cap.
    assert_eq!(agreement(&prompt, [&[1, 2, 3, 4, 5, 6][..]]), (4, vec![5, 6]));
    // Nothing shared, or no lineages at all.
    assert_eq!(agreement(&prompt, [&[8, 8][..]]), (0, vec![]));
    assert_eq!(agreement(&prompt, std::iter::empty::<&[u32]>()), (0, vec![]));
    // The tail is bounded.
    let long: Vec<u32> = std::iter::once(1).chain(std::iter::repeat_n(4, 40)).collect();
    let (pos, tail) = agreement(&prompt, [long.as_slice()]);
    assert_eq!((pos, tail.len()), (1, 16));
}

#[test]
fn boundary_rule_materialises_only_a_long_unresumed_agreement() {
    // Two runs shared 2 000 tokens, nothing was resumable: materialise there.
    assert_eq!(boundary_position(2000, 0, 3000, 1024), Some(2000));
    // Below the threshold: not worth a disk entry.
    assert_eq!(boundary_position(900, 0, 3000, 1024), None);
    // Threshold 0 disables the feature entirely.
    assert_eq!(boundary_position(2000, 0, 3000, 0), None);
    // A pure extension (one growing conversation): agreement == reused.
    assert_eq!(boundary_position(2000, 2000, 3000, 1024), None);
    // Resumed past the agreement never happens, but must not materialise.
    assert_eq!(boundary_position(2000, 2500, 3000, 1024), None);
    // The agreement may sit at the last feedable position ...
    assert_eq!(boundary_position(2999, 0, 3000, 1024), Some(2999));
    // ... but never at or past the prompt end (a token must remain to feed).
    assert_eq!(boundary_position(3000, 0, 3000, 1024), None);
    assert_eq!(boundary_position(1, 0, 0, 1), None);
    // Exactly the threshold counts.
    assert_eq!(boundary_position(1024, 0, 3000, 1024), Some(1024));
}

#[test]
fn common_prefix_length() {
    assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 4]), 2);
    assert_eq!(common_prefix_len(&[1, 2], &[1, 2, 3]), 2);
    assert_eq!(common_prefix_len(&[], &[1]), 0);
    assert_eq!(common_prefix_len(&[5], &[1]), 0);
}

#[test]
fn resume_rule_prefers_live_end_then_checkpoints() {
    // Strict prefix: the live end, when resumable.
    assert_eq!(resume_position(&[1, 2, 3], &[2, 3], true, &[1, 2, 3, 4]), Some(3));
    // Live end not resumable (a disk entry without one): latest checkpoint.
    assert_eq!(resume_position(&[1, 2, 3], &[2], false, &[1, 2, 3, 4]), Some(2));
    // Divergence after 2: the checkpoint at 2 (not 3).
    assert_eq!(resume_position(&[1, 2, 3, 4], &[2, 3], true, &[1, 2, 9, 9]), Some(2));
    // Identical prompt: never the last token.
    assert_eq!(resume_position(&[1, 2, 3], &[2, 3], true, &[1, 2, 3]), Some(2));
    // Nothing shared, or nothing usable.
    assert_eq!(resume_position(&[5, 6], &[2], true, &[1, 2, 3]), None);
    assert_eq!(resume_position(&[1, 2, 3], &[3], true, &[1, 2, 9]), None);
}
