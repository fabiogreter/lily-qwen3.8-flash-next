use super::{common_prefix_len, resume_position};

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
