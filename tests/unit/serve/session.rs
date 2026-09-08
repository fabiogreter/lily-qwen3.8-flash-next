use super::common_prefix_len;

#[test]
fn common_prefix_length() {
    assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 4]), 2);
    assert_eq!(common_prefix_len(&[1, 2], &[1, 2, 3]), 2);
    assert_eq!(common_prefix_len(&[], &[1]), 0);
    assert_eq!(common_prefix_len(&[5], &[1]), 0);
}
