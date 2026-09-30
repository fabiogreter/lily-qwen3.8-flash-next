use std::sync::atomic::AtomicBool;

use super::*;

fn written(segments: &[Segment], cancel: Option<&AtomicBool>) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    // SAFETY: every host range below points into locals that outlive the call.
    unsafe { write_layout(segments, &mut out, cancel) }?;
    Ok(out)
}

#[test]
fn a_layout_writes_its_segments_in_order_and_compares_by_bytes_alone() {
    let rows = vec![5u8; 10 << 20];
    let a = [Segment::Bytes(vec![1, 2]), Segment::host(&rows), Segment::host(&[])];
    let bytes = written(&a, None).expect("write");
    assert_eq!(bytes.len(), 2 + rows.len());
    assert_eq!(layout_len(&a), bytes.len());
    assert_eq!(&bytes[..3], &[1, 2, 5]);
    // The same bytes split differently are the same layout; one byte off or
    // one byte short is not.
    let b = [
        Segment::Bytes(vec![1]),
        Segment::Bytes(vec![2, 5]),
        Segment::host(&rows[1..]),
    ];
    let mut other = rows.clone();
    *other.last_mut().unwrap() = 6;
    let c = [Segment::Bytes(vec![1, 2]), Segment::host(&other)];
    let d = [Segment::Bytes(vec![1, 2]), Segment::host(&rows[1..])];
    // SAFETY: as above.
    unsafe {
        assert!(layouts_equal(&a, &b));
        assert!(!layouts_equal(&a, &c));
        assert!(!layouts_equal(&a, &d));
        assert!(layouts_equal(&[], &[Segment::host(&[])]));
    }
}

#[test]
fn a_cancelled_layout_stops_between_pieces() {
    let rows = vec![0u8; 20 << 20];
    let cancel = AtomicBool::new(true);
    let error = written(&[Segment::host(&rows)], Some(&cancel)).expect_err("cancelled");
    assert_eq!(error.to_string(), "cancelled");
}
