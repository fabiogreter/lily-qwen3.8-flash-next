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

/// A batched step's phases are consecutive intervals between its marks, so
/// they add up to its wall time; the GPU's marks sit between `committed`
/// and `woke`.
#[test]
fn a_batched_steps_phases_cover_its_wall_time() {
    let t = RowsStepTiming {
        began: 10.0,
        staged: 10.0002,
        encoded: 10.0012,
        committed: 10.0013,
        gpu: PassTiming { gpu_start_secs: 10.0016, gpu_end_secs: 10.0206 },
        woke: 10.0207,
        ended: 10.0208,
        parked_staging: None,
    };
    let p = t.phases();
    let close = |a: f64, b: f64| (a - b).abs() < 1e-6;
    assert!(
        close(p.stage_ms, 0.2) && close(p.encode_ms, 1.0) && close(p.commit_ms, 0.1)
    );
    assert!(close(p.submit_ms, 0.3) && close(p.gpu_ms, 19.0) && close(p.wake_ms, 0.1));
    assert!(close(p.finish_ms, 0.1));
    let sum = p.stage_ms
        + p.encode_ms
        + p.commit_ms
        + p.submit_ms
        + p.gpu_ms
        + p.wake_ms
        + p.finish_ms;
    assert!(close(sum, t.wall_ms()), "{sum} vs {}", t.wall_ms());
}

/// A parked step is encoded and committed while the step before it runs and
/// staged only after that step's draws were read: its staging phase is the
/// release (`parked_staging` to `staged`), its encoding counts from `began`,
/// and its submission gap from the release, negative when the GPU started
/// the pass (running it up to its wait) before the host released it.
#[test]
fn a_parked_steps_phases_measure_its_release() {
    let t = RowsStepTiming {
        began: 10.0,
        encoded: 10.0010,
        committed: 10.0011,
        parked_staging: Some(10.0153),
        staged: 10.0155,
        gpu: PassTiming { gpu_start_secs: 10.0150, gpu_end_secs: 10.0300 },
        woke: 10.0301,
        ended: 10.0302,
    };
    let p = t.phases();
    let close = |a: f64, b: f64| (a - b).abs() < 1e-6;
    assert!(close(p.encode_ms, 1.0) && close(p.commit_ms, 0.1), "{p:?}");
    assert!(close(p.stage_ms, 0.2) && close(p.submit_ms, -0.5), "{p:?}");
    assert!(close(p.gpu_ms, 15.0) && close(p.wake_ms, 0.1), "{p:?}");
    assert!(close(p.finish_ms, 0.1), "{p:?}");
}
