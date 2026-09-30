use super::*;

#[test]
fn counters_are_per_thread_and_subtract_to_what_a_phase_added() {
    let before = counters();
    record(|c| {
        c.gather.batches += 1;
        c.gather.rows += 16;
        c.gather.pages += 50;
        c.gather.cold_pages += 3;
        c.gather.secs += 0.002;
        c.prefill.gpu_secs += 1.5;
    });
    let added = counters().since(before);
    assert_eq!(
        (
            added.gather.batches,
            added.gather.rows,
            added.gather.pages,
            added.gather.cold_pages
        ),
        (1, 16, 50, 3)
    );
    assert!((added.gather.secs - 0.002).abs() < 1e-12);
    assert!((added.prefill.gpu_secs - 1.5).abs() < 1e-12);
    assert_eq!(added.prefill.chunks, 0);
    // Another thread starts from zero and does not see this one's counts.
    let other = std::thread::spawn(counters).join().unwrap();
    assert_eq!(other, Counters::default());
    // Never negative, whichever way round the samples are passed.
    assert_eq!(before.since(counters()).gather.rows, 0);
}

#[test]
fn mach_layouts_match_the_sdk_headers() {
    use std::mem::{offset_of, size_of};
    assert_eq!(size_of::<sys::VmStatistics64>(), 152);
    assert_eq!(offset_of!(sys::VmStatistics64, pageins), 32);
    assert_eq!(offset_of!(sys::VmStatistics64, decompressions), 96);
    assert_eq!(offset_of!(sys::VmStatistics64, swapouts), 120);
    assert_eq!(sys::VmStatistics64::COUNT, 38);
    assert_eq!(size_of::<sys::TaskVmInfo>(), 152);
    assert_eq!(offset_of!(sys::TaskVmInfo, compressed), 120);
    assert_eq!(offset_of!(sys::TaskVmInfo, phys_footprint), 144);
    assert_eq!(sys::TaskVmInfo::COUNT, 38);
}

#[test]
fn system_memory_can_be_sampled() {
    let a = vm_counters().expect("host_statistics64");
    let b = vm_counters().expect("host_statistics64");
    // Cumulative counters: a later sample never has fewer.
    assert!(b.pageins >= a.pageins && b.compressions >= a.compressions);
    assert_eq!(a.since(b), VmCounters::default());
    let task = task_memory().expect("task_info");
    assert!(task.phys_footprint > 0);
    // The sysctl is refused inside some sandboxes; when it answers, the
    // level is one of the three the kernel reports.
    if let Some(level) = pressure_level() {
        assert!(matches!(level, 1 | 2 | 4), "pressure level {level}");
        assert_ne!(pressure_name(level), "unknown");
    }
}
