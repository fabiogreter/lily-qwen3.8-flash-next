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
        c.gather.hidden_secs += 0.25;
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
    assert!((added.gather.hidden_secs - 0.25).abs() < 1e-12);
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

#[test]
fn cpu_load_is_the_busy_share_of_the_ticks_between_two_samples() {
    let at = |user, system, idle, nice| CpuTicks { user, system, idle, nice };
    let earlier = at(100, 50, 850, 0);
    assert_eq!(at(130, 60, 900, 10).load_since(earlier), Some(0.5));
    assert_eq!(at(100, 50, 950, 0).load_since(earlier), Some(0.0));
    // No tick passed, or a counter wrapped: no load to report.
    assert_eq!(earlier.load_since(earlier), None);
    assert_eq!(at(10, 50, 900, 0).load_since(earlier), None);
}

#[test]
fn machine_figures_can_be_sampled() {
    let a = cpu_ticks().expect("host_statistics");
    let b = cpu_ticks().expect("host_statistics");
    assert!(b.idle >= a.idle && b.user >= a.user);
    let memory = memory_use().expect("memory");
    assert!(memory.total > 0 && memory.used > 0 && memory.used <= memory.total);
    // `vm.swapusage` is refused inside some sandboxes; when it answers, the
    // swap in use is no more than the machine could page out.
    if let Some(swap) = swap_used() {
        assert!(swap < 64 * memory.total, "swap {swap}");
    }
}

#[test]
fn own_cpu_energy_is_cumulative() {
    let a = own_energy_nj().expect("proc_pid_rusage");
    // Some work for the counter to count.
    let mut x = 0u64;
    for i in 0..2_000_000u64 {
        x = std::hint::black_box(x.wrapping_mul(31).wrapping_add(i));
    }
    let b = own_energy_nj().expect("proc_pid_rusage");
    assert!(b >= a, "energy went back: {a} -> {b} ({x})");
}
