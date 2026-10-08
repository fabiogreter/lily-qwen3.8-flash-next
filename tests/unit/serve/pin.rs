use std::sync::atomic::{AtomicU32, Ordering};

use super::*;
use crate::metal::MetalContext;

const GIB: u64 = 1 << 30;
/// `vm.user_wire_limit` on a 128 GB machine (85 % of 128 GiB).
const WIRE_128: u64 = 116_823_110_451;
/// The same share of a 64 GB machine.
const WIRE_64: u64 = WIRE_128 / 2;
/// The fully resident checkpoint's weights, vision tower included.
const FULL: u64 = 74_000_000_000;
/// What stays resident next to an expert cache (3.1 GB) plus the tower.
const NON_EXPERT: u64 = 4_000_000_000;

fn inputs(mode: PinMode) -> PinInputs {
    PinInputs {
        mode,
        planned_memory: Some(128 * GIB),
        physical_memory: Some(128 * GIB),
        wire_limit: Some(WIRE_128),
        pin_bytes: FULL,
        locked_elsewhere: 0,
        expert_cache: false,
    }
}

fn skip_reason(i: &PinInputs) -> String {
    match decide(i) {
        PinDecision::Skip(why) => why,
        PinDecision::Pin => panic!("expected a skip for {i:?}"),
    }
}

#[test]
fn auto_pins_the_fully_resident_model_on_128_gb() {
    assert_eq!(decide(&inputs(PinMode::Auto)), PinDecision::Pin);
    assert_eq!(decide(&inputs(PinMode::Always)), PinDecision::Pin);
}

#[test]
fn auto_skips_a_real_64_gb_machine_with_the_expert_cache() {
    let i = PinInputs {
        planned_memory: Some(64 * GIB),
        physical_memory: Some(64 * GIB),
        wire_limit: Some(WIRE_64),
        pin_bytes: NON_EXPERT,
        expert_cache: true,
        ..inputs(PinMode::Auto)
    };
    assert_eq!(skip_reason(&i), "expert cache active");
}

#[test]
fn a_simulated_64_gb_machine_decides_like_one() {
    let i = PinInputs {
        planned_memory: Some(64 * GIB),
        pin_bytes: NON_EXPERT,
        expert_cache: true,
        ..inputs(PinMode::Auto)
    };
    assert_eq!(skip_reason(&i), "expert cache active");
    // The planned memory, not the physical one, sets the margin: 74 GB of
    // weights do not leave the reserve free of 64 GiB even if some load
    // kept them all resident.
    let i = PinInputs { planned_memory: Some(64 * GIB), ..inputs(PinMode::Auto) };
    assert!(skip_reason(&i).contains("planned memory"), "{}", skip_reason(&i));
    // A budget above the physical memory never plans with more than it.
    let i = PinInputs {
        planned_memory: Some(512 * GIB),
        physical_memory: Some(80 * GIB),
        ..inputs(PinMode::Auto)
    };
    assert!(skip_reason(&i).contains("planned memory"), "{}", skip_reason(&i));
}

#[test]
fn always_pins_the_non_expert_weights_of_a_64_gb_machine_within_the_wire_limit() {
    let i = PinInputs {
        planned_memory: Some(64 * GIB),
        physical_memory: Some(64 * GIB),
        wire_limit: Some(WIRE_64),
        pin_bytes: NON_EXPERT,
        expert_cache: true,
        ..inputs(PinMode::Always)
    };
    assert_eq!(decide(&i), PinDecision::Pin);
    // `always` ignores the planned-memory margin, not the wire limit.
    let i = PinInputs { pin_bytes: 60 * GIB, ..i };
    assert!(skip_reason(&i).contains("wire limit"), "{}", skip_reason(&i));
}

#[test]
fn a_too_small_wire_limit_skips() {
    for mode in [PinMode::Auto, PinMode::Always] {
        let i = PinInputs { wire_limit: Some(80_000_000_000), ..inputs(mode) };
        assert!(skip_reason(&i).contains("wire limit"), "{}", skip_reason(&i));
        // What `--ngram-lock` locks counts against the same limit.
        let i = PinInputs { locked_elsewhere: 32 * GIB, ..inputs(mode) };
        let why = skip_reason(&i);
        assert!(why.contains("--ngram-lock"), "{why}");
    }
    // An unreadable limit counts as half the physical memory: too little
    // for the full model, enough for a few gigabytes.
    let unknown = PinInputs { wire_limit: None, ..inputs(PinMode::Auto) };
    assert!(skip_reason(&unknown).contains("unreadable"), "{}", skip_reason(&unknown));
    let small = PinInputs { pin_bytes: 4 * GIB, ..unknown };
    assert_eq!(decide(&small), PinDecision::Pin);
    let blind = PinInputs { physical_memory: None, ..unknown };
    assert!(matches!(decide(&blind), PinDecision::Skip(_)));
}

#[test]
fn off_never_pins_and_nothing_to_pin_is_skipped() {
    assert_eq!(skip_reason(&inputs(PinMode::Off)), "--pin-weights off");
    let none = PinInputs { pin_bytes: 0, ..inputs(PinMode::Always) };
    assert_eq!(skip_reason(&none), "no weight buffers to pin");
}

#[test]
fn modes_parse() {
    assert_eq!("auto".parse::<PinMode>().unwrap(), PinMode::Auto);
    assert_eq!("off".parse::<PinMode>().unwrap(), PinMode::Off);
    assert_eq!("always".parse::<PinMode>().unwrap(), PinMode::Always);
    assert!("on".parse::<PinMode>().is_err());
}

// --- the pin itself, on small Metal buffers -------------------------------

/// The user wiring of the VM entry that holds `address`: nonzero while it
/// is `mlock`ed (`mach_vm_region`, `VM_REGION_BASIC_INFO_64`).
fn user_wired_count(address: usize) -> u16 {
    #[repr(C, packed(4))]
    #[derive(Default)]
    struct BasicInfo64 {
        protection: i32,
        max_protection: i32,
        inheritance: u32,
        shared: u32,
        reserved: u32,
        offset: u64,
        behavior: i32,
        user_wired_count: u16,
    }
    unsafe extern "C" {
        static mach_task_self_: u32;
        fn mach_vm_region(
            task: u32,
            address: *mut u64,
            size: *mut u64,
            flavor: i32,
            info: *mut i32,
            count: *mut u32,
            object_name: *mut u32,
        ) -> i32;
    }
    const VM_REGION_BASIC_INFO_64: i32 = 9;
    let mut info = BasicInfo64::default();
    let mut count = (std::mem::size_of::<BasicInfo64>() / 4) as u32;
    let (mut addr, mut size, mut object) = (address as u64, 0u64, 0u32);
    // SAFETY: out-pointers to locals of the flavor's layout and length.
    let rc = unsafe {
        mach_vm_region(
            mach_task_self_,
            &mut addr,
            &mut size,
            VM_REGION_BASIC_INFO_64,
            (&raw mut info).cast(),
            &mut count,
            &mut object,
        )
    };
    assert_eq!(rc, 0, "mach_vm_region");
    assert!(addr as usize <= address, "no region holds {address:#x}");
    info.user_wired_count
}

/// The Metal tests below run one at a time: small buffers of two tests can
/// share a VM entry, whose wiring the other test's pin would then show.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn buffers(ctx: &MetalContext) -> Vec<Buffer> {
    // A few hundred MB in sizes like the weights', plus small ones that may
    // share pages.
    [128 << 20, 64 << 20, 96 << 20, 4096, 100, 1 << 20]
        .into_iter()
        .map(|len| ctx.new_buffer(len).expect("buffer"))
        .collect()
}

fn wired(buffers: &[Buffer]) -> Vec<bool> {
    buffers
        .iter()
        .map(|b| user_wired_count(b.contents().as_ptr() as usize) > 0)
        .collect()
}

static PRESSURE: AtomicU32 = AtomicU32::new(1);

fn fake_pressure() -> Option<u32> {
    Some(PRESSURE.load(Ordering::SeqCst))
}

fn normal_pressure() -> Option<u32> {
    Some(1)
}

#[test]
fn page_ranges_cover_every_buffer_once_in_whole_pages() {
    let _serial = serial();
    let ctx = MetalContext::new().expect("metal");
    let bufs = buffers(&ctx);
    let (ranges, total) = page_ranges(&bufs);
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let payload: usize = bufs.iter().map(|b| b.length()).sum();
    assert!(
        total as usize >= payload && total as usize <= payload + 2 * page * bufs.len()
    );
    for pair in ranges.windows(2) {
        assert!(pair[0].0 + pair[0].1 < pair[1].0, "ranges overlap or touch: {pair:?}");
    }
    for &(start, len) in &ranges {
        assert_eq!((start % page, len % page), (0, 0));
    }
    for b in &bufs {
        let (s, e) = (
            b.contents().as_ptr() as usize,
            b.contents().as_ptr() as usize + b.length(),
        );
        assert!(ranges.iter().any(|&(start, len)| start <= s && e <= start + len));
    }
}

#[test]
fn pins_at_the_first_request_holds_then_releases_after_the_hold() {
    let _serial = serial();
    let ctx = MetalContext::new().expect("metal");
    let bufs = buffers(&ctx);
    let mut pin = WeightPin::new(bufs.clone(), PinDecision::Pin, 60, normal_pressure)
        .expect("pin");
    assert!(!pin.pinned());
    assert_eq!(wired(&bufs), vec![false; bufs.len()]);
    assert_eq!(pin.keep_warm_remaining(Instant::now()), None, "no request yet");

    let t0 = Instant::now();
    assert!(pin.before_request(t0).pinned(), "pins at the first request");
    assert_eq!(wired(&bufs), vec![true; bufs.len()]);
    pin.after_request(t0);
    assert_eq!(pin.hold_remaining(t0), Some(Duration::from_secs(60)));
    // The keep-alive's window is the hold.
    assert_eq!(pin.keep_warm_remaining(t0), Some(Duration::from_secs(60)));
    let t = t0 + Duration::from_secs(25);
    assert_eq!(pin.keep_warm_remaining(t), Some(Duration::from_secs(35)));

    // A request inside the hold finds it pinned; the hold restarts.
    let t1 = t0 + Duration::from_secs(30);
    assert!(pin.before_request(t1).pinned());
    pin.after_request(t1);
    pin.release_if_held_out(t1 + Duration::from_secs(59));
    assert!(pin.pinned(), "still inside the hold");

    let t2 = t1 + Duration::from_secs(60);
    assert_eq!(pin.hold_remaining(t2), Some(Duration::ZERO));
    assert_eq!(pin.keep_warm_remaining(t2), Some(Duration::ZERO), "window over");
    pin.release_if_held_out(t2);
    assert!(!pin.pinned());
    assert_eq!(wired(&bufs), vec![false; bufs.len()]);
    assert_eq!(pin.hold_remaining(t2), None, "nothing to hold");
    assert_eq!(pin.keep_warm_remaining(t2), None, "released: no keep-alive");

    // The next active period pins again.
    assert!(pin.before_request(t2 + Duration::from_secs(600)).pinned());
    assert_eq!(wired(&bufs), vec![true; bufs.len()]);
    drop(pin);
    assert_eq!(wired(&bufs), vec![false; bufs.len()], "dropping the pin unlocks");
}

#[test]
fn the_pin_keeps_its_buffers_alive_and_unlocks_before_releasing_them() {
    let _serial = serial();
    let ctx = MetalContext::new().expect("metal");
    let bufs = buffers(&ctx);
    let first = bufs[0].contents().as_ptr() as usize;
    let handles: Vec<_> = bufs.iter().map(std::rc::Rc::downgrade).collect();
    let mut pin =
        WeightPin::new(bufs, PinDecision::Pin, 0, normal_pressure).expect("pin");
    let t0 = Instant::now();
    assert!(pin.before_request(t0).pinned());
    assert_eq!(pin.hold_remaining(Instant::now()), None, "a hold of 0 never ends");
    // The keep-alive still ends: a minute after the last request.
    pin.after_request(t0);
    assert_eq!(pin.keep_warm_remaining(t0), Some(Duration::from_secs(60)));
    assert_eq!(
        pin.keep_warm_remaining(t0 + Duration::from_secs(90)),
        Some(Duration::ZERO)
    );
    assert!(pin.pinned(), "while the pin itself holds");
    // The engine drops its model (the other handles) while the pin lives:
    // the memory stays allocated, and locked.
    assert!(handles.iter().all(|h| h.upgrade().is_some()));
    assert!(user_wired_count(first) > 0);
    // Dropping the pin unlocks first, then frees.
    drop(pin);
    assert!(
        handles.iter().all(|h| h.upgrade().is_none()),
        "the buffers went with the pin"
    );
}

#[test]
fn a_warning_between_requests_releases_the_pin_and_blocks_repinning_until_normal() {
    let _serial = serial();
    let ctx = MetalContext::new().expect("metal");
    let bufs = buffers(&ctx);
    PRESSURE.store(1, Ordering::SeqCst);
    let mut pin =
        WeightPin::new(bufs.clone(), PinDecision::Pin, 60, fake_pressure).expect("pin");
    let t0 = Instant::now();
    assert!(pin.before_request(t0).pinned());
    pin.after_request(t0);
    PRESSURE.store(2, Ordering::SeqCst);
    // The monitor polls every second.
    let deadline = Instant::now() + Duration::from_secs(5);
    while pin.pinned() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(!pin.pinned(), "released under warning");
    assert_eq!(wired(&bufs), vec![false; bufs.len()]);
    assert_eq!(pin.keep_warm_remaining(t0), None, "the release ends the keep-alive");
    // No re-pin while the pressure is raised, even in the same period...
    assert!(!pin.before_request(t0 + Duration::from_secs(1)).pinned());
    assert!(!pin.pinned());
    // ...and back at the next request once it is normal.
    PRESSURE.store(1, Ordering::SeqCst);
    assert!(pin.before_request(t0 + Duration::from_secs(2)).pinned());
    assert_eq!(wired(&bufs), vec![true; bufs.len()]);
}

/// Polls `done` every 50 ms for up to 5 s (the monitor reads the pressure
/// once a second).
fn wait_for(done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !done() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn a_warning_during_a_request_releases_the_pin_when_the_request_ends() {
    let _serial = serial();
    let ctx = MetalContext::new().expect("metal");
    let bufs = buffers(&ctx);
    PRESSURE.store(1, Ordering::SeqCst);
    let mut pin =
        WeightPin::new(bufs.clone(), PinDecision::Pin, 60, fake_pressure).expect("pin");
    let request = pin.before_request(Instant::now());
    assert!(request.pinned());
    PRESSURE.store(2, Ordering::SeqCst);
    wait_for(|| pin.shared.lock().deferred.is_some());
    assert_eq!(pin.shared.lock().deferred, Some(WARNING), "the monitor deferred");
    assert!(pin.pinned(), "not released under the running request");
    assert_eq!(wired(&bufs), vec![true; bufs.len()]);
    // The request's end releases, at once rather than at the next poll.
    drop(request);
    assert!(!pin.pinned(), "released at the request's end");
    assert_eq!(wired(&bufs), vec![false; bufs.len()]);
    let t = Instant::now();
    pin.after_request(t);
    assert_eq!(pin.keep_warm_remaining(t), None, "the release ends the keep-alive");
    // No re-pin while the warning lasts, then back once it is normal.
    assert!(!pin.before_request(t).pinned());
    PRESSURE.store(1, Ordering::SeqCst);
    assert!(pin.before_request(t).pinned());
}

#[test]
fn critical_releases_the_pin_mid_request() {
    let _serial = serial();
    let ctx = MetalContext::new().expect("metal");
    let bufs = buffers(&ctx);
    PRESSURE.store(1, Ordering::SeqCst);
    let mut pin =
        WeightPin::new(bufs.clone(), PinDecision::Pin, 60, fake_pressure).expect("pin");
    let request = pin.before_request(Instant::now());
    assert!(request.pinned());
    PRESSURE.store(4, Ordering::SeqCst);
    wait_for(|| !pin.pinned());
    assert!(!pin.pinned(), "released with the request in flight");
    assert_eq!(wired(&bufs), vec![false; bufs.len()]);
    drop(request);
    assert!(!pin.pinned());
    PRESSURE.store(1, Ordering::SeqCst);
}

#[test]
fn a_warning_found_at_the_start_of_a_request_releases_before_it_runs() {
    let _serial = serial();
    let ctx = MetalContext::new().expect("metal");
    let bufs = buffers(&ctx);
    PRESSURE.store(1, Ordering::SeqCst);
    let mut pin =
        WeightPin::new(bufs.clone(), PinDecision::Pin, 60, fake_pressure).expect("pin");
    let t0 = Instant::now();
    assert!(pin.before_request(t0).pinned());
    pin.after_request(t0);
    PRESSURE.store(2, Ordering::SeqCst);
    // Whether the monitor read the warning first or the request does, the
    // request starts unpinned instead of deferring the release past itself.
    let request = pin.before_request(t0 + Duration::from_secs(1));
    assert!(!request.pinned());
    assert!(!pin.pinned());
    assert_eq!(wired(&bufs), vec![false; bufs.len()]);
    drop(request);
    PRESSURE.store(1, Ordering::SeqCst);
}

// --- the pressure policy, without Metal ---------------------------------

#[test]
fn warning_waits_for_the_requests_in_flight_and_anything_above_it_does_not() {
    use super::PressureAction::{Defer, Keep, Release};
    for running in [0, 1, 2] {
        assert_eq!(pressure_action(None, running), Keep, "unreadable");
        assert_eq!(pressure_action(Some(1), running), Keep, "normal");
        assert_eq!(pressure_action(Some(4), running), Release(4), "critical");
        // A level without a name above warning counts as above it.
        assert_eq!(pressure_action(Some(3), running), Release(3));
    }
    assert_eq!(pressure_action(Some(WARNING), 0), Release(WARNING));
    assert_eq!(pressure_action(Some(WARNING), 1), Defer(WARNING));
    assert_eq!(pressure_action(Some(WARNING), 2), Defer(WARNING));
}

/// A pinned state over no memory (unlocking nothing): the policy's state
/// machine without Metal or `mlock`.
fn pinned_state(in_flight: usize) -> State {
    State { ranges: Vec::new(), pinned: true, stop: false, in_flight, deferred: None }
}

#[test]
fn a_deferred_release_waits_for_the_last_request_in_flight() {
    let mut state = pinned_state(2);
    state.on_pressure(Some(WARNING), 2);
    assert!(state.pinned, "deferred, not released");
    assert_eq!(state.deferred, Some(WARNING));
    // The next poll reads it again: still one deferral.
    state.on_pressure(Some(WARNING), 2);
    assert_eq!(state.deferred, Some(WARNING));
    state.end_request();
    assert!(state.pinned, "another request is still in flight");
    state.end_request();
    assert!(!state.pinned, "released as the last one ended");
    assert_eq!((state.in_flight, state.deferred), (0, None));
}

#[test]
fn a_warning_seen_during_a_request_releases_at_its_end_even_once_normal() {
    let mut state = pinned_state(1);
    state.on_pressure(Some(WARNING), 1);
    state.on_pressure(Some(1), 1);
    assert!(state.pinned);
    assert_eq!(state.deferred, Some(WARNING), "the warning stays decided");
    state.end_request();
    assert!(!state.pinned);
}

#[test]
fn critical_after_a_deferral_releases_at_once() {
    let mut state = pinned_state(1);
    state.on_pressure(Some(WARNING), 1);
    state.on_pressure(Some(4), 1);
    assert!(!state.pinned, "released mid-request");
    assert_eq!(state.deferred, None);
    // The request's end finds nothing left to release.
    state.end_request();
    assert_eq!((state.pinned, state.in_flight), (false, 0));
}

#[test]
fn a_request_end_without_a_warning_keeps_the_pin() {
    let mut state = pinned_state(1);
    state.on_pressure(Some(1), 1);
    state.on_pressure(None, 1);
    state.end_request();
    assert!(state.pinned);
    assert_eq!((state.in_flight, state.deferred), (0, None));
    // An unbalanced end cannot wrap the count.
    state.end_request();
    assert_eq!(state.in_flight, 0);
}

#[test]
fn a_skipped_decision_never_pins() {
    let _serial = serial();
    let ctx = MetalContext::new().expect("metal");
    let bufs = buffers(&ctx);
    let mut pin = WeightPin::new(
        bufs.clone(),
        PinDecision::Skip("--pin-weights off".into()),
        60,
        normal_pressure,
    )
    .expect("pin");
    let t0 = Instant::now();
    assert!(!pin.before_request(t0).pinned());
    pin.after_request(t0);
    assert_eq!(wired(&bufs), vec![false; bufs.len()]);
    assert!(pin.monitor.is_none(), "no monitor thread without a pin");
    // `--pin-weights off`, the expert cache under `auto`: no keep-alive.
    assert_eq!(pin.keep_warm_remaining(t0), None);
}

#[test]
fn a_failed_lock_unlocks_what_it_locked_and_waits_for_the_next_period() {
    let _serial = serial();
    let ctx = MetalContext::new().expect("metal");
    let bufs = buffers(&ctx);
    let mut pin = WeightPin::new(bufs.clone(), PinDecision::Pin, 60, normal_pressure)
        .expect("pin");
    // A last range that cannot be locked: an address nothing maps.
    let bogus = (8usize << 40, 1usize << 20);
    pin.shared.lock().ranges.push(bogus);
    let t0 = Instant::now();
    assert!(!pin.before_request(t0).pinned(), "the lock failed");
    assert_eq!(
        wired(&bufs),
        vec![false; bufs.len()],
        "the earlier ranges were unlocked"
    );
    pin.after_request(t0);
    assert_eq!(pin.keep_warm_remaining(t0), None, "unpinned: no keep-alive");
    // Not retried within the period...
    pin.shared.lock().ranges.pop();
    assert!(!pin.before_request(t0 + Duration::from_secs(30)).pinned());
    pin.after_request(t0 + Duration::from_secs(30));
    // ...but at the next one.
    assert!(pin.before_request(t0 + Duration::from_secs(120)).pinned());
    assert_eq!(wired(&bufs), vec![true; bufs.len()]);
}
