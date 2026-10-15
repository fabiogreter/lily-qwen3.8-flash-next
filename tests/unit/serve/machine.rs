use std::collections::VecDeque;

use super::*;

/// A reading with distinguishable figures.
fn reading(sys_w: f64) -> Reading {
    Reading {
        cpu_w: 1.0,
        gpu_w: 0.5,
        ane_w: 0.0,
        sys_w: Some(sys_w),
        gpu_active: 0.25,
        gpu_freq_mhz: 338.0,
        cpu_temp_c: Some(48.0),
        gpu_temp_c: Some(47.0),
        lily: None,
    }
}

fn sample(start_ms: u64, end_ms: u64, sys_w: f64) -> Sample {
    Sample { start_ms, end_ms, reading: reading(sys_w) }
}

/// Every call a source got: when, and the window it was asked for.
type Calls = Arc<Mutex<Vec<(Instant, Duration)>>>;

fn calls() -> Calls {
    Arc::default()
}

/// A source that plays `script` (then readings of 10 W) and notes each call.
/// Like the real one it waits out the window it is asked for (`honours`),
/// which is none for the zero window of every call after the first.
struct Script {
    script: VecDeque<Result<Reading, String>>,
    calls: Calls,
    honours: bool,
    panics: bool,
}

impl Script {
    fn new(script: Vec<Result<Reading, String>>, calls: &Calls) -> Self {
        Self {
            script: script.into(),
            calls: calls.clone(),
            honours: true,
            panics: false,
        }
    }
}

impl Source for Script {
    fn read(&mut self, window: Duration) -> Result<Reading, String> {
        lock(&self.calls).push((Instant::now(), window));
        if self.honours {
            std::thread::sleep(window);
        }
        if self.panics {
            panic!("IOReport exploded");
        }
        self.script.pop_front().unwrap_or_else(|| Ok(reading(10.0)))
    }
}

/// Waits short enough for a test to run in real time.
const FAST: Intervals = Intervals {
    busy: Duration::from_millis(10),
    idle: Duration::from_millis(80),
    page_open: Duration::from_millis(150),
    first: Duration::from_millis(20),
};

/// Idle waits of half a minute, for tests that show a wait being cut short.
const SLOW: Intervals = Intervals {
    busy: Duration::from_millis(10),
    idle: Duration::from_secs(30),
    page_open: Duration::from_secs(10),
    first: Duration::from_millis(10),
};

/// Waits up to 10 s for `done`.
fn eventually(mut done: impl FnMut() -> bool) -> bool {
    let until = Instant::now() + Duration::from_secs(10);
    while Instant::now() < until {
        if done() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    false
}

fn count(calls: &Calls) -> usize {
    lock(calls).len()
}

/// The gap between the last two calls.
fn last_gap(calls: &Calls) -> Option<Duration> {
    let calls = lock(calls);
    let n = calls.len();
    (n >= 2).then(|| calls[n - 1].0.duration_since(calls[n - 2].0))
}

fn spawn_script(
    script: Vec<Result<Reading, String>>,
    busy: impl Fn() -> bool + Send + 'static,
    intervals: Intervals,
) -> (Arc<Machine>, Calls) {
    let played = calls();
    let machine = Machine::spawn(
        {
            let played = played.clone();
            move || Ok(Script::new(script, &played))
        },
        busy,
        intervals,
    );
    (machine, played)
}

// --- the waits -----------------------------------------------------------------

#[test]
fn the_short_wait_runs_while_busy_or_the_page_is_open() {
    let i = Intervals::default();
    assert_eq!(
        (i.busy, i.idle, i.page_open, i.first),
        (BUSY_INTERVAL, IDLE_INTERVAL, PAGE_OPEN, FIRST_WINDOW)
    );
    assert_eq!(BUSY_INTERVAL, Duration::from_secs(2));
    assert_eq!(IDLE_INTERVAL, Duration::from_secs(30));
    let s = Duration::from_secs;
    // Idle and nobody looking: the long wait.
    assert_eq!(i.pick(false, None), IDLE_INTERVAL);
    // Busy, whether or not anybody looks.
    assert_eq!(i.pick(true, None), BUSY_INTERVAL);
    assert_eq!(i.pick(true, Some(s(600))), BUSY_INTERVAL);
    // A dashboard request within the last 10 s: the page is open.
    assert_eq!(i.pick(false, Some(Duration::ZERO)), BUSY_INTERVAL);
    assert_eq!(i.pick(false, Some(s(3))), BUSY_INTERVAL);
    assert_eq!(i.pick(false, Some(PAGE_OPEN)), BUSY_INTERVAL);
    assert_eq!(
        i.pick(false, Some(PAGE_OPEN + Duration::from_millis(1))),
        IDLE_INTERVAL
    );
    assert_eq!(i.pick(false, Some(s(3600))), IDLE_INTERVAL);
}

#[test]
fn a_dashboard_request_stamps_the_page_open_and_wakes_the_sampler_only_when_it_was_closed()
 {
    let machine = Machine::new(OFF, Duration::from_millis(100));
    assert_eq!(machine.since_touch(), None, "never touched");
    let woken = |m: &Machine| std::mem::take(&mut *lock(&m.woken));
    assert!(!woken(&machine));
    machine.touch();
    assert!(woken(&machine), "the page was closed");
    let age = machine.since_touch().expect("touched");
    assert!(age < Duration::from_secs(1), "{age:?}");
    // A page that keeps asking costs the sampler nothing.
    machine.touch();
    machine.touch();
    assert!(!woken(&machine), "the page was open");
    std::thread::sleep(Duration::from_millis(130));
    assert!(machine.since_touch().is_some_and(|a| a >= Duration::from_millis(100)));
    machine.touch();
    assert!(woken(&machine), "it had closed in between");
}

#[test]
fn a_window_is_the_monotonic_time_since_the_previous_call_placed_by_the_wall_clock() {
    let two = Duration::from_secs(2);
    let ms = Duration::from_millis;
    assert_eq!(window_of(ms(2_000), 12_000, two), (10_000, 12_000));
    assert_eq!(window_of(ms(2_013), 12_013, two), (10_000, 12_013));
    // A stall (the machine asleep between calls) is credited at most twice
    // the wait and a second, ending at the call's end.
    assert_eq!(window_of(Duration::from_secs(600), 600_000, two), (595_000, 600_000));
    // The wall clock only places it: a jump (the clock stepped by an hour
    // either way) moves the window and never changes its length.
    let (a, b) = window_of(ms(2_000), 1_000_000, two);
    let (c, d) = window_of(ms(2_000), 1_000_000 + 3_600_000, two);
    let (e, f) = window_of(ms(2_000), 1_000_000 - 500_000, two);
    assert_eq!((b - a, d - c, f - e), (2_000, 2_000, 2_000));
    // A clock near its origin does not underflow.
    assert_eq!(window_of(two, 500, two), (0, 500));
}

// --- the readings --------------------------------------------------------------

#[test]
fn a_system_power_that_is_zero_or_below_the_parts_is_unknown() {
    let with = |sys_w: Option<f64>| Reading {
        cpu_w: 8.0,
        gpu_w: 40.0,
        ane_w: 0.5,
        sys_w,
        ..reading(0.0)
    };
    // The SMC's key failed: macmon reports 0 with components above it.
    assert_eq!(with(Some(0.0)).normalised().sys_w, None);
    // Below the sum cannot be a real reading either.
    assert_eq!(with(Some(48.4)).normalised().sys_w, None);
    // At the sum (macmon's floor) or above, it is known.
    assert_eq!(with(Some(48.5)).normalised().sys_w, Some(48.5));
    assert_eq!(with(Some(61.0)).normalised().sys_w, Some(61.0));
    assert_eq!(with(None).normalised().sys_w, None);
    // An idle machine whose parts are all 0 and whose system power is not.
    let idle = Reading { cpu_w: 0.0, gpu_w: 0.0, ane_w: 0.0, ..with(Some(9.5)) };
    assert_eq!(idle.normalised().sys_w, Some(9.5));
    // Unknown is not implausible: the reading is still good.
    assert_eq!(with(Some(0.0)).normalised().check(), Ok(()));
}

#[test]
fn the_gpu_activity_is_clamped_and_what_cannot_be_right_is_refused() {
    let r = Reading { gpu_active: 1.0004, ..reading(10.0) };
    assert_eq!(r.normalised().gpu_active, 1.0);
    let r = Reading { gpu_active: -0.0004, ..reading(10.0) };
    assert_eq!(r.normalised().gpu_active, 0.0);
    assert_eq!(reading(10.0).normalised().check(), Ok(()));
    // The edges of plausible pass.
    let edge = Reading {
        cpu_w: 1000.0,
        sys_w: Some(1000.0),
        gpu_freq_mhz: 10_000.0,
        cpu_temp_c: Some(-20.0),
        gpu_temp_c: Some(150.0),
        ..reading(0.0)
    };
    assert_eq!(edge.normalised().check(), Ok(()));
    for (bad, what) in [
        (Reading { sys_w: Some(f64::NAN), ..reading(1.0) }, "system power"),
        (Reading { sys_w: Some(1000.5), ..reading(1.0) }, "system power"),
        (Reading { cpu_w: 1000.5, ..reading(1.0) }, "cpu power"),
        (Reading { gpu_w: 5000.0, ..reading(1.0) }, "gpu power"),
        (Reading { ane_w: 1001.0, ..reading(1.0) }, "ane power"),
        (Reading { gpu_w: -0.5, ..reading(1.0) }, "gpu power"),
        (Reading { gpu_freq_mhz: 10_001.0, ..reading(1.0) }, "gpu frequency"),
        (Reading { gpu_freq_mhz: f64::INFINITY, ..reading(1.0) }, "gpu frequency"),
        (Reading { gpu_freq_mhz: -1.0, ..reading(1.0) }, "gpu frequency"),
        (Reading { cpu_temp_c: Some(150.5), ..reading(1.0) }, "cpu temperature"),
        (Reading { gpu_temp_c: Some(-20.5), ..reading(1.0) }, "gpu temperature"),
        (Reading { cpu_temp_c: Some(f64::NAN), ..reading(1.0) }, "cpu temperature"),
        (Reading { gpu_active: f64::NAN, ..reading(1.0) }, "gpu activity"),
    ] {
        let error = bad.normalised().check().expect_err(what);
        assert!(error.starts_with(&format!("an implausible {what}")), "{error}");
    }
}

// --- freshness -----------------------------------------------------------------

#[test]
fn a_reading_is_shown_for_three_of_its_windows_then_null() {
    let machine = Machine::running();
    let s = sample(0, 2_000, 12.0);
    machine.publish(BUSY_INTERVAL, s);
    let now = Instant::now();
    assert_eq!(machine.latest(now), Some(s));
    assert_eq!(machine.latest(now + Duration::from_millis(5_900)), Some(s));
    assert_eq!(machine.latest(now + Duration::from_millis(6_500)), None);
    // A 30 s window's reading stays for 90 s.
    machine.publish(IDLE_INTERVAL, s);
    let now = Instant::now();
    assert_eq!(machine.latest(now + Duration::from_secs(89)), Some(s));
    assert_eq!(machine.latest(now + Duration::from_secs(91)), None);
    // However short the window, a reading lives for three seconds.
    machine.publish(Duration::from_millis(10), s);
    let now = Instant::now();
    assert_eq!(machine.latest(now + Duration::from_millis(2_900)), Some(s));
    assert_eq!(machine.latest(now + Duration::from_millis(3_100)), None);
}

#[test]
fn the_recorders_queue_keeps_the_newest_readings_and_drains_once() {
    let machine = Machine::running();
    for i in 0..UNRECORDED as u64 + 5 {
        machine.publish(BUSY_INTERVAL, sample(i * 1000, i * 1000 + 1000, 10.0));
    }
    let drained = machine.drain();
    assert_eq!(drained.len(), UNRECORDED);
    assert_eq!(drained[0].start_ms, 5_000, "the oldest five were dropped");
    assert!(machine.drain().is_empty(), "taken once");
    assert!(machine.latest(Instant::now()).is_some(), "the latest stays");
}

// --- the thread ----------------------------------------------------------------

#[test]
fn the_first_call_has_a_window_and_the_rest_wait_in_lily_and_ask_for_none() {
    let (machine, played) = spawn_script(vec![], || false, FAST);
    assert_eq!(machine.state(), State::Running);
    let mut taken = Vec::new();
    assert!(eventually(|| {
        taken.extend(machine.drain());
        taken.len() >= 3
    }));
    {
        let calls = lock(&played);
        assert_eq!(calls[0].1, FAST.first, "a baseline needs a real window");
        assert!(calls[1..].iter().all(|c| c.1 == Duration::ZERO));
        // Idle: each call at least `idle` after the one before.
        for pair in calls.windows(2) {
            assert!(pair[1].0 - pair[0].0 >= FAST.idle, "{:?}", pair[1].0 - pair[0].0);
        }
    }
    assert!(
        machine.latest(Instant::now()).is_some_and(|s| s.reading.sys_w == Some(10.0))
    );
    // A window is the time since the call before: the first its own, the
    // rest at least the idle wait.
    assert!(taken[0].end_ms - taken[0].start_ms >= FAST.first.as_millis() as u64);
    for s in &taken[1..] {
        assert!(s.end_ms - s.start_ms >= FAST.idle.as_millis() as u64, "{s:?}");
    }
    machine.finish(Duration::from_secs(5));
    assert!(machine.finished());
    assert_eq!(machine.state(), State::Running, "stopped, not failed");
    assert_eq!(machine.error(), None);
}

#[test]
fn the_wait_follows_the_engine_and_the_page() {
    let busy = Arc::new(AtomicBool::new(false));
    let (machine, played) = spawn_script(
        vec![],
        {
            let busy = busy.clone();
            move || busy.load(Ordering::Relaxed)
        },
        FAST,
    );
    // Idle, nobody looking: calls at least `idle` apart.
    assert!(eventually(|| last_gap(&played).is_some_and(|g| g >= FAST.idle)));
    // The engine is busy: calls close together.
    busy.store(true, Ordering::Relaxed);
    assert!(eventually(|| last_gap(&played).is_some_and(|g| g < FAST.idle / 2)));
    // Idle again, the page asks: close together while it keeps asking.
    busy.store(false, Ordering::Relaxed);
    assert!(eventually(|| last_gap(&played).is_some_and(|g| g >= FAST.idle)));
    assert!(eventually(|| {
        machine.touch();
        last_gap(&played).is_some_and(|g| g < FAST.idle / 2)
    }));
    // The page left: spaced again once `page_open` has passed.
    assert!(eventually(|| last_gap(&played).is_some_and(|g| g >= FAST.idle)));
    machine.finish(Duration::from_secs(5));
}

#[test]
fn a_page_that_opens_in_the_middle_of_an_idle_wait_wakes_it_at_once() {
    let (machine, played) = spawn_script(vec![], || false, SLOW);
    assert!(eventually(|| count(&played) == 1), "the first call");
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(count(&played), 1, "idle: the next one is half a minute away");
    let asked = Instant::now();
    machine.touch();
    assert!(eventually(|| count(&played) == 2));
    assert!(asked.elapsed() < Duration::from_secs(2), "{:?}", asked.elapsed());
    // Its window is the whole time since the first, measured by lily.
    let mut taken = Vec::new();
    assert!(
        eventually(|| {
            taken.extend(machine.drain());
            taken.len() >= 2
        }),
        "the reading from the woken call is published"
    );
    let woken = taken[1];
    assert!(woken.end_ms >= woken.start_ms);
    machine.finish(Duration::from_secs(5));
}

/// A `busy` predicate that reads `flag` and counts how often the sampler
/// looked.
fn counted(
    flag: &Arc<AtomicBool>,
    looks: &Arc<AtomicU64>,
) -> impl Fn() -> bool + Send + 'static {
    let (flag, looks) = (flag.clone(), looks.clone());
    move || {
        looks.fetch_add(1, Ordering::Relaxed);
        flag.load(Ordering::Relaxed)
    }
}

#[test]
fn an_idle_sampler_wakes_only_when_a_reading_is_due() {
    let (busy, looks) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)));
    let intervals = Intervals { idle: Duration::from_millis(150), ..FAST };
    let (machine, played) = spawn_script(vec![], counted(&busy, &looks), intervals);
    std::thread::sleep(Duration::from_millis(800));
    machine.finish(Duration::from_secs(5));
    let readings = count(&played);
    assert!((3..=7).contains(&readings), "{readings} readings in 800 ms");
    // Each wait looks twice: right after the call before it, to pick the
    // window, and once when it wakes, due. Nothing polls in between (every
    // 5 ms would be some 30 looks a reading).
    let looks = looks.load(Ordering::Relaxed) as usize;
    assert!(looks <= 2 * readings, "{looks} looks for {readings} readings");
    assert!(!machine.short_cadence(), "an ended sampler has no cadence");
}

#[test]
fn an_opened_request_wakes_an_idle_sampler_at_once() {
    use crate::serve::dashboard::Board;
    let board = Arc::new(Board::default());
    let looks = Arc::new(AtomicU64::new(0));
    let played = calls();
    let machine = Machine::spawn(
        {
            let played = played.clone();
            move || Ok(Script::new(vec![], &played))
        },
        {
            let (board, looks) = (Arc::downgrade(&board), looks.clone());
            move || {
                looks.fetch_add(1, Ordering::Relaxed);
                board.upgrade().is_some_and(|b| b.busy())
            }
        },
        SLOW,
    );
    board.attach_machine(machine.clone());
    assert!(eventually(|| count(&played) == 1), "the first call");
    assert!(eventually(|| machine.listening.load(Ordering::Relaxed)), "waiting");
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(count(&played), 1, "idle: the next one is half a minute away");
    assert_eq!(looks.load(Ordering::Relaxed), 1, "one look, then asleep");
    assert!(!machine.short_cadence());
    let asked = Instant::now();
    board.opened();
    assert!(eventually(|| count(&played) >= 4), "readings at the busy cadence");
    assert!(asked.elapsed() < Duration::from_secs(2), "{:?}", asked.elapsed());
    assert!(machine.short_cadence());
    machine.finish(Duration::from_secs(5));
}

#[test]
fn a_reload_wakes_an_idle_sampler_at_once() {
    use crate::serve::dashboard::Board;
    let board = Arc::new(Board::default());
    let (loading, looks) =
        (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)));
    let (machine, played) = spawn_script(vec![], counted(&loading, &looks), SLOW);
    board.attach_machine(machine.clone());
    assert!(eventually(|| count(&played) == 1));
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(count(&played), 1);
    // What the engine does: the state first, then the nudge.
    let asked = Instant::now();
    loading.store(true, Ordering::Relaxed);
    board.working();
    assert!(eventually(|| count(&played) >= 3), "calls at the busy cadence");
    assert!(asked.elapsed() < Duration::from_secs(2), "{:?}", asked.elapsed());
    machine.finish(Duration::from_secs(5));
}

#[test]
fn a_nudge_without_a_waiting_sampler_wakes_nothing() {
    let machine = Machine::new(OFF, PAGE_OPEN);
    machine.nudge();
    assert!(!*lock(&machine.woken));
    // Waiting out an idle window: the first nudge wakes it, the next finds
    // the flag already lowered.
    machine.listening.store(true, Ordering::Relaxed);
    machine.nudge();
    assert!(std::mem::take(&mut *lock(&machine.woken)));
    machine.nudge();
    assert!(!*lock(&machine.woken));
}

#[test]
fn stop_ends_a_long_wait_within_a_bound_and_finish_joins_the_thread() {
    let (machine, played) = spawn_script(vec![], || false, SLOW);
    assert!(eventually(|| count(&played) == 1));
    let asked = Instant::now();
    machine.stop();
    assert!(eventually(|| machine.finished()));
    assert!(asked.elapsed() < Duration::from_secs(1), "{:?}", asked.elapsed());
    machine.finish(Duration::from_secs(1));
    assert!(lock(&machine.thread).is_none(), "joined");
    // `finish` is the shutdown's call: it does the stop itself, bounded.
    let (machine, played) = spawn_script(vec![], || false, SLOW);
    assert!(eventually(|| count(&played) == 1));
    let asked = Instant::now();
    machine.finish(Duration::from_secs(1));
    assert!(machine.finished());
    assert!(asked.elapsed() < Duration::from_secs(1), "{:?}", asked.elapsed());
    // And it never waits past its cap for a thread inside a call.
    struct Stuck;
    impl Source for Stuck {
        fn read(&mut self, _: Duration) -> Result<Reading, String> {
            std::thread::sleep(Duration::from_secs(5));
            Err("late".into())
        }
    }
    let machine = Machine::spawn(|| Ok(Stuck), || false, FAST);
    std::thread::sleep(Duration::from_millis(50));
    let asked = Instant::now();
    machine.finish(Duration::from_millis(200));
    assert!(asked.elapsed() < Duration::from_secs(2), "{:?}", asked.elapsed());
    assert!(!machine.finished(), "still inside its call");
}

#[test]
fn a_source_that_returns_at_once_cannot_spin_the_thread() {
    let played = calls();
    let machine = Machine::spawn(
        {
            let played = played.clone();
            move || {
                let mut source = Script::new(vec![], &played);
                source.honours = false;
                Ok(source)
            }
        },
        || true,
        FAST,
    );
    std::thread::sleep(Duration::from_millis(300));
    machine.stop();
    // The busy wait is 10 ms: at most about 30 calls in 300 ms, not
    // thousands.
    assert!(count(&played) <= 35, "{} calls", count(&played));
    assert!(eventually(|| machine.finished()));
    // The first call asked for a window and came back at once: not trusted,
    // skipped; the zero-window calls after it are good.
    assert_eq!(machine.skipped(), 1);
    assert_eq!(machine.state(), State::Running);
    assert!(!machine.drain().is_empty());
}

// --- failures ------------------------------------------------------------------

#[test]
fn a_sampler_that_cannot_be_created_fails_once_and_the_thread_ends() {
    let machine = Machine::spawn(
        || -> Result<Script, String> { Err("Failed to create subscription".into()) },
        || false,
        FAST,
    );
    assert!(eventually(|| machine.finished()));
    assert_eq!(machine.state(), State::Failed);
    let error = machine.error().expect("the cause is kept");
    assert_eq!(error, "creating the sampler failed: Failed to create subscription");
    assert_eq!(machine.latest(Instant::now()), None);
    assert!(machine.drain().is_empty());
    // The failure is recorded once: a second report changes nothing.
    machine.fail("again".into());
    assert_eq!(machine.state(), State::Failed);
}

#[test]
fn one_bad_reading_is_skipped_and_a_good_one_resets_the_count() {
    // Four failures in a row, a good reading, four more, a good one: never
    // five in a row.
    let bad = || Err("IOReport hiccup".to_owned());
    let good = |w| Ok(reading(w));
    let mut script = vec![good(11.0), bad(), bad(), bad(), bad(), good(12.0)];
    script.extend([bad(), bad(), bad(), bad(), good(13.0)]);
    let (machine, _) = spawn_script(script, || true, FAST);
    let mut taken = Vec::new();
    assert!(eventually(|| {
        taken.extend(machine.drain());
        taken.len() >= 6
    }));
    assert_eq!(machine.state(), State::Running);
    assert_eq!(machine.error(), None);
    assert_eq!(machine.skipped(), 8);
    let watts: Vec<_> = taken.iter().take(3).map(|s| s.reading.sys_w).collect();
    assert_eq!(watts, [Some(11.0), Some(12.0), Some(13.0)]);
    // A reading after a skip covers the time since the call that was skipped,
    // not the time before it.
    assert!(taken[1].end_ms - taken[1].start_ms < 1_000);
    machine.finish(Duration::from_secs(5));
}

#[test]
fn five_failures_in_a_row_end_the_sampler_with_the_last_error() {
    let mut script = vec![Ok(reading(11.0)), Ok(reading(12.0))];
    script.extend((1..=5).map(|i| Err(format!("boom {i}"))));
    let (machine, played) = spawn_script(script, || true, FAST);
    assert!(eventually(|| machine.finished()));
    assert_eq!(machine.state(), State::Failed);
    assert_eq!(
        machine.error().as_deref(),
        Some("sampling failed 5 times in a row, the last: sampling failed: boom 5")
    );
    assert_eq!(machine.skipped(), 5);
    assert_eq!(count(&played), 7, "no call after the failure");
    // The page shows nothing from a sampler that stopped; the recorder still
    // gets the two windows that were measured.
    assert_eq!(machine.latest(Instant::now()), None);
    let taken = machine.drain();
    assert_eq!(
        taken.iter().map(|s| s.reading.sys_w).collect::<Vec<_>>(),
        [Some(11.0), Some(12.0)]
    );
}

#[test]
fn a_panic_ends_the_sampler_at_once() {
    let played = calls();
    let machine = Machine::spawn(
        {
            let played = played.clone();
            move || {
                let mut source = Script::new(vec![], &played);
                source.panics = true;
                Ok(source)
            }
        },
        || false,
        FAST,
    );
    assert!(eventually(|| machine.finished()));
    assert_eq!(machine.state(), State::Failed);
    assert_eq!(
        machine.error().as_deref(),
        Some("sampling panicked: IOReport exploded")
    );
    assert_eq!(machine.skipped(), 0, "not a skip");
    assert_eq!(count(&played), 1);
}

#[test]
fn a_panic_while_creating_the_sampler_is_caught_too() {
    let machine = Machine::spawn(
        || -> Result<Script, String> { panic!("no IOReport") },
        || false,
        FAST,
    );
    assert!(eventually(|| machine.finished()));
    assert_eq!(machine.state(), State::Failed);
    assert_eq!(
        machine.error().as_deref(),
        Some("creating the sampler panicked: no IOReport")
    );
}

#[test]
fn implausible_readings_are_skipped_and_five_in_a_row_end_the_sampler() {
    for (bad, what) in [
        (Reading { sys_w: Some(f64::NAN), ..reading(1.0) }, "system power"),
        (Reading { sys_w: Some(1000.5), ..reading(1.0) }, "system power"),
        (Reading { cpu_w: 2000.0, ..reading(1.0) }, "cpu power"),
        (Reading { gpu_w: -0.5, ..reading(1.0) }, "gpu power"),
        (Reading { gpu_freq_mhz: 10_001.0, ..reading(1.0) }, "gpu frequency"),
        (Reading { cpu_temp_c: Some(151.0), ..reading(1.0) }, "cpu temperature"),
        (Reading { gpu_temp_c: Some(-21.0), ..reading(1.0) }, "gpu temperature"),
    ] {
        let mut script = vec![Ok(reading(11.0)), Ok(bad), Ok(reading(12.0))];
        script.extend(std::iter::repeat_n(Ok(bad), 5));
        let (machine, _) = spawn_script(script, || true, FAST);
        assert!(eventually(|| machine.finished()));
        assert_eq!(machine.state(), State::Failed, "{bad:?}");
        let error = machine.error().expect("error");
        assert!(
            error.starts_with("sampling failed 5 times in a row, the last: sampling returned an implausible")
                && error.contains(what),
            "{error}"
        );
        // The skip between two good readings did not end it, and the good
        // ones were kept.
        assert_eq!(machine.skipped(), 6);
        let watts: Vec<_> = machine.drain().iter().map(|s| s.reading.sys_w).collect();
        assert_eq!(watts, [Some(11.0), Some(12.0)], "{what}");
    }
}

#[test]
fn an_unknown_system_power_and_a_wild_activity_are_published_not_skipped() {
    let unknown = Reading { cpu_w: 8.0, gpu_w: 40.0, sys_w: Some(0.0), ..reading(0.0) };
    let wild = Reading { gpu_active: 1.3, ..reading(10.0) };
    let (machine, _) = spawn_script(vec![Ok(unknown), Ok(wild)], || true, FAST);
    let mut taken = Vec::new();
    assert!(eventually(|| {
        taken.extend(machine.drain());
        taken.len() >= 2
    }));
    assert_eq!(machine.skipped(), 0);
    assert_eq!(taken[0].reading.sys_w, None, "unknown, not 0 W");
    assert_eq!(taken[0].reading.gpu_w, 40.0, "the parts are kept");
    assert_eq!(taken[1].reading.gpu_active, 1.0, "clamped before it is stored");
    machine.finish(Duration::from_secs(5));
}

// --- lily's own share -------------------------------------------------------

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn lilys_gpu_share_is_its_busy_time_over_the_gpus_active_time() {
    let second = Duration::from_secs(1);
    // The GPU active half the second at 40 W; lily busy a quarter second: half
    // of the GPU's work, half its power. 2 J of CPU energy over 2 s is 1 W.
    let p = attribute(40.0, 0.5, second, 0, 250_000_000).expect("figure");
    assert!(close(p.gpu_w, 20.0), "{p:?}");
    let p = attribute(40.0, 0.5, 2 * second, 2_000_000_000, 0).expect("figure");
    assert!(close(p.cpu_w, 1.0) && p.gpu_w == 0.0, "lily idle: none of the GPU");
    assert!(close(p.w(), 1.0));
    // lily busy for longer than IOReport says the GPU was active (its spans
    // are wall time, stalls included): the whole GPU, never more.
    let p = attribute(40.0, 0.25, second, 0, 900_000_000).expect("figure");
    assert!(close(p.gpu_w, 40.0), "{p:?}");
}

#[test]
fn an_idle_gpu_by_ioreports_account_is_all_lilys_if_lily_was_busy() {
    let second = Duration::from_secs(1);
    let p = attribute(0.3, 0.0, second, 0, 1_000).expect("figure");
    assert!(close(p.gpu_w, 0.3));
    let p = attribute(0.3, 0.0, second, 0, 0).expect("figure");
    assert_eq!(p.gpu_w, 0.0);
    // Active for less than a microsecond counts as idle too.
    let p = attribute(0.3, 1e-7, second, 0, 0).expect("figure");
    assert_eq!(p.gpu_w, 0.0);
}

#[test]
fn no_figure_for_a_zero_window_or_one_that_cannot_be_right() {
    assert_eq!(attribute(40.0, 0.5, Duration::ZERO, 1_000, 1_000), None);
    // 10 kJ in a second: no CPU draws 10 kW.
    assert_eq!(attribute(40.0, 0.5, Duration::from_secs(1), 10u64.pow(13), 0), None);
    assert_eq!(attribute(f64::NAN, 0.5, Duration::from_secs(1), 0, 1_000), None);
}

#[test]
fn own_counters_subtract_and_refuse_to_go_back() {
    let at = |energy_nj, gpu_busy_ns| OwnCounters { energy_nj, gpu_busy_ns };
    assert_eq!(at(Some(30), 7).since(at(Some(10), 2)), Some((20, 5)));
    assert_eq!(at(Some(5), 7).since(at(Some(10), 2)), None, "energy went back");
    assert_eq!(at(Some(30), 1).since(at(Some(10), 2)), None, "busy went back");
    assert_eq!(at(None, 7).since(at(Some(10), 2)), None, "the call failed");
    assert_eq!(at(Some(30), 7).since(at(None, 2)), None);
}

#[test]
fn lilys_counters_are_read_after_every_call_and_a_bad_figure_is_dropped_not_the_reading()
 {
    let at = |energy, busy| OwnCounters { energy_nj: Some(energy), gpu_busy_ns: busy };
    let counters = Arc::new(Mutex::new(VecDeque::from([
        at(0, 0),                   // the baseline, before the first call
        at(100_000_000, 2_000_000), // after the first: a figure
        at(50_000_000, 3_000_000),  // energy went back: none
        at(60_000_000, 9_000_000),  // after the failed call
        at(70_000_000, 9_000_000),  // no GPU time since the failed call
    ])));
    let reads = Arc::new(AtomicU64::new(0));
    let played = calls();
    let script = vec![
        Ok(reading(11.0)),
        Ok(reading(12.0)),
        Err("IOReport hiccup".to_owned()),
        Ok(reading(13.0)),
    ];
    let machine = Machine::spawn_counting(
        {
            let played = played.clone();
            move || Ok(Script::new(script, &played))
        },
        {
            let (counters, reads) = (counters.clone(), reads.clone());
            move || {
                reads.fetch_add(1, Ordering::Relaxed);
                lock(&counters).pop_front().unwrap_or_default()
            }
        },
        || true,
        FAST,
    );
    let mut taken = Vec::new();
    assert!(eventually(|| {
        taken.extend(machine.drain());
        taken.len() >= 3
    }));
    machine.finish(Duration::from_secs(5));
    assert_eq!(machine.skipped(), 1, "only the failed call was skipped");
    let first = taken[0].reading.lily.expect("a figure from the baseline on");
    assert!(first.cpu_w > 0.0 && first.gpu_w > 0.0 && first.gpu_w <= 0.5, "{first:?}");
    assert_eq!(taken[1].reading.lily, None, "a counter that went back");
    assert_eq!(taken[1].reading.sys_w, Some(12.0), "the reading itself stays");
    // Measured from the counters read after the failed call: no GPU time.
    let third = taken[2].reading.lily.expect("a figure");
    assert_eq!(third.gpu_w, 0.0, "{third:?}");
    assert!(third.cpu_w > 0.0);
    // Once before the first call and once after each.
    assert_eq!(reads.load(Ordering::Relaxed) as usize, count(&played) + 1);
    // Counters without an energy (the call failed): no figure, readings go on.
    assert!(taken.iter().skip(3).all(|s| s.reading.lily.is_none()));
}

#[test]
fn skips_are_logged_the_first_time_and_every_hundredth() {
    let mut failures = Failures::default();
    let mut logged = Vec::new();
    for n in 1..=250u64 {
        failures.bad();
        failures.good();
        if failures.log_this() {
            logged.push(n);
        }
    }
    assert_eq!(logged, [1, 100, 200]);
    assert!(!Failures::default().log_this());
}

#[test]
fn off_has_no_thread_and_no_readings() {
    let machine = Machine::off();
    assert_eq!(machine.state(), State::Off);
    assert!(machine.finished());
    assert_eq!(machine.latest(Instant::now()), None);
    assert_eq!(machine.error(), None);
    machine.finish(Duration::from_millis(10));
}

#[test]
fn machine_stats_takes_on_or_off() {
    assert_eq!("on".parse::<MachineStats>().unwrap(), MachineStats::On);
    assert_eq!("off".parse::<MachineStats>().unwrap(), MachineStats::Off);
    assert_eq!(MachineStats::default(), MachineStats::On);
    let error = "auto".parse::<MachineStats>().unwrap_err().to_string();
    assert!(error.contains("on or off"), "{error}");
}

// --- the real hardware ----------------------------------------------------------

/// Reads the real machine through `macmon`: IOReport and the SMC are blocked
/// inside a sandbox, so this runs outside it, on an Apple Silicon Mac:
///
/// `cargo test -j 4 --lib serve::machine::tests::real_macmon -- --ignored --nocapture --test-threads=1`
///
/// It takes about 8 s: 0.7 s to open the sampler, a 1 s window, 20 warm-up
/// calls and 300 calls of 10 ms windows with lily's footprint measured around
/// them, then the thread's own path (one first window).
///
/// `macmon` 0.9.0 keeps IOHID and SMC handles for the life of the sampler
/// (one per thread, so no leak that grows) and takes a `mach_host_self` send
/// right per call; both would show here as a footprint that climbs with the
/// calls. The bound is generous (8 MiB over 300 calls) so only a leak that
/// matters fails; the growth is printed either way. Not vendored.
#[test]
#[ignore = "needs IOReport and the SMC: run outside the sandbox on Apple Silicon"]
fn real_macmon_reads_plausible_figures_and_does_not_leak() {
    let mut source = Macmon::new().expect("IOReport opens outside a sandbox");
    let first = source.read(Duration::from_millis(1000)).expect("the first window");
    let r = first.normalised();
    r.check().expect("finite and in range");
    assert!(r.cpu_w >= 0.0 && r.cpu_w < 300.0, "cpu power {} W", r.cpu_w);
    assert!(r.gpu_w >= 0.0 && r.gpu_w < 300.0, "gpu power {} W", r.gpu_w);
    let sys = r.sys_w.expect("the SMC reports the system's power on this machine");
    assert!(sys > 1.0 && sys < 500.0, "system power {sys} W");
    assert!(sys >= r.cpu_w + r.gpu_w + r.ane_w - 1e-3, "the system is the sum or more");
    assert!((0.0..=1.0).contains(&r.gpu_active), "gpu active {}", r.gpu_active);
    assert!(r.gpu_freq_mhz < 5000.0, "gpu frequency {} MHz", r.gpu_freq_mhz);
    for (name, t) in [("cpu", r.cpu_temp_c), ("gpu", r.gpu_temp_c)] {
        if let Some(t) = t {
            assert!((10.0..=120.0).contains(&t), "{name} temperature {t} C");
        }
    }
    eprintln!("first window: {r:?}");
    // A zero window reads since the previous call, as the production loop does.
    std::thread::sleep(Duration::from_millis(500));
    let since = source.read(Duration::ZERO).expect("a zero window").normalised();
    since.check().expect("finite and in range");
    eprintln!("zero window after 0.5 s: {since:?}");

    // Footprint over many calls.
    let footprint = || crate::stats::task_memory().expect("task_info").phys_footprint;
    for _ in 0..20 {
        source.read(Duration::from_millis(10)).expect("warm-up");
    }
    let before = footprint();
    for _ in 0..300 {
        source.read(Duration::from_millis(10)).expect("a short window");
    }
    let after = footprint();
    let growth = after as i64 - before as i64;
    eprintln!("footprint over 300 calls: {before} -> {after} ({growth:+} bytes)");
    assert!(growth < 8 << 20, "lily's footprint grew by {growth} bytes over 300 calls");
    drop(source);

    // The same through the thread: one first window, published and queued.
    let machine = Machine::spawn(Macmon::new, || false, Intervals::default());
    assert!(eventually(|| machine.latest(Instant::now()).is_some()));
    assert_eq!(machine.state(), State::Running);
    machine.finish(Duration::from_secs(5));
    assert!(machine.finished(), "the thread ends at the stop");
}
