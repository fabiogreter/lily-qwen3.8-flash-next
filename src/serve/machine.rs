//! The dashboard's machine sampling (docs/dashboard.md § 6): CPU, GPU, ANE
//! and system power, the GPU's activity and frequency and the CPU and GPU
//! temperatures, read from IOReport and the SMC by the `macmon` crate, which
//! needs no root. This is the one module that knows `macmon` ([`Macmon`]);
//! [`Source`] is the whole of what the rest of lily asks of it, so a
//! replacement (our own IOReport bindings) changes nothing outside this file.
//!
//! **A thread of its own.** [`Machine::start`] spawns the sampler thread at
//! the utility quality of service (background work that should not compete
//! with the engine's threads for the performance cores). The thread creates
//! the `macmon` sampler itself (about 0.7 s, so never on the main thread, and
//! never while the server starts). Nothing here runs on the engine's thread
//! or on an HTTP thread (§ 2.3), reads the engine's state beyond one cheap
//! predicate, or wakes the engine (§ 2.2): the sampler only asks "is anything
//! running" through the closure it was given and reads IOReport, which is the
//! machine's, not lily's.
//!
//! **Waiting is lily's, the measuring is `macmon`'s.** `macmon` measures from
//! the end of its previous call to its next one, and sleeps only if the
//! duration asked for has not passed since. So the first call asks for
//! [`FIRST_WINDOW`] (a baseline has to exist, and a window of nothing would be
//! noise); every later one asks for [`Duration::ZERO`] after lily itself has
//! waited, on a condition variable, until the next reading is due. A reading's
//! window is then exactly the time since the previous call, and the wait can be
//! cut short: [`Machine::touch`] wakes it when the page opens, [`Machine::nudge`]
//! when the engine starts work (a request taken on, a reload, a recovery),
//! [`Machine::stop`] when the server shuts down. Nothing waits out a 30 s call,
//! and nothing polls: an idle sampler wakes once per idle window.
//!
//! **Waking on work** costs the engine one fence and one relaxed load per
//! request taken on (and per reload or recovery). The sampler raises a flag
//! before each look at the engine and lowers it when the look picks the busy
//! window; [`Machine::nudge`] notifies the condition variable only when it
//! finds the flag raised, so only the idle-to-busy edge pays for a lock and a
//! notify. The two sequentially consistent fences (one after the sampler
//! raises the flag, one before the nudge reads it) make sure that either the
//! sampler's look sees the work or the nudge sees the flag: work that starts
//! during an idle wait is never left for the next idle reading.
//!
//! **The window** ([`Intervals::pick`]) is chosen while waiting: the next
//! reading is due 2 s after the previous one while the engine is busy or the
//! page is open (a dashboard request within the last 10 s), 30 s after it
//! otherwise, so the 7-day history has idle samples too. A page that opens in
//! the middle of an idle wait gets its first fresh reading at once (the last
//! idle reading shows meanwhile, as it is still fresh), and the 2 s cadence
//! starts from it; so does work that starts then ([`Machine::nudge`]). Work
//! that no request accounts for (tokens drawn outside one) is noticed at the
//! next reading only. Nothing can call faster than the busy window, so a source
//! that returns at once cannot spin the thread; and the first call, the only
//! one with a window to wait out, is not trusted if it comes back in less than
//! half of it ([`Failures`]).
//!
//! **Freshness** ([`Machine::latest`]). The sampler publishes each reading,
//! timestamped, into one slot whose lock is held only to swap it. The status
//! route shows it while it is younger than three windows (6 s after a 2 s
//! window, 90 s after a 30 s one) and shows `null` otherwise: a sampler that
//! stalls does not leave a stale reading on the page.
//!
//! **What the recorder gets.** Every reading also goes to a bounded queue
//! ([`UNRECORDED`] samples, the oldest dropped) that the recorder drains on its
//! own tick ([`Machine::drain`]); the sampler never waits for it. A [`Sample`]
//! carries the window it covers in unix milliseconds, so the recorder spreads
//! it over the minutes it overlaps and integrates energy over covered time
//! only (`serve::recorder`). The window's length is the monotonic time
//! between the two calls ([`window_of`]), anchored at the wall clock's now, so
//! a clock that jumps shifts a window and never stretches it.
//!
//! **System power can be unknown.** `macmon` reads it from the SMC's `PSTR`
//! key and reports 0 when that fails, and it never reports less than the sum
//! of the three components. A reading whose system power is not above that sum
//! (or is 0) has none ([`Reading::sys_w`] is `None`): the status shows
//! `null` for it, and the recorder counts the window for the GPU series but
//! not for system power or energy.
//!
//! **A broken probe never breaks serving** (§ 2.4), within limits. A call
//! that fails, returns a value that is not finite or is out of range
//! ([`Reading::check`]), or comes back too early is skipped and counted; one
//! good reading resets the count. After [`MAX_FAILURES`] failures in a row,
//! or at once for a panic or if the sampler cannot be created (IOReport
//! refuses, as it does inside a sandbox), the error is logged once with its
//! cause, the state becomes [`State::Failed`], the published reading is
//! cleared and the thread ends. Skips are logged on the first and then every
//! 100th. The status then says `machine.sampler: failed` with the power, GPU
//! and temperatures `null`; nothing else is affected.
//!
//! What `catch_unwind` covers, exactly: a Rust panic raised in this module or
//! in `macmon`'s Rust code, on the sampler's thread, is caught and ends the
//! sampler. It does **not** cover a fault that is not a panic. `macmon` calls
//! IOReport and CoreFoundation through `unsafe` FFI and does not check every
//! pointer it gets back: a NULL CoreFoundation object dereferenced there is a
//! `SIGSEGV`, which ends the whole lily process, as does an abort. A call that
//! hangs inside IOReport would leave the sampler thread stuck (the status shows
//! `null` once the reading is stale, the shutdown does not wait for it past its
//! cap). `macmon` 0.9.0 is pinned (`=0.9.0`) so an update is a decision; the
//! protection that would remove the risk is a subprocess, below.
//!
//! TODO(dashboard): run `macmon` in a child process (a probe subcommand
//! writing one JSON line per reading) so a crash or a hang there cannot take
//! the server down; the sampler thread would then read the child's lines and
//! restart it. Owner decision pending (more moving parts, a second process).
//!
//! `--machine-stats off` ([`MachineStats`]) starts no thread: the state is
//! [`State::Off`].

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering, fence};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::Result;

/// The wait between readings while the engine is busy or the page is open.
pub const BUSY_INTERVAL: Duration = Duration::from_secs(2);
/// The wait between readings otherwise.
pub const IDLE_INTERVAL: Duration = Duration::from_secs(30);
/// How long after a dashboard request the page counts as open.
pub const PAGE_OPEN: Duration = Duration::from_secs(10);
/// The first call's window: `macmon` has no baseline yet, and a reading over
/// no time at all would be noise.
pub const FIRST_WINDOW: Duration = Duration::from_secs(1);
/// A wait this close to its end counts as over: a timed wait on the
/// condition variable can return a little early, and waiting out the rest
/// would be a second wake-up for nothing (the window is measured, not
/// assumed).
const DUE_SLACK: Duration = Duration::from_millis(2);
/// A reading is shown while it is younger than this many of its windows.
pub const FRESH_WINDOWS: u32 = 3;
/// The shortest window freshness is measured in, however short the one
/// asked for.
const FRESH_FLOOR: Duration = Duration::from_secs(1);
/// Readings waiting for the recorder, the oldest dropped past this many
/// (about eight minutes at the busy cadence).
pub const UNRECORDED: usize = 256;
/// Failures in a row that end the sampler.
pub const MAX_FAILURES: u32 = 5;
/// A skipped reading is logged the first time and then every this many.
const LOG_SKIPS_EVERY: u64 = 100;
/// The most any power reading can plausibly be, in watts.
const MAX_WATTS: f64 = 1000.0;
/// The highest GPU frequency that can be plausible, in MHz.
const MAX_MHZ: f64 = 10_000.0;
/// The temperatures that can be plausible, in degrees Celsius.
const TEMPERATURES_C: std::ops::RangeInclusive<f64> = -20.0..=150.0;

/// `--machine-stats`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MachineStats {
    /// Sample the machine (default).
    #[default]
    On,
    /// No sampler thread: the dashboard shows no power, GPU or temperatures.
    Off,
}

impl FromStr for MachineStats {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            other => {
                anyhow::bail!("unknown machine-stats mode {other:?}; use on or off")
            }
        }
    }
}

/// One reading of the machine over a window. Power in watts, `gpu_active` a
/// ratio 0..1, the frequency in MHz, temperatures in degrees Celsius (`None`
/// where no sensor reported).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Reading {
    pub cpu_w: f64,
    pub gpu_w: f64,
    pub ane_w: f64,
    /// The whole system's power; `None` where it is not known (see the module
    /// docs).
    pub sys_w: Option<f64>,
    pub gpu_active: f64,
    pub gpu_freq_mhz: f64,
    pub cpu_temp_c: Option<f64>,
    pub gpu_temp_c: Option<f64>,
}

impl Reading {
    /// The reading with what cannot be right put right: the GPU's activity
    /// clamped to 0..1, and a system power that is 0 or below the sum of the
    /// components (the SMC key failed) unknown.
    pub fn normalised(mut self) -> Self {
        if self.gpu_active.is_finite() {
            self.gpu_active = self.gpu_active.clamp(0.0, 1.0);
        }
        let parts = self.cpu_w + self.gpu_w + self.ane_w;
        // `macmon` takes the larger of the SMC's figure and this sum, so a
        // real reading is never below it; a milliwatt of slack for the
        // rounding of the sum.
        // A value that is not a number stays, for the check to refuse.
        self.sys_w =
            self.sys_w.filter(|&w| !w.is_finite() || (w > 0.0 && w >= parts - 1e-3));
        self
    }

    /// Why this reading cannot be trusted, if it cannot: a value that is not
    /// finite, a power or frequency that is negative or beyond anything a
    /// Mac can do ([`MAX_WATTS`], [`MAX_MHZ`]), a temperature outside
    /// [`TEMPERATURES_C`].
    pub fn check(&self) -> Result<(), String> {
        let powers = [
            ("cpu power", Some(self.cpu_w)),
            ("gpu power", Some(self.gpu_w)),
            ("ane power", Some(self.ane_w)),
            ("system power", self.sys_w),
        ];
        for (name, value) in powers {
            if let Some(v) = value
                && !(v.is_finite() && (0.0..=MAX_WATTS).contains(&v))
            {
                return Err(format!("an implausible {name}: {v}"));
            }
        }
        if !(self.gpu_active.is_finite() && (0.0..=1.0).contains(&self.gpu_active)) {
            return Err(format!("an implausible gpu activity: {}", self.gpu_active));
        }
        if !(self.gpu_freq_mhz.is_finite()
            && (0.0..=MAX_MHZ).contains(&self.gpu_freq_mhz))
        {
            return Err(format!("an implausible gpu frequency: {}", self.gpu_freq_mhz));
        }
        for (name, value) in
            [("cpu temperature", self.cpu_temp_c), ("gpu temperature", self.gpu_temp_c)]
        {
            if let Some(v) = value
                && !(v.is_finite() && TEMPERATURES_C.contains(&v))
            {
                return Err(format!("an implausible {name}: {v}"));
            }
        }
        Ok(())
    }
}

/// A reading and the span it covers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// The window, unix milliseconds (`start_ms <= end_ms`).
    pub start_ms: u64,
    pub end_ms: u64,
    pub reading: Reading,
}

/// Where the readings come from. `read(window)` returns what the machine did
/// since the previous call, after waiting until `window` has passed since it
/// (no wait for [`Duration::ZERO`]); the first call has no previous one and
/// measures over `window`. The production source is [`Macmon`]; tests play
/// one. A source is created and used on the sampler thread only, so it need
/// not be `Send`.
pub trait Source {
    fn read(&mut self, window: Duration) -> Result<Reading, String>;
}

/// `macmon`'s sampler as a [`Source`]. Its `get_metrics(duration)` is exactly
/// that contract.
///
/// Known costs of `macmon` 0.9.0, accepted rather than vendored: it creates
/// IOHID and SMC handles that live as long as the sampler (one sampler for
/// the life of the thread, never one per reading), and every `get_metrics`
/// takes a `mach_host_self` send right, which a loop over thousands of calls
/// can leak; the ignored real-hardware test measures lily's footprint over
/// 300 calls so a leak that matters shows.
pub struct Macmon(macmon::Sampler);

impl Macmon {
    /// Opens IOReport and the SMC; about 0.7 s. Fails where IOReport is
    /// blocked (a sandbox).
    pub fn new() -> Result<Self, String> {
        macmon::Sampler::new().map(Self).map_err(|error| error.to_string())
    }
}

impl Source for Macmon {
    fn read(&mut self, window: Duration) -> Result<Reading, String> {
        let ms = window.as_millis().min(u32::MAX as u128) as u32;
        let m = self.0.get_metrics(ms).map_err(|error| error.to_string())?;
        Ok(Reading {
            cpu_w: f64::from(m.cpu_power),
            gpu_w: f64::from(m.gpu_power),
            ane_w: f64::from(m.ane_power),
            sys_w: Some(f64::from(m.sys_power)),
            gpu_active: f64::from(m.gpu_active_ratio),
            gpu_freq_mhz: f64::from(m.gpu_freq_mhz),
            cpu_temp_c: m.temp.cpu_temp_avg.map(f64::from),
            gpu_temp_c: m.temp.gpu_temp_avg.map(f64::from),
        })
    }
}

/// The waits the sampler uses and how long a dashboard request keeps the
/// page open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Intervals {
    pub busy: Duration,
    pub idle: Duration,
    pub page_open: Duration,
    /// The first call's window.
    pub first: Duration,
}

impl Default for Intervals {
    fn default() -> Self {
        Self {
            busy: BUSY_INTERVAL,
            idle: IDLE_INTERVAL,
            page_open: PAGE_OPEN,
            first: FIRST_WINDOW,
        }
    }
}

impl Intervals {
    /// The wait until the next reading: the short one while `busy` (a
    /// request is running) or the last dashboard request was `since_touch`
    /// ago, at most `page_open` (`None`: never), the long one otherwise.
    pub fn pick(&self, busy: bool, since_touch: Option<Duration>) -> Duration {
        let page_open = since_touch.is_some_and(|age| age <= self.page_open);
        if busy || page_open { self.busy } else { self.idle }
    }
}

/// The window of a reading taken at `end_ms` (unix milliseconds) `since` the
/// previous call, by the monotonic clock: that long back from `end_ms`, at
/// most twice `requested` plus a second so a long stall is not counted as
/// covered time. The wall clock only places the window; a jump of it moves
/// the window and never stretches it.
pub fn window_of(since: Duration, end_ms: u64, requested: Duration) -> (u64, u64) {
    let cap = (requested.as_millis() as u64).saturating_mul(2).saturating_add(1000);
    let len = (since.as_millis() as u64).min(cap);
    (end_ms.saturating_sub(len), end_ms)
}

/// Failures in a row and skips so far: what decides between skipping a
/// reading and ending the sampler.
#[derive(Debug, Default)]
struct Failures {
    in_a_row: u32,
    skipped: u64,
}

impl Failures {
    fn good(&mut self) {
        self.in_a_row = 0;
    }

    /// Counts a failure; whether it is the one that ends the sampler.
    fn bad(&mut self) -> bool {
        self.in_a_row += 1;
        self.skipped += 1;
        self.in_a_row >= MAX_FAILURES
    }

    /// Whether this skip is logged.
    fn log_this(&self) -> bool {
        self.skipped > 0
            && (self.skipped == 1 || self.skipped.is_multiple_of(LOG_SKIPS_EVERY))
    }
}

/// Where the sampler is, as the status shows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// `--machine-stats off`.
    Off,
    /// The thread runs (or is creating the sampler: no reading yet).
    Running,
    /// Creating the sampler failed, or it stopped after repeated errors; the
    /// thread has ended.
    Failed,
}

const OFF: u8 = 0;
const RUNNING: u8 = 1;
const FAILED: u8 = 2;

#[derive(Debug, Clone, Copy)]
struct Published {
    at: Instant,
    /// The wait before this reading, which freshness is measured in.
    interval: Duration,
    sample: Sample,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The sampler's shared side: its state, the latest reading, the queue for
/// the recorder and the stamp of the last dashboard request. Every method is
/// cheap and holds its lock only for a swap, a push or a small copy.
#[derive(Debug)]
pub struct Machine {
    state: AtomicU8,
    error: Mutex<Option<String>>,
    latest: Mutex<Option<Published>>,
    unrecorded: Mutex<VecDeque<Sample>>,
    /// Milliseconds since `epoch` of the last dashboard request, plus one
    /// (0: none yet).
    touched: AtomicU64,
    /// How long a request keeps the page open, which decides whether one
    /// wakes the sampler.
    page_open: Duration,
    epoch: Instant,
    stop: AtomicBool,
    /// Set by whatever wakes the sampler's wait early.
    woken: Mutex<bool>,
    wake: Condvar,
    /// The sampler is looking at the engine or waiting out the idle window:
    /// work that starts wakes it ([`Machine::nudge`]).
    listening: AtomicBool,
    /// The sampler's last look picked the busy window (the engine busy or
    /// the page open); the recorder's cadence follows it.
    short: AtomicBool,
    skipped: AtomicU64,
    thread: Mutex<Option<JoinHandle<()>>>,
    /// The thread has ended (or never started).
    finished: AtomicBool,
}

impl Machine {
    fn new(state: u8, page_open: Duration) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU8::new(state),
            error: Mutex::new(None),
            latest: Mutex::new(None),
            unrecorded: Mutex::new(VecDeque::new()),
            touched: AtomicU64::new(0),
            page_open,
            epoch: Instant::now(),
            stop: AtomicBool::new(false),
            woken: Mutex::new(false),
            wake: Condvar::new(),
            listening: AtomicBool::new(false),
            short: AtomicBool::new(false),
            skipped: AtomicU64::new(0),
            thread: Mutex::new(None),
            finished: AtomicBool::new(false),
        })
    }

    /// No sampling (`--machine-stats off`, and tests that need none).
    pub fn off() -> Arc<Self> {
        let machine = Self::new(OFF, PAGE_OPEN);
        machine.finished.store(true, Ordering::Release);
        machine
    }

    /// Starts the production sampler ([`Macmon`], the default
    /// [`Intervals`]); `busy` says whether the engine is working. Returns at
    /// once: the thread creates the sampler.
    pub fn start(busy: impl Fn() -> bool + Send + 'static) -> Arc<Self> {
        Self::spawn(Macmon::new, busy, Intervals::default())
    }

    /// Starts a sampler thread that creates its source with `make` (on the
    /// thread) and reads it as often as `intervals`, `busy` and the
    /// dashboard's requests say.
    pub fn spawn<S, F, B>(make: F, busy: B, intervals: Intervals) -> Arc<Self>
    where
        S: Source + 'static,
        F: FnOnce() -> Result<S, String> + Send + 'static,
        B: Fn() -> bool + Send + 'static,
    {
        let machine = Self::new(RUNNING, intervals.page_open);
        let thread = {
            let machine = machine.clone();
            std::thread::Builder::new()
                .name("lily-machine".into())
                .spawn(move || machine.run(make, busy, intervals))
        };
        match thread {
            Ok(thread) => *lock(&machine.thread) = Some(thread),
            Err(error) => {
                machine.fail(format!("spawning the sampler thread failed: {error}"));
                machine.finished.store(true, Ordering::Release);
            }
        }
        machine
    }

    /// A sampler in the running state with no thread: a test plays it with
    /// [`Machine::publish`].
    #[cfg(test)]
    pub fn running() -> Arc<Self> {
        Self::new(RUNNING, PAGE_OPEN)
    }

    /// The loop of the sampler thread.
    fn run<S, F, B>(&self, make: F, busy: B, intervals: Intervals)
    where
        S: Source,
        F: FnOnce() -> Result<S, String>,
        B: Fn() -> bool,
    {
        let _ended = Ended(self);
        utility_priority();
        let mut source = match guarded("creating the sampler", make) {
            Ok(source) => source,
            Err(fault) => return self.fail(fault.message()),
        };
        eprintln!(
            "machine: sampling power, GPU and temperatures every {} s while the engine is busy or the page is open, every {} s otherwise",
            intervals.busy.as_secs_f64(),
            intervals.idle.as_secs_f64()
        );
        let mut failures = Failures::default();
        // When the previous call returned: `macmon`'s baseline, near enough.
        let mut previous: Option<Instant> = None;
        while !self.stop.load(Ordering::Acquire) {
            let (window, interval) = match previous {
                None => (intervals.first, intervals.first),
                Some(at) => match self.wait_until_due(&busy, &intervals, at) {
                    Some(interval) => (Duration::ZERO, interval),
                    None => break,
                },
            };
            let call_start = Instant::now();
            let result = guarded("sampling", || source.read(window));
            let end = Instant::now();
            let end_ms = super::now_ms();
            let since = end.saturating_duration_since(previous.unwrap_or(call_start));
            previous = Some(end);
            let outcome = match result {
                Err(fault @ Fault::Panic(_)) => {
                    self.fail(fault.message());
                    break;
                }
                Err(Fault::Error(why)) => Err(why),
                Ok(reading) => {
                    let took = end.saturating_duration_since(call_start);
                    if window > Duration::ZERO && took < window / 2 {
                        // A source that does not wait for its window would
                        // read over nothing.
                        Err(format!(
                            "sampling returned after {} ms for a {} ms window",
                            took.as_millis(),
                            window.as_millis()
                        ))
                    } else {
                        let reading = reading.normalised();
                        reading
                            .check()
                            .map(|()| reading)
                            .map_err(|why| format!("sampling returned {why}"))
                    }
                }
            };
            match outcome {
                Ok(reading) => {
                    failures.good();
                    let (start_ms, end_ms) = window_of(since, end_ms, interval);
                    self.publish(interval, Sample { start_ms, end_ms, reading });
                }
                Err(why) => {
                    let last = failures.bad();
                    self.skipped.store(failures.skipped, Ordering::Relaxed);
                    if last {
                        self.fail(format!(
                            "sampling failed {MAX_FAILURES} times in a row, the last: {why}"
                        ));
                        break;
                    }
                    if failures.log_this() {
                        eprintln!(
                            "machine: skipped a reading ({why}); {} skipped so far",
                            failures.skipped
                        );
                    }
                }
            }
        }
        // A source that failed may not drop cleanly either.
        let _ = catch_unwind(AssertUnwindSafe(move || drop(source)));
    }

    /// Waits until the next reading is due, `at` being when the previous call
    /// returned: the busy or the idle interval after it, as the engine and the
    /// page say at each look (on entry, and whenever the wait is woken: the
    /// page opened, the engine started work). `None` when asked to stop.
    fn wait_until_due(
        &self,
        busy: &impl Fn() -> bool,
        intervals: &Intervals,
        at: Instant,
    ) -> Option<Duration> {
        loop {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            // Raised before the look: work that starts after it nudges this
            // wait (the module docs say why the fences).
            self.listening.store(true, Ordering::Relaxed);
            fence(Ordering::SeqCst);
            let interval = intervals.pick(busy(), self.since_touch());
            let short = interval == intervals.busy;
            self.short.store(short, Ordering::Relaxed);
            if short {
                // A short wait need not be cut short by work.
                self.listening.store(false, Ordering::Relaxed);
            }
            let remaining = (at + interval).saturating_duration_since(Instant::now());
            if remaining <= DUE_SLACK {
                return Some(interval);
            }
            let mut woken = lock(&self.woken);
            if !*woken {
                woken = self
                    .wake
                    .wait_timeout(woken, remaining)
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .0;
            }
            *woken = false;
        }
    }

    /// The engine started work (a request taken on, a reload, a recovery):
    /// wakes the sampler if it is waiting out the idle window, so the busy
    /// cadence starts now. A fence and a relaxed load when it is not (busy
    /// already, or no sampler thread); the lock and the notify only on the
    /// idle-to-busy edge.
    pub fn nudge(&self) {
        fence(Ordering::SeqCst);
        if self.listening.load(Ordering::Relaxed)
            && self.listening.swap(false, Ordering::Relaxed)
        {
            self.wake();
        }
    }

    /// Whether the sampler's last look picked the busy window (the engine
    /// busy or the page open). `false` without a running thread.
    pub fn short_cadence(&self) -> bool {
        self.short.load(Ordering::Relaxed)
    }

    fn wake(&self) {
        *lock(&self.woken) = true;
        self.wake.notify_all();
    }

    /// Records the failure: logged once, the reading cleared, the state
    /// [`State::Failed`].
    pub fn fail(&self, why: String) {
        *lock(&self.latest) = None;
        *lock(&self.error) = Some(why.clone());
        if self.state.swap(FAILED, Ordering::AcqRel) != FAILED {
            eprintln!(
                "machine: {why}; the dashboard shows no power, GPU or temperatures (the server serves on; later sampler errors are not logged)"
            );
        }
    }

    /// Publishes `sample`, taken after the wait `interval`, as the latest
    /// reading and queues it for the recorder.
    pub fn publish(&self, interval: Duration, sample: Sample) {
        *lock(&self.latest) = Some(Published { at: Instant::now(), interval, sample });
        let mut queue = lock(&self.unrecorded);
        if queue.len() == UNRECORDED {
            queue.pop_front();
        }
        queue.push_back(sample);
    }

    /// The latest reading, while it is younger than [`FRESH_WINDOWS`] of its
    /// windows at `now`; `None` otherwise, and after a failure.
    pub fn latest(&self, now: Instant) -> Option<Sample> {
        let published = (*lock(&self.latest))?;
        let limit = published.interval.max(FRESH_FLOOR) * FRESH_WINDOWS;
        (now.saturating_duration_since(published.at) <= limit)
            .then_some(published.sample)
    }

    pub fn state(&self) -> State {
        match self.state.load(Ordering::Acquire) {
            OFF => State::Off,
            RUNNING => State::Running,
            _ => State::Failed,
        }
    }

    /// Why the sampler failed.
    pub fn error(&self) -> Option<String> {
        lock(&self.error).clone()
    }

    /// Readings skipped so far (a failed call, an implausible value, a call
    /// back too early); tests only, not part of the dashboard's contract.
    #[cfg(test)]
    pub fn skipped(&self) -> u64 {
        self.skipped.load(Ordering::Relaxed)
    }

    /// The readings the recorder has not taken yet, oldest first.
    pub fn drain(&self) -> Vec<Sample> {
        lock(&self.unrecorded).drain(..).collect()
    }

    /// A dashboard request came in: the page counts as open for
    /// [`Intervals::page_open`] from now. One relaxed swap; the sampler is
    /// woken only when the page was closed, so a page that polls costs it
    /// nothing.
    pub fn touch(&self) {
        let now = self.epoch.elapsed().as_millis() as u64 + 1;
        let before = self.touched.swap(now, Ordering::Relaxed);
        let was_open = before != 0
            && now.saturating_sub(before) <= self.page_open.as_millis() as u64;
        if !was_open {
            self.wake();
        }
    }

    /// How long ago the last dashboard request came in (`None`: none yet).
    pub fn since_touch(&self) -> Option<Duration> {
        let touched = self.touched.load(Ordering::Relaxed).checked_sub(1)?;
        let now = self.epoch.elapsed().as_millis() as u64;
        Some(Duration::from_millis(now.saturating_sub(touched)))
    }

    /// Asks the thread to end; it does at once unless it is inside a call
    /// (the first window, or creating the sampler).
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
        self.wake();
    }

    /// Stops the thread and waits for it at most `cap`; a thread still inside
    /// a call then is left to the process's exit.
    pub fn finish(&self, cap: Duration) {
        self.stop();
        let deadline = Instant::now() + cap;
        while !self.finished() {
            if Instant::now() >= deadline {
                eprintln!(
                    "machine: the sampler is still inside a call after {cap:?}; stopping without it"
                );
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        if let Some(thread) = lock(&self.thread).take() {
            let _ = thread.join();
        }
    }

    /// The thread has ended, or there never was one.
    pub fn finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }
}

/// Marks the thread ended; one that panicked outside the guarded calls
/// leaves the sampler failed.
struct Ended<'a>(&'a Machine);

impl Drop for Ended<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.fail("the sampler thread panicked".to_owned());
        }
        // Nothing samples at either cadence any more.
        self.0.short.store(false, Ordering::Relaxed);
        self.0.listening.store(false, Ordering::Relaxed);
        self.0.finished.store(true, Ordering::Release);
    }
}

/// What went wrong in a guarded call.
enum Fault {
    /// The call returned an error.
    Error(String),
    /// The call panicked.
    Panic(String),
}

impl Fault {
    fn message(self) -> String {
        match self {
            Self::Error(m) | Self::Panic(m) => m,
        }
    }
}

/// Runs `call`, turning an error or a panic (the calls reach into IOReport
/// and the SMC through FFI) into what was being done and what happened.
fn guarded<T>(
    what: &str,
    call: impl FnOnce() -> Result<T, String>,
) -> Result<T, Fault> {
    match catch_unwind(AssertUnwindSafe(call)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(Fault::Error(format!("{what} failed: {error}"))),
        Err(panic) => {
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "no message".to_owned());
            Err(Fault::Panic(format!("{what} panicked: {message}")))
        }
    }
}

/// Puts the calling thread at the utility quality of service: below the
/// engine's threads, above the background class (which would also throttle
/// its I/O). A failure is not worth a log line.
fn utility_priority() {
    // SAFETY: a plain call on the current thread with a class from the enum.
    let _ = unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0)
    };
}

#[cfg(test)]
#[path = "../../tests/unit/serve/machine.rs"]
mod tests;
