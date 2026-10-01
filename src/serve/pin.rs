//! Pinning the weights in memory while the server is in use
//! (`--pin-weights`, `--pin-hold`).
//!
//! The weights are anonymous shared-storage Metal buffers, so while the
//! engine waits between an agent's requests (a tool run of 16 to 35 s) macOS
//! compresses them like any other idle memory, and the next request waits
//! seconds in `prefill_phases.wait_ms` for millions of pages to come back
//! from the compressor before 0.1 to 2 s of GPU work. Being in the queue's
//! residency set does not prevent that; `mlock` wires the pages, and wired
//! pages are not compressed.
//!
//! The pin follows use: the first request of an active period locks the
//! weights before its prefill (that request pays for it, about 2 to 3 s for
//! the full model, more when they were already compressed), the pin holds
//! while requests keep coming and is released `--pin-hold` (default 1m)
//! after the last one finished, as soon as the memory pressure level reaches
//! warning (a monitor thread polls it every second), and before the engine
//! drops its buffers (idle unload, GPU fault, shutdown). A failed `mlock`
//! unlocks what it had locked, is logged, and is not retried before the
//! next active period; the server serves unpinned meanwhile.
//!
//! The hold is also the GPU keep-alive's window
//! ([`WeightPin::keep_warm_remaining`]): while the weights are pinned and
//! the hold runs, the idle engine signals the GPU once a second so the
//! queue's residency set is not dropped; once the pin is released, for
//! whatever reason, it submits nothing.
//!
//! What is pinned: [`LanguageModel::weight_buffers`], the buffers the load
//! read the weights into, the vision tower included. Never the expert
//! cache's slab, the session caches, the scratch or the paged n-gram table.
//! Pinning acts only on buffers the load already allocated, so it changes
//! nothing in the memory plan. Whether to pin is [`decide`], a pure
//! function of the plan and the machine's limits.
//!
//! [`LanguageModel::weight_buffers`]: crate::engine::LanguageModel::weight_buffers

use std::str::FromStr;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use objc2_metal::MTLBuffer as _;

use crate::metal::Buffer;

/// `--pin-weights`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PinMode {
    /// Pin when the whole model is resident (no expert cache) and the pin
    /// leaves the margins of [`decide`] free.
    #[default]
    Auto,
    /// Never pin.
    Off,
    /// Pin whenever it fits below the wire limit (with its margin), on a
    /// small machine too; the expert cache's slab is still never pinned.
    Always,
}

impl FromStr for PinMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            "always" => Ok(Self::Always),
            other => {
                anyhow::bail!("unknown pin mode {other:?}; use auto, off or always")
            }
        }
    }
}

/// What [`decide`] looks at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinInputs {
    pub mode: PinMode,
    /// The memory the load planned for: `--memory-gb` when given, else the
    /// physical memory (`LanguageModel::planned_memory`).
    pub planned_memory: Option<u64>,
    /// `hw.memsize`.
    pub physical_memory: Option<u64>,
    /// [`crate::stats::user_wire_limit`].
    pub wire_limit: Option<u64>,
    /// The bytes the pin would lock, rounded out to whole pages.
    pub pin_bytes: u64,
    /// Bytes this process locks otherwise (`--ngram-lock`'s table), which
    /// count against the same per-process wire limit.
    pub locked_elsewhere: u64,
    /// Whether the load serves the experts from an expert cache.
    pub expert_cache: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PinDecision {
    Pin,
    /// Why not, for the `pin skipped: <reason>` log line.
    Skip(String),
}

const GIB: u64 = 1 << 30;

/// What `auto` leaves free of the planned memory next to the pinned weights:
/// the expert-cache plan's own reserve for the OS, other applications and
/// the page cache (a sixth of memory, at least 12 GiB) plus its 5 GiB for
/// scratch and caches (`auto_expert_slots` in `qwen4exp::weights`). A pin
/// that leaves this much is one the plan would have let the weights have
/// resident anyway, so it takes nothing the plan kept for others; on 128 GB
/// that is 28.3 GB, next to 74 GB of weights.
fn memory_margin(planned: u64) -> u64 {
    (planned / 6).max(12 * GIB) + 5 * GIB
}

/// Left below the wire limit: an eighth of it, at least 8 GiB (14.6 GB of
/// 116.8 GB). The global part of the limit counts every page the system has
/// wired already (the kernel, other processes' `mlock`s and GPU memory in
/// use), which varies, so a pin that would end right at the limit fails or
/// starves the next wiring; the margin keeps that headroom.
fn wire_margin(limit: u64) -> u64 {
    (limit / 8).max(8 * GIB)
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

/// Whether to pin, as a pure function of the plan and the limits:
///
/// - `off` never pins; nothing to pin never pins.
/// - `auto` skips when the expert cache is active: the resident non-expert
///   weights are about 3 GB there and not where the stalls come from, and
///   wiring them would come out of the reserve the plan keeps for the page
///   cache that streams the other experts. Otherwise it pins only when the
///   pinned bytes leave `memory_margin` free of the planned memory (the
///   smaller of the plan and the physical memory, so `--memory-gb 64` on a
///   128 GB machine decides like a 64 GB machine).
/// - `auto` and `always` both pin only when the pinned bytes plus what the
///   process locks otherwise leave `wire_margin` below the wire limit.
///   An unreadable limit counts as half the physical memory (too little
///   for the full model), and with neither known nothing is pinned.
pub fn decide(i: &PinInputs) -> PinDecision {
    let skip = |why: String| PinDecision::Skip(why);
    if i.mode == PinMode::Off {
        return skip("--pin-weights off".into());
    }
    if i.pin_bytes == 0 {
        return skip("no weight buffers to pin".into());
    }
    if i.mode == PinMode::Auto {
        if i.expert_cache {
            return skip("expert cache active".into());
        }
        let planned = match (i.planned_memory, i.physical_memory) {
            (Some(p), Some(m)) => p.min(m),
            (p, m) => match p.or(m) {
                Some(known) => known,
                None => return skip("planned memory unknown".into()),
            },
        };
        let margin = memory_margin(planned);
        if i.pin_bytes.saturating_add(margin) > planned {
            return skip(format!(
                "{:.1} GB of weights would leave less than {:.1} GB of the {:.1} GB planned memory \
                 unpinned (--pin-weights always overrides)",
                gb(i.pin_bytes),
                gb(margin),
                gb(planned)
            ));
        }
    }
    let (limit, assumed) = match (i.wire_limit, i.physical_memory) {
        (Some(limit), _) => (limit, false),
        (None, Some(physical)) => (physical / 2, true),
        (None, None) => {
            return skip("the wire limit and the physical memory are unknown".into());
        }
    };
    let margin = wire_margin(limit);
    let wired = i.pin_bytes.saturating_add(i.locked_elsewhere);
    if wired.saturating_add(margin) > limit {
        return skip(format!(
            "{:.1} GB of weights{} would leave less than {:.1} GB below the {:.1} GB wire limit{}",
            gb(i.pin_bytes),
            if i.locked_elsewhere > 0 {
                format!(
                    " and {:.1} GB locked otherwise (--ngram-lock)",
                    gb(i.locked_elsewhere)
                )
            } else {
                String::new()
            },
            gb(margin),
            gb(limit),
            if assumed {
                " (vm.user_wire_limit unreadable, half the physical memory assumed)"
            } else {
                ""
            }
        ));
    }
    PinDecision::Pin
}

/// The memorystatus level from which the pin is released (2: warning).
const WARNING: u32 = 2;
/// How often the monitor thread reads the pressure level while pinned.
const PRESSURE_POLL: Duration = Duration::from_secs(1);
/// The GPU keep-alive's window after the last request when the hold is
/// unlimited: the default `--pin-hold`.
const UNLIMITED_HOLD_KEEP_WARM: Duration = Duration::from_secs(60);

/// `buffers`' memory as page-aligned `(start, len)` ranges, sorted and with
/// overlapping or touching ones merged (a page two small buffers share is
/// locked once), and their total bytes.
pub fn page_ranges(buffers: &[Buffer]) -> (Vec<(usize, usize)>, u64) {
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let mut ranges: Vec<(usize, usize)> = buffers
        .iter()
        .map(|b| {
            let start = b.contents().as_ptr() as usize;
            let end = start + b.length();
            (start - start % page, end.next_multiple_of(page))
        })
        .collect();
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    let total = merged.iter().map(|(s, e)| (e - s) as u64).sum();
    (merged.into_iter().map(|(s, e)| (s, e - s)).collect(), total)
}

/// Locks every range, or none: on a failure the ranges locked before it are
/// unlocked again (the failing call itself wires nothing; the kernel rolls a
/// partial wiring back).
fn lock_all(ranges: &[(usize, usize)]) -> std::result::Result<(), String> {
    for (i, &(start, len)) in ranges.iter().enumerate() {
        // SAFETY: a page-aligned range of buffers the pin keeps alive.
        if unsafe { libc::mlock(start as *const libc::c_void, len) } != 0 {
            let error = std::io::Error::last_os_error();
            unlock_all(&ranges[..i]);
            return Err(format!(
                "mlock of {:.2} GB (range {} of {}, {:.1} GB locked before it): {error}",
                gb(len as u64),
                i + 1,
                ranges.len(),
                gb(ranges[..i].iter().map(|r| r.1 as u64).sum())
            ));
        }
    }
    Ok(())
}

fn unlock_all(ranges: &[(usize, usize)]) {
    for &(start, len) in ranges {
        // SAFETY: a range this pin locked, of buffers it keeps alive.
        if unsafe { libc::munlock(start as *const libc::c_void, len) } != 0 {
            eprintln!(
                "pin: munlock of {:.2} GB failed: {}",
                gb(len as u64),
                std::io::Error::last_os_error()
            );
        }
    }
}

/// What the engine thread and the monitor thread share. The ranges are raw
/// addresses; the memory behind them stays allocated because the
/// [`WeightPin`] holds its buffers until after the last unlock.
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    /// The pressure level source: [`crate::stats::pressure_level`] in the
    /// server, a fake in the tests.
    pressure: fn() -> Option<u32>,
}

struct State {
    ranges: Vec<(usize, usize)>,
    pinned: bool,
    stop: bool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The raised pressure level, if it is warning or higher.
    fn raised_pressure(&self) -> Option<u32> {
        (self.pressure)().filter(|&level| level >= WARNING)
    }
}

/// While pinned, releases the pin as soon as the pressure level reaches
/// warning; sleeps on the condition variable otherwise. Unlocking while a
/// request runs is fine: `munlock` only makes the pages pageable again.
fn monitor(shared: Arc<Shared>) {
    let mut state = shared.lock();
    loop {
        if state.stop {
            return;
        }
        if !state.pinned {
            state = shared.wake.wait(state).unwrap_or_else(|p| p.into_inner());
            continue;
        }
        state = shared
            .wake
            .wait_timeout(state, PRESSURE_POLL)
            .unwrap_or_else(|p| p.into_inner())
            .0;
        if state.stop || !state.pinned {
            continue;
        }
        if let Some(level) = shared.raised_pressure() {
            unlock_all(&state.ranges);
            state.pinned = false;
            eprintln!(
                "released pin: memory pressure {}",
                crate::stats::pressure_name(level)
            );
        }
    }
}

/// The engine's pin on its weights. Lives with the engine on its thread and
/// is dropped before the model: its drop unlocks, then releases its handles
/// on the buffers, so no locked memory is ever freed and nothing is locked
/// after it was freed.
pub struct WeightPin {
    decision: PinDecision,
    /// `--pin-hold`; `None` holds until pressure or the unload.
    hold: Option<Duration>,
    shared: Arc<Shared>,
    monitor: Option<JoinHandle<()>>,
    bytes: u64,
    /// When the last request finished.
    last_request: Option<Instant>,
    /// An `mlock` failed in this active period; retried in the next one.
    failed: bool,
    /// A skip under pressure was logged in this active period.
    pressure_logged: bool,
    /// Keeps the pinned memory allocated until the pin is gone.
    _buffers: Vec<Buffer>,
}

impl WeightPin {
    /// A released pin on `buffers` that pins at [`Self::before_request`]
    /// when `decision` says so and `pressure` reads below warning. A
    /// `hold_secs` of 0 keeps the pin until pressure or the unload.
    pub fn new(
        buffers: Vec<Buffer>,
        decision: PinDecision,
        hold_secs: u64,
        pressure: fn() -> Option<u32>,
    ) -> Result<Self> {
        let (ranges, bytes) = page_ranges(&buffers);
        let shared = Arc::new(Shared {
            state: Mutex::new(State { ranges, pinned: false, stop: false }),
            wake: Condvar::new(),
            pressure,
        });
        let monitor = if decision == PinDecision::Pin {
            let shared = shared.clone();
            Some(
                std::thread::Builder::new()
                    .name("lily-pin-monitor".into())
                    .spawn(move || monitor(shared))
                    .context("spawning the pin's pressure monitor")?,
            )
        } else {
            None
        };
        Ok(Self {
            decision,
            hold: (hold_secs > 0).then(|| Duration::from_secs(hold_secs)),
            shared,
            monitor,
            bytes,
            last_request: None,
            failed: false,
            pressure_logged: false,
            _buffers: buffers,
        })
    }

    pub fn pinned(&self) -> bool {
        self.shared.lock().pinned
    }

    /// Called before a request's prefill: pins unless the decision says
    /// not to, the pin is held already, an `mlock` failed earlier in this
    /// active period, or the memory pressure is raised. Returns whether the
    /// weights are pinned for the request.
    pub fn before_request(&mut self, now: Instant) -> bool {
        if self.decision != PinDecision::Pin {
            return false;
        }
        // The previous active period ended when no request came for the hold.
        if let (Some(last), Some(hold)) = (self.last_request, self.hold)
            && now.saturating_duration_since(last) >= hold
        {
            self.failed = false;
            self.pressure_logged = false;
        }
        let mut state = self.shared.lock();
        if state.pinned {
            return true;
        }
        if self.failed {
            return false;
        }
        if let Some(level) = self.shared.raised_pressure() {
            if !self.pressure_logged {
                eprintln!(
                    "pin skipped: memory pressure {}",
                    crate::stats::pressure_name(level)
                );
                self.pressure_logged = true;
            }
            return false;
        }
        let started = Instant::now();
        match lock_all(&state.ranges) {
            Ok(()) => {
                state.pinned = true;
                self.pressure_logged = false;
                self.shared.wake.notify_all();
                eprintln!(
                    "pinned {:.1} GB of weights in {:.1}s",
                    gb(self.bytes),
                    started.elapsed().as_secs_f64()
                );
                true
            }
            Err(error) => {
                self.failed = true;
                eprintln!(
                    "pin failed: {error}; serving unpinned, retried at the next active period"
                );
                false
            }
        }
    }

    /// Called when a request finished: the hold counts from here.
    pub fn after_request(&mut self, now: Instant) {
        self.last_request = Some(now);
    }

    /// How long the pin is held without another request: `None` when
    /// nothing is pinned or the hold is unlimited, zero when it is over.
    pub fn hold_remaining(&self, now: Instant) -> Option<Duration> {
        let (last, hold) = (self.last_request?, self.hold?);
        self.pinned().then(|| hold.saturating_sub(now.saturating_duration_since(last)))
    }

    /// What is left of the GPU keep-alive's window ([`super::KeepAlive`]):
    /// the hold after the last request while the weights are pinned, and
    /// with an unlimited hold (`--pin-hold 0`) the default hold's minute,
    /// so the ticks always end. `None` when nothing is pinned (never pinned,
    /// a skipped decision, a failed `mlock`, released after the hold or
    /// under memory pressure), zero when the window is over.
    pub fn keep_warm_remaining(&self, now: Instant) -> Option<Duration> {
        let last = self.last_request?;
        let window = self.hold.unwrap_or(UNLIMITED_HOLD_KEEP_WARM);
        self.pinned()
            .then(|| window.saturating_sub(now.saturating_duration_since(last)))
    }

    /// Releases the pin once its hold is over.
    pub fn release_if_held_out(&mut self, now: Instant) {
        if self.hold_remaining(now) == Some(Duration::ZERO) {
            let hold = self.hold.map_or(0, |h| h.as_secs());
            self.release(&format!(
                "released pin after {} idle",
                super::describe_secs(hold)
            ));
        }
    }

    /// Unlocks when pinned and logs `line` then.
    fn release(&self, line: &str) {
        let mut state = self.shared.lock();
        if state.pinned {
            unlock_all(&state.ranges);
            state.pinned = false;
            eprintln!("{line}");
        }
    }
}

impl Drop for WeightPin {
    fn drop(&mut self) {
        self.release("released pin: unloading");
        self.shared.lock().stop = true;
        self.shared.wake.notify_all();
        if let Some(monitor) = self.monitor.take() {
            let _ = monitor.join();
        }
        // `_buffers` drops after this, with the pin already released.
    }
}

#[cfg(test)]
#[path = "../../tests/unit/serve/pin.rs"]
mod tests;
