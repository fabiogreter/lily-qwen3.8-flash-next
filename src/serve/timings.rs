//! Lily's `timings` response extension and the ring buffer behind
//! `GET /v1/timings`.
//!
//! Every completed request already measures what the per-request log line
//! prints: how much of the prompt the session cache supplied, how long the
//! rest took to prefill, how long the decode ran, and how the draft head
//! did. [`Timings`] is that same set of numbers as a JSON object, attached
//! to the response next to `usage` and kept for the last few requests so a
//! client whose SDK drops unknown response fields can read them out of band.
//!
//! Rates are per *computed* token: the prefill rate divides by the tokens
//! actually run through the model, never by the whole prompt, so a cache hit
//! reports what it cost rather than an absurd number. A rate whose numerator
//! or denominator is zero is `null` instead of a division by zero, and the
//! speculation fields are `null` when the draft head is off for the request,
//! so they never read as a 0 % acceptance.
//!
//! Next to the headline numbers sit the diagnostics that say why a request
//! was slow: the time it queued, where `prefill_ms` went ([`PrefillPhases`]),
//! how the paged n-gram table's gathers fared ([`NgramStats`]) and what the
//! system's memory did meanwhile ([`MemoryStats`]). `GET /v1/timings` always
//! carries them; the log line prints them only when something stands out
//! ([`Timings::log_details`]).

use std::collections::VecDeque;
use std::sync::Mutex;

use serde::Serialize;

use crate::stats::{self, VmCounters};

/// What the draft head did for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Speculation {
    /// Draft tokens proposed across the request's speculative steps.
    pub drafted: usize,
    /// Of those, the ones the verify pass confirmed.
    pub accepted: usize,
}

/// One request's timings, as they appear in the JSON. Stable field names,
/// `snake_case`, all counts in tokens, all durations in milliseconds and all
/// rates in tokens per second.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Timings {
    /// Every token of the rendered prompt.
    pub prompt_tokens: usize,
    /// Prompt tokens the session cache supplied: a resident prefix, a forked
    /// session or a checkpoint restored from the disk tier.
    pub cached_tokens: usize,
    /// Prompt tokens actually run through the model (`prompt_tokens`
    /// - `cached_tokens`).
    pub prefill_tokens: usize,
    /// Wall time from the cache lookup to the checkpoint that ends the
    /// prefill phase: the `prefix` figure of the server log line, so it
    /// includes a disk-tier restore when there was one.
    pub prefill_ms: f64,
    /// `prefill_tokens` per second. `null` when nothing was prefilled (a full
    /// cache hit), because the rate of zero tokens says nothing.
    pub prefill_per_second: Option<f64>,
    /// Tokens the model drew, including the ones a stop string swallowed.
    pub generated_tokens: usize,
    /// Wall time of the decode loop.
    pub decode_ms: f64,
    /// `generated_tokens` per second, `null` when nothing was generated.
    pub decode_per_second: Option<f64>,
    /// Recurrent-state checkpoints the decode took (one every
    /// `--decode-checkpoint-tokens`), thinned ones included.
    pub decode_checkpoints: usize,
    /// Wall time taking them, part of `decode_ms`.
    pub decode_checkpoint_ms: f64,
    /// Draft tokens proposed, `null` when speculative decoding is off.
    pub drafted_tokens: Option<usize>,
    /// Draft tokens accepted, `null` when speculative decoding is off.
    pub accepted_tokens: Option<usize>,
    /// `accepted_tokens / drafted_tokens` in `0.0..=1.0`, `null` when
    /// speculative decoding is off or proposed nothing.
    pub acceptance_ratio: Option<f64>,
    /// How far the prompt agreed with any lineage the session cache knew,
    /// resident or on disk, capped at `prompt_tokens - 1`. Always at least
    /// `cached_tokens`; a gap between the two is a shared prefix nothing
    /// could resume from (a durable prefix entry closes it for later runs).
    pub agreement_tokens: usize,
    /// Position of the durable prefix entry this request wrote, when it
    /// wrote one; absent otherwise, so nothing reads as a zero-length write.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub durable_prefix_tokens: Option<usize>,
    /// Prompt tokens that are image placeholders (all of the request's
    /// images, cached or not); absent for a text request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_tokens: Option<usize>,
    /// Wall time of the vision tower over the images the session cache did
    /// not already hold (part of `prefill_ms`); absent for a text request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_ms: Option<f64>,
    /// What stopped the request before it finished: `"client"` (its
    /// connection closed, see `http::Watched`) or `"shutdown"` (the stop
    /// signal's grace ran out); absent for a request that ran to its end.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancelled_by: Option<&'static str>,
    /// The prompt position at which a cancellation stopped the prefill, a
    /// chunk boundary where the session cache kept the state for a retry;
    /// `prefill_tokens` are then the tokens run up to it. Absent when the
    /// prefill completed (a cancellation during the decode has only
    /// `cancelled_by`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancelled_at: Option<usize>,
    /// Wall time from the request entering the engine's queue to the engine
    /// starting it: other requests ahead of it, a reload after an idle
    /// unload, or pinning the weights (`pinned`). Not part of `prefill_ms`.
    pub queue_ms: f64,
    /// Whether the weights were pinned in memory (`--pin-weights`) when
    /// the request started; pinning them for it is part of `queue_ms`.
    pub pinned: bool,
    /// Where `prefill_ms` went.
    pub prefill_phases: PrefillPhases,
    /// What evicting sessions cost the request: while acquiring its session
    /// (part of `prefill_phases.session_ms`) and while returning it to the
    /// cache after the decode (before the response's last chunk).
    pub evictions: EvictionTimings,
    /// The paged n-gram table's gathers during the prefill and the decode.
    pub ngram: NgramStats,
    /// The system's memory over the request.
    pub memory: MemoryStats,
}

/// Where `prefill_ms` went, in milliseconds. These phases plus `vision_ms`
/// add up to `prefill_ms`; `other_ms` is what none of them measured
/// (bookkeeping between phases, and anything unexpected). A GPU time that
/// looks normal next to an exploding total means the stall was on the host.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct PrefillPhases {
    /// The session cache: finding the prefix to resume, forking or restoring
    /// a resident session, reading a checkpoint back from the disk tier, and
    /// any eviction to disk that making room for the request caused.
    pub session_ms: f64,
    /// Growing the decode state's caches and the prefill scratch.
    pub alloc_ms: f64,
    /// Hashing the n-gram ids and gathering their rows, as far as the GPU
    /// waited for it: the first chunk's staging and whatever of a later
    /// chunk's outlasted the chunk before it (`ngram.prefill` breaks the
    /// gathers down; its `hidden_ms` is the part that overlapped the GPU).
    pub ngram_ms: f64,
    /// Encoding the chunks' passes on the host.
    pub encode_ms: f64,
    /// The chunks' GPU execution, from the commit feedback's timestamps.
    pub gpu_ms: f64,
    /// Commit to completion minus the GPU execution: submission latency, the
    /// GPU busy with other work, residency being established.
    pub wait_ms: f64,
    /// Writing a durable prefix entry to the disk tier (0 without one).
    pub durable_ms: f64,
    /// The recurrent-state checkpoint that ends the prefill.
    pub checkpoint_ms: f64,
    pub other_ms: f64,
    /// Prefill chunks run (at most 4 096 tokens each).
    pub chunks: u64,
}

/// Sessions evicted from GPU memory during one step of a request, and how
/// (`docs/architecture.md`, "The session cache"). An eviction written ahead
/// in the background is a drop; one that was not writes the session on the
/// spot (`spilled`, `spill_ms`, the fallback); one that reached the session
/// being written ahead waited for that write (`waited_ms`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct EvictionPhase {
    /// Sessions evicted.
    pub evicted: usize,
    /// Of those, the ones whose copy was written ahead: dropped, no I/O.
    pub written_ahead: usize,
    /// Of those, the ones written to the disk tier synchronously.
    pub spilled: usize,
    /// Time in the synchronous writes.
    pub spill_ms: f64,
    /// Time waiting for a write ahead that had not finished.
    pub waited_ms: f64,
    /// Writes ahead cancelled because the request resumed that session.
    pub cancelled_writes: usize,
}

impl EvictionPhase {
    pub fn new(e: &super::session::Evictions) -> Self {
        Self {
            evicted: e.evicted,
            written_ahead: e.written_ahead,
            spilled: e.spilled,
            spill_ms: round(e.spill_secs * 1e3, 1e3),
            waited_ms: round(e.waited_secs * 1e3, 1e3),
            cancelled_writes: e.cancelled,
        }
    }

    /// Whether anything here cost time (a drop costs none).
    fn notable(&self) -> bool {
        self.spilled > 0 || self.waited_ms > 0.0 || self.cancelled_writes > 0
    }

    fn describe(&self) -> String {
        let mut parts = vec![format!("{} evicted", self.evicted)];
        if self.written_ahead > 0 {
            parts.push(format!("{} written ahead", self.written_ahead));
        }
        if self.spilled > 0 {
            parts.push(format!(
                "{} spilled in {:.2}s",
                self.spilled,
                self.spill_ms / 1e3
            ));
        }
        if self.waited_ms > 0.0 {
            parts
                .push(format!("waited {:.2}s for a write ahead", self.waited_ms / 1e3));
        }
        if self.cancelled_writes > 0 {
            parts.push(format!("{} write ahead cancelled", self.cancelled_writes));
        }
        parts.join(", ")
    }
}

/// [`EvictionPhase`] for the two places a request evicts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct EvictionTimings {
    /// Making room for the request's session (inside `session_ms`).
    pub acquire: EvictionPhase,
    /// Trimming the cache back to budget after the decode.
    pub release: EvictionPhase,
}

/// The wall-clock pieces [`PrefillPhases::split`] adds to the engine's
/// counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrefillParts {
    pub session_secs: f64,
    pub durable_secs: f64,
    pub checkpoint_secs: f64,
    /// What the prefill loop recorded over the request's prefill.
    pub counters: stats::Prefill,
}

impl PrefillPhases {
    /// Splits a prefill of `prefill_secs`, `vision_secs` of which the vision
    /// tower took, into its phases; the remainder becomes `other_ms` (never
    /// negative: overlapping clocks round the other way at worst).
    pub fn split(prefill_secs: f64, vision_secs: f64, parts: PrefillParts) -> Self {
        let c = parts.counters;
        let measured = parts.session_secs
            + c.alloc_secs
            + c.ngram_secs
            + c.encode_secs
            + c.gpu_secs
            + c.wait_secs
            + parts.durable_secs
            + parts.checkpoint_secs
            + vision_secs;
        let ms = |secs: f64| round(secs * 1e3, 1e3);
        Self {
            session_ms: ms(parts.session_secs),
            alloc_ms: ms(c.alloc_secs),
            ngram_ms: ms(c.ngram_secs),
            encode_ms: ms(c.encode_secs),
            gpu_ms: ms(c.gpu_secs),
            wait_ms: ms(c.wait_secs),
            durable_ms: ms(parts.durable_secs),
            checkpoint_ms: ms(parts.checkpoint_secs),
            other_ms: ms((prefill_secs - measured).max(0.0)),
            chunks: c.chunks,
        }
    }

    /// Prefill time spent anywhere but on the GPU, in milliseconds (the
    /// vision tower, a GPU phase of its own, excluded too).
    fn host_ms(&self) -> f64 {
        self.session_ms
            + self.alloc_ms
            + self.ngram_ms
            + self.encode_ms
            + self.wait_ms
            + self.durable_ms
            + self.checkpoint_ms
            + self.other_ms
    }
}

/// One phase's gathers from the paged n-gram table (all zero for a
/// resident table).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct GatherStats {
    /// Staged batches: one per prefill chunk, decode step or verify step.
    pub batches: u64,
    /// Table rows copied (16 per token).
    pub rows: u64,
    /// Rows whose pages were checked for residency right before the copy:
    /// every row in the decode, every 16th in the prefill.
    pub checked_rows: u64,
    /// Pages the checked rows lie on (three to six per row).
    pub pages: u64,
    /// Of those, the ones that were not resident, each a read from the SSD.
    pub cold_pages: u64,
    /// Rows outside the prefill's 1-in-16 sample that were checked and
    /// hinted too, because the sample found their chunk cold (1 page in 32
    /// or more); 0 for a warm prefill and in the decode.
    pub hinted_rows: u64,
    /// Wall time in the gathers, the checks included.
    pub gather_ms: f64,
    /// Staging time (hashing and gathers) that ran while the GPU executed
    /// the previous prefill chunk and so cost nothing; the exposed rest is
    /// `prefill_phases.ngram_ms`. Always 0 in the decode.
    pub hidden_ms: f64,
}

impl From<stats::Gather> for GatherStats {
    fn from(g: stats::Gather) -> Self {
        Self {
            batches: g.batches,
            rows: g.rows,
            checked_rows: g.checked_rows,
            pages: g.pages,
            cold_pages: g.cold_pages,
            hinted_rows: g.hinted_rows,
            gather_ms: round(g.secs * 1e3, 1e3),
            hidden_ms: round(g.hidden_secs * 1e3, 1e3),
        }
    }
}

/// The n-gram table's gathers of the two phases.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct NgramStats {
    pub prefill: GatherStats,
    /// The decode loop's gathers, the last prompt token's included.
    pub decode: GatherStats,
}

/// The system's memory over the request. The paging counters are
/// system-wide deltas in pages of 16 KB (`host_statistics64`), sampled when
/// the engine started the request, when the prefill ended and when the
/// decode ended; `null` where a sample failed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct MemoryStats {
    /// The memorystatus pressure level at the end: 1 normal, 2 warning,
    /// 4 critical.
    pub pressure_level: Option<u32>,
    /// The same as a word.
    pub pressure: Option<&'static str>,
    /// The process's physical footprint at the end, in bytes: what the
    /// system charges it with (its GPU buffers included, the page cache it
    /// maps not).
    pub phys_footprint_bytes: Option<u64>,
    /// Bytes of the process's memory in the compressor at the end.
    pub compressed_bytes: Option<u64>,
    pub prefill: Option<VmCounters>,
    pub decode: Option<VmCounters>,
}

impl MemoryStats {
    /// Builds the object from three samples of the paging counters (start,
    /// end of prefill, end of decode) and the end-of-request readings.
    pub fn from_samples(
        samples: [Option<VmCounters>; 3],
        pressure_level: Option<u32>,
        task: Option<stats::TaskMemory>,
    ) -> Self {
        let delta = |a: Option<VmCounters>, b: Option<VmCounters>| {
            a.zip(b).map(|(a, b)| b.since(a))
        };
        Self {
            pressure_level,
            pressure: pressure_level.map(stats::pressure_name),
            phys_footprint_bytes: task.map(|t| t.phys_footprint),
            compressed_bytes: task.map(|t| t.compressed),
            prefill: delta(samples[0], samples[1]),
            decode: delta(samples[1], samples[2]),
        }
    }

    /// Whether the request ran under memory pressure worth a log line: a
    /// raised pressure level, at least [`LOG_SWAP_PAGES`] swapped, or at
    /// least [`LOG_COMPRESSOR_PAGES`] through the compressor in one phase.
    /// A busy machine swaps a few pages and compresses tens of thousands in
    /// almost every request without any effect on it; what stalls lily is
    /// its own weights coming back from the compressor, millions of pages.
    fn notable(&self) -> bool {
        let busy = |vm: &Option<VmCounters>| {
            vm.is_some_and(|v| {
                v.swapins + v.swapouts >= LOG_SWAP_PAGES
                    || v.compressions + v.decompressions >= LOG_COMPRESSOR_PAGES
            })
        };
        self.pressure_level.is_some_and(|l| l > 1)
            || busy(&self.prefill)
            || busy(&self.decode)
    }
}

/// The log line shows the queue time from this many milliseconds on.
const LOG_QUEUE_MS: f64 = 1_000.0;
/// ... the prefill phases once this much of the prefill was spent off the
/// GPU (a normal prefill spends a few tens of milliseconds there).
const LOG_HOST_MS: f64 = 500.0;
/// ... the n-gram gathers once the prefill waited this long for them (the
/// gather time the GPU did not hide) ...
const LOG_PREFILL_GATHER_MS: f64 = 500.0;
/// ... or a decode step spent this long in them on average, more than the
/// parked window that hides the host's staging (docs/architecture.md,
/// "Hiding the host round trip") ...
const LOG_DECODE_GATHER_MS_PER_STEP: f64 = 1.0;
/// ... and the memory counters once one phase swapped 256 MB ...
const LOG_SWAP_PAGES: u64 = 16_384;
/// ... or moved 4 GB through the compressor. They also appear whenever the
/// prefill phases do, since a stall on the host is read against them.
const LOG_COMPRESSOR_PAGES: u64 = 262_144;

/// Tokens per second, or `None` when either side is zero: a rate over no
/// tokens is meaningless and a division by a zero duration is worse.
fn rate(tokens: usize, secs: f64) -> Option<f64> {
    (tokens > 0 && secs > 0.0 && secs.is_finite())
        .then(|| round(tokens as f64 / secs, 100.0))
}

/// Keeps the JSON readable: the numbers come from a wall clock, so the
/// digits past this are noise either way.
fn round(value: f64, scale: f64) -> f64 {
    (value * scale).round() / scale
}

impl Timings {
    /// Builds the object from what [`crate::serve`] already measured.
    /// `speculation` is `None` when the draft head is off for the request.
    pub fn measure(
        prompt_tokens: usize,
        cached_tokens: usize,
        prefill_secs: f64,
        generated_tokens: usize,
        decode_secs: f64,
        speculation: Option<Speculation>,
    ) -> Self {
        let cached_tokens = cached_tokens.min(prompt_tokens);
        let prefill_tokens = prompt_tokens - cached_tokens;
        Self {
            prompt_tokens,
            cached_tokens,
            prefill_tokens,
            prefill_ms: round(prefill_secs * 1e3, 1e3),
            prefill_per_second: rate(prefill_tokens, prefill_secs),
            generated_tokens,
            decode_ms: round(decode_secs * 1e3, 1e3),
            decode_per_second: rate(generated_tokens, decode_secs),
            decode_checkpoints: 0,
            decode_checkpoint_ms: 0.0,
            drafted_tokens: speculation.map(|s| s.drafted),
            accepted_tokens: speculation.map(|s| s.accepted),
            acceptance_ratio: speculation.filter(|s| s.drafted > 0).map(|s| {
                round(s.accepted.min(s.drafted) as f64 / s.drafted as f64, 1e4)
            }),
            agreement_tokens: cached_tokens,
            durable_prefix_tokens: None,
            image_tokens: None,
            vision_ms: None,
            cancelled_by: None,
            cancelled_at: None,
            queue_ms: 0.0,
            pinned: false,
            prefill_phases: PrefillPhases::default(),
            evictions: EvictionTimings::default(),
            ngram: NgramStats::default(),
            memory: MemoryStats::default(),
        }
    }

    /// Adds the diagnostics: the time the request queued, where the prefill
    /// went, the n-gram gathers and the memory samples.
    pub fn with_diagnostics(
        mut self,
        queue_secs: f64,
        prefill_phases: PrefillPhases,
        ngram: NgramStats,
        memory: MemoryStats,
    ) -> Self {
        self.queue_ms = round(queue_secs * 1e3, 1e3);
        self.prefill_phases = prefill_phases;
        self.ngram = ngram;
        self.memory = memory;
        self
    }

    /// The diagnostics worth appending to the request's log line, each group
    /// only when it stands out: a long queue, a prefill that spent more than
    /// [`LOG_HOST_MS`] off the GPU, n-gram gathers that cost noticeable time,
    /// memory pressure (and the memory state next to every slow prefill).
    /// Empty for an ordinary request; `GET /v1/timings` has everything.
    pub fn log_details(&self) -> String {
        let secs = |ms: f64| format!("{:.2}s", ms / 1e3);
        let mut parts = Vec::new();
        if self.queue_ms >= LOG_QUEUE_MS {
            parts.push(format!("queued {}", secs(self.queue_ms)));
        }
        let p = &self.prefill_phases;
        let slow_prefill = p.host_ms() >= LOG_HOST_MS;
        if slow_prefill {
            let mut phases = vec![
                format!("session {}", secs(p.session_ms)),
                format!("alloc {}", secs(p.alloc_ms)),
                format!("ngram {}", secs(p.ngram_ms)),
                format!("encode {}", secs(p.encode_ms)),
                format!("gpu {}", secs(p.gpu_ms)),
                format!("wait {}", secs(p.wait_ms)),
            ];
            if p.durable_ms > 0.0 {
                phases.push(format!("durable {}", secs(p.durable_ms)));
            }
            phases.push(format!("checkpoint {}", secs(p.checkpoint_ms)));
            phases.push(format!("other {}", secs(p.other_ms)));
            parts.push(format!("prefill phases: {}", phases.join(", ")));
        }
        let ev = &self.evictions;
        if ev.acquire.notable() || ev.release.notable() {
            parts.push(format!(
                "evictions: acquire {}; release {}",
                ev.acquire.describe(),
                ev.release.describe()
            ));
        }
        let (np, nd) = (&self.ngram.prefill, &self.ngram.decode);
        let decode_per_step = nd.gather_ms / nd.batches.max(1) as f64;
        if np.gather_ms - np.hidden_ms >= LOG_PREFILL_GATHER_MS
            || decode_per_step >= LOG_DECODE_GATHER_MS_PER_STEP
        {
            parts.push(format!(
                "ngram cold pages: prefill {}/{} checked (gather {}, {} hidden), decode {}/{} (gather {})",
                np.cold_pages,
                np.pages,
                secs(np.gather_ms),
                secs(np.hidden_ms),
                nd.cold_pages,
                nd.pages,
                secs(nd.gather_ms),
            ));
        }
        let m = &self.memory;
        if (slow_prefill && m.prefill.is_some()) || m.notable() {
            let vm = |label: &str, v: &Option<VmCounters>| match v {
                Some(v) => format!(
                    "{label} pageins {} pageouts {} swapins {} swapouts {} compressions {} decompressions {}",
                    v.pageins,
                    v.pageouts,
                    v.swapins,
                    v.swapouts,
                    v.compressions,
                    v.decompressions
                ),
                None => format!("{label} unavailable"),
            };
            parts.push(format!(
                "memory: pressure {}, footprint {}; {}; {}",
                m.pressure.unwrap_or("unknown"),
                match (m.phys_footprint_bytes, m.compressed_bytes) {
                    (Some(f), Some(c)) => format!(
                        "{:.1} GB ({:.1} GB compressed)",
                        f as f64 / 1e9,
                        c as f64 / 1e9
                    ),
                    _ => "unknown".to_owned(),
                },
                vm("prefill", &m.prefill),
                vm("decode", &m.decode),
            ));
        }
        parts.iter().map(|part| format!("; {part}")).collect()
    }

    /// Adds the decode checkpoints the generation took and their time.
    pub fn with_decode_checkpoints(mut self, taken: usize, secs: f64) -> Self {
        self.decode_checkpoints = taken;
        self.decode_checkpoint_ms = round(secs * 1e3, 1e3);
        self
    }

    /// Adds what evicting sessions cost at the acquire and the release.
    pub fn with_evictions(mut self, evictions: EvictionTimings) -> Self {
        self.evictions = evictions;
        self
    }

    /// Records a cancellation: who stopped the request and, when it was
    /// stopped during the prefill, the position it got to, which is also
    /// what `prefill_tokens` and the prefill rate then count.
    pub fn with_cancel(mut self, by: Option<&'static str>, at: Option<usize>) -> Self {
        self.cancelled_by = by;
        self.cancelled_at = at;
        if let Some(at) = at {
            self.prefill_tokens =
                at.clamp(self.cached_tokens, self.prompt_tokens) - self.cached_tokens;
            self.prefill_per_second = rate(self.prefill_tokens, self.prefill_ms / 1e3);
        }
        self
    }

    /// Records whether the weights were pinned when the request started.
    pub fn with_pinned(mut self, pinned: bool) -> Self {
        self.pinned = pinned;
        self
    }

    /// Adds the request's images: how many prompt tokens they take and how
    /// long the tower ran for the ones that had to be encoded. A request
    /// without images (`image_tokens == 0`) leaves both fields absent.
    pub fn with_vision(mut self, image_tokens: usize, vision_secs: f64) -> Self {
        if image_tokens > 0 {
            self.image_tokens = Some(image_tokens);
            self.vision_ms = Some(round(vision_secs * 1e3, 1e3));
        }
        self
    }

    /// Adds what the session cache saw beyond the reused prefix: the
    /// agreement with any lineage (never less than `cached_tokens`) and the
    /// durable prefix entry written for it, if one was.
    pub fn with_agreement(
        mut self,
        agreement_tokens: usize,
        durable_prefix_tokens: Option<usize>,
    ) -> Self {
        self.agreement_tokens =
            agreement_tokens.clamp(self.cached_tokens, self.prompt_tokens);
        self.durable_prefix_tokens = durable_prefix_tokens;
        self
    }
}

/// One entry of the log behind `GET /v1/timings`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TimingsEntry {
    /// The response id the request was answered with (`chatcmpl-…`,
    /// `cmpl-…`), which is also what the server log line is keyed by.
    pub id: String,
    pub model: &'static str,
    /// Unix seconds, the same `created` the response carries.
    pub created: u64,
    pub timings: Timings,
}

/// The last [`TimingsLog::capacity`] completed requests, newest first.
///
/// The engine thread appends one entry per request and the connection
/// threads read the whole buffer; the lock is held for a push or a clone of
/// at most `capacity` small entries, never across I/O or a GPU call.
#[derive(Debug)]
pub struct TimingsLog {
    entries: Mutex<VecDeque<TimingsEntry>>,
    capacity: usize,
}

impl TimingsLog {
    /// A log of at most `capacity` entries (at least one).
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self { entries: Mutex::new(VecDeque::with_capacity(capacity)), capacity }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Appends one entry, dropping the oldest once the buffer is full.
    pub fn record(&self, entry: TimingsEntry) {
        let mut entries =
            self.entries.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if entries.len() == self.capacity {
            entries.pop_front();
        }
        entries.push_back(entry);
    }

    /// The entries, newest first.
    pub fn recent(&self) -> Vec<TimingsEntry> {
        let entries =
            self.entries.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        entries.iter().rev().cloned().collect()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/serve/timings.rs"]
mod tests;
