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

use std::collections::VecDeque;
use std::sync::Mutex;

use serde::Serialize;

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
}

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
            drafted_tokens: speculation.map(|s| s.drafted),
            accepted_tokens: speculation.map(|s| s.accepted),
            acceptance_ratio: speculation.filter(|s| s.drafted > 0).map(|s| {
                round(s.accepted.min(s.drafted) as f64 / s.drafted as f64, 1e4)
            }),
            agreement_tokens: cached_tokens,
            durable_prefix_tokens: None,
            image_tokens: None,
            vision_ms: None,
        }
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
