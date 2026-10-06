//! What every request's answer is made of, built once for both ways of
//! serving it: one request at a time (`Engine::run`) and the batch scheduler
//! (`Engine::serve_batched`). [`Facts`] is what a request's admission
//! measured; [`Outcome`] how its decode went. From the two come its
//! `timings` ([`timings`]), its log line ([`answer_line`], or
//! [`stopped_line`] for a prefill that was stopped) and the end of its
//! response ([`stream_end`], [`response_body`]).

use std::time::Duration;

use serde_json::{Value, json};

use super::api::Kind;
use super::batch::BatchStats;
use super::session::{Evictions, SessionStore};
use super::timings::{
    EvictionPhase, EvictionTimings, MemoryStats, NgramStats, PrefillParts,
    PrefillPhases, Speculation, Timings,
};
use super::{Collected, call_id, chunk, describe_reuse, text_chunk};
use crate::engine::LanguageModel;
use crate::stats::{Counters, TaskMemory, VmCounters};

/// What a request's admission measured, for its timings and log line.
#[derive(Debug, Clone, Default)]
pub(super) struct Facts {
    /// Prompt tokens.
    pub(super) n: usize,
    /// Prompt tokens the session cache supplied.
    pub(super) reused: usize,
    /// How far the prompt agreed with any lineage the cache knew.
    pub(super) agreement: usize,
    /// The cache cut its session back in place, by this many tokens.
    pub(super) cut_back: Option<usize>,
    /// The cache forked its session from a lineage.
    pub(super) forked: bool,
    /// The reused prefix came from the disk tier, in this long.
    pub(super) from_disk: Option<Duration>,
    /// The durable prefix entry written, and how long writing it took.
    pub(super) durable: Option<(usize, f64)>,
    /// The durable prefix phase's wall time, snapshot included.
    pub(super) durable_secs: f64,
    /// The request's images, how many of them the HTTP thread decoded and
    /// how long preparing the request took there.
    pub(super) images: usize,
    pub(super) images_decoded: usize,
    pub(super) prepare_secs: f64,
    /// Prompt tokens that are image placeholders.
    pub(super) image_tokens: usize,
    /// The vision tower's wall time, and how many images it encoded.
    pub(super) vision_secs: f64,
    pub(super) encoded_images: usize,
    pub(super) queued_secs: f64,
    pub(super) pinned: bool,
    pub(super) session_secs: f64,
    /// The checkpoint that ends the prefill (0 for a stopped prefill).
    pub(super) checkpoint_secs: f64,
    /// From the cache lookup to the end of the prefill.
    pub(super) prefix_secs: f64,
    pub(super) acquire_evictions: Evictions,
    /// The engine thread's counters and the system's paging counters at
    /// the start and at the end of the prefill.
    pub(super) counters_start: Counters,
    pub(super) counters_prefill: Counters,
    pub(super) vm_start: Option<VmCounters>,
    pub(super) vm_prefill: Option<VmCounters>,
}

/// How a request's decode went, for its timings and log line; all zero
/// (`Default`) for a prefill that was stopped, apart from the cancellation.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Outcome<'a> {
    /// Tokens drawn.
    pub(super) generated: usize,
    pub(super) decode_secs: f64,
    /// What the draft head did; `None` when it is off.
    pub(super) speculation: Option<Speculation>,
    pub(super) decode_checkpoints: usize,
    pub(super) decode_checkpoint_secs: f64,
    pub(super) cancelled_by: Option<&'static str>,
    /// Where a cancellation stopped the prefill.
    pub(super) cancelled_at: Option<usize>,
    /// How the request shared the GPU; `None` with batching off.
    pub(super) batch: Option<&'a BatchStats>,
}

/// The request's `timings`. `counters_end` and `vm_end` are the samples
/// at the end of the decode (of the prefill, when it was stopped);
/// `pressure_level` and `task` the end-of-request readings.
pub(super) fn timings(
    f: &Facts,
    o: &Outcome<'_>,
    release: &Evictions,
    counters_end: Counters,
    vm_end: Option<VmCounters>,
    pressure_level: Option<u32>,
    task: Option<TaskMemory>,
) -> Timings {
    let measured = Timings::measure(
        f.n,
        f.reused,
        f.prefix_secs,
        o.generated,
        o.decode_secs,
        o.speculation,
    )
    .with_agreement(f.agreement, f.durable.map(|(b, _)| b))
    .with_vision(f.image_tokens, f.vision_secs)
    .with_decode_checkpoints(o.decode_checkpoints, o.decode_checkpoint_secs)
    .with_pinned(f.pinned);
    let measured = match o.batch {
        Some(batch) => measured.with_batch(batch.timings()),
        None => measured,
    };
    measured
        .with_cancel(o.cancelled_by, o.cancelled_at)
        .with_evictions(EvictionTimings {
            acquire: EvictionPhase::new(&f.acquire_evictions),
            release: EvictionPhase::new(release),
        })
        .with_diagnostics(
            f.queued_secs,
            PrefillPhases::split(
                f.prefix_secs,
                if f.images == 0 { 0.0 } else { f.vision_secs },
                PrefillParts {
                    session_secs: f.session_secs,
                    durable_secs: f.durable_secs,
                    checkpoint_secs: f.checkpoint_secs,
                    counters: f.counters_prefill.since(f.counters_start).prefill,
                },
            ),
            NgramStats {
                prefill: f.counters_prefill.since(f.counters_start).gather.into(),
                decode: counters_end.since(f.counters_prefill).gather.into(),
            },
            MemoryStats::from_samples(
                [f.vm_start, f.vm_prefill, vm_end],
                pressure_level,
                task,
            ),
        )
}

/// The session cache's state for the end of a log line: resident sessions,
/// their bytes against the budget, and the disk tier's entries and bytes.
pub(super) fn store_line(
    sessions: usize,
    used_bytes: usize,
    budget_bytes: usize,
    disk: Option<(usize, u64)>,
) -> String {
    format!(
        "sessions={} ({:.1}/{:.1} GB){}",
        sessions,
        used_bytes as f64 / 1e9,
        budget_bytes as f64 / 1e9,
        disk.map(|(len, used)| format!(", disk {} ({:.1} GB)", len, used as f64 / 1e9))
            .unwrap_or_default(),
    )
}

/// [`store_line`] of `sessions` as they are now.
pub(super) fn describe_store<M: LanguageModel>(sessions: &SessionStore<M>) -> String {
    store_line(
        sessions.len(),
        sessions.used_bytes(),
        sessions.budget_bytes(),
        sessions.disk().map(|d| (d.len(), d.used_bytes())),
    )
}

fn from_disk(from_disk: Option<Duration>) -> String {
    from_disk
        .map(|d| format!(", from disk in {:.2}s", d.as_secs_f64()))
        .unwrap_or_default()
}

/// The log line of a request whose prefill `by` (the client or the
/// shutdown) stopped at `at`; `store` is [`store_line`], `details`
/// [`Timings::log_details`].
pub(super) fn stopped_line(
    id: &str,
    f: &Facts,
    by: &str,
    at: usize,
    store: &str,
    details: &str,
) -> String {
    format!(
        "{id}: {} prompt tokens ({} cached{}{}), cancelled by the {by} at {at} after {} prefilled in {:.2}s, kept {at} tokens as a session, {store}{details}",
        f.n,
        f.reused,
        describe_reuse(f.cut_back, f.forked),
        from_disk(f.from_disk),
        at - f.reused,
        f.prefix_secs,
    )
}

/// The log line of an answered request; `store` is [`store_line`],
/// `details` [`Timings::log_details`]. The batch scheduler's requests carry
/// a "batched: ..." group ([`BatchStats::describe`]) after the finish.
pub(super) fn answer_line(
    id: &str,
    f: &Facts,
    o: &Outcome<'_>,
    finish_reason: &str,
    store: &str,
    details: &str,
) -> String {
    format!(
        "{}: {} prompt tokens ({} cached{}{}{}{}){}, {} generated, prefix {:.2}s, decode {:.2}s ({:.1} tok/s){}, finish={finish_reason}{}{}, {store}{details}",
        id,
        f.n,
        f.reused,
        describe_reuse(f.cut_back, f.forked),
        from_disk(f.from_disk),
        if f.agreement > f.reused {
            format!(", agreement {}", f.agreement)
        } else {
            String::new()
        },
        f.durable
            .map(|(b, secs)| format!(", durable prefix {b} written in {secs:.2}s"))
            .unwrap_or_default(),
        if f.images == 0 {
            String::new()
        } else {
            format!(
                ", images {} ({} tokens{}{}, prepared in {:.3}s), tower {:.2}s",
                f.images,
                f.image_tokens,
                if f.encoded_images < f.images {
                    format!(", {} encoded", f.encoded_images)
                } else {
                    String::new()
                },
                if f.images_decoded < f.images {
                    format!(", {} decoded", f.images_decoded)
                } else {
                    String::new()
                },
                f.prepare_secs,
                f.vision_secs,
            )
        },
        o.generated,
        f.prefix_secs,
        o.decode_secs,
        o.generated as f64 / o.decode_secs.max(1e-9),
        format_args!(
            "{}{}",
            match o.speculation.filter(|s| s.drafted > 0) {
                Some(s) => format!(", drafts {}/{} accepted", s.accepted, s.drafted),
                None => String::new(),
            },
            if o.decode_checkpoints > 0 {
                format!(
                    ", {} decode checkpoints in {:.2}s",
                    o.decode_checkpoints, o.decode_checkpoint_secs
                )
            } else {
                String::new()
            }
        ),
        o.cancelled_by
            .map(|by| format!(" (cancelled by the {by} during the decode)"))
            .unwrap_or_default(),
        o.batch.map(|b| b.describe(o.generated)).unwrap_or_default(),
    )
}

/// What the end of an answered request's response is made of.
pub(super) struct Closing<'a> {
    pub(super) kind: Kind,
    pub(super) id: &'a str,
    pub(super) created: u64,
    pub(super) model: &'a str,
    pub(super) finish_reason: &'a str,
    /// The `usage` object.
    pub(super) usage: Value,
    /// The `timings` object.
    pub(super) timings: Value,
}

/// A streamed answer's last events before `data: [DONE]`. `timings` rides
/// the last chunk the stream already sends: the usage chunk when the client
/// asked for one (`include_usage`), the finish chunk otherwise. No client
/// ever sees a chunk shape it did not already get, and the extension costs
/// nothing to those that ignore it.
pub(super) fn stream_end(c: Closing<'_>, include_usage: bool) -> Vec<Value> {
    let mut events = Vec::with_capacity(2);
    let mut last = if c.kind == Kind::Chat {
        chunk(c.id, c.created, c.model, json!({}), Some(c.finish_reason))
    } else {
        text_chunk(c.id, c.created, c.model, "", Some(c.finish_reason))
    };
    if include_usage {
        events.push(last);
        last = json!({
            "id": c.id,
            "object": if c.kind == Kind::Chat { "chat.completion.chunk" } else { "text_completion" },
            "created": c.created,
            "model": c.model,
            "choices": [],
            "usage": c.usage,
        });
    }
    last["timings"] = c.timings;
    events.push(last);
    events
}

/// A non-streamed answer's body, from the text and tool calls `collected`.
pub(super) fn response_body(c: Closing<'_>, collected: Collected) -> Value {
    if c.kind == Kind::Chat {
        let mut message = json!({"role": "assistant", "content": collected.content});
        if !collected.reasoning.is_empty() {
            message["reasoning_content"] = Value::String(collected.reasoning);
        }
        if !collected.tool_calls.is_empty() {
            if collected.content.is_empty() {
                message["content"] = Value::Null;
            }
            message["tool_calls"] = Value::Array(
                collected
                    .tool_calls
                    .iter()
                    .enumerate()
                    .map(|(i, call)| {
                        json!({
                            "id": call_id(c.id, i),
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        })
                    })
                    .collect(),
            );
        }
        json!({
            "id": c.id,
            "object": "chat.completion",
            "created": c.created,
            "model": c.model,
            "choices": [{"index": 0, "message": message, "finish_reason": c.finish_reason}],
            "usage": c.usage,
            "timings": c.timings,
        })
    } else {
        json!({
            "id": c.id,
            "object": "text_completion",
            "created": c.created,
            "model": c.model,
            "choices": [{"index": 0, "text": collected.content, "finish_reason": c.finish_reason, "logprobs": null}],
            "usage": c.usage,
            "timings": c.timings,
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/serve/request.rs"]
mod tests;
