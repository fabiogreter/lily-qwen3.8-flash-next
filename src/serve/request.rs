//! One request's way through the engine, shared by both ways of serving
//! it: one request at a time ([`Engine::serve`]) and the batch scheduler
//! ([`Engine::serve_batched`]). A request is taken on ([`Engine::open`]),
//! decoded by its caller, answered ([`Engine::answer`]) and closed
//! ([`Engine::close`]). [`Facts`] is what its admission measured;
//! [`Outcome`] how its decode went. From the two come its `timings`
//! ([`timings`]), its log line ([`answer_line`], or [`stopped_line`] for a
//! prefill that was stopped) and the end of its response ([`stream_end`],
//! [`response_body`]).

use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use serde_json::{Value, json};

use super::api::{self, Kind, Prepared};
use super::batch::BatchStats;
use super::pin::InFlight;
use super::session::{
    CachedImage, DecodeCheckpoints, Evictions, Session, SessionStore,
};
use super::stream::{Event, OutputParser};
use super::timings::{
    EvictionPhase, EvictionTimings, MemoryStats, NgramStats, PrefillParts,
    PrefillPhases, Speculation, Timings, TimingsEntry,
};
use super::{
    Collected, Engine, Job, Sink, call_id, chunk, describe_reuse, error_json,
    print_kernel_profile, text_chunk,
};
use crate::engine::{DecodeStateApi, LanguageModel};
use crate::generate::FinishReason;
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

// --- a request's way out -----------------------------------------------------

pub(super) type Detokenize<'g> = Box<dyn FnMut(&[u32]) -> Result<String> + 'g>;
/// A request's output parser, detokenizing with the generator's tokenizer.
pub(super) type Parser<'g> = OutputParser<Detokenize<'g>>;

/// Where a request's output goes: SSE chunks or the collected answer.
pub(super) struct Output {
    pub(super) id: String,
    pub(super) created: u64,
    pub(super) model: &'static str,
    pub(super) kind: Kind,
    pub(super) stream: bool,
    pub(super) collected: Collected,
    pub(super) tool_index: usize,
}

impl Output {
    /// Sends `events` to the client as SSE chunks, or collects them for the
    /// body when the request is not streamed.
    pub(super) fn deliver(&mut self, events: Vec<Event>, sink: &Sink) {
        let (id, created, model) = (self.id.as_str(), self.created, self.model);
        for event in events {
            match event {
                Event::Reasoning(text) => {
                    if self.stream {
                        sink.sse(&chunk(
                            id,
                            created,
                            model,
                            json!({"reasoning_content": text}),
                            None,
                        ));
                    } else {
                        self.collected.reasoning.push_str(&text);
                    }
                }
                Event::Content(text) => {
                    if self.stream {
                        if self.kind == Kind::Chat {
                            sink.sse(&chunk(
                                id,
                                created,
                                model,
                                json!({"content": text}),
                                None,
                            ));
                        } else {
                            sink.sse(&text_chunk(id, created, model, &text, None));
                        }
                    } else {
                        self.collected.content.push_str(&text);
                    }
                }
                Event::ToolCall(call) => {
                    if self.stream {
                        let delta = json!({"tool_calls": [{
                            "index": self.tool_index,
                            "id": call_id(id, self.tool_index),
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }]});
                        sink.sse(&chunk(id, created, model, delta, None));
                    } else {
                        self.collected.tool_calls.push(call);
                    }
                    self.tool_index += 1;
                }
            }
        }
    }
}

/// A request brought to its decode: its session holds the prompt but its
/// last token, checkpointed there, and the response has its id (and, when
/// streamed, its first chunk). Its caller draws the tokens and hands it
/// to [`Engine::answer`].
pub(super) struct Admitted<'g, M: LanguageModel> {
    pub(super) p: Prepared,
    pub(super) session: Session<M>,
    /// The prompt's images as the session cache keys them.
    pub(super) images: Vec<CachedImage>,
    pub(super) facts: Facts,
    /// The decode checkpoints, counted from the prefill's checkpoint.
    pub(super) decode_checkpoints: DecodeCheckpoints<M::State>,
    pub(super) parser: Parser<'g>,
    pub(super) out: Output,
}

/// How an admitted request was decoded, for [`Engine::answer`].
pub(super) struct Decoded<'a> {
    /// Every token drawn; the state holds all but the last, or all of them
    /// when a parked step consumed the last one.
    pub(super) tokens: &'a [u32],
    pub(super) finish: FinishReason,
    /// Draft tokens proposed and accepted.
    pub(super) drafted: usize,
    pub(super) accepted: usize,
    /// Taken right before the first draw.
    pub(super) started: Instant,
    /// How it shared the GPU (the batch scheduler); `None` with batching off.
    pub(super) batch: Option<&'a BatchStats>,
}

/// A request the engine took on: in flight for the weights' pin until
/// `in_flight` drops, which [`Engine::close`] does after the response's end.
pub(super) struct Opened {
    pub(super) p: Prepared,
    pub(super) sink: Sink,
    pub(super) in_flight: InFlight,
    /// How long it waited for the engine (a reload and the pin included).
    pub(super) queued: Duration,
}

/// How a request's response ends, for [`Engine::close`].
pub(super) enum Ending<'e> {
    /// Answered, or nothing to say: the client left.
    Answered,
    /// Failed with this error; the response names the GPU fault the context
    /// recorded by then, if any.
    Failed(&'e anyhow::Error),
    /// Failed with this error under the fault its caller read once for
    /// several requests (a failed batched step fails all of its rows).
    FailedUnder(&'e anyhow::Error, Option<&'e str>),
}

/// Ends a response with an error: an error body when nothing was sent yet,
/// an error event and the end of the stream when an event stream started.
fn error_end(sink: &mut Sink, stream: bool, status: u16, message: &str) {
    if !sink.started {
        sink.start(status, "application/json");
        sink.send(error_json("server_error", message));
    } else if stream {
        sink.sse(&json!({"error": {"message": message, "type": "server_error"}}));
        sink.send(b"data: [DONE]\n\n".to_vec());
    }
}

/// Answers a request that failed: logs the error and ends the response
/// with a 503 on a GPU fault (the engine reloads) or a 500.
pub(super) fn answer_failure(
    sink: &mut Sink,
    stream: bool,
    fault: Option<&str>,
    error: &anyhow::Error,
) {
    let (status, message) = match fault {
        Some(fault) => {
            eprintln!("request failed on a GPU fault (the engine reloads): {error:#}");
            (
                503,
                format!(
                    "the GPU command queue failed ({fault}); the engine is reloading, retry shortly"
                ),
            )
        }
        None => {
            eprintln!("request failed: {error:#}");
            (500, "internal server error".to_owned())
        }
    };
    error_end(sink, stream, status, &message);
    sink.end();
}

impl<M: LanguageModel> Engine<M> {
    /// Takes `job` on. Pinned before the prefill (the first request of an
    /// active period pays for it, inside its queue time). The request is in
    /// flight for the pin until the returned guard drops: a memory pressure
    /// warning meanwhile releases the pin then, not under the prefill or the
    /// decode.
    pub(super) fn open(&mut self, job: Job) -> Opened {
        let Job { prepared, sink, queued_at } = job;
        let in_flight = self.pin.before_request(Instant::now());
        let queued = queued_at.elapsed();
        Opened { p: prepared, sink, in_flight, queued }
    }

    /// Ends a request [`Self::open`] took on: the pin's hold counts from
    /// here, the response ends (with the error when it failed), and only
    /// then does the request leave flight, so a deferred release's `munlock`
    /// does not hold up the client (and happens before the caller can drop
    /// the engine). Returns the GPU fault the context recorded, if any.
    pub(super) fn close(
        &mut self,
        mut sink: Sink,
        in_flight: InFlight,
        stream: bool,
        ending: Ending<'_>,
    ) -> Option<String> {
        self.pin.after_request(Instant::now());
        let fault = self.ctx.fault();
        match ending {
            Ending::Answered => sink.end(),
            Ending::Failed(error) => {
                answer_failure(&mut sink, stream, fault.as_deref(), error)
            }
            Ending::FailedUnder(error, under) => {
                answer_failure(&mut sink, stream, under, error)
            }
        }
        drop(in_flight);
        fault
    }

    /// Answers a decoded request: the session goes back to the cache with
    /// what the decode fed and its decode checkpoints, then the timings,
    /// the log line and the end of the response. `in_flight_others` are
    /// the bytes of the sessions other requests hold checked out (0 for one
    /// request at a time).
    pub(super) fn answer(
        &mut self,
        admitted: Admitted<'_, M>,
        sink: &mut Sink,
        decoded: Decoded<'_>,
        in_flight_others: usize,
    ) -> Result<()> {
        let Admitted {
            p,
            mut session,
            images,
            facts: f,
            decode_checkpoints,
            mut parser,
            mut out,
        } = admitted;
        let final_events = parser.finish();
        out.deliver(final_events, sink);
        let decode_secs = decoded.started.elapsed().as_secs_f64();
        let counters_end = crate::stats::counters();
        let vm_end = crate::stats::vm_counters();

        // Bookkeeping: the state holds the prompt plus the generated tokens
        // that were fed: all but the last (drawn, never fed), or all of them
        // when a parked step consumed the final one.
        let n = f.n;
        let pos = session.state.pos();
        let fed_generated = pos
            .checked_sub(n)
            .ok_or_else(|| anyhow::anyhow!("decode state did not advance"))?;
        ensure!(
            fed_generated + 1 == decoded.tokens.len()
                || fed_generated == decoded.tokens.len(),
            "decode state at {pos} for {n} prompt tokens and {} drawn",
            decoded.tokens.len()
        );
        session.tokens.truncate(f.reused);
        session.tokens.extend_from_slice(&p.prompt[f.reused..]);
        session.tokens.extend_from_slice(&decoded.tokens[..fed_generated]);
        ensure!(
            session.state.pos() == session.tokens.len(),
            "session token/state position mismatch"
        );
        // Every decode checkpoint sits at a point the state passed: a prefix
        // of the tokens just recorded.
        let decode_checkpoints_taken = decode_checkpoints.taken();
        let decode_checkpoint_secs = decode_checkpoints.secs();
        session.add_decode_checkpoints(decode_checkpoints.into_snapshots(), &images)?;
        // The sessions other requests hold are checked out: the store makes
        // room among the others.
        self.sessions.set_in_flight_bytes(in_flight_others);
        let released =
            self.sessions.release(&self.ctx, session, &images, p.cache_key.as_deref());

        // TODO(batch): the batch scheduler never printed the kernel profile
        // (nor took the recorded passes); kept so here, fixed on its own.
        if decoded.batch.is_none() && self.ctx.profiling() {
            print_kernel_profile(&crate::metal::profile::take());
        }
        let completion_tokens = decoded.tokens.len();
        let finish_reason = match decoded.finish {
            FinishReason::Length => "length",
            _ if parser.tool_calls_emitted() > 0 => "tool_calls",
            _ => "stop",
        };
        // The numbers the line below prints, as JSON: attached to the
        // response below and kept for `GET /v1/timings`. Recorded before the
        // cancellation checks so the log and the ring buffer never disagree.
        // A request whose client left (or the server stopped) during the
        // decode is marked; its answer goes nowhere, or is a 503 below.
        let cancelled_by = if sink.cancelled() {
            Some("client")
        } else if self.shutdown.cancel() {
            Some("shutdown")
        } else {
            None
        };
        let outcome = Outcome {
            generated: completion_tokens,
            decode_secs,
            speculation: (self.drafts > 0).then_some(Speculation {
                drafted: decoded.drafted,
                accepted: decoded.accepted,
            }),
            decode_checkpoints: decode_checkpoints_taken,
            decode_checkpoint_secs,
            cancelled_by,
            cancelled_at: None,
            batch: decoded.batch,
        };
        let measured = timings(
            &f,
            &outcome,
            &released,
            counters_end,
            vm_end,
            crate::stats::pressure_level(),
            crate::stats::task_memory(),
        );
        eprintln!(
            "{}",
            answer_line(
                &out.id,
                &f,
                &outcome,
                finish_reason,
                &describe_store(&self.sessions),
                &measured.log_details()
            )
        );
        self.timings.record(TimingsEntry {
            id: out.id.clone(),
            model: M::MODEL_ID,
            created: out.created,
            timings: measured,
        });
        if sink.cancelled() {
            return Ok(());
        }
        if self.shutdown.cancel() {
            // Stopped by the server, not the client: say so instead of
            // handing out a truncated answer as a finished one.
            error_end(sink, p.stream, 503, "the server is shutting down");
            return Ok(());
        }
        let usage = api::usage(
            n,
            f.reused,
            completion_tokens,
            (p.kind == Kind::Chat).then(|| parser.reasoning_tokens()),
            outcome.speculation.filter(|s| s.drafted > 0),
        );
        let closing = Closing {
            kind: p.kind,
            id: &out.id,
            created: out.created,
            model: M::MODEL_ID,
            finish_reason,
            usage,
            timings: serde_json::to_value(measured)?,
        };
        if p.stream {
            for event in stream_end(closing, p.include_usage) {
                sink.sse(&event);
            }
            sink.send(b"data: [DONE]\n\n".to_vec());
        } else {
            let body = response_body(closing, out.collected);
            sink.start(200, "application/json");
            sink.send(serde_json::to_vec(&body)?);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/serve/request.rs"]
mod tests;
