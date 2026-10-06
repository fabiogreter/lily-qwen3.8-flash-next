//! One request's way through the engine, shared by both ways of serving
//! it: one request at a time ([`Engine::serve`]) and the batch scheduler
//! ([`Engine::serve_batched`]). A request is taken on ([`Engine::open`]),
//! admitted ([`Engine::admit`]: everything before its decode, which either
//! answers it there or hands back an [`Admitted`] request), decoded by its
//! caller, answered ([`Engine::answer`]) and closed ([`Engine::close`]).
//! The two ways differ only in how the prefill's chunks run
//! ([`PrefillChunks`]: straight through, or between batched decode steps
//! of the requests already decoding) and in how the decode runs (the
//! production loop, or the scheduler's rows); a change to admission or to
//! the answer is made here, once.
//!
//! [`Facts`] is what admission measured; [`Outcome`] how the decode went.
//! From the two come the request's `timings` ([`timings`]), its log line
//! ([`answer_line`], or [`stopped_line`] for a prefill that was stopped)
//! and the end of its response ([`stream_end`], [`response_body`]).

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};

use super::api::{self, Kind, Prepared};
use super::batch::BatchStats;
use super::pin::InFlight;
use super::session::{
    CachedImage, DecodeCheckpoints, Evictions, Session, SessionStore,
    boundary_position, last_user_turn,
};
use super::stream::{Event, OutputParser, ParserConfig};
use super::timings::{
    EvictionPhase, EvictionTimings, MemoryStats, NgramStats, PrefillParts,
    PrefillPhases, Speculation, Timings, TimingsEntry,
};
use super::{
    Collected, Engine, Job, Sink, call_id, chunk, describe_reuse, error_json, now,
    print_kernel_profile, response_id, text_chunk, user_turn_opener,
};
use crate::engine::{DecodeStateApi, LanguageModel};
use crate::generate::{FinishReason, Generator};
use crate::qwen4exp::{ImageEmbeds, VisionInput, positions_for_prompt};
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

// --- a request's way in ------------------------------------------------------

/// How admission runs a request's prefill: [`Alone`] for one request at a
/// time, the batch scheduler's `Interleaved` with batched decode steps of
/// the requests already decoding between its chunks. Everything else about
/// admission is the same for both.
pub(super) trait PrefillChunks<M: LanguageModel> {
    /// Prefills `tokens` into `session` at its position, stopping early
    /// when `stop` says so before a chunk; returns the tokens fed.
    fn prefill(
        &mut self,
        engine: &mut Engine<M>,
        session: &mut Session<M>,
        tokens: &[u32],
        vision: Option<&VisionInput<'_>>,
        stop: &dyn Fn() -> bool,
    ) -> Result<usize>;

    /// Bytes of the sessions other requests hold checked out, which the
    /// store cannot evict: it makes room among the others at the acquire
    /// and at a stopped prefill's release (0 for one request at a time).
    fn in_flight_bytes(&self) -> usize;

    /// What sharing the GPU did for the request so far, for a stopped
    /// prefill's timings; `None` without batching.
    fn batch(&self) -> Option<&BatchStats>;
}

/// One request at a time: the prefill runs straight through, chunk after
/// chunk, and no other session is in flight.
pub(super) struct Alone;

impl<M: LanguageModel> PrefillChunks<M> for Alone {
    fn prefill(
        &mut self,
        engine: &mut Engine<M>,
        session: &mut Session<M>,
        tokens: &[u32],
        vision: Option<&VisionInput<'_>>,
        stop: &dyn Fn() -> bool,
    ) -> Result<usize> {
        engine.model.prefill_until(
            &engine.ctx,
            &mut session.state,
            &mut engine.scratch,
            tokens,
            vision,
            stop,
        )
    }

    fn in_flight_bytes(&self) -> usize {
        0
    }

    fn batch(&self) -> Option<&BatchStats> {
        None
    }
}

impl<M: LanguageModel> Engine<M> {
    /// Everything before a request's decode: the session cache lookup, the
    /// vision tower, the durable prefix entry, the prefill up to the last
    /// prompt token (its chunks run by `prefill`) and the checkpoint there,
    /// then the response's id, parser and, when streamed, its first chunk.
    /// `queued` is how long the request waited for the engine (a reload and
    /// the pin included), which its timings report next to the prefill,
    /// with whether the weights were `pinned`.
    ///
    /// Returns `None` when the request was answered here: a departed client
    /// or the end of the shutdown's grace stopped its prefill, and the
    /// prefix fed so far stays a session for a retry.
    pub(super) fn admit<'g>(
        &mut self,
        p: Prepared,
        sink: &mut Sink,
        queued: Duration,
        pinned: bool,
        generator: &'g Generator,
        prefill: &mut impl PrefillChunks<M>,
    ) -> Result<Option<Admitted<'g, M>>> {
        // The HTTP thread validated against the same limit; this only guards
        // the engine's buffers if the two ever disagree.
        ensure!(
            p.prompt.len() < self.max_seq,
            "prompt exceeds the engine context: {} prompt tokens, {} tokens of context",
            p.prompt.len(),
            self.max_seq
        );
        let n = p.prompt.len();
        let images: Vec<CachedImage> =
            p.images.iter().map(|i| CachedImage::new(i.span, i.digest)).collect();
        let image_tokens: usize = p.images.iter().map(|i| i.span.len).sum();
        // The diagnostics' samples: the engine thread's counters and the
        // system's paging counters, here, at the end of the prefill and at
        // the end of the decode (a few microseconds each).
        let counters_start = crate::stats::counters();
        let vm_start = crate::stats::vm_counters();
        let started = Instant::now();
        // Sessions other requests hold are checked out: the store makes room
        // among the others.
        self.sessions.set_in_flight_bytes(prefill.in_flight_bytes());
        let acquired = self.sessions.acquire(
            &self.ctx,
            &self.model,
            &p.prompt,
            &images,
            p.cache_key.as_deref(),
        )?;
        let session_secs = started.elapsed().as_secs_f64();
        let mut session = acquired.session;
        let reused = acquired.reused;
        let agreement = acquired.agreement;
        ensure!(reused < n, "session cache returned the whole prompt");
        ensure!(
            reused <= agreement,
            "session cache resumed at {reused} past the agreement {agreement}"
        );

        // Rotary positions are a pure function of the prompt, so every
        // acquired session takes the prompt's delta here, whatever state it
        // was restored from (text gives 0). With images, the tower runs for
        // every image that has rows at or beyond the reused prefix; an image
        // entirely inside it is already in the caches, span and digest
        // matched by the cache lookup.
        let spans: Vec<_> = p.images.iter().map(|i| i.span).collect();
        let positions = if spans.is_empty() {
            None
        } else {
            Some(positions_for_prompt(&p.prompt, &spans).context("image positions")?)
        };
        session
            .state
            .set_rope_delta(positions.as_ref().map_or(0, |pos| pos.rope_delta))?;
        let vision_started = Instant::now();
        let mut encoded: Vec<(usize, crate::tensor::Tensor)> = Vec::new();
        for (k, image) in p.images.iter().enumerate() {
            if image.span.end() > reused {
                let pixels =
                    image.rows().with_context(|| format!("image {}", k + 1))?;
                let rows = self
                    .model
                    .encode_image(
                        &self.ctx,
                        &mut self.scratch,
                        &pixels,
                        image.span.grid_h,
                        image.span.grid_w,
                    )
                    .with_context(|| format!("vision tower over image {}", k + 1))?;
                encoded.push((k, rows));
            }
        }
        let vision_secs = vision_started.elapsed().as_secs_f64();
        let encoded_images = encoded.len();
        let embeds: Vec<ImageEmbeds<'_>> = encoded
            .iter()
            .map(|(k, rows)| ImageEmbeds { span: p.images[*k].span, rows })
            .collect();
        let vision = positions
            .as_ref()
            .map(|pos| VisionInput { positions: pos, images: &embeds });

        // A shared prefix the cache could not resume from becomes a durable
        // disk entry: prefill up to the boundary, write the caches and the
        // recurrent state there, then carry on. The prefill is split at the
        // boundary on purpose: chunks are 4 096 tokens and the kernels are
        // not row-count invariant, so a later run that resumes at the boundary
        // must prefill the rest in the same chunks this run did. The snapshot
        // is dropped, not kept as a checkpoint: durable entries live on disk
        // only (see the session module).
        let min_tokens = self.sessions.durable_min_tokens();
        // The boundary snaps back to the start of the last user message
        // opened before the agreement (see `boundary_position`).
        let boundary = self.sessions.disk().and_then(|_| {
            let user_turn = (min_tokens > 0 && agreement > reused)
                .then(|| {
                    let opener = user_turn_opener(generator.tokenizer());
                    last_user_turn(&p.prompt[..agreement.min(n)], &opener)
                })
                .flatten();
            boundary_position(
                agreement,
                reused,
                n,
                min_tokens,
                acquired.cut_back.is_some(),
                user_turn,
            )
        });
        // A client that went away (or the stop signal's grace running out)
        // stops the prefill at the next chunk boundary, before that chunk is
        // committed, instead of holding the engine for the rest of it.
        let shutdown = self.shutdown.clone();
        let cancelled = sink.cancelled.clone();
        let stop = || cancelled.load(Ordering::Relaxed) || shutdown.cancel();
        let mut cancelled_at: Option<usize> = None;
        let mut durable: Option<(usize, f64)> = None;
        let mut durable_secs = 0.0;
        let mut prefilled = reused;
        if let Some(b) = boundary {
            if prefilled < b {
                prefilled += prefill.prefill(
                    self,
                    &mut session,
                    &p.prompt[prefilled..b],
                    vision.as_ref(),
                    &stop,
                )?;
            }
            // A stopped request writes no durable entry: one it never wrote
            // is one fewer than a finished request would have, never more.
            if prefilled < b || stop() {
                cancelled_at = Some(prefilled);
            } else {
                let write_started = Instant::now();
                let snapshot = session.state.snapshot(&self.ctx)?;
                match self.sessions.store_durable(
                    &p.prompt[..b],
                    &images,
                    p.cache_key.as_deref(),
                    &session.state,
                    &snapshot,
                ) {
                    Ok(Some(_)) => {
                        durable = Some((b, write_started.elapsed().as_secs_f64()))
                    }
                    Ok(None) => eprintln!(
                        "session cache: the disk tier did not keep the durable prefix at {b}"
                    ),
                    Err(error) => eprintln!(
                        "session cache: writing the durable prefix at {b} failed: {error:#}"
                    ),
                }
                drop(snapshot);
                durable_secs = write_started.elapsed().as_secs_f64();
            }
        }

        // Prefix up to the last prompt token, then checkpoint there so an
        // identical or extended prompt can resume without re-feeding it.
        if cancelled_at.is_none() {
            if prefilled < n - 1 {
                prefilled += prefill.prefill(
                    self,
                    &mut session,
                    &p.prompt[prefilled..n - 1],
                    vision.as_ref(),
                    &stop,
                )?;
            }
            if prefilled < n - 1 || stop() {
                cancelled_at = Some(prefilled);
            }
        }
        if let Some(at) = cancelled_at {
            let prefix_secs = started.elapsed().as_secs_f64();
            // The prefix fed so far stays an ordinary session whose live end
            // is the chunk boundary, so a retry of the prompt resumes there.
            // It is released like any request's session: under the budget
            // (evictions write ahead or spill as usual), never written ahead
            // itself while it is the latest, its checkpoints those it had,
            // all at or below the position it was acquired at (the live end
            // needs none). An image the stop cut through is left out of the
            // lineage's spans, so no prompt resumes inside it.
            session.stop_at(&p.prompt, reused, at)?;
            self.sessions.set_in_flight_bytes(prefill.in_flight_bytes());
            let released = self.sessions.release(
                &self.ctx,
                session,
                &images,
                p.cache_key.as_deref(),
            );
            let counters_end = crate::stats::counters();
            let vm_end = crate::stats::vm_counters();
            let created = now();
            let id = response_id(p.kind, created, &mut self.next_id);
            let by = if sink.cancelled() { "client" } else { "shutdown" };
            // The prefill ended at the stop: its samples are the stop's.
            let facts = Facts {
                n,
                reused,
                agreement,
                cut_back: acquired.cut_back,
                forked: acquired.forked,
                from_disk: acquired.from_disk,
                durable,
                durable_secs,
                images: p.images.len(),
                images_decoded: p.images_decoded,
                prepare_secs: p.prepare_secs,
                image_tokens,
                vision_secs,
                encoded_images,
                queued_secs: queued.as_secs_f64(),
                pinned,
                session_secs,
                checkpoint_secs: 0.0,
                prefix_secs,
                acquire_evictions: acquired.evictions,
                counters_start,
                counters_prefill: counters_end,
                vm_start,
                vm_prefill: vm_end,
            };
            let measured = timings(
                &facts,
                &Outcome {
                    cancelled_by: Some(by),
                    cancelled_at: Some(at),
                    batch: prefill.batch(),
                    ..Outcome::default()
                },
                &released,
                counters_end,
                vm_end,
                crate::stats::pressure_level(),
                crate::stats::task_memory(),
            );
            eprintln!(
                "{}",
                stopped_line(
                    &id,
                    &facts,
                    by,
                    at,
                    &describe_store(&self.sessions),
                    &measured.log_details()
                )
            );
            self.timings.record(TimingsEntry {
                id,
                model: M::MODEL_ID,
                created,
                timings: measured,
            });
            if !sink.cancelled() {
                // Stopped by the server: the client is still there to hear it.
                sink.start(503, "application/json");
                sink.send(error_json("server_error", "the server is shutting down"));
            }
            return Ok(None);
        }
        // The last prompt token is fed by the generator through the text
        // prefill: it is text after every image, and the state's rope delta
        // places it. The image rows are no longer needed.
        drop(embeds);
        drop(encoded);
        let checkpoint_started = Instant::now();
        let snapshot = session.state.snapshot(&self.ctx)?;
        session.add_checkpoint(snapshot);
        let checkpoint_secs = checkpoint_started.elapsed().as_secs_f64();
        let prefix_secs = started.elapsed().as_secs_f64();
        let counters_prefill = crate::stats::counters();
        let vm_prefill = crate::stats::vm_counters();

        // A long shared prefix that nothing could resume from: show the seam
        // once, as the text either side of it in this prompt and what the
        // cached lineage continued with. This is how a client that renders
        // the same preamble differently between runs is found at a glance.
        // Unlike the durable boundary, the threshold here is absolute: a
        // short fork inside one conversation (a re-rendered tool call, a
        // re-tokenized answer) is exactly what this line should surface.
        // An ordinary hit (`agreement == reused`) diverges too, at the user's
        // message, and says nothing worth a line of prompt text in the log.
        if min_tokens > 0
            && agreement >= min_tokens
            && agreement > reused
            && agreement < n - 1
        {
            let tokenizer = generator.tokenizer();
            let text = |ids: &[u32]| {
                tokenizer
                    .decode(ids, false)
                    .unwrap_or_else(|e| format!("<undecodable: {e}>"))
            };
            let window = 12;
            eprintln!(
                "divergence at {agreement}: prompt {:?} | {:?}, cached lineage continued {:?}",
                text(&p.prompt[agreement.saturating_sub(window)..agreement]),
                text(&p.prompt[agreement..(agreement + window).min(n)]),
                text(
                    &acquired.divergent_tail
                        [..acquired.divergent_tail.len().min(window)]
                ),
            );
        }

        let created = now();
        let id = response_id(p.kind, created, &mut self.next_id);
        let tokenizer = generator.tokenizer();
        let detokenize: Detokenize<'g> =
            Box::new(move |ids: &[u32]| tokenizer.decode(ids, false));
        let parser = OutputParser::new(
            detokenize,
            ParserConfig {
                thinking_open: p.thinking_open,
                tools: p.tools.clone(),
                stop_strings: p.stop_strings.clone(),
                raw: p.kind == Kind::Completion,
            },
        );
        let out = Output {
            id,
            created,
            model: M::MODEL_ID,
            kind: p.kind,
            stream: p.stream,
            collected: Collected::default(),
            tool_index: 0,
        };
        if p.stream {
            sink.start(200, "text/event-stream");
            if p.kind == Kind::Chat {
                sink.sse(&chunk(
                    &out.id,
                    created,
                    M::MODEL_ID,
                    json!({"role": "assistant", "content": ""}),
                    None,
                ));
            }
        }
        let facts = Facts {
            n,
            reused,
            agreement,
            cut_back: acquired.cut_back,
            forked: acquired.forked,
            from_disk: acquired.from_disk,
            durable,
            durable_secs,
            images: p.images.len(),
            images_decoded: p.images_decoded,
            prepare_secs: p.prepare_secs,
            image_tokens,
            vision_secs,
            encoded_images,
            queued_secs: queued.as_secs_f64(),
            pinned,
            session_secs,
            checkpoint_secs,
            prefix_secs,
            acquire_evictions: acquired.evictions,
            counters_start,
            counters_prefill,
            vm_start,
            vm_prefill,
        };
        Ok(Some(Admitted {
            p,
            session,
            images,
            facts,
            // Counted from the checkpoint the prefill just took.
            decode_checkpoints: self.sessions.decode_checkpoints(n - 1),
            parser,
            out,
        }))
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

impl<M: LanguageModel> Admitted<'_, M> {
    /// GPU bytes the request holds checked out: its session and the decode
    /// checkpoints it took so far, which join the session only at the
    /// answer. One checkpoint is a whole recurrent snapshot (about 113 MB
    /// with the full model), so a long-running row undercounted by them can
    /// let admission and the store overcommit by hundreds of MB.
    pub(super) fn bytes(&self) -> usize {
        self.session.bytes() + self.decode_checkpoints.bytes()
    }
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

        // Every pass recorded since the last answer: under batching, other
        // requests' steps and prefills included.
        if self.ctx.profiling() {
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
