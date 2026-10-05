//! Continuous batching across sessions (`--max-batch` above 1).
//!
//! With batching on, `engine_loop` hands each job to
//! [`Engine::serve_batched`], which serves it and every job that arrives
//! while anything is active, and returns once the last of them has its
//! answer. Up to `max_batch` requests decode together: each is a [`Row`],
//! one session in its decode phase, and a step over all rows is one GPU
//! pass ([`LanguageModel::decode_rows`]) that shares the weights' reads.
//!
//! The scheduler, per iteration: admit one waiting job when a row is free
//! (its prefill is exclusive on the GPU, one chunk at a time, with a few
//! batched steps of the running rows between chunks); with one row and no
//! job waiting, run that row in the production single-session loop
//! ([`Generator::resume`], speculation and parking included), stopped at its
//! next token when a job arrives; with two or more rows, one batched step.
//! A lone request is therefore served by exactly the passes `Engine::run`
//! would run for it. Rows decode plainly while they share a step; the draft
//! head is caught up on every row, so a row that is alone again speculates
//! with complete head caches.
//!
//! Admission and the answer mirror `Engine::run` (session lookup, the
//! tower, the durable boundary, cancellation, the checkpoint, timings, the
//! log line, the response). The two are kept separate so that batching off
//! is the unchanged code; a change to one usually needs the other.
//! `docs/continuous-batching-draft.md` has the design and its status.

use std::cell::Cell;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};

use super::api::{self, Kind, Prepared};
use super::session::{CachedImage, Evictions, Session, boundary_position};
use super::stream::{Event, OutputParser, ParserConfig};
use super::timings::{
    BatchTimings, EvictionPhase, EvictionTimings, MemoryStats, NgramStats,
    PrefillParts, PrefillPhases, Speculation, Timings, TimingsEntry,
};
use super::{
    Cmd, Collected, Engine, EngineQueue, Job, Sink, call_id, chunk, error_json, now,
    response_id, text_chunk,
};
use crate::engine::{BatchRow, CountsSlot, DecodeStateApi, Draw, LanguageModel};
use crate::generate::{FinishReason, GenerateOptions, Generator};
use crate::qwen4exp::{ImageEmbeds, VisionInput, positions_for_prompt};
use crate::stats::{Counters, VmCounters};

/// Batched decode steps of the running rows between two prefill chunks of a
/// request being admitted. A chunk is a 1.6 to 1.8 s pass the rows cannot
/// share; 8 steps (about 0.12 s at 15 ms a step) give them 5 tokens a
/// second meanwhile and cost the newcomer about 7 % of its prefill. More
/// favours the running rows, fewer the newcomer's time to first token.
pub(super) const DECODE_STEPS_PER_PREFILL_CHUNK: usize = 8;

/// Decode tokens the admission budgets a new session for beyond its prompt
/// (one capacity step of the caches; they grow on demand past it).
const ADMIT_GROWTH_TOKENS: usize = 8192;

// --- bookkeeping (host only, unit-tested) -----------------------------------

/// The batch slots: each has its own sampler scratch in the model, which
/// holds the penalty counts of the row that owns the slot.
#[derive(Debug)]
pub(super) struct Slots {
    free: Vec<bool>,
}

impl Slots {
    pub(super) fn new(n: usize) -> Self {
        Self { free: vec![true; n] }
    }

    /// The lowest free slot, now taken.
    pub(super) fn take(&mut self) -> Option<usize> {
        let slot = self.free.iter().position(|&f| f)?;
        self.free[slot] = false;
        Some(slot)
    }

    pub(super) fn give(&mut self, slot: usize) {
        if let Some(f) = self.free.get_mut(slot) {
            debug_assert!(!*f, "slot {slot} given back twice");
            *f = true;
        }
    }

    pub(super) fn in_use(&self) -> usize {
        self.free.iter().filter(|&&f| !f).count()
    }
}

/// Whether a request whose session is estimated at `estimate` bytes may
/// start next to `rows` running rows holding `in_flight` bytes: there must
/// be a free row, and its session must fit the cache budget beside the
/// sessions in flight (which the store cannot evict). With nothing running
/// a request always starts, as it does without batching.
pub(super) fn admits(
    rows: usize,
    max_rows: usize,
    in_flight: usize,
    estimate: usize,
    budget: usize,
) -> bool {
    rows == 0 || (rows < max_rows && in_flight.saturating_add(estimate) <= budget)
}

/// What the scheduler does once admission had its turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Next {
    /// Nothing is running: the batch period is over.
    Done,
    /// One row: the single-session loop, stopped at the next token when a
    /// job arrives if `preempt` (not while a job is held back for the
    /// budget, which could not start anyway).
    Solo { preempt: bool },
    /// Two rows or more: one batched step.
    Step,
}

pub(super) fn next_action(rows: usize, held: bool) -> Next {
    match rows {
        0 => Next::Done,
        1 => Next::Solo { preempt: !held },
        _ => Next::Step,
    }
}

/// How one request shared the GPU with others, for its timings and log line.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct BatchStats {
    /// Tokens drawn in batched steps (the rest came from the single-session
    /// loop).
    pub(super) batched_tokens: usize,
    /// Over those steps, the sum of their row counts (this row included).
    pub(super) rows_sum: usize,
    /// The most rows a step of this request had.
    pub(super) max_rows: usize,
    /// The other requests (scheduler sequence numbers) it shared a step with.
    pub(super) peers: Vec<u64>,
    /// Tokens drawn by the single-session loop.
    pub(super) solo_tokens: usize,
    /// Times its single-session loop was stopped for an arriving request.
    pub(super) preemptions: usize,
    /// Batched steps of other requests run between its prefill chunks, and
    /// their wall time (part of its prefill time).
    pub(super) interleaved_steps: usize,
    pub(super) interleaved_secs: f64,
}

impl BatchStats {
    /// One batched step of the rows `members` (sequence numbers), `own` among
    /// them.
    pub(super) fn record_step(&mut self, own: u64, members: &[u64]) {
        self.batched_tokens += 1;
        self.rows_sum += members.len();
        self.max_rows = self.max_rows.max(members.len());
        for &m in members {
            if m != own && !self.peers.contains(&m) {
                self.peers.push(m);
            }
        }
    }

    /// The `batch` object of the request's timings.
    pub(super) fn timings(&self) -> BatchTimings {
        BatchTimings {
            batched_tokens: self.batched_tokens,
            solo_tokens: self.solo_tokens,
            mean_rows: self.mean_rows(),
            max_rows: self.max_rows,
            shared_with: self.peers.len(),
            preemptions: self.preemptions,
            prefill_interleaved_steps: self.interleaved_steps,
            prefill_interleaved_ms: self.interleaved_secs * 1e3,
        }
    }

    /// Mean rows per batched step; `None` without one.
    pub(super) fn mean_rows(&self) -> Option<f64> {
        (self.batched_tokens > 0)
            .then(|| self.rows_sum as f64 / self.batched_tokens as f64)
    }

    /// The log line's group for a request of `generated` tokens: how many of
    /// them came from shared steps and with whom, how often its lone decode
    /// was stopped for a newcomer, and what other requests' steps added to
    /// its prefill. Empty when it never met another request.
    pub(super) fn describe(&self, generated: usize) -> String {
        let mut parts = Vec::new();
        if let Some(mean) = self.mean_rows() {
            parts.push(format!(
                "{}/{generated} tokens in steps of up to {} rows (mean {mean:.2}) shared with {} other request{}",
                self.batched_tokens,
                self.max_rows,
                self.peers.len(),
                if self.peers.len() == 1 { "" } else { "s" },
            ));
        }
        if self.preemptions > 0 {
            parts.push(format!(
                "lone decode stopped {} time{} for a new request",
                self.preemptions,
                if self.preemptions == 1 { "" } else { "s" },
            ));
        }
        if self.interleaved_steps > 0 {
            parts.push(format!(
                "prefill shared with {} decode steps ({:.2}s)",
                self.interleaved_steps, self.interleaved_secs
            ));
        }
        if parts.is_empty() {
            String::new()
        } else {
            format!(", batched: {}", parts.join(", "))
        }
    }
}

// --- rows --------------------------------------------------------------------

type Detokenize<'g> = Box<dyn FnMut(&[u32]) -> Result<String> + 'g>;
type Parser<'g> = OutputParser<Detokenize<'g>>;

/// Where a request's output goes: SSE chunks or the collected answer.
struct Output {
    id: String,
    created: u64,
    model: &'static str,
    kind: Kind,
    stream: bool,
    collected: Collected,
    tool_index: usize,
}

impl Output {
    /// `Engine::run`'s `deliver`.
    fn deliver(&mut self, events: Vec<Event>, sink: &Sink) {
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

/// What the admission measured, for the timings and the log line.
struct Facts {
    n: usize,
    reused: usize,
    agreement: usize,
    forked: bool,
    from_disk: Option<Duration>,
    durable: Option<(usize, f64)>,
    durable_secs: f64,
    image_tokens: usize,
    vision_secs: f64,
    encoded_images: usize,
    queued_secs: f64,
    pinned: bool,
    session_secs: f64,
    checkpoint_secs: f64,
    prefix_secs: f64,
    acquire_evictions: Evictions,
    counters_start: Counters,
    counters_prefill: Counters,
    vm_start: Option<VmCounters>,
    vm_prefill: Option<VmCounters>,
}

/// One request in its decode phase.
struct Row<'g, M: LanguageModel> {
    /// The scheduler's sequence number (for the sharing statistics).
    seq: u64,
    p: Prepared,
    sink: Sink,
    session: Session<M>,
    images: Vec<CachedImage>,
    facts: Facts,
    parser: Parser<'g>,
    out: Output,
    /// Every token drawn so far; the last one is not fed into the state yet
    /// (`state.pos() == prompt + generated.len() - 1`) whenever the row is at
    /// rest between steps.
    generated: Vec<u32>,
    /// Its batch slot; the slot's sampler holds its penalty counts except
    /// while it runs the single-session loop.
    slot: usize,
    decode_started: Instant,
    drafted: usize,
    accepted: usize,
    stats: BatchStats,
    finish: Option<FinishReason>,
}

impl<M: LanguageModel> Row<'_, M> {
    /// A draw of this row: what the production loops do with one (a stop
    /// token ends the row undelivered; otherwise the parser gets it, and a
    /// stop string, a departed client, the end of the shutdown grace or
    /// `max_tokens` end it).
    fn take_draw(
        &mut self,
        token: u32,
        generator: &Generator,
        shutdown_cancel: bool,
    ) -> Result<()> {
        self.generated.push(token);
        if generator.stop_tokens().contains(&token) {
            self.finish = Some(FinishReason::StopToken);
            return Ok(());
        }
        let events = self.parser.push(token)?;
        self.out.deliver(events, &self.sink);
        if self.sink.cancelled() || self.parser.stopped || shutdown_cancel {
            self.finish = Some(FinishReason::Callback);
        } else if self.generated.len() >= self.p.max_tokens {
            self.finish = Some(FinishReason::Length);
        }
        Ok(())
    }
}

/// A sink that goes nowhere: what the admission leaves behind once the row
/// took the request's sink.
fn detached_sink() -> Sink {
    let (tx, _) = std::sync::mpsc::channel();
    Sink {
        tx,
        cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        started: false,
    }
}

/// Answers a request that failed: `Engine::serve`'s error path.
fn answer_failure(
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
    if !sink.started {
        sink.start(status, "application/json");
        sink.send(error_json("server_error", message));
    } else if stream {
        sink.sse(&json!({"error": {"message": message, "type": "server_error"}}));
        sink.send(b"data: [DONE]\n\n".to_vec());
    }
    sink.end();
}

// --- the scheduler -----------------------------------------------------------

impl<M: LanguageModel> Engine<M> {
    /// Serves `first` and every job that arrives while anything is active,
    /// up to `max_batch` decoding together (see the module docs). Returns the
    /// GPU fault the context recorded, if any; every request still active
    /// then got a 503, and the engine must be replaced.
    pub(super) fn serve_batched(
        &mut self,
        first: Job,
        rx: &Receiver<Cmd>,
        queue: &EngineQueue,
    ) -> Option<String> {
        let _activity = crate::activity::Activity::begin("lily: serving requests");
        let generator = self.generator.clone();
        let mut rows: Vec<Row<'_, M>> = Vec::with_capacity(self.max_batch);
        let mut slots = Slots::new(self.max_batch);
        let mut held = Some(first);
        let mut seq = 0u64;
        while self.ctx.fault().is_none() {
            if held.is_none() && rows.len() < self.max_batch {
                held = self.next_job(rx, queue);
            }
            if let Some(job) = held.take() {
                if self.fits(&job, &rows) {
                    seq += 1;
                    self.admit(job, seq, &generator, &mut rows, &mut slots);
                    continue;
                }
                held = Some(job);
            }
            match next_action(rows.len(), held.is_some()) {
                Next::Done => break,
                Next::Solo { preempt } => {
                    self.run_solo(&generator, &mut rows, &mut slots, preempt, queue)
                }
                Next::Step => self.step_rows(&generator, &mut rows, &mut slots, 0),
            }
        }
        let fault = self.ctx.fault();
        if let Some(fault) = &fault {
            let error = anyhow::anyhow!("GPU fault while batching: {fault}");
            for row in rows.drain(..) {
                self.fail_row(row, &mut slots, Some(fault), &error);
            }
            if let Some(job) = held.take() {
                let mut sink = job.sink;
                answer_failure(&mut sink, job.prepared.stream, Some(fault), &error);
            }
        }
        debug_assert!(rows.is_empty() && held.is_none() && slots.in_use() == 0);
        self.sessions.set_in_flight_bytes(0);
        fault
    }

    /// The next job in the engine's channel, consuming the control messages
    /// before it as `engine_loop` would; `None` when there is none. A job
    /// that arrives after a stop request is refused.
    fn next_job(&self, rx: &Receiver<Cmd>, queue: &EngineQueue) -> Option<Job> {
        loop {
            match rx.try_recv() {
                Ok(Cmd::Job(job)) => {
                    queue.left();
                    if self.shutdown.requested() {
                        job.reject(503, "the server is shutting down");
                        continue;
                    }
                    return Some(*job);
                }
                // The GPU is busy: nothing to wake.
                Ok(Cmd::Arrival) => queue.arrival_taken(),
                Ok(Cmd::Wake) => {}
                Err(_) => return None,
            }
        }
    }

    /// Whether `job` may be admitted beside `rows` (see [`admits`]). Its
    /// session is estimated at the prompt plus [`ADMIT_GROWTH_TOKENS`] of
    /// decode (clients that leave `max_tokens` unset get the whole remaining
    /// context, which would keep every second request out), with one
    /// checkpoint.
    fn fits(&self, job: &Job, rows: &[Row<'_, M>]) -> bool {
        let growth = job.prepared.max_tokens.min(ADMIT_GROWTH_TOKENS);
        let tokens = (job.prepared.prompt.len() + growth).min(self.max_seq);
        let estimate = self.model.session_bytes(tokens, 1).unwrap_or(0) as usize;
        let in_flight = rows.iter().map(|r| r.session.bytes()).sum();
        admits(
            rows.len(),
            self.max_batch,
            in_flight,
            estimate,
            self.sessions.budget_bytes(),
        )
    }

    /// Bytes of the sessions in `rows`, which the store counts as in flight.
    fn in_flight(rows: &[Row<'_, M>]) -> usize {
        rows.iter().map(|r| r.session.bytes()).sum()
    }

    /// Runs the one row in the single-session loop until it finishes or, with
    /// `preempt`, a job arrives (it then stops at its next token, at rest).
    fn run_solo<'g>(
        &mut self,
        generator: &'g Generator,
        rows: &mut Vec<Row<'g, M>>,
        slots: &mut Slots,
        preempt: bool,
        queue: &EngineQueue,
    ) {
        let outcome = self.try_solo(generator, &mut rows[0], preempt, queue);
        match outcome {
            Ok(()) => {
                if rows[0].finish.is_some() {
                    let row = rows.remove(0);
                    self.finish_row(row, slots, 0);
                }
            }
            Err(error) => {
                let row = rows.remove(0);
                let fault = self.ctx.fault();
                self.fail_row(row, slots, fault.as_deref(), &error);
            }
        }
    }

    fn try_solo(
        &mut self,
        generator: &Generator,
        row: &mut Row<'_, M>,
        preempt: bool,
        queue: &EngineQueue,
    ) -> Result<()> {
        let Engine { ctx, model, scratch, shutdown, drafts, .. } = self;
        model.move_sampler_counts(
            ctx,
            scratch,
            CountsSlot::Batch(row.slot),
            CountsSlot::Engine,
        )?;
        let options = GenerateOptions {
            max_tokens: row.p.max_tokens,
            sampling: &row.p.sampling,
            stop_tokens: &[],
            drafts: *drafts,
        };
        let before = row.generated.len();
        let mut preempted = false;
        let resumed = {
            let Row { parser, out, sink, session, generated, .. } = row;
            let mut on_token = |token: u32| -> Result<bool> {
                let events = parser.push(token)?;
                out.deliver(events, sink);
                if sink.cancelled() || parser.stopped || shutdown.cancel() {
                    return Ok(false);
                }
                if preempt && queue.has_waiting() {
                    preempted = true;
                    return Ok(false);
                }
                Ok(true)
            };
            generator.resume(
                ctx,
                model,
                &mut session.state,
                scratch,
                generated,
                &options,
                None,
                &mut on_token,
            )?
        };
        row.drafted += resumed.drafted;
        row.accepted += resumed.accepted;
        row.stats.solo_tokens += row.generated.len() - before;
        if preempted && resumed.finish == FinishReason::Callback {
            row.stats.preemptions += 1;
            match resumed.parked_draw {
                // The parked step fed the last draw and drew the next one:
                // that is the row's next draw, as the loop would have used it.
                Some(token) => row.take_draw(token, generator, shutdown.cancel())?,
                None if row.generated.len() >= row.p.max_tokens => {
                    row.finish = Some(FinishReason::Length)
                }
                None => {}
            }
            if row.finish.is_none() {
                model.move_sampler_counts(
                    ctx,
                    scratch,
                    CountsSlot::Engine,
                    CountsSlot::Batch(row.slot),
                )?;
            }
        } else {
            row.finish = Some(resumed.finish);
        }
        Ok(())
    }

    /// One batched step over `rows`, then the rows that finished are answered.
    /// `extra_in_flight` is the session of a request being admitted, which
    /// the store must count when a finished row is released. A failed step
    /// fails every row in it.
    fn step_rows<'g>(
        &mut self,
        generator: &'g Generator,
        rows: &mut Vec<Row<'g, M>>,
        slots: &mut Slots,
        extra_in_flight: usize,
    ) {
        if rows.is_empty() {
            return;
        }
        if let Err(error) = self.try_step(generator, rows) {
            let fault = self.ctx.fault();
            for row in rows.drain(..) {
                self.fail_row(row, slots, fault.as_deref(), &error);
            }
            return;
        }
        let mut i = 0;
        while i < rows.len() {
            if rows[i].finish.is_some() {
                let row = rows.swap_remove(i);
                let others = Self::in_flight(rows) + extra_in_flight;
                self.finish_row(row, slots, others);
            } else {
                i += 1;
            }
        }
    }

    fn try_step(
        &mut self,
        generator: &Generator,
        rows: &mut [Row<'_, M>],
    ) -> Result<()> {
        // The GPU is idle between steps: a row at its capacity grows here.
        for row in rows.iter_mut() {
            let state = &mut row.session.state;
            if state.pos() >= state.capacity() {
                state.ensure_capacity(&self.ctx, state.pos() + 1)?;
            }
        }
        let members: Vec<u64> = rows.iter().map(|r| r.seq).collect();
        let draws = {
            let mut batch: Vec<BatchRow<'_, M::State>> = rows
                .iter_mut()
                .map(|r| BatchRow {
                    token: *r.generated.last().expect("a row has drawn"),
                    draw: Draw { params: &r.p.sampling, step: r.generated.len() },
                    slot: r.slot,
                    state: &mut r.session.state,
                })
                .collect();
            self.model.decode_rows(&self.ctx, &mut self.scratch, &mut batch)?
        };
        ensure!(
            draws.len() == rows.len(),
            "a batched step of {} rows drew {} tokens",
            rows.len(),
            draws.len()
        );
        let cancel = self.shutdown.cancel();
        for (row, token) in rows.iter_mut().zip(draws) {
            row.stats.record_step(row.seq, &members);
            row.take_draw(token, generator, cancel)?;
        }
        Ok(())
    }

    /// Drops a failed row's session (its state is not trustworthy) and
    /// answers it with the error.
    fn fail_row(
        &mut self,
        row: Row<'_, M>,
        slots: &mut Slots,
        fault: Option<&str>,
        error: &anyhow::Error,
    ) {
        let Row { mut sink, p, slot, .. } = row;
        slots.give(slot);
        self.pin.after_request(Instant::now());
        answer_failure(&mut sink, p.stream, fault, error);
    }

    // --- admission ----------------------------------------------------------

    /// Admits `job`: everything `Engine::run` does before its decode loop,
    /// with the prefill interleaved with batched steps of `rows`. The
    /// request becomes a row, or is answered here when it ends before its
    /// decode (cancelled, refused, failed, or finished by its first draw).
    fn admit<'g>(
        &mut self,
        job: Job,
        seq: u64,
        generator: &'g Generator,
        rows: &mut Vec<Row<'g, M>>,
        slots: &mut Slots,
    ) {
        let Job { prepared, mut sink, queued_at } = job;
        let stream = prepared.stream;
        let pinned = self.pin.before_request(Instant::now());
        let queued = queued_at.elapsed();
        if sink.cancelled() {
            self.pin.after_request(Instant::now());
            sink.end();
            return;
        }
        match self
            .try_admit(prepared, &mut sink, queued, pinned, seq, generator, rows, slots)
        {
            // The row holds the request's sink now; `sink` is detached.
            Ok(Some(row)) => match row.finish {
                Some(_) => {
                    let others = Self::in_flight(rows);
                    self.finish_row(row, slots, others);
                }
                None => rows.push(row),
            },
            // Answered (a cancelled prefill keeps its session, see there).
            Ok(None) => {
                self.pin.after_request(Instant::now());
                sink.end();
            }
            Err(error) => {
                self.pin.after_request(Instant::now());
                let fault = self.ctx.fault();
                answer_failure(&mut sink, stream, fault.as_deref(), &error);
            }
        }
    }

    /// `Engine::run` up to its first draw. A returned row has taken over
    /// `sink` (a detached one is left behind); `None` and errors leave it.
    #[allow(clippy::too_many_arguments)]
    fn try_admit<'g>(
        &mut self,
        p: Prepared,
        sink: &mut Sink,
        queued: Duration,
        pinned: bool,
        seq: u64,
        generator: &'g Generator,
        rows: &mut Vec<Row<'g, M>>,
        slots: &mut Slots,
    ) -> Result<Option<Row<'g, M>>> {
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
        let counters_start = crate::stats::counters();
        let vm_start = crate::stats::vm_counters();
        let started = Instant::now();
        // The running rows' sessions are checked out: the store makes room
        // among the others.
        self.sessions.set_in_flight_bytes(Self::in_flight(rows));
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

        // Positions and the tower, as in `Engine::run`.
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
                let merged = self
                    .model
                    .encode_image(
                        &self.ctx,
                        &mut self.scratch,
                        &pixels,
                        image.span.grid_h,
                        image.span.grid_w,
                    )
                    .with_context(|| format!("vision tower over image {}", k + 1))?;
                encoded.push((k, merged));
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

        let mut stats = BatchStats::default();
        let min_tokens = self.sessions.durable_min_tokens();
        let boundary = self
            .sessions
            .disk()
            .and_then(|_| boundary_position(agreement, reused, n, min_tokens));
        let shutdown = self.shutdown.clone();
        let cancelled = sink.cancelled.clone();
        let stop = || {
            cancelled.load(std::sync::atomic::Ordering::Relaxed) || shutdown.cancel()
        };
        let mut cancelled_at: Option<usize> = None;
        let mut durable: Option<(usize, f64)> = None;
        let mut durable_secs = 0.0;
        let mut prefilled = reused;
        if let Some(b) = boundary {
            if prefilled < b {
                prefilled += self.prefill_interleaved(
                    &mut session,
                    &p.prompt[prefilled..b],
                    vision.as_ref(),
                    &stop,
                    generator,
                    rows,
                    slots,
                    &mut stats,
                )?;
            }
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
        if cancelled_at.is_none() {
            if prefilled < n - 1 {
                prefilled += self.prefill_interleaved(
                    &mut session,
                    &p.prompt[prefilled..n - 1],
                    vision.as_ref(),
                    &stop,
                    generator,
                    rows,
                    slots,
                    &mut stats,
                )?;
            }
            if prefilled < n - 1 || stop() {
                cancelled_at = Some(prefilled);
            }
        }
        if let Some(at) = cancelled_at {
            // `Engine::run`'s cancelled prefill: the prefix stays a session.
            let prefix_secs = started.elapsed().as_secs_f64();
            session.stop_at(&p.prompt, reused, at)?;
            self.sessions.set_in_flight_bytes(Self::in_flight(rows));
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
            let measured = Timings::measure(n, reused, prefix_secs, 0, 0.0, None)
                .with_agreement(agreement, durable.map(|(b, _)| b))
                .with_vision(image_tokens, vision_secs)
                .with_pinned(pinned)
                .with_batch(stats.timings())
                .with_cancel(Some(by), Some(at))
                .with_evictions(EvictionTimings {
                    acquire: EvictionPhase::new(&acquired.evictions),
                    release: EvictionPhase::new(&released),
                })
                .with_diagnostics(
                    queued.as_secs_f64(),
                    PrefillPhases::split(
                        prefix_secs,
                        if p.images.is_empty() { 0.0 } else { vision_secs },
                        PrefillParts {
                            session_secs,
                            durable_secs,
                            checkpoint_secs: 0.0,
                            counters: counters_end.since(counters_start).prefill,
                        },
                    ),
                    NgramStats {
                        prefill: counters_end.since(counters_start).gather.into(),
                        decode: counters_end.since(counters_end).gather.into(),
                    },
                    MemoryStats::from_samples(
                        [vm_start, vm_end, vm_end],
                        crate::stats::pressure_level(),
                        crate::stats::task_memory(),
                    ),
                );
            eprintln!(
                "{id}: {n} prompt tokens ({reused} cached{}{}), cancelled by the {by} at {at} after {} prefilled in {prefix_secs:.2}s, kept {at} tokens as a session, sessions={} ({:.1}/{:.1} GB){}{}",
                if acquired.forked { ", forked" } else { "" },
                acquired
                    .from_disk
                    .map(|d| format!(", from disk in {:.2}s", d.as_secs_f64()))
                    .unwrap_or_default(),
                at - reused,
                self.sessions.len(),
                self.sessions.used_bytes() as f64 / 1e9,
                self.sessions.budget_bytes() as f64 / 1e9,
                self.sessions
                    .disk()
                    .map(|d| format!(
                        ", disk {} ({:.1} GB)",
                        d.len(),
                        d.used_bytes() as f64 / 1e9
                    ))
                    .unwrap_or_default(),
                measured.log_details(),
            );
            self.timings.record(TimingsEntry {
                id,
                model: M::MODEL_ID,
                created,
                timings: measured,
            });
            if !sink.cancelled() {
                sink.start(503, "application/json");
                sink.send(error_json("server_error", "the server is shutting down"));
            }
            return Ok(None);
        }
        drop(embeds);
        drop(encoded);
        let checkpoint_started = Instant::now();
        let snapshot = session.state.snapshot(&self.ctx)?;
        session.add_checkpoint(snapshot);
        let checkpoint_secs = checkpoint_started.elapsed().as_secs_f64();
        let prefix_secs = started.elapsed().as_secs_f64();
        let counters_prefill = crate::stats::counters();
        let vm_prefill = crate::stats::vm_counters();

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

        // The first draw (`Generator::generate`'s start), into the engine's
        // sampler, whose counts it resets; the row then takes a slot and the
        // counts go with it.
        let decode_started = Instant::now();
        let first = generator.begin(
            &self.ctx,
            &self.model,
            &mut session.state,
            &mut self.scratch,
            &p.prompt[n - 1..],
            &p.sampling,
        )?;
        let slot = slots.take().ok_or_else(|| {
            anyhow::anyhow!("no free batch slot for an admitted request")
        })?;
        let facts = Facts {
            n,
            reused,
            agreement,
            forked: acquired.forked,
            from_disk: acquired.from_disk,
            durable,
            durable_secs,
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
        // The row takes the request's sink; the caller keeps a detached one,
        // and gets the real one back if the first draw fails.
        let mut row = Row {
            seq,
            p,
            sink: std::mem::replace(sink, detached_sink()),
            session,
            images,
            facts,
            parser,
            out,
            generated: Vec::with_capacity(256),
            slot,
            decode_started,
            drafted: 0,
            accepted: 0,
            stats,
            finish: None,
        };
        let mut taken = row.take_draw(first, generator, self.shutdown.cancel());
        if taken.is_ok() && row.finish.is_none() {
            taken = self.model.move_sampler_counts(
                &self.ctx,
                &mut self.scratch,
                CountsSlot::Engine,
                CountsSlot::Batch(slot),
            );
        }
        if let Err(error) = taken {
            slots.give(slot);
            *sink = std::mem::replace(&mut row.sink, detached_sink());
            return Err(error);
        }
        Ok(Some(row))
    }

    /// Prefills `tokens` into `session` at its position, stopping early when
    /// `stop` says so before a chunk; returns the tokens fed. While `rows`
    /// decode, the prefill runs one chunk per call with
    /// [`DECODE_STEPS_PER_PREFILL_CHUNK`] batched steps of the rows between
    /// chunks. The chunk grid is that of one call (every call starts a chunk
    /// at its first token), so the numerics are an uninterleaved prefill's.
    #[allow(clippy::too_many_arguments)]
    fn prefill_interleaved<'g>(
        &mut self,
        session: &mut Session<M>,
        tokens: &[u32],
        vision: Option<&VisionInput<'_>>,
        stop: &dyn Fn() -> bool,
        generator: &'g Generator,
        rows: &mut Vec<Row<'g, M>>,
        slots: &mut Slots,
        stats: &mut BatchStats,
    ) -> Result<usize> {
        let mut fed = 0;
        while fed < tokens.len() {
            if rows.is_empty() {
                fed += self.model.prefill_until(
                    &self.ctx,
                    &mut session.state,
                    &mut self.scratch,
                    &tokens[fed..],
                    vision,
                    stop,
                )?;
                break;
            }
            // Let exactly one chunk through: the stop is asked before every
            // chunk, the first included.
            let first = Cell::new(true);
            let one_chunk = || stop() || !first.replace(false);
            let chunk = self.model.prefill_until(
                &self.ctx,
                &mut session.state,
                &mut self.scratch,
                &tokens[fed..],
                vision,
                &one_chunk,
            )?;
            fed += chunk;
            if chunk == 0 || stop() {
                break;
            }
            if fed < tokens.len() {
                let between = Instant::now();
                let own = session.bytes();
                for _ in 0..DECODE_STEPS_PER_PREFILL_CHUNK {
                    if rows.is_empty() {
                        break;
                    }
                    self.step_rows(generator, rows, slots, own);
                    stats.interleaved_steps += 1;
                }
                stats.interleaved_secs += between.elapsed().as_secs_f64();
            }
        }
        Ok(fed)
    }

    // --- the answer -----------------------------------------------------------

    /// Answers a row that finished (`Engine::run` after its decode loop):
    /// the session goes back to the cache, then timings, the log line and
    /// the response. `in_flight_others` are the bytes of the sessions other
    /// requests still hold.
    fn finish_row(
        &mut self,
        mut row: Row<'_, M>,
        slots: &mut Slots,
        in_flight_others: usize,
    ) {
        slots.give(row.slot);
        let stream = row.p.stream;
        let mut sink = std::mem::replace(&mut row.sink, detached_sink());
        let result = self.try_finish(row, &mut sink, in_flight_others);
        self.pin.after_request(Instant::now());
        match result {
            Ok(()) => sink.end(),
            Err(error) => {
                let fault = self.ctx.fault();
                answer_failure(&mut sink, stream, fault.as_deref(), &error);
            }
        }
    }

    fn try_finish(
        &mut self,
        mut row: Row<'_, M>,
        sink: &mut Sink,
        in_flight_others: usize,
    ) -> Result<()> {
        let final_events = row.parser.finish();
        row.out.deliver(final_events, sink);
        let decode_secs = row.decode_started.elapsed().as_secs_f64();
        let counters_end = crate::stats::counters();
        let vm_end = crate::stats::vm_counters();
        let f = &row.facts;
        let n = f.n;
        let p = &row.p;
        let finish = row.finish.unwrap_or(FinishReason::Callback);

        // The state holds the prompt and every draw but the last, or all of
        // them when a parked step consumed the last one.
        let pos = row.session.state.pos();
        let fed_generated = pos
            .checked_sub(n)
            .ok_or_else(|| anyhow::anyhow!("decode state did not advance"))?;
        ensure!(
            fed_generated + 1 == row.generated.len()
                || fed_generated == row.generated.len(),
            "decode state at {pos} for {n} prompt tokens and {} drawn",
            row.generated.len()
        );
        let mut session = row.session;
        session.tokens.truncate(f.reused);
        session.tokens.extend_from_slice(&p.prompt[f.reused..]);
        session.tokens.extend_from_slice(&row.generated[..fed_generated]);
        ensure!(
            session.state.pos() == session.tokens.len(),
            "session token/state position mismatch"
        );
        self.sessions.set_in_flight_bytes(in_flight_others);
        let released = self.sessions.release(
            &self.ctx,
            session,
            &row.images,
            p.cache_key.as_deref(),
        );

        let completion_tokens = row.generated.len();
        let finish_reason = match finish {
            FinishReason::Length => "length",
            _ if row.parser.tool_calls_emitted() > 0 => "tool_calls",
            _ => "stop",
        };
        let cancelled_by = if sink.cancelled() {
            Some("client")
        } else if self.shutdown.cancel() {
            Some("shutdown")
        } else {
            None
        };
        let speculation = (self.drafts > 0)
            .then_some(Speculation { drafted: row.drafted, accepted: row.accepted });
        let measured = Timings::measure(
            n,
            f.reused,
            f.prefix_secs,
            completion_tokens,
            decode_secs,
            speculation,
        )
        .with_agreement(f.agreement, f.durable.map(|(b, _)| b))
        .with_vision(f.image_tokens, f.vision_secs)
        .with_pinned(f.pinned)
        .with_batch(row.stats.timings())
        .with_cancel(cancelled_by, None)
        .with_evictions(EvictionTimings {
            acquire: EvictionPhase::new(&f.acquire_evictions),
            release: EvictionPhase::new(&released),
        })
        .with_diagnostics(
            f.queued_secs,
            PrefillPhases::split(
                f.prefix_secs,
                if p.images.is_empty() { 0.0 } else { f.vision_secs },
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
                crate::stats::pressure_level(),
                crate::stats::task_memory(),
            ),
        );
        let id = row.out.id.clone();
        let created = row.out.created;
        eprintln!(
            "{}: {} prompt tokens ({} cached{}{}{}{}){}, {} generated, prefix {:.2}s, decode {:.2}s ({:.1} tok/s){}, finish={finish_reason}{}{}, sessions={} ({:.1}/{:.1} GB){}{}",
            id,
            n,
            f.reused,
            if f.forked { ", forked" } else { "" },
            f.from_disk
                .map(|d| format!(", from disk in {:.2}s", d.as_secs_f64()))
                .unwrap_or_default(),
            if f.agreement > f.reused {
                format!(", agreement {}", f.agreement)
            } else {
                String::new()
            },
            f.durable
                .map(|(b, secs)| format!(", durable prefix {b} written in {secs:.2}s"))
                .unwrap_or_default(),
            if p.images.is_empty() {
                String::new()
            } else {
                format!(
                    ", images {} ({} tokens{}{}, prepared in {:.3}s), tower {:.2}s",
                    p.images.len(),
                    f.image_tokens,
                    if f.encoded_images < p.images.len() {
                        format!(", {} encoded", f.encoded_images)
                    } else {
                        String::new()
                    },
                    if p.images_decoded < p.images.len() {
                        format!(", {} decoded", p.images_decoded)
                    } else {
                        String::new()
                    },
                    p.prepare_secs,
                    f.vision_secs,
                )
            },
            completion_tokens,
            f.prefix_secs,
            decode_secs,
            completion_tokens as f64 / decode_secs.max(1e-9),
            if row.drafted > 0 {
                format!(", drafts {}/{} accepted", row.accepted, row.drafted)
            } else {
                String::new()
            },
            cancelled_by
                .map(|by| format!(" (cancelled by the {by} during the decode)"))
                .unwrap_or_default(),
            row.stats.describe(completion_tokens),
            self.sessions.len(),
            self.sessions.used_bytes() as f64 / 1e9,
            self.sessions.budget_bytes() as f64 / 1e9,
            self.sessions
                .disk()
                .map(|d| format!(
                    ", disk {} ({:.1} GB)",
                    d.len(),
                    d.used_bytes() as f64 / 1e9
                ))
                .unwrap_or_default(),
            measured.log_details(),
        );
        self.timings.record(TimingsEntry {
            id: id.clone(),
            model: M::MODEL_ID,
            created,
            timings: measured,
        });
        if sink.cancelled() {
            return Ok(());
        }
        if self.shutdown.cancel() {
            if !sink.started {
                sink.start(503, "application/json");
                sink.send(error_json("server_error", "the server is shutting down"));
            } else if p.stream {
                sink.sse(&json!({"error": {"message": "the server is shutting down", "type": "server_error"}}));
                sink.send(b"data: [DONE]\n\n".to_vec());
            }
            return Ok(());
        }
        let usage = api::usage(
            n,
            f.reused,
            completion_tokens,
            (p.kind == Kind::Chat).then(|| row.parser.reasoning_tokens()),
            (row.drafted > 0).then_some(Speculation {
                drafted: row.drafted,
                accepted: row.accepted,
            }),
        );
        let timings_json = serde_json::to_value(measured)?;
        if p.stream {
            let mut last = if p.kind == Kind::Chat {
                chunk(&id, created, M::MODEL_ID, json!({}), Some(finish_reason))
            } else {
                text_chunk(&id, created, M::MODEL_ID, "", Some(finish_reason))
            };
            if p.include_usage {
                sink.sse(&last);
                last = json!({
                    "id": id,
                    "object": if p.kind == Kind::Chat { "chat.completion.chunk" } else { "text_completion" },
                    "created": created,
                    "model": M::MODEL_ID,
                    "choices": [],
                    "usage": usage,
                });
            }
            last["timings"] = timings_json;
            sink.sse(&last);
            sink.send(b"data: [DONE]\n\n".to_vec());
        } else {
            let collected = std::mem::take(&mut row.out.collected);
            let body = if p.kind == Kind::Chat {
                let mut message =
                    json!({"role": "assistant", "content": collected.content});
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
                                    "id": call_id(&id, i),
                                    "type": "function",
                                    "function": {"name": call.name, "arguments": call.arguments},
                                })
                            })
                            .collect(),
                    );
                }
                json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": created,
                    "model": M::MODEL_ID,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                    "usage": usage,
                    "timings": timings_json,
                })
            } else {
                json!({
                    "id": id,
                    "object": "text_completion",
                    "created": created,
                    "model": M::MODEL_ID,
                    "choices": [{"index": 0, "text": collected.content, "finish_reason": finish_reason, "logprobs": null}],
                    "usage": usage,
                    "timings": timings_json,
                })
            };
            sink.start(200, "application/json");
            sink.send(serde_json::to_vec(&body)?);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/serve/batch.rs"]
mod tests;
