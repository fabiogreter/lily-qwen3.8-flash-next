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
//! next token when a job arrives; with two or more rows, a stretch of
//! batched steps until a row finishes or a waiting job could be admitted.
//! A lone request is therefore served by exactly the passes `Engine::run`
//! would run for it. Rows decode plainly while they share a step; the draft
//! head is caught up on every row, so a row that is alone again speculates
//! with complete head caches.
//!
//! Within a stretch the next step is committed parked behind the current
//! one ([`LanguageModel::park_rows`]) and released once the current one's
//! draws are taken, so the host's round trip between two steps overlaps
//! the GPU's work instead of following it ([`park_next`] says when). A
//! stretch returns with nothing in flight, so everything else that needs
//! the GPU, the batched scratch or a row's state runs between stretches.
//!
//! Admission and the answer are the ones `Engine::run` uses
//! (`serve/request.rs`): [`Engine::admit`] with [`Interleaved`] running the
//! prefill, which puts batched steps of the running rows between its chunks
//! where `Engine::run` prefills straight through, and [`Engine::answer`]
//! and [`Engine::close`] for a row that finished. What is the scheduler's
//! own is the first draw into a batch slot, the rows' decode, and how a
//! request shared the GPU (the `batch` object of its timings and the
//! "batched" group of its log line).
//! `docs/architecture.md`, "Continuous batching", has the design and what
//! it was measured to give.

use std::cell::Cell;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use anyhow::{Result, ensure};

use super::pin::InFlight;
use super::request::{self, Admitted, Decoded, Ending, Opened, PrefillChunks};
use super::session::Session;
use super::timings::BatchTimings;
use super::{Cmd, Engine, EngineQueue, Job, Sink};
use crate::engine::{
    BatchRow, CountsSlot, DecodeStateApi, Draw, LanguageModel, RowsInFlight,
};
use crate::generate::{DecodeCheckpointer, FinishReason, GenerateOptions, Generator};
use crate::qwen4exp::VisionInput;

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

/// What the scheduler knows of a row, right after committing a batched
/// step, when it decides whether to park the next one ([`park_next`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RowAhead {
    /// The position the committed step leaves the row at (its state's
    /// position, which counts the step's token as fed).
    pub(super) pos: usize,
    /// Tokens the row's caches hold before they have to grow.
    pub(super) capacity: usize,
    /// A decode checkpoint is due at `pos`.
    pub(super) checkpoint_due: bool,
    /// The committed step's draw will end the row whatever it is: it is the
    /// row's `max_tokens`-th, or the client has left.
    pub(super) ending: bool,
}

/// Whether to commit the next batched step parked behind the one just
/// committed, before that one's draws are read. Only between two steps of
/// the same rows in the same stretch: not when the stretch ends after this
/// step (`another_step` false: its step budget is spent), when a waiting
/// job could be admitted next (`admission`; its prefill needs the GPU and
/// the batched scratch), when the server's shutdown grace is over
/// (`cancel`: every row ends on this step's draw), or when any row ends on
/// this step's draw, has a decode checkpoint due where this step leaves it
/// (taken at rest, with nothing in flight), or would need its caches grown
/// for the step after the parked one (which cannot happen with a pass in
/// flight; `pos + 1 < capacity`, as the decode loop checks). A row that
/// finishes on a draw nobody saw coming (a stop token, a stop string) while
/// the next step is parked is the drain's business, not this one's.
pub(super) fn park_next(
    rows: &[RowAhead],
    another_step: bool,
    admission: bool,
    cancel: bool,
) -> bool {
    another_step
        && !admission
        && !cancel
        && rows.iter().all(|r| !r.checkpoint_due && !r.ending && r.pos + 1 < r.capacity)
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

/// One request in its decode phase.
struct Row<'g, M: LanguageModel> {
    /// The scheduler's sequence number (for the sharing statistics).
    seq: u64,
    /// The request as its admission left it. Its decode checkpoints are
    /// taken in the single-session loop and between batched steps alike.
    req: Admitted<'g, M>,
    sink: Sink,
    /// The request is in flight for the weight pin until this drops, after
    /// its response's end ([`Engine::close`]).
    in_flight: InFlight,
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
        let events = self.req.parser.push(token)?;
        self.req.out.deliver(events, &self.sink);
        if self.sink.cancelled() || self.req.parser.stopped || shutdown_cancel {
            self.finish = Some(FinishReason::Callback);
        } else if self.generated.len() >= self.req.p.max_tokens {
            self.finish = Some(FinishReason::Length);
        }
        Ok(())
    }
}

/// The rows of the batched step that feeds each row's last draw, or with
/// `ahead` 1 of the step parked behind it (whose token is not read: it is
/// that step's draw, still on the GPU).
fn batch_of<'r, M: LanguageModel>(
    rows: &'r mut [Row<'_, M>],
    ahead: usize,
) -> Vec<BatchRow<'r, M::State>> {
    rows.iter_mut()
        .map(|r| BatchRow {
            token: *r.generated.last().expect("a row has drawn"),
            draw: Draw { params: &r.req.p.sampling, step: r.generated.len() + ahead },
            slot: r.slot,
            state: &mut r.req.session.state,
        })
        .collect()
}

/// A batched step's draws, one per row in order, taken as the production
/// loops take a draw.
fn take_draws<M: LanguageModel>(
    rows: &mut [Row<'_, M>],
    draws: &[u32],
    members: &[u64],
    generator: &Generator,
    cancel: bool,
) -> Result<()> {
    ensure!(
        draws.len() == rows.len(),
        "a batched step of {} rows drew {} tokens",
        rows.len(),
        draws.len()
    );
    for (row, &token) in rows.iter_mut().zip(draws) {
        row.stats.record_step(row.seq, members);
        row.take_draw(token, generator, cancel)?;
    }
    Ok(())
}

/// At rest between steps (nothing in flight), each row holding every draw
/// but the last: a decode checkpoint due here is taken as the
/// single-session loop takes one, only for a row that goes on.
fn take_due_checkpoints<M: LanguageModel>(
    ctx: &crate::metal::MetalContext,
    rows: &mut [Row<'_, M>],
) -> Result<()> {
    for row in rows.iter_mut() {
        let pos = row.req.session.state.pos();
        if row.finish.is_none() && row.req.decode_checkpoints.due(pos) {
            row.req.decode_checkpoints.take(ctx, &row.req.session.state)?;
        }
    }
    Ok(())
}

/// The scheduler's prefill: while rows decode, one chunk at a time with
/// [`DECODE_STEPS_PER_PREFILL_CHUNK`] batched steps of the rows between
/// chunks (a row that finishes meanwhile is answered there); straight
/// through once none is left.
struct Interleaved<'a, 'g, M: LanguageModel> {
    generator: &'g Generator,
    rows: &'a mut Vec<Row<'g, M>>,
    slots: &'a mut Slots,
    /// The admitted request's sharing statistics so far (the steps between
    /// its chunks); its row takes them over.
    stats: BatchStats,
}

impl<M: LanguageModel> PrefillChunks<M> for Interleaved<'_, '_, M> {
    /// The chunk grid is that of one call (every call starts a chunk at its
    /// first token), so the numerics are an uninterleaved prefill's.
    fn prefill(
        &mut self,
        engine: &mut Engine<M>,
        session: &mut Session<M>,
        tokens: &[u32],
        vision: Option<&VisionInput<'_>>,
        stop: &dyn Fn() -> bool,
    ) -> Result<usize> {
        let mut fed = 0;
        while fed < tokens.len() {
            if self.rows.is_empty() {
                fed += engine.model.prefill_until(
                    &engine.ctx,
                    &mut session.state,
                    &mut engine.scratch,
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
            let chunk = engine.model.prefill_until(
                &engine.ctx,
                &mut session.state,
                &mut engine.scratch,
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
                // Stretches end where a row finishes (it is answered here),
                // and always drained: the next chunk has the GPU and the
                // batched scratch to itself. Nothing is admitted meanwhile.
                let mut steps = 0;
                while steps < DECODE_STEPS_PER_PREFILL_CHUNK && !self.rows.is_empty() {
                    let ran = engine.step_rows(
                        self.generator,
                        self.rows,
                        self.slots,
                        own,
                        DECODE_STEPS_PER_PREFILL_CHUNK - steps,
                        &|| false,
                    );
                    if ran == 0 {
                        break;
                    }
                    steps += ran;
                }
                self.stats.interleaved_steps += steps;
                self.stats.interleaved_secs += between.elapsed().as_secs_f64();
            }
        }
        Ok(fed)
    }

    /// The running rows' sessions.
    fn in_flight_bytes(&self) -> usize {
        Engine::in_flight(self.rows)
    }

    fn batch(&self) -> Option<&BatchStats> {
        Some(&self.stats)
    }
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
                    self.start_row(job, seq, &generator, &mut rows, &mut slots);
                    continue;
                }
                held = Some(job);
            }
            match next_action(rows.len(), held.is_some()) {
                Next::Done => break,
                Next::Solo { preempt } => {
                    self.run_solo(&generator, &mut rows, &mut slots, preempt, queue)
                }
                Next::Step => {
                    // A stretch of steps until a row finishes or a waiting
                    // job could be admitted. A job held back for the budget
                    // cannot start before a row finishes (the sessions in
                    // flight only grow), and with every row taken none can.
                    let can_admit = held.is_none() && rows.len() < self.max_batch;
                    let admission = || can_admit && queue.has_waiting();
                    self.step_rows(
                        &generator,
                        &mut rows,
                        &mut slots,
                        0,
                        usize::MAX,
                        &admission,
                    );
                }
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
                request::answer_failure(
                    &mut sink,
                    job.prepared.stream,
                    Some(fault),
                    &error,
                );
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
        let in_flight = rows.iter().map(|r| r.req.session.bytes()).sum();
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
        rows.iter().map(|r| r.req.session.bytes()).sum()
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
            max_tokens: row.req.p.max_tokens,
            sampling: &row.req.p.sampling,
            stop_tokens: &[],
            drafts: *drafts,
        };
        let before = row.generated.len();
        let mut preempted = false;
        let resumed = {
            let Row {
                req: Admitted { parser, out, session, decode_checkpoints, .. },
                sink,
                generated,
                ..
            } = row;
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
                Some(decode_checkpoints as &mut dyn DecodeCheckpointer<M::State>),
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
                None if row.generated.len() >= row.req.p.max_tokens => {
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

    /// A stretch of batched steps over `rows` (see [`Self::try_stretch`]),
    /// then the rows that finished are answered. Returns the steps it ran.
    /// `extra_in_flight` is the session of a request being admitted, which
    /// the store must count when a finished row is released. A failed step
    /// fails every row in it.
    fn step_rows<'g>(
        &mut self,
        generator: &'g Generator,
        rows: &mut Vec<Row<'g, M>>,
        slots: &mut Slots,
        extra_in_flight: usize,
        max_steps: usize,
        admission: &dyn Fn() -> bool,
    ) -> usize {
        if rows.is_empty() {
            return 0;
        }
        let steps = match self.try_stretch(generator, rows, max_steps, admission) {
            Ok(steps) => steps,
            Err(error) => {
                // A parked step dropped on the error path released itself;
                // this clears the model's record of it.
                let _ = self.model.release_parked(&self.scratch);
                let fault = self.ctx.fault();
                for row in rows.drain(..) {
                    self.fail_row(row, slots, fault.as_deref(), &error);
                }
                return 0;
            }
        };
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
        steps
    }

    /// Batched steps over `rows` until `max_steps` ran (at least one), a
    /// row finished, or `admission` says a waiting job could be admitted;
    /// returns the steps run. Between two steps of the stretch the next one
    /// is committed parked behind the current one when [`park_next`] allows
    /// it, and released once the current one's draws are taken. Whatever
    /// else needs the GPU, the batched scratch or the rows' states (an
    /// admission's prefill, a cache growth, a decode checkpoint, a finished
    /// row's answer, the single-session loop) runs only once the stretch
    /// returned, and it returns with nothing in flight: a step parked when a
    /// row finished is drained first, as an ordinary step for the rows that
    /// go on (the finished row's draw from it is dropped; it fed the row's
    /// final token, as the single-session loop's parked step does).
    fn try_stretch(
        &mut self,
        generator: &Generator,
        rows: &mut [Row<'_, M>],
        max_steps: usize,
        admission: &dyn Fn() -> bool,
    ) -> Result<usize> {
        // TODO(batch): speculation inside a batch. A row shares a step
        // plainly and speculates again once it is alone; batching the
        // verify pass (rows = the sum of 1 + drafts over sessions) needs a
        // per-session spec scratch (mid-step GDN states, conv inputs), one
        // GPU-side accepted count and control block per session in the draft
        // pass, per-session rollback, and parked next passes for every
        // combination of accepted counts. That is a second engine-shape
        // change, and the measured gain does not ask for it yet; see
        // docs/architecture.md, "Continuous batching".
        let Engine { ctx, model, scratch, shutdown, .. } = self;
        let parking = model.supports_rows_parking();
        let members: Vec<u64> = rows.iter().map(|r| r.seq).collect();
        let mut steps = 0;
        // The next step, committed parked behind the current one.
        let mut parked: Option<RowsInFlight<'_>> = None;
        loop {
            let current = match parked.take() {
                Some(mut step) => {
                    model.release_rows(scratch, &mut step, &mut batch_of(rows, 0))?;
                    step
                }
                None => {
                    // The GPU is idle between steps: a row at its capacity
                    // grows here.
                    for row in rows.iter_mut() {
                        let state = &mut row.req.session.state;
                        if state.pos() >= state.capacity() {
                            state.ensure_capacity(ctx, state.pos() + 1)?;
                        }
                    }
                    model.commit_rows(ctx, scratch, &mut batch_of(rows, 0), false)?
                }
            };
            steps += 1;
            if parking {
                let ahead: Vec<RowAhead> = rows
                    .iter_mut()
                    .map(|row| {
                        let state = &row.req.session.state;
                        let pos = state.pos();
                        RowAhead {
                            pos,
                            capacity: state.capacity(),
                            checkpoint_due: row.req.decode_checkpoints.due(pos),
                            ending: row.generated.len() + 1 >= row.req.p.max_tokens
                                || row.sink.cancelled(),
                        }
                    })
                    .collect();
                if park_next(&ahead, steps < max_steps, admission(), shutdown.cancel())
                {
                    parked = Some(model.park_rows(
                        ctx,
                        scratch,
                        &mut batch_of(rows, 1),
                        &current,
                        false,
                    )?);
                }
            }
            let (draws, _) = model.finish_rows(scratch, current)?;
            take_draws(rows, &draws, &members, generator, shutdown.cancel())?;
            let finished = rows.iter().any(|r| r.finish.is_some());
            match parked.take() {
                // Parked: nothing was due where this step left the rows, and
                // the parked step is the next one.
                Some(step) if !finished => parked = Some(step),
                Some(mut step) => {
                    // A row finished on this step's draw: drain. The parked
                    // step feeds every row's last draw, the finished rows'
                    // final ones included.
                    model.release_rows(scratch, &mut step, &mut batch_of(rows, 0))?;
                    let (draws, _) = model.finish_rows(scratch, step)?;
                    steps += 1;
                    ensure!(
                        draws.len() == rows.len(),
                        "a batched step of {} rows drew {} tokens",
                        rows.len(),
                        draws.len()
                    );
                    let cancel = shutdown.cancel();
                    for (row, token) in rows.iter_mut().zip(draws) {
                        if row.finish.is_none() {
                            row.stats.record_step(row.seq, &members);
                            row.take_draw(token, generator, cancel)?;
                        }
                    }
                    take_due_checkpoints(ctx, rows)?;
                    return Ok(steps);
                }
                None => {
                    take_due_checkpoints(ctx, rows)?;
                    if finished || steps >= max_steps || admission() {
                        return Ok(steps);
                    }
                }
            }
        }
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
        let Row { sink, req, slot, in_flight, .. } = row;
        slots.give(slot);
        self.close(sink, in_flight, req.p.stream, Ending::FailedUnder(error, fault));
    }

    // --- admission ----------------------------------------------------------

    /// Starts `job`: its admission (everything before the decode, with the
    /// prefill interleaved with batched steps of `rows`), then its first
    /// draw. The request becomes a row, or is answered here when it ends
    /// before its decode (cancelled, failed, or finished by its first draw).
    fn start_row<'g>(
        &mut self,
        job: Job,
        seq: u64,
        generator: &'g Generator,
        rows: &mut Vec<Row<'g, M>>,
        slots: &mut Slots,
    ) {
        let Opened { p, mut sink, in_flight, queued } = self.open(job);
        let stream = p.stream;
        if sink.cancelled() {
            self.close(sink, in_flight, stream, Ending::Answered);
            return;
        }
        let mut interleaved =
            Interleaved { generator, rows, slots, stats: BatchStats::default() };
        let admitted = self.admit(
            p,
            &mut sink,
            queued,
            in_flight.pinned(),
            generator,
            &mut interleaved,
        );
        let Interleaved { stats, .. } = interleaved;
        let mut req = match admitted {
            Ok(Some(req)) => req,
            // Answered: its prefill was stopped (see `Engine::admit`).
            Ok(None) => {
                self.close(sink, in_flight, stream, Ending::Answered);
                return;
            }
            Err(error) => {
                self.close(sink, in_flight, stream, Ending::Failed(&error));
                return;
            }
        };
        // The first draw (`Generator::generate`'s start), into the engine's
        // sampler, whose counts it resets; the row then takes a slot and the
        // counts go with it. A failed request's session is dropped before
        // its answer, as everywhere.
        let n = req.facts.n;
        let decode_started = Instant::now();
        let drawn = generator
            .begin(
                &self.ctx,
                &self.model,
                &mut req.session.state,
                &mut self.scratch,
                &req.p.prompt[n - 1..],
                &req.p.sampling,
            )
            .and_then(|first| {
                let slot = slots.take().ok_or_else(|| {
                    anyhow::anyhow!("no free batch slot for an admitted request")
                })?;
                Ok((first, slot))
            });
        let (first, slot) = match drawn {
            Ok(drawn) => drawn,
            Err(error) => {
                drop(req);
                self.close(sink, in_flight, stream, Ending::Failed(&error));
                return;
            }
        };
        let mut row = Row {
            seq,
            req,
            sink,
            in_flight,
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
            let Row { req, sink, in_flight, .. } = row;
            drop(req);
            self.close(sink, in_flight, stream, Ending::Failed(&error));
            return;
        }
        match row.finish {
            Some(_) => {
                let others = Self::in_flight(rows);
                self.finish_row(row, slots, others);
            }
            None => rows.push(row),
        }
    }

    // --- the answer -----------------------------------------------------------

    /// Answers a row that finished ([`Engine::answer`]) and closes it.
    /// `in_flight_others` are the bytes of the sessions other requests
    /// still hold.
    fn finish_row(
        &mut self,
        row: Row<'_, M>,
        slots: &mut Slots,
        in_flight_others: usize,
    ) {
        let Row {
            req,
            mut sink,
            in_flight,
            generated,
            slot,
            decode_started,
            drafted,
            accepted,
            stats,
            finish,
            ..
        } = row;
        slots.give(slot);
        let stream = req.p.stream;
        let decoded = Decoded {
            tokens: &generated,
            finish: finish.unwrap_or(FinishReason::Callback),
            drafted,
            accepted,
            started: decode_started,
            batch: Some(&stats),
        };
        let result = self.answer(req, &mut sink, decoded, in_flight_others);
        let ending = match &result {
            Ok(()) => Ending::Answered,
            Err(error) => Ending::Failed(error),
        };
        self.close(sink, in_flight, stream, ending);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/serve/batch.rs"]
mod tests;
