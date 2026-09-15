//! Lily's OpenAI-compatible API server.
//!
//! One engine thread owns the GPU and runs requests strictly one at a time
//! from a bounded queue. The HTTP thread parses and validates requests (400s
//! never touch the engine), and a small responder thread per request relays
//! the engine's output to the socket, so a slow or vanished client never
//! stalls the decode loop; its write failure cancels the generation at the
//! next token.
//!
//! The engine thread also owns the model's lifetime: with `--idle-unload`
//! it spills the resident sessions to the disk tier and drops the whole
//! engine (weights, scratch, caches, Metal context, the n-gram mapping) once
//! no request has run for that long, and loads it again for the next request,
//! which waits instead of failing. SIGTERM/SIGINT stop the listener, give the
//! running request a bounded grace, spill the sessions and exit 0.
//!
//! A GPU fault (a Metal 4 command-queue error such as a timeout, reported
//! through the commit feedback) makes the context permanently unusable, so
//! the engine thread answers the running request with 503, drops the whole
//! engine without spilling (the caches on a faulted queue are not
//! trustworthy) and loads it again; `/health` says `recovering` meanwhile.
//! More than [`MAX_RECOVERIES`] faults within [`RECOVERY_WINDOW`] exit the
//! process with status 1 for the supervisor to restart it.

pub mod api;
pub mod disk;
pub mod http;
mod session;
pub mod stream;
pub mod timings;
pub mod tools;

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};

use crate::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi};
use crate::generate::{FinishReason, GenerateOptions, Generator};
use crate::kernels::attention::MAX_SEQ;
use crate::kernels::sample::SamplingParams;
use crate::metal::MetalContext;
use crate::model::Qwen3_5Model;
use crate::qwen4exp::{NgramStorage, Qwen4ExpModel};
use api::{Defaults, Kind, Prepared};
use session::SessionStore;
use stream::{Event, OutputParser, ParserConfig};
use timings::{Speculation, Timings, TimingsEntry, TimingsLog};
use tools::ParsedToolCall;

const MAX_REQUEST_BYTES: usize = 32 << 20;
/// Recurrent-state checkpoints kept per session (the newest ones).
const CHECKPOINTS_PER_SESSION: usize = 3;
/// Completed requests `GET /v1/timings` remembers (a ring buffer; a client
/// polls it right after its request, so a handful of entries is plenty).
const TIMINGS_LOG_CAPACITY: usize = 32;
/// Left free when the cache budget is derived automatically: room for the OS
/// and the applications that share the machine with the server (a browser,
/// containers, an IDE), so a full cache does not push them into swap.
const BUDGET_HEADROOM_BYTES: usize = 8 << 30;
/// The derived budget never goes below this (two full 131k contexts of the
/// Qwen3.8 caches), whatever the arithmetic says; `--cache-bytes` overrides.
const BUDGET_FLOOR_BYTES: usize = 8 << 30;

/// The default session-cache budget: what the device's recommended working
/// set leaves after the weights already allocated, the paged weights that
/// live in the page cache (the n-gram table) and the headroom, floored.
/// Returns the budget and whether the floor applied.
fn derive_cache_budget(working_set: usize, allocated: usize, paged: usize) -> (usize, bool) {
    let derived = working_set.saturating_sub(allocated).saturating_sub(paged).saturating_sub(BUDGET_HEADROOM_BYTES);
    (derived.max(BUDGET_FLOOR_BYTES), derived < BUDGET_FLOOR_BYTES)
}
/// How long a running request may keep going after a stop signal before it
/// is cancelled at its next token.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);
/// How long the listener waits for connection threads to finish writing
/// their responses after the engine stopped.
const SHUTDOWN_DRAIN: Duration = Duration::from_secs(3);

/// Parses `3d`, `12h`, `90m`, `45s` or a bare number of seconds.
pub fn parse_duration_secs(text: &str) -> Result<u64> {
    let text = text.trim();
    let (digits, unit) = text
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .map_or((text, ""), |i| text.split_at(i));
    let value: f64 = digits.parse().with_context(|| format!("invalid duration {text:?}"))?;
    let scale = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "s" => 1.0,
        "m" => 60.0,
        "h" => 3600.0,
        "d" => 86_400.0,
        other => anyhow::bail!("unknown duration unit {other:?} (use s, m, h or d)"),
    };
    Ok((value * scale) as u64)
}

/// Sampling defaults from the command line; unset fields fall back to the
/// checkpoint's `generation_config.json`, then to OpenAI's defaults.
#[derive(Debug, Clone, Default)]
pub struct SamplingOverrides {
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub bind: String,
    pub max_seq: usize,
    pub cache_bytes: Option<usize>,
    pub max_sessions: usize,
    pub ngram_storage: NgramStorage,
    pub ngram_preload: bool,
    /// Pin the preloaded table in memory (`mlock`).
    pub ngram_lock: bool,
    /// Draft tokens per speculative step (0 disables the draft head).
    pub mtp_drafts: usize,
    /// Where evicted sessions are kept on disk (`None` disables the tier).
    pub disk_cache_dir: Option<std::path::PathBuf>,
    /// Most bytes the disk tier may hold.
    pub disk_cache_bytes: u64,
    /// Seconds an entry may go unused on disk before it is deleted (0: never).
    pub disk_cache_ttl_secs: u64,
    pub thinking: bool,
    pub reasoning_effort: Option<String>,
    pub queue: usize,
    pub sampling: SamplingOverrides,
    /// Seconds without a request after which the engine is unloaded (0: never).
    pub idle_unload_secs: u64,
    /// Testing only: record a Metal fault on the N-th request the engine
    /// serves (1-based, counted across reloads) to exercise the recovery
    /// path; `None` in normal operation.
    pub inject_metal_fault: Option<u64>,
}

// --- lifecycle ---------------------------------------------------------------

/// Where the engine is in its life, as `/health` reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum State {
    /// The first load; requests are refused with 503.
    Loading = 0,
    /// Loaded and serving.
    Ready = 1,
    /// Unloaded after the idle timeout; the next request reloads it.
    Idle = 2,
    /// Loading again for a request that is waiting.
    Reloading = 3,
    /// A stop signal arrived; new requests are refused with 503.
    Stopping = 4,
    /// A GPU fault took the engine down; it is being dropped and loaded
    /// again. Requests wait for it (their cached prefixes are gone).
    Recovering = 5,
}

impl State {
    pub fn as_str(self) -> &'static str {
        match self {
            State::Loading => "loading",
            State::Ready => "ready",
            State::Idle => "idle",
            State::Reloading => "reloading",
            State::Stopping => "stopping",
            State::Recovering => "recovering",
        }
    }

    /// Whether the server takes requests in this state (they may have to
    /// wait for a reload).
    pub fn accepting(self) -> bool {
        matches!(self, State::Ready | State::Idle | State::Reloading | State::Recovering)
    }

    /// The `/health` status code and `status` field. `ok`/`loading` keep the
    /// meaning they had before idle unloading existed: a client that only
    /// looks at the code sees 200 whenever a request would be served, with
    /// one exception: `recovering` is 503 although requests are queued,
    /// because something went wrong that an operator should notice.
    pub fn health(self) -> (u16, &'static str) {
        match self {
            State::Loading => (503, "loading"),
            State::Ready | State::Idle | State::Reloading => (200, "ok"),
            State::Stopping => (503, "stopping"),
            State::Recovering => (503, "recovering"),
        }
    }
}

/// The engine's state, shared between the engine thread and the HTTP threads.
pub struct Lifecycle {
    state: AtomicU8,
}

impl Lifecycle {
    pub fn new(state: State) -> Self {
        Self { state: AtomicU8::new(state as u8) }
    }

    pub fn set(&self, state: State) {
        self.state.store(state as u8, Ordering::Release);
    }

    pub fn get(&self) -> State {
        match self.state.load(Ordering::Acquire) {
            0 => State::Loading,
            1 => State::Ready,
            2 => State::Idle,
            3 => State::Reloading,
            4 => State::Stopping,
            _ => State::Recovering,
        }
    }
}

/// Most automatic recoveries from GPU faults within [`RECOVERY_WINDOW`]; one
/// more exits the process so the supervisor restarts it with its throttle.
pub const MAX_RECOVERIES: usize = 3;
pub const RECOVERY_WINDOW: Duration = Duration::from_secs(10 * 60);

/// Counts GPU-fault recoveries in a sliding window, so a GPU that keeps
/// faulting does not keep the server in a reload loop that never serves.
pub struct RecoveryBudget {
    max: usize,
    window: Duration,
    faults: Vec<Instant>,
}

impl RecoveryBudget {
    pub fn new(max: usize, window: Duration) -> Self {
        Self { max, window, faults: Vec::new() }
    }

    /// Records a fault at `now`. `Some(n)`: recovering is allowed and this
    /// is the n-th recovery within the window; `None`: the budget is used up.
    pub fn record(&mut self, now: Instant) -> Option<usize> {
        self.faults.retain(|&at| now.saturating_duration_since(at) < self.window);
        self.faults.push(now);
        (self.faults.len() <= self.max).then_some(self.faults.len())
    }

    /// Faults recorded within the window as of the last `record`.
    pub fn faults_in_window(&self) -> usize {
        self.faults.len()
    }
}

/// The idle-unload clock: how long the engine may wait for the next request
/// before it is worth unloading.
pub struct IdleTimer {
    timeout: Option<Duration>,
    last_active: Instant,
}

impl IdleTimer {
    /// `timeout_secs == 0` never unloads.
    pub fn new(timeout_secs: u64, now: Instant) -> Self {
        Self { timeout: (timeout_secs > 0).then(|| Duration::from_secs(timeout_secs)), last_active: now }
    }

    /// Records activity (a request just finished, or the engine just loaded).
    pub fn touch(&mut self, now: Instant) {
        self.last_active = now;
    }

    /// How long to wait for the next request before the engine counts as
    /// idle: `None` when it never does, zero when it already is.
    pub fn remaining(&self, now: Instant) -> Option<Duration> {
        let timeout = self.timeout?;
        Some(timeout.saturating_sub(now.saturating_duration_since(self.last_active)))
    }

    pub fn expired(&self, now: Instant) -> bool {
        self.remaining(now) == Some(Duration::ZERO)
    }
}

/// Stop coordination: `requested` closes the door (no new requests, the
/// engine winds down after the running one); `cancel` is raised once the
/// grace period is over and stops the running request at its next token.
#[derive(Default)]
struct Shutdown {
    requested: AtomicBool,
    cancel: AtomicBool,
}

impl Shutdown {
    fn requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    fn cancel(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

// --- HTTP plumbing -----------------------------------------------------------

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

fn error_json(kind: &'static str, message: impl Into<String>) -> Vec<u8> {
    serde_json::to_vec(&ErrorEnvelope { error: ErrorBody { message: message.into(), kind } })
        .unwrap_or_else(|_| b"{\"error\":{\"message\":\"error\"}}".to_vec())
}

fn send_json<T: Serialize>(stream: TcpStream, status: u16, value: &T) {
    let body = serde_json::to_vec(value).unwrap_or_default();
    if let Err(error) = http::respond(stream, status, "application/json", &body) {
        eprintln!("response error: {error:#}");
    }
}

fn send_error(stream: TcpStream, status: u16, kind: &'static str, message: impl Into<String>) {
    if let Err(error) = http::respond(stream, status, "application/json", &error_json(kind, message)) {
        eprintln!("response error: {error:#}");
    }
}

/// Answers a request the engine never sees and logs why, so a client that
/// gives up on a 4xx/5xx can be traced in the server log.
fn refuse(stream: TcpStream, request: &str, status: u16, kind: &'static str, message: impl Into<String>) {
    let message = message.into();
    eprintln!("rejected {request} with {status}: {message}");
    send_error(stream, status, kind, message);
}

/// What the engine sends the responder thread for one request.
enum Out {
    Start { status: u16, content_type: &'static str },
    Body(Vec<u8>),
    End,
}

/// The engine's handle on a request's response.
struct Sink {
    tx: Sender<Out>,
    cancelled: Arc<AtomicBool>,
    started: bool,
}

impl Sink {
    fn start(&mut self, status: u16, content_type: &'static str) {
        if !self.started {
            self.started = true;
            let _ = self.tx.send(Out::Start { status, content_type });
        }
    }

    fn send(&self, body: Vec<u8>) {
        let _ = self.tx.send(Out::Body(body));
    }

    fn sse(&self, value: &Value) {
        let mut line = b"data: ".to_vec();
        line.extend_from_slice(serde_json::to_string(value).unwrap_or_default().as_bytes());
        line.extend_from_slice(b"\n\n");
        self.send(line);
    }

    fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    fn end(&self) {
        let _ = self.tx.send(Out::End);
    }
}

struct Job {
    prepared: Prepared,
    sink: Sink,
    queued_at: Instant,
}

impl Job {
    /// Answers the request without running it.
    fn reject(self, status: u16, message: &str) {
        let mut sink = self.sink;
        sink.start(status, "application/json");
        sink.send(error_json("server_error", message));
        sink.end();
    }
}

/// What the HTTP threads (and the stop handler) send the engine thread.
enum Cmd {
    Job(Box<Job>),
    /// Wakes the engine so it notices a stop request; carries nothing.
    Wake,
}

/// Relays one request's output from the engine to the client on the
/// connection's own thread; a failed write or the client's EOF flags
/// cancellation for the engine.
fn relay(stream: TcpStream, rx: Receiver<Out>, cancelled: Arc<AtomicBool>) {
    let (status, content_type) = match rx.recv() {
        Ok(Out::Start { status, content_type }) => (status, content_type),
        _ => {
            send_error(stream, 500, "server_error", "internal server error");
            return;
        }
    };
    let mut response = match http::ChunkedResponse::start(stream, status, content_type, cancelled.clone()) {
        Ok(response) => response,
        Err(_) => {
            cancelled.store(true, Ordering::Relaxed);
            return;
        }
    };
    while let Ok(message) = rx.recv() {
        match message {
            Out::Body(bytes) => {
                if response.write_chunk(&bytes).is_err() {
                    // Keep draining so the engine's sends never block.
                    for _ in rx.iter() {}
                    return;
                }
            }
            Out::End => break,
            Out::Start { .. } => {}
        }
    }
    if cancelled.load(Ordering::Relaxed) {
        return;
    }
    if let Err(error) = response.finish() {
        eprintln!("response error: {error:#}");
    }
}

// --- engine ------------------------------------------------------------------

struct Engine<M: LanguageModel> {
    ctx: MetalContext,
    model: M,
    generator: Arc<Generator>,
    sessions: SessionStore<M>,
    scratch: M::Scratch,
    max_seq: usize,
    drafts: usize,
    next_id: u64,
    shutdown: Arc<Shutdown>,
    /// Where each finished request's numbers go for `GET /v1/timings`.
    timings: Arc<TimingsLog>,
}

/// Where a request's text ends up when not streaming.
#[derive(Default)]
struct Collected {
    reasoning: String,
    content: String,
    tool_calls: Vec<ParsedToolCall>,
}

impl<M: LanguageModel> Engine<M> {
    fn load(model_dir: &Path, options: &ServeOptions, shared: &Shared, next_id: u64) -> Result<Self> {
        let Shared { generator, shutdown, timings } = shared.clone();
        let ctx = MetalContext::new()?;
        let started = Instant::now();
        let model = M::load(&ctx, model_dir, &LoadOptions { ngram_storage: options.ngram_storage, mtp_drafts: options.mtp_drafts })?;
        let drafts = options.mtp_drafts.min(model.max_drafts());
        eprintln!(
            "loaded {} in {:.1}s ({:.1} GB resident){}",
            M::MODEL_ID,
            started.elapsed().as_secs_f64(),
            ctx.current_allocated() as f64 / 1e9,
            if drafts > 0 { format!(", speculative decoding with {drafts} drafts per step") } else { String::new() }
        );
        if options.ngram_preload || options.ngram_lock {
            let started = Instant::now();
            let bytes = model.warm_storage(options.ngram_lock)?;
            if bytes > 0 {
                eprintln!(
                    "paged weights: {:.1} GB resident after preload in {:.1}s{}",
                    bytes as f64 / 1e9,
                    started.elapsed().as_secs_f64(),
                    if options.ngram_lock { " (locked)" } else { "" }
                );
            }
        }
        let max_seq = effective_max_seq(options.max_seq, model.max_position_embeddings());
        ensure!(max_seq > 1, "max_seq must be at least 2");
        let mut scratch = model.new_scratch_with_capacity(&ctx, max_seq)?;
        warm_up(&ctx, &model, &mut scratch)?;

        let allocated = ctx.current_allocated();
        let working_set = ctx.recommended_working_set();
        let paged = model.paged_storage_bytes();
        let gb = |bytes: usize| bytes as f64 / 1e9;
        let budget = match options.cache_bytes {
            Some(b) => b,
            None => {
                let (budget, floored) = derive_cache_budget(working_set, allocated, paged);
                eprintln!(
                    "session cache budget: {:.1} GB = {:.1} GB recommended working set - {:.1} GB allocated \
                     (weights, scratch) - {:.1} GB paged weights in the page cache - {:.1} GB headroom for \
                     other applications{}; override with --cache-bytes",
                    gb(budget),
                    gb(working_set),
                    gb(allocated),
                    gb(paged),
                    gb(BUDGET_HEADROOM_BYTES),
                    if floored { format!(", raised to the {:.1} GB floor", gb(BUDGET_FLOOR_BYTES)) } else { String::new() },
                );
                budget
            }
        };
        let per_request = model.bytes_per_token() * max_seq;
        eprintln!(
            "memory: {:.1} GB allocated, {:.1} GB recommended working set, {:.1} GB session cache budget \
             ({} B/token of context; a full {}-token request needs {:.1} GB)",
            gb(allocated),
            gb(working_set),
            gb(budget),
            model.bytes_per_token(),
            max_seq,
            gb(per_request),
        );
        if per_request > budget {
            eprintln!(
                "warning: a request using the whole {max_seq}-token context exceeds the cache budget; \
                 lower --max-seq or raise --cache-bytes"
            );
        }
        let mut sessions = SessionStore::new(budget, options.max_sessions, CHECKPOINTS_PER_SESSION);
        if let (Some(dir), true) = (&options.disk_cache_dir, options.disk_cache_bytes > 0) {
            match model.persistence_format() {
                Some(format) => {
                    let disk = disk::DiskStore::open(dir, &format, options.disk_cache_bytes, options.disk_cache_ttl_secs)?;
                    eprintln!(
                        "session cache: disk tier at {} ({} entries, {:.1}/{:.1} GB, entries expire after {})",
                        disk.dir().display(),
                        disk.len(),
                        disk.used_bytes() as f64 / 1e9,
                        disk.budget_bytes() as f64 / 1e9,
                        if disk.max_age_secs() == 0 { "never".to_owned() } else { format!("{:.1} days unused", disk.max_age_secs() as f64 / 86_400.0) }
                    );
                    sessions = sessions.with_disk(disk);
                }
                None => eprintln!("session cache: {} cannot persist sessions; disk tier off", M::MODEL_ID),
            }
        }
        Ok(Self {
            ctx,
            model,
            generator,
            sessions,
            scratch,
            max_seq,
            drafts,
            next_id,
            shutdown,
            timings,
        })
    }

    /// Takes the engine down: every resident session goes to the disk tier
    /// (or is lost when there is none), then the scratch, the sessions, the
    /// model and finally the Metal context are dropped, which releases the
    /// GPU buffers and the n-gram mapping. Returns the request counter so
    /// ids stay unique across a reload.
    fn unload(self, reason: &str) -> u64 {
        let started = Instant::now();
        let Engine { ctx, model, generator: _, mut sessions, scratch, next_id, .. } = self;
        let resident = ctx.current_allocated();
        // A faulted queue cannot run the snapshot blits, and what its last
        // command buffers left in the caches is not trustworthy either: the
        // sessions are dropped and clients re-prefill (the disk tier's
        // earlier copies were written by a healthy queue and stay valid).
        let faulted = ctx.fault().is_some();
        let (spilled, dropped) = if faulted { (0, sessions.drop_all()) } else { sessions.spill_all(&ctx) };
        let spill_secs = started.elapsed().as_secs_f64();
        drop(scratch);
        drop(sessions);
        drop(model);
        // Only the context's own arenas may remain here; anything else
        // would be a buffer something outside the engine still holds.
        let left = ctx.current_allocated();
        let allocations = ctx.resident_allocations();
        drop(ctx);
        eprintln!(
            "{reason}: unloaded {} in {:.1}s ({:.1} GB was resident; {spilled} sessions spilled to disk, \
             {dropped} dropped{}, in {spill_secs:.1}s; {:.2} GB in {allocations} pool allocations \
             released with the context)",
            M::MODEL_ID,
            started.elapsed().as_secs_f64(),
            resident as f64 / 1e9,
            if faulted { " without spilling (GPU faulted; their state is not trustworthy)" } else { "" },
            left as f64 / 1e9,
        );
        next_id
    }

    /// Testing only: makes the context fail its next submission the way a
    /// reported GPU error would.
    fn inject_fault(&self) {
        self.ctx.inject_fault("injected by --debug-inject-metal-fault");
    }

    /// Runs one request and answers it. Returns the GPU fault the context
    /// recorded, if any: the engine is then unusable and must be replaced,
    /// even when this request happened to finish before the feedback arrived.
    fn serve(&mut self, job: Job) -> Option<String> {
        let mut sink = job.sink;
        let stream = job.prepared.stream;
        let kind = job.prepared.kind;
        let queued = job.queued_at.elapsed();
        let result = self.run(job.prepared, &mut sink);
        let fault = self.ctx.fault();
        if let Err(error) = result {
            let (status, message) = match &fault {
                Some(fault) => {
                    eprintln!("request failed on a GPU fault (the engine reloads): {error:#}");
                    (503, format!("the GPU command queue failed ({fault}); the engine is reloading, retry shortly"))
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
        }
        sink.end();
        let _ = (kind, queued);
        fault
    }

    fn run(&mut self, p: Prepared, sink: &mut Sink) -> Result<()> {
        if sink.cancelled() {
            return Ok(());
        }
        let Engine { ctx, model, generator, sessions, scratch, max_seq, drafts, next_id, shutdown, timings } = self;
        // The HTTP thread validated against the same limit; this only guards
        // the engine's buffers if the two ever disagree.
        ensure!(
            p.prompt.len() < *max_seq,
            "prompt exceeds the engine context: {} prompt tokens, {max_seq} tokens of context",
            p.prompt.len()
        );
        let n = p.prompt.len();
        let started = Instant::now();
        let acquired = sessions.acquire(ctx, model, &p.prompt, p.cache_key.as_deref())?;
        let mut session = acquired.session;
        let reused = acquired.reused;
        ensure!(reused < n, "session cache returned the whole prompt");

        // Prefix up to the last prompt token, then checkpoint there so an
        // identical or extended prompt can resume without re-feeding it.
        if reused < n - 1 {
            model.prefill(ctx, &mut session.state, scratch, &p.prompt[reused..n - 1], None)?;
        }
        let snapshot = session.state.snapshot(ctx)?;
        session.add_checkpoint(snapshot);
        let prefix_secs = started.elapsed().as_secs_f64();

        let created = now();
        let id = format!("{}-{}-{}", if p.kind == Kind::Chat { "chatcmpl" } else { "cmpl" }, created, *next_id);
        *next_id += 1;
        let tokenizer = generator.tokenizer();
        let mut parser = OutputParser::new(
            |ids: &[u32]| tokenizer.decode(ids, false),
            ParserConfig {
                thinking_open: p.thinking_open,
                tools: p.tools.clone(),
                stop_strings: p.stop_strings.clone(),
                raw: p.kind == Kind::Completion,
            },
        );
        let mut collected = Collected::default();
        let mut tool_index = 0usize;
        if p.stream {
            sink.start(200, "text/event-stream");
            if p.kind == Kind::Chat {
                sink.sse(&chunk(&id, created, M::MODEL_ID, json!({"role": "assistant", "content": ""}), None));
            }
        }
        let mut deliver = |events: Vec<Event>, sink: &Sink| {
            for event in events {
                match event {
                    Event::Reasoning(text) => {
                        if p.stream {
                            sink.sse(&chunk(&id, created, M::MODEL_ID, json!({"reasoning_content": text}), None));
                        } else {
                            collected.reasoning.push_str(&text);
                        }
                    }
                    Event::Content(text) => {
                        if p.stream {
                            if p.kind == Kind::Chat {
                                sink.sse(&chunk(&id, created, M::MODEL_ID, json!({"content": text}), None));
                            } else {
                                sink.sse(&text_chunk(&id, created, M::MODEL_ID, &text, None));
                            }
                        } else {
                            collected.content.push_str(&text);
                        }
                    }
                    Event::ToolCall(call) => {
                        if p.stream {
                            let delta = json!({"tool_calls": [{
                                "index": tool_index,
                                "id": call_id(&id, tool_index),
                                "type": "function",
                                "function": {"name": call.name, "arguments": call.arguments},
                            }]});
                            sink.sse(&chunk(&id, created, M::MODEL_ID, delta, None));
                        } else {
                            collected.tool_calls.push(call);
                        }
                        tool_index += 1;
                    }
                }
            }
        };

        let options = GenerateOptions { max_tokens: p.max_tokens, sampling: &p.sampling, stop_tokens: &[], drafts: *drafts };
        let decode_started = Instant::now();
        let generation = generator.generate(
            ctx,
            model,
            &mut session.state,
            scratch,
            &p.prompt[n - 1..],
            &options,
            &mut |token| {
                let events = parser.push(token)?;
                deliver(events, sink);
                Ok(!sink.cancelled() && !parser.stopped && !shutdown.cancel())
            },
        )?;
        let final_events = parser.finish();
        deliver(final_events, sink);
        let decode_secs = decode_started.elapsed().as_secs_f64();

        // Bookkeeping: the state holds the prompt plus the generated tokens
        // that were fed: all but the last (drawn, never fed), or all of them
        // when a parked step consumed the final one. `fed` counts the last
        // prompt token too.
        let fed_generated = generation.fed.checked_sub(1).ok_or_else(|| anyhow::anyhow!("decode state did not advance"))?;
        ensure!(
            fed_generated + 1 == generation.tokens.len() || fed_generated == generation.tokens.len(),
            "decode state advanced {} tokens for {} drawn",
            generation.fed,
            generation.tokens.len()
        );
        session.tokens.truncate(reused);
        session.tokens.extend_from_slice(&p.prompt[reused..]);
        session.tokens.extend_from_slice(&generation.tokens[..fed_generated]);
        ensure!(session.state.pos() == session.tokens.len(), "session token/state position mismatch");
        sessions.release(ctx, session, p.cache_key.as_deref());

        let completion_tokens = generation.tokens.len();
        let finish_reason = match generation.finish {
            FinishReason::Length => "length",
            _ if parser.tool_calls_emitted() > 0 => "tool_calls",
            _ => "stop",
        };
        eprintln!(
            "{}: {} prompt tokens ({} cached{}{}), {} generated, prefix {:.2}s, decode {:.2}s ({:.1} tok/s){}, finish={finish_reason}, sessions={} ({:.1}/{:.1} GB){}",
            id,
            n,
            reused,
            if acquired.forked { ", forked" } else { "" },
            acquired.from_disk.map(|d| format!(", from disk in {:.2}s", d.as_secs_f64())).unwrap_or_default(),
            completion_tokens,
            prefix_secs,
            decode_secs,
            completion_tokens as f64 / decode_secs.max(1e-9),
            if generation.drafted > 0 {
                format!(", drafts {}/{} accepted", generation.accepted, generation.drafted)
            } else {
                String::new()
            },
            sessions.len(),
            sessions.used_bytes() as f64 / 1e9,
            sessions.budget_bytes() as f64 / 1e9,
            sessions.disk().map(|d| format!(", disk {} ({:.1} GB)", d.len(), d.used_bytes() as f64 / 1e9)).unwrap_or_default(),
        );
        // The same numbers the line above prints, as JSON: attached to the
        // response below and kept for `GET /v1/timings`. Recorded before the
        // cancellation checks so the log and the ring buffer never disagree.
        let measured = Timings::measure(
            n,
            reused,
            prefix_secs,
            completion_tokens,
            decode_secs,
            (*drafts > 0).then_some(Speculation { drafted: generation.drafted, accepted: generation.accepted }),
        );
        timings.record(TimingsEntry { id: id.clone(), model: M::MODEL_ID, created, timings: measured });
        if sink.cancelled() {
            return Ok(());
        }
        if shutdown.cancel() {
            // Stopped by the server, not the client: say so instead of
            // handing out a truncated answer as a finished one.
            if !sink.started {
                sink.start(503, "application/json");
                sink.send(error_json("server_error", "the server is shutting down"));
            } else if p.stream {
                sink.sse(&json!({"error": {"message": "the server is shutting down", "type": "server_error"}}));
                sink.send(b"data: [DONE]\n\n".to_vec());
            }
            return Ok(());
        }
        let mut usage = json!({
            "prompt_tokens": n,
            "completion_tokens": completion_tokens,
            "total_tokens": n + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": reused},
        });
        if generation.drafted > 0 {
            usage["completion_tokens_details"] = json!({
                "accepted_prediction_tokens": generation.accepted,
                "rejected_prediction_tokens": generation.drafted - generation.accepted,
            });
        }
        let timings_json = serde_json::to_value(measured)?;
        if p.stream {
            // `timings` rides the last chunk the stream already sends: the
            // usage chunk when the client asked for one, the finish chunk
            // otherwise. No client ever sees a chunk shape it did not
            // already get, and the extension costs nothing to those that
            // ignore it.
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
            let body = if p.kind == Kind::Chat {
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

fn call_id(request_id: &str, index: usize) -> String {
    let digest = request_id.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
    format!("call_{:016x}{index:02}", digest)
}

fn chunk(id: &str, created: u64, model: &str, delta: Value, finish_reason: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    })
}

fn text_chunk(id: &str, created: u64, model: &str, text: &str, finish_reason: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason, "logprobs": null}],
    })
}

/// Compiles the Metal pipelines a short request needs by running a throwaway
/// two-token prompt and two decode steps, so the first real request does not
/// pay the shader compile (tens of seconds on the 48-layer model). Kernels
/// only long contexts reach (sparse attention) still compile on first use.
fn warm_up<M: LanguageModel>(ctx: &MetalContext, model: &M, scratch: &mut M::Scratch) -> Result<()> {
    let started = Instant::now();
    let greedy = SamplingParams::greedy();
    let mut state = model.new_state(ctx, 4)?;
    scratch.begin_request();
    // Two arbitrary in-vocabulary ids: the values do not matter, only that the
    // prefill and decode graphs get encoded once.
    model.prefill(ctx, &mut state, scratch, &[1, 2], Some(Draw { params: &greedy, step: 0 }))?;
    for (step, (slot_in, slot_out)) in [(0, 1), (1, 0)].into_iter().enumerate() {
        let token = scratch.next_token().view(slot_in, &[1])?.to_u32()?[0];
        let encoded = model.encode_decode_step(ctx, &state, scratch, slot_in, slot_out, Draw { params: &greedy, step: step + 1 })?;
        model.prepare_step_inputs(&mut state, scratch, token)?;
        let pending = encoded.commit()?;
        state.advance(1);
        pending.wait()?;
    }
    // Snapshot/restore compile no shaders but exercise the blit path once.
    let snapshot = state.snapshot(ctx)?;
    state.restore(ctx, &snapshot)?;
    eprintln!("warm-up done in {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}

// --- startup -----------------------------------------------------------------

/// The `model_type` a checkpoint's `config.json` declares.
pub fn checkpoint_model_type(model_dir: &Path) -> Result<String> {
    let config = read_config(model_dir)?;
    config
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .with_context(|| format!("{} has no model_type", model_dir.join("config.json").display()))
}

fn read_config(model_dir: &Path) -> Result<Value> {
    let path = model_dir.join("config.json");
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).context("parsing config.json")
}

/// `eos_token_id` from `config.json` (top level or `text_config`), int or list.
fn checkpoint_eos_ids(model_dir: &Path) -> Result<Vec<u32>> {
    let config = read_config(model_dir)?;
    let value = config
        .get("eos_token_id")
        .or_else(|| config.get("text_config").and_then(|t| t.get("eos_token_id")));
    Ok(match value {
        Some(Value::Number(n)) => n.as_u64().map(|v| v as u32).into_iter().collect(),
        Some(Value::Array(items)) => items.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect(),
        _ => Vec::new(),
    })
}

/// `max_position_embeddings` from `config.json` (top level or `text_config`);
/// zero when the checkpoint does not declare one.
fn checkpoint_max_position_embeddings(model_dir: &Path) -> Result<usize> {
    let config = read_config(model_dir)?;
    Ok(config
        .get("max_position_embeddings")
        .or_else(|| config.get("text_config").and_then(|t| t.get("max_position_embeddings")))
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize)
}

/// The per-request context the server enforces: the flag, capped by the
/// kernels' limit and by the window the checkpoint declares (the engine
/// applies the same caps from the loaded model, so both sides agree).
fn effective_max_seq(requested: usize, declared: usize) -> usize {
    let max_seq = requested.min(MAX_SEQ);
    if declared > 0 { max_seq.min(declared) } else { max_seq }
}

/// Sampling defaults: `generation_config.json` over OpenAI's defaults, then
/// the command-line overrides.
fn sampling_defaults(model_dir: &Path, overrides: &SamplingOverrides) -> Result<SamplingParams> {
    let mut params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repetition_penalty: 1.0,
        seed: 0,
    };
    let path = model_dir.join("generation_config.json");
    if let Ok(bytes) = std::fs::read(&path) {
        let cfg: Value = serde_json::from_slice(&bytes).context("parsing generation_config.json")?;
        let f = |key: &str| cfg.get(key).and_then(Value::as_f64).map(|v| v as f32);
        if cfg.get("do_sample").and_then(Value::as_bool) == Some(false) {
            params.temperature = 0.0;
        }
        if let Some(t) = f("temperature") {
            params.temperature = t;
        }
        if let Some(p) = f("top_p") {
            params.top_p = p;
        }
        if let Some(k) = cfg.get("top_k").and_then(Value::as_u64) {
            params.top_k = k as usize;
        }
        if let Some(m) = f("min_p") {
            params.min_p = m;
        }
        if let Some(r) = f("repetition_penalty") {
            params.repetition_penalty = r;
        }
        if let Some(p) = f("presence_penalty") {
            params.presence_penalty = p;
        }
    }
    if let Some(v) = overrides.temperature {
        params.temperature = v;
    }
    if let Some(v) = overrides.top_k {
        params.top_k = v;
    }
    if let Some(v) = overrides.top_p {
        params.top_p = v;
    }
    if let Some(v) = overrides.min_p {
        params.min_p = v;
    }
    if let Some(v) = overrides.presence_penalty {
        params.presence_penalty = v;
    }
    if let Some(v) = overrides.frequency_penalty {
        params.frequency_penalty = v;
    }
    if let Some(v) = overrides.repetition_penalty {
        params.repetition_penalty = v;
    }
    params.validate()?;
    Ok(params)
}

/// Serves the checkpoint at `model_dir` with the engine its `model_type` names.
pub fn run(model_dir: &Path, options: ServeOptions) -> Result<()> {
    // Before any other thread exists, so they all inherit the mask and the
    // stop signals only ever reach the thread that waits for them.
    let signals = signal::Signals::block()?;
    match checkpoint_model_type(model_dir)?.as_str() {
        "qwen3_5_moe" => run_with::<Qwen3_5Model>(model_dir, options, signals),
        "qwen4_exp" => run_with::<Qwen4ExpModel>(model_dir, options, signals),
        other => anyhow::bail!(
            "unsupported model_type {other:?}; lily serves qwen3_5_moe \
             (Qwen3.6-35B-A3B) and qwen4_exp (Qwen3.8-Flash-Next)"
        ),
    }
}

/// The handles the HTTP threads and the engine thread share.
#[derive(Clone)]
struct Shared {
    generator: Arc<Generator>,
    shutdown: Arc<Shutdown>,
    /// Where each finished request's numbers go for `GET /v1/timings`.
    timings: Arc<TimingsLog>,
}

/// Everything the HTTP thread needs without the engine.
struct Front {
    shared: Shared,
    defaults: Defaults,
    max_seq: usize,
    jobs: SyncSender<Cmd>,
    lifecycle: Arc<Lifecycle>,
    /// Connection threads still running (the listener waits for them a
    /// little at shutdown so responses in flight get written).
    connections: Arc<AtomicUsize>,
    model_id: &'static str,
    idle_unload_secs: u64,
}

/// The engine thread's body: the first load, then requests until a stop is
/// requested, with the idle unload and reload in between. Exits the process
/// with status 1 when a load fails, so a supervisor restarts it (with its
/// backoff) instead of leaving a server up that can never answer.
fn engine_loop<M: LanguageModel>(
    model_dir: &Path,
    options: &ServeOptions,
    shared: Shared,
    rx: Receiver<Cmd>,
    lifecycle: &Lifecycle,
    address: SocketAddr,
) {
    let Shared { shutdown, .. } = &shared;
    // Exits with status 1 after logging `what` and refusing the queued
    // requests with `status`/`client_message`.
    let exit_failed = |what: String, status: u16, client_message: &str, rx: &Receiver<Cmd>| -> ! {
        eprintln!("{what}");
        lifecycle.set(State::Stopping);
        let mut refused = 0usize;
        while let Ok(Cmd::Job(job)) = rx.try_recv() {
            job.reject(status, client_message);
            refused += 1;
        }
        if refused > 0 {
            eprintln!("exiting: refused {refused} queued requests with {status}");
        }
        std::process::exit(1)
    };
    let fatal = |what: &str, error: anyhow::Error, rx: &Receiver<Cmd>| -> ! {
        exit_failed(format!("{what}: {error:#}"), 500, "the model failed to load", rx)
    };
    let mut engine = match Engine::<M>::load(model_dir, options, &shared, 1) {
        Ok(engine) => Some(engine),
        Err(error) => fatal("engine failed to start", error, &rx),
    };
    let mut next_id = 1;
    let mut recoveries = RecoveryBudget::new(MAX_RECOVERIES, RECOVERY_WINDOW);
    let mut inject_fault_at = options.inject_metal_fault;
    let mut served = 0u64;
    lifecycle.set(State::Ready);
    eprintln!(
        "ready: serving {} on http://{address}{}",
        M::MODEL_ID,
        if options.idle_unload_secs > 0 {
            format!(" (unloading after {} idle)", describe_secs(options.idle_unload_secs))
        } else {
            String::new()
        }
    );
    let mut idle = IdleTimer::new(options.idle_unload_secs, Instant::now());
    while !shutdown.requested() {
        // An unloaded engine has nothing to time out; wait for a request.
        let wait = engine.as_ref().and_then(|_| idle.remaining(Instant::now()));
        let cmd = match wait {
            None => rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
            Some(wait) => rx.recv_timeout(wait),
        };
        match cmd {
            Ok(Cmd::Job(job)) => {
                if shutdown.requested() {
                    job.reject(503, "the server is shutting down");
                    break;
                }
                if engine.is_none() {
                    lifecycle.set(State::Reloading);
                    let started = Instant::now();
                    match Engine::<M>::load(model_dir, options, &shared, next_id) {
                        Ok(loaded) => {
                            eprintln!("reloaded {} in {:.1}s for a waiting request", M::MODEL_ID, started.elapsed().as_secs_f64());
                            engine = Some(loaded);
                            lifecycle.set(State::Ready);
                        }
                        Err(error) => {
                            job.reject(500, "the model failed to reload");
                            fatal("engine failed to reload", error, &rx);
                        }
                    }
                }
                served += 1;
                if inject_fault_at.take_if(|at| *at == served).is_some() {
                    eprintln!("debug: injecting a Metal fault on request {served} (--debug-inject-metal-fault)");
                    engine.as_ref().expect("engine loaded").inject_fault();
                }
                let fault = engine.as_mut().expect("engine loaded").serve(*job);
                idle.touch(Instant::now());
                if let Some(fault) = fault {
                    let faulted = engine.take().expect("engine loaded");
                    lifecycle.set(State::Recovering);
                    let window = describe_secs(RECOVERY_WINDOW.as_secs());
                    let Some(attempt) = recoveries.record(Instant::now()) else {
                        exit_failed(
                            format!(
                                "GPU fault: {fault}; fault {} within {window} exceeds the {MAX_RECOVERIES} automatic \
                                 recoveries, exiting for the supervisor to restart the process",
                                recoveries.faults_in_window()
                            ),
                            503,
                            "the server is restarting after repeated GPU faults",
                            &rx,
                        );
                    };
                    eprintln!(
                        "GPU fault: {fault}; dropping the engine and loading it again \
                         (recovery {attempt} of {MAX_RECOVERIES} within {window})"
                    );
                    next_id = faulted.unload("GPU fault");
                    let started = Instant::now();
                    match Engine::<M>::load(model_dir, options, &shared, next_id) {
                        Ok(loaded) => {
                            eprintln!(
                                "recovered: reloaded {} in {:.1}s after the GPU fault; resident sessions were \
                                 dropped, clients re-prefill (disk tier entries still apply)",
                                M::MODEL_ID,
                                started.elapsed().as_secs_f64()
                            );
                            engine = Some(loaded);
                            lifecycle.set(State::Ready);
                            idle.touch(Instant::now());
                        }
                        Err(error) => fatal("engine failed to reload after a GPU fault", error, &rx),
                    }
                }
            }
            Ok(Cmd::Wake) => {}
            Err(RecvTimeoutError::Timeout) => {
                if let Some(loaded) = engine.take_if(|_| idle.expired(Instant::now())) {
                    next_id = loaded.unload(&format!("idle for {}", describe_secs(options.idle_unload_secs)));
                    lifecycle.set(State::Idle);
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    lifecycle.set(State::Stopping);
    let mut refused = 0usize;
    while let Ok(Cmd::Job(job)) = rx.try_recv() {
        job.reject(503, "the server is shutting down");
        refused += 1;
    }
    if refused > 0 {
        eprintln!("stopping: refused {refused} queued requests");
    }
    if let Some(loaded) = engine {
        loaded.unload("stopping");
    }
}

fn describe_secs(secs: u64) -> String {
    match secs {
        s if s % 3600 == 0 => format!("{}h", s / 3600),
        s if s % 60 == 0 => format!("{}m", s / 60),
        s => format!("{s}s"),
    }
}

/// The address a client on this machine reaches the listener at (a bind to
/// the unspecified address is reachable on loopback).
fn loopback_of(address: SocketAddr) -> SocketAddr {
    let ip = match address.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        ip => ip,
    };
    SocketAddr::new(ip, address.port())
}

fn run_with<M: LanguageModel + 'static>(model_dir: &Path, options: ServeOptions, signals: signal::Signals) -> Result<()> {
    let address = options
        .bind
        .to_socket_addrs()
        .with_context(|| format!("resolving bind address {}", options.bind))?
        .next()
        .with_context(|| format!("bind address {} resolved to nothing", options.bind))?;
    let mut generator = Generator::from_model_dir(model_dir)?;
    generator.add_stop_tokens(&checkpoint_eos_ids(model_dir)?);
    let generator = Arc::new(generator);
    let defaults = Defaults {
        sampling: sampling_defaults(model_dir, &options.sampling)?,
        thinking: options.thinking,
        reasoning_effort: options.reasoning_effort.clone(),
    };
    eprintln!(
        "defaults: temperature {} top_k {} top_p {} min_p {} repetition_penalty {} presence {} frequency {}; thinking {}{}",
        defaults.sampling.temperature,
        defaults.sampling.top_k,
        defaults.sampling.top_p,
        defaults.sampling.min_p,
        defaults.sampling.repetition_penalty,
        defaults.sampling.presence_penalty,
        defaults.sampling.frequency_penalty,
        if defaults.thinking { "on" } else { "off" },
        defaults.reasoning_effort.as_deref().map(|e| format!(" (effort {e})")).unwrap_or_default(),
    );
    let max_seq = effective_max_seq(options.max_seq, checkpoint_max_position_embeddings(model_dir)?);
    if max_seq < options.max_seq {
        eprintln!("context: --max-seq {} capped to {max_seq} (the checkpoint's window or the kernel limit)", options.max_seq);
    }

    let listener = TcpListener::bind(address).with_context(|| format!("binding http://{}", options.bind))?;
    let (jobs, job_rx) = mpsc::sync_channel::<Cmd>(options.queue.max(1));
    let lifecycle = Arc::new(Lifecycle::new(State::Loading));
    let shutdown = Arc::new(Shutdown::default());
    let shared = Shared { generator, shutdown: shutdown.clone(), timings: Arc::new(TimingsLog::new(TIMINGS_LOG_CAPACITY)) };
    let engine_thread = {
        let model_dir = model_dir.to_path_buf();
        let options = options.clone();
        let lifecycle = lifecycle.clone();
        let shared = shared.clone();
        std::thread::Builder::new()
            .name("lily-engine".into())
            .spawn(move || engine_loop::<M>(&model_dir, &options, shared, job_rx, &lifecycle, address))
            .context("spawning the engine thread")?
    };
    {
        // SIGTERM (launchd's stop) or SIGINT: close the door, give the
        // running request its grace, then cancel it. The engine thread and
        // the listener notice the flag; a loopback connection wakes the
        // listener out of `accept`.
        let shutdown = shutdown.clone();
        let lifecycle = lifecycle.clone();
        let jobs = jobs.clone();
        signals.spawn_handler(move |signal| {
            if shutdown.requested() {
                eprintln!("second {signal}: exiting now");
                std::process::exit(130);
            }
            eprintln!("{signal}: stopping (no new requests; a running request has {}s to finish)", SHUTDOWN_GRACE.as_secs());
            shutdown.requested.store(true, Ordering::Release);
            lifecycle.set(State::Stopping);
            let _ = jobs.try_send(Cmd::Wake);
            let _ = TcpStream::connect_timeout(&loopback_of(address), Duration::from_secs(1));
            std::thread::sleep(SHUTDOWN_GRACE);
            shutdown.cancel.store(true, Ordering::Relaxed);
        })?;
    }
    eprintln!("listening on http://{address} (loading model)");

    let connections = Arc::new(AtomicUsize::new(0));
    let front = Arc::new(Front {
        shared,
        defaults,
        max_seq,
        jobs,
        lifecycle,
        connections: connections.clone(),
        model_id: M::MODEL_ID,
        idle_unload_secs: options.idle_unload_secs,
    });
    for stream in listener.incoming() {
        if shutdown.requested() {
            break;
        }
        let stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("accept error: {error}");
                continue;
            }
        };
        let front = front.clone();
        connections.fetch_add(1, Ordering::AcqRel);
        if let Err(error) = std::thread::Builder::new().name("lily-http".into()).spawn(move || {
            handle(&front, stream);
            front.connections.fetch_sub(1, Ordering::AcqRel);
        }) {
            connections.fetch_sub(1, Ordering::AcqRel);
            eprintln!("failed to spawn connection thread: {error}");
        }
    }
    drop(listener);
    eprintln!("stopping: listener closed, waiting for the engine");
    let _ = engine_thread.join();
    let deadline = Instant::now() + SHUTDOWN_DRAIN;
    while connections.load(Ordering::Acquire) > 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    eprintln!("stopped");
    Ok(())
}

fn handle(front: &Front, mut stream: TcpStream) {
    let request = match http::read_request(&mut stream, MAX_REQUEST_BYTES) {
        Ok(request) => request,
        Err(error) => {
            refuse(stream, "request", 400, "invalid_request_error", format!("{error:#}"));
            return;
        }
    };
    let path = request.path.split('?').next().unwrap_or("").to_string();
    let what = format!("{} {path}", request.method);
    let state = if front.shared.shutdown.requested() { State::Stopping } else { front.lifecycle.get() };
    match (request.method.as_str(), path.as_str()) {
        ("GET", "/health") => {
            let (code, status) = state.health();
            send_json(
                stream,
                code,
                &json!({
                    "status": status,
                    "state": state.as_str(),
                    "model": front.model_id,
                    "idle_unload_secs": front.idle_unload_secs,
                }),
            );
        }
        // Lily's own extension: the `timings` object of the most recent
        // completed requests, newest first, for clients whose SDK drops
        // unknown response fields.
        ("GET", "/v1/timings") => {
            send_json(stream, 200, &json!({"object": "list", "data": front.shared.timings.recent()}));
        }
        ("GET", "/v1/models") => send_json(
            stream,
            200,
            &json!({
                "object": "list",
                "data": [{"id": front.model_id, "object": "model", "created": 0, "owned_by": "lily"}]
            }),
        ),
        ("POST", "/v1/chat/completions" | "/v1/completions") => {
            let prepared = if path == "/v1/chat/completions" {
                serde_json::from_slice::<api::ChatRequest>(&request.body)
                    .context("parsing the chat request")
                    .and_then(|r| api::prepare_chat(r, front.shared.generator.tokenizer(), &front.defaults, front.max_seq))
            } else {
                serde_json::from_slice::<api::CompletionRequest>(&request.body)
                    .context("parsing the completion request")
                    .and_then(|r| api::prepare_completion(r, front.shared.generator.tokenizer(), &front.defaults, front.max_seq))
            };
            let prepared = match prepared {
                Ok(p) => p,
                Err(error) => {
                    refuse(stream, &what, 400, "invalid_request_error", format!("{error:#}"));
                    return;
                }
            };
            if let Some(asked) = prepared.clamped_from {
                eprintln!(
                    "warning: {what}: max_tokens {asked} clamped to {} (the prompt has {} tokens, the server context \
                     is {}); the response ends with finish_reason \"length\" if it uses them all",
                    prepared.max_tokens,
                    prepared.prompt.len(),
                    front.max_seq
                );
            }
            match state {
                State::Stopping => {
                    refuse(stream, &what, 503, "server_error", "the server is shutting down");
                    return;
                }
                State::Loading => {
                    refuse(stream, &what, 503, "server_error", "model is still loading");
                    return;
                }
                State::Ready | State::Idle | State::Reloading | State::Recovering => {}
            }
            let (tx, rx) = mpsc::channel();
            let cancelled = Arc::new(AtomicBool::new(false));
            let job = Job { prepared, sink: Sink { tx, cancelled: cancelled.clone(), started: false }, queued_at: Instant::now() };
            match front.jobs.try_send(Cmd::Job(Box::new(job))) {
                Ok(()) => relay(stream, rx, cancelled),
                Err(TrySendError::Full(_)) => {
                    refuse(stream, &what, 503, "server_error", "the request queue is full; retry later");
                }
                Err(TrySendError::Disconnected(_)) => {
                    refuse(stream, &what, 500, "server_error", "engine stopped");
                }
            }
        }
        (_, "/health" | "/v1/models" | "/v1/timings" | "/v1/chat/completions" | "/v1/completions") => {
            refuse(stream, &what, 405, "invalid_request_error", "method not allowed");
        }
        _ => refuse(stream, &what, 404, "invalid_request_error", "not found"),
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

/// Stop signals, taken synchronously on one thread. SIGTERM and SIGINT are
/// blocked process-wide (every thread inherits the mask of the one that
/// spawned it, so this must happen before the first spawn) and a dedicated
/// thread `sigwait`s for them, which keeps the handler an ordinary function
/// instead of an async-signal context.
mod signal {
    use std::mem::MaybeUninit;

    use anyhow::{Result, ensure};

    pub struct Signals {
        set: libc::sigset_t,
    }

    impl Signals {
        pub fn block() -> Result<Self> {
            let mut set = MaybeUninit::<libc::sigset_t>::uninit();
            // SAFETY: plain libc calls on a set we own; `set` is initialised
            // by `sigemptyset` before anything reads it.
            let set = unsafe {
                // A non-interactive shell starts background jobs with SIGINT
                // ignored, and an ignored signal is discarded before
                // `sigwait` could take it: restore the default disposition
                // so the stop path works however the process was started.
                libc::signal(libc::SIGTERM, libc::SIG_DFL);
                libc::signal(libc::SIGINT, libc::SIG_DFL);
                libc::sigemptyset(set.as_mut_ptr());
                libc::sigaddset(set.as_mut_ptr(), libc::SIGTERM);
                libc::sigaddset(set.as_mut_ptr(), libc::SIGINT);
                let rc = libc::pthread_sigmask(libc::SIG_BLOCK, set.as_ptr(), std::ptr::null_mut());
                ensure!(rc == 0, "blocking SIGTERM/SIGINT failed: {}", std::io::Error::from_raw_os_error(rc));
                set.assume_init()
            };
            Ok(Self { set })
        }

        /// Runs `on_signal` with the signal's name on a new thread each
        /// time one of the blocked signals arrives.
        pub fn spawn_handler(self, mut on_signal: impl FnMut(&'static str) + Send + 'static) -> Result<()> {
            std::thread::Builder::new()
                .name("lily-signals".into())
                .spawn(move || {
                    loop {
                        let mut signal = 0;
                        // SAFETY: `set` is a valid, initialised signal set
                        // and `signal` a valid out-pointer.
                        let rc = unsafe { libc::sigwait(&self.set, &mut signal) };
                        if rc != 0 {
                            eprintln!("sigwait failed: {}", std::io::Error::from_raw_os_error(rc));
                            return;
                        }
                        on_signal(match signal {
                            libc::SIGTERM => "SIGTERM",
                            libc::SIGINT => "SIGINT",
                            _ => "signal",
                        });
                    }
                })
                .map(drop)
                .map_err(Into::into)
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/serve.rs"]
mod tests;
