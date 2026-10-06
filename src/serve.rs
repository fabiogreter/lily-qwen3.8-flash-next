//! Lily's OpenAI-compatible API server.
//!
//! One engine thread owns the GPU and runs requests strictly one at a time
//! from a bounded queue. The HTTP thread parses and validates requests (400s
//! never touch the engine), and a small responder thread per request relays
//! the engine's output to the socket, so a slow or vanished client never
//! stalls the decode loop. The connection is watched from the moment the
//! request is queued; a client that left cancels it before it starts, at the
//! next prefill chunk or at the next token, and a prefill stopped that way
//! keeps its prefix as a session for the retry.
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
pub mod batch;
pub mod data_uri;
pub mod disk;
pub mod http;
pub mod pin;
pub mod request;
pub mod session;
pub mod stream;
pub mod timings;
pub mod tools;

use std::net::{
    IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs,
};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc::{
    self, Receiver, RecvTimeoutError, Sender, SyncSender, TrySendError,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};

use crate::engine::{
    DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi, SnapshotApi,
    VisionMode, VisionTower,
};
use crate::generate::{DecodeCheckpointer, GenerateOptions, Generator};
use crate::kernels::attention::MAX_SEQ;
use crate::kernels::sample::SamplingParams;
use crate::metal::MetalContext;
use crate::qwen4exp::image::ImageLimits;
use crate::qwen4exp::ngram::BackgroundPreload;
use crate::qwen4exp::weights::SessionContext;
use crate::qwen4exp::{NgramStorage, Qwen4ExpModel};
use api::{Defaults, ImagePolicy, Kind, Prepared};
use request::{Admitted, Alone, Decoded, Ending, Opened};
use session::SessionStore;
use timings::TimingsLog;
use tools::ParsedToolCall;

/// Largest request body. Agent clients resend every image of the history
/// as base64 on every turn, about 2.5 MB per 2000 x 1182 screenshot, so 32
/// MiB refused a session after a dozen of them; the context (about 2 000
/// tokens an image) bounds a request near 130 screenshots, well inside
/// this. The body buffer is zero-filled pages until read into.
const MAX_REQUEST_BYTES: usize = 1 << 30;
/// Recurrent-state checkpoints kept per session (the newest ones).
const CHECKPOINTS_PER_SESSION: usize = 3;
/// Decode checkpoints a generation holds and a session keeps of its latest
/// request (113 MB each on the full model): a 2 048-token interval stays at
/// that spacing up to an 8 192-token answer and thins to 4 096 up to 16 384.
const DECODE_CHECKPOINTS_PER_SESSION: usize = 4;
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
/// The write ahead keeps the evictions for a new session of at least this
/// context free of writes (the size of the last few new sessions raises
/// it): an agent client's new conversation or subagent starts from a
/// preamble of 10 000 to 30 000 tokens.
const WRITE_AHEAD_FLOOR_TOKENS: usize = 32_768;
/// How often the engine looks whether a write ahead finished.
const WRITE_AHEAD_POLL: Duration = Duration::from_millis(50);
/// How long the GPU may go without a submission while the pin is held
/// before the engine sends it a bare fence signal ([`KeepAlive`]): well
/// under the ~1.5 s after which the queue's residency set is dropped.
const KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(1);

/// The default session-cache budget: what the device's recommended working
/// set leaves after the weights already allocated, the paged weights that
/// live in the page cache (the n-gram table) and the headroom, floored.
/// Returns the budget and whether the floor applied.
fn derive_cache_budget(
    working_set: usize,
    allocated: usize,
    paged: usize,
) -> (usize, bool) {
    let derived = working_set
        .saturating_sub(allocated)
        .saturating_sub(paged)
        .saturating_sub(BUDGET_HEADROOM_BYTES);
    (derived.max(BUDGET_FLOOR_BYTES), derived < BUDGET_FLOOR_BYTES)
}
/// How long a running request may keep going after a stop signal before it
/// is cancelled at its next token or prefill chunk.
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
    let value: f64 =
        digits.parse().with_context(|| format!("invalid duration {text:?}"))?;
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
    /// Memory the engine may plan for, in bytes (`None`: the machine's).
    pub memory_budget: Option<u64>,
    /// Whether to load the vision tower when the checkpoint has one.
    pub vision: VisionMode,
    /// An image with more pixels is scaled down to fit before the tower.
    pub image_max_pixels: usize,
    /// An image with fewer pixels is scaled up to reach it.
    pub image_min_pixels: usize,
    /// Where evicted sessions are kept on disk (`None` disables the tier).
    pub disk_cache_dir: Option<std::path::PathBuf>,
    /// Most bytes the disk tier may hold.
    pub disk_cache_bytes: u64,
    /// Seconds an entry may go unused on disk before it is deleted (0: never).
    pub disk_cache_ttl_secs: u64,
    /// A prefix at least this long that two prompts shared without either
    /// being able to resume there is written to the disk tier as a durable
    /// prefix entry, so later prompts resume from it (0 disables).
    pub durable_min_tokens: usize,
    /// A generation takes a recurrent-state checkpoint every this many
    /// tokens, so a next prompt that diverges inside the answer resumes
    /// near the divergence (0 disables).
    pub decode_checkpoint_tokens: usize,
    pub thinking: bool,
    pub reasoning_effort: Option<String>,
    pub queue: usize,
    /// Requests that decode together in one batched step (`--max-batch`);
    /// 1 serves one request at a time, exactly as before batching existed.
    /// Capped by what the model supports.
    pub max_batch: usize,
    pub sampling: SamplingOverrides,
    /// Seconds without a request after which the engine is unloaded (0: never).
    pub idle_unload_secs: u64,
    /// Whether to pin the weights in memory while requests come (`mlock`).
    pub pin_weights: pin::PinMode,
    /// Seconds the pin is held after the last request (0: until memory
    /// pressure or the unload).
    pub pin_hold_secs: u64,
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
        matches!(
            self,
            State::Ready | State::Idle | State::Reloading | State::Recovering
        )
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
        Self {
            timeout: (timeout_secs > 0).then(|| Duration::from_secs(timeout_secs)),
            last_active: now,
        }
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

/// The GPU keep-alive: when the engine's last submission was, and when the
/// next bare fence signal is due so the queue's residency set is never
/// dropped (docs/architecture.md, "The Metal 4 transport"). It ticks only
/// inside a window the caller passes, the pin's hold
/// ([`pin::WeightPin::keep_warm_remaining`]), so an idle server without a
/// held pin submits nothing.
pub struct KeepAlive {
    interval: Duration,
    /// The engine's last GPU submission; `None` after a failed tick, which
    /// stops the ticks until the engine submits again.
    last_gpu: Option<Instant>,
}

impl KeepAlive {
    pub fn new(interval: Duration, now: Instant) -> Self {
        Self { interval, last_gpu: Some(now) }
    }

    /// Records a GPU submission (a request served, a wake, a tick, a load).
    pub fn touch(&mut self, now: Instant) {
        self.last_gpu = Some(now);
    }

    /// Stops the ticks until the next [`Self::touch`].
    pub fn stop(&mut self) {
        self.last_gpu = None;
    }

    /// When the next tick is due, given what is left of the window at
    /// `now`: `None` when the window is closed (`None` or zero), when the
    /// tick would fall at or after its end, or after a failed tick.
    pub fn next_tick(&self, now: Instant, window: Option<Duration>) -> Option<Instant> {
        let ends = now + window.filter(|left| !left.is_zero())?;
        let at = self.last_gpu? + self.interval;
        (at < ends).then_some(at)
    }
}

/// Waits for the engine's next command until `deadline` (`None`: for as
/// long as it takes), and meanwhile calls `tick` whenever `keep_alive` says
/// one is due inside `window` (read again before every tick, so a pin
/// released by another thread stops the ticks at once). Returns what
/// `recv_timeout` would have for `deadline`: a command the moment it
/// arrives, even between ticks, a timeout once the deadline passed. A tick
/// due together with the deadline yields to it. A failed tick is logged
/// once and stops the ticks until the engine touches `keep_alive` again.
/// The ticks run on the engine thread itself, so they never overlap the
/// engine's own GPU work.
fn recv_keeping_warm<T>(
    rx: &Receiver<T>,
    deadline: Option<Instant>,
    keep_alive: &mut KeepAlive,
    window: impl Fn(Instant) -> Option<Duration>,
    mut tick: impl FnMut() -> Result<()>,
) -> std::result::Result<T, RecvTimeoutError> {
    loop {
        let now = Instant::now();
        let next_tick = keep_alive
            .next_tick(now, window(now))
            .filter(|&at| deadline.is_none_or(|end| at < end));
        let received = match next_tick.or(deadline) {
            None => rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
            Some(at) => rx.recv_timeout(at.saturating_duration_since(now)),
        };
        match received {
            Err(RecvTimeoutError::Timeout) if next_tick.is_some() => {
                let now = Instant::now();
                if deadline.is_some_and(|end| now >= end) {
                    // Woken late, past the deadline: it goes first.
                    return Err(RecvTimeoutError::Timeout);
                }
                if keep_alive.next_tick(now, window(now)).is_none_or(|at| at > now) {
                    // The window closed (or moved) while waiting.
                    continue;
                }
                match tick() {
                    Ok(()) => keep_alive.touch(Instant::now()),
                    Err(error) => {
                        // A faulted queue; the next request reports it.
                        eprintln!("GPU keep-alive failed: {error:#}");
                        keep_alive.stop();
                    }
                }
            }
            other => return other,
        }
    }
}

/// Stop coordination: `requested` closes the door (no new requests, the
/// engine winds down after the running one); `cancel` is raised once the
/// grace period is over and stops the running request at its next token or
/// prefill chunk.
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
    serde_json::to_vec(&ErrorEnvelope {
        error: ErrorBody { message: message.into(), kind },
    })
    .unwrap_or_else(|_| b"{\"error\":{\"message\":\"error\"}}".to_vec())
}

fn send_json<T: Serialize>(stream: TcpStream, status: u16, value: &T) {
    let body = serde_json::to_vec(value).unwrap_or_default();
    if let Err(error) = http::respond(stream, status, "application/json", &body) {
        eprintln!("response error: {error:#}");
    }
}

fn send_error(
    stream: TcpStream,
    status: u16,
    kind: &'static str,
    message: impl Into<String>,
) {
    if let Err(error) =
        http::respond(stream, status, "application/json", &error_json(kind, message))
    {
        eprintln!("response error: {error:#}");
    }
}

/// Answers a request the engine never sees and logs why, so a client that
/// gives up on a 4xx/5xx can be traced in the server log.
fn refuse(
    stream: TcpStream,
    request: &str,
    status: u16,
    kind: &'static str,
    message: impl Into<String>,
) {
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
        line.extend_from_slice(
            serde_json::to_string(value).unwrap_or_default().as_bytes(),
        );
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
    /// A generation request has arrived and is being parsed and tokenized:
    /// the engine wakes the GPU now (`MetalContext::wake`) so the residency
    /// the first submission after an idle second waits for overlaps that
    /// host work instead of following it. At most one is in the channel.
    Arrival,
}

/// The bookkeeping next to the engine's channel. `--queue` limits the jobs
/// waiting in it, which the channel's capacity no longer does because it
/// also carries control messages: its capacity is the limit plus one
/// arrival plus one stop wake, and this counts the jobs.
struct EngineQueue {
    limit: usize,
    waiting: AtomicUsize,
    arrival_pending: AtomicBool,
}

impl EngineQueue {
    fn new(limit: usize) -> Self {
        Self {
            limit: limit.max(1),
            waiting: AtomicUsize::new(0),
            arrival_pending: AtomicBool::new(false),
        }
    }

    /// The channel's capacity: every job the limit admits plus the two
    /// control messages, so neither can be refused for lack of room.
    fn capacity(&self) -> usize {
        self.limit + 2
    }

    /// Reserves a place for one job; false when the queue is full. A
    /// reservation is given back by [`Self::left`] once the job is received,
    /// or when sending it failed.
    fn admit(&self) -> bool {
        if self.waiting.fetch_add(1, Ordering::AcqRel) < self.limit {
            true
        } else {
            self.left();
            false
        }
    }

    fn left(&self) {
        self.waiting.fetch_sub(1, Ordering::AcqRel);
    }

    /// Whether the caller should send a [`Cmd::Arrival`]: true unless one
    /// is already in the channel.
    fn claim_arrival(&self) -> bool {
        !self.arrival_pending.swap(true, Ordering::AcqRel)
    }

    /// The engine took the arrival (or sending it failed).
    fn arrival_taken(&self) {
        self.arrival_pending.store(false, Ordering::Release);
    }

    /// Whether a job is waiting in the channel (the batch scheduler's cue to
    /// stop a request decoding alone at its next token and admit the job).
    fn has_waiting(&self) -> bool {
        self.waiting.load(Ordering::Acquire) > 0
    }
}

/// Relays one request's output from the engine to the client on the
/// connection's own thread. The connection is watched from here on, while
/// the request waits in the queue and the engine prefills it, not only once
/// the response has started: a failed write, a reset or a confirmed EOF
/// flags cancellation for the engine, which stops at the next prefill chunk
/// or decoded token.
fn relay(
    stream: TcpStream,
    http11: bool,
    rx: Receiver<Out>,
    cancelled: Arc<AtomicBool>,
) {
    let watched = http::Watched::new(stream, http11, cancelled.clone());
    let (status, content_type) = match rx.recv() {
        Ok(Out::Start { status, content_type }) => (status, content_type),
        // The engine ended the request without a response: the client is
        // gone (nothing to say) or something failed before it started.
        _ => {
            if !cancelled.load(Ordering::Relaxed) {
                let body = error_json("server_error", "internal server error");
                if let Err(error) = watched.respond(500, "application/json", &body) {
                    eprintln!("response error: {error:#}");
                }
            }
            return;
        }
    };
    let mut response = match http::ChunkedResponse::start(watched, status, content_type)
    {
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
    /// Requests decoded together at most (`--max-batch` capped by the
    /// model); 1 serves them one at a time.
    max_batch: usize,
    next_id: u64,
    shutdown: Arc<Shutdown>,
    /// Where each finished request's numbers go for `GET /v1/timings`.
    timings: Arc<TimingsLog>,
    /// The paged n-gram table's preload while it runs in the background.
    preload: Option<BackgroundPreload>,
    /// The weights' pin (`--pin-weights`); dropped before the model.
    pin: pin::WeightPin,
}

/// Where a request's text ends up when not streaming.
#[derive(Default)]
struct Collected {
    reasoning: String,
    content: String,
    tool_calls: Vec<ParsedToolCall>,
}

impl<M: LanguageModel> Engine<M> {
    fn load(
        model_dir: &Path,
        options: &ServeOptions,
        shared: &Shared,
        next_id: u64,
    ) -> Result<Self> {
        let Shared { generator, shutdown, timings, queue: _ } = shared.clone();
        // `LILY_KERNEL_PROFILE=1`: per-kernel GPU times per pass, printed
        // after every request (diagnostic; the profile transport serializes
        // dispatches, so throughput under it is not comparable).
        let ctx = if std::env::var_os("LILY_KERNEL_PROFILE").is_some() {
            MetalContext::new_with_profile(true)?
        } else {
            MetalContext::new()?
        };
        let started = Instant::now();
        let model = M::load(
            &ctx,
            model_dir,
            &LoadOptions {
                ngram_storage: options.ngram_storage,
                mtp_drafts: options.mtp_drafts,
                vision: options.vision,
                expert_slots: None,
                expert_usage: None,
                memory_budget: options.memory_budget,
                // The usage the cache measures goes next to the disk tier.
                expert_usage_out: options
                    .disk_cache_dir
                    .as_ref()
                    .map(|d| d.join("expert-usage.json")),
                // Under an expert cache the plan keeps one full session at
                // this context free, and the budget below is exactly that.
                session_context: Some(SessionContext {
                    max_seq: effective_max_seq(options.max_seq, 0),
                    checkpoints: CHECKPOINTS_PER_SESSION,
                }),
            },
        )?;
        let drafts = options.mtp_drafts.min(model.max_drafts());
        let max_batch = options.max_batch.clamp(1, model.max_batch_rows().max(1));
        if options.max_batch > 1 && max_batch == 1 {
            eprintln!(
                "batching: off, {} cannot batch decode steps in this configuration (the expert cache)",
                M::MODEL_ID
            );
        } else if options.max_batch > 1 {
            eprintln!(
                "batching: up to {max_batch} requests decode together{}",
                if max_batch < options.max_batch {
                    format!(
                        " (--max-batch {} capped by what {} batches)",
                        options.max_batch,
                        M::MODEL_ID
                    )
                } else {
                    String::new()
                }
            );
        }
        let tower = model.vision_tower();
        eprintln!(
            "loaded {} in {:.1}s ({:.1} GB resident{}){}",
            M::MODEL_ID,
            started.elapsed().as_secs_f64(),
            ctx.current_allocated() as f64 / 1e9,
            match tower {
                Some(VisionTower::Loaded { bytes, .. }) => {
                    format!(", {:.2} GB of it the vision tower", bytes as f64 / 1e9)
                }
                _ => String::new(),
            },
            if drafts > 0 {
                format!(", speculative decoding with {drafts} drafts per step")
            } else {
                String::new()
            }
        );
        if let Some(tower) = tower {
            eprintln!(
                "vision tower: {}",
                match tower {
                    VisionTower::Loaded { bytes, blocks } => {
                        format!(
                            "loaded ({:.2} GB, {blocks} blocks)",
                            bytes as f64 / 1e9
                        )
                    }
                    VisionTower::Absent => "absent from the checkpoint".to_string(),
                    VisionTower::Off => "off".to_string(),
                }
            );
        }
        // The weights' pin acts only on the buffers the load allocated, so
        // it changes nothing in the plan; it pins at the first request.
        let weight_buffers = model.weight_buffers();
        let (_, pin_bytes) = pin::page_ranges(&weight_buffers);
        let decision = pin::decide(&pin::PinInputs {
            mode: options.pin_weights,
            planned_memory: model.planned_memory(),
            physical_memory: crate::qwen4exp::weights::physical_memory(),
            wire_limit: crate::stats::user_wire_limit(),
            pin_bytes,
            locked_elsewhere: match model.paged_table() {
                Some(table) if options.ngram_lock => table.bytes(),
                _ => 0,
            },
            expert_cache: model.expert_cache_stats().is_some(),
        });
        match &decision {
            pin::PinDecision::Pin => eprintln!(
                "weights: pinning {:.1} GB in memory from the first request on, released {} or under \
                 memory pressure",
                pin_bytes as f64 / 1e9,
                if options.pin_hold_secs > 0 {
                    format!(
                        "after {} without a request",
                        describe_secs(options.pin_hold_secs)
                    )
                } else {
                    "at the unload".to_owned()
                }
            ),
            pin::PinDecision::Skip(why) => eprintln!("pin skipped: {why}"),
        }
        let pin = pin::WeightPin::new(
            weight_buffers,
            decision,
            options.pin_hold_secs,
            crate::stats::pressure_level,
        )?;
        let max_seq =
            effective_max_seq(options.max_seq, model.max_position_embeddings());
        ensure!(max_seq > 1, "max_seq must be at least 2");
        let mut scratch = model.new_scratch_with_capacity(&ctx, max_seq)?;
        warm_up(&ctx, &model, &mut scratch)?;

        let allocated = ctx.current_allocated();
        let working_set = ctx.recommended_working_set();
        let paged = model.paged_storage_bytes();
        let gb = |bytes: usize| bytes as f64 / 1e9;
        let budget = match (options.cache_bytes, model.session_reserve()) {
            (Some(b), _) => b,
            // The expert cache's plan kept exactly one full session free:
            // budget that, not the working-set arithmetic, whose floor would
            // come on top of the plan and eat the reserve it keeps for the
            // system and the page cache the experts stream through.
            (None, Some(reserve)) => {
                eprintln!(
                    "session cache budget: {:.1} GB = one full {max_seq}-token session, planned into the \
                     expert cache's memory; more sessions spill to the disk tier; override with --cache-bytes",
                    gb(reserve as usize),
                );
                reserve as usize
            }
            (None, None) => {
                let (budget, floored) =
                    derive_cache_budget(working_set, allocated, paged);
                eprintln!(
                    "session cache budget: {:.1} GB = {:.1} GB recommended working set - {:.1} GB allocated \
                     (weights, scratch) - {:.1} GB paged weights in the page cache - {:.1} GB headroom for \
                     other applications{}; override with --cache-bytes",
                    gb(budget),
                    gb(working_set),
                    gb(allocated),
                    gb(paged),
                    gb(BUDGET_HEADROOM_BYTES),
                    if floored {
                        format!(
                            ", raised to the {:.1} GB floor",
                            gb(BUDGET_FLOOR_BYTES)
                        )
                    } else {
                        String::new()
                    },
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
        let mut sessions =
            SessionStore::new(budget, options.max_sessions, CHECKPOINTS_PER_SESSION);
        // The expert cache's plan reserved one session with exactly
        // `CHECKPOINTS_PER_SESSION` snapshots; more would outgrow it.
        if model.session_reserve().is_some() {
            if options.decode_checkpoint_tokens > 0 {
                eprintln!(
                    "session cache: no decode checkpoints (the expert cache budgets one session with {CHECKPOINTS_PER_SESSION} checkpoints)"
                );
            }
        } else if options.decode_checkpoint_tokens > 0 {
            sessions = sessions.with_decode_checkpoints(
                options.decode_checkpoint_tokens,
                DECODE_CHECKPOINTS_PER_SESSION,
            );
            eprintln!(
                "session cache: decode checkpoints every {} generated tokens, at most {DECODE_CHECKPOINTS_PER_SESSION} per session{}",
                options.decode_checkpoint_tokens,
                model
                    .session_bytes(4, 1)
                    .zip(model.session_bytes(4, 0))
                    .map(|(with, without)| format!(
                        " ({:.0} MB each)",
                        with.saturating_sub(without) as f64 / 1e6
                    ))
                    .unwrap_or_default()
            );
        }
        if let (Some(dir), true) =
            (&options.disk_cache_dir, options.disk_cache_bytes > 0)
        {
            match model.persistence_format() {
                Some(format) => {
                    let disk = disk::DiskStore::open(
                        dir,
                        &format,
                        options.disk_cache_bytes,
                        options.disk_cache_ttl_secs,
                    )?;
                    eprintln!(
                        "session cache: disk tier at {} ({} entries, {} durable, {:.1}/{:.1} GB, entries expire after {}; \
                         durable prefix entries {})",
                        disk.dir().display(),
                        disk.len(),
                        disk.durable_len(),
                        disk.used_bytes() as f64 / 1e9,
                        disk.budget_bytes() as f64 / 1e9,
                        if disk.max_age_secs() == 0 {
                            "never".to_owned()
                        } else {
                            format!(
                                "{:.1} days unused",
                                disk.max_age_secs() as f64 / 86_400.0
                            )
                        },
                        if options.durable_min_tokens == 0 {
                            "off".to_owned()
                        } else {
                            format!("from {} shared tokens", options.durable_min_tokens)
                        },
                    );
                    sessions = sessions
                        .with_disk(disk)
                        .with_durable_min_tokens(options.durable_min_tokens);
                    // Under the expert cache the budget is exactly the one
                    // session the plan reserved: every new session evicts the
                    // last one, and the small machine keeps its behaviour.
                    if model.expert_cache_stats().is_none()
                        && model.session_reserve().is_none()
                    {
                        let floor = model
                            .session_bytes(WRITE_AHEAD_FLOOR_TOKENS, 0)
                            .map_or(1 << 30, |b| b as usize);
                        sessions = sessions.with_write_ahead(floor);
                        eprintln!(
                            "session cache: evictions are written ahead in the background between requests, \
                             keeping room for a new session of at least {:.1} GB ({WRITE_AHEAD_FLOOR_TOKENS} tokens) \
                             free of writes",
                            gb(floor)
                        );
                    } else {
                        eprintln!(
                            "session cache: no write ahead (the expert cache budgets one session)"
                        );
                    }
                }
                None => eprintln!(
                    "session cache: {} cannot persist sessions; disk tier off",
                    M::MODEL_ID
                ),
            }
        }
        // The table's preload starts last, once the weights are resident and
        // warm-up is done, and runs while the engine serves: the first
        // requests do not wait for 32 GB of reads, and a row they need before
        // the preload reaches it is read on demand.
        let preload = match model.paged_table() {
            Some(table) if options.ngram_preload || options.ngram_lock => {
                let lock = options.ngram_lock;
                eprintln!(
                    "paged weights: preloading {:.1} GB in the background{}",
                    table.bytes() as f64 / 1e9,
                    if lock { ", locking it" } else { "" }
                );
                Some(table.preload_in_background(lock, move |outcome, secs| {
                    match outcome {
                        Ok(done) => eprintln!(
                            "paged weights: {:.1} GB resident after the background preload {} in {secs:.1}s \
                             ({:.1} GB read){}",
                            done.resident as f64 / 1e9,
                            if done.stopped { "stopped" } else { "finished" },
                            done.read as f64 / 1e9,
                            if lock && !done.stopped { ", locked" } else { "" }
                        ),
                        Err(error) => eprintln!(
                            "paged weights: the background preload failed after {secs:.1}s: {error:#}"
                        ),
                    }
                })?)
            }
            _ => None,
        };
        Ok(Self {
            ctx,
            model,
            generator,
            sessions,
            scratch,
            max_seq,
            drafts,
            max_batch,
            next_id,
            shutdown,
            timings,
            preload,
            pin,
        })
    }

    /// Takes the engine down: every resident session goes to the disk tier
    /// (or is lost when there is none), then the scratch, the sessions, the
    /// model and finally the Metal context are dropped, which releases the
    /// GPU buffers and the n-gram mapping. Returns the request counter so
    /// ids stay unique across a reload.
    fn unload(self, reason: &str) -> u64 {
        let started = Instant::now();
        let Engine {
            ctx,
            model,
            generator: _,
            mut sessions,
            scratch,
            next_id,
            preload,
            pin,
            ..
        } = self;
        // Unlocked before any buffer is released: the pin holds handles on
        // the weights, so they cannot be freed while locked.
        drop(pin);
        // A preload still running holds the table's mappings; it stops at its
        // next 8 MB piece and logs how far it got.
        if let Some(mut preload) = preload {
            preload.stop();
        }
        let resident = ctx.current_allocated();
        // A faulted queue cannot run the snapshot blits, and what its last
        // command buffers left in the caches is not trustworthy either: the
        // sessions are dropped and clients re-prefill (the disk tier's
        // earlier copies were written by a healthy queue and stay valid).
        let faulted = ctx.fault().is_some();
        let (spilled, dropped) =
            if faulted { (0, sessions.drop_all()) } else { sessions.spill_all(&ctx) };
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
            if faulted {
                " without spilling (GPU faulted; their state is not trustworthy)"
            } else {
                ""
            },
            left as f64 / 1e9,
        );
        next_id
    }

    /// Between requests: lets the session cache write the next evictions
    /// ahead (see [`SessionStore::write_ahead`]). Returns whether a write is
    /// in flight, which the engine loop then polls for.
    fn write_ahead(&mut self) -> bool {
        self.sessions.write_ahead(&self.ctx)
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
        let Opened { p, mut sink, in_flight, queued } = self.open(job);
        let stream = p.stream;
        let result = self.run(p, &mut sink, queued, in_flight.pinned());
        let ending = match &result {
            Ok(()) => Ending::Answered,
            Err(error) => Ending::Failed(error),
        };
        self.close(sink, in_flight, stream, ending)
    }

    /// One request alone: its admission with the prefill straight through
    /// ([`Engine::admit`]), the production decode loop
    /// ([`Generator::generate_checkpointed`], speculation and parking
    /// included), then its answer ([`Engine::answer`]). `queued` and `pinned`
    /// go to its timings.
    fn run(
        &mut self,
        p: Prepared,
        sink: &mut Sink,
        queued: Duration,
        pinned: bool,
    ) -> Result<()> {
        // Work on the user's behalf for as long as the request runs.
        let _activity = crate::activity::Activity::begin("lily: serving a request");
        if sink.cancelled() {
            return Ok(());
        }
        // The parser borrows the tokenizer for the request's life: not
        // through `self`, which admission and the answer take.
        let generator = self.generator.clone();
        let Some(mut admitted) =
            self.admit(p, sink, queued, pinned, &generator, &mut Alone)?
        else {
            return Ok(());
        };

        let Engine { ctx, model, scratch, drafts, shutdown, .. } = self;
        let n = admitted.facts.n;
        let Admitted { p, session, decode_checkpoints, parser, out, .. } =
            &mut admitted;
        let options = GenerateOptions {
            max_tokens: p.max_tokens,
            sampling: &p.sampling,
            stop_tokens: &[],
            drafts: *drafts,
        };
        let checkpointer: &mut dyn DecodeCheckpointer<M::State> = decode_checkpoints;
        let decode_started = Instant::now();
        let generation = generator.generate_checkpointed(
            ctx,
            model,
            &mut session.state,
            scratch,
            &p.prompt[n - 1..],
            &options,
            Some(checkpointer),
            &mut |token| {
                let events = parser.push(token)?;
                out.deliver(events, sink);
                Ok(!sink.cancelled() && !parser.stopped && !shutdown.cancel())
            },
        )?;
        self.answer(
            admitted,
            sink,
            Decoded {
                tokens: &generation.tokens,
                finish: generation.finish,
                drafted: generation.drafted,
                accepted: generation.accepted,
                started: decode_started,
                batch: None,
            },
            0,
        )
    }
}

/// The tokens that open a user turn in the chat template,
/// `<|im_start|>user\n`, encoded as the chat prompt encodes the template's
/// own opener (the marker as the special token, then the text; for
/// Qwen3.8-Flash-Next that is `[248045, 846, 198]`, the golden prompts'
/// ids). Message text cannot produce it: its special spellings are encoded
/// as text ([`crate::tokenizer::Tokenizer::encode_chat`]). Empty, which
/// turns the durable boundary's snap off, when the vocabulary has no
/// `<|im_start|>`.
fn user_turn_opener(tokenizer: &crate::tokenizer::Tokenizer) -> Vec<u32> {
    let Some(start) = tokenizer.token_id("<|im_start|>") else {
        return Vec::new();
    };
    match tokenizer.encode("<|im_start|>user\n") {
        Ok(ids) if ids.len() > 1 && ids[0] == start => ids,
        _ => Vec::new(),
    }
}

/// How the session cache reused a lineage behind its live end, for the
/// request's log line: cut back in place (by how many tokens) or forked.
fn describe_reuse(cut_back: Option<usize>, forked: bool) -> String {
    match (cut_back, forked) {
        (Some(dropped), _) => format!(", cut back by {dropped}"),
        (None, true) => ", forked".to_owned(),
        (None, false) => String::new(),
    }
}

/// A response id, `chatcmpl-<created>-<n>` or `cmpl-…`, unique within the
/// process; advances the counter.
fn response_id(kind: Kind, created: u64, next_id: &mut u64) -> String {
    let id = format!(
        "{}-{created}-{}",
        if kind == Kind::Chat { "chatcmpl" } else { "cmpl" },
        *next_id
    );
    *next_id += 1;
    id
}

fn call_id(request_id: &str, index: usize) -> String {
    let digest = request_id.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    });
    format!("call_{:016x}{index:02}", digest)
}

fn chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish_reason: Option<&str>,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    })
}

fn text_chunk(
    id: &str,
    created: u64,
    model: &str,
    text: &str,
    finish_reason: Option<&str>,
) -> Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason, "logprobs": null}],
    })
}

/// Runs a throwaway two-token prompt and two decode steps, so the first real
/// request does not pay the first-use costs: building the pipelines a short
/// request needs (kernels only long contexts reach, sparse attention, still
/// build on first use) and the first submissions after the weights were
/// allocated, which wait while their residency is established. Building is
/// cheap once Metal's shader cache holds the libraries (56 pipelines in
/// 20 ms on the four-layer checkpoint; about 1.5 s for every source cold);
/// the first submission is the larger part (about 10 ms per GB allocated
/// on an idle machine). The log line splits the time up to tell which
/// dominates on this machine, with the system's paging over it.
fn warm_up<M: LanguageModel>(
    ctx: &MetalContext,
    model: &M,
    scratch: &mut M::Scratch,
) -> Result<()> {
    let started = Instant::now();
    let (built_before, build_secs_before) = ctx.build_stats();
    let counters_before = crate::stats::counters();
    let vm_before = crate::stats::vm_counters();
    let greedy = SamplingParams::greedy();
    let mut state = model.new_state(ctx, 4)?;
    scratch.begin_request();
    // Two arbitrary in-vocabulary ids: the values do not matter, only that the
    // prefill and decode graphs get encoded once.
    model.prefill(
        ctx,
        &mut state,
        scratch,
        &[1, 2],
        Some(Draw { params: &greedy, step: 0 }),
    )?;
    let decode_started = Instant::now();
    for (step, (slot_in, slot_out)) in [(0, 1), (1, 0)].into_iter().enumerate() {
        let token = scratch.next_token().view(slot_in, &[1])?.to_u32()?[0];
        let encoded = model.encode_decode_step(
            ctx,
            &state,
            scratch,
            slot_in,
            slot_out,
            Draw { params: &greedy, step: step + 1 },
        )?;
        model.prepare_step_inputs(&mut state, scratch, token)?;
        let pending = encoded.commit()?;
        state.advance(1);
        pending.wait()?;
    }
    let decode_secs = decode_started.elapsed().as_secs_f64();
    // Snapshot/restore compile no shaders but exercise the blit path once.
    let snapshot = state.snapshot(ctx)?;
    // The session cache persists a state at rest from its own buffers
    // (`live_layout`, what the write ahead and a spill write as the live
    // end) instead of a snapshot of it; the two must be the same bytes, or a
    // session read back from disk would resume from another recurrent state.
    // Checked before the restore, which resets the conv window slot.
    if model.persistence_format().is_some() {
        let (live, taken) = (state.live_layout()?, snapshot.layout()?);
        // SAFETY: the state and the snapshot are borrowed here and the GPU
        // is idle on both (the snapshot's blit was waited for).
        ensure!(
            unsafe { crate::engine::layouts_equal(&live, &taken) },
            "the live recurrent state does not lay out like its snapshot"
        );
    }
    state.restore(ctx, &snapshot)?;
    // The expert cache's plan reserves a session from the shapes alone
    // (`LanguageModel::session_bytes`); a drift from what a real state and
    // snapshot hold would silently under- or over-reserve.
    if let (Some(live), Some(with_one)) =
        (model.session_bytes(4, 0), model.session_bytes(4, 1))
    {
        let (state_bytes, snapshot_bytes) =
            (state.bytes() as u64, snapshot.bytes() as u64);
        if live != state_bytes || with_one - live != snapshot_bytes {
            eprintln!(
                "warning: the session size formula gives {live} B per state and {} B per snapshot, \
                 the loaded model holds {state_bytes} and {snapshot_bytes}; the expert cache's session \
                 reserve is off by that much",
                with_one - live
            );
        }
    }
    let (built, build_secs) = ctx.build_stats();
    let prefill = crate::stats::counters().since(counters_before).prefill;
    let vm = crate::stats::vm_counters()
        .zip(vm_before)
        .map(|(after, before)| after.since(before));
    eprintln!(
        "warm-up done in {:.1}s: {} pipelines built in {:.2}s; prefill encode {:.2}s, GPU {:.2}s, \
         waiting {:.2}s; two decode steps {:.2}s{}",
        started.elapsed().as_secs_f64(),
        built - built_before,
        build_secs - build_secs_before,
        prefill.encode_secs,
        prefill.gpu_secs,
        prefill.wait_secs,
        decode_secs,
        vm.map(|v| format!(
            "; system paging meanwhile: pageins {} swapins {} swapouts {} compressions {} decompressions {}",
            v.pageins, v.swapins, v.swapouts, v.compressions, v.decompressions
        ))
        .unwrap_or_default(),
    );
    Ok(())
}

// --- startup -----------------------------------------------------------------

/// The `model_type` a checkpoint's `config.json` declares.
pub fn checkpoint_model_type(model_dir: &Path) -> Result<String> {
    let config = read_config(model_dir)?;
    config.get("model_type").and_then(|v| v.as_str()).map(str::to_string).with_context(
        || format!("{} has no model_type", model_dir.join("config.json").display()),
    )
}

fn read_config(model_dir: &Path) -> Result<Value> {
    let path = model_dir.join("config.json");
    let bytes =
        std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
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
        Some(Value::Array(items)) => {
            items.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect()
        }
        _ => Vec::new(),
    })
}

/// `max_position_embeddings` from `config.json` (top level or `text_config`);
/// zero when the checkpoint does not declare one.
fn checkpoint_max_position_embeddings(model_dir: &Path) -> Result<usize> {
    let config = read_config(model_dir)?;
    Ok(config
        .get("max_position_embeddings")
        .or_else(|| {
            config.get("text_config").and_then(|t| t.get("max_position_embeddings"))
        })
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

/// Whether the checkpoint carries a vision tower lily can load: the
/// converter's `lily.vision` block in `config.json` (the engine checks the
/// same block against the weights when it loads).
fn checkpoint_has_vision_tower(model_dir: &Path) -> Result<bool> {
    let config = read_config(model_dir)?;
    Ok(config.get("lily").and_then(|l| l.get("vision")).is_some_and(Value::is_object))
}

/// What the HTTP threads may accept as image content, decided before the
/// engine loads from the same facts the engine's load applies: the flag and
/// the checkpoint. A request with an image is refused with the reason
/// otherwise, and never reaches the engine.
fn image_policy(model_dir: &Path, options: &ServeOptions) -> Result<ImagePolicy> {
    let available = if options.vision == VisionMode::Off {
        Err("the vision tower is not loaded (--vision off)".to_owned())
    } else if !checkpoint_has_vision_tower(model_dir)? {
        Err("this checkpoint carries no vision tower; images are not supported"
            .to_owned())
    } else {
        Ok(())
    };
    ensure!(
        options.image_min_pixels > 0
            && options.image_min_pixels <= options.image_max_pixels,
        "--image-min-pixels ({}) must be positive and at most --image-max-pixels ({})",
        options.image_min_pixels,
        options.image_max_pixels
    );
    Ok(ImagePolicy {
        limits: ImageLimits {
            max_pixels: options.image_max_pixels,
            min_pixels: options.image_min_pixels,
            ..ImageLimits::default()
        },
        available,
        memo: Default::default(),
    })
}

/// Sampling defaults: `generation_config.json` over OpenAI's defaults, then
/// the command-line overrides.
fn sampling_defaults(
    model_dir: &Path,
    overrides: &SamplingOverrides,
) -> Result<SamplingParams> {
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
        let cfg: Value =
            serde_json::from_slice(&bytes).context("parsing generation_config.json")?;
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
    // One lily per machine, taken before the port is bound so a second
    // server is refused with the holder's pid rather than "address in use",
    // and held for the process's life: the idle unload and the reloads keep
    // it (`crate::instance`).
    crate::instance::acquire()?;
    // Before any other thread exists, so they all inherit the mask and the
    // stop signals only ever reach the thread that waits for them.
    let signals = signal::Signals::block()?;
    match checkpoint_model_type(model_dir)?.as_str() {
        "qwen4_exp" => run_with::<Qwen4ExpModel>(model_dir, options, signals),
        other => anyhow::bail!(
            "unsupported model_type {other:?}; lily serves qwen4_exp \
             (Qwen3.8-Flash-Next)"
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
    queue: Arc<EngineQueue>,
}

/// Everything the HTTP thread needs without the engine.
struct Front {
    shared: Shared,
    defaults: Defaults,
    /// What chat requests may carry as images.
    images: ImagePolicy,
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
/// with status 1 when a load fails (75 when another instance refused it), so
/// a supervisor restarts it (with its backoff) instead of leaving a server up
/// that can never answer.
fn engine_loop<M: LanguageModel>(
    model_dir: &Path,
    options: &ServeOptions,
    shared: Shared,
    rx: Receiver<Cmd>,
    lifecycle: &Lifecycle,
    address: SocketAddr,
) {
    let Shared { shutdown, queue, .. } = &shared;
    // Exits with `code` after logging `what` and refusing the queued
    // requests with `status`/`client_message`.
    let exit_failed = |what: String,
                       status: u16,
                       client_message: &str,
                       rx: &Receiver<Cmd>,
                       code: u8|
     -> ! {
        eprintln!("{what}");
        lifecycle.set(State::Stopping);
        let refused = refuse_queued(rx, queue, status, client_message);
        if refused > 0 {
            eprintln!("exiting: refused {refused} queued requests with {status}");
        }
        std::process::exit(code.into())
    };
    // A load that another instance refused exits 75 like a refused start
    // (`crate::instance`); every other failure 1.
    let fatal = |what: &str, error: anyhow::Error, rx: &Receiver<Cmd>| -> ! {
        let code = crate::instance::exit_status(&error);
        exit_failed(
            format!("{what}: {error:#}"),
            500,
            "the model failed to load",
            rx,
            code,
        )
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
            format!(
                " (unloading after {} idle)",
                describe_secs(options.idle_unload_secs)
            )
        } else {
            String::new()
        }
    );
    let mut idle = IdleTimer::new(options.idle_unload_secs, Instant::now());
    let mut keep_alive = KeepAlive::new(KEEP_ALIVE_INTERVAL, Instant::now());
    while !shutdown.requested() {
        // An unloaded engine has nothing to time out; wait for a request.
        // A loaded one wakes for the idle unload or the end of the pin's hold.
        // A write ahead in flight is polled for, so its entry is indexed and
        // the next one started while the engine is still idle. While the
        // pin is held, the wait also keeps the GPU's residency warm.
        let now = Instant::now();
        let cmd = match engine.as_ref() {
            None => recv_keeping_warm(&rx, None, &mut keep_alive, |_| None, || Ok(())),
            Some(loaded) => {
                let writing = loaded.sessions.writing().then_some(WRITE_AHEAD_POLL);
                let deadline =
                    [idle.remaining(now), loaded.pin.hold_remaining(now), writing]
                        .into_iter()
                        .flatten()
                        .min()
                        .map(|wait| now + wait);
                recv_keeping_warm(
                    &rx,
                    deadline,
                    &mut keep_alive,
                    |now| loaded.pin.keep_warm_remaining(now),
                    || loaded.ctx.wake(),
                )
            }
        };
        match cmd {
            Ok(Cmd::Job(job)) => {
                queue.left();
                if shutdown.requested() {
                    job.reject(503, "the server is shutting down");
                    break;
                }
                if engine.is_none() {
                    lifecycle.set(State::Reloading);
                    let started = Instant::now();
                    match Engine::<M>::load(model_dir, options, &shared, next_id) {
                        Ok(loaded) => {
                            eprintln!(
                                "reloaded {} in {:.1}s for a waiting request",
                                M::MODEL_ID,
                                started.elapsed().as_secs_f64()
                            );
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
                    eprintln!(
                        "debug: injecting a Metal fault on request {served} (--debug-inject-metal-fault)"
                    );
                    engine.as_ref().expect("engine loaded").inject_fault();
                }
                let loaded = engine.as_mut().expect("engine loaded");
                // Batching off: one request at a time, exactly as before.
                // On: this job and every job that arrives while anything is
                // active, decoded together (`batch`).
                let fault = if loaded.max_batch > 1 {
                    loaded.serve_batched(*job, &rx, queue)
                } else {
                    loaded.serve(*job)
                };
                idle.touch(Instant::now());
                keep_alive.touch(Instant::now());
                if fault.is_none() {
                    engine.as_mut().expect("engine loaded").write_ahead();
                }
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
                            1,
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
                            keep_alive.touch(Instant::now());
                        }
                        Err(error) => fatal(
                            "engine failed to reload after a GPU fault",
                            error,
                            &rx,
                        ),
                    }
                }
            }
            Ok(Cmd::Wake) => {}
            Ok(Cmd::Arrival) => {
                queue.arrival_taken();
                if let Some(loaded) = engine.as_ref() {
                    match loaded.ctx.wake() {
                        Ok(()) => keep_alive.touch(Instant::now()),
                        // A faulted queue; the request that follows reports it.
                        Err(error) => {
                            eprintln!("GPU wake at request arrival failed: {error:#}")
                        }
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if let Some(loaded) = engine.as_mut() {
                    loaded.pin.release_if_held_out(Instant::now());
                    loaded.write_ahead();
                }
                if let Some(loaded) = engine.take_if(|_| idle.expired(Instant::now())) {
                    next_id = loaded.unload(&format!(
                        "idle for {}",
                        describe_secs(options.idle_unload_secs)
                    ));
                    lifecycle.set(State::Idle);
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    lifecycle.set(State::Stopping);
    let refused = refuse_queued(&rx, queue, 503, "the server is shutting down");
    if refused > 0 {
        eprintln!("stopping: refused {refused} queued requests");
    }
    if let Some(loaded) = engine {
        loaded.unload("stopping");
    }
}

/// Answers every job still in the channel with `status` and skips the
/// control messages between them; returns how many were refused.
fn refuse_queued(
    rx: &Receiver<Cmd>,
    queue: &EngineQueue,
    status: u16,
    message: &str,
) -> usize {
    let mut refused = 0usize;
    while let Ok(cmd) = rx.try_recv() {
        match cmd {
            Cmd::Job(job) => {
                queue.left();
                job.reject(status, message);
                refused += 1;
            }
            Cmd::Arrival => queue.arrival_taken(),
            Cmd::Wake => {}
        }
    }
    refused
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

fn run_with<M: LanguageModel + 'static>(
    model_dir: &Path,
    options: ServeOptions,
    signals: signal::Signals,
) -> Result<()> {
    let address = options
        .bind
        .to_socket_addrs()
        .with_context(|| format!("resolving bind address {}", options.bind))?
        .next()
        .with_context(|| {
            format!("bind address {} resolved to nothing", options.bind)
        })?;
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
        defaults
            .reasoning_effort
            .as_deref()
            .map(|e| format!(" (effort {e})"))
            .unwrap_or_default(),
    );
    // Decided here, before any thread exists, so a bad flag fails the start
    // instead of a server that is already loading.
    let images = image_policy(model_dir, &options)?;
    eprintln!(
        "images: {}",
        match &images.available {
            Ok(()) => format!(
                "PNG and JPEG data URIs, {} to {} pixels after resizing",
                images.limits.min_pixels, images.limits.max_pixels
            ),
            Err(why) => format!("refused ({why})"),
        }
    );
    let max_seq = effective_max_seq(
        options.max_seq,
        checkpoint_max_position_embeddings(model_dir)?,
    );
    if max_seq < options.max_seq {
        eprintln!(
            "context: --max-seq {} capped to {max_seq} (the checkpoint's window or the kernel limit)",
            options.max_seq
        );
    }

    let listener = TcpListener::bind(address)
        .with_context(|| format!("binding http://{}", options.bind))?;
    let queue = Arc::new(EngineQueue::new(options.queue));
    let (jobs, job_rx) = mpsc::sync_channel::<Cmd>(queue.capacity());
    let lifecycle = Arc::new(Lifecycle::new(State::Loading));
    let shutdown = Arc::new(Shutdown::default());
    let shared = Shared {
        generator,
        shutdown: shutdown.clone(),
        timings: Arc::new(TimingsLog::new(TIMINGS_LOG_CAPACITY)),
        queue,
    };
    let engine_thread = {
        let model_dir = model_dir.to_path_buf();
        let options = options.clone();
        let lifecycle = lifecycle.clone();
        let shared = shared.clone();
        std::thread::Builder::new()
            .name("lily-engine".into())
            .spawn(move || {
                engine_loop::<M>(
                    &model_dir, &options, shared, job_rx, &lifecycle, address,
                )
            })
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
        images,
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
        if let Err(error) =
            std::thread::Builder::new().name("lily-http".into()).spawn(move || {
                handle(&front, stream);
                front.connections.fetch_sub(1, Ordering::AcqRel);
            })
        {
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
            refuse(
                stream,
                "request",
                400,
                "invalid_request_error",
                format!("{error:#}"),
            );
            return;
        }
    };
    let path = request.path.split('?').next().unwrap_or("").to_string();
    let what = format!("{} {path}", request.method);
    let state = if front.shared.shutdown.requested() {
        State::Stopping
    } else {
        front.lifecycle.get()
    };
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
            send_json(
                stream,
                200,
                &json!({"object": "list", "data": front.shared.timings.recent()}),
            );
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
            // Before parsing and tokenizing, so the GPU's return from idle
            // runs meanwhile. Only a loaded engine has a GPU to wake; a busy
            // one takes the message after its request, a no-op then.
            let queue = &front.shared.queue;
            if state == State::Ready
                && queue.claim_arrival()
                && front.jobs.try_send(Cmd::Arrival).is_err()
            {
                queue.arrival_taken();
            }
            let prepare_started = Instant::now();
            let prepared = if path == "/v1/chat/completions" {
                serde_json::from_slice::<api::ChatRequest>(&request.body)
                    .context("parsing the chat request")
                    .and_then(|r| {
                        api::prepare_chat(
                            r,
                            front.shared.generator.tokenizer(),
                            &front.defaults,
                            front.max_seq,
                            &front.images,
                        )
                    })
            } else {
                serde_json::from_slice::<api::CompletionRequest>(&request.body)
                    .context("parsing the completion request")
                    .and_then(|r| {
                        api::prepare_completion(
                            r,
                            front.shared.generator.tokenizer(),
                            &front.defaults,
                            front.max_seq,
                        )
                    })
            };
            let mut prepared = match prepared {
                Ok(p) => p,
                Err(error) => {
                    refuse(
                        stream,
                        &what,
                        400,
                        "invalid_request_error",
                        format!("{error:#}"),
                    );
                    return;
                }
            };
            prepared.prepare_secs = prepare_started.elapsed().as_secs_f64();
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
                    refuse(
                        stream,
                        &what,
                        503,
                        "server_error",
                        "the server is shutting down",
                    );
                    return;
                }
                State::Loading => {
                    refuse(
                        stream,
                        &what,
                        503,
                        "server_error",
                        "model is still loading",
                    );
                    return;
                }
                State::Ready | State::Idle | State::Reloading | State::Recovering => {}
            }
            let (tx, rx) = mpsc::channel();
            let cancelled = Arc::new(AtomicBool::new(false));
            let job = Job {
                prepared,
                sink: Sink { tx, cancelled: cancelled.clone(), started: false },
                queued_at: Instant::now(),
            };
            if !queue.admit() {
                refuse(
                    stream,
                    &what,
                    503,
                    "server_error",
                    "the request queue is full; retry later",
                );
                return;
            }
            match front.jobs.try_send(Cmd::Job(Box::new(job))) {
                Ok(()) => relay(stream, request.http11, rx, cancelled),
                Err(TrySendError::Full(_)) => {
                    queue.left();
                    refuse(
                        stream,
                        &what,
                        503,
                        "server_error",
                        "the request queue is full; retry later",
                    );
                }
                Err(TrySendError::Disconnected(_)) => {
                    queue.left();
                    refuse(stream, &what, 500, "server_error", "engine stopped");
                }
            }
        }
        (
            _,
            "/health"
            | "/v1/models"
            | "/v1/timings"
            | "/v1/chat/completions"
            | "/v1/completions",
        ) => {
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
                let rc = libc::pthread_sigmask(
                    libc::SIG_BLOCK,
                    set.as_ptr(),
                    std::ptr::null_mut(),
                );
                ensure!(
                    rc == 0,
                    "blocking SIGTERM/SIGINT failed: {}",
                    std::io::Error::from_raw_os_error(rc)
                );
                set.assume_init()
            };
            Ok(Self { set })
        }

        /// Runs `on_signal` with the signal's name on a new thread each
        /// time one of the blocked signals arrives.
        pub fn spawn_handler(
            self,
            mut on_signal: impl FnMut(&'static str) + Send + 'static,
        ) -> Result<()> {
            std::thread::Builder::new()
                .name("lily-signals".into())
                .spawn(move || {
                    loop {
                        let mut signal = 0;
                        // SAFETY: `set` is a valid, initialised signal set
                        // and `signal` a valid out-pointer.
                        let rc = unsafe { libc::sigwait(&self.set, &mut signal) };
                        if rc != 0 {
                            eprintln!(
                                "sigwait failed: {}",
                                std::io::Error::from_raw_os_error(rc)
                            );
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

/// The per-kernel GPU profile of the passes recorded since the last call,
/// aggregated per pass label (decode, verify, draft, prefill): the twelve
/// largest kernels by ms per pass. `LILY_KERNEL_PROFILE=1`.
fn print_kernel_profile(passes: &[crate::metal::profile::PassProfile]) {
    let mut labels: Vec<&'static str> = Vec::new();
    for pass in passes {
        if !labels.contains(&pass.label) {
            labels.push(pass.label);
        }
    }
    for label in labels {
        let group: Vec<_> = passes.iter().filter(|p| p.label == label).collect();
        let n = group.len() as f64;
        let mut by_kernel: std::collections::HashMap<&'static str, (usize, f64)> =
            std::collections::HashMap::new();
        let (mut kernel_secs, mut span_secs) = (0.0f64, 0.0f64);
        for pass in &group {
            span_secs += pass.span_secs;
            for k in &pass.kernels {
                kernel_secs += k.gpu_secs;
                let e = by_kernel.entry(k.name).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += k.gpu_secs;
            }
        }
        let mut rows: Vec<_> = by_kernel.into_iter().collect();
        rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        eprintln!(
            "kernel profile [{label}]: {} passes, kernel sum {:.2} ms/pass, span {:.2} ms/pass",
            group.len(),
            1e3 * kernel_secs / n,
            1e3 * span_secs / n
        );
        for (name, (calls, secs)) in rows.iter().take(12) {
            eprintln!(
                "  {:>9.3} ms  {:>7.1} calls  {name}",
                1e3 * secs / n,
                *calls as f64 / n
            );
        }
    }
}
