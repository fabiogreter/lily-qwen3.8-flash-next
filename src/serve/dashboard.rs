//! The dashboard (docs/dashboard.md): the Vue app built into
//! `dashboard/dist/` at `/`, and the read-only `/v1/lily/*` routes it polls.
//!
//! **The dashboard never wakes the engine** (§ 2.2). No route here sends the
//! engine a command, claims an arrival, touches the GPU, takes the activity
//! assertion or moves the idle timer or the pin's hold. What it shows of the
//! engine is what the engine **publishes** into the [`Board`] at points where
//! that state changes anyway: a request taken on, admitted, answered and
//! closed, the session cache after an acquire, a release, a write ahead or a
//! spill, the load and the unload. Each publish is a few small allocations
//! and a lock held for a pointer swap or a push, never across GPU work; JSON
//! is rendered on the HTTP thread, never on the engine's (§ 2.3). The live
//! token rates come from two counters the engine bumps (a relaxed atomic add
//! per drawn token, one per prefill chunk), sampled by the HTTP threads.
//! Machine figures (CPU load, memory, swap, pressure) are read on the HTTP
//! thread directly.
//!
//! **This machine only** (§ 3.2). Every dashboard route refuses with 403 a
//! peer that is not loopback, whatever `--bind` says, and a `Host` that is
//! not lily's own loopback address (a DNS-rebinding page reaches loopback
//! under a name of its own). The settings writes also need
//! `Content-Type: application/json` and, when an `Origin` is sent, lily's
//! own origin, which a cross-site page cannot fake without a CORS preflight
//! that lily never answers. Every answer forbids framing and carries a
//! content security policy the built app runs under.
//!
//! **Numbers only** (§ 2.1): nothing here holds prompt text, generated text,
//! token ids, image data or tool arguments; the request rows are the counts,
//! durations and rates of [`Timings`].
//!
//! What phase 2 cannot answer yet: the history routes need the recorder
//! (503), the settings writes need runtime settings (409, after the write
//! guard), the per-session figures cover the requests still in the
//! [`REQUEST_ROWS`] ring, and the machine's power, GPU and temperatures wait
//! for the sampler (`machine.sampler` is `off`).

pub mod contract;
pub mod settings;

use std::borrow::Cow;
use std::collections::{HashSet, VecDeque};
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use contract::{
    CancelledBy, DiskTier, ErrorBody, ErrorDetail, ExpertTier, GpuTier, LifecycleState,
    LiveStatus, MachineStatus, MemoryPressure, NgramTier, QueueStatus, RecorderStatus,
    RequestRow, RequestsResponse, SamplerState, Session, SessionTier, SessionsResponse,
    Setting, SettingsLogResponse, SettingsResponse, Status, Tiers, WeightsTier,
};
use serde::Serialize;

use super::http::Request;
use super::session::{EntryView, Lineage, Lineages, SessionStore};
use super::timings::Timings;
use crate::engine::LanguageModel;
use crate::qwen4exp::ngram::PagedTable;
use crate::thinking::ClosedBy;

/// The files of `dashboard/dist/`, embedded by `build.rs`.
mod assets {
    include!(concat!(env!("OUT_DIR"), "/dashboard_assets.rs"));
}

/// Request rows the board keeps (newest last): `GET /v1/lily/requests`
/// serves the newest of them and the per-session figures are computed over
/// all of them until the recorder keeps history. About 150 bytes a row.
pub const REQUEST_ROWS: usize = 1024;
/// `GET /v1/lily/requests` without a `limit`.
const DEFAULT_REQUEST_LIMIT: usize = 40;
/// The most rows one `GET /v1/lily/requests` returns.
const MAX_REQUEST_LIMIT: usize = 200;
/// The span the live token rates and the CPU load are measured over: the
/// page polls every 2 s, so a rate covers the last two or three polls.
const LIVE_WINDOW: Duration = Duration::from_secs(6);
/// The shortest span a rate is measured over; a status read sooner after
/// the previous one (another tab) reuses the older sample.
const LIVE_MIN_SPAN: Duration = Duration::from_secs(1);
/// How long a measured n-gram residency is reused: the `mincore` scan over
/// the table's 32 GB takes a few milliseconds.
const NGRAM_RESIDENCY_TTL: Duration = Duration::from_secs(10);
/// The live request rate and acceptance cover this span of finished rows.
const LIVE_ROWS_MS: u64 = 60_000;
/// Peers whose refusal is logged; past this many, refusals are not logged.
const REFUSED_PEERS_LOGGED: usize = 1024;

/// The headers every dashboard answer carries: no MIME sniffing, no framing
/// (clickjacking), and a policy that lets the page load only its own files.
/// The built app has no inline script; it does set inline styles (static
/// `style` attributes in its templates, and Vue's style bindings), so
/// `style-src` allows them. The favicon is a `data:` URI.
pub const SECURITY_HEADERS: &[(&str, &str)] = &[
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    (
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; \
         img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; \
         base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
    ),
];

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Logs `what` the first time `flag` is raised, never again.
fn log_once(flag: &AtomicBool, what: impl FnOnce() -> String) {
    if !flag.swap(true, Ordering::Relaxed) {
        eprintln!("{}", what());
    }
}

// --- what the engine publishes ----------------------------------------------

/// The session cache as the engine last published it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StoreView {
    /// The resident sessions (the requests' sessions checked out are not
    /// among them; the board adds those).
    pub gpu: GpuTier,
    /// `None` without a disk tier.
    pub disk: Option<DiskTier>,
    pub entries: Vec<EntryView>,
    /// Small-machine mode only.
    pub experts: Option<ExpertTier>,
}

impl StoreView {
    /// The view of `sessions` now, with the expert cache's figures for the
    /// model `model` when it serves its experts from one.
    pub fn of<M: LanguageModel>(sessions: &SessionStore<M>, model: Option<&M>) -> Self {
        let disk = sessions.disk().map(|d| {
            let durable = d.entries().iter().filter(|e| e.durable);
            DiskTier {
                used_bytes: d.used_bytes(),
                budget_bytes: d.budget_bytes(),
                entries: d.len(),
                durable_entries: d.durable_len(),
                durable_bytes: durable.map(|e| e.bytes).sum(),
            }
        });
        let experts = model.and_then(|m| {
            let stats = m.expert_cache_stats()?;
            let slots = m.expert_cache_slots()?;
            Some(ExpertTier {
                slots,
                // The load fills every slot (`ExpertCache::fill_and_serve`).
                used: slots,
                // Cumulative since the load.
                hit_rate: (stats.lookups > 0).then(|| {
                    round(1.0 - stats.misses as f64 / stats.lookups as f64, 1e4)
                }),
            })
        });
        Self {
            gpu: GpuTier {
                used_bytes: sessions.used_bytes() as u64,
                budget_bytes: Some(sessions.budget_bytes() as u64),
                entries: sessions.len(),
                max_entries: sessions.max_sessions(),
            },
            disk,
            entries: sessions.entries_view(),
            experts,
        }
    }
}

/// The model's n-gram table, as a load found it.
#[derive(Debug, Clone)]
pub struct NgramFacts {
    pub bytes: u64,
    /// The paged table, by a handle that does not keep its mappings alive
    /// past the unload; `None` for a resident table (all of it in memory).
    pub paged: Option<Weak<PagedTable>>,
}

/// What a load settled that the dashboard shows.
#[derive(Debug, Clone)]
pub struct Loaded {
    /// Requests decoding together at most (`--max-batch` capped by the
    /// model).
    pub max_batch: usize,
    /// The weights' bytes (what `--pin-weights` locks).
    pub weights_bytes: u64,
    /// `None` for a model without an n-gram table.
    pub ngram: Option<NgramFacts>,
}

/// A request's session while the request holds it checked out.
#[derive(Debug, Clone, Copy)]
struct Active {
    lineage: Lineage,
    tokens: usize,
    bytes: u64,
}

#[derive(Debug, Default)]
struct Activity {
    /// `None` while unloaded (or before the first load finished).
    loaded: Option<Loaded>,
    /// The last load's facts, kept across an unload for the read-only parts
    /// (the weights' size, the batch cap).
    last_loaded: Option<Loaded>,
    /// Loads so far: a measurement of the previous load's table is stale.
    loads: u64,
    /// Requests between `opened` and `closed`.
    open: usize,
    active: Vec<Active>,
    last_finished: Option<Instant>,
}

/// The engine's side of the dashboard: what it publishes, shared with the
/// HTTP threads. Every method is cheap and takes its lock only for a swap,
/// a push or a small clone.
#[derive(Debug)]
pub struct Board {
    store: Mutex<Arc<StoreView>>,
    activity: Mutex<Activity>,
    rows: Mutex<VecDeque<RequestRow>>,
    decoded: AtomicU64,
    prefilled: AtomicU64,
    pinned: Arc<AtomicBool>,
    lineages: Arc<Lineages>,
}

impl Default for Board {
    fn default() -> Self {
        Self {
            store: Mutex::new(Arc::new(StoreView::default())),
            activity: Mutex::default(),
            rows: Mutex::new(VecDeque::with_capacity(REQUEST_ROWS)),
            decoded: AtomicU64::new(0),
            prefilled: AtomicU64::new(0),
            pinned: Arc::new(AtomicBool::new(false)),
            lineages: Arc::new(Lineages::default()),
        }
    }
}

impl Board {
    /// The process's lineage counter, which every session store shares.
    pub fn lineages(&self) -> Arc<Lineages> {
        self.lineages.clone()
    }

    /// The flag the weights' pin mirrors its state into.
    pub fn pin_mirror(&self) -> Arc<AtomicBool> {
        self.pinned.clone()
    }

    /// The engine loaded.
    pub fn loaded(&self, loaded: Loaded) {
        let mut activity = lock(&self.activity);
        activity.last_loaded = Some(loaded.clone());
        activity.loaded = Some(loaded);
        activity.loads += 1;
    }

    /// The engine unloaded; `store` is the cache after the spill (the disk
    /// tier as it was left; nothing resident).
    pub fn unloaded(&self, store: StoreView) {
        self.publish(StoreView { experts: None, ..store });
        let mut activity = lock(&self.activity);
        activity.loaded = None;
        activity.active.clear();
    }

    /// The session cache changed.
    pub fn publish(&self, store: StoreView) {
        *lock(&self.store) = Arc::new(store);
    }

    /// A request was taken on.
    pub fn opened(&self) {
        lock(&self.activity).open += 1;
    }

    /// A request's session was acquired: `tokens` of prompt, `bytes` of it
    /// on the GPU.
    pub fn admitted(&self, lineage: Lineage, tokens: usize, bytes: u64) {
        lock(&self.activity).active.push(Active { lineage, tokens, bytes });
    }

    /// A request taken on ended (answered, stopped or failed); `lineage` is
    /// its session's when it was admitted.
    pub fn closed(&self, lineage: Option<u64>, now: Instant) {
        let mut activity = lock(&self.activity);
        activity.open = activity.open.saturating_sub(1);
        if let Some(id) = lineage
            && let Some(i) = activity.active.iter().position(|a| a.lineage.id == id)
        {
            activity.active.swap_remove(i);
        }
        activity.last_finished = Some(now);
    }

    /// A finished request's numbers.
    pub fn record(&self, row: RequestRow) {
        let mut rows = lock(&self.rows);
        if rows.len() == REQUEST_ROWS {
            rows.pop_front();
        }
        rows.push_back(row);
    }

    /// One token drawn (generated or inserted by a thinking control).
    pub fn drew(&self) {
        self.decoded.fetch_add(1, Ordering::Relaxed);
    }

    /// `tokens` prompt tokens prefilled.
    pub fn prefilled(&self, tokens: usize) {
        self.prefilled.fetch_add(tokens as u64, Ordering::Relaxed);
    }

    /// The token counters: drawn, prefilled.
    pub fn token_counts(&self) -> (u64, u64) {
        (self.decoded.load(Ordering::Relaxed), self.prefilled.load(Ordering::Relaxed))
    }

    fn store(&self) -> Arc<StoreView> {
        lock(&self.store).clone()
    }
}

/// `Timings::cancelled_by` as the contract names it; a value the contract
/// does not know is logged once and left out.
fn cancelled_by(by: Option<&str>) -> Option<CancelledBy> {
    static UNKNOWN: AtomicBool = AtomicBool::new(false);
    match by? {
        "client" => Some(CancelledBy::Client),
        "shutdown" => Some(CancelledBy::Shutdown),
        other => {
            log_once(&UNKNOWN, || {
                format!(
                    "dashboard: request rows leave out an unknown cancellation {other:?}"
                )
            });
            None
        }
    }
}

/// A finished request's row: `t` its finish (unix ms), `lineage` its
/// session's, the rest from its `timings`. `thinking` is how its thinking
/// controls closed the reasoning block, if they did.
pub fn request_row(
    t: u64,
    lineage: Option<u64>,
    timings: &Timings,
    thinking: Option<ClosedBy>,
) -> RequestRow {
    RequestRow {
        t,
        lineage,
        prompt_tokens: timings.prompt_tokens,
        cached_tokens: timings.cached_tokens,
        prefill_tokens: timings.prefill_tokens,
        prefill_ms: timings.prefill_ms,
        prefill_tps: timings.prefill_per_second,
        generated_tokens: timings.generated_tokens,
        decode_ms: timings.decode_ms,
        decode_tps: timings.decode_per_second,
        drafted_tokens: timings.drafted_tokens,
        accepted_tokens: timings.accepted_tokens,
        queue_ms: timings.queue_ms,
        // A request that never shared a step decoded alone: a batch of one.
        batch: timings.batch.map_or(1, |b| b.max_rows.max(1)),
        cancelled_by: cancelled_by(timings.cancelled_by),
        // The budget's close, and the end of turn it replaced inside the
        // block; a tool call ending the block is the model's own doing.
        thinking_forced: matches!(
            thinking,
            Some(ClosedBy::Budget | ClosedBy::EndOfTurn)
        ),
        pinned: timings.pinned,
        failed: false,
    }
}

/// The row of a request that failed (an error, a GPU fault): it counts as a
/// request, and of its numbers only the time and the lineage are known.
pub fn failed_row(t: u64, lineage: Option<u64>) -> RequestRow {
    RequestRow {
        t,
        lineage,
        prompt_tokens: 0,
        cached_tokens: 0,
        prefill_tokens: 0,
        prefill_ms: 0.0,
        prefill_tps: None,
        generated_tokens: 0,
        decode_ms: 0.0,
        decode_tps: None,
        drafted_tokens: None,
        accepted_tokens: None,
        queue_ms: 0.0,
        batch: 1,
        cancelled_by: None,
        thinking_forced: false,
        pinned: false,
        failed: true,
    }
}

// --- routing and the guards --------------------------------------------------

/// A request the dashboard answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// `/`, `/assets/*` and every other GET or HEAD outside `/v1` and
    /// `/health`.
    Page,
    /// Anything under `/v1/lily/`.
    Api,
}

/// Whether the dashboard answers `method path` (`path` without its query).
/// The OpenAI routes, `/health` and everything else under `/v1` are not its.
pub fn route(method: &str, path: &str) -> Option<Route> {
    if path == "/v1/lily" || path.starts_with("/v1/lily/") {
        Some(Route::Api)
    } else if matches!(method, "GET" | "HEAD")
        && path.starts_with('/')
        && path != "/v1"
        && !path.starts_with("/v1/")
        && path != "/health"
    {
        Some(Route::Page)
    } else {
        None
    }
}

/// Whether `peer` is this machine (an IPv4-mapped loopback address
/// included, as a dual-stack listener reports it).
pub fn is_loopback(peer: Option<SocketAddr>) -> bool {
    peer.is_some_and(|peer| peer.ip().to_canonical().is_loopback())
}

/// The names lily's own page is reached under on `port`, as `Host` headers
/// carry them (`Origin` adds the scheme).
fn own_hosts(port: u16) -> [String; 3] {
    [format!("127.0.0.1:{port}"), format!("localhost:{port}"), format!("[::1]:{port}")]
}

/// Whether `host` (the `Host` header) names lily's own loopback address on
/// `port`. A page that DNS-rebound its own name to 127.0.0.1 sends that
/// name, and is refused.
pub fn host_allowed(host: Option<&str>, port: u16) -> bool {
    host.map(str::trim).is_some_and(|host| {
        own_hosts(port).iter().any(|own| own.eq_ignore_ascii_case(host))
    })
}

/// The cross-site write guard: `Content-Type` must be the media type
/// `application/json` (parameters such as a charset pass), and an `Origin`,
/// when sent, must be lily's own on `port`. Returns why it refused.
pub fn write_guard(
    content_type: Option<&str>,
    origin: Option<&str>,
    port: u16,
) -> Result<(), &'static str> {
    let media = content_type.and_then(|v| v.split(';').next()).map(str::trim);
    if !media.is_some_and(|m| m.eq_ignore_ascii_case("application/json")) {
        return Err("writes need Content-Type: application/json");
    }
    if let Some(origin) = origin
        && !own_hosts(port).iter().any(|own| origin == format!("http://{own}"))
    {
        return Err("writes are only accepted from lily's own page");
    }
    Ok(())
}

/// A file of the app: its content type, its `Cache-Control` and its bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Asset {
    pub content_type: &'static str,
    pub cache_control: &'static str,
    pub bytes: &'static [u8],
}

const NO_CACHE: &str = "no-cache";
const IMMUTABLE: &str = "public, max-age=31536000, immutable";
/// The one file under `assets/` whose name carries no content hash: its
/// content can change under the same name.
const UNHASHED_ASSET: &str = "licenses.txt";

fn embedded(path: &str) -> Option<(&'static str, &'static [u8])> {
    assets::FILES.iter().find(|(p, _, _)| *p == path).map(|&(_, t, b)| (t, b))
}

/// The file a page route serves, by exact match against the embedded table
/// (nothing is resolved, so `..` and encodings find nothing): `/assets/<file>`
/// (cached for good, its name carries the build's hash), any other path with
/// a file extension as it is, and every other path `index.html` so the app's
/// own routes survive a reload. `None`: a file that does not exist.
pub fn page(path: &str) -> Option<Asset> {
    if let Some(name) = path.strip_prefix("/assets/") {
        let (content_type, bytes) = embedded(&format!("assets/{name}"))?;
        let cache_control = if name == UNHASHED_ASSET { NO_CACHE } else { IMMUTABLE };
        return Some(Asset { content_type, cache_control, bytes });
    }
    let file = path.rsplit('/').next().unwrap_or("");
    let target =
        if file.contains('.') { path.trim_start_matches('/') } else { "index.html" };
    let (content_type, bytes) = embedded(target)?;
    Some(Asset { content_type, cache_control: NO_CACHE, bytes })
}

// --- answers -----------------------------------------------------------------

/// An answer for the connection thread to write.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub cache_control: &'static str,
    /// The app's files are served from the binary as they are.
    pub body: Cow<'static, [u8]>,
}

impl Response {
    fn json<T: Serialize>(status: u16, value: &T) -> Self {
        match serde_json::to_vec(value) {
            Ok(body) => Self {
                status,
                content_type: "application/json",
                cache_control: "no-store",
                body: Cow::Owned(body),
            },
            Err(error) => {
                eprintln!("dashboard: serializing an answer failed: {error}");
                Self {
                    status: 500,
                    content_type: "application/json",
                    cache_control: "no-store",
                    body: Cow::Borrowed(
                        br#"{"error":{"message":"the answer could not be serialized"}}"#,
                    ),
                }
            }
        }
    }

    fn error(status: u16, message: &str, setting: Option<&str>) -> Self {
        Self::json(
            status,
            &ErrorBody {
                error: ErrorDetail {
                    message: message.to_owned(),
                    setting: setting.map(str::to_owned),
                },
            },
        )
    }

    /// The headers besides `Content-Type` and `Content-Length`.
    pub fn headers(&self) -> Vec<(&'static str, &'static str)> {
        let mut headers = SECURITY_HEADERS.to_vec();
        headers.push(("Cache-Control", self.cache_control));
        headers
    }
}

/// The connection a request came in on.
#[derive(Debug, Clone, Copy)]
pub struct Connection {
    /// The remote address (`None` when the socket could not say).
    pub peer: Option<SocketAddr>,
    /// The port lily listens on, which its own page's origin carries.
    pub port: u16,
}

/// What the HTTP thread knows of the engine without asking it.
#[derive(Debug, Clone, Copy)]
pub struct EngineNow {
    pub state: super::State,
    /// Jobs waiting in the engine's channel.
    pub waiting: usize,
}

/// The dashboard's configuration: what the command line fixed.
#[derive(Debug, Clone)]
pub struct Config {
    pub model: &'static str,
    /// `--idle-unload`, 0 for never.
    pub idle_unload_secs: u64,
    /// `--pin-hold`, 0 for until pressure or the unload.
    pub pin_hold_secs: u64,
    /// `--max-batch` or its default, until a load caps it.
    pub max_batch: usize,
    /// `--queue`.
    pub queue: usize,
    /// `--max-sessions`.
    pub max_sessions: usize,
    /// `--cache-bytes`; without it the budget is derived at the load.
    pub cache_bytes: Option<u64>,
    pub settings: Vec<Setting>,
}

#[derive(Debug, Clone, Copy)]
struct Sample {
    at: Instant,
    decoded: u64,
    prefilled: u64,
    cpu: Option<crate::stats::CpuTicks>,
}

/// The HTTP threads' memory between status reads: recent samples of the
/// engine's token counters and the CPU ticks, and the last n-gram residency
/// (which load it measured, when, the share).
#[derive(Debug, Default)]
struct Meter {
    samples: VecDeque<Sample>,
    ngram: Option<(u64, Instant, f64)>,
}

impl Meter {
    /// Adds `now` and returns the oldest sample within [`LIVE_WINDOW`] that
    /// is at least [`LIVE_MIN_SPAN`] older, to measure against.
    fn sample(&mut self, now: Sample) -> Option<Sample> {
        while self
            .samples
            .front()
            .is_some_and(|s| now.at.saturating_duration_since(s.at) > LIVE_WINDOW)
        {
            self.samples.pop_front();
        }
        let base = self
            .samples
            .front()
            .copied()
            .filter(|s| now.at.saturating_duration_since(s.at) >= LIVE_MIN_SPAN);
        // A burst of reads keeps one sample per second at most.
        if self
            .samples
            .back()
            .is_none_or(|s| now.at.saturating_duration_since(s.at) >= LIVE_MIN_SPAN)
        {
            self.samples.push_back(now);
        }
        base
    }
}

/// Tokens per second between two counter readings.
fn rate(from: u64, to: u64, secs: f64) -> f64 {
    if secs > 0.0 { round(to.saturating_sub(from) as f64 / secs, 10.0) } else { 0.0 }
}

fn round(value: f64, scale: f64) -> f64 {
    (value * scale).round() / scale
}

/// `part / whole`, `None` over nothing.
fn ratio(part: u64, whole: u64) -> Option<f64> {
    (whole > 0).then(|| round(part as f64 / whole as f64, 1e4))
}

/// A lineage's figures over the rows the board holds (docs/dashboard.md § 5:
/// averages over busy time, the maximum the best minute).
#[derive(Debug, Default, Clone, PartialEq)]
struct Figures {
    requests: usize,
    decode_tps_avg: Option<f64>,
    decode_tps_max: Option<f64>,
    prefill_tps_avg: Option<f64>,
    cache_hit: Option<f64>,
    acceptance: Option<f64>,
}

fn figures<'a>(rows: impl Iterator<Item = &'a RequestRow>) -> Figures {
    let (mut requests, mut generated, mut decode_ms) = (0usize, 0u64, 0.0f64);
    let (mut prefilled, mut prefill_ms) = (0u64, 0.0f64);
    let (mut cached, mut prompt, mut drafted, mut accepted) = (0u64, 0u64, 0u64, 0u64);
    // Decode tokens and busy milliseconds per minute, for the best minute.
    let mut minutes: Vec<(u64, u64, f64)> = Vec::new();
    for row in rows {
        requests += 1;
        generated += row.generated_tokens as u64;
        decode_ms += row.decode_ms;
        if row.prefill_tokens > 0 {
            prefilled += row.prefill_tokens as u64;
            prefill_ms += row.prefill_ms;
        }
        cached += row.cached_tokens as u64;
        prompt += row.prompt_tokens as u64;
        if let (Some(d), Some(a)) = (row.drafted_tokens, row.accepted_tokens) {
            drafted += d as u64;
            accepted += a.min(d) as u64;
        }
        if row.generated_tokens > 0 && row.decode_ms > 0.0 {
            let minute = row.t / 60_000;
            match minutes.iter_mut().find(|m| m.0 == minute) {
                Some(m) => {
                    m.1 += row.generated_tokens as u64;
                    m.2 += row.decode_ms;
                }
                None => {
                    minutes.push((minute, row.generated_tokens as u64, row.decode_ms))
                }
            }
        }
    }
    let per_second = |tokens: u64, ms: f64| {
        (tokens > 0 && ms > 0.0).then(|| round(tokens as f64 / (ms / 1e3), 100.0))
    };
    Figures {
        requests,
        decode_tps_avg: per_second(generated, decode_ms),
        decode_tps_max: minutes
            .iter()
            .filter_map(|&(_, tokens, ms)| per_second(tokens, ms))
            .reduce(f64::max),
        prefill_tps_avg: per_second(prefilled, prefill_ms),
        cache_hit: ratio(cached, prompt),
        acceptance: ratio(accepted, drafted),
    }
}

/// The HTTP side of the dashboard: answers its routes from the [`Board`],
/// the configuration and the machine.
#[derive(Debug)]
pub struct Dashboard {
    board: Arc<Board>,
    config: Config,
    started: Instant,
    meter: Mutex<Meter>,
    /// Peers whose refusal was logged.
    refused: Mutex<HashSet<IpAddr>>,
}

impl Dashboard {
    /// The dashboard over `board`, which until the first load shows the
    /// session cache's limits from the command line.
    pub fn new(board: Arc<Board>, config: Config) -> Self {
        board.publish(StoreView {
            gpu: GpuTier {
                used_bytes: 0,
                budget_bytes: config.cache_bytes,
                entries: 0,
                max_entries: config.max_sessions,
            },
            ..StoreView::default()
        });
        Self {
            board,
            config,
            started: Instant::now(),
            meter: Mutex::default(),
            refused: Mutex::default(),
        }
    }

    /// Answers `request` on `route`; `path` and `query` are its target's
    /// two halves. Never waits on the engine. A HEAD is answered as its GET
    /// (the caller writes the head only).
    pub fn respond(
        &self,
        route: Route,
        request: &Request,
        (path, query): (&str, &str),
        connection: Connection,
        engine: EngineNow,
    ) -> Response {
        let refusal = if !is_loopback(connection.peer) {
            Some("the dashboard answers this machine only")
        } else if !host_allowed(request.header("Host"), connection.port) {
            Some("the dashboard answers under its own loopback address only")
        } else {
            None
        };
        if let Some(why) = refusal {
            self.log_refusal(connection.peer, request, path, why);
            return Response::error(403, why, None);
        }
        match route {
            Route::Page => match page(path) {
                Some(asset) => Response {
                    status: 200,
                    content_type: asset.content_type,
                    cache_control: asset.cache_control,
                    body: Cow::Borrowed(asset.bytes),
                },
                None => Response::error(404, "not found", None),
            },
            Route::Api => self.api(request, path, query, connection.port, engine),
        }
    }

    /// Logs a refusal the first time a peer address is refused.
    fn log_refusal(
        &self,
        peer: Option<SocketAddr>,
        request: &Request,
        path: &str,
        why: &str,
    ) {
        let Some(ip) = peer.map(|p| p.ip()) else { return };
        let mut refused = lock(&self.refused);
        if refused.len() < REFUSED_PEERS_LOGGED && refused.insert(ip) {
            eprintln!(
                "dashboard: refused {} {path} from {ip} with 403 ({why}); later refusals of {ip} are not logged",
                request.method
            );
        }
    }

    fn api(
        &self,
        request: &Request,
        path: &str,
        query: &str,
        port: u16,
        engine: EngineNow,
    ) -> Response {
        let method = match request.method.as_str() {
            "HEAD" => "GET",
            other => other,
        };
        let rest = path.strip_prefix("/v1/lily").unwrap_or("");
        let segments: Vec<&str> = rest.split('/').skip(1).collect();
        if matches!(method, "PATCH" | "DELETE")
            && let Err(why) = write_guard(
                request.header("Content-Type"),
                request.header("Origin"),
                port,
            )
        {
            return Response::error(403, why, None);
        }
        match (method, segments.as_slice()) {
            ("GET", ["status"]) => Response::json(200, &self.status(engine)),
            ("GET", ["sessions"]) => Response::json(200, &self.sessions()),
            ("GET", ["requests"]) => match request_limit(query) {
                Ok(limit) => Response::json(200, &self.requests(limit)),
                Err(why) => Response::error(400, why, None),
            },
            ("GET", ["settings"]) => Response::json(
                200,
                &SettingsResponse { settings: self.config.settings.clone() },
            ),
            ("GET", ["settings", "log"]) => {
                Response::json(200, &SettingsLogResponse { entries: Vec::new() })
            }
            ("GET", ["history"]) => {
                Response::error(503, "history needs the recorder", None)
            }
            ("GET", ["sessions", lineage, "history"]) => match lineage.parse::<u64>() {
                Ok(_) => Response::error(503, "history needs the recorder", None),
                Err(_) => Response::error(404, "unknown lineage", None),
            },
            ("PATCH", ["settings"]) => {
                Response::error(409, "runtime settings are not editable yet", None)
            }
            ("DELETE", ["settings", name]) => {
                if self.config.settings.iter().any(|s| s.name == *name) {
                    Response::error(
                        409,
                        "runtime settings are not editable yet",
                        Some(name),
                    )
                } else {
                    Response::error(404, "unknown setting", Some(name))
                }
            }
            (
                _,
                ["status"]
                | ["sessions"]
                | ["requests"]
                | ["settings"]
                | ["settings", _]
                | ["history"]
                | ["sessions", _, "history"],
            ) => Response::error(405, "method not allowed", None),
            _ => Response::error(404, "not found", None),
        }
    }

    /// `GET /v1/lily/status`.
    pub fn status(&self, engine: EngineNow) -> Status {
        let now = Instant::now();
        let now_ms = super::now_ms();
        let store = self.board.store();
        let (loaded, max_batch, weights_bytes, loads, open, active, last_finished) = {
            let a = lock(&self.board.activity);
            let facts = a.loaded.as_ref().or(a.last_loaded.as_ref());
            (
                a.loaded.clone(),
                facts.map_or(self.config.max_batch, |l| l.max_batch),
                facts.map_or(0, |l| l.weights_bytes),
                a.loads,
                a.open,
                a.active.clone(),
                a.last_finished,
            )
        };
        let busy = open > 0 || engine.waiting > 0;
        let state = match engine.state {
            super::State::Loading
            | super::State::Reloading
            | super::State::Recovering => LifecycleState::Loading,
            super::State::Idle => LifecycleState::Unloaded,
            super::State::Stopping => LifecycleState::Stopping,
            super::State::Ready if busy => LifecycleState::Busy,
            super::State::Ready => LifecycleState::Ready,
        };
        let idle_for = if open > 0 {
            Duration::ZERO
        } else {
            now.saturating_duration_since(last_finished.unwrap_or(self.started))
        };
        let pinned = self.board.pinned.load(Ordering::Acquire);
        let pin_hold_s = (pinned && open == 0 && self.config.pin_hold_secs > 0)
            .then(|| self.config.pin_hold_secs.saturating_sub(idle_for.as_secs()));

        // The live figures: the token counters over the last few seconds,
        // the finished rows over the last minute.
        let cpu_now = crate::stats::cpu_ticks();
        let (decoded, prefilled) = self.board.token_counts();
        let sample = Sample { at: now, decoded, prefilled, cpu: cpu_now };
        let base = lock(&self.meter).sample(sample);
        let (decode_tps, prefill_tps) = match base {
            Some(base) if open > 0 => {
                let secs = now.saturating_duration_since(base.at).as_secs_f64();
                (
                    rate(base.decoded, sample.decoded, secs),
                    rate(base.prefilled, sample.prefilled, secs),
                )
            }
            _ => (0.0, 0.0),
        };
        let cpu_load = base
            .and_then(|b| b.cpu.zip(cpu_now))
            .and_then(|(before, after)| after.load_since(before))
            .map(|load| round(load, 1e3));
        let (requests_per_min, acceptance) = {
            let rows = lock(&self.board.rows);
            let recent = rows
                .iter()
                .filter(|r| r.t + LIVE_ROWS_MS >= now_ms)
                .collect::<Vec<_>>();
            let (drafted, accepted) = recent.iter().fold((0u64, 0u64), |(d, a), r| {
                match (r.drafted_tokens, r.accepted_tokens) {
                    (Some(rd), Some(ra)) => (d + rd as u64, a + ra.min(rd) as u64),
                    _ => (d, a),
                }
            });
            (recent.len(), ratio(accepted, drafted).filter(|_| open > 0))
        };

        let memory = crate::stats::memory_use();
        let machine = MachineStatus {
            sampler: SamplerState::Off,
            cpu_load,
            gpu_active: None,
            gpu_freq_mhz: None,
            memory_pressure: crate::stats::pressure_level()
                .and_then(MemoryPressure::of_level),
            ram_total_bytes: memory.map(|m| m.total),
            ram_used_bytes: memory.map(|m| m.used),
            lily_resident_bytes: crate::stats::task_memory().map(|t| t.phys_footprint),
            swap_used_bytes: crate::stats::swap_used(),
            power: None,
            temp_c: None,
        };

        // Only a loaded model has a table to show.
        let ngram = loaded.and_then(|l| l.ngram).map(|facts| NgramTier {
            bytes: facts.bytes,
            resident_share: match &facts.paged {
                Some(table) => self.ngram_residency(loads, now, facts.bytes, table),
                None => 1.0,
            },
        });
        // A request's session that the published cache still lists (the
        // publish that follows an admission or precedes a close has not
        // happened yet) is counted once.
        let mut gpu = store.gpu.clone();
        for a in &active {
            let listed = store
                .entries
                .iter()
                .any(|e| !e.on_disk && e.lineage.is_some_and(|l| l.id == a.lineage.id));
            if !listed {
                gpu.used_bytes += a.bytes;
                gpu.entries += 1;
            }
        }
        Status {
            now: now_ms,
            state,
            model: self.config.model.to_owned(),
            uptime_s: self.started.elapsed().as_secs(),
            pin_hold_s,
            idle_unload_s: (self.config.idle_unload_secs > 0)
                .then_some(self.config.idle_unload_secs),
            idle_for_s: idle_for.as_secs(),
            queue: QueueStatus {
                waiting: engine.waiting,
                running: open,
                max_batch,
                capacity: self.config.queue.max(1),
            },
            live: LiveStatus {
                decode_tps,
                prefill_tps,
                acceptance,
                requests_per_min,
                active_sessions: active.len(),
            },
            machine,
            tiers: Tiers {
                gpu,
                disk: store.disk.clone(),
                experts: store.experts.clone(),
                ngram,
                weights: WeightsTier { bytes: weights_bytes, pinned },
            },
            recorder: RecorderStatus {
                ok: false,
                dropped_rows: 0,
                db_bytes: 0,
                oldest: None,
            },
        }
    }

    /// The share of the paged table in the page cache, measured at most
    /// every [`NGRAM_RESIDENCY_TTL`] within one load (`load`); 0 when the
    /// table's mappings are already gone.
    fn ngram_residency(
        &self,
        load: u64,
        now: Instant,
        bytes: u64,
        table: &Weak<PagedTable>,
    ) -> f64 {
        if let Some((of, at, share)) = lock(&self.meter).ngram
            && of == load
            && now.saturating_duration_since(at) < NGRAM_RESIDENCY_TTL
        {
            return share;
        }
        let share = table
            .upgrade()
            .and_then(|t| t.resident_bytes().ok())
            .map_or(0.0, |resident| round(resident as f64 / bytes.max(1) as f64, 1e3));
        lock(&self.meter).ngram = Some((load, now, share));
        share
    }

    /// `GET /v1/lily/sessions`: the requests' sessions first (active), then
    /// the cache's entries, most recently used first.
    pub fn sessions(&self) -> SessionsResponse {
        let store = self.board.store();
        let active = lock(&self.board.activity).active.clone();
        let rows: Vec<RequestRow> = lock(&self.board.rows).iter().cloned().collect();
        let now_ms = super::now_ms();
        let of = |lineage: Option<u64>| match lineage {
            Some(id) => figures(rows.iter().filter(|r| r.lineage == Some(id))),
            None => Figures::default(),
        };
        let session = |lineage: Option<Lineage>,
                       tier,
                       durable,
                       active,
                       tokens,
                       bytes,
                       last_used_t: u64| {
            let f = of(lineage.map(|l| l.id));
            Session {
                lineage: lineage.map(|l| l.id),
                tier,
                durable,
                active,
                tokens,
                bytes,
                last_used_t,
                requests: f.requests,
                decode_tps_avg: f.decode_tps_avg,
                decode_tps_max: f.decode_tps_max,
                prefill_tps_avg: f.prefill_tps_avg,
                cache_hit: f.cache_hit,
                acceptance: f.acceptance,
                created_t: lineage.map(|l| l.created_ms),
            }
        };
        let mut sessions: Vec<Session> = active
            .iter()
            .map(|a| {
                session(
                    Some(a.lineage),
                    SessionTier::Gpu,
                    false,
                    true,
                    a.tokens,
                    a.bytes,
                    now_ms,
                )
            })
            .collect();
        let mut entries: Vec<&EntryView> = store
            .entries
            .iter()
            // The session a request holds is listed as that request's, not
            // again as an entry (a stale publish, or its copy on disk).
            .filter(|e| {
                !e.lineage.is_some_and(|l| active.iter().any(|a| a.lineage.id == l.id))
            })
            .collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.last_used_ms));
        sessions.extend(entries.into_iter().map(|e| {
            session(
                e.lineage,
                if e.on_disk { SessionTier::Disk } else { SessionTier::Gpu },
                e.durable,
                false,
                e.tokens,
                e.bytes,
                e.last_used_ms,
            )
        }));
        SessionsResponse { sessions }
    }

    /// `GET /v1/lily/requests?limit=`: the newest `limit` rows, newest first.
    pub fn requests(&self, limit: usize) -> RequestsResponse {
        let rows = lock(&self.board.rows);
        RequestsResponse { requests: rows.iter().rev().take(limit).cloned().collect() }
    }
}

/// The `limit` of `GET /v1/lily/requests`: [`DEFAULT_REQUEST_LIMIT`] when
/// absent, capped at [`MAX_REQUEST_LIMIT`].
fn request_limit(query: &str) -> Result<usize, &'static str> {
    let Some(text) = query.split('&').find_map(|pair| pair.strip_prefix("limit="))
    else {
        return Ok(DEFAULT_REQUEST_LIMIT);
    };
    text.parse::<usize>()
        .map(|n| n.min(MAX_REQUEST_LIMIT))
        .map_err(|_| "limit must be a non-negative integer")
}

#[cfg(test)]
#[path = "../../tests/unit/serve/dashboard.rs"]
mod tests;
