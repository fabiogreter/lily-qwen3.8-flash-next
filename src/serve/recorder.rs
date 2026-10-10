//! The dashboard's recorder (docs/dashboard.md § 4): the SQLite database
//! behind the 7-day history, opened and written by one thread of its own.
//!
//! **What feeds it.** Nothing new on the engine's side. The engine already
//! hands each finished request's row to the [`Board`]; the board also offers
//! it to the recorder's bounded channel with `try_send`, so a recorder that
//! falls behind costs the engine nothing: the row is dropped and counted
//! ([`RecorderStatus::dropped_rows`], § 2.3). Every [`SAMPLE_EVERY`] the
//! recorder thread samples the machine itself (CPU load, memory pressure,
//! swap, RAM in use, lily's resident memory, the same calls the status route
//! makes) and, once a minute, takes the session cache's tiers from the view
//! the engine last published. It never sends the engine anything, never
//! takes a lock the engine holds across work (a board lock only for a clone,
//! never across a SQLite call) and never runs on the engine's thread.
//!
//! **Opening.** The thread opens the database, never the main thread, so the
//! server's start never waits on SQLite. An open that fails is retried after
//! 1 s, 2 s, 5 s and then every 30 s; meanwhile the status says `opening`
//! with the error, and rows wait in the channel (or a buffer of the same
//! size), counted when they are dropped. Once open, the board's lineage
//! counter is raised above every id the database holds. Sessions created
//! before that took ids from the counter's clock seed (start time in ms ×
//! 1000), which is already above every id an earlier process handed out
//! unless the clock went back, so the late raise only matters then.
//!
//! **One writer, many readers.** The recorder thread owns the only
//! read-write connection (WAL, `synchronous=NORMAL`, the WAL truncated to
//! [`JOURNAL_SIZE_LIMIT`] after checkpoints). The HTTP threads read through
//! read-only connections of their own, opened per answer ([`Recorder::read`])
//! with a short busy timeout, so a lock held elsewhere answers 503 quickly
//! rather than holding an HTTP thread.
//!
//! **Busy time** (§ 5). A row says when its request finished (`t`, taken
//! after the session was returned to the cache) and how long its prefill
//! and its decode ran. The recorder places the decode as the `decode_ms`
//! that end at `t` and the prefill as the `prefill_ms` that end where the
//! decode starts; the release between the decode's end and `t` (a few
//! milliseconds of eviction at most) is not told apart. A request is at
//! most [`MAX_SPAN_MS`] long and never starts before the process did: a
//! longer (bogus) duration is clamped, and stored clamped. A minute's
//! **decode busy** time is the length of the union of the decode intervals
//! that overlap it, clipped to the minute, so two requests decoding together
//! for 10 s count 10 s, not 20 s, and their summed tokens over it are the
//! batch's throughput. **Prefill busy** is the same over the prefill
//! intervals of the requests that computed tokens (a full cache hit computes
//! nothing, so its restore does not dilute the prefill rate), and **busy**
//! the union of whole requests (prefill start to finish). A request's tokens
//! are spread evenly over its interval, so a minute gets the share it
//! overlapped. The per-session rows are the same computation restricted to
//! one lineage. Each row recomputes every minute it overlaps from the stored
//! rows, so a request finishing later still counts in the minutes it ran
//! through, and recomputing is idempotent.
//!
//! **Requests, cache hits, acceptance and queue times** belong to the
//! minute a request finished in; a failed request has no queue time.
//!
//! **A broken probe never breaks serving** (§ 2.4). A file that is not a
//! database or fails `quick_check` is renamed aside (`<path>.corrupt-<unix
//! seconds>`, its `-wal` and `-shm` with it), logged, and a fresh database
//! takes its place. A database with a newer schema is left untouched and the
//! recorder stays failed (history off). A write that fails is logged once
//! per kind and marks the recorder not ok until that kind succeeds again; a
//! batch of request rows that fails is retried once with the next, then its
//! rows are dropped and counted; a single row SQLite refuses is skipped and
//! counted. The thread never panics on a SQLite error, and if it ends anyway
//! the rows sent to it are counted as dropped.
//!
//! **Settings** (§ 7.3). The thread is the one writer of the runtime
//! settings after startup. Once the database opens, before the recorder
//! says `open`, it applies the stored overrides over the command line
//! (`RuntimeSettings::apply_stored`), logs one line naming them and records a
//! `startup` entry for each; a row it cannot apply is logged and kept. A
//! change from the dashboard ([`Recorder::change_settings`]) comes through
//! the same channel as the rows: the thread plans it against the state now,
//! writes the `settings` rows and the `settings_log` entries in one
//! transaction and only then puts it into effect, so the server and the
//! database never disagree. The HTTP thread waits for that at most
//! [`SETTINGS_WAIT`]; a change it stopped waiting for before the thread took
//! it up is not made.
//!
//! **Shutdown.** [`Recorder::finish`] asks the thread to stop, waits for it
//! at most a bounded time, and the thread's last act is a WAL checkpoint
//! that truncates the WAL.
//!
//! **Numbers only** (§ 2.1): every table holds counts, durations, rates,
//! lineage ids and times; the settings tables hold setting names and values.
//! The directory lily creates for the database is `0700`, the database and
//! its WAL files `0600`.

use std::collections::{BTreeSet, HashSet};
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use rusqlite::{Connection, ErrorCode, OpenFlags, Transaction, params};

use super::dashboard::contract::{
    CancelledBy, RecorderState, RecorderStatus, RequestRow, SettingValue,
    SettingsLogEntry, SettingsLogSource,
};
use super::dashboard::{Board, StoreView};
use super::runtime::{Change, Entry, Plan, RuntimeSettings, same_value};
use crate::stats::CpuTicks;

/// The schema this lily writes (`PRAGMA user_version`). Migrations only go
/// forward; a database with a larger version is left alone.
pub const SCHEMA_VERSION: i64 = 1;
/// How long everything is kept.
pub const RETENTION_MS: u64 = 7 * 86_400_000;
/// Kept on top of [`RETENTION_MS`], pruned at the hour: the 7d range's
/// oldest bucket (an hour) stays whole.
const PRUNE_MARGIN_MS: u64 = 3_600_000;
pub const MINUTE_MS: u64 = 60_000;
/// The longest a request can have run, as the rollups count it.
pub const MAX_SPAN_MS: u64 = 86_400_000;
/// Rows the channel from the board holds: minutes of the busiest agent
/// traffic, so a full channel means the recorder is stuck, not slow.
pub const ROW_CHANNEL: usize = 1024;
/// How often the recorder samples the machine.
pub const SAMPLE_EVERY: Duration = Duration::from_secs(10);
/// How often rows older than the retention are deleted.
const PRUNE_EVERY: Duration = Duration::from_secs(3600);
/// The most rows written in one transaction.
const BATCH_ROWS: usize = 256;
/// How long the writer waits for a lock (another process reading the file
/// with the `sqlite3` shell, say).
const WRITE_BUSY_TIMEOUT: Duration = Duration::from_secs(2);
/// How long an HTTP thread's read waits for a lock before answering 503.
pub const READ_BUSY_TIMEOUT: Duration = Duration::from_millis(250);
/// The waits between failed opens; the last repeats.
const OPEN_BACKOFF: [Duration; 4] = [
    Duration::from_secs(1),
    Duration::from_secs(2),
    Duration::from_secs(5),
    Duration::from_secs(30),
];
/// The WAL is truncated to this after a checkpoint.
pub const JOURNAL_SIZE_LIMIT: i64 = 4 << 20;
/// The kinds of write a settings change makes (see [`Writer::outcome`]).
const SETTINGS_WRITES: [&str; 2] = ["writing settings", "writing the settings log"];
/// How long a settings change waits for the recorder to make it durable:
/// the writer's own busy timeout and a margin.
pub const SETTINGS_WAIT: Duration = Duration::from_secs(3);
/// The most change log entries `GET /v1/lily/settings/log` returns.
pub const SETTINGS_LOG_LIMIT: usize = 500;

/// The forward-only migrations: `MIGRATIONS[v]` takes version `v` to `v + 1`.
const MIGRATIONS: [&str; SCHEMA_VERSION as usize] = [SCHEMA_V1];

/// Times are unix milliseconds (`t`, `*_t`) or unix minutes (`minute`, unix
/// milliseconds / 60 000); tokens in the rollups are fractional, a request's
/// tokens being shared out over the minutes it ran through. `requests` holds
/// the durations as the rollups count them (clamped, § the module docs) and
/// the accepted draft tokens clamped to the drafted; a failed request's
/// `queue_ms` is NULL.
const SCHEMA_V1: &str = "
CREATE TABLE requests (
    t INTEGER NOT NULL,
    start_t INTEGER NOT NULL,
    lineage INTEGER,
    prompt_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    prefill_tokens INTEGER NOT NULL,
    prefill_ms REAL NOT NULL,
    prefill_tps REAL,
    generated_tokens INTEGER NOT NULL,
    decode_ms REAL NOT NULL,
    decode_tps REAL,
    drafted_tokens INTEGER,
    accepted_tokens INTEGER,
    queue_ms REAL,
    batch INTEGER NOT NULL,
    cancelled INTEGER NOT NULL,
    thinking_forced INTEGER NOT NULL,
    pinned INTEGER NOT NULL,
    failed INTEGER NOT NULL
);
CREATE INDEX requests_t ON requests (t);
CREATE INDEX requests_lineage ON requests (lineage, t);
CREATE TABLE minute (
    minute INTEGER PRIMARY KEY,
    requests INTEGER NOT NULL,
    decode_tokens REAL NOT NULL,
    prefill_tokens REAL NOT NULL,
    decode_busy_ms REAL NOT NULL,
    prefill_busy_ms REAL NOT NULL,
    busy_ms REAL NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    drafted_tokens INTEGER NOT NULL,
    accepted_tokens INTEGER NOT NULL,
    sessions_max INTEGER NOT NULL
);
CREATE TABLE minute_session (
    lineage INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    requests INTEGER NOT NULL,
    decode_tokens REAL NOT NULL,
    prefill_tokens REAL NOT NULL,
    decode_busy_ms REAL NOT NULL,
    prefill_busy_ms REAL NOT NULL,
    busy_ms REAL NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    drafted_tokens INTEGER NOT NULL,
    accepted_tokens INTEGER NOT NULL,
    PRIMARY KEY (lineage, minute)
) WITHOUT ROWID;
CREATE INDEX minute_session_minute ON minute_session (minute);
CREATE TABLE machine (
    minute INTEGER PRIMARY KEY,
    samples INTEGER NOT NULL,
    cpu_load_min REAL,
    cpu_load_mean REAL,
    cpu_load_max REAL,
    memory_pressure INTEGER,
    swap_bytes_mean REAL,
    swap_bytes_max INTEGER,
    ram_used_bytes_mean REAL,
    lily_resident_bytes_max INTEGER,
    gpu_active_mean REAL,
    gpu_freq_mhz_min REAL,
    gpu_freq_mhz_mean REAL,
    gpu_freq_mhz_max REAL,
    power_w_min REAL,
    power_w_mean REAL,
    power_w_max REAL,
    gpu_power_w_mean REAL
);
CREATE TABLE tiers (
    minute INTEGER PRIMARY KEY,
    gpu_bytes INTEGER NOT NULL,
    gpu_budget_bytes INTEGER,
    gpu_entries INTEGER NOT NULL,
    active_sessions INTEGER NOT NULL,
    disk_bytes INTEGER,
    disk_budget_bytes INTEGER,
    disk_entries INTEGER,
    disk_durable_entries INTEGER,
    disk_durable_bytes INTEGER,
    expert_slots INTEGER,
    expert_hit_rate REAL,
    weights_pinned INTEGER NOT NULL
);
CREATE TABLE settings (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    changed_t INTEGER NOT NULL
);
CREATE TABLE settings_log (
    t INTEGER NOT NULL,
    name TEXT NOT NULL,
    old TEXT,
    new TEXT,
    source TEXT NOT NULL
);
CREATE INDEX settings_log_t ON settings_log (t);
";

/// The tables pruning trims and the column that dates their rows (`true`:
/// in minutes).
const PRUNED: [(&str, &str, bool); 6] = [
    ("requests", "t", false),
    ("minute", "minute", true),
    ("minute_session", "minute", true),
    ("machine", "minute", true),
    ("tiers", "minute", true),
    ("settings_log", "t", false),
];

// --- health --------------------------------------------------------------------

/// What the recorder thread and the board's row sink report to the status
/// route, read without touching the database.
#[derive(Debug)]
pub struct Health {
    /// Where the recorder is, and its error while opening or failed.
    phase: Mutex<(RecorderState, Option<String>)>,
    /// The database was opened once: reads can go to it.
    opened: AtomicBool,
    /// No kind of write is failing.
    ok: AtomicBool,
    dropped: AtomicU64,
    db_bytes: AtomicU64,
    /// Unix ms of the oldest stored sample; 0 when nothing is stored.
    oldest: AtomicU64,
    /// The kinds of error logged so far, each once.
    logged: Mutex<HashSet<&'static str>>,
    /// The `settings` table could not be read at the open (twice): the
    /// command line stays in effect and settings changes are refused.
    settings_unreadable: AtomicBool,
}

impl Default for Health {
    fn default() -> Self {
        Self {
            phase: Mutex::new((RecorderState::Opening, None)),
            opened: AtomicBool::new(false),
            ok: AtomicBool::new(false),
            dropped: AtomicU64::new(0),
            db_bytes: AtomicU64::new(0),
            oldest: AtomicU64::new(0),
            logged: Mutex::default(),
            settings_unreadable: AtomicBool::new(false),
        }
    }
}

impl Health {
    /// Logs `error` the first time an error of `kind` happens.
    fn log_once(&self, kind: &'static str, error: &dyn Display) {
        let first = self.logged.lock().unwrap_or_else(|p| p.into_inner()).insert(kind);
        if first {
            eprintln!(
                "recorder: {kind} failed: {error}; the history has a gap (later {kind} errors are not logged)"
            );
        }
    }

    fn set_phase(&self, state: RecorderState, error: Option<String>) {
        *self.phase.lock().unwrap_or_else(|p| p.into_inner()) = (state, error);
    }

    fn phase(&self) -> (RecorderState, Option<String>) {
        self.phase.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    fn drop_rows(&self, n: usize) {
        self.dropped.fetch_add(n as u64, Ordering::Relaxed);
    }
}

/// What the board, the settings routes and [`Recorder::finish`] send the
/// recorder thread.
#[derive(Debug)]
pub enum Msg {
    Row(Box<RequestRow>),
    Settings(Box<SettingsWrite>),
    /// Write what is pending, checkpoint and end.
    Stop,
}

/// A settings change for the recorder to write and put into effect.
#[derive(Debug)]
pub struct SettingsWrite {
    change: Change,
    /// The entries of what changed, or why nothing did.
    reply: SyncSender<Result<Vec<Entry>, WriteError>>,
    /// Who has it: the recorder (taken) or the HTTP thread (abandoned).
    claim: Arc<Claim>,
}

/// Why a settings change was not made, or may have been, as the route
/// answers it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteError {
    /// 503 nothing changed, 504 the outcome is unknown, 500 the change
    /// could not be worked out.
    pub status: u16,
    pub message: String,
}

impl WriteError {
    /// Nothing changed (503).
    pub fn not_applied(message: impl Into<String>) -> Self {
        Self { status: 503, message: message.into() }
    }
}

/// The message of a change whose outcome the HTTP thread cannot know.
pub const OUTCOME_UNKNOWN: &str =
    "the change may still apply; reload to see the current value";

/// Who decides a settings change's fate once the HTTP thread stops
/// waiting: the recorder takes it before writing, the HTTP thread abandons
/// it on its timeout; whichever comes first wins, so a timeout tells a
/// change that will never be made (abandoned) from one being written.
#[derive(Debug, Default)]
pub struct Claim(AtomicU8);

impl Claim {
    const PENDING: u8 = 0;
    const TAKEN: u8 = 1;
    const ABANDONED: u8 = 2;

    fn swap_from_pending(&self, to: u8) -> bool {
        self.0
            .compare_exchange(Self::PENDING, to, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// The recorder's: `true` when it may make the change.
    pub fn take(&self) -> bool {
        self.swap_from_pending(Self::TAKEN)
    }

    /// The HTTP thread's: `true` when the change will never be made.
    pub fn abandon(&self) -> bool {
        self.swap_from_pending(Self::ABANDONED)
    }
}

/// The board's end of the channel to the recorder.
#[derive(Debug)]
pub struct RowSink {
    rows: SyncSender<Msg>,
    health: Arc<Health>,
}

impl RowSink {
    /// Offers `row` to the recorder; a full channel, or a recorder that is
    /// gone, drops it and counts the drop. Never blocks.
    pub fn send(&self, row: RequestRow) {
        if let Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) =
            self.rows.try_send(Msg::Row(Box::new(row)))
        {
            self.health.drop_rows(1);
        }
    }
}

/// A sink over a channel of `capacity`, for tests of the drop counter.
#[cfg(test)]
pub fn sink(capacity: usize) -> (RowSink, Receiver<Msg>, Arc<Health>) {
    let (rows, rx) = mpsc::sync_channel(capacity);
    let health = Arc::new(Health::default());
    (RowSink { rows, health: health.clone() }, rx, health)
}

// --- the database --------------------------------------------------------------

/// Why a database could not be prepared.
#[derive(Debug)]
enum OpenError {
    /// Not a database, or a corrupt one: set it aside.
    Corrupt(String),
    /// A newer lily's schema: leave it alone.
    Newer(i64),
    /// Anything else (permissions, a full disk, a lock): try again later.
    Failed(rusqlite::Error),
}

impl From<rusqlite::Error> for OpenError {
    fn from(error: rusqlite::Error) -> Self {
        match error.sqlite_error_code() {
            Some(ErrorCode::NotADatabase | ErrorCode::DatabaseCorrupt) => {
                Self::Corrupt(error.to_string())
            }
            _ => Self::Failed(error),
        }
    }
}

/// Sets up `conn` for the recorder: the schema version checked first (a
/// newer one is left alone), incremental auto-vacuum on a database that has
/// no tables yet, WAL with `synchronous=NORMAL` and a WAL size limit, a quick
/// integrity check, then the migrations up to [`SCHEMA_VERSION`].
fn prepare(conn: &Connection) -> Result<(), OpenError> {
    conn.busy_timeout(WRITE_BUSY_TIMEOUT)?;
    let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if version > SCHEMA_VERSION {
        return Err(OpenError::Newer(version));
    }
    if version == 0 {
        // Only takes effect before the first table exists.
        conn.execute_batch("PRAGMA auto_vacuum = INCREMENTAL")?;
    }
    // Answers the mode in effect (`memory` for an in-memory database).
    let _: String = conn.query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0))?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    let _: i64 = conn.query_row(
        &format!("PRAGMA journal_size_limit = {JOURNAL_SIZE_LIMIT}"),
        [],
        |r| r.get(0),
    )?;
    let check: String = conn.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
    if check != "ok" {
        return Err(OpenError::Corrupt(format!("quick_check says {check:?}")));
    }
    for (from, migration) in MIGRATIONS.iter().enumerate().skip(version.max(0) as usize)
    {
        let tx = conn.unchecked_transaction()?;
        tx.execute_batch(migration)?;
        tx.pragma_update(None, "user_version", from as i64 + 1)?;
        tx.commit()?;
    }
    Ok(())
}

/// `path` with `suffix` appended to its file name.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Creates `path`'s missing directories (`0700`) and the file itself
/// (`0600`) when it does not exist, so SQLite never creates it with the
/// process's umask.
fn create_private(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _};
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)?;
    }
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
    {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(e),
    }
}

/// Makes the database and its WAL files `0600` where they exist (SQLite
/// creates the WAL files with the database's mode; this is a second line).
fn restrict(path: &Path) {
    use std::os::unix::fs::PermissionsExt as _;
    for p in [path.to_path_buf(), with_suffix(path, "-wal"), with_suffix(path, "-shm")]
    {
        if p.exists() {
            let _ =
                std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600));
        }
    }
}

fn try_open(path: &Path) -> Result<Connection, OpenError> {
    create_private(path).map_err(|e| {
        OpenError::Failed(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CANTOPEN),
            Some(format!("creating {}: {e}", path.display())),
        ))
    })?;
    let conn = Connection::open(path)?;
    prepare(&conn)?;
    restrict(path);
    Ok(conn)
}

/// What [`open`] did with a corrupt file: where it went and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetAside {
    pub to: PathBuf,
    pub why: String,
}

/// Why [`open`] gave up.
#[derive(Debug)]
pub enum OpenFailure {
    /// The database has schema `found`, newer than [`SCHEMA_VERSION`]: it
    /// is left untouched and the history stays off.
    Newer { found: i64 },
    /// Anything else; worth another try later.
    Other(anyhow::Error),
}

impl Display for OpenFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Newer { found } => {
                write!(f, "schema {found}, this lily knows {SCHEMA_VERSION}")
            }
            Self::Other(error) => write!(f, "{error:#}"),
        }
    }
}

/// Opens the database at `path` for writing, creating it and its directory
/// as needed. A corrupt file (not a database, failing `quick_check`) is
/// renamed to `<path>.corrupt-<now_s>` with its `-wal` and `-shm`, and a
/// fresh database is created in its place; a newer schema is refused.
pub fn open(
    path: &Path,
    now_s: u64,
) -> Result<(Connection, Option<SetAside>), OpenFailure> {
    use anyhow::Context as _;
    let other = |e: rusqlite::Error, what: &str| {
        OpenFailure::Other(
            anyhow::Error::new(e).context(format!("{what} {}", path.display())),
        )
    };
    let why = match try_open(path) {
        Ok(conn) => return Ok((conn, None)),
        Err(OpenError::Newer(found)) => return Err(OpenFailure::Newer { found }),
        Err(OpenError::Failed(error)) => return Err(other(error, "opening")),
        Err(OpenError::Corrupt(why)) => why,
    };
    let to = with_suffix(path, &format!(".corrupt-{now_s}"));
    let set_aside = || -> anyhow::Result<()> {
        std::fs::rename(path, &to)
            .with_context(|| format!("setting {} aside", path.display()))?;
        for side in ["-wal", "-shm"] {
            let from = with_suffix(path, side);
            if from.exists() {
                std::fs::rename(&from, with_suffix(&to, side))
                    .with_context(|| format!("setting {} aside", from.display()))?;
            }
        }
        Ok(())
    };
    set_aside().map_err(OpenFailure::Other)?;
    match try_open(path) {
        Ok(conn) => Ok((conn, Some(SetAside { to, why }))),
        Err(OpenError::Newer(found)) => Err(OpenFailure::Newer { found }),
        Err(OpenError::Failed(error)) => Err(other(error, "creating")),
        Err(OpenError::Corrupt(still)) => Err(OpenFailure::Other(anyhow::anyhow!(
            "a fresh {} is unusable too: {still}",
            path.display()
        ))),
    }
}

/// An in-memory database with the schema, for tests and the fixtures.
#[cfg(test)]
pub fn in_memory() -> rusqlite::Result<Connection> {
    let conn = Connection::open_in_memory()?;
    match prepare(&conn) {
        Ok(()) => Ok(conn),
        Err(OpenError::Failed(error)) => Err(error),
        Err(other) => panic!("a fresh in-memory database is unusable: {other:?}"),
    }
}

/// A read-only connection to the recorder's database at `path`, waiting at
/// most [`READ_BUSY_TIMEOUT`] for a lock.
pub fn open_reader(path: &Path) -> rusqlite::Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.busy_timeout(READ_BUSY_TIMEOUT)?;
    Ok(conn)
}

/// The largest lineage id the database holds.
pub fn max_lineage(conn: &Connection) -> rusqlite::Result<Option<u64>> {
    conn.query_row("SELECT max(lineage) FROM requests", [], |r| r.get(0))
}

/// Unix ms of the oldest stored sample, `None` when nothing is stored.
pub fn oldest(conn: &Connection) -> rusqlite::Result<Option<u64>> {
    conn.query_row(
        "SELECT min(x) FROM (
             SELECT min(t) AS x FROM requests
             UNION ALL SELECT min(minute) * 60000 FROM minute
             UNION ALL SELECT min(minute) * 60000 FROM machine
         )",
        [],
        |r| r.get(0),
    )
}

/// The overrides stored in the database: name and JSON text.
pub fn stored_settings(conn: &Connection) -> rusqlite::Result<Vec<(String, String)>> {
    let mut select = conn.prepare("SELECT name, value FROM settings ORDER BY name")?;
    let rows = select.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
    rows.collect()
}

/// A stored value (JSON text, NULL for none) as the contract's value; text
/// that is not a JSON number, string or boolean reads as `None`.
fn stored_value(text: Option<String>) -> Option<SettingValue> {
    text.and_then(|t| serde_json::from_str(&t).ok())
        .and_then(|v: serde_json::Value| SettingValue::from_json(&v))
}

/// The newest `limit` change log entries, newest first. An entry with a
/// source this lily does not know is left out.
pub fn settings_log(
    conn: &Connection,
    limit: usize,
) -> rusqlite::Result<Vec<SettingsLogEntry>> {
    let mut select = conn.prepare(
        "SELECT t, name, old, new, source FROM settings_log
         ORDER BY t DESC, rowid DESC LIMIT ?1",
    )?;
    let rows = select.query_map([limit as i64], |r| {
        let (t, name, old, new, source): (u64, String, _, _, String) =
            (r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?);
        Ok(SettingsLogSource::parse(&source).map(|source| SettingsLogEntry {
            t,
            name,
            old: stored_value(old),
            new: stored_value(new),
            source,
        }))
    })?;
    rows.filter_map(Result::transpose).collect()
}

/// A value as the database stores it: JSON text.
fn json_text(value: &Option<SettingValue>) -> Option<String> {
    value.as_ref().and_then(|v| serde_json::to_string(v).ok())
}

/// Inserts `entries` into the change log at `t` from `source`.
fn log_entries(
    tx: &Transaction<'_>,
    entries: &[Entry],
    source: SettingsLogSource,
    t: u64,
) -> rusqlite::Result<()> {
    let mut insert = tx.prepare_cached(
        "INSERT INTO settings_log (t, name, old, new, source) VALUES (?1, ?2, ?3, ?4, ?5)",
    )?;
    for e in entries {
        insert.execute(params![
            t,
            e.name,
            json_text(&e.old),
            json_text(&e.new),
            source.as_str()
        ])?;
    }
    Ok(())
}

// --- request rows and their minutes ------------------------------------------

/// The numbers of a stored request the rollups need.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    /// Finish, unix ms.
    pub t: u64,
    pub lineage: Option<u64>,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub prefill_tokens: u64,
    pub prefill_ms: f64,
    pub generated_tokens: u64,
    pub decode_ms: f64,
    /// Drafted and accepted (at most the drafted), or neither when the
    /// draft head was off.
    pub speculation: Option<(u64, u64)>,
}

/// A duration as a non-negative finite number of milliseconds.
fn ms(value: f64) -> f64 {
    if value.is_finite() { value.max(0.0) } else { 0.0 }
}

impl Span {
    /// `row` as the rollups count it: at most [`MAX_SPAN_MS`] long and not
    /// starting before `floor_ms` (the process's start), the decode clamped
    /// first.
    pub fn of(row: &RequestRow, floor_ms: u64) -> Self {
        let limit = MAX_SPAN_MS.min(row.t.saturating_sub(floor_ms)) as f64;
        let decode_ms = ms(row.decode_ms).min(limit);
        let prefill_ms = ms(row.prefill_ms).min(limit - decode_ms);
        Self {
            t: row.t,
            lineage: row.lineage,
            prompt_tokens: row.prompt_tokens as u64,
            cached_tokens: row.cached_tokens as u64,
            prefill_tokens: row.prefill_tokens as u64,
            prefill_ms,
            generated_tokens: row.generated_tokens as u64,
            decode_ms,
            speculation: row
                .drafted_tokens
                .zip(row.accepted_tokens)
                .map(|(d, a)| (d as u64, a.min(d) as u64)),
        }
    }

    /// The decode: the `decode_ms` that end at the finish.
    fn decode(&self) -> (f64, f64) {
        let end = self.t as f64;
        (end - self.decode_ms, end)
    }

    /// The prefill: the `prefill_ms` that end where the decode starts.
    fn prefill(&self) -> (f64, f64) {
        let end = self.t as f64 - self.decode_ms;
        (end - self.prefill_ms, end)
    }

    /// The whole request, prefill start to finish.
    fn whole(&self) -> (f64, f64) {
        (self.prefill().0, self.t as f64)
    }

    /// Unix ms the request started, rounded down.
    fn start_t(&self) -> u64 {
        self.whole().0.max(0.0).floor() as u64
    }

    /// The first and last minute the request ran in.
    pub fn minutes(&self) -> std::ops::RangeInclusive<u64> {
        (self.start_t() / MINUTE_MS)..=(self.t / MINUTE_MS)
    }
}

/// The length of the union of `intervals`, each clipped to `[from, to)`.
fn union_ms(mut intervals: Vec<(f64, f64)>, from: f64, to: f64) -> f64 {
    intervals.retain_mut(|(s, e)| {
        *s = s.max(from);
        *e = e.min(to);
        e > s
    });
    intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut total = 0.0;
    let mut current: Option<(f64, f64)> = None;
    for (s, e) in intervals {
        match &mut current {
            Some((_, ce)) if s <= *ce => *ce = ce.max(e),
            _ => {
                if let Some((cs, ce)) = current {
                    total += ce - cs;
                }
                current = Some((s, e));
            }
        }
    }
    total + current.map_or(0.0, |(s, e)| e - s)
}

/// The most of `intervals` that overlap at one time within `[from, to)`.
fn most_at_once(
    intervals: impl Iterator<Item = (f64, f64)>,
    from: f64,
    to: f64,
) -> u64 {
    let mut edges: Vec<(f64, i64)> = Vec::new();
    for (s, e) in intervals {
        let (s, e) = (s.max(from), e.min(to));
        if e > s {
            edges.push((s, 1));
            edges.push((e, -1));
        }
    }
    // An interval ending where another starts does not overlap it.
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let (mut now, mut most) = (0i64, 0i64);
    for (_, step) in edges {
        now += step;
        most = most.max(now);
    }
    most as u64
}

/// `tokens` spread evenly over `(s, e)`: the share inside `[from, to)`. A
/// zero-length interval puts them all at its end.
fn share(tokens: u64, (s, e): (f64, f64), from: f64, to: f64) -> f64 {
    if e > s {
        let overlap = (e.min(to) - s.max(from)).max(0.0);
        tokens as f64 * overlap / (e - s)
    } else if e >= from && e < to {
        tokens as f64
    } else {
        0.0
    }
}

/// One minute's figures, over every request or one lineage's (see the
/// module docs for how busy time and tokens are attributed).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Minute {
    /// Requests that finished in the minute.
    pub requests: u64,
    pub decode_tokens: f64,
    pub prefill_tokens: f64,
    pub decode_busy_ms: f64,
    pub prefill_busy_ms: f64,
    pub busy_ms: f64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    /// The most requests running at once.
    pub sessions_max: u64,
}

impl Minute {
    /// Minute `minute` over `spans` (those that finished in it or ran
    /// through it; others contribute nothing). `None` when none did.
    pub fn of<'a>(
        minute: u64,
        spans: impl Iterator<Item = &'a Span> + Clone,
    ) -> Option<Self> {
        let (from, to) =
            ((minute * MINUTE_MS) as f64, ((minute + 1) * MINUTE_MS) as f64);
        let mut m = Minute {
            decode_busy_ms: union_ms(
                spans.clone().filter(|s| s.decode_ms > 0.0).map(Span::decode).collect(),
                from,
                to,
            ),
            prefill_busy_ms: union_ms(
                spans
                    .clone()
                    .filter(|s| s.prefill_tokens > 0 && s.prefill_ms > 0.0)
                    .map(Span::prefill)
                    .collect(),
                from,
                to,
            ),
            busy_ms: union_ms(spans.clone().map(Span::whole).collect(), from, to),
            sessions_max: most_at_once(spans.clone().map(Span::whole), from, to),
            ..Minute::default()
        };
        let mut any = m.busy_ms > 0.0;
        for span in spans {
            m.decode_tokens += share(span.generated_tokens, span.decode(), from, to);
            if span.prefill_tokens > 0 {
                m.prefill_tokens +=
                    share(span.prefill_tokens, span.prefill(), from, to);
            }
            if span.t / MINUTE_MS == minute {
                any = true;
                m.requests += 1;
                m.prompt_tokens += span.prompt_tokens;
                m.cached_tokens += span.cached_tokens;
                if let Some((drafted, accepted)) = span.speculation {
                    m.drafted_tokens += drafted;
                    m.accepted_tokens += accepted;
                }
            }
        }
        any.then_some(m)
    }
}

const INSERT_REQUEST: &str = "INSERT INTO requests (
    t, start_t, lineage, prompt_tokens, cached_tokens, prefill_tokens, prefill_ms,
    prefill_tps, generated_tokens, decode_ms, decode_tps, drafted_tokens,
    accepted_tokens, queue_ms, batch, cancelled, thinking_forced, pinned, failed
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)";

/// Whether `error` is about the row itself (a value SQLite cannot hold, a
/// constraint), not the database: the row is skipped, the batch goes on.
fn refuses_the_row(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::ToSqlConversionFailure(_)
            | rusqlite::Error::IntegralValueOutOfRange(..)
    ) || error.sqlite_error_code() == Some(ErrorCode::ConstraintViolation)
}

/// Stores one row, as the rollups count it.
fn insert(tx: &Transaction<'_>, row: &RequestRow, span: &Span) -> rusqlite::Result<()> {
    let cancelled = match row.cancelled_by {
        None => 0,
        Some(CancelledBy::Client) => 1,
        Some(CancelledBy::Shutdown) => 2,
    };
    tx.prepare_cached(INSERT_REQUEST)?.execute(params![
        row.t,
        span.start_t(),
        row.lineage,
        span.prompt_tokens,
        span.cached_tokens,
        span.prefill_tokens,
        span.prefill_ms,
        row.prefill_tps,
        span.generated_tokens,
        span.decode_ms,
        row.decode_tps,
        span.speculation.map(|(d, _)| d),
        span.speculation.map(|(_, a)| a),
        // A failed request never waited for anything it got.
        (!row.failed).then_some(ms(row.queue_ms)),
        row.batch as u64,
        cancelled,
        row.thinking_forced,
        row.pinned,
        row.failed,
    ])?;
    Ok(())
}

/// The stored rows that finished in or ran through minute `minute`: those
/// finishing from its start to [`MAX_SPAN_MS`] past its end (no request is
/// longer) that started before its end.
fn spans_in(tx: &Transaction<'_>, minute: u64) -> rusqlite::Result<Vec<Span>> {
    let (from, to) = (minute * MINUTE_MS, (minute + 1) * MINUTE_MS);
    let mut select = tx.prepare_cached(
        "SELECT t, lineage, prompt_tokens, cached_tokens, prefill_tokens, prefill_ms,
                generated_tokens, decode_ms, drafted_tokens, accepted_tokens
         FROM requests WHERE t >= ?1 AND t < ?2 + ?3 AND start_t < ?2",
    )?;
    let rows = select.query_map(params![from, to, MAX_SPAN_MS], |r| {
        let drafted: Option<u64> = r.get(8)?;
        let accepted: Option<u64> = r.get(9)?;
        Ok(Span {
            t: r.get(0)?,
            lineage: r.get(1)?,
            prompt_tokens: r.get(2)?,
            cached_tokens: r.get(3)?,
            prefill_tokens: r.get(4)?,
            prefill_ms: r.get(5)?,
            generated_tokens: r.get(6)?,
            decode_ms: r.get(7)?,
            speculation: drafted.zip(accepted),
        })
    })?;
    rows.collect()
}

/// Recomputes minute `minute` and its per-session rows from the stored rows.
fn roll_up(tx: &Transaction<'_>, minute: u64) -> rusqlite::Result<()> {
    let spans = spans_in(tx, minute)?;
    tx.prepare_cached("DELETE FROM minute WHERE minute = ?1")?.execute([minute])?;
    tx.prepare_cached("DELETE FROM minute_session WHERE minute = ?1")?
        .execute([minute])?;
    if let Some(m) = Minute::of(minute, spans.iter()) {
        tx.prepare_cached(
            "INSERT INTO minute (minute, requests, decode_tokens, prefill_tokens,
                 decode_busy_ms, prefill_busy_ms, busy_ms, prompt_tokens, cached_tokens,
                 drafted_tokens, accepted_tokens, sessions_max)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        )?
        .execute(params![
            minute,
            m.requests,
            m.decode_tokens,
            m.prefill_tokens,
            m.decode_busy_ms,
            m.prefill_busy_ms,
            m.busy_ms,
            m.prompt_tokens,
            m.cached_tokens,
            m.drafted_tokens,
            m.accepted_tokens,
            m.sessions_max,
        ])?;
    }
    let lineages: BTreeSet<u64> = spans.iter().filter_map(|s| s.lineage).collect();
    for lineage in lineages {
        let of_lineage = spans.iter().filter(|s| s.lineage == Some(lineage));
        let Some(m) = Minute::of(minute, of_lineage) else { continue };
        tx.prepare_cached(
            "INSERT INTO minute_session (lineage, minute, requests, decode_tokens,
                 prefill_tokens, decode_busy_ms, prefill_busy_ms, busy_ms, prompt_tokens,
                 cached_tokens, drafted_tokens, accepted_tokens)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        )?
        .execute(params![
            lineage,
            minute,
            m.requests,
            m.decode_tokens,
            m.prefill_tokens,
            m.decode_busy_ms,
            m.prefill_busy_ms,
            m.busy_ms,
            m.prompt_tokens,
            m.cached_tokens,
            m.drafted_tokens,
            m.accepted_tokens,
        ])?;
    }
    Ok(())
}

// --- the machine and the tiers ------------------------------------------------

/// Count, sum, least and most of one series' samples in a minute.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Stat {
    n: u32,
    sum: f64,
    min: f64,
    max: f64,
}

impl Stat {
    pub fn add(&mut self, value: Option<f64>) {
        let Some(v) = value.filter(|v| v.is_finite()) else { return };
        if self.n == 0 {
            (self.min, self.max) = (v, v);
        } else {
            (self.min, self.max) = (self.min.min(v), self.max.max(v));
        }
        self.n += 1;
        self.sum += v;
    }

    pub fn mean(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }

    pub fn min(&self) -> Option<f64> {
        (self.n > 0).then_some(self.min)
    }

    pub fn max(&self) -> Option<f64> {
        (self.n > 0).then_some(self.max)
    }
}

/// One reading of the machine; `None` where its probe has nothing. Power,
/// GPU activity and frequency wait for the sampler (§ 6) and are `None`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MachineSample {
    pub cpu_load: Option<f64>,
    /// `kern.memorystatus_vm_pressure_level`: 1 normal, 2 warning, 4 critical.
    pub pressure_level: Option<u32>,
    pub swap_bytes: Option<u64>,
    pub ram_used_bytes: Option<u64>,
    pub lily_resident_bytes: Option<u64>,
    pub gpu_active: Option<f64>,
    pub gpu_freq_mhz: Option<f64>,
    pub power_w: Option<f64>,
    pub gpu_power_w: Option<f64>,
}

impl MachineSample {
    /// The machine now; the CPU load is measured since `ticks`, which
    /// becomes the reading taken now.
    fn now(ticks: &mut Option<CpuTicks>) -> Self {
        let current = crate::stats::cpu_ticks();
        let cpu_load =
            ticks.zip(current).and_then(|(before, after)| after.load_since(before));
        *ticks = current;
        Self {
            cpu_load,
            pressure_level: crate::stats::pressure_level(),
            swap_bytes: crate::stats::swap_used(),
            ram_used_bytes: crate::stats::memory_use().map(|m| m.used),
            lily_resident_bytes: crate::stats::task_memory().map(|t| t.phys_footprint),
            ..Self::default()
        }
    }
}

/// A minute's machine samples so far.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MachineMinute {
    pub minute: u64,
    pub samples: u32,
    pub cpu_load: Stat,
    /// The worst level seen.
    pub pressure_level: Option<u32>,
    pub swap_bytes: Stat,
    pub ram_used_bytes: Stat,
    pub lily_resident_bytes: Stat,
    pub gpu_active: Stat,
    pub gpu_freq_mhz: Stat,
    pub power_w: Stat,
    pub gpu_power_w: Stat,
}

impl MachineMinute {
    pub fn new(minute: u64) -> Self {
        Self { minute, ..Self::default() }
    }

    pub fn add(&mut self, s: &MachineSample) {
        self.samples += 1;
        self.cpu_load.add(s.cpu_load);
        // The levels grow with the pressure (1, 2, 4).
        self.pressure_level = self.pressure_level.max(s.pressure_level);
        self.swap_bytes.add(s.swap_bytes.map(|b| b as f64));
        self.ram_used_bytes.add(s.ram_used_bytes.map(|b| b as f64));
        self.lily_resident_bytes.add(s.lily_resident_bytes.map(|b| b as f64));
        self.gpu_active.add(s.gpu_active);
        self.gpu_freq_mhz.add(s.gpu_freq_mhz);
        self.power_w.add(s.power_w);
        self.gpu_power_w.add(s.gpu_power_w);
    }
}

/// The session cache's tiers at one moment, as the engine last published
/// them.
#[derive(Debug, Clone, PartialEq)]
pub struct TierSnapshot {
    pub store: StoreView,
    /// Sessions requests held checked out, on top of the store's.
    pub active_sessions: usize,
    pub active_bytes: u64,
    pub weights_pinned: bool,
}

// --- the writer ------------------------------------------------------------------

/// The recorder's connection and what it reports: every write goes through
/// here, and none of them panics on a SQLite error.
pub struct Writer {
    conn: Connection,
    path: Option<PathBuf>,
    health: Arc<Health>,
    /// Kinds of write whose last attempt failed.
    failing: HashSet<&'static str>,
    /// A batch of request rows that failed once, tried again with the next.
    retry: Option<Vec<RequestRow>>,
    /// No request starts before this (unix ms; the process's start).
    floor_ms: u64,
}

impl Writer {
    /// A writer over `conn` (prepared by [`open`] or `in_memory`), whose
    /// file is `path` (`None` in memory), for a process started at
    /// `floor_ms`.
    pub fn new(
        conn: Connection,
        path: Option<PathBuf>,
        health: Arc<Health>,
        floor_ms: u64,
    ) -> Self {
        let writer =
            Self { conn, path, health, failing: HashSet::new(), retry: None, floor_ms };
        writer.health.ok.store(true, Ordering::Relaxed);
        writer
    }

    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    pub fn health(&self) -> &Arc<Health> {
        &self.health
    }

    /// Notes how a write of `kind` went: logs a failure once per kind, and
    /// the recorder is ok while no kind of write is failing.
    fn outcome<T>(
        &mut self,
        kind: &'static str,
        result: rusqlite::Result<T>,
    ) -> Option<T> {
        let value = match result {
            Ok(value) => {
                self.failing.remove(kind);
                // A settings write that failed is a single change the
                // dashboard was told about, not a kind of write that keeps
                // failing: any success clears it.
                self.failing.retain(|k| !SETTINGS_WRITES.contains(k));
                Some(value)
            }
            Err(error) => {
                self.failing.insert(kind);
                self.health.log_once(kind, &error);
                None
            }
        };
        self.health.ok.store(self.failing.is_empty(), Ordering::Relaxed);
        value
    }

    /// A read that is not a write: a failure is logged once and leaves the
    /// recorder's health alone.
    fn read<T>(&self, kind: &'static str, result: rusqlite::Result<T>) -> Option<T> {
        result.map_err(|error| self.health.log_once(kind, &error)).ok()
    }

    /// Stores `rows` and recomputes every minute they ran through, in one
    /// transaction; a batch that failed before is tried once more first,
    /// and dropped (its rows counted) if it fails again. A batch that fails
    /// now is kept for the next call. Returns whether `rows` were stored.
    pub fn record(&mut self, rows: Vec<RequestRow>) -> bool {
        if let Some(old) = self.retry.take()
            && !self.write(&old)
        {
            self.health.drop_rows(old.len());
        }
        let stored = self.write(&rows);
        if !stored {
            self.retry = Some(rows);
        }
        stored
    }

    /// Gives up the batch waiting for a retry, counting its rows.
    pub fn give_up_retry(&mut self) {
        if let Some(old) = self.retry.take()
            && !self.write(&old)
        {
            self.health.drop_rows(old.len());
        }
    }

    fn write(&mut self, rows: &[RequestRow]) -> bool {
        let result = Self::try_write(&mut self.conn, rows, self.floor_ms, &self.health);
        let Some(skipped) = self.outcome("writing request rows", result) else {
            return false;
        };
        self.health.drop_rows(skipped);
        if self.health.oldest.load(Ordering::Relaxed) == 0 {
            self.publish_extent();
        }
        true
    }

    /// Inserts each row on its own (a row SQLite refuses is skipped and
    /// counted in the returned number), then recomputes the minutes.
    fn try_write(
        conn: &mut Connection,
        rows: &[RequestRow],
        floor_ms: u64,
        health: &Health,
    ) -> rusqlite::Result<usize> {
        let tx = conn.transaction()?;
        let mut minutes = BTreeSet::new();
        let mut skipped = 0;
        for row in rows {
            let span = Span::of(row, floor_ms);
            match insert(&tx, row, &span) {
                Ok(()) => minutes.extend(span.minutes()),
                Err(error) if refuses_the_row(&error) => {
                    skipped += 1;
                    health.log_once("storing a request row", &error);
                }
                Err(error) => return Err(error),
            }
        }
        for minute in minutes {
            roll_up(&tx, minute)?;
        }
        tx.commit()?;
        Ok(skipped)
    }

    /// Stores a finished minute of machine samples. A minute already stored
    /// is replaced: after the clock went back, the latest samples of a
    /// minute are the ones kept.
    pub fn machine(&mut self, m: &MachineMinute) -> bool {
        let result = self
            .conn
            .prepare_cached(
                "INSERT OR REPLACE INTO machine (minute, samples, cpu_load_min, cpu_load_mean,
                     cpu_load_max, memory_pressure, swap_bytes_mean, swap_bytes_max,
                     ram_used_bytes_mean, lily_resident_bytes_max, gpu_active_mean,
                     gpu_freq_mhz_min, gpu_freq_mhz_mean, gpu_freq_mhz_max, power_w_min,
                     power_w_mean, power_w_max, gpu_power_w_mean)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                     ?16, ?17, ?18)",
            )
            .and_then(|mut insert| {
                insert.execute(params![
                    m.minute,
                    m.samples,
                    m.cpu_load.min(),
                    m.cpu_load.mean(),
                    m.cpu_load.max(),
                    m.pressure_level,
                    m.swap_bytes.mean(),
                    m.swap_bytes.max().map(|b| b as u64),
                    m.ram_used_bytes.mean(),
                    m.lily_resident_bytes.max().map(|b| b as u64),
                    m.gpu_active.mean(),
                    m.gpu_freq_mhz.min(),
                    m.gpu_freq_mhz.mean(),
                    m.gpu_freq_mhz.max(),
                    m.power_w.min(),
                    m.power_w.mean(),
                    m.power_w.max(),
                    m.gpu_power_w.mean(),
                ])
            });
        let ok = self.outcome("writing machine samples", result).is_some();
        if ok && self.health.oldest.load(Ordering::Relaxed) == 0 {
            self.publish_extent();
        }
        ok
    }

    /// Stores the tiers as they were at the end of minute `minute`. Nothing
    /// serves this table yet: it is kept per § 4.1 for a later tier-history
    /// view. A minute already stored is replaced (keep-latest, as for the
    /// machine).
    pub fn tiers(&mut self, minute: u64, t: &TierSnapshot) -> bool {
        let disk = t.store.disk.as_ref();
        let experts = t.store.experts.as_ref();
        let result = self
            .conn
            .prepare_cached(
                "INSERT OR REPLACE INTO tiers (minute, gpu_bytes, gpu_budget_bytes, gpu_entries,
                     active_sessions, disk_bytes, disk_budget_bytes, disk_entries,
                     disk_durable_entries, disk_durable_bytes, expert_slots, expert_hit_rate,
                     weights_pinned)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            )
            .and_then(|mut insert| {
                insert.execute(params![
                    minute,
                    t.store.gpu.used_bytes + t.active_bytes,
                    t.store.gpu.budget_bytes,
                    (t.store.gpu.entries + t.active_sessions) as u64,
                    t.active_sessions as u64,
                    disk.map(|d| d.used_bytes),
                    disk.map(|d| d.budget_bytes),
                    disk.map(|d| d.entries as u64),
                    disk.map(|d| d.durable_entries as u64),
                    disk.map(|d| d.durable_bytes),
                    experts.map(|e| e.slots as u64),
                    experts.and_then(|e| e.hit_rate),
                    t.weights_pinned,
                ])
            });
        self.outcome("writing tier snapshots", result).is_some()
    }

    /// Where pruning at `now_ms` cuts: the retention plus a margin, rounded
    /// down to the hour, so the 7d range's oldest bucket is whole.
    pub fn prune_cutoff(now_ms: u64) -> u64 {
        now_ms.saturating_sub(RETENTION_MS + PRUNE_MARGIN_MS) / 3_600_000 * 3_600_000
    }

    /// Deletes everything older than [`Self::prune_cutoff`], hands the
    /// freed pages back (incremental vacuum) and truncates the WAL.
    pub fn prune(&mut self, now_ms: u64) -> bool {
        let cutoff = Self::prune_cutoff(now_ms);
        let result = (|| {
            let tx = self.conn.transaction()?;
            for (table, column, in_minutes) in PRUNED {
                let before = if in_minutes { cutoff / MINUTE_MS } else { cutoff };
                tx.execute(
                    &format!("DELETE FROM {table} WHERE {column} < ?1"),
                    [before],
                )?;
            }
            tx.commit()?;
            // One freed page per step: step it to the end.
            let mut vacuum = self.conn.prepare("PRAGMA incremental_vacuum")?;
            let mut steps = vacuum.query([])?;
            while steps.next()?.is_some() {}
            drop(steps);
            drop(vacuum);
            self.checkpoint()
        })();
        let ok = self.outcome("pruning", result).is_some();
        self.publish_extent();
        ok
    }

    /// Checkpoints the WAL into the database and truncates it.
    fn checkpoint(&self) -> rusqlite::Result<()> {
        self.conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()))
    }

    /// Publishes the oldest sample and the file's size for the status route.
    pub fn publish_extent(&mut self) {
        if let Some(oldest) = self.read("reading the oldest sample", oldest(&self.conn))
        {
            self.health.oldest.store(oldest.unwrap_or(0), Ordering::Relaxed);
        }
        if let Some(path) = &self.path {
            let size = |p: &Path| std::fs::metadata(p).map_or(0, |m| m.len());
            let bytes = size(path) + size(&with_suffix(path, "-wal"));
            self.health.db_bytes.store(bytes, Ordering::Relaxed);
        }
    }

    /// The database's largest lineage id, read once at the start; a failure
    /// is logged and nothing more.
    pub fn max_lineage(&self) -> Option<u64> {
        self.read("reading lineage ids", max_lineage(&self.conn)).flatten()
    }

    /// The overrides the database holds; a failure is logged and reads as
    /// none (the command line stays in effect).
    pub fn stored_settings(&self) -> Option<Vec<(String, String)>> {
        self.read("reading the settings", stored_settings(&self.conn))
    }

    /// Writes `plan` at `t`: each changed setting's override set or deleted
    /// and its change logged from the dashboard, in one transaction.
    /// Returns why it failed (nothing is written then).
    pub fn settings(&mut self, plan: &Plan, t: u64) -> Result<(), String> {
        let result = (|| {
            let tx = self.conn.transaction()?;
            // A reset deletes the row by name, also one that was never in
            // effect (a stale row the start ignored).
            for name in plan.reset_rows() {
                tx.execute("DELETE FROM settings WHERE name = ?1", [name])?;
            }
            for e in &plan.entries {
                if let Some(value) = plan.override_of(&e.name) {
                    tx.execute(
                        "INSERT OR REPLACE INTO settings (name, value, changed_t) VALUES (?1, ?2, ?3)",
                        params![e.name, json_text(&Some(value.clone())), t],
                    )?;
                }
            }
            log_entries(&tx, &plan.entries, SettingsLogSource::Dashboard, t)?;
            tx.commit()
        })();
        let why = result.as_ref().err().map(|e| e.to_string());
        match (self.outcome("writing settings", result), why) {
            (Some(()), _) => Ok(()),
            (None, why) => Err(format!(
                "the database refused the change ({}); nothing changed",
                why.unwrap_or_default()
            )),
        }
    }

    /// Logs `entries` at `t` as applied at startup.
    pub fn startup_entries(&mut self, entries: &[Entry], t: u64) -> bool {
        let result = (|| {
            let tx = self.conn.transaction()?;
            log_entries(&tx, entries, SettingsLogSource::Startup, t)?;
            tx.commit()
        })();
        self.outcome("writing the settings log", result).is_some()
    }

    /// The last act before the thread ends: the retry, then a checkpoint.
    fn close(&mut self) {
        self.give_up_retry();
        let result = self.checkpoint();
        self.outcome("checkpointing at shutdown", result);
    }
}

// --- the thread --------------------------------------------------------------------

/// Marks the recorder not ok when its thread ends, a panic included.
struct Ended(Arc<Health>);

impl Drop for Ended {
    fn drop(&mut self) {
        self.0.ok.store(false, Ordering::Relaxed);
        let (state, _) = self.0.phase();
        if std::thread::panicking() {
            eprintln!("recorder: the thread panicked; serving on without history");
            self.0
                .set_phase(RecorderState::Failed, Some("the recorder panicked".into()));
        } else if state != RecorderState::Failed {
            self.0
                .set_phase(RecorderState::Failed, Some("the recorder stopped".into()));
        }
    }
}

/// The recorder thread's state.
struct Recording {
    path: PathBuf,
    health: Arc<Health>,
    rows: Receiver<Msg>,
    board: Weak<Board>,
    started_ms: u64,
    machine: Option<MachineMinute>,
    ticks: Option<CpuTicks>,
    /// The runtime settings it persists (`None`: tests without any).
    settings: Option<Arc<RuntimeSettings>>,
    /// Settings changes received and not answered yet.
    changes: Vec<SettingsWrite>,
}

/// What a wait on the channel saw.
enum Waited {
    /// The wait ran out.
    Time,
    /// Stop, or every sender is gone.
    Stop,
}

impl Recording {
    fn run(mut self) {
        let _ended = Ended(self.health.clone());
        self.record();
        // Rows still queued will never be written.
        let queued = self.rows.try_iter().filter(|m| matches!(m, Msg::Row(_))).count();
        self.health.drop_rows(queued);
    }

    fn record(&mut self) {
        let mut pending = Vec::new();
        let Some(mut writer) = self.open(&mut pending) else { return };
        if !pending.is_empty() {
            writer.record(std::mem::take(&mut pending));
        }
        let mut next_sample = Instant::now();
        let mut next_prune = Instant::now();
        loop {
            let until = next_sample.min(next_prune);
            let mut batch = Vec::new();
            let waited = self.receive(until, &mut batch, BATCH_ROWS);
            if !batch.is_empty() {
                writer.record(batch);
            }
            self.change_settings(&mut writer);
            if let Waited::Stop = waited {
                break;
            }
            let now = Instant::now();
            if now >= next_sample {
                if !self.sample(&mut writer, super::now_ms()) {
                    break;
                }
                next_sample = now + SAMPLE_EVERY;
            }
            if now >= next_prune {
                writer.prune(super::now_ms());
                next_prune = now + PRUNE_EVERY;
            }
        }
        if let Some(m) = self.machine.take() {
            writer.machine(&m);
        }
        writer.close();
    }

    /// Receives rows into `into` until `until`, a stop, or `cap` rows (the
    /// excess counted as dropped), settings changes into `self.changes`;
    /// returns at the first message once one came.
    fn receive(
        &mut self,
        until: Instant,
        into: &mut Vec<RequestRow>,
        cap: usize,
    ) -> Waited {
        let wait = until.saturating_duration_since(Instant::now());
        match self.rows.recv_timeout(wait) {
            Ok(msg) => {
                if self.take(msg, into, cap) {
                    return Waited::Stop;
                }
            }
            Err(RecvTimeoutError::Disconnected) => return Waited::Stop,
            Err(RecvTimeoutError::Timeout) => return Waited::Time,
        }
        // Whatever else is queued joins the batch.
        while into.len() < cap {
            match self.rows.try_recv() {
                Ok(msg) => {
                    if self.take(msg, into, cap) {
                        return Waited::Stop;
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => return Waited::Stop,
                Err(mpsc::TryRecvError::Empty) => break,
            }
        }
        Waited::Time
    }

    /// Files `msg`: a row into `into` (past `cap` counted as dropped), a
    /// settings change for later. `true` for a stop.
    fn take(&mut self, msg: Msg, into: &mut Vec<RequestRow>, cap: usize) -> bool {
        match msg {
            Msg::Row(row) if into.len() < cap => into.push(*row),
            Msg::Row(_) => self.health.drop_rows(1),
            Msg::Settings(write) => self.changes.push(*write),
            Msg::Stop => return true,
        }
        false
    }

    /// Answers every settings change received with `why`, making none.
    fn refuse_changes(&mut self, why: &str) {
        for write in self.changes.drain(..) {
            let _ = write.reply.try_send(Err(WriteError::not_applied(why)));
        }
    }

    /// The first attempt to open is over: the disk tier may expire with
    /// the TTL in effect now.
    fn settle(&self) {
        if let Some(settings) = &self.settings {
            settings.settle();
        }
    }

    /// Makes the settings changes received: each planned against the state
    /// now, written, then put into effect, and answered. A change whose
    /// sender stopped waiting is not made.
    fn change_settings(&mut self, writer: &mut Writer) {
        for write in std::mem::take(&mut self.changes) {
            let SettingsWrite { change, reply, claim } = write;
            if !claim.take() {
                // Its sender stopped waiting first: it is never made.
                continue;
            }
            let result = match &self.settings {
                None => Err(WriteError::not_applied(
                    "this recorder keeps no settings; nothing changed",
                )),
                Some(settings) => match settings.plan(&change) {
                    Err(why) => Err(WriteError {
                        status: 500,
                        message: format!(
                            "the change could not be worked out ({why}); nothing changed"
                        ),
                    }),
                    Ok(plan) => {
                        let written = if plan.writes_anything() {
                            writer.settings(&plan, super::now_ms())
                        } else {
                            Ok(())
                        };
                        match written {
                            Ok(()) => {
                                let entries = plan.entries.clone();
                                settings.publish(plan);
                                Ok(entries)
                            }
                            // Nothing written, nothing published.
                            Err(why) => Err(WriteError::not_applied(why)),
                        }
                    }
                },
            };
            let _ = reply.try_send(result);
        }
    }

    /// Applies the overrides the database holds over the command line, logs
    /// which, and records a `startup` entry for each.
    fn apply_stored_settings(&self, writer: &mut Writer) {
        let Some(settings) = &self.settings else { return };
        // Read twice before giving up on them.
        let Some(rows) = writer.stored_settings().or_else(|| writer.stored_settings())
        else {
            eprintln!(
                "settings: the database's overrides could not be read; the command line stays in effect and settings changes are refused"
            );
            self.health.settings_unreadable.store(true, Ordering::Release);
            return;
        };
        if rows.is_empty() {
            return;
        }
        let startup = settings.apply_stored(&rows);
        for (name, why) in &startup.ignored {
            eprintln!(
                "settings: ignoring the database's {name}: {why}; the row is kept"
            );
        }
        if !startup.applied.is_empty() {
            let list: Vec<String> = startup
                .applied
                .iter()
                .map(|e| {
                    format!("{} {}", e.name, json_text(&e.new).unwrap_or_default())
                })
                .collect();
            eprintln!(
                "settings: from the database, over the command line: {}",
                list.join(", ")
            );
            // The log records what changed: an override equal to the value
            // it replaces changed nothing.
            let changed: Vec<Entry> = startup
                .applied
                .iter()
                .filter(|e| !same_value(&e.old, &e.new))
                .cloned()
                .collect();
            if !changed.is_empty() {
                writer.startup_entries(&changed, super::now_ms());
            }
        }
    }

    /// Opens the database, retrying with [`OPEN_BACKOFF`] while rows wait in
    /// `pending` (at most [`ROW_CHANNEL`], the rest counted as dropped).
    /// `None`: a newer schema, or a stop before the open succeeded.
    fn open(&mut self, pending: &mut Vec<RequestRow>) -> Option<Writer> {
        let mut attempt = 0;
        let mut last_error = String::new();
        loop {
            match open(&self.path, super::now_ms() / 1000) {
                Ok((conn, aside)) => {
                    if let Some(SetAside { to, why }) = &aside {
                        eprintln!(
                            "recorder: {} was unusable ({why}); set aside as {} and started afresh",
                            self.path.display(),
                            to.display()
                        );
                    }
                    let mut writer = Writer::new(
                        conn,
                        Some(self.path.clone()),
                        self.health.clone(),
                        self.started_ms,
                    );
                    if let Some(max) = writer.max_lineage()
                        && let Some(board) = self.board.upgrade()
                    {
                        board.lineages().raise_above(max);
                    }
                    writer.publish_extent();
                    // Before `open`: no change from the dashboard races it.
                    self.apply_stored_settings(&mut writer);
                    self.settle();
                    self.health.opened.store(true, Ordering::Release);
                    self.health.set_phase(RecorderState::Open, None);
                    eprintln!(
                        "recorder: {} (7-day history){}",
                        self.path.display(),
                        if attempt > 0 {
                            format!(", opened after {} attempts", attempt + 1)
                        } else {
                            String::new()
                        }
                    );
                    return Some(writer);
                }
                Err(OpenFailure::Newer { found }) => {
                    let message = format!(
                        "database {} has schema {found}, this lily knows {SCHEMA_VERSION}: history off",
                        self.path.display()
                    );
                    eprintln!("recorder: {message}");
                    self.health.set_phase(RecorderState::Failed, Some(message));
                    self.settle();
                    self.health.drop_rows(pending.len());
                    self.refuse_changes(
                        "the recorder's database cannot be used; nothing changed",
                    );
                    return None;
                }
                Err(failure @ OpenFailure::Other(_)) => {
                    let error = failure.to_string();
                    if error != last_error {
                        eprintln!(
                            "recorder: cannot open the database: {error}; retrying, serving without history meanwhile"
                        );
                    }
                    self.health.set_phase(RecorderState::Opening, Some(error.clone()));
                    last_error = error;
                    // The flags stay in effect until an open succeeds.
                    self.settle();
                }
            }
            let wait = OPEN_BACKOFF[attempt.min(OPEN_BACKOFF.len() - 1)];
            attempt += 1;
            let until = Instant::now() + wait;
            loop {
                let waited = self.receive(until, pending, ROW_CHANNEL);
                self.refuse_changes(
                    "the recorder's database is not open yet (see the server log); nothing changed",
                );
                match waited {
                    Waited::Stop => {
                        self.health.drop_rows(pending.len());
                        return None;
                    }
                    Waited::Time if Instant::now() >= until => break,
                    Waited::Time => {}
                }
            }
        }
    }

    /// Samples the machine at `now_ms`; when a minute ended since the last
    /// sample (or the clock went back to another), stores it and the tiers.
    /// `false` once the board is gone.
    fn sample(&mut self, writer: &mut Writer, now_ms: u64) -> bool {
        let Some(board) = self.board.upgrade() else { return false };
        let minute = now_ms / MINUTE_MS;
        if let Some(m) = self.machine.take_if(|m| m.minute != minute) {
            writer.machine(&m);
            writer.tiers(m.minute, &board.tier_snapshot());
            writer.publish_extent();
        }
        drop(board);
        let sample = MachineSample::now(&mut self.ticks);
        self.machine.get_or_insert_with(|| MachineMinute::new(minute)).add(&sample);
        true
    }
}

// --- the HTTP side ------------------------------------------------------------------

/// Why the recorder cannot answer a read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unavailable {
    /// `--stats-db off`.
    Off,
    /// The database is not open yet.
    Opening,
    /// The database will not be opened (a newer schema).
    Failed,
    /// Reading it failed now: a lock held too long, or an error (logged
    /// once).
    Read,
}

impl Unavailable {
    /// For the change log, which lives in the same database.
    pub fn log_message(self) -> &'static str {
        match self {
            Self::Off => {
                "the change log is kept in the recorder's database, which is off (--stats-db off)"
            }
            other => other.message(),
        }
    }

    pub fn message(self) -> &'static str {
        match self {
            Self::Off => "history is off (--stats-db off)",
            Self::Opening => {
                "the recorder is still opening its database; see the server log"
            }
            Self::Failed => {
                "the recorder's database cannot be used; see the server log"
            }
            Self::Read => {
                "the recorder's database could not be read just now; see the server log"
            }
        }
    }
}

#[derive(Debug)]
struct Inner {
    /// `None`: `--stats-db off`.
    path: Option<PathBuf>,
    health: Arc<Health>,
    /// The thread's channel, for the stop.
    stop: Option<SyncSender<Msg>>,
    thread: Mutex<Option<JoinHandle<()>>>,
    read_failed: AtomicBool,
}

/// The dashboard's handle on the recorder: its health, read-only access to
/// its database, and its stop.
#[derive(Debug, Clone)]
pub struct Recorder {
    inner: Arc<Inner>,
}

impl Recorder {
    fn with(path: Option<PathBuf>, health: Arc<Health>) -> Self {
        Self {
            inner: Arc::new(Inner {
                path,
                health,
                stop: None,
                thread: Mutex::new(None),
                read_failed: AtomicBool::new(false),
            }),
        }
    }

    /// No recorder (`--stats-db off`, and tests that need none).
    pub fn off() -> Self {
        Self::with(None, Arc::default())
    }

    /// Starts the recorder thread on the database at `path` (`None`:
    /// `--stats-db off`) and attaches the board's row sink; the thread
    /// persists `settings`. Returns at once; the thread opens the database
    /// (see the module docs).
    pub fn start(
        path: Option<&Path>,
        board: &Arc<Board>,
        settings: Option<Arc<RuntimeSettings>>,
    ) -> Self {
        let Some(path) = path else {
            eprintln!("recorder: off (--stats-db off); the dashboard shows no history");
            if let Some(settings) = &settings {
                settings.settle();
            }
            return Self::off();
        };
        let settle = settings.clone();
        let health = Arc::new(Health::default());
        let (rows, rx) = mpsc::sync_channel(ROW_CHANNEL);
        let recording = Recording {
            path: path.to_path_buf(),
            health: health.clone(),
            rows: rx,
            board: Arc::downgrade(board),
            started_ms: super::now_ms(),
            machine: None,
            ticks: None,
            settings,
            changes: Vec::new(),
        };
        let thread = std::thread::Builder::new()
            .name("lily-recorder".into())
            .spawn(move || recording.run());
        let thread = match thread {
            Ok(thread) => Some(thread),
            Err(error) => {
                let message = format!("spawning the recorder thread: {error}");
                eprintln!("recorder: {message}; serving without history");
                health.set_phase(RecorderState::Failed, Some(message));
                if let Some(settings) = &settle {
                    settings.settle();
                }
                None
            }
        };
        board.attach_recorder(RowSink { rows: rows.clone(), health: health.clone() });
        Self {
            inner: Arc::new(Inner {
                path: Some(path.to_path_buf()),
                health,
                stop: Some(rows),
                thread: Mutex::new(thread),
                read_failed: AtomicBool::new(false),
            }),
        }
    }

    /// A recorder over an existing database at `path` with `health`, no
    /// thread: tests of the read side.
    #[cfg(test)]
    pub fn reading(path: &Path, health: Arc<Health>) -> Self {
        health.opened.store(true, Ordering::Release);
        health.set_phase(RecorderState::Open, None);
        Self::with(Some(path.to_path_buf()), health)
    }

    /// Stops the thread: it writes what it holds, checkpoints and ends.
    /// Waits at most `cap`; a thread still busy then is left to the
    /// process's exit (SQLite's WAL keeps the database whole).
    pub fn finish(&self, cap: Duration) {
        let Some(stop) = &self.inner.stop else { return };
        let deadline = Instant::now() + cap;
        loop {
            match stop.try_send(Msg::Stop) {
                Ok(()) | Err(TrySendError::Disconnected(_)) => break,
                Err(TrySendError::Full(_)) if Instant::now() >= deadline => {
                    eprintln!(
                        "recorder: still busy after {cap:?}; stopping without it"
                    );
                    return;
                }
                Err(TrySendError::Full(_)) => {
                    std::thread::sleep(Duration::from_millis(5))
                }
            }
        }
        let Some(thread) =
            self.inner.thread.lock().unwrap_or_else(|p| p.into_inner()).take()
        else {
            return;
        };
        while !thread.is_finished() {
            if Instant::now() >= deadline {
                eprintln!("recorder: still busy after {cap:?}; stopping without it");
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let _ = thread.join();
    }

    /// The status route's `recorder`.
    pub fn status(&self) -> RecorderStatus {
        let health = &self.inner.health;
        if self.inner.path.is_none() {
            return RecorderStatus {
                state: RecorderState::Off,
                ok: false,
                dropped_rows: 0,
                db_bytes: 0,
                oldest: None,
                error: None,
            };
        }
        let (state, error) = health.phase();
        let oldest = health.oldest.load(Ordering::Relaxed);
        RecorderStatus {
            state,
            ok: state == RecorderState::Open && health.ok.load(Ordering::Relaxed),
            dropped_rows: health.dropped.load(Ordering::Relaxed),
            db_bytes: health.db_bytes.load(Ordering::Relaxed),
            oldest: (oldest > 0).then_some(oldest),
            error,
        }
    }

    /// Why a settings change cannot be made now, as the route answers it
    /// (503); `None` while the database is open.
    pub fn settings_unavailable(&self) -> Option<&'static str> {
        if self.inner.path.is_none() {
            return Some(
                "settings changes are kept in the recorder's database, which is off (--stats-db off); nothing changed",
            );
        }
        match self.inner.health.phase().0 {
            RecorderState::Open
                if self.inner.health.settings_unreadable.load(Ordering::Acquire) =>
            {
                Some(
                    "settings could not be read from the database (see the server log); nothing changed",
                )
            }
            RecorderState::Open => None,
            RecorderState::Off | RecorderState::Opening => Some(
                "the recorder's database is not open yet (see the server log); nothing changed",
            ),
            RecorderState::Failed => Some(
                "the recorder's database cannot be used (see the server log); nothing changed",
            ),
        }
    }

    /// Has the recorder thread write `change` and put it into effect, and
    /// waits at most `wait` for that. Returns the entries of what changed,
    /// or why nothing did (503), or that the outcome is unknown (504: the
    /// thread took the change up but did not answer in time), or that it
    /// could not be worked out (500). A change the wait gave up on before
    /// the thread took it up is never made.
    pub fn change_settings(
        &self,
        change: Change,
        wait: Duration,
    ) -> Result<Vec<Entry>, WriteError> {
        if let Some(why) = self.settings_unavailable() {
            return Err(WriteError::not_applied(why));
        }
        let Some(sender) = &self.inner.stop else {
            return Err(WriteError::not_applied(
                "the recorder is not running; nothing changed",
            ));
        };
        let (reply, answer) = mpsc::sync_channel(1);
        let claim = Arc::new(Claim::default());
        let write = SettingsWrite { change, reply, claim: claim.clone() };
        match sender.try_send(Msg::Settings(Box::new(write))) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                return Err(WriteError::not_applied(
                    "the recorder is busy; nothing changed, try again",
                ));
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err(WriteError::not_applied(
                    "the recorder stopped; nothing changed",
                ));
            }
        }
        match answer.recv_timeout(wait) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) if claim.abandon() => {
                Err(WriteError::not_applied(format!(
                    "the database did not take the change up within {} s; nothing changed",
                    wait.as_secs()
                )))
            }
            // Taken and being written: it may land after this answer.
            Err(RecvTimeoutError::Timeout) => {
                Err(WriteError { status: 504, message: OUTCOME_UNKNOWN.to_owned() })
            }
            Err(RecvTimeoutError::Disconnected) => Err(WriteError::not_applied(
                "the recorder stopped before writing the change; nothing changed",
            )),
        }
    }

    /// A recorder that says `open` over a channel the test reads itself
    /// (the test plays the recorder thread).
    #[cfg(test)]
    pub fn on_channel(capacity: usize) -> (Self, Receiver<Msg>) {
        let (tx, rx) = mpsc::sync_channel(capacity);
        let health = Arc::new(Health::default());
        health.opened.store(true, Ordering::Release);
        health.set_phase(RecorderState::Open, None);
        let recorder = Self {
            inner: Arc::new(Inner {
                path: Some(PathBuf::from("/nonexistent/lily.sqlite")),
                health,
                stop: Some(tx),
                thread: Mutex::new(None),
                read_failed: AtomicBool::new(false),
            }),
        };
        (recorder, rx)
    }

    /// Runs `read` on a read-only connection of its own. A failure is
    /// logged once and answered as [`Unavailable::Read`].
    pub fn read<T>(
        &self,
        read: impl FnOnce(&Connection) -> rusqlite::Result<T>,
    ) -> Result<T, Unavailable> {
        let Some(path) = &self.inner.path else { return Err(Unavailable::Off) };
        if !self.inner.health.opened.load(Ordering::Acquire) {
            return Err(match self.inner.health.phase().0 {
                RecorderState::Failed => Unavailable::Failed,
                _ => Unavailable::Opening,
            });
        }
        open_reader(path).and_then(|conn| read(&conn)).map_err(|error| {
            if !self.inner.read_failed.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "recorder: reading {} failed: {error}; later read errors are not logged",
                    path.display()
                );
            }
            Unavailable::Read
        })
    }
}

/// `--stats-db`'s default: `~/Library/Application Support/lily/lily.sqlite`.
pub fn default_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join("Library").join("Application Support").join("lily").join("lily.sqlite")
}

/// `--stats-db`'s text as a path: `None` for `off`, the default without the
/// flag. An empty or relative path is refused (no `~` expansion, as for
/// `--disk-cache-dir`).
pub fn stats_db(flag: Option<&str>) -> Result<Option<PathBuf>, String> {
    match flag {
        None => Ok(Some(default_path())),
        Some("off") => Ok(None),
        Some("") => Err("--stats-db needs an absolute path, or off".to_owned()),
        Some(path) if !Path::new(path).is_absolute() => Err(format!(
            "--stats-db needs an absolute path, or off; {path:?} is relative (a leading ~ is not expanded)"
        )),
        Some(path) => Ok(Some(PathBuf::from(path))),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/serve/recorder.rs"]
mod tests;
