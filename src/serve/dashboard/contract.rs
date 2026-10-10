//! The response shapes of `/v1/lily/*`, field for field those of
//! `dashboard/src/api/types.ts`, which is the contract (docs/dashboard.md
//! § 3.4): snake_case names, unix milliseconds, bytes, milliseconds, tokens
//! per second, ratios in `0..=1`, and `null` (never an absent field, never 0)
//! for a value lily does not have. The fixtures in
//! `dashboard/src/api/fixtures.ts` are these types serialized; a test compares
//! them, and the dashboard's typecheck holds them against `types.ts`.
//!
//! The settings writes' answers are not here yet: their routes answer 409
//! until runtime settings (phase 4) exist.

use serde::Serialize;

/// Body of every 4xx and 5xx answer of the dashboard's routes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    /// The setting the error is about, when there is one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub setting: Option<String>,
}

/// lily's own memory-pressure names (`stats::pressure_level`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryPressure {
    Normal,
    Warning,
    Critical,
}

impl MemoryPressure {
    /// The name of a `kern.memorystatus_vm_pressure_level`; `None` for a
    /// level the kernel does not document.
    pub fn of_level(level: u32) -> Option<Self> {
        match level {
            1 => Some(Self::Normal),
            2 => Some(Self::Warning),
            4 => Some(Self::Critical),
            _ => None,
        }
    }
}

// --- GET /v1/lily/status ---------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LifecycleState {
    Unloaded,
    Loading,
    Ready,
    Busy,
    Stopping,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QueueStatus {
    pub waiting: usize,
    pub running: usize,
    pub max_batch: usize,
    pub capacity: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LiveStatus {
    pub decode_tps: f64,
    pub prefill_tps: f64,
    pub acceptance: Option<f64>,
    pub requests_per_min: usize,
    pub active_sessions: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Power {
    pub cpu_w: f64,
    pub gpu_w: f64,
    pub ane_w: f64,
    pub sys_w: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SamplerState {
    Ok,
    Off,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Temperatures {
    pub cpu: f64,
    pub gpu: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MachineStatus {
    pub sampler: SamplerState,
    pub cpu_load: Option<f64>,
    pub gpu_active: Option<f64>,
    pub gpu_freq_mhz: Option<f64>,
    pub memory_pressure: Option<MemoryPressure>,
    pub ram_total_bytes: Option<u64>,
    pub ram_used_bytes: Option<u64>,
    pub lily_resident_bytes: Option<u64>,
    pub swap_used_bytes: Option<u64>,
    pub power: Option<Power>,
    pub temp_c: Option<Temperatures>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GpuTier {
    pub used_bytes: u64,
    /// `None` until a load derived it, unless `--cache-bytes` set it.
    pub budget_bytes: Option<u64>,
    pub entries: usize,
    pub max_entries: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct DiskTier {
    pub used_bytes: u64,
    pub budget_bytes: u64,
    pub entries: usize,
    pub durable_entries: usize,
    pub durable_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExpertTier {
    pub slots: usize,
    pub used: usize,
    /// Cumulative since the load; `None` before the first lookup.
    pub hit_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NgramTier {
    pub bytes: u64,
    pub resident_share: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WeightsTier {
    pub bytes: u64,
    pub pinned: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Tiers {
    pub gpu: GpuTier,
    pub disk: Option<DiskTier>,
    pub experts: Option<ExpertTier>,
    /// `None` without a loaded model that has a table.
    pub ngram: Option<NgramTier>,
    pub weights: WeightsTier,
}

/// Where the recorder is: `off` (`--stats-db off`), `opening` (its thread
/// is opening the database, retrying after a failure), `open`, `failed` (it
/// will not open, or its thread ended).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RecorderState {
    Off,
    Opening,
    Open,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecorderStatus {
    pub state: RecorderState,
    pub ok: bool,
    pub dropped_rows: u64,
    pub db_bytes: u64,
    pub oldest: Option<u64>,
    /// Why it is `opening` or `failed`; `None` otherwise.
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Status {
    pub now: u64,
    pub state: LifecycleState,
    pub model: String,
    pub uptime_s: u64,
    pub pin_hold_s: Option<u64>,
    pub idle_unload_s: Option<u64>,
    pub idle_for_s: u64,
    pub queue: QueueStatus,
    pub live: LiveStatus,
    pub machine: MachineStatus,
    pub tiers: Tiers,
    pub recorder: RecorderStatus,
}

// --- GET /v1/lily/history and /v1/lily/sessions/{lineage}/history -----------

/// A history's interval; each has its own bucket width ([`Range::step_ms`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Range {
    #[serde(rename = "15m")]
    QuarterHour,
    #[serde(rename = "1h")]
    Hour,
    #[serde(rename = "24h")]
    Day,
    #[serde(rename = "7d")]
    Week,
}

impl Range {
    pub const ALL: [Self; 4] = [Self::QuarterHour, Self::Hour, Self::Day, Self::Week];

    /// The range a `range=` value names, as the contract spells it.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "15m" => Some(Self::QuarterHour),
            "1h" => Some(Self::Hour),
            "24h" => Some(Self::Day),
            "7d" => Some(Self::Week),
            _ => None,
        }
    }

    /// The interval, milliseconds.
    pub fn span_ms(self) -> u64 {
        match self {
            Self::QuarterHour => 15 * 60_000,
            Self::Hour => 3_600_000,
            Self::Day => 86_400_000,
            Self::Week => 7 * 86_400_000,
        }
    }

    /// The bucket width (docs/dashboard.md § 3.4), milliseconds: a whole
    /// number of minutes that divides the interval.
    pub fn step_ms(self) -> u64 {
        match self {
            Self::QuarterHour | Self::Hour => 60_000,
            Self::Day => 900_000,
            Self::Week => 3_600_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HistoryPoint {
    /// Bucket end, unix ms.
    pub t: u64,
    pub busy_share: f64,
    pub requests: u64,
    pub decode_tps_avg: Option<f64>,
    pub decode_tps_max: Option<f64>,
    pub prefill_tps_avg: Option<f64>,
    pub prefill_tps_max: Option<f64>,
    pub decode_tokens: u64,
    pub prefill_tokens: u64,
    pub acceptance: Option<f64>,
    pub cache_hit: Option<f64>,
    pub queue_ms_p50: Option<f64>,
    pub queue_ms_max: Option<f64>,
    pub sessions_max: u64,
    /// The machine series: `None` throughout a session's history.
    pub cpu_load: Option<f64>,
    pub gpu_active: Option<f64>,
    pub memory_pressure: Option<MemoryPressure>,
    pub swap_bytes: Option<u64>,
    pub power_w: Option<f64>,
    pub gpu_power_w: Option<f64>,
    pub gpu_freq_mhz: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Rate {
    pub avg: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HistorySummary {
    pub decode: Rate,
    pub prefill: Rate,
    pub tokens_generated: u64,
    pub tokens_prefilled: u64,
    pub requests: u64,
    pub busy_share: f64,
    pub cache_hit: Option<f64>,
    pub acceptance: Option<f64>,
    pub power_avg_w: Option<f64>,
    pub energy_wh: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct History {
    pub range: Range,
    pub step_s: u64,
    /// The complete grid, oldest first.
    pub points: Vec<HistoryPoint>,
    pub summary: HistorySummary,
}

// --- GET /v1/lily/sessions ---------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionTier {
    Gpu,
    Disk,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Session {
    pub lineage: Option<u64>,
    pub tier: SessionTier,
    pub durable: bool,
    pub active: bool,
    pub tokens: usize,
    pub bytes: u64,
    pub last_used_t: u64,
    pub requests: usize,
    pub decode_tps_avg: Option<f64>,
    pub decode_tps_max: Option<f64>,
    pub prefill_tps_avg: Option<f64>,
    pub cache_hit: Option<f64>,
    pub acceptance: Option<f64>,
    /// `None` for an entry written before lineage ids.
    pub created_t: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SessionsResponse {
    pub sessions: Vec<Session>,
}

// --- GET /v1/lily/requests ---------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CancelledBy {
    Client,
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RequestRow {
    pub t: u64,
    pub lineage: Option<u64>,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub prefill_tokens: usize,
    pub prefill_ms: f64,
    pub prefill_tps: Option<f64>,
    pub generated_tokens: usize,
    pub decode_ms: f64,
    pub decode_tps: Option<f64>,
    pub drafted_tokens: Option<usize>,
    pub accepted_tokens: Option<usize>,
    pub queue_ms: f64,
    pub batch: usize,
    pub cancelled_by: Option<CancelledBy>,
    pub thinking_forced: bool,
    pub pinned: bool,
    /// The request failed (an error, a GPU fault); its counts are 0.
    pub failed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RequestsResponse {
    pub requests: Vec<RequestRow>,
}

// --- /v1/lily/settings -------------------------------------------------------

/// A setting's value: JSON number, string or boolean. Integers stay
/// integers in the JSON (`131072`, not `131072.0`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum SettingValue {
    Int(u64),
    Number(f64),
    Text(String),
    Bool(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SettingType {
    Number,
    Int,
    Enum,
    String,
    Bool,
    Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SettingSource {
    Default,
    Flag,
    Database,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Setting {
    pub name: String,
    pub group: String,
    #[serde(rename = "type")]
    pub kind: SettingType,
    pub editable: bool,
    pub flag: String,
    pub default: Option<SettingValue>,
    pub flag_value: Option<SettingValue>,
    pub value: Option<SettingValue>,
    pub value_s: Option<u64>,
    pub source: SettingSource,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub options: Option<Vec<String>>,
    pub help: Option<String>,
    pub applies: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SettingsResponse {
    pub settings: Vec<Setting>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SettingsLogSource {
    Dashboard,
    Startup,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SettingsLogEntry {
    pub t: u64,
    pub name: String,
    pub old: Option<SettingValue>,
    pub new: Option<SettingValue>,
    pub source: SettingsLogSource,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SettingsLogResponse {
    pub entries: Vec<SettingsLogEntry>,
}
