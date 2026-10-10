/**
 * The dashboard's API contract: what `/v1/lily/*` returns (docs/dashboard.md
 * § 3.1). The Rust server implements these shapes; the mock backend and the
 * HTTP backend are both typed against them.
 *
 * Conventions, everywhere:
 *   - field names are snake_case; JSON numbers, never strings
 *   - a unit suffix says the unit: `_bytes`, `_ms`, `_s` (seconds), `_w`
 *     (watts), `_mhz`, `_c` (degrees Celsius), `_tps` (tokens per second),
 *     `_tokens`
 *   - fields named `*_share`, `acceptance`, `cache_hit`, `cpu_load`,
 *     `gpu_active`, `hit_rate` are ratios 0..1
 *   - `t`, `*_t` and `now` are unix time in milliseconds (UTC)
 *   - `null` means "no value": nothing was measured, the probe is
 *     unavailable, or the series has a gap; it is never 0 in disguise. Every
 *     field is always present (explicit `| null`, no optional fields), so the
 *     server writes `null` rather than omitting a key
 *   - numbers only (docs/dashboard.md § 2.1): no prompt text, generated text,
 *     token ids, image data or tool arguments appear in any response
 *
 * Errors are JSON `ErrorBody` with a 4xx status and are shown inline as
 * written:
 *   - 400 a value is invalid (the message is the command-line parser's)
 *   - 403 the write guard refused (not loopback, wrong Content-Type or Origin)
 *   - 404 unknown setting (or unknown lineage: neither a cache entry now nor
 *     in the recorder's 7 days)
 *   - 409 the setting needs a restart and cannot be changed here
 *   - 413 a settings write's body is larger than 64 KiB
 * The history routes and the change log answer 503 while the recorder's
 * database is not open. A settings write is made durable before it takes
 * effect, and its 5xx say what happened:
 *   - 503 nothing changed (no open database, `--stats-db off`, settings the
 *     database could not be read, the write refused, or not taken up in time)
 *   - 504 the outcome is unknown: the database took the change up but did
 *     not confirm it in time ("the change may still apply; reload to see the
 *     current value"); the page reloads the settings and the log
 *   - 500 the change could not be worked out; nothing changed
 */

// ---------------------------------------------------------------------------
// Shared

export const RANGES = ['15m', '1h', '24h', '7d'] as const;
/**
 * Chart interval and its bucket width: 15m = 60 s, 1h = 60 s, 24h = 900 s,
 * 7d = 3600 s.
 */
export type Range = (typeof RANGES)[number];

/** Body of every 4xx answer. */
export interface ErrorBody {
  error: {
    message: string;
    /** The setting the error is about, when there is one. */
    setting?: string;
  };
}

/** lily's own memory-pressure names (`stats::pressure_level`). */
export type MemoryPressure = 'normal' | 'warning' | 'critical';

// ---------------------------------------------------------------------------
// GET /v1/lily/status

export type LifecycleState = 'unloaded' | 'loading' | 'ready' | 'busy' | 'stopping';

export interface QueueStatus {
  /** Requests waiting for a batch row. */
  waiting: number;
  /** Requests running now (occupied batch rows). */
  running: number;
  max_batch: number;
  /** Queue capacity (`--queue`). */
  capacity: number;
}

/**
 * What the engine is doing right now. All zero/null while idle. The rates are
 * measured over about the last 6 s (the first poll after a pause reads 0);
 * prefill tokens are counted per chunk as each finishes.
 */
export interface LiveStatus {
  /** Decode throughput now, tokens per second, summed over the running requests. */
  decode_tps: number;
  /** Prefill throughput now, per computed token; 0 when no prefill is running. */
  prefill_tps: number;
  /** MTP acceptance ratio over the requests finished in the last minute; null when nothing is decoding. */
  acceptance: number | null;
  requests_per_min: number;
  active_sessions: number;
}

/** Watts by part; the whole object is null while the sampler is not running. */
export interface Power {
  cpu_w: number;
  gpu_w: number;
  ane_w: number;
  sys_w: number;
}

/** `ok` sampling, `off` (`--machine-stats off`), `failed` (an error after startup stopped it). */
export type SamplerState = 'ok' | 'off' | 'failed';

/** Every field is null when its probe is unavailable. */
export interface MachineStatus {
  sampler: SamplerState;
  /** CPU load, 0..1. */
  cpu_load: number | null;
  /** GPU active ratio, 0..1. */
  gpu_active: number | null;
  gpu_freq_mhz: number | null;
  memory_pressure: MemoryPressure | null;
  ram_total_bytes: number | null;
  ram_used_bytes: number | null;
  /** lily's own resident memory (`stats::task_memory`). */
  lily_resident_bytes: number | null;
  swap_used_bytes: number | null;
  power: Power | null;
  temp_c: { cpu: number; gpu: number } | null;
}

export interface GpuTier {
  used_bytes: number;
  /** `--cache-bytes`, or derived at the load; null until loaded unless `--cache-bytes` set it. */
  budget_bytes: number | null;
  entries: number;
  /** `--max-sessions`. */
  max_entries: number;
}

export interface DiskTier {
  /** All entries, durable ones included. */
  used_bytes: number;
  /** `--disk-cache-bytes`. */
  budget_bytes: number;
  entries: number;
  durable_entries: number;
  durable_bytes: number;
}

export interface ExpertTier {
  slots: number;
  /** Slots holding an expert; the load fills every slot, so this equals `slots`. */
  used: number;
  /** 0..1, cumulative since the load; null before the first lookup. */
  hit_rate: number | null;
}

export interface NgramTier {
  /** Size of the n-gram table. */
  bytes: number;
  /** Share of the table currently in the page cache, 0..1; 1 for a table resident in GPU memory. */
  resident_share: number;
}

export interface WeightsTier {
  bytes: number;
  /** Weights are mlocked right now. */
  pinned: boolean;
}

export interface Tiers {
  gpu: GpuTier;
  /** null when the disk tier is off (disk budget 0). */
  disk: DiskTier | null;
  /** Present only in small-machine mode. */
  experts: ExpertTier | null;
  /** null while no model is loaded, or the model has no n-gram table. */
  ngram: NgramTier | null;
  weights: WeightsTier;
}

/**
 * `off` (`--stats-db off`), `opening` (the recorder's thread is opening the
 * database, retrying after a failure), `open`, `failed` (it will not open,
 * e.g. a newer lily's schema, or its thread ended).
 */
export type RecorderState = 'off' | 'opening' | 'open' | 'failed';

export interface RecorderStatus {
  state: RecorderState;
  /** Open and no kind of write failing. */
  ok: boolean;
  /** Request rows dropped since start: a full channel, a batch that failed twice, a row SQLite refused. */
  dropped_rows: number;
  db_bytes: number;
  /** Unix ms of the oldest stored sample; null when the database is empty or off. */
  oldest: number | null;
  /** Why it is `opening` or `failed`, as the server log says it; null otherwise. */
  error: string | null;
}

export interface Status {
  /** The server's clock, unix ms. The page compares it with its own to flag a stale answer. */
  now: number;
  state: LifecycleState;
  model: string;
  uptime_s: number;
  /** Seconds until the pin is released; null while busy or unpinned. */
  pin_hold_s: number | null;
  /** Idle-unload delay in seconds; null = never unloads. */
  idle_unload_s: number | null;
  idle_for_s: number;
  queue: QueueStatus;
  live: LiveStatus;
  machine: MachineStatus;
  tiers: Tiers;
  recorder: RecorderStatus;
}

// ---------------------------------------------------------------------------
// GET /v1/lily/history?range=  and  GET /v1/lily/sessions/{lineage}/history?range=

/**
 * One bucket. Throughput averages are tokens over *busy* time in the bucket,
 * maxima are the best 1-minute bucket within it (docs/dashboard.md § 5); both
 * are null for an idle bucket, which the charts draw as a gap.
 *
 * A bucket with no data at all (the server was off, the recorder had nothing)
 * has every value field null, `busy_share` 0 and every count 0.
 */
export interface HistoryPoint {
  /**
   * Bucket end, unix ms. The data runs to the last complete minute, so the
   * last bucket of 24h and 7d is usually partial: its `t` is its nominal end
   * (later than now) and its `busy_share` is over the minutes it covers.
   */
  t: number;
  /** Share of the bucket the engine was busy, 0..1 (of the covered part, for a partial last bucket). */
  busy_share: number;
  requests: number;
  decode_tps_avg: number | null;
  decode_tps_max: number | null;
  prefill_tps_avg: number | null;
  prefill_tps_max: number | null;
  decode_tokens: number;
  prefill_tokens: number;
  /** MTP acceptance ratio 0..1. */
  acceptance: number | null;
  /** Cached / prompt tokens, 0..1. */
  cache_hit: number | null;
  queue_ms_p50: number | null;
  queue_ms_max: number | null;
  /** Maximum concurrent sessions in the bucket. */
  sessions_max: number;
  /* Machine series. They describe the machine, so the per-session history
     carries null in all of them. */
  cpu_load: number | null;
  gpu_active: number | null;
  /** Worst memory pressure in the bucket. */
  memory_pressure: MemoryPressure | null;
  swap_bytes: number | null;
  /** System power, watts. */
  power_w: number | null;
  gpu_power_w: number | null;
  gpu_freq_mhz: number | null;
}

export interface Rate {
  /** Tokens over busy time; null if nothing ran in the interval. */
  avg: number | null;
  /** The best 1-minute bucket in the interval; null if nothing ran. */
  max: number | null;
}

export interface HistorySummary {
  decode: Rate;
  prefill: Rate;
  tokens_generated: number;
  tokens_prefilled: number;
  requests: number;
  /** Busy time over wall time, 0..1. */
  busy_share: number;
  cache_hit: number | null;
  acceptance: number | null;
  /** Mean system power over the interval; null when unavailable. */
  power_avg_w: number | null;
  /** Energy over the interval, watt-hours; null when unavailable. */
  energy_wh: number | null;
}

export interface History {
  range: Range;
  /** Bucket width in seconds. */
  step_s: number;
  /**
   * The complete contiguous grid from now minus the range to now at `step_s`,
   * oldest first, the last bucket holding the last complete minute (and
   * possibly partial, see `HistoryPoint.t`). Every bucket is present: the
   * page places a bucket by its index, never by its timestamp.
   */
  points: HistoryPoint[];
  summary: HistorySummary;
}

// ---------------------------------------------------------------------------
// GET /v1/lily/sessions

export type SessionTier = 'gpu' | 'disk';

/**
 * One cache entry, identified by its lineage id (docs/dashboard.md § 4.2).
 * The per-session figures (`requests`, the rates, `cache_hit`, `acceptance`)
 * cover the recorder's 7 days; without the recorder (`--stats-db off`) they
 * cover the server's last 1024 requests.
 */
export interface Session {
  /** null for an entry written before lineage ids existed; it has no history. */
  lineage: number | null;
  tier: SessionTier;
  durable: boolean;
  /** A request of this lineage is running now. */
  active: boolean;
  tokens: number;
  bytes: number;
  /** Last use, unix ms. */
  last_used_t: number;
  requests: number;
  decode_tps_avg: number | null;
  decode_tps_max: number | null;
  prefill_tps_avg: number | null;
  cache_hit: number | null;
  acceptance: number | null;
  /** Creation time, unix ms; null for an entry written before lineage ids existed. */
  created_t: number | null;
}

export interface SessionsResponse {
  sessions: Session[];
}

// ---------------------------------------------------------------------------
// GET /v1/lily/requests?limit=   (newest first)

/** What stopped a request before it finished (`Timings.cancelled_by`). */
export type CancelledBy = 'client' | 'shutdown';

export interface RequestRow {
  /** Finish time, unix ms. */
  t: number;
  /** null for an entry without a lineage id. */
  lineage: number | null;
  prompt_tokens: number;
  cached_tokens: number;
  /** Tokens computed (prompt minus cached). */
  prefill_tokens: number;
  prefill_ms: number;
  /** null when no tokens were prefilled. */
  prefill_tps: number | null;
  generated_tokens: number;
  decode_ms: number;
  /** null when nothing was generated. */
  decode_tps: number | null;
  /** Draft tokens proposed; null when the draft head is off. */
  drafted_tokens: number | null;
  /** Draft tokens accepted; null when the draft head is off. */
  accepted_tokens: number | null;
  queue_ms: number;
  /** Batch size while the request ran. */
  batch: number;
  cancelled_by: CancelledBy | null;
  /** The thinking budget closed the reasoning block. */
  thinking_forced: boolean;
  pinned: boolean;
  /** The request failed (an error, a GPU fault); its counts are 0 and its rates null. */
  failed: boolean;
}

export interface RequestsResponse {
  requests: RequestRow[];
}

// ---------------------------------------------------------------------------
// /v1/lily/settings

export type SettingValue = number | string | boolean;
export type SettingType = 'number' | 'int' | 'enum' | 'string' | 'bool' | 'duration';
export type SettingSource = 'default' | 'flag' | 'database';

export interface Setting {
  /** snake_case setting name; the key in PATCH bodies and the DELETE path. */
  name: string;
  /** Section in the settings view, e.g. Sampling, Thinking, Lifecycle, Caches, Engine. */
  group: string;
  type: SettingType;
  /**
   * false = shown read-only: a restart is required, or (runtime settings
   * with `--stats-db off`) nothing could keep the change; `help` then says so.
   */
  editable: boolean;
  /** The command-line flag, e.g. `--top-p`. */
  flag: string;
  /** Built-in default; null where the flag has no value (unset). */
  default: SettingValue | null;
  /** The value the flag set; null when the flag was not given. */
  flag_value: SettingValue | null;
  /**
   * Effective value now; null where unset. A duration is the flag text
   * ("30m", "90", "1.5h"), exactly as `parse_duration_secs` accepts it.
   */
  value: SettingValue | null;
  /** Seconds, computed by the server, for `duration` settings; null otherwise. */
  value_s: number | null;
  /** Where `value` comes from: database override > flag > default. */
  source: SettingSource;
  /** Constraints for `number` and `int` settings; null otherwise. */
  min: number | null;
  max: number | null;
  step: number | null;
  /** Choices for `enum` settings; null otherwise. */
  options: string[] | null;
  /** One-line definition, shown as a tooltip. */
  help: string | null;
  /** When a change takes effect ("next request", "hold re-armed on change", ...); null when it cannot change. */
  applies: string | null;
}

export interface SettingsResponse {
  settings: Setting[];
}

/**
 * PATCH body: `{ "<name>": value, ... }`, applied atomically (all or none),
 * each name once (a name twice is a 400). A value is the flag's text (not
 * trimmed) or a JSON number or boolean standing for it, checked by the
 * flag's own parser (the 400 carries its message).
 */
export type SettingsPatch = Record<string, SettingValue>;

export interface SettingsUpdateResponse {
  /**
   * The settings that changed, as they are now (listing order). A setting
   * set to the value already in effect did not change and is left out (no
   * override is made, nothing is logged); the flag's value set over an
   * override keeps an override with that value (it pins it).
   */
  updated: Setting[];
}

/** DELETE /v1/lily/settings/{name}: back to the flag or default value; the database row is deleted by name even when no override was in effect (needs the database, 503 without it). */
export interface SettingResetResponse {
  setting: Setting;
}

/** One change: from the dashboard, or an override applied at startup. */
export interface SettingsLogEntry {
  /** Unix ms. */
  t: number;
  name: string;
  old: SettingValue | null;
  new: SettingValue | null;
  source: 'dashboard' | 'startup';
}

export interface SettingsLogResponse {
  /** Newest first; the recorder keeps 7 days. */
  entries: SettingsLogEntry[];
}
