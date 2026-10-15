// Mock data for the dashboard: pure, seeded generators typed against the API
// contract (src/api/types.ts). The numbers are invented and deterministic
// (given the clock); nothing here is a measurement. Sampling defaults in
// particular are placeholders: the real ones come from the checkpoint's
// generation_config.json.
//
// Time: everything is relative to a `now` the caller passes (the mock backend
// passes its clock). A history bucket's numbers depend only on its own time, so
// the grid slides with `now` without the curves changing under the reader; only
// a partial last bucket (24h, 7d) grows as its minutes pass, as on the server.

import type {
  CancelledBy,
  History,
  HistoryPoint,
  HistorySummary,
  Range,
  RequestRow,
  Session,
  Setting,
  SettingSource,
  SettingsLogEntry,
  SettingType,
  SettingValue,
  Status,
} from '../api/types';
import { parseDurationSecs } from '../lib/duration';

/** Deterministic PRNG (mulberry32). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const GiB = 1024 ** 3;
/** A fixed "now" for tests and as the backend's clock origin. */
export const NOW = Date.UTC(2026, 9, 10, 14, 20, 0);
export const DAY = 86400e3;

const round1 = (x: number): number => Math.round(x * 10) / 10;
const round3 = (x: number): number => Math.round(x * 1000) / 1000;

/** The random stream of one bucket: a function of the seed and the bucket's own index only. */
function bucketRng(seed: number, bucket: number): () => number {
  return rng(Math.imul(bucket ^ Math.imul(seed + 1, 0x9e3779b1), 0x85ebca6b) >>> 0);
}

/** Activity level 0..1 at a given time: workday hours busy, nights quiet, weekend lighter, with bursts. */
function activity(t: number, r: () => number): number {
  const d = new Date(t);
  const hour = d.getUTCHours() + d.getUTCMinutes() / 60;
  const weekday = d.getUTCDay();
  const weekend = weekday === 0 || weekday === 6;
  let base = 0;
  if (hour >= 7.5 && hour <= 19) base = 0.55 + 0.35 * Math.sin(((hour - 7.5) / 11.5) * Math.PI);
  else if (hour > 19 && hour < 23) base = 0.15;
  if (weekend) base *= 0.3;
  const burst = r() < 0.12 ? r() * 0.6 : 0;
  return Math.max(0, Math.min(1, base * (0.6 + 0.6 * r()) + burst));
}

/** Range, its span and its bucket width (contract: 15m 60 s, 1h 60 s, 24h 900 s, 7d 3600 s). */
export const RANGE_CFG: Record<Range, { span: number; step: number }> = {
  '15m': { span: 15 * 60e3, step: 60e3 },
  '1h': { span: 60 * 60e3, step: 60e3 },
  '24h': { span: DAY, step: 15 * 60e3 },
  '7d': { span: 7 * DAY, step: 60 * 60e3 },
};

/** Every third day the server is "off" for a few small hours: buckets with no data at all. */
function offline(t: number): boolean {
  const hour = new Date(t).getUTCHours();
  return Math.floor(t / DAY) % 3 === 0 && hour >= 2 && hour < 5;
}

function emptyPoint(t: number): HistoryPoint {
  return {
    t,
    busy_share: 0,
    requests: 0,
    decode_tps_avg: null,
    decode_tps_max: null,
    prefill_tps_avg: null,
    prefill_tps_max: null,
    decode_tokens: 0,
    prefill_tokens: 0,
    acceptance: null,
    cache_hit: null,
    queue_ms_p50: null,
    queue_ms_max: null,
    sessions_max: 0,
    cpu_load: null,
    gpu_active: null,
    memory_pressure: null,
    swap_bytes: null,
    power_w: null,
    gpu_power_w: null,
    gpu_freq_mhz: null,
  };
}

function point(t: number, seed: number, stepMs: number): HistoryPoint {
  if (offline(t)) return emptyPoint(t);
  const stepS = stepMs / 1000;
  const r = bucketRng(seed, Math.round(t / stepMs));
  const a = activity(t, r);
  const busy = a > 0.05;
  const sessions = busy ? Math.max(1, Math.round(a * 4 + r())) : 0;
  // Decode per session drops with batch size (drafting is solo only).
  const decodeAvg = busy
    ? (sessions === 1 ? 62 + r() * 18 : 34 + r() * 10) * (sessions > 1 ? 1 + 0.55 * (sessions - 1) : 1)
    : null;
  const prefillAvg = busy ? 1450 + r() * 900 : null;
  const share = busy ? Math.min(1, a * 1.1) : 0;
  return {
    t,
    busy_share: share,
    requests: busy ? Math.round(((a * 14 + r() * 3) * stepS) / 60) : 0,
    decode_tps_avg: decodeAvg && round1(decodeAvg),
    decode_tps_max: decodeAvg && round1(decodeAvg * (1.12 + r() * 0.25)),
    prefill_tps_avg: prefillAvg && Math.round(prefillAvg),
    prefill_tps_max: prefillAvg && Math.round(prefillAvg * (1.15 + r() * 0.4)),
    decode_tokens: decodeAvg ? Math.round(decodeAvg * share * stepS) : 0,
    prefill_tokens: prefillAvg ? Math.round(prefillAvg * share * 0.12 * stepS) : 0,
    acceptance: busy ? round3(0.7 + r() * 0.15) : null,
    cache_hit: busy ? round3(0.82 + r() * 0.15) : null,
    queue_ms_p50: busy ? Math.round(a * 40 * r()) : null,
    queue_ms_max: busy ? Math.round(a * 900 * r()) : null,
    sessions_max: sessions,
    cpu_load: round3(0.04 + a * 0.28 + r() * 0.05),
    gpu_active: round3(busy ? 0.25 + a * 0.7 : r() * 0.02),
    memory_pressure: a > 0.92 && r() < 0.3 ? 'warning' : 'normal',
    swap_bytes: Math.round((0.4 + a * 1.2) * GiB),
    power_w: round1(busy ? 18 + a * 62 + r() * 6 : 7 + r() * 3),
    gpu_power_w: round1(busy ? 8 + a * 44 + r() * 4 : 0.4 + r()),
    gpu_freq_mhz: busy ? Math.round(1240 + a * 380) : 340,
  };
}

type RateKey = 'decode_tps_avg' | 'decode_tps_max' | 'prefill_tps_avg' | 'prefill_tps_max' | 'acceptance' | 'cache_hit';

/**
 * The interval summary the real server computes (docs/dashboard.md § 5):
 * averages weighted by busy time, maxima the best bucket, busy time over the
 * wall time the grid covers. `lastCoveredMs` is how much of the last bucket
 * the data covers (all of it unless the bucket is partial).
 */
export function summarize(points: HistoryPoint[], lastCoveredMs?: number): HistorySummary {
  const busy = points.filter((p) => p.decode_tps_avg != null);
  const sum = (f: (p: HistoryPoint) => number): number => points.reduce((s, p) => s + f(p), 0);
  const max = (k: RateKey): number | null => (busy.length ? busy.reduce((m, p) => Math.max(m, p[k] ?? 0), 0) : null);
  const wavg = (k: RateKey, digits: 1 | 3): number | null => {
    const w = busy.reduce((s, p) => s + p.busy_share, 0);
    if (!w) return null;
    const v = busy.reduce((s, p) => s + (p[k] ?? 0) * p.busy_share, 0) / w;
    return digits === 1 ? round1(v) : round3(v);
  };
  const stepMs = points.length > 1 ? (points[1]?.t ?? 0) - (points[0]?.t ?? 0) : 0;
  // Each bucket's share of the wall time: the step, the last one what it covers.
  const covered = (i: number): number => (i === points.length - 1 && lastCoveredMs != null ? lastCoveredMs : stepMs);
  const wall = points.reduce((s, _, i) => s + covered(i), 0);
  const busyMs = points.reduce((s, p, i) => s + p.busy_share * covered(i), 0);
  const withPower = points.filter((p) => p.power_w != null);
  const powerSum = withPower.reduce((s, p) => s + (p.power_w ?? 0), 0);
  return {
    decode: { avg: wavg('decode_tps_avg', 1), max: max('decode_tps_max') },
    prefill: { avg: wavg('prefill_tps_avg', 1), max: max('prefill_tps_max') },
    tokens_generated: sum((p) => p.decode_tokens),
    tokens_prefilled: sum((p) => p.prefill_tokens),
    requests: sum((p) => p.requests),
    busy_share: wall > 0 ? round3(busyMs / wall) : points.length ? round3(sum((p) => p.busy_share) / points.length) : 0,
    cache_hit: wavg('cache_hit', 3),
    acceptance: wavg('acceptance', 3),
    power_avg_w: withPower.length ? round1(powerSum / withPower.length) : null,
    energy_wh: withPower.length ? round1((powerSum * stepMs) / 3600e3) : null,
  };
}

/**
 * The grid at `now` as the server lays it out (src/serve/dashboard/history.rs):
 * the data runs to the last complete minute (`dataEnd`), the last bucket is the
 * one that minute falls in, stamped with its nominal end (`end`, later than now
 * when the bucket is partial), and `lastCovered` is how much of it the data
 * covers.
 */
export function grid(range: Range, now: number): { end: number; dataEnd: number; lastCovered: number; buckets: number; step: number } {
  const { span, step } = RANGE_CFG[range];
  const dataEnd = Math.floor(now / 60e3) * 60e3;
  const end = Math.floor((dataEnd - 1) / step) * step + step;
  return { end, dataEnd, lastCovered: dataEnd - (end - step), buckets: span / step, step };
}

/** The complete grid of buckets (see `grid`), oldest first. */
export function buildHistory(range: Range, now: number, seed = 7): History {
  const { end, lastCovered, buckets: n, step } = grid(range, now);
  const points: HistoryPoint[] = [];
  for (let k = 0; k < n; k++) points.push(point(end - (n - 1 - k) * step, seed, step));
  // A partial last bucket holds the share of its counts its minutes saw; its
  // ratios (busy share, rates, machine means) are over those minutes.
  const last = points[n - 1];
  if (last && lastCovered < step) {
    const f = lastCovered / step;
    last.requests = Math.round(last.requests * f);
    last.decode_tokens = Math.round(last.decode_tokens * f);
    last.prefill_tokens = Math.round(last.prefill_tokens * f);
  }
  return { range, step_s: step / 1000, points, summary: summarize(points, lastCovered) };
}

// ---------------------------------------------------------------------------
// Cache entries (lineage ids). No content, only numbers.

export function buildSessions(now: number): Session[] {
  const r = rng(42);
  const tiers = ['gpu', 'gpu', 'gpu', 'gpu', 'disk', 'disk', 'disk', 'disk', 'disk', 'disk', 'disk'] as const;
  return tiers.map((tier, i) => {
    const durable = tier === 'disk' && i >= 9;
    const tokens = durable ? Math.round(9000 + r() * 4000) : Math.round(4000 + r() * 90000);
    // KV + recurrent state per token at q8_0; invented.
    const bytes = Math.round(tokens * (tier === 'gpu' ? 38 : 36) * 1024 + 0.6 * GiB);
    const active = tier === 'gpu' && i < 2;
    const lineage = 1180 + i * 7 + Math.round(r() * 5);
    const lastUsedS = active ? Math.round(r() * 5) : Math.round((i + 1) * (300 + r() * 4000));
    const requests = durable ? Math.round(30 + r() * 80) : Math.round(2 + r() * 60);
    return {
      // One entry predates lineage ids: no id, so no history.
      lineage: i === 8 ? null : lineage,
      tier,
      durable,
      active,
      tokens,
      bytes,
      last_used_t: now - lastUsedS * 1000,
      requests,
      decode_tps_avg: round1(48 + r() * 30),
      decode_tps_max: round1(78 + r() * 22),
      prefill_tps_avg: Math.round(1300 + r() * 900),
      cache_hit: round3(0.75 + r() * 0.24),
      acceptance: round3(0.68 + r() * 0.18),
      created_t: now - Math.round((i + 1) * 3600 * (1 + r() * 6)) * 1000,
    };
  });
}

/**
 * One entry's history: the overall shape, thinned to the stretches the entry
 * was active (runs of buckets picked by time, seeded by lineage) and scaled so
 * its averages match the entry's row in the sessions list. Machine fields are
 * null (they describe the machine, not the entry). null for an unknown lineage.
 */
export function buildSessionHistory(sessions: Session[], lineage: number, range: Range, now: number): History | null {
  const entry = sessions.find((x) => x.lineage === lineage);
  if (!entry) return null;
  const h = buildHistory(range, now, lineage);
  const stepMs = h.step_s * 1000;
  for (const p of h.points) {
    const run = bucketRng(lineage * 31 + 5, Math.floor(p.t / stepMs / 6));
    const on = run() < 0.45;
    if (!on) Object.assign(p, emptyPoint(p.t));
    p.cpu_load = null;
    p.gpu_active = null;
    p.memory_pressure = null;
    p.swap_bytes = null;
    p.power_w = null;
    p.gpu_power_w = null;
    p.gpu_freq_mhz = null;
  }
  const covered = grid(range, now).lastCovered;
  const base = summarize(h.points, covered);
  const kd = base.decode.avg && entry.decode_tps_avg != null ? entry.decode_tps_avg / base.decode.avg : 1;
  const kp = base.prefill.avg && entry.prefill_tps_avg != null ? entry.prefill_tps_avg / base.prefill.avg : 1;
  for (const p of h.points) {
    if (p.decode_tps_avg == null) continue;
    p.decode_tps_avg = round1(p.decode_tps_avg * kd);
    p.decode_tps_max = round1((p.decode_tps_max ?? 0) * kd);
    p.prefill_tps_avg = Math.round((p.prefill_tps_avg ?? 0) * kp);
    p.prefill_tps_max = Math.round((p.prefill_tps_max ?? 0) * kp);
    p.decode_tokens = Math.round(p.decode_tokens * kd);
    p.prefill_tokens = Math.round(p.prefill_tokens * kp);
  }
  h.summary = summarize(h.points, covered);
  return h;
}

export function buildRequests(sessions: Session[], now: number): RequestRow[] {
  const r = rng(99);
  const rows: RequestRow[] = [];
  let t = now - 2e3;
  for (let i = 0; i < 40; i++) {
    const s = sessions[Math.floor(r() * 6)] as Session;
    const prompt = Math.round(s.tokens * (0.9 + r() * 0.1));
    const cold = r() < 0.12;
    const full = !cold && r() < 0.04; // everything came from the cache
    const cached = full ? prompt : cold ? Math.round(prompt * r() * 0.3) : Math.round(prompt - 200 - r() * 4000);
    const prefill = prompt - cached;
    const prefillTps = Math.round(1200 + r() * 1100 - (cold ? 300 : 0));
    const generated = Math.round(40 + r() * (r() < 0.2 ? 6000 : 900));
    const decodeTps = round1(40 + r() * 40);
    const draftOff = r() < 0.08;
    const drafted = Math.round(generated * 1.3);
    const cancelled: CancelledBy | null = r() < 0.05 ? (r() < 0.8 ? 'client' : 'shutdown') : null;
    rows.push({
      t,
      lineage: s.lineage,
      prompt_tokens: prompt,
      cached_tokens: cached,
      prefill_tokens: prefill,
      prefill_ms: prefill ? Math.round((prefill / prefillTps) * 1000) : 0,
      prefill_tps: prefill ? prefillTps : null,
      generated_tokens: generated,
      decode_ms: Math.round((generated / decodeTps) * 1000),
      decode_tps: decodeTps,
      drafted_tokens: draftOff ? null : drafted,
      accepted_tokens: draftOff ? null : Math.round(drafted * (0.65 + r() * 0.2)),
      queue_ms: r() < 0.25 ? Math.round(r() * 1400) : Math.round(r() * 8),
      batch: 1 + Math.floor(r() * r() * 4),
      cancelled_by: cancelled,
      thinking_forced: r() < 0.08,
      pinned: true,
      // One failed request in the list, without drawing from the stream.
      failed: i === 17,
    });
    t -= Math.round(4e3 + r() * 90e3);
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Settings. Restart-only ones are editable: false.

interface SettingOpts {
  flag: string;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  help?: string;
  /** When a change takes effect. */
  applies: string;
  value?: SettingValue;
  source?: SettingSource;
}

const NEXT_REQUEST = 'next request';

function s(name: string, group: string, type: SettingType, def: SettingValue, o: SettingOpts): Setting {
  const value = o.value !== undefined ? o.value : def;
  return {
    name,
    group,
    type,
    editable: true,
    flag: o.flag,
    default: def,
    flag_value: o.source === 'flag' && o.value !== undefined ? o.value : null,
    value,
    value_s: type === 'duration' ? parseDurationSecs(String(value)) : null,
    source: o.source ?? 'default',
    min: o.min ?? null,
    max: o.max ?? null,
    step: o.step ?? null,
    options: o.options ?? null,
    help: o.help ?? null,
    applies: o.applies,
  };
}

function ro(name: string, group: string, value: SettingValue, flag: string, source: SettingSource): Setting {
  return {
    name,
    group,
    type: typeof value === 'number' ? 'int' : 'string',
    editable: false,
    flag,
    default: value,
    flag_value: null,
    value,
    value_s: null,
    source,
    min: null,
    max: null,
    step: null,
    options: null,
    help: null,
    applies: null,
  };
}

export function buildSettings(): Setting[] {
  return [
    // Sampling (runtime). Values are placeholders for generation_config.json.
    s('temperature', 'Sampling', 'number', 1.0, { flag: '--temperature', min: 0, max: 2, step: 0.05, help: 'Default when a request sets none.', applies: NEXT_REQUEST }),
    s('top_p', 'Sampling', 'number', 0.95, { flag: '--top-p', min: 0, max: 1, step: 0.01, applies: NEXT_REQUEST }),
    s('top_k', 'Sampling', 'int', 20, { flag: '--top-k', min: 0, max: 200, applies: NEXT_REQUEST }),
    s('min_p', 'Sampling', 'number', 0.0, { flag: '--min-p', min: 0, max: 1, step: 0.01, applies: NEXT_REQUEST }),
    s('presence_penalty', 'Sampling', 'number', 0.0, { flag: '--presence-penalty', min: -2, max: 2, step: 0.1, applies: NEXT_REQUEST }),
    s('frequency_penalty', 'Sampling', 'number', 0.0, { flag: '--frequency-penalty', min: -2, max: 2, step: 0.1, applies: NEXT_REQUEST }),
    s('repetition_penalty', 'Sampling', 'number', 1.0, { flag: '--repetition-penalty', min: 0.5, max: 2, step: 0.01, value: 1.05, source: 'database', help: 'Anti-loop measure for agent use.', applies: NEXT_REQUEST }),
    // Thinking (runtime).
    s('reasoning_effort', 'Thinking', 'enum', 'xhigh', { flag: '--reasoning-effort', options: ['low', 'medium', 'xhigh'], value: 'low', source: 'flag', applies: NEXT_REQUEST }),
    s('thinking_budget', 'Thinking', 'string', 'low=8000,medium=8000,xhigh=16000', { flag: '--thinking-budget', help: 'Tokens by reasoning effort, or one number for all.', applies: NEXT_REQUEST }),
    s('thinking_budget_tool_turn_factor', 'Thinking', 'number', 1.0, { flag: '--thinking-budget-tool-turn-factor', min: 0.05, max: 4, step: 0.05, applies: NEXT_REQUEST }),
    s('thinking_budget_grace', 'Thinking', 'int', 128, { flag: '--thinking-budget-grace', min: 0, max: 2048, applies: NEXT_REQUEST }),
    s('thinking_nudges', 'Thinking', 'bool', true, { flag: '--thinking-nudges', applies: NEXT_REQUEST }),
    s('tool_call_ends_thinking', 'Thinking', 'bool', true, { flag: '--tool-call-ends-thinking', applies: NEXT_REQUEST }),
    // Lifecycle and caches (runtime).
    s('idle_unload', 'Lifecycle', 'duration', '0', { flag: '--idle-unload', value: '30m', source: 'flag', help: '0 never unloads.', applies: 'timer re-armed on change' }),
    s('pin_hold', 'Lifecycle', 'duration', '1m', { flag: '--pin-hold', applies: 'hold re-armed on change' }),
    s('disk_cache_ttl', 'Caches', 'duration', '3d', { flag: '--disk-cache-ttl', applies: 'next expiry pass' }),
    s('durable_min_tokens', 'Caches', 'int', 1024, { flag: '--durable-min-tokens', min: 0, help: '0 turns durable prefix entries off.', applies: NEXT_REQUEST }),
    // Restart-only (read-only in the dashboard).
    ro('model', 'Engine', '~/models/Qwen3.8-Flash-Next-lily-q4', '--model', 'flag'),
    ro('bind', 'Engine', '127.0.0.1:8000', '--bind', 'default'),
    ro('max_seq', 'Engine', 131072, '--max-seq', 'default'),
    ro('kv_cache', 'Engine', 'q8_0', '--kv-cache', 'default'),
    ro('max_batch', 'Engine', 4, '--max-batch', 'default'),
    ro('mtp_drafts', 'Engine', 2, '--mtp-drafts', 'default'),
    ro('queue', 'Engine', 32, '--queue', 'default'),
    ro('vision', 'Engine', 'auto', '--vision', 'default'),
    ro('stats_db', 'Engine', '~/Library/Application Support/lily/lily.sqlite', '--stats-db', 'default'),
    ro('machine_stats', 'Engine', 'on', '--machine-stats', 'default'),
    ro('cache_bytes', 'Caches', 'auto', '--cache-bytes', 'default'),
    ro('max_sessions', 'Caches', 16, '--max-sessions', 'default'),
    ro('disk_cache_bytes', 'Caches', '100G', '--disk-cache-bytes', 'default'),
    ro('decode_checkpoint_tokens', 'Caches', 2048, '--decode-checkpoint-tokens', 'default'),
    ro('ngram_table', 'Caches', 'paged', '--ngram-table', 'default'),
    ro('pin_weights', 'Lifecycle', 'auto', '--pin-weights', 'default'),
  ];
}

export function buildSettingsLog(now: number): SettingsLogEntry[] {
  return [
    { t: now - 2 * 3600e3, name: 'repetition_penalty', old: 1.0, new: 1.05, source: 'dashboard' },
    { t: now - 26 * 3600e3, name: 'thinking_nudges', old: false, new: true, source: 'dashboard' },
    { t: now - 3 * DAY, name: 'repetition_penalty', old: 1.1, new: 1.0, source: 'dashboard' },
    { t: now - 4 * DAY, name: 'idle_unload', old: '0', new: '30m', source: 'startup' },
  ];
}

// ---------------------------------------------------------------------------
// Live status. `tick` advances by one per poll; deterministic per tick. The
// mock alternates busy and idle: six busy polls, then three idle ones.

export function buildStatus(tick: number, sessions: Session[], now: number): Status {
  const r = rng(1000 + tick);
  const running = tick % 9 < 6 ? 1 + (tick % 3 === 0 ? 1 : 0) : 0;
  const busy = running > 0;
  const gpuSessions = sessions.filter((x) => x.tier === 'gpu');
  const disk = sessions.filter((x) => x.tier === 'disk');
  const durable = disk.filter((x) => x.durable);
  const sum = (xs: Session[]): number => xs.reduce((a, x) => a + x.bytes, 0);
  return {
    now,
    state: busy ? 'busy' : 'ready',
    model: 'qwen3.8-flash-next',
    uptime_s: 4 * 86400 + 3 * 3600 + 1200 + tick * 2,
    pin_hold_s: busy ? null : 60 - ((tick * 2) % 60),
    idle_unload_s: 1800,
    idle_for_s: busy ? 0 : ((tick % 9) - 5) * 2,
    queue: { waiting: busy && r() < 0.2 ? 1 : 0, running, max_batch: 4, capacity: 32 },
    live: {
      decode_tps: busy ? round1(running === 1 ? 64 + r() * 14 : 92 + r() * 12) : 0,
      prefill_tps: busy && r() < 0.35 ? Math.round(1500 + r() * 800) : 0,
      acceptance: busy ? round3(0.72 + r() * 0.1) : null,
      requests_per_min: busy ? Math.round(6 + r() * 6) : 1,
      active_sessions: running,
    },
    machine: {
      sampler: 'ok',
      cpu_load: round3(busy ? 0.18 + r() * 0.1 : 0.04 + r() * 0.03),
      gpu_active: round3(busy ? 0.82 + r() * 0.15 : 0.01),
      gpu_freq_mhz: busy ? Math.round(1560 + r() * 60) : 338,
      memory_pressure: 'normal',
      ram_total_bytes: 128 * GiB,
      ram_used_bytes: Math.round(97.4 * GiB + r() * GiB),
      lily_resident_bytes: Math.round(71.2 * GiB),
      swap_used_bytes: Math.round(1.1 * GiB),
      power: {
        cpu_w: round1(busy ? 6 + r() * 3 : 1.2),
        gpu_w: round1(busy ? 44 + r() * 8 : 0.5),
        ane_w: 0,
        sys_w: round1(busy ? 68 + r() * 10 : 9.5),
        lily: busy ? { w: 47.6, cpu_w: 3.4, gpu_w: 44.2 } : { w: 0.3, cpu_w: 0.2, gpu_w: 0.1 },
      },
      temp_c: { cpu: Math.round(busy ? 64 + r() * 6 : 41), gpu: Math.round(busy ? 71 + r() * 5 : 39) },
    },
    tiers: {
      gpu: { used_bytes: sum(gpuSessions), budget_bytes: 24 * GiB, entries: gpuSessions.length, max_entries: 16 },
      disk: {
        used_bytes: sum(disk),
        budget_bytes: 100 * GiB,
        entries: disk.length,
        durable_entries: durable.length,
        durable_bytes: sum(durable),
      },
      experts: null,
      ngram: { bytes: Math.round(30.6 * GiB), resident_share: round3(0.86 + r() * 0.02) },
      weights: { bytes: Math.round(42.8 * GiB), pinned: true },
    },
    recorder: { state: 'open', ok: true, dropped_rows: 0, db_bytes: Math.round(6.3 * 1024 ** 2), oldest: now - 7 * DAY, error: null },
  };
}
