/**
 * Turns lily's `timings` object into the one line the TUI slot shows.
 *
 * Pure, so it can be exercised without a terminal: `node --test format.test.ts`.
 */

/** One entry of lily's `GET /v1/timings`, and the `timings` object responses carry. */
export type Timings = {
  prompt_tokens: number
  cached_tokens: number
  prefill_tokens: number
  prefill_ms: number
  prefill_per_second: number | null
  generated_tokens: number
  decode_ms: number
  decode_per_second: number | null
  drafted_tokens: number | null
  accepted_tokens: number | null
  acceptance_ratio: number | null
}

export type TimingsEntry = {
  id: string
  model: string
  created: number
  timings: Timings
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function isRate(value: unknown): value is number | null {
  return value === null || isNumber(value)
}

/**
 * Whether a value really is lily's timings object. Anything else — another
 * server answering on the same port, an older lily — is ignored, so the
 * plugin shows nothing rather than `NaN`.
 */
export function isTimings(value: unknown): value is Timings {
  if (!value || typeof value !== "object") return false
  const t = value as Record<string, unknown>
  return (
    isNumber(t.prompt_tokens) &&
    isNumber(t.cached_tokens) &&
    isNumber(t.prefill_tokens) &&
    isNumber(t.prefill_ms) &&
    isNumber(t.generated_tokens) &&
    isNumber(t.decode_ms) &&
    isRate(t.prefill_per_second) &&
    isRate(t.decode_per_second)
  )
}

/** The newest entry of a `GET /v1/timings` body, or undefined if it is not one. */
export function newestEntry(body: unknown): TimingsEntry | undefined {
  if (!body || typeof body !== "object") return undefined
  const data = (body as { data?: unknown }).data
  if (!Array.isArray(data) || data.length === 0) return undefined
  const entry = data[0] as Record<string, unknown> | undefined
  if (!entry || typeof entry.id !== "string" || !isTimings(entry.timings)) return undefined
  return {
    id: entry.id,
    model: typeof entry.model === "string" ? entry.model : "",
    created: isNumber(entry.created) ? entry.created : 0,
    timings: entry.timings,
  }
}

/**
 * A plugin setting: the value from the plugin's options in `tui.json`, else
 * the environment variable, else the default. Blank strings do not count, so
 * an empty entry cannot silently point the plugin at nothing.
 */
export function setting(
  options: Record<string, unknown> | undefined,
  key: string,
  variable: string,
  env: Record<string, string | undefined>,
  fallback: string,
): string {
  const configured = options?.[key]
  if (typeof configured === "string" && configured.trim()) return configured.trim()
  const fromEnvironment = env[variable]
  if (typeof fromEnvironment === "string" && fromEnvironment.trim()) return fromEnvironment.trim()
  return fallback
}

/** `18`, `1.2k`, `24.5k`, `131k`. */
export function formatCount(tokens: number): string {
  if (!isNumber(tokens) || tokens < 0) return "?"
  if (tokens < 1000) return String(Math.round(tokens))
  const thousands = tokens / 1000
  return `${thousands < 100 ? thousands.toFixed(1) : Math.round(thousands)}k`
}

/** `101/s`, `1.2k/s`, or `—` when the server reported no rate. */
export function formatRate(rate: number | null | undefined): string {
  if (!isNumber(rate) || rate <= 0) return "—"
  if (rate >= 1000) return `${(rate / 1000).toFixed(1)}k/s`
  return `${rate >= 100 ? Math.round(rate) : rate.toFixed(1)}/s`
}

/**
 * The status line: what the prefill cost, what the decode cost, how much of
 * the prompt the session cache saved, and how the draft head did. The last
 * two parts are left out when they do not apply, so a request without a
 * cache hit or without speculative decoding stays short.
 */
export function formatTimings(timings: Timings): string {
  const parts = [
    `pf ${formatCount(timings.prefill_tokens)} ${formatRate(timings.prefill_per_second)}`,
    `dec ${formatCount(timings.generated_tokens)} ${formatRate(timings.decode_per_second)}`,
  ]
  if (timings.cached_tokens > 0 && timings.prompt_tokens > 0) {
    parts.push(`cached ${Math.round((100 * timings.cached_tokens) / timings.prompt_tokens)}%`)
  }
  if (isNumber(timings.acceptance_ratio)) {
    parts.push(`spec ${Math.round(100 * timings.acceptance_ratio)}%`)
  }
  return parts.join(" · ")
}

/**
 * What every request of this terminal UI's life added up to. Times are
 * milliseconds and tokens are tokens, so every rate derived from them is
 * token-weighted: a total divided by a total, never the mean of per-request
 * rates, which would let a 16-token answer outvote a 2 000-token one.
 *
 * `drafted_tokens` and `accepted_tokens` only ever take numbers from
 * requests that actually ran with the draft head, so a request without
 * speculation is absent from the ratio rather than a zero in it.
 */
export type Totals = {
  requests: number
  prompt_tokens: number
  cached_tokens: number
  prefill_tokens: number
  prefill_ms: number
  generated_tokens: number
  decode_ms: number
  drafted_tokens: number
  accepted_tokens: number
}

/** The running totals and the response ids already in them. */
export type Aggregate = {
  totals: Totals
  counted: ReadonlySet<string>
}

export function emptyTotals(): Totals {
  return {
    requests: 0,
    prompt_tokens: 0,
    cached_tokens: 0,
    prefill_tokens: 0,
    prefill_ms: 0,
    generated_tokens: 0,
    decode_ms: 0,
    drafted_tokens: 0,
    accepted_tokens: 0,
  }
}

export function emptyAggregate(): Aggregate {
  return { totals: emptyTotals(), counted: new Set() }
}

/**
 * Adds one request to the running totals, once. An id already counted
 * returns the aggregate unchanged (by identity), so a poll that hands back
 * an entry the plugin has already seen cannot inflate anything.
 */
export function accumulate(aggregate: Aggregate, entry: TimingsEntry): Aggregate {
  if (aggregate.counted.has(entry.id)) return aggregate
  const t = entry.timings
  const speculative = isNumber(t.drafted_tokens)
  const previous = aggregate.totals
  const counted = new Set(aggregate.counted)
  counted.add(entry.id)
  return {
    counted,
    totals: {
      requests: previous.requests + 1,
      prompt_tokens: previous.prompt_tokens + t.prompt_tokens,
      cached_tokens: previous.cached_tokens + t.cached_tokens,
      prefill_tokens: previous.prefill_tokens + t.prefill_tokens,
      prefill_ms: previous.prefill_ms + t.prefill_ms,
      generated_tokens: previous.generated_tokens + t.generated_tokens,
      decode_ms: previous.decode_ms + t.decode_ms,
      drafted_tokens: previous.drafted_tokens + (speculative ? t.drafted_tokens : 0),
      accepted_tokens: previous.accepted_tokens + (speculative && isNumber(t.accepted_tokens) ? t.accepted_tokens : 0),
    },
  }
}

/** What the totals say, each `null` when its denominator is zero. */
export type Derived = {
  prefillPerSecond: number | null
  decodePerSecond: number | null
  acceptanceRatio: number | null
  cachedShare: number | null
}

/** Tokens per second over a total duration, rounded like the server's own. */
function weightedRate(tokens: number, milliseconds: number): number | null {
  if (!(tokens > 0) || !(milliseconds > 0)) return null
  return Math.round((100 * 1000 * tokens) / milliseconds) / 100
}

function share(part: number, whole: number): number | null {
  if (!(whole > 0)) return null
  return Math.round((10000 * part) / whole) / 10000
}

/**
 * The aggregate figures, all token-weighted: total tokens over total time,
 * total accepted over total drafted, total cached over the total prompt.
 */
export function derive(totals: Totals): Derived {
  return {
    prefillPerSecond: weightedRate(totals.prefill_tokens, totals.prefill_ms),
    decodePerSecond: weightedRate(totals.generated_tokens, totals.decode_ms),
    acceptanceRatio: share(totals.accepted_tokens, totals.drafted_tokens),
    cachedShare: share(totals.cached_tokens, totals.prompt_tokens),
  }
}

function percent(ratio: number | null): string {
  return ratio === null ? "—" : `${Math.round(100 * ratio)}%`
}

/** `Σ 7 req · pf 1.1k/s · dec 98/s · cached 62% · spec 71%`. */
export function formatTotals(totals: Totals): string {
  const derived = derive(totals)
  const parts = [
    `Σ ${formatCount(totals.requests)} req`,
    `pf ${formatRate(derived.prefillPerSecond)}`,
    `dec ${formatRate(derived.decodePerSecond)}`,
    `cached ${percent(derived.cachedShare)}`,
  ]
  if (derived.acceptanceRatio !== null) parts.push(`spec ${percent(derived.acceptanceRatio)}`)
  return parts.join(" · ")
}

/**
 * The whole line: this request on the left, the session's totals on the
 * right, at the widest form that fits `columns`.
 *
 * Width is taken from the aggregate first — it is the half a reader can
 * reconstruct later from the server log, and the point of the slot is the
 * request that just ran. The last rung is the latest request alone, so it
 * survives any width; if even that does not fit, it is returned anyway
 * rather than replaced by nothing.
 */
export function formatLine(latest: Timings, totals: Totals, columns: number): string {
  const derived = derive(totals)
  const now = `now ${formatTimings(latest)}`
  const rungs = [
    `${now}  ${formatTotals(totals)}`,
    `${now}  Σ ${formatCount(totals.requests)} · ${formatRate(derived.prefillPerSecond)} · ${formatRate(derived.decodePerSecond)} · cached ${percent(derived.cachedShare)}`,
    `${now}  Σ ${formatRate(derived.decodePerSecond)} · ${percent(derived.cachedShare)}`,
    now,
    `now pf ${formatRate(latest.prefill_per_second)} · dec ${formatRate(latest.decode_per_second)}`,
    `now ${formatRate(latest.decode_per_second)}`,
  ]
  const width = isNumber(columns) && columns > 0 ? columns : Number.POSITIVE_INFINITY
  return rungs.find((rung) => rung.length <= width) ?? rungs[rungs.length - 1]
}
