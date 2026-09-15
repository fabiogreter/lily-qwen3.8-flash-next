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
