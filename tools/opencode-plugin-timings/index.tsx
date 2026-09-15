/**
 * opencode TUI plugin: lily's own prefill and decode numbers, next to the
 * session prompt.
 *
 * The community throughput plugins estimate tokens from the characters they
 * see streaming past. This one asks the server, which counted them: after
 * every assistant message from the configured provider it reads
 * `GET /v1/timings` and shows what that request actually cost, plus the
 * running totals of every request this process has attributed to a session
 * turn. The totals live in memory for the life of the terminal UI and are
 * token-weighted, never an average of per-request rates.
 *
 * It reads the endpoint rather than the response's `timings` field because
 * nothing carries that field to a plugin: opencode's message and part
 * schemas have no free-form provider field for it (`AssistantMessage` in
 * `@opencode-ai/sdk` is a closed shape), and the AI SDK's
 * openai-compatible provider drops response keys it does not know.
 *
 * Every failure is silent: another provider, no server, an older lily, a
 * body that is not lily's — the slot then renders nothing at all, because
 * this plugin is loaded in sessions that have nothing to do with lily.
 */
import type { PluginOptions } from "@opencode-ai/plugin"
import type { TuiPlugin, TuiPluginApi, TuiPluginModule } from "@opencode-ai/plugin/tui"
import { Show, createSignal } from "solid-js"

import { type Aggregate, type Timings, accumulate, emptyAggregate, formatLine, newestEntry, setting } from "./format.ts"

/** Where lily listens, unless configured otherwise. */
const DEFAULT_BASE_URL = "http://127.0.0.1:8000"
/** The opencode provider id whose messages the numbers belong to. */
const DEFAULT_PROVIDER = "lily"
/** A local server answers in a millisecond; this only bounds a hung socket. */
const FETCH_TIMEOUT_MS = 1500
/**
 * The engine records an entry before it writes the last bytes of the
 * response, so it is normally there already. One short retry covers the
 * ordering anyway.
 */
const RETRY_MS = 200
/**
 * The share of the terminal the line may use. The prompt, the agent and the
 * model name have the rest of the row, and the slot is never told its own
 * width, so the aggregate half gives way early rather than being clipped.
 */
const WIDTH_SHARE = 0.6
/** Assumed width when neither the renderer nor stdout says. */
const FALLBACK_COLUMNS = 120

type Reading = {
  /** The response id the numbers came from, so the same one is not shown twice. */
  id: string
  timings: Timings
}

/** The terminal's width, from the renderer if it knows, else from stdout. */
function terminalColumns(api: TuiPluginApi): number {
  const rendered = (api.renderer as unknown as { width?: number } | undefined)?.width
  if (typeof rendered === "number" && rendered > 0) return rendered
  const columns = process.stdout?.columns
  return typeof columns === "number" && columns > 0 ? columns : FALLBACK_COLUMNS
}

const tui: TuiPlugin = async (api: TuiPluginApi, options?: PluginOptions) => {
  const env = process.env
  const baseUrl = setting(options, "baseUrl", "LILY_TIMINGS_BASE_URL", env, DEFAULT_BASE_URL).replace(/\/+$/, "")
  const provider = setting(options, "provider", "LILY_TIMINGS_PROVIDER", env, DEFAULT_PROVIDER)
  const [readings, setReadings] = createSignal(new Map<string, Reading>())
  // One running total for the whole process, not per session: it answers
  // "how is this engine doing today", and `accumulate` keeps a response id
  // from being counted twice.
  const [aggregate, setAggregate] = createSignal<Aggregate>(emptyAggregate())
  const [columns, setColumns] = createSignal(terminalColumns(api))
  const cleanups: Array<() => void> = []

  const onResize = () => setColumns(terminalColumns(api))
  process.stdout?.on?.("resize", onResize)
  cleanups.push(() => {
    process.stdout?.off?.("resize", onResize)
  })

  async function fetchNewest() {
    try {
      const response = await fetch(`${baseUrl}/v1/timings`, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) })
      if (!response.ok) return undefined
      return newestEntry(await response.json())
    } catch {
      // No server, not lily, no such route, a timeout: show nothing.
      return undefined
    }
  }

  async function update(sessionID: string) {
    const previous = readings().get(sessionID)
    for (let attempt = 0; attempt < 2; attempt++) {
      const entry = await fetchNewest()
      if (entry && entry.id !== previous?.id) {
        // Only what this turn is attributed reaches the totals, so they
        // describe the session rather than everything that touched lily.
        setAggregate((current) => accumulate(current, entry))
        setReadings((current) => new Map(current).set(sessionID, { id: entry.id, timings: entry.timings }))
        return
      }
      if (attempt === 0) await new Promise((resolve) => setTimeout(resolve, RETRY_MS))
    }
  }

  cleanups.push(
    api.event.on("message.updated", (event) => {
      const info = event.properties.info
      if (!info || info.role !== "assistant") return
      // Someone else's provider: leave the slot to whatever it showed.
      if (info.providerID !== provider) return
      // Only a finished message has numbers to report.
      if (!info.time?.completed) return
      void update(event.properties.sessionID || info.sessionID)
    }),
  )

  api.slots.register({
    order: 60,
    slots: {
      session_prompt_right(_ctx, props) {
        const reading = () => readings().get(props.session_id)
        return (
          <Show when={reading()} fallback={<box flexShrink={0} />}>
            {(current) => (
              <box flexDirection="row" flexShrink={0}>
                <text fg={api.theme.current.textMuted}>
                  {formatLine(current().timings, aggregate().totals, Math.round(columns() * WIDTH_SHARE))}
                </text>
              </box>
            )}
          </Show>
        )
      },
    },
  })

  api.lifecycle.onDispose(() => {
    for (const cleanup of cleanups) cleanup()
    setReadings(new Map())
    setAggregate(emptyAggregate())
  })
}

const plugin: TuiPluginModule = { id: "lily-timings", tui }

export default plugin
