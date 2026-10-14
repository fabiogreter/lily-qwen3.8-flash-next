/**
 * opencode TUI plugin: lily's own prefill and decode numbers, in the
 * prompt footer's status row.
 *
 * The community throughput plugins estimate tokens from the characters they
 * see streaming past. This one asks the server, which counted them: after
 * every assistant step from the configured provider it reads
 * `GET /v1/timings` and shows what that request actually cost, plus the
 * running totals of every request this process has attributed to a session
 * turn. The totals live in memory for the life of the terminal UI and are
 * token-weighted, never an average of per-request rates.
 *
 * It reads the endpoint rather than the response's `timings` field because
 * nothing carries that field to a plugin: opencode's message and part
 * schemas have no free-form provider field for it, and the AI SDK's
 * openai-compatible provider drops response keys it does not know.
 *
 * Every failure is silent: another provider, no server, an older lily, a
 * body that is not lily's — the slot then renders nothing at all, because
 * this plugin is loaded in sessions that have nothing to do with lily.
 */
import { Plugin } from "@opencode/plugin/tui"
import type { CliRenderer } from "@opentui/core"
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
function terminalColumns(renderer: CliRenderer | undefined): number {
  const rendered = renderer?.width
  if (typeof rendered === "number" && rendered > 0) return rendered
  const columns = process.stdout?.columns
  return typeof columns === "number" && columns > 0 ? columns : FALLBACK_COLUMNS
}

export default Plugin.define({
  id: "lily-timings",
  setup(context) {
    const env = process.env
    const baseUrl = setting(context.options, "baseUrl", "LILY_TIMINGS_BASE_URL", env, DEFAULT_BASE_URL).replace(/\/+$/, "")
    const provider = setting(context.options, "provider", "LILY_TIMINGS_PROVIDER", env, DEFAULT_PROVIDER)
    const [readings, setReadings] = createSignal(new Map<string, Reading>())
    // One running total for the whole process, not per session: it answers
    // "how is this engine doing today", and `accumulate` keeps a response id
    // from being counted twice.
    const [aggregate, setAggregate] = createSignal<Aggregate>(emptyAggregate())
    const [columns, setColumns] = createSignal(terminalColumns(context.renderer))

    const onResize = () => setColumns(terminalColumns(context.renderer))
    process.stdout?.on?.("resize", onResize)

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

    // V2 has no `message.updated`; a step ends when one model request to the
    // step's assistant message finishes, which is exactly one lily request.
    const stopSteps = context.data.on("session.step.ended", (event) => {
      void (async () => {
        const { sessionID, assistantMessageID } = event.data
        let message = context.data.session.message.get(sessionID, assistantMessageID)
        if (!message) {
          try {
            await context.data.session.message.sync(sessionID)
          } catch {
            // Server restarting, session deleted mid-step: stay silent.
            return
          }
          message = context.data.session.message.get(sessionID, assistantMessageID)
        }
        // Unknown or someone else's provider: leave the slot to whatever it
        // showed.
        if (message?.type !== "assistant") return
        if (message.model?.providerID !== provider) return
        void update(sessionID)
      })()
    })

    const unregisterSlot = context.ui.slot({
      append: "prompt.footer.status",
      // The input is reactive and the footer outlives a session switch, so
      // `sessionID` is read inside the accessor, never destructured.
      render: (input) => {
        const reading = () => (input.sessionID ? readings().get(input.sessionID) : undefined)
        return (
          <Show when={reading()} fallback={<box flexShrink={0} />}>
            {(current) => (
              <box flexDirection="row" flexShrink={0}>
                <text fg={context.theme.text.muted}>
                  {formatLine(current().timings, aggregate().totals, Math.round(columns() * WIDTH_SHARE))}
                </text>
              </box>
            )}
          </Show>
        )
      },
    })

    return () => {
      stopSteps()
      unregisterSlot()
      process.stdout?.off?.("resize", onResize)
      setReadings(new Map())
      setAggregate(emptyAggregate())
    }
  },
})
