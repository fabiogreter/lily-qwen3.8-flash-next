import assert from "node:assert/strict"
import { test } from "node:test"

import { type Timings, formatCount, formatRate, formatTimings, newestEntry, setting } from "./format.ts"

const timings: Timings = {
  prompt_tokens: 24538,
  cached_tokens: 0,
  prefill_tokens: 24538,
  prefill_ms: 21180.0,
  prefill_per_second: 1158.55,
  generated_tokens: 19,
  decode_ms: 190.0,
  decode_per_second: 100.6,
  drafted_tokens: 14,
  accepted_tokens: 11,
  acceptance_ratio: 0.7857,
}

test("counts and rates stay short", () => {
  assert.equal(formatCount(18), "18")
  assert.equal(formatCount(1200), "1.2k")
  assert.equal(formatCount(131072), "131k")
  assert.equal(formatRate(1158.55), "1.2k/s")
  assert.equal(formatRate(100.6), "101/s")
  assert.equal(formatRate(3.36), "3.4/s")
  // The server reports no rate when nothing was computed.
  assert.equal(formatRate(null), "—")
  assert.equal(formatRate(0), "—")
})

test("the line carries prefill, decode, the cache share and acceptance", () => {
  assert.equal(formatTimings(timings), "pf 24.5k 1.2k/s · dec 19 101/s · spec 79%")
  assert.equal(
    formatTimings({ ...timings, cached_tokens: 24000, prefill_tokens: 538, prefill_per_second: 200 }),
    "pf 538 200/s · dec 19 101/s · cached 98% · spec 79%",
  )
  // Speculation off: no percentage that could read as a zero acceptance.
  assert.equal(
    formatTimings({ ...timings, drafted_tokens: null, accepted_tokens: null, acceptance_ratio: null }),
    "pf 24.5k 1.2k/s · dec 19 101/s",
  )
  // A full cache hit has no prefill rate to show.
  assert.equal(
    formatTimings({ ...timings, cached_tokens: 24538, prefill_tokens: 0, prefill_per_second: null }),
    "pf 0 — · dec 19 101/s · cached 100% · spec 79%",
  )
})

test("only lily's own body is accepted", () => {
  const entry = newestEntry({ data: [{ id: "chatcmpl-1-1", model: "m", created: 7, timings }] })
  assert.equal(entry?.id, "chatcmpl-1-1")
  assert.deepEqual(entry?.timings, timings)
  // Anything else on that port, or an empty log, shows nothing.
  assert.equal(newestEntry({ data: [] }), undefined)
  assert.equal(newestEntry({ object: "list" }), undefined)
  assert.equal(newestEntry({ data: [{ id: "x", timings: { tokens_per_second: 12 } }] }), undefined)
  assert.equal(newestEntry("not json at all"), undefined)
})

test("a setting comes from the options, then the environment, then the default", () => {
  const key = "baseUrl"
  const variable = "LILY_TIMINGS_BASE_URL"
  const fallback = "http://127.0.0.1:8000"
  assert.equal(setting(undefined, key, variable, {}, fallback), fallback)
  assert.equal(setting({}, key, variable, { [variable]: "http://box:9000" }, fallback), "http://box:9000")
  // The option wins over the environment, and both are trimmed.
  assert.equal(
    setting({ baseUrl: " http://box:1234 " }, key, variable, { [variable]: "http://box:9000" }, fallback),
    "http://box:1234",
  )
  // A blank or wrongly typed entry falls through instead of pointing at nothing.
  assert.equal(setting({ baseUrl: "   " }, key, variable, {}, fallback), fallback)
  assert.equal(setting({ baseUrl: 8000 }, key, variable, {}, fallback), fallback)
  assert.equal(setting({}, key, variable, { [variable]: "" }, fallback), fallback)
})
