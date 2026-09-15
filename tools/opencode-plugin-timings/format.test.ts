import assert from "node:assert/strict"
import { test } from "node:test"

import {
  type Timings,
  type TimingsEntry,
  accumulate,
  derive,
  emptyAggregate,
  emptyTotals,
  formatCount,
  formatLine,
  formatRate,
  formatTimings,
  formatTotals,
  newestEntry,
  setting,
} from "./format.ts"

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

/** A timings object with everything zeroed but the fields a case is about. */
function some(fields: Partial<Timings>): Timings {
  return {
    prompt_tokens: 0,
    cached_tokens: 0,
    prefill_tokens: 0,
    prefill_ms: 0,
    prefill_per_second: null,
    generated_tokens: 0,
    decode_ms: 0,
    decode_per_second: null,
    drafted_tokens: null,
    accepted_tokens: null,
    acceptance_ratio: null,
    ...fields,
  }
}

function entry(id: string, fields: Partial<Timings>): TimingsEntry {
  return { id, model: "Qwen3.8-Flash-Next", created: 0, timings: some(fields) }
}

function totalsOf(...entries: TimingsEntry[]) {
  return entries.reduce(accumulate, emptyAggregate()).totals
}

test("aggregate rates are token-weighted, not the mean of per-request rates", () => {
  // 16 tokens at 93/s (172.043 ms) then 2 000 at 80/s (25 000 ms):
  // 2 016 tokens over 25.172043 s is 80.09/s. The mean of the two rates
  // would be 86.5, which a 16-token answer has no business producing.
  const totals = totalsOf(
    entry("a", { generated_tokens: 16, decode_ms: 172.043 }),
    entry("b", { generated_tokens: 2000, decode_ms: 25000 }),
  )
  assert.equal(totals.generated_tokens, 2016)
  assert.equal(totals.decode_ms, 25172.043)
  assert.equal(derive(totals).decodePerSecond, 80.09)
  assert.notEqual(derive(totals).decodePerSecond, 86.5)

  // The prefill side is weighted the same way: 24 000 tokens in 20 s and
  // 100 in 5 s is 24 100 / 25 s = 964/s, not the 620.5 the two rates average.
  const prefill = totalsOf(
    entry("c", { prefill_tokens: 24000, prefill_ms: 20000 }),
    entry("d", { prefill_tokens: 100, prefill_ms: 5000 }),
  )
  assert.equal(derive(prefill).prefillPerSecond, 964)
})

test("a response id is counted once however often it comes back", () => {
  const first = entry("chatcmpl-1", { prompt_tokens: 100, generated_tokens: 10, decode_ms: 100 })
  let aggregate = accumulate(emptyAggregate(), first)
  const again = accumulate(aggregate, first)
  assert.equal(again, aggregate, "a repeat returns the same aggregate, untouched")
  aggregate = accumulate(aggregate, entry("chatcmpl-2", { prompt_tokens: 50, generated_tokens: 5, decode_ms: 100 }))
  aggregate = accumulate(aggregate, first)
  assert.equal(aggregate.totals.requests, 2)
  assert.equal(aggregate.totals.prompt_tokens, 150)
  assert.equal(aggregate.totals.generated_tokens, 15)
})

test("every derived figure is null while its denominator is zero", () => {
  assert.deepEqual(derive(emptyTotals()), {
    prefillPerSecond: null,
    decodePerSecond: null,
    acceptanceRatio: null,
    cachedShare: null,
  })
  // A request whose decode took no measurable time cannot set a rate either.
  assert.equal(derive(totalsOf(entry("a", { generated_tokens: 4, decode_ms: 0 }))).decodePerSecond, null)
  assert.equal(formatTotals(emptyTotals()), "Σ 0 req · pf — · dec — · cached —")
})

test("requests without speculation stay out of the acceptance ratio", () => {
  // 8 of 10 accepted, then a request that ran without the draft head: the
  // ratio is still 80%, not 40% as it would be if its nulls counted as zeros.
  const totals = totalsOf(
    entry("a", { generated_tokens: 20, decode_ms: 1000, drafted_tokens: 10, accepted_tokens: 8 }),
    entry("b", { generated_tokens: 30, decode_ms: 1000 }),
  )
  assert.equal(totals.drafted_tokens, 10)
  assert.equal(totals.accepted_tokens, 8)
  assert.equal(derive(totals).acceptanceRatio, 0.8)
  // Its tokens and time still count towards the decode rate.
  assert.equal(derive(totals).decodePerSecond, 25)
  // No request used speculation: no percentage at all.
  const none = totalsOf(entry("c", { prompt_tokens: 100, generated_tokens: 5, decode_ms: 1000 }))
  assert.equal(derive(none).acceptanceRatio, null)
  assert.equal(formatTotals(none), "Σ 1 req · pf — · dec 5.0/s · cached 0%")
})

test("the cache share is over all the prompt tokens of the session", () => {
  const totals = totalsOf(
    entry("a", { prompt_tokens: 592, cached_tokens: 0 }),
    entry("b", { prompt_tokens: 34136, cached_tokens: 34114 }),
  )
  // 34 114 of 34 728 prompt tokens, kept to four decimals.
  assert.equal(derive(totals).cachedShare, 0.9823)
  assert.equal(formatTotals(totals).includes("cached 98%"), true)
})

test("a narrow terminal drops the aggregate first and keeps the latest request", () => {
  const latest = some({
    prompt_tokens: 21931,
    prefill_tokens: 21931,
    prefill_ms: 17969.741,
    prefill_per_second: 1220.44,
    generated_tokens: 16,
    decode_ms: 138.889,
    decode_per_second: 115.2,
    drafted_tokens: 10,
    accepted_tokens: 10,
    acceptance_ratio: 1,
  })
  const totals = totalsOf(
    entry("a", { prompt_tokens: 592, prefill_tokens: 592, prefill_ms: 3102.093, generated_tokens: 219, decode_ms: 10545.802, drafted_tokens: 230, accepted_tokens: 103 }),
    entry("b", { prompt_tokens: 34136, cached_tokens: 34114, prefill_tokens: 22, prefill_ms: 8630, generated_tokens: 124, decode_ms: 1700, drafted_tokens: 124, accepted_tokens: 61 }),
    { id: "c", model: "m", created: 3, timings: latest },
  )
  const at = (columns: number) => formatLine(latest, totals, columns)
  assert.equal(at(200), "now pf 21.9k 1.2k/s · dec 16 115/s · spec 100%  Σ 3 req · pf 759/s · dec 29.0/s · cached 60% · spec 48%")
  assert.equal(at(90), "now pf 21.9k 1.2k/s · dec 16 115/s · spec 100%  Σ 3 · 759/s · 29.0/s · cached 60%")
  assert.equal(at(70), "now pf 21.9k 1.2k/s · dec 16 115/s · spec 100%  Σ 29.0/s · 60%")
  assert.equal(at(60), "now pf 21.9k 1.2k/s · dec 16 115/s · spec 100%")
  assert.equal(at(44), "now pf 1.2k/s · dec 115/s")
  assert.equal(at(24), "now 115/s")
  // Narrower than anything on the ladder still shows the latest request.
  assert.equal(at(4), "now 115/s")
  // Every rung fits the width it is chosen for.
  for (const columns of [200, 90, 70, 60, 44, 24]) assert.ok(at(columns).length <= columns, `${columns}`)
})
