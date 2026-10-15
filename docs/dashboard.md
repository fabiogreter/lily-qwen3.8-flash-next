# Dashboard

Status: **spec, agreed with the owner 2026-10-10; all five phases built**
(branch `dashboard`, not merged: § 10 lists what is still to verify before
merge). Prototypes are done (§ 9). Build order: the Vue app on mock data (§ 8), then lily serving it
and the read routes (§ 3), the recorder (§ 4), runtime settings (§ 7) and
machine sampling (§ 6). All five are in (`src/serve/dashboard.rs` says what
each route serves, `src/serve/recorder.rs` what the recorder stores and
how, `src/serve/machine.rs` how the machine is sampled); with `--stats-db
off` the per-session figures cover the server's last 1 024 requests and the
history routes answer 503.

A small web dashboard served by lily itself at `/`: what the server is doing
now, what it did over the last 7 days, what the machine pays for it, and the
settings that can change without a restart.

## 1. Goals and non-goals

Goals:

- Answer, at first look: **is it up and busy**, and **what does the machine
  pay** (load, memory pressure, later watts).
- Throughput overall and per session (cache entry), over 15 min, 1 h, 24 h
  and 7 d, averaged and max.
- Cache usage by tier.
- Show every setting; edit the runtime ones, persisted across restarts.

Non-goals (for now):

- Access from other devices. Loopback only, always (§ 3.2).
- Authentication. The guard in § 3.2 is against cross-site writes from a
  browser, not against local processes.
- Editing restart-only settings (`--max-seq`, `--kv-cache`, `--max-batch`,
  ...). They are shown read-only. § 7.4 keeps the door open for moving them
  from the launchd plist into the database later.
- Actions on the engine (evict a session, unload now, drop the disk tier). Not
  in the first version; revisit after the prototypes.
- A log tail.

## 2. Hard rules

1. **Numbers only.** Nothing stored or served by the dashboard contains prompt
   text, generated text, token ids, image data or tool arguments. Request
   rows hold counts, durations and rates. Agent traces are customer data; the
   dashboard must not become a place they leak into.
2. **The dashboard never wakes the engine.** No dashboard route loads the
   model, wakes the GPU, takes the activity assertion (`src/activity.rs`),
   counts as an arrival, or resets the idle-unload timer or the pin hold.
   Everything it shows comes from state the server already keeps or from the
   recorder (§ 4).
3. **Never on the engine thread's critical path.** The engine hands finished
   request numbers to the recorder through a bounded channel with `try_send`;
   a full channel drops the row and counts the drop (shown on the page). No
   SQLite call, no machine sampling and no JSON rendering on the engine
   thread.
4. **A broken probe never breaks serving.** SQLite unavailable, IOReport
   failing after a macOS update, a corrupt database: the dashboard shows the
   gap and the server serves on.

## 3. Server

### 3.1 Routes

All on lily's existing listener (`--bind`, default `127.0.0.1:8000`).

| Method | Path | What |
|---|---|---|
| GET | `/` | The Vue app's `index.html` |
| GET | `/assets/*` | The app's hashed JS, CSS, fonts (immutable cache headers) |
| GET | `/v1/lily/status` | Now: lifecycle state, queue, batch occupancy, sessions per tier, machine sample, recorder health |
| GET | `/v1/lily/history?range=15m\|1h\|24h\|7d` | Rolled-up time series (§ 5) |
| GET | `/v1/lily/sessions` | Cache entries: tier, tokens, last used, durable, lineage id, throughput summary |
| GET | `/v1/lily/sessions/{lineage}/history?range=...` | One entry's throughput series |
| GET | `/v1/lily/requests?limit=` | Recent request rows (numbers only) |
| GET | `/v1/lily/settings` | Every setting: value, source (flag, default, database), editable, constraints |
| PATCH | `/v1/lily/settings` | Change runtime settings (§ 7) |
| DELETE | `/v1/lily/settings/{name}` | Reset one setting to its flag or default value |
| GET | `/v1/lily/settings/log` | Change history |

The response shapes are defined once, in `dashboard/src/api/types.ts`: that
file is the contract the server implements (§ 3.4).

Existing routes (`/health`, `/v1/models`, `/v1/timings`, the completions)
stay as they are. An unknown GET or HEAD serves `index.html`, so client-side
routes survive a reload, only when it is a browser's navigation: a path with
no extension, not under `/api/` or `/v1/`, with an `Accept` that names
`text/html`. Every other unknown path (a client probing `/models`,
`/api/tags`, `/version`, `/metrics`; a file of the app that does not exist)
gets the server's own 404 (`{"error": {"message": "not found", "type":
"invalid_request_error"}}`), exactly as before the dashboard, and does not
count as the page being open.

### 3.2 Loopback only, and the write guard

- Every `/`, `/assets/*` and `/v1/lily/*` request checks the **peer address**
  and refuses anything not loopback with 403, whatever `--bind` says. The
  OpenAI routes keep their current behaviour.
- Writes (`PATCH`, `DELETE`) also require `Content-Type: application/json`
  (a cross-site page cannot send that without a CORS preflight, which lily
  never answers) and, when an `Origin` header is present, an origin equal to
  lily's own (`http://127.0.0.1:<port>` or `http://localhost:<port>`).
  Anything else is 403. About ten lines; the owner's own page never notices.

### 3.3 Serving the app

The built `dashboard/dist/` is committed and embedded with `include_bytes!`
through a generated manifest (path, content type, bytes), so `cargo build`
never needs node. The HTTP layer (`src/serve/http.rs`) gains a static
response with `Content-Type` and `Cache-Control`; `index.html` is
`no-cache`, hashed assets `immutable`.

### 3.4 Contract rules

`dashboard/src/api/types.ts` holds every shape; these rules hold across it:

- snake_case fields; timestamps unix ms; bytes, ms, tok/s, ratios 0..1.
- A value lily does not have is `null`, never 0 or an absent field: a rate
  of zero tokens, a missing machine sample, a tier that is off (disk budget
  0, the expert cache outside small-machine mode).
- Lists come wrapped (`{sessions: [...]}`), leaving room for totals later.
- History is the complete grid from `now - range` to `now` at `step_s`,
  oldest first; buckets with no data are all `null` (counts 0). Steps:
  15m and 1h at 60 s, 24h at 900 s, 7d at 3 600 s. Data runs to the last
  complete minute, so the last bucket may be partial: it carries its nominal
  end as `t` and its `busy_share` over the minutes it covers.
- `recorder.state` is `off` (`--stats-db off`), `opening` (the database is
  being opened, with retries; `error` says why it is not open yet), `open` or
  `failed` (`error` says why); the history routes answer 503 unless `open`.
- Memory pressure is lily's own names everywhere: `normal`, `warning`,
  `critical` or `null`; a history bucket carries the worst in it.
- Machine power is `{cpu_w, gpu_w, ane_w, sys_w, lily}` or `null`, with
  `machine.sampler` saying why (`ok`, `off`, `failed`); `lily` is lily's own
  estimated share, `{w, cpu_w, gpu_w}` or `null` (§ 6).
- Durations in settings are the flag text, parsed exactly as
  `parse_duration_secs` parses the flag, plus `value_s`.
- `PATCH /v1/lily/settings` takes several names, applied all or none, and
  returns `{updated: [...]}`; `DELETE` returns `{setting}`. Errors are
  `{"error": {"message", "setting"?}}` with 404 (unknown setting), 409
  (restart-only setting), 400 (invalid value), 403 (write guard), 413 (a
  write body over 64 KiB), 503 (no database, or it timed out before taking
  the change: nothing changed), 504 (the outcome is unknown: the change may
  still apply) or 500. After any 5xx on a write the page reloads the
  settings and the log. The guard matches the media type `application/json`,
  so a `charset` parameter passes. Setting the value already in effect is a
  no-op and is left out of `updated`.
- **API-only for now**: the server fills these fields and the app's types
  carry them, but no view shows them yet: `live.active_sessions`,
  `machine.power.ane_w`, `recorder.db_bytes`, `recorder.oldest`, the history's
  `queue_ms_p50`, `queue_ms_max` and `sessions_max`, and a setting's
  `value_s`. They stay in the contract (and its tests); a view may use them
  later without a server change.

## 4. Recorder and database

A **recorder thread** owns the database. It receives:

- one `RequestRow` per finished request from the engine (`try_send`, § 2.3),
- machine samples from the sampler thread (§ 6),
- tier snapshots it takes itself once a minute from shared counters (no
  engine involvement).

Storage: SQLite via `rusqlite` with the `bundled` feature (no system
library), WAL mode, one connection on the recorder thread. HTTP handlers read
through a second read-only connection.

Location: new flag `--stats-db <path>`, default
`~/Library/Application Support/lily/lily.sqlite` (assumption; the owner may
prefer next to `--disk-cache-dir`). `--stats-db off` disables the recorder
and the history views, nothing else.

As built (phase 3):

- The engine gains no hook: the board it already publishes request rows to
  offers each one to the recorder's channel (1 024 rows, `try_send`). The
  recorder samples the machine itself (CPU load, pressure, swap, RAM in use,
  lily's footprint) every 10 s, every 60 s while lily is quiet (§ 6, the
  idle wake-up budget), and snapshots the tiers from the published cache
  view once a minute.
- HTTP handlers open a read-only connection per answer with a 250 ms busy
  timeout (a lock held elsewhere answers 503 quickly); nothing they do
  holds a lock the engine takes.
- The recorder thread opens the database, never the main thread: a failed
  open is retried after 1 s, 2 s, 5 s, then every 30 s, the status saying
  `opening` and the error; rows wait meanwhile (counted if dropped).
- Schema version in `PRAGMA user_version`, migrations forward-only; v1 sets
  `auto_vacuum = INCREMENTAL` before its tables, the connection
  `journal_size_limit` 4 MiB. A file that is not a database or fails
  `quick_check` is renamed to `<path>.corrupt-<unix seconds>` (with its
  `-wal`, `-shm`) and a fresh one started. A newer schema is never renamed:
  the recorder is `failed` ("database <path> has schema N, this lily knows
  M: history off") and the file untouched.
- A batch of rows that fails is retried once with the next, then dropped
  and counted; a row SQLite refuses (a value it cannot hold) is skipped and
  counted. A SQLite error is logged once per kind of write and shows as
  `recorder.ok = false` until that write succeeds again; reads at the start
  (lineage ids, the oldest sample) are only logged.
- A request counts as at most 24 h long and never starting before the
  process; the clamped durations are what is stored. Accepted draft tokens
  are stored clamped to the drafted, and a failed request's queue time is
  NULL (left out of the queue figures).
- Once the database is open the lineage counter is raised above the largest
  id in `requests`, as it is above the disk tier's. Sessions created before
  that took clock-seeded ids, above every earlier process's unless the
  clock went back.
- Pruning hourly at the hour 7 d + 1 h back (the 7d range's oldest bucket
  stays whole), then `incremental_vacuum` and `wal_checkpoint(TRUNCATE)`.
  A machine or tier minute stored twice (the clock went back) keeps the
  latest.
- On shutdown `Recorder::finish` stops the thread, which writes what it
  holds and checkpoints the WAL; the wait is capped at 2 s.
- `--stats-db` takes an absolute path or `off` (no `~` expansion); a
  directory lily creates is `0700`, the database and its WAL files `0600`.
- The settings tables hold the runtime overrides and their log (§ 7.3); `settings_log` is pruned
  at 7 days with the rest, `settings` never.

### 4.1 Tables

| Table | Rows | Kept |
|---|---|---|
| `requests` | one per finished request: time, lineage id, prompt/cached/prefill/generated tokens, prefill/decode ms and rates, drafted/accepted, queue ms, batch size, cancelled, pinned, forced thinking close | 7 d |
| `minute` | one per minute: per-series sum, count, max, busy ms (§ 5) | 7 d |
| `minute_session` | one per minute and lineage id active in it: tokens, busy ms, max rate | 7 d |
| `machine` | one per minute: load, memory pressure, swap (min/mean/max); from the sampler: system power (min/mean/max), GPU power, activity and frequency, hottest temperatures, covered milliseconds (§ 6) | 7 d |
| `tiers` | one per minute: bytes and entries per cache tier | 7 d |
| `settings` | one per overridden setting: name, JSON value, changed at | until reset |
| `settings_log` | every change: time, name, old, new, source (`dashboard`, `startup`) | 7 d (open item, § 10) |

Pruning runs on the recorder thread once an hour. Size estimate: about 10k
rows per minute-table for 7 days, plus the request rows; a few MB.

### 4.2 Session identity (lineage id)

Today a resident session has no id, and a disk entry gets a fresh `s<n>` each
time it is written and is consumed by a hit (`src/serve/disk.rs`,
`src/serve/session.rs`). The dashboard adds a **lineage id**: a `u64` counter
given to a session when it is created, written into the disk entry's meta
file (new field, `#[serde(default)]`, so older meta files load as "unknown"),
and taken back when the session is restored from disk. A fork gets a new
lineage id. Durable prefix entries keep their own. Nothing more: no attempt to
follow a conversation across forks.

## 5. Metrics

Throughput, per series (decode, prefill, both directions summed):

- **Average over an interval** = tokens in the interval / **busy** time in the
  interval (time a request of that kind was running), not wall time: wall
  time would mostly measure idleness.
- **Busy time, as measured**: a request row gives its finish time and its
  prefill and decode durations; the decode is placed as the `decode_ms`
  ending at the finish, the prefill as the `prefill_ms` ending where the
  decode starts. A minute's decode busy time is the length of the union of
  the decode intervals overlapping it (two requests decoding together for
  10 s count 10 s, so the average is the batch's throughput), prefill busy
  the same over the prefill intervals of requests that computed tokens, and
  `busy_share` the union of whole requests over the bucket. A request's
  tokens are spread evenly over its interval, so a minute gets the share it
  overlapped. Request counts, cache hit, acceptance and queue times belong
  to the minute a request finished in.
- **Max over an interval** = the best 1-minute bucket in it, not the best
  single request: short requests give noisy peaks.
- Prefill rates are per computed token (as `Timings` already defines them):
  a cache hit does not inflate them.
- Per session: the same, restricted to the lineage id.

Also kept as series: requests per minute, MTP acceptance ratio, cache hit %
(cached / prompt tokens), queue ms (p50, max), concurrent sessions (max per
minute), batch occupancy.

Cache tiers (bytes used / budget, entries):

- GPU-resident sessions (`--cache-bytes`, `--max-sessions`)
- disk tier, split durable / evicted (`DiskStore::used_bytes`, `durable_len`)
- expert cache (`ExpertCacheStats`: slots, hit rate) when present
- n-gram table residency (page-cache share, from the gather stats)
- pinned weights (pin state, held until)

Machine: CPU load, memory pressure level (`stats::pressure_level`), swap
used, lily's own resident memory (`stats::task_memory`), and from the
sampler (§ 6) system, CPU, GPU and ANE power, the GPU's activity and
frequency and the temperatures. Energy and mean power cover only the time
the sampler measured (§ 6).

Ranges in the UI: 15 min, 1 h, 24 h, 7 d, in the buckets of § 3.4. Live
values: polled every 2 s, only while the page is visible; history, sessions
and requests every 60 s and on a range change.

## 6. Machine sampling (IOReport)

`macmon` 0.9.0 (MIT, 2026-10-07) is a Rust library over IOReport and SMC
that needs no root. With `default-features = false` its dependencies are
`core-foundation`, `libc`, `serde`, `serde_json`. It gives CPU/GPU/ANE/system
watts, CPU and GPU frequency and active ratio, temperatures, RAM and swap.

Plan:

1. **Probe first**: a small throwaway binary calling `Sampler::get_metrics`
   on this machine (M5 Max), checked against Activity Monitor and against the
   watts quoted in `src/activity.rs`. If it works, integrate in the first
   version; if not, ship load and pressure only and keep this as a follow-up.
2. A **sampler thread** owns the `Sampler` (its call blocks for the sample
   interval). Interval: 2 s while the engine is busy or the page is open, 30 s
   otherwise, so the 7-day history has idle samples too.
3. New flag `--machine-stats on|off` (default on). Any error after startup
   logs once, marks power "unavailable" in `/v1/lily/status`, and stops the
   thread; serving is unaffected.
4. Wrapped in one module (`src/serve/machine.rs`) so a replacement (own
   IOReport bindings modelled on macmon) touches nothing else.

As built (phase 5, `src/serve/machine.rs`; the probe's measurements were
taken outside the sandbox by the owner: `Sampler::new()` about 0.7 s,
`get_metrics(1000)` blocks the 1 000 ms, idle system 12 to 15 W, GPU 0.2 W at
338 MHz, temperatures about 48 °C):

- The thread (utility QoS) **creates the sampler itself**; `Source` is the
  whole of what the rest of lily asks of `macmon`. **lily does the waiting**:
  the first call asks `macmon` for a 1 s window (a baseline has to exist), every
  later one for none after lily waited on a condition variable until the next
  reading is due, so a window is exactly the time since the previous call. The
  wait is woken by the page opening (`Machine::touch`, only when the page was
  closed), by the engine starting work (`Board::opened` and, for a reload or
  a recovery, `Board::working`, through `Machine::nudge`: a fence and a
  relaxed load, the notify only on the idle-to-busy edge) and by the shutdown
  (`Machine::finish`, a 1 s cap). Nothing polls the engine in between.
- **Failures**: a call that errors, a value that is not finite or is out of
  range (power or system power above 1 000 W, frequency above 10 000 MHz,
  temperature outside -20 to 150 °C), or a first call back in under half its
  window is skipped and counted (logged on the first and every 100th, never in
  the contract); a good reading resets the count. **Five in a row**, a panic,
  or a sampler that cannot be created (IOReport is blocked inside a sandbox:
  "Failed to create subscription") logs once with the cause, marks
  `machine.sampler` `failed`, clears the reading and ends the thread.
  `--machine-stats off` starts none (`sampler: off`); `ok` means the thread
  runs, the figures can be `null` until the first fresh reading. GPU activity is
  clamped to 0..1 before it is stored.
- **What `catch_unwind` covers**: a Rust panic on the sampler thread, in lily's
  code or `macmon`'s Rust code, is caught and ends the sampler. It does not
  cover a fault that is not a panic: `macmon` calls IOReport and CoreFoundation
  through `unsafe` FFI without checking every pointer, and a NULL CoreFoundation
  object dereferenced there is a `SIGSEGV` that ends the whole lily process (an
  abort does too); a call that hangs leaves the thread stuck (the figures go
  `null` when stale, the shutdown does not wait past its cap). `macmon` is
  pinned (`=0.9.0`). Removing the risk takes a subprocess probe (a child
  process writing one JSON line per reading): `TODO(dashboard)` in
  `src/serve/machine.rs`, **owner decision pending**.
- **Busy** is `Board::busy`: a request between `opened` and `closed`, or
  tokens drawn or prefilled since the last look, or the engine loading,
  reloading or recovering. **The page is open** while the last request that
  passed the dashboard's loopback and `Host` guards (a page file or any
  `/v1/lily/*` route) is at most 10 s old; the status poll every 2 s keeps it
  so, and a hidden tab lets it lapse.
- **Freshness**: the status shows the latest reading while it is younger than
  three of the waits before it (6 s, 90 s after an idle wait, never less than
  3 s), and `null` otherwise or after a failure.
- **System power can be unknown**: `macmon` reads it from the SMC's `PSTR` and
  reports 0 when that fails, and never less than CPU + GPU + ANE; a reading whose
  system power is 0 or below that sum has none. `machine.power.sys_w` is then
  `null` (the parts are still shown; the UI shows "–"), and the recorder counts
  the window for the GPU series but not for system power or energy.
- **lily's own power** (`machine.power.lily`, "lily CPU+GPU (est.)" on the
  page) is an attribution, not a measurement: macOS has no per-process power
  meter. Per reading window (`serve::machine::attribute`):

  ```
  lily_cpu_w = Δ ri_energy_nj (proc_pid_rusage, RUSAGE_INFO_V6, own pid) / Δt
  lily_gpu_w = gpu_w × min(1, lily_gpu_busy / (gpu_active × Δt))
  lily_w     = lily_cpu_w + lily_gpu_w
  ```

  With the GPU (nearly, under 1 µs) idle by IOReport's account the share is
  1 if lily was busy at all and 0 otherwise. `ri_energy_nj` is the kernel's
  per-process energy estimate over both core clusters (`ri_penergy_nj` would
  be the performance cores only); `libc` stops at `rusage_info_v4`, so
  `stats.rs` declares the SDK's 464-byte `rusage_info_v6` itself.
  `lily_gpu_busy` comes from lily's own Metal commit feedback (`metal::busy`):
  each command buffer's `GPUStartTime`..`GPUEndTime`, merged per pass into
  their union when the pass's last feedback arrives (feedback arrives out of
  order), and that union added to a set of the intervals of the last second,
  so overlapping passes count once. **Limits**: the ANE (lily uses none), DRAM
  (large while decoding) and the rest of the SoC are not attributed; a span is
  wall time with stalls, so the share over-attributes when other processes'
  GPU work runs inside lily's spans; busy time is booked when a pass's
  feedback completes, so a long prefill pass can land in the 2 s window after
  the one it ran in (the clamp at 1 hides the overshoot, the next window
  shows the shortfall); a pass that never completes is not counted. The
  counters are re-read after every call to `macmon`, success, error or skip,
  because its baseline moves on every call; the first reading uses counters
  read just before the first call. A figure that cannot be right (a counter
  call failed, a counter went back, not finite, outside 0 to 1 000 W) is
  `null` and never skips the reading or counts toward the five failures. The
  cost on the feedback path (a vector per pass, a sort, a lock held for a
  merge on Metal's completion thread) is not measured. A cross-check on the
  hardware: `ioreg -r -c IOAccelerator -l` lists `accumulatedGPUTime` per
  process (not read by lily: a second FFI surface next to `macmon`).
- **Sources of the rest**: CPU load stays lily's tick-based figure
  (`stats::cpu_ticks`: it works without the sampler, measures the same span as
  the other live rates, and the history's 10 s samples use it); memory,
  swap and pressure stay lily's Mach and sysctl probes (they work with the
  sampler off, and are read at the status poll's cadence rather than the
  sampler's). From `macmon`: `cpu_power`, `gpu_power`, `ane_power`, `sys_power`,
  `gpu_active_ratio`, `gpu_freq_mhz` and the two average temperatures (`temp_c`
  needs both).
- **Recorder** (schema 2): every reading goes to a bounded queue the recorder
  drains on its 10 s tick (60 s while quiet). A reading carries its window, the monotonic time
  since the previous call placed by the wall clock at its end, so a clock that
  jumps moves a window and never stretches it. The recorder spreads it over the
  minutes it overlaps and merges each part into the minute's row (time-weighted
  means, least and most of the windows, hottest temperatures, `power_covered_ms`
  and `sys_covered_ms`), so parts arriving later, or from an earlier process, add
  up; a minute never holds more than 60 000 ms of coverage. **Energy integrates
  over covered time only**: the history's power series are weighted by the
  covered time (system power by `sys_covered_ms`, the GPU series by
  `power_covered_ms`), a bucket's `energy_wh` is the sum of mean system power
  times `sys_covered_ms`, `power_avg_w` the mean over the same time, and a minute
  without coverage adds nothing and is not filled in.
- **Idle wake-up budget** (also in [server.md](server.md), "Idle wake-ups"):

  | state | sampler | recorder |
  |---|---|---|
  | engine unloaded, no page within 10 s | every 30 s | every 60 s, and hourly to prune |
  | engine loaded and idle, no page | every 30 s | every 10 s |
  | page open | every 2 s | every 10 s |
  | request running, load, reload, recovery | every 2 s | every 10 s |

  The recorder's 60 s cadence needs all three: the engine unloaded after a
  load with no request open, no dashboard request within 10 s, the sampler
  not at its 2 s cadence. It blocks on its channel until the next sample, so
  rows and settings changes still arrive at once and a row that finds lily
  busy brings the next sample forward; the minute rows aggregate the samples a
  minute got, so the history stays complete. Work that starts during its 60 s
  wait is sampled when the wait ends, at most a minute later.
- **Resources**: a sampler thread at the utility QoS, a call every 2 s at most
  and every 30 s idle. `macmon` 0.9.0 holds IOHID and SMC handles for the
  sampler's life and takes a `mach_host_self` send right per call (a possible
  slow leak); the ignored real-hardware test measures lily's footprint over 300
  calls. Not vendored. Whether the sampler costs the engine anything is not
  measured yet: § 10.

## 7. Settings

### 7.1 Runtime-editable

| Setting | Flag | Applies |
|---|---|---|
| sampling defaults: temperature, top_k, top_p, min_p, presence, frequency, repetition penalty | `--temperature` ... | next request |
| default reasoning effort | `--reasoning-effort` | next request |
| thinking budget | `--thinking-budget` | next request |
| tool-turn factor | `--thinking-budget-tool-turn-factor` | next request |
| budget grace | `--thinking-budget-grace` | next request |
| thinking nudges | `--thinking-nudges` | next request |
| a tool call ends thinking | `--tool-call-ends-thinking` | next request |
| idle unload | `--idle-unload` | timer re-armed on change |
| pin hold | `--pin-hold` | hold re-armed on change (a shorter hold releases at once) |
| disk cache TTL | `--disk-cache-ttl` | next expiry pass |
| durable prefix threshold | `--durable-min-tokens` | next request |

Checked in the code (phase 4): `--durable-min-tokens` is read once per
request at admission, so it is runtime. `--queue` sizes the engine's
channel at startup and `--decode-checkpoint-tokens` is planned into the
session budget and the batch admission (and turned off by the expert
cache's plan), so both stay restart-only. Every other setting is shown
read-only with its value and where it came from.

"Timer re-armed on change": the new delay counts from the last request, as
the old one did (the dashboard never moves the timer, § 2.2); a delay
shorter than the time already idle unloads at once.

Changes never touch a running request.

### 7.2 Mechanism

Today these values are copied by value at startup into `Defaults`, the idle
timer, the pin and the disk store (`src/serve.rs`). They move into one shared
`RuntimeSettings` (an `ArcSwap`-like snapshot or a `Mutex` read at request
start; the cheapest that keeps reads off any lock the engine holds while
running). Each consumer reads it when it needs the value instead of keeping a
copy. Validation reuses the CLI parsers, so a value the flag would refuse is
refused here too (400 with the same message).

### 7.3 Persistence

Precedence at startup: database override > CLI flag > built-in default. The
startup log line says which settings came from the database. Each setting in
the UI shows its source and has "reset", which deletes the override and
returns to the flag or default value. Every change goes to `settings_log`.

As built (phase 4): the recorder thread applies the stored overrides once
the database is open (the flags are in effect until then; the disk tier runs
no expiry pass until then, or 30 s, so a longer stored TTL is never
undercut), and logs a `startup` entry for each override that differs from
the value it replaces. A change is written first and in effect after,
so the server and the database never disagree; without an open database
(`--stats-db off`, opening, failed) a write that would change something
answers 503 and changes nothing (`types.ts` has no "not persisted" answer,
and an in-memory change would silently vanish at the restart). With
`--stats-db off` the runtime settings are listed read-only. Durations are
capped at 365 days, for the flags too.

### 7.4 Later: all settings in the database

Not now. The design leaves room: the settings table already holds any
setting by name and JSON value, and `/v1/lily/settings` already describes
restart-only ones. Moving them means `lily-service.sh` would launch with only
`--model`, `--bind` and `--stats-db`, and a "save and restart" action would
be added. Decide after living with the runtime ones.

## 8. Frontend

- Vue 3 + Vite + TypeScript in `lily/dashboard/`, npm.
- `nodejs 24.21.0` added to `lily/.tool-versions` (owner approved
  2026-10-10).
- `dist/` committed; a check script (`npm run build` and `git diff --exit-code
  dashboard/dist`) catches a stale build.
- No charting library: hand-written SVG charts.
- Fonts vendored from `@fontsource` (Shippori Mincho, IBM Plex Sans; OFL),
  so the page works offline. `dist/assets/licenses.txt` carries the Vue
  (MIT) and font (OFL) notices into the binary.
- Light and dark, following the system, with a manual toggle; the paper
  texture (smooth, grain, charcoal; default charcoal) is a per-browser
  preference in the Settings view, stored in `localStorage`.
- Views: Overview (the first look), Sessions, Requests, Engine (the flow
  diagram), Settings.
- `npm run dev:mock` runs it on the deterministic mock in `src/mock/`, which
  the tests use too.

## 9. Design prototypes (done)

Outcome (owner, 2026-10-10): the calm structure in the sumi-e skin, with the
engine schematic as its own view; Shippori Mincho headings with IBM Plex Sans
for data; charcoal paper; no explanatory captions (definitions live in
tooltips). The prototype files were throwaway and are not in the repo. The
four directions explored:

Four standalone HTML files with mock data in `dashboard/prototypes/`, not
wired to lily. Each shows the Overview and Settings at least, in light and
dark. Shared mock data generator so all four show the same numbers.

Top of every prototype: **is it up and busy** (state, sessions, queue, live
tok/s) and **machine cost** (load, memory pressure, watts).

1. **Instrument panel**: Grafana-like operations look. Neutral dark grey
   panels, thin grid, green/amber/red thresholds, many small panels and
   sparklines, monospace numbers. Most information per screen.
2. **Calm overview**: Linear/Vercel product look. Crisp neutral greys, one
   accent, a few large tabular numbers, subtle cards; details one click down.
3. **Engine schematic**: a clean flow diagram of lily (queue, batch, GPU,
   tiers: GPU sessions, disk, expert cache, n-gram table) with live values on
   the nodes and flow width as tokens or bytes per second.
4. **Lily (sumi-e)**: ink and paper. A brushed ensō is the main gauge
   (throughput as how far the circle closes), a small indigo (ai) seal with a
   lily mark signs the page. Night paper in dark mode: warm charcoal with
   pale ink. Balanced precision: key numbers always visible, the brushwork
   carries state and trends. Three type variants switchable in the
   prototype: refined serif with tabular figures; Mincho-style serif for
   headings with sans for data; light humanist sans throughout.

## 10. Open items

- Sampler cost on decode throughput: an A/B bench (decode tok/s with
  `--machine-stats on` against `off`, the page open) needs the GPU and was
  deferred; until it runs, the sampler's cost to the engine is unmeasured.
- Owner decision: run `macmon` in a subprocess probe so a crash in its FFI
  cannot end the server (§ 6, `TODO(dashboard)` in `src/serve/machine.rs`).
- `settings_log` retention: 7 days like the rest, or kept forever (it is
  tiny)? Default 7 days.
- ~~Whether the background service may be stopped for testing against the
  real server (phase 2 on).~~ Owner: not during development; the end-to-end
  run against the real server is pending (below).
- ~~`--stats-db` default location (§ 4).~~ Resolved as built:
  `~/Library/Application Support/lily/lily.sqlite` (`recorder::default_path`).
- ~~macmon probe result on this machine (§ 6.1).~~ Phase 5: it works outside
  the sandbox; the real-hardware test is `serve::machine::tests::real_macmon_reads_plausible_figures`
  (`#[ignore]`, run with `-- --ignored`).
- ~~Whether `--queue`, `--durable-min-tokens`, `--decode-checkpoint-tokens`
  can change at runtime (§ 7.1).~~ Phase 4: `--durable-min-tokens` yes, the
  other two stay restart-only (§ 7.1).
- The drop counter's place on the page (the channel holds 1 024 rows).

**Before merge**: the verification still to do. Status 2026-10-12, after the
rebase on main 7ca696c: items 1 and 2 done (668 lib tests with Metal, the
`real_macmon` test; the service under launchd with the database at
`~/Library/Application Support/lily/` 0700/0600 and the sampler running); 4
done for dark and light; 5 done for the rebound `Host` (403), framing headers
and cross-site writes (missing JSON type or a foreign `Origin`: 403); a
spill on restart and the restore from disk keep lineage ids. Still open: 3,
6, 7, 8, the non-loopback peer of 5.

1. The Metal test suites (`serve::tests` against the GPU, the engine's) on
   the branch rebased on main.
2. A launchd run of the service with the dashboard on: the recorder's
   database created where § 4 says with its permissions, the sampler running.
3. Idle cost measured: wake-ups per thread against the budget in § 6 (engine
   unloaded, page closed) and the sampler's cost to decode (the A/B above).
4. A browser pass over every view, light and dark, against the real server.
5. The guards from a browser: a non-loopback peer, a rebound `Host`, a
   cross-site write, framing.
6. Robustness: a corrupt and a newer-schema database, a locked one, the
   sampler failing, `--stats-db off` and `--machine-stats off`.
7. A long run (a day or more): the database's size, pruning, the WAL, no
   leaked memory in the sampler.
8. A real opencode session through the dashboard-enabled server: rows,
   sessions and history as opencode's turns produce them, and unknown-path
   probes still answered 404.
