# Dashboard

Status: **spec, agreed with the owner 2026-10-10; being built.** Prototypes are
done (§ 9). Build order: the Vue app on mock data (§ 8), then lily serving it
and the read routes (§ 3), the recorder (§ 4), runtime settings (§ 7) and
machine sampling (§ 6). The first three are in (`src/serve/dashboard.rs`
says what each route serves, `src/serve/recorder.rs` what the recorder
stores and how); with `--stats-db off` the per-session figures cover the
server's last 1 024 requests and the history routes answer 503.

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
stay as they are. Unknown paths under `/` that are not `/v1/*` or
`/assets/*` serve `index.html`, so client-side routes survive a reload.

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
- Machine power is `{cpu_w, gpu_w, ane_w, sys_w}` or `null`, with
  `machine.sampler` saying why (`ok`, `off`, `failed`).
- Durations in settings are the flag text, parsed exactly as
  `parse_duration_secs` parses the flag, plus `value_s`.
- `PATCH /v1/lily/settings` takes several names, applied all or none, and
  returns `{updated: [...]}`; `DELETE` returns `{setting}`. Errors are
  `{"error": {"message", "setting"?}}` with 404 (unknown setting), 409
  (restart-only setting), 400 (invalid value) or 403 (write guard). The guard
  matches the media type `application/json`, so a `charset` parameter passes.

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
  recorder samples the machine itself every 10 s (CPU load, pressure, swap,
  RAM in use, lily's footprint) and snapshots the tiers from the published
  cache view once a minute.
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
- The settings tables exist (empty until phase 4); `settings_log` is pruned
  at 7 days with the rest, `settings` never.

### 4.1 Tables

| Table | Rows | Kept |
|---|---|---|
| `requests` | one per finished request: time, lineage id, prompt/cached/prefill/generated tokens, prefill/decode ms and rates, drafted/accepted, queue ms, batch size, cancelled, pinned, forced thinking close | 7 d |
| `minute` | one per minute: per-series sum, count, max, busy ms (§ 5) | 7 d |
| `minute_session` | one per minute and lineage id active in it: tokens, busy ms, max rate | 7 d |
| `machine` | one per minute: load, memory pressure, swap, later watts and GPU frequency (min/mean/max) | 7 d |
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

Machine (first version): CPU load, memory pressure level
(`stats::pressure_level`), swap used, lily's own resident memory
(`stats::task_memory`).

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
| pin hold | `--pin-hold` | next release |
| disk cache TTL | `--disk-cache-ttl` | next expiry pass |

To be checked in the code before they are added: `--queue`,
`--durable-min-tokens`, `--decode-checkpoint-tokens`. Every other setting is
shown read-only with its value and where it came from.

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

- `settings_log` retention: 7 days like the rest, or kept forever (it is
  tiny)? Default 7 days.
- Whether the background service may be stopped for testing against the
  real server (phase 2 on).
- `--stats-db` default location (§ 4).
- macmon probe result on this machine (§ 6.1).
- Whether `--queue`, `--durable-min-tokens`, `--decode-checkpoint-tokens` can
  change at runtime (§ 7.1).
- The drop counter's place on the page (the channel holds 1 024 rows).
