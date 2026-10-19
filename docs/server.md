# The server

What the HTTP server does beyond the OpenAI API it speaks, for clients and
operators. `lily --help` lists every flag; how the engine works inside is in
[architecture.md](architecture.md).

## The API

`POST /v1/chat/completions` (streaming or not), `POST /v1/completions`,
`GET /v1/models` and `GET /health`, with tools, `reasoning_content` and
`prompt_cache_key`. It is developed against opencode. Where it differs from
other OpenAI-compatible servers:

- **Images** are base64 data URIs (PNG or JPEG, no count limit beyond the
  context); the server never fetches URLs. A 1920 x 1080 screenshot costs
  about 2 000 prompt tokens. No video.
- **Message text is always text.** A special token spelled in a message, a
  tool result or a tool definition (`<|im_end|>`, `<|image_pad|>`) is
  tokenized as the characters it is, never as conversation structure or an
  image placeholder; only the chat template's own markup is special.
  `/v1/completions` takes its raw prompt as written, special tokens
  included.
- **`max_tokens`** beyond the context is clamped, not refused.
- **Every response carries a `timings` object** with prefill and decode
  rates, cached tokens and draft acceptance. `GET /v1/timings` keeps the
  last 32; `tools/opencode-plugin-timings/` shows them in opencode.

## The dashboard

`http://127.0.0.1:8000/` (the `--bind` address) serves a dashboard built into
the binary: what the server is doing, its cache tiers, recent requests, the
last 7 days and every setting. It reads `/v1/lily/status`, `/sessions`,
`/requests`, `/history` and `/settings`, which answer from what the engine
already publishes and from the recorder's database, so an open page never
wakes an idle or unloaded engine. Every dashboard route answers loopback
peers only, under `127.0.0.1`, `localhost` or `[::1]` with lily's port as
`Host` (403 otherwise, whatever `--bind` says), and forbids framing. A path
that is not the dashboard's (no `/v1/lily/` route, no file of the app) gets
the app only when a browser navigates to it: a GET or HEAD of a path with no
extension, outside `/api/` and `/v1/`, whose `Accept` names `text/html`, so
the app's own routes survive a reload. Anything else (a client probing
`/models`, `/api/tags`, `/version` or `/metrics`) answers the server's
`404` `{"error": {"message": "not found", "type": "invalid_request_error"}}`,
as before the dashboard.

The history lives in a SQLite database, `--stats-db <path>` (an absolute
path; default `~/Library/Application Support/lily/lily.sqlite`, no `~`
expansion): one row of numbers per finished request (counts, durations,
rates, never text or tokens), per-minute rollups, machine load and the cache
tiers, all kept 7 days and an hour. A directory lily creates for it is
`0700`, the database and its WAL files `0600`. A recorder thread of its own
opens and writes it, so the start never waits on SQLite: an open that fails
is retried (1 s, 2 s, 5 s, then every 30 s) while `/v1/lily/status` says
`recorder.state: "opening"` with the error. The engine only offers each row
to a bounded channel; a full one drops the row (counted as
`recorder.dropped_rows`, as are rows of a batch that failed twice). A file
that is not a database or is corrupt is renamed to
`<path>.corrupt-<unix seconds>` and a fresh one started; a database written
by a newer lily is left untouched and the history stays off
(`state: "failed"`). On shutdown the recorder gets up to 2 s to write what
it holds and checkpoint the WAL. `--stats-db off` records nothing: the
history routes answer 503 and the per-session figures cover the last 1 024
requests. The design is [dashboard.md](dashboard.md).

### Machine power, GPU and temperatures

`--machine-stats on|off` (default `on`, restart-only, listed read-only in the
Settings view) runs a sampler thread of its own, at the utility quality of
service, over the `macmon` crate (IOReport and the SMC, no root): system, CPU,
GPU and ANE power, the GPU's activity and frequency and the CPU and GPU
temperatures. The thread creates the sampler (about 0.7 s), never while the
server starts, and takes a reading 2 s after the previous one while a request
is running or a dashboard request came in within the last 10 s, 30 s after it
otherwise, so the history has idle samples too. lily does the waiting itself
(`macmon` is asked for no window after the first, so it measures exactly the
time since its last call): the wait is cut short when the page opens (a
dashboard request after the page was closed) and when the engine starts work
(a request taken on, a reload, a recovery after a GPU fault), and ends at once
at shutdown, so a page opened or a request arriving in the middle of an idle
wait gets a fresh reading immediately. Nothing polls in between: the engine
pays a fence and a relaxed load per request, and a lock and a notify only when
it finds the sampler in an idle wait.

**Idle wake-ups.** How often lily's own threads wake, by state (the engine's
own idle timers, the pin's hold and the disk tier's write-ahead, are not the
dashboard's and are unchanged):

| state | sampler (`lily-machine`) | recorder (`lily-recorder`) |
|---|---|---|
| engine unloaded, no page within 10 s | every 30 s | every 60 s (and at the hour, to prune) |
| engine loaded and idle, no page | every 30 s | every 10 s |
| page open (a dashboard request within 10 s) | every 2 s | every 10 s |
| a request running, a load, a reload, a recovery | every 2 s | every 10 s |
| `--machine-stats off` | no thread | as above, the sampler's column never short |
| `--stats-db off` | as above | no thread |

The recorder's 60 s cadence holds while the engine is unloaded after a load
with no request open, no dashboard request came within 10 s and the sampler is
not at its 2 s cadence. It blocks on its channel until the next sample is
due, so a request's row and a settings change still arrive at once, and a row
that finds lily busy again brings the next sample forward; the minute rows
aggregate whatever samples a minute got, so the history stays complete. A page
or a request that starts during the recorder's 60 s wait is sampled at its
end, at most a minute late (the sampler's power readings are not delayed).

`/v1/lily/status` shows the latest reading as `machine.power`, `gpu_active`,
`gpu_freq_mhz` and `temp_c` while it is younger than three waits (6 s, or 90 s
after an idle one) and `null` otherwise. `machine.power.sys_w` is `null` when
the SMC does not report the system's power: macmon then gives 0, or exactly
the sum of the parts (it never reports less, so a sum means the SMC read below
them). The parts are still shown. `machine.sampler` is `ok` (the thread
runs; the figures can be `null` until the first fresh reading), `off`
(`--machine-stats off`: no thread) or `failed` (it could not start, or stopped
after repeated errors).

`machine.power.lily` is lily's own share of that power, **an estimate**:
`{ w, cpu_w, gpu_w }` in watts, `w` being the sum. `cpu_w` is the change of
lily's CPU energy over the reading's window (the kernel's per-process account,
`proc_pid_rusage` `ri_energy_nj`, both core clusters), `gpu_w` the reading's
GPU power times lily's GPU busy time (its command buffers' spans from Metal's
commit feedback) over the GPU's active time in the window, at most all of it.
The ANE, DRAM and the rest of the SoC are not attributed, so `w` stays below
what lily costs the machine, by a lot while decoding (DRAM). It is `null`
when it cannot be told for a reading (a counter call failed, a counter went
back, a value out of range); the rest of the reading is shown regardless.
The dashboard labels it "lily CPU+GPU (est.)"; dashboard.md § 6 has the
method and its limits.

`machine.power.rest_w` is the system power the parts do not account for:
`sys_w` minus CPU, GPU and ANE (never below 0, `null` with `sys_w`). It is
memory, fabric, display, storage, board and conversion losses, measured only
as a whole. While lily decoded on 2026-10-15 (macOS 26.6) it was about half
the system's power, the memory side about 40 W of it; macOS 27 gives no split
of it, so it is attributed to no one (dashboard.md § 6).

A reading that fails, is not finite, or is out of range (a power or the system
power above 1 000 W, a frequency above 10 000 MHz, a temperature outside
-20 to 150 °C) is skipped and counted, and one good reading resets the count;
the first skip and every 100th are logged. Five failures in a row, a panic, or
a sampler that cannot be created (IOReport refuses where a sandbox blocks it)
log `machine: <what> failed: <cause>; the dashboard shows no power, ...` once;
the sampler is then `failed`, its thread ends and the figures stay `null`;
nothing else is affected, and a restart tries again. A Rust panic in the
sampler is caught; a crash inside `macmon`'s FFI (a NULL CoreFoundation object
dereferenced) would end the whole process, which a subprocess probe would
prevent (a `TODO(dashboard)` in `src/serve/machine.rs`; the owner's decision).
CPU load, memory, swap and pressure are lily's own probes and do not depend on
the sampler.

The recorder keeps each window in the `machine` table, spread over the
minutes it overlaps: per minute the system power's least, time-weighted mean
and most, the GPU's power, activity and frequency, the hottest temperatures,
`power_covered_ms` (the milliseconds of the minute the windows covered, never
more than 60 000) and `sys_covered_ms` (of those, the ones that had a system
power). The history's power series are those means weighted by the covered
time, and the summary's **energy is the sum over minutes of mean system power
times the time that had a system power**: a minute the sampler did not cover (it
was off or failed, lily was not running, a stall) or that had no system power
adds nothing and is never filled in, so `summary.energy_wh` is what the machine
used while it was being measured, not an extrapolation over the range, and
`power_avg_w` is the mean over the same time. Both are `null` while no minute
has a system power. lily's own estimated power is kept the same way, one
series with a coverage of its own (`lily_w_mean`, `lily_covered_ms`): the
buckets' `lily_power_w` and the summary's `lily_energy_wh`, `null` where no
reading had a figure. The database is at schema 3 (`power_covered_ms`,
`sys_covered_ms` and the temperature columns are added to a v1 database, the
two lily columns to a v1 or v2 one, when it is opened). A lily from before
schema 3 refuses a v3 database as newer and runs with history off (the file
untouched): rolling the service back to such a binary stops the history until
a schema-3 lily runs again.

### Runtime settings

Seventeen settings change without a restart, from the dashboard's Settings
view or its routes; every other flag is shown read-only.

| settings | a change applies |
|---|---|
| `--temperature`, `--top-p`, `--top-k`, `--min-p`, `--presence-penalty`, `--frequency-penalty`, `--repetition-penalty` (the defaults a request that sets none gets) | at the next request |
| `--reasoning-effort` (`low`, `medium` or `xhigh`; `high` and `max` are stored as `xhigh`), `--thinking-budget`, `--thinking-budget-tool-turn-factor`, `--thinking-budget-grace`, `--thinking-nudges`, `--tool-call-ends-thinking` | at the next request |
| `--durable-min-tokens` | at the next request's admission |
| `--idle-unload` | at once: the timer is re-armed, the new delay counting from the last request (a delay shorter than the time already idle unloads now; the page asks first) |
| `--pin-hold` | at once, the same way: a shorter hold than the time since the last request releases the pin now |
| `--disk-cache-ttl` | at the next expiry pass (every lookup and store) |

A request is prepared from one look at the settings, so a change never
reaches a request already running. `--queue` (it sizes the engine's
channel when the server starts) and `--decode-checkpoint-tokens` (a
running generation keeps the interval it was admitted with while the batch
admission budgets newcomers by the store's, and the expert cache's plan
turns checkpoints off whatever the flag says) stay restart-only.

**Test and benchmark servers** run with `--stats-db off`. Every lily on the
machine shares the default database, so a server started for an e2e run, a
benchmark or a bisect would otherwise take the settings stored from the
dashboard over its own flags (a stored temperature or thinking budget
changes what is measured) and add its requests to the history of the
service. With `--stats-db off` the flags are the whole configuration and
nothing is recorded. `tools/e2e/durable.py` and `tools/e2e/vision.py` pass
it to the servers they start; a server started by hand for
`tools/bench/http_bench.py` needs it on its command line. `lily-bench` and
`lily-probe` read and write no database.

**Limits**, the same for a flag and a runtime setting: durations at most
365 days (`duration "400d" is longer than the 365-day cap`), thinking
budgets and the grace at most 1 048 576 tokens, the tool-turn factor a
positive number of at most 100.

**Where a value comes from**: the database's override, else the flag, else
the default. Overrides live in the `settings` table of `--stats-db` and are
applied over the command line once the recorder has opened the database
(until then the flags are in effect, and the startup lines that state the
idle unload, the pin hold and the disk TTL say "the flag's value; database
overrides may apply"); the log then says
`settings: from the database, over the command line: <name> <value>, ...`
at every start. The disk tier runs no expiry pass before that (at most
30 s after the start), so the flag's TTL never deletes an entry the stored
one keeps. A stored override for a setting this lily cannot change at
runtime, or whose value no longer passes the flag's checks, is logged,
left out and kept in the database; if the `settings` table cannot be read
at all, the flags stay in effect and changes are refused (503) while the
history works on.

**The change log** (`settings_log`, kept 7 days) holds every change from
the dashboard, and an entry at startup for each stored override that
differs from the flag's or default value it replaces. Setting a value that
is already in effect changes nothing and logs nothing; setting the flag's
value while an override is in effect keeps an override with that value
(it pins it against a later change of the flag).

**Routes** (same loopback and write guard as the rest of the dashboard:
writes need `Content-Type: application/json` and, when an `Origin` is
sent, lily's own; a write's body is at most 64 KiB, 413 past it):

- `GET /v1/lily/settings`: every setting with its value, `source`
  (`database`, `flag` or `default`), `editable` and `applies`. With
  `--stats-db off` the runtime settings are listed `editable: false`, the
  reason in their `help`.
- `PATCH /v1/lily/settings` with `{"<name>": value, ...}`: all of it or
  none, each name once (twice is a 400). A value is the flag's text, not
  trimmed (`"30m"`, `"low=8000,xhigh=16000"`), or a JSON number or boolean
  standing for it, checked by the flag's own parser and limits, so a value
  the command line would refuse is refused with the same message. Answers
  `{"updated": [...]}` (the settings that changed, as they are now) once
  the change is in the database and in effect; 404 for an unknown name,
  409 for a restart-only one, 400 for an invalid value (`error.setting`
  names it), 403 from the guard.
- `DELETE /v1/lily/settings/{name}`: deletes the override (the row by
  name, also a stale one the start ignored); the setting is back at its
  flag or default value. Answers `{"setting": {...}}`.
- `GET /v1/lily/settings/log`: the change log, newest first.

A change is written before it takes effect, and a write's 5xx says what
happened: 503 nothing changed (no open database, settings that could not
be read, a write the database refused, or one the recorder did not take
up within 3 s), 504 the outcome is unknown (the recorder took the change
up but did not confirm it within 3 s: "the change may still apply; reload
to see the current value"; the page reloads the settings and the log after
any 5xx), 500 the change could not be worked out. The change log answers
503 without the database.

## Notes for upgrading

A server that starts from a script or the launchd service may refuse flags
it took before the dashboard: a duration over 365 days (`--idle-unload`,
`--pin-hold`, `--disk-cache-ttl`), a thinking budget or
`--thinking-budget-grace` over 1 048 576 tokens, and
`--thinking-budget-tool-turn-factor` over 100 now stop the start with the
flag's message. Two threads are new: `lily-machine` (the sampler, unless
`--machine-stats off`) and `lily-recorder` (unless `--stats-db off`). The
recorder creates `~/Library/Application Support/lily/lily.sqlite` by default
(its directory `0700` if lily creates it, the database and its WAL files
`0600`); a service whose `HOME` is unset or relative keeps no history unless
`--stats-db` names a file. Unknown paths answer `404` as before,
except a browser's navigation (a path with no extension whose `Accept` names
`text/html`), which gets the dashboard (see The dashboard above).

## Batching

Up to four requests decode together (`--max-batch`, continuous batching): a
request that arrives while another decodes gets its prefill between the
other's steps instead of waiting for its answer, then the two share each
decode step. A request decoding alone keeps speculative decoding; batched
ones decode without drafts. Prefills run one at a time; more requests queue,
with 503 when the queue is full (`--queue`). `--max-batch 1` serves one
request at a time. On a machine that does not hold the model (the expert
cache) requests are always served one at a time, and an explicit
`--max-batch` above 1 refuses to start.

## Memory

The session cache keeps recent conversations' state in GPU memory and
spills the rest to the disk tier (`--disk-cache-bytes`). The defaults:

| | default | otherwise |
|---|---|---|
| `--kv-cache` | `q8`: 8-bit attention K/V caches (q8_0), 18 304 bytes per token of context | `bf16`: the model's precision, 30 784 bytes per token |
| `--cache-bytes` | what the working set leaves after the weights and 8 GiB of headroom, at least 5 GiB with q8 (8 GiB with bf16, about the same context) | any size with q8; with bf16 at least 8 GiB, or the server refuses to start |
| `--max-batch` | 4 | 1 serves one request at a time |

q8 runs at the same speed as bf16, at a teacher-forced KL of about 2e-3 to
4e-3 against it (see [architecture.md](architecture.md), "The 8-bit K/V
cache"). Each format and each checkpoint (q4, q4-xl, a re-conversion)
keeps its own disk-tier directory, so sessions persisted under one are not
read under another.

On a machine that does not hold the model (the expert cache,
[low-ram-experts.md](low-ram-experts.md)) the caches are q8 and requests
are served one at a time: `--kv-cache bf16` or `--max-batch` above 1
refuses to start there. Its default budget is the one full `--max-seq`
session the plan reserved (2.7 GB at 131 072 tokens), not the floor
above.

## Thinking

Thinking is on by default. `reasoning_effort` (`none`, `low`, `medium`,
`high`) sets it per request; `--thinking` and `--reasoning-effort` set the
server's default.

### Thinking controls

The model sometimes reasons for many thousands of tokens without closing its
`<think>` block, writes a tool call inside the block, or ends its turn there,
so an agent receives nothing but reasoning. Three controls act on the token
stream while it is drawn; for chat requests they are on by default:

| flag | default | request field |
|---|---|---|
| `--thinking-budget` | `low=8000,medium=8000,xhigh=16000` | `thinking_budget` |
| `--thinking-nudges` | `true` | `thinking_nudges` |
| `--tool-call-ends-thinking` | `true` | `tool_call_ends_thinking` |

On a captured opencode turn where the model kept reasoning (56K tokens of
context, `low` effort, 30 samples each, 4-bit checkpoint), the defaults
took the runaway reasoning blocks from 16 of 30 to none and the turns that
went on to a tool call from 9 to 18. The 16 000 at `xhigh` is not measured.

- **`thinking_budget`** (a positive token count; a negative one turns a
  server default off, 0 is refused). The server's budget follows the
  template's reasoning effort (`high` is `xhigh`, which is also the
  template's default), or one number for all, or `off`. Once the reasoning
  block holds that many of the model's tokens, it is closed at the next line
  end (after a grace window also a sentence end, after a second one
  anywhere), never inside a code fence or a tool call (it waits for them to
  end; a fence never closed means no close), with a short action-oriented
  preface and `</think>`. With a budget, an end of turn the model draws
  inside the block is replaced, once, by `</think>` alone, and the model
  goes on to its tool call or answer. `--thinking-budget-tool-turn-factor`
  scales the server's budget for a turn whose last message is a tool
  result; `--thinking-budget-grace` sets the windows (128 tokens).
- **`thinking_nudges`** (with a budget): sentences of increasingly firm
  wording inserted into the reasoning at 50, 75 and 90 % of the budget, each
  only at a line end within the grace window (otherwise skipped).
- **`tool_call_ends_thinking`** (chat requests with tools): a `<tool_call>`
  at a line start inside the reasoning block, outside a code fence
  (CommonMark's rules) and outside a tool call the block kept, ends the
  block, with `</think>` inserted in front of it. One mid-line or in a fence
  is reasoning text.

Request fields go at the top level or in `chat_template_kwargs` (which
wins). `/v1/completions` takes only the request fields, and only for a
prompt that ends with `<think>\n`. `--thinking-budget off`,
`--thinking-nudges false` and `--tool-call-ends-thinking false` turn the
defaults off. `--thinking-texts` replaces the inserted texts with a JSON
file (`{"close": ["..."], "nudges": [{"at": 0.5, "texts": ["..."]}]}`);
texts containing special tokens are refused.

**What the client sees.** Inserted tokens are fed to the model like
generated ones, appear in the stream (the texts in `reasoning_content`) and
count as completion tokens. They are the ids the whole text encodes to, so
a next turn that sends the reasoning back reuses the session cache through
them.

**What it costs.** Inside the reasoning block, a budget makes a decode that
would pipeline its steps rest at every token (the plain loop, and batched
steps, which park nothing while such a row thinks);
`tool_call_ends_thinking` alone does so at every line start there. Measured:
nothing for a request decoding alone with drafts, about 1 % without drafts,
2 to 4 % with two to four batched requests. Module docs:
`src/thinking.rs`; the decode-loop details: [architecture.md](architecture.md).
