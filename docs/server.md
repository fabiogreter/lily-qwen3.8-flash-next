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
`Host` (403 otherwise, whatever `--bind` says), and forbids framing.

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
