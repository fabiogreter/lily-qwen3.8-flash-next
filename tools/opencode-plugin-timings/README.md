# opencode TUI plugin: lily timings

Shows what the last request actually cost, to the right of the session
prompt in [opencode](https://opencode.ai):

```
pf 21.9k 365/s · dec 16 38.5/s · spec 100%
```

The numbers are lily's own. The community throughput plugins estimate tokens
from the characters they watch streaming past and divide by a stopwatch; this
one reads the counters the server already kept, so prefill and decode are
separated and the draft head's acceptance is real.

## What each number means

| part | meaning |
|------|---------|
| `pf 21.9k 365/s` | 21 934 prompt tokens were run through the model, at 365 tokens per second. The rate is over the tokens actually computed, never over the whole prompt, so a cache hit cannot inflate it. |
| `dec 16 38.5/s` | 16 tokens were generated, at 38.5 tokens per second. |
| `cached 98%` | Only shown when the session cache supplied part of the prompt: the share of prompt tokens that came from a resident prefix, a forked session or the disk tier, and so cost no prefill. |
| `spec 100%` | Only shown when speculative decoding ran: the share of proposed draft tokens the verify pass accepted. Absent, never `0%`, when the draft head is off. |

A rate is `—` when the server reported none, which happens when nothing was
computed in that phase. The line is refreshed after every assistant message
and stays on screen until the next one.

## Where the numbers come from

`GET /v1/timings` on the lily server, read right after each assistant message
completes; it returns the last 32 completed requests, newest first.

Not from the response itself, although lily also puts the same `timings`
object next to `usage` in every response: nothing carries a custom response
field to a plugin. opencode's `AssistantMessage` is a closed schema (`tokens`,
`cost`, `finish`, `providerID`, …) with no free-form provider field, and the
AI SDK's `@ai-sdk/openai-compatible` provider drops response keys it does not
know. Checked on a real turn: the stored message for it held
`"tokens":{"total":21950,"input":21934,"output":16,…}` and the string
`timings` appeared in no row of opencode's database.

## Installing

TUI plugins are listed in **`~/.config/opencode/tui.json`**, not in
`opencode.json`/`opencode.jsonc`. The `plugin` array in `opencode.jsonc` (the
one holding `opencode-pty`) is the *server* plugin list: a module listed there
is loaded for its `server` entrypoint and its `tui` entrypoint is never
reached, so the slot stays empty. Verified against opencode 1.18.31 with this
plugin in both places.

Add it to the `plugin` array of `~/.config/opencode/tui.json`, keeping what is
already there:

```json
{
  "plugin": [
    "@jimicze-opencode/opencode-tps",
    "file:///Users/you/projects/personal/local-llms/lily/tools/opencode-plugin-timings"
  ]
}
```

A project can do the same in its own `<project>/.opencode/tui.json`.

## Configuring

Both settings are optional; the defaults match a stock lily.

| setting | option key | environment variable | default |
|---------|-----------|----------------------|---------|
| server base URL | `baseUrl` | `LILY_TIMINGS_BASE_URL` | `http://127.0.0.1:8000` |
| opencode provider id whose messages the numbers belong to | `provider` | `LILY_TIMINGS_PROVIDER` | `lily` |

Options go in the tuple form of the plugin entry:

```json
{
  "plugin": [
    [
      "file:///Users/you/projects/personal/local-llms/lily/tools/opencode-plugin-timings",
      { "baseUrl": "http://127.0.0.1:8000", "provider": "lily" }
    ]
  ]
}
```

The plugin is silent and invisible whenever it has nothing to say: a message
from another provider is ignored without a request, and a server that is not
there, is not lily, or does not answer `/v1/timings` leaves the slot empty. It
is safe to leave installed in sessions that never touch lily.

## Developing

```sh
node --test          # the formatting, the body guard and the settings
```

`index.tsx` is loaded as TSX by opencode, which applies the solid transform
itself (`tsconfig.json` therefore keeps `"jsx": "preserve"`), and resolves
`solid-js` to the runtime it already has; nothing needs to be installed or
built.
