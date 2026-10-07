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
cache"). Each format keeps its own disk-tier directory, so sessions
persisted under one are not read under the other.

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
