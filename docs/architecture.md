# Architecture

How the engine is put together: the model graph it runs, the Metal transport
below it, the kernels, speculative decoding, the memory layout and the
session cache. Measured numbers in this document are from an M5 Max
(40-core GPU, 128 GB) running the full Qwen3.8-Flash-Next conversion; the
method behind them is in [performance.md](performance.md).

## The checkpoint's architecture

Qwen3.8-Flash-Next (`model_type` `qwen4_exp`) is a 48-layer decoder with
hidden size 2560, vocabulary 248 320 and a maximum kernel context of
262 144 tokens. Six things make it different from a plain transformer.

### Hybrid attention: Gated DeltaNet and sparse attention, 3:1

Of the 48 layers, 36 are **Gated DeltaNet** layers and 12 are **Qwen Sparse
Attention** layers, interleaved 3:1. A GDN layer carries a recurrent state
instead of a growing cache: 48 x 128 x 128 f32 per layer, 3 MB per layer and
113 MB for the model. Its step reads and writes that state and a short
convolution window. Because the state is a fixed size, a GDN layer costs the
same at any context length, but it cannot be truncated to an arbitrary
position: a session that wants to resume at an earlier token needs a
checkpoint of the state at that position (see "Session cache").

An attention layer has 24 query heads over 2 key/value heads, head dimension
256, keys and values in bf16. Its cache is 24 KiB per token across the 12
layers and is truncatable by construction.

### The sparse-attention indexer

Up to a dense limit of 2 051 tokens an attention layer attends to everything.
Past it, an indexer decides what to look at. Keys are compressed 4:1 into
block keys, one per 4 tokens; for each query the indexer scores every
completed block, selects the top 512 by score, and attention then runs over
those 2 048 selected tokens plus the uncompressed tail. Selection is exact
and deterministic (ties resolve by block id), so it is not a source of
numerical drift. The indexer's own state is 3 KiB of raw keys plus 0.75 KiB
of block keys per token across the 12 layers, which makes the total per-token
cache 28 416 bytes.

A prefill chunk that extends past the dense limit takes the sparse path for
all of its queries, including those below the limit, where selecting the top
512 blocks of a shorter window selects all of them and the result is the
dense one.

### Mixture-of-experts feed-forward

Every layer's feed-forward is 512 routed experts of intermediate width 640
plus one shared expert. A router picks the top 10 experts per token. Routed
expert weights are 67.9 GB of the checkpoint, and a decode token reads
10 of 512 per layer, 1.327 GB. That single number sets the decode budget:
different tokens route to different experts, so nothing about the expert
traffic amortizes over a single sequence.

### Hyper-connections

The residual stream is four streams wide (4 x 2560 = 10 240) rather than one.
Each block reads a mixed view of the four streams through low-rank Q8
projections (`down` [320, 10240], `up` [10240, 320]), and injects its output
back through a per-layer gate (`inject` [4, 10240]). The mixers are small,
0.68 GB in total, but they are read twice per layer per token, which puts
them among the larger per-step costs.

### The hashed n-gram embedding

At layer 2 the model adds a parameter-lookup embedding: 16 hash functions
over the recent token history index a table of 320 001 536 rows of 160
values, quantized at 4 bits with group 32, which is 100 bytes per row and
32.0 GB in total. A token reads 16 rows, 1 600 bytes. The hashing constants
(layer multipliers, per-head vocabulary sizes and offsets) are copied out of
the source checkpoint into `config.json`.

Because the gather is so sparse, the table does not live on the GPU. The
server memory-maps the checkpoint shards and copies a step's 16 rows into a
small staging buffer on the host; the GPU dequantizes them with the same
gather kernel as before. Hashing runs on the host, which owns the token
history anyway. `--ngram-table resident` restores the fully resident layout
for comparison.

### The multi-token-prediction head

A separate 1.5 GB head (one trunk-style attention plus MoE block, two input
projections, its own stream mixer) predicts the next token from the trunk's
hidden state. Its input is built per hyper-connection stream:
`fc_hidden(norm(stream))` for each of the four streams, plus
`fc_embedding(norm(embed(next token)))` broadcast to all of them. The block
then runs like a trunk attention layer, with its own KV and indexer caches at
trunk positions, and its mixer feeds the shared LM head. During prefill the
head runs over the chunk paired with the next token so its caches keep up
with the trunk; that catch-up costs about 4.7% of prefill time.

The head drives speculative decoding. It is optional: `--mtp-drafts 0`, or a
conversion made with `--no-mtp`, runs the single-token decode graph.

## The Metal 4 transport

`src/metal.rs` owns the GPU. Everything above it works with a `ComputePass`:
allocate buffers, encode dispatches into a pass, commit, wait. The transport
under that API is Metal 4, which removes most of the implicit behaviour the
classic Metal API provided and replaces it with explicit contracts.

| Metal 4 has no ...                   | lily therefore ...                                                                                                                                                            |
|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| implicit ordering or hazard tracking | puts a cache-flushing barrier after every dispatch of a serial pass, keeps explicit level barriers between the dependency levels of a concurrent pass, and opens every command buffer with a queue-stage barrier so it sees the previous pass's writes |
| `setBuffer` / `setBytes`             | binds by GPU address through one argument table per pass; inline parameters are copied into a per-pass arena buffer, 16-byte aligned, and bound by address                        |
| residency tracking                   | keeps every buffer it allocates in one residency set attached to the queue, removes it on drop, and re-commits the set before the next submission when membership changed         |
| completion status on command buffers | follows every submission with a queue-level signal on a fence event; waits poll or block on that counter, and errors and GPU timestamps arrive through the commit feedback handler |
| reusable command buffers             | encodes into a command allocator taken from a pool; the allocator returns once its pass completed or was dropped un-committed                                                     |
| in-buffer event waits                | splits a pass that needs a mid-pass wait or signal into several command buffers with queue-level waits and signals between them                                                   |
| a blit encoder                       | copies through the compute encoder                                                                                                                                               |

Barriers use `MTL4VisibilityOptions::Device`. The `None` variant does not
flush caches and degenerates into an execution barrier, which is not enough
for a producer/consumer dependency.

**Inline parameters can come from the GPU.** Because a pass binds parameters
by address, `ComputePass::dispatch_with` accepts either host bytes, copied
into the pass's arena as usual, or a word of a resident buffer (`Param::Gpu`).
A kernel's `constant uint& base_pos` cannot tell the difference. This is what
lets a dispatch early in a pass decide a value that later dispatches of the
same pass consume, with no host round trip. Speculative decoding is built on
it.

**The buffer-lifetime contract: a pass holds buffers by address only.** There
is no reference from an encoded pass back to the buffers it reads. Every
buffer a pass touches must outlive the pass's completion. A temporary dropped
after encoding has its residency removed and its memory freed before the GPU
runs the pass. Production code binds scratch and state tensors, which live as
long as the session; test code has to keep its inputs in locals.

Bookkeeping costs, from the transport's own probes: the commit feedback after
the fence arrives 0.067 ms after it (median), a residency commit on the first
submission after an allocation costs 0.003 ms with about 4 000 resident
allocations, a buffer drop 0.002 ms, and a blocking fence wait costs 0.07 ms
more than polling on a 1 ms pass.

### Hiding the host round trip

The n-gram table lives in the page cache, so every decode step needs the host
once: it must see the sampled token, hash it, and stage the 16 rows before
the next step can use them. Measured on a 12.9 ms step at a 1K prompt, that
round trip was 0.56 ms of GPU idle: 0.09 ms until the blocked host thread
woke, 0.06 ms of host work, and 0.40 ms from committing the next pass to the
GPU starting it.

The engine removes it by **parking** the next pass. Step N+1 is encoded and
committed while step N runs, and waits on a shared event immediately before
its first host-staged input, the n-gram gather in layer 1. The embedding and
layer 0 run while the host wakes, stages the rows and signals. Waits are
paced: every pass signals a done event as its last command and the host
sleeps until shortly before the predicted completion, which brings wake-up
latency from 0.09 ms to 0.04 ms. Polling a command buffer's status instead
does not work; it is not updated promptly.

What parking costs in return is a fixed penalty: a pass whose wait was still
unsatisfied when the command buffer was scheduled runs about 0.3 ms longer,
whatever the release time, and lateness adds on top of that. So parking
converts 0.40 ms of submission latency into a 0.3 ms segment penalty. The
host is no longer on the critical path in either the plain or the speculative
loop.

When a generation stops with a step parked, that step runs anyway, fed with
the final token, so the number of tokens fed into the state can exceed the
number drawn. The server accounts for both cases, because a continued
conversation wants that token in its state.

## The kernel set and where time goes

Kernels live in `src/kernels/` as a Rust dispatch wrapper plus a `.metal`
source, compiled at runtime. The families are: dense Q4 and Q8 GEMV
(`quant`), tensor-op GEMM and its grouped expert variant (`gemm`), small-row
GEMMs for batched passes (`skinny`), MoE gather and combine (`moe`),
hyper-connection read and inject (`hc`), the n-gram gather (`ple`), sparse
attention scoring, selection and attention (`qsa`), dense attention
(`attention`), the Gated DeltaNet step and prefill scan (`gdn`), norms and
elementwise passes, the GPU sampler (`sample`), and the speculative accept
and rollback kernels (`spec`).

### Decode

A decode step is about 920 to 990 dispatches encoded as one concurrent pass
with level barriers. Per token it reads 4.135 GB of weights plus the 226 MB
of GDN recurrent state and 25 to 50 MB of attention caches, so about 4.4 GB.
Against a probed dense-GEMV bandwidth of 575 GB/s that is a 7.6 ms floor; the
step measures about 11.0 ms at a 1K prompt, which is 400 GB/s, 70% of the
probed peak.

Where the 1K step goes, by profiled share:

| kernel group                                        | share |
|-----------------------------------------------------|-------|
| dense Q4 GEMV (GDN, attention, shared expert)       | 31.5% |
| MoE gather GEMV, gate/up and down/combine           | 25%   |
| fused hyper-connection read (down, up and mix)      | 17.3% |
| LM head                                             | 4.8%  |
| GDN step                                            | 4.5%  |
| router GEMV and top-k                               | 6.5%  |
| dense attention                                     | 2.5%  |
| inject, n-gram and indexer GEMVs, conv, norms, rope, scatter, sampler, gathers | ~8% |

Past the dense limit the sparse-attention kernels enter: scoring, block
selection and the split attention add 2.2 to 2.5 ms per step at 8K and 32K.
That is the entire difference between a 1K and an 8K decode step.

The hyper-connection read is fused into two dispatches instead of six. The
down kernel uses one threadgroup per output row with one simdgroup per
stream, dotting each stream's Q8 blocks against the normalized residual in
f32 while accumulating the sum of squares in the same loop, so the RMS
normalization folds into the projection. The up kernel owns a group of output
columns per simdgroup, applies SiLU in registers and finishes with the mix
epilogue.

### Prefill

Prefill runs in chunks of 4 096 tokens (one shorter chunk for a shorter
prompt). Weights are read once per chunk, 71.1 GB, which is 17.4 MB per
token, 240 times less than decode's 4.1 GB per token. Adding activation
traffic (the 84 MB wide residual read or written about eight times per layer,
the MoE row gather, the intermediates) gives a bandwidth floor of about
0.25 s per chunk. A chunk measures 2.95 s at an 8K prompt. **Prefill is
compute- and efficiency-bound, not bandwidth-bound.**

Where a 4 096-token chunk goes at an 8K prompt:

| kernel                                     | share | achieved   |
|--------------------------------------------|-------|------------|
| sparse attention (`qsa_attn_split`)        | 40.2% | 2.1 TFLOP/s |
| grouped Q4 expert GEMM                     | 19.9% | 33 TFLOP/s |
| dense bf16 GEMM (tensor ops)               | 18.8% | 54 TFLOP/s |
| GDN prefill scan                           | 9.2%  | sequential recurrence |
| elementwise passes over the wide residual  | 3.7%  |            |
| MoE input gather                           | 1.8%  |            |

At a 1K prompt, where attention is dense, the grouped expert GEMM (40.4%) and
the dense GEMM (29.3%) dominate instead.

The sparse-attention prefill kernel is the outlier. Past the dense limit the
whole chunk runs through the decode-style split kernel in 256-query
sub-batches: score, select, then attend per query, gathering that query's
512 blocks of K and V with no reuse across queries and no tensor operations.
At 2.1 TFLOP/s next to a dense kernel at 29 on the same head shapes, it is
the single largest remaining item in the engine.

## Speculative decoding

With the draft head loaded, a decode step becomes two GPU passes.

**Verify.** The pending token plus `k` drafts run through one batched trunk
pass, which draws one token per row with the request's sampler. The draw
index equals the output index, so a seeded replay is unchanged by the draft
count. Acceptance is **exact equality**: draft `i` is accepted when it equals
the trunk's own draw for the prefix that precedes it. Every emitted token is
therefore a token the trunk itself drew, and the output does not depend on
the draft count at all. What the draft count changes is only how many rows a
pass confirms.

The outputs can still differ from the single-token decode graph on near-ties,
because the batched projections reduce in a different order. One such
divergence was found at a logit gap of 0.03 in 160 tokens. Both are valid
roundings of the same model.

**Accept and draft.** The draft pass is encoded once per step and committed
directly behind the verify pass, before the host knows anything. Its first
dispatch compares the trunk's draws with the drafts, counts the leading
matches `a`, and writes a control block: `a`, `a + 1`, and for each chained
row its position, the indexer block that position completes and a 0/1 flag
whether it does. Later dispatches of the same pass read those words as inline
parameters through `Param::Gpu`. **The accepted count is decided on the
GPU**, so the loop has no host dependency between verify and draft; measured
GPU idle between the two passes is 0.013 ms.

A GPU-supplied position works because every host decision that depends on a
position is monotone in it, so sizing for the whole candidate range is exact:
the dense path is taken only when the largest candidate fits the dense limit,
the block-key dispatch is one threadgroup gated in-kernel by the GPU's count
word, grids and strides come from the largest candidate and the kernels
already mask per row. Such a position is restricted to single-row passes
whose candidate range spans less than one indexer block, which keeps "at most
one block completes" true.

**Rollback needs no recomputation.** The GDN prefill scan records the state
after every row, so the accepted one is copied back by index; the convolution
windows are rewound from the rows' saved inputs; attention caches are
position-indexed and simply get overwritten. The head is then caught up on
all `m` rows with the trunk's draws (the rows past `a` compute values that
later rows overwrite before anything valid reads them) and chains further
drafts from its own residual.

**Drafts are greedy.** The head proposes with argmax while the trunk may
sample with temperature, so a sampled request accepts fewer drafts than a
greedy one and gains less. Making acceptance exact under sampling would mean
standard speculative sampling with rejection against the trunk's
distribution, which changes the verify pass's sampler.

Cost: a verify pass over `m` rows is the batched prefill graph, and the extra
rows are mostly the extra experts they route to, up to 10 more per layer per
row. A 3-row pass measures about 18.9 ms at 1K and 20.6 ms at 8K against
about 11 and 12.9 ms for the decode graph. With 2 drafts a step yields about
2.2 to 2.6 tokens.

## Memory layout

On a 128 GB machine with the full checkpoint:

| what                                | bytes    | where                              |
|-------------------------------------|----------|------------------------------------|
| resident weights without the table  | 71.1 GB  | GPU, one residency set             |
| n-gram table                        | 32.0 GB  | page cache, memory-mapped, evictable |
| draft head                          | 1.5 GB   | in the 71.1 GB above when converted |
| per-token cache, all layers         | 28 416 B | session cache                      |
| GDN recurrent checkpoint            | 113 MB   | session cache, up to 3 per session |
| prefill scratch                     | ~1.5 GB  | grown on demand to the 4 096-token chunk |

Moving the n-gram table off the GPU is what makes the model fit without
raising `iogpu.wired_limit_mb`: 71 GB fits under the 96 GiB default. The
table still wants its 32 GB of physical memory as page cache for decode to
stay fast, so the session cache budget accounts for it.

The table is read once at startup (`--ngram-preload`, 3 to 6 s), residency is
verified with `mincore`, and `madvise(WILLNEED)` is issued for a step's rows
before they are copied so any remaining faults overlap. `--ngram-lock` pins
it. Weights are read with `F_NOCACHE` so a model load does not evict the
table. Without preload a token with new n-grams costs 0.6 ms of host time
with 8 reader threads, or 1.8 ms serially, against 0.05 ms warm.

**Session cache budget.** By default it is the device's recommended working
set minus what is already allocated minus the paged weights minus 8 GiB of
headroom for other applications, floored at 8 GiB. On a 128 GB machine that
is 115.4 - 73.0 - 32.0 - 8.6 GB, so the floor applies and the budget is
8 GiB, which is two full 131 072-token contexts. The disk tier holds the
rest. The server logs the derivation at load.

## The session cache

A session is one token lineage: its tokens, a decode state whose attention
and indexer buffers grow in 8 192-token steps, and up to three **checkpoints**
of the recurrent part (GDN states, convolution windows) taken at
`prompt_len - 1` of recent requests. Attention caches are truncatable by
construction, so a checkpoint makes every prefix up to its position
resumable. The position is `prompt_len - 1` and not `prompt_len` because an
identical prompt, a regeneration, must still feed one token to produce
logits.

Acquiring a session for a new prompt: for every session, find the longest
common prefix with the prompt and the latest checkpoint at or below
`min(lcp, prompt_len - 1)`, where the live end counts as a checkpoint. Take
the largest such position. A pure extension of the live end reuses the
session in place. A rollback **forks**: the KV prefix and the checkpoint are
copied into a new session, so a parallel conversation that shares only a
system prompt never destroys a long context. Sessions are evicted least
recently used under the byte budget.

**The disk tier.** An evicted session is written to
`<disk-cache-dir>/<format>/<id>/` as its per-token caches for all tokens,
every checkpoint and a snapshot of the live end, and indexed in memory. A
later prompt sharing its prefix reads the prefix back, one to two seconds for
a full context, instead of recomputing it. A hit at the live end moves the
session back to the GPU and drops the files; a hit at an earlier checkpoint
forks from the file. Entries are evicted least recently used under the disk
budget and deleted once unused for the TTL, which is checked at startup,
before every lookup and on every write, so the tier drains on its own. Only
sessions of 256 tokens or more are kept and an 8 GB free-space margin is
respected. The index is rebuilt from the meta files at startup. The directory
is tagged with the model's persistence format, so a different model or a
changed cache shape never reads the files.

Writes happen on the engine thread at eviction time, at a few gigabytes per
second, which puts them on the request path.

## The server

One engine thread owns the model's lifetime and runs one generation at a
time; a bounded queue holds the rest. Tokenization and chat-template
rendering run on the connection thread, so they never touch the engine
thread. Detokenization and the output parser run per token inside the token
callback, which executes while the parked next step is already running.

The HTTP layer is a small HTTP/1.1 implementation over `std::net`, one
connection per thread with `Connection: close`. It is hand-written because
disconnect detection needs the socket: a general-purpose crate buffered
writes and swallowed the errors, so a departed client kept the GPU busy to
`max_tokens`.

Sampling runs on the GPU, so that only the token id crosses to the host. Two
kernels per step: a wide one applies the penalties (presence, frequency and
repetition over generated tokens through a per-request count table) and
temperature and reduces the maximum; one threadgroup then selects the top-k
by a two-level bucket search on the distance below the maximum (4 096 bins
over a range that widens when needed, refined once, boundary resolution
2^-24 of the range), applies top-p and min-p over the sorted candidates, and
draws by inverse CDF from a counter-based hash RNG seeded by `seed` and the
step index. Greedy without penalties takes the exact argmax kernel. Measured
per draw at the 248 320-token vocabulary: greedy 24 us, top-k 20 with
top-p 0.95 350 us, the 1 024-candidate cap 170 us.

Stop signals are taken synchronously rather than in a signal handler: SIGTERM
and SIGINT are blocked before the first thread is spawned, so every thread
inherits the mask, and one thread `sigwait`s for them. The inherited
disposition is reset to default first, because a non-interactive shell starts
background jobs with SIGINT ignored and an ignored signal is discarded before
`sigwait` could take it.

A Metal command-queue error is treated as a transport failure rather than a
request failure, because the queue's state after one is unknown: the context
records the first error the commit feedback reports and fails every later
submission and wait with it, so the engine is dropped and reloaded on the
same thread while `/health` reports `recovering`. Sessions are dropped rather
than spilled in that case, since a faulted queue cannot run the snapshot copy
and the caches are not trustworthy; disk entries written earlier by a healthy
queue stay valid. A budget of 3 recoveries per sliding 10-minute window
bounds the loop; the fourth fault exits 1 for the supervisor.

## What is deliberately not here

Constrained decoding (`response_format: json_schema`), batching across
requests, the vision tower, and session persistence for the Qwen3.6-35B path
(the engine trait's defaults disable the disk tier for it).
