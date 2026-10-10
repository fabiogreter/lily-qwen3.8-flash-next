# Running on machines the experts do not fit: measurements and design

Written 2026-09-18. The checkpoint is 104.6 GB: 68 GB of routed experts
(48 layers x 512 experts, 2.76 MB each: gate and up projections 0.92 MB
each, down 0.92 MB), a 32 GB n-gram table already served from the page
cache by host reads, and about 5 GB of everything else. A 64 GB machine
leaves roughly 45 GB for experts after the OS and the engine's scratch:
about two thirds of them. Everything below is measured on the M5 Max
(128 GB) unless marked as an estimate.

## What was measured

**Expert usage is skewed enough to matter.** `lily-experts` over 40
real-text prompts of 8K tokens (documentation and code from three
repositories): 49% of the (layer, expert) slices carry 90% of the
routing, 62% carry 95%, 81% carry 99%; within a layer the busiest half of
the experts carry 80 to 95% of its routing, more in the late layers; 59
of 24 576 slices were never used. So a resident set of two thirds of the
slices, chosen by usage, serves roughly 96% of expert reads; the
remaining 4% are about 19 expert slices per decoded token (53 MB) and
about a third of every expert per prefill chunk (23 GB per 4 096
tokens).

**Transparent paging of GPU buffers does not work for that pattern.**
`MetalContext::new_buffer_mapped` wraps a file mapping in a no-copy Metal
buffer; the OS then drops its pages under pressure and the GPU faults them
back. Under a 100 GB locked balloon (`mapped_buffer_read_bandwidth`,
`mapped_buffer_random_slice_bandwidth`):

| access                              | cold (faulting)      | warm      |
|-------------------------------------|----------------------|-----------|
| 17 GB sequential                    | 3.9 to 4.5 GB/s      | 563 GB/s  |
| 64 random slices of 2.8 MB          | 64 ms per slice      | 172 GB/s  |

Sequential faults get readahead; random ones are serviced a 16 KB page at
a time, about 0.37 ms each, so one expert miss would cost 64 ms and a
token with 19 misses over a second. Mapping is therefore only useful
where the host pre-touches what the GPU will read.

**Host reads of expert slices are cheap.** `pread` of 64 random 2.8 MB
slices from the checkpoint shards: 0.39 ms mean (0.61 worst) without
pressure, 0.40 to 0.49 ms (worst 3.5) under the same balloon, 6 to 7 GB/s.

**The checkpoint's expert tensors are not all 16-byte aligned** (91 of
147 code tensors start 8 bytes off a 16-byte boundary in their shard), so
mapping the shards as GPU buffers is out for them anyway; host reads into
GPU memory have no such constraint.

## Running it

Nothing to set on the small machine: at load the engine compares the
checkpoint with `hw.memsize` and engages the cache when it does not fit.
`LILY_MEMORY_GB=64` (for `lily` and `lily-bench` alike) plans for that
much memory instead of the machine's, which is how to try the mode here:
a 64 GB machine picks about 16 250 slots on this checkpoint (the load
prints the number) and keeps speculative decoding on. Before 2026-10-08 this
was `--memory-gb`, which now refuses to start and names its replacements.

`--memory-limit-gb G` is the other knob: the most lily should take. The
plan sizes the process for the machine's memory less its reserve (12 GB,
or a sixth of memory, for the OS, other apps and the page cache); a limit
below that lowers the target, GB for GB out of the slab, and a limit above
it changes nothing, so it can never hand lily the reserve. The target
covers the process footprint (weights, scratch, the session cache), not
the page cache that serves the paged n-gram table and the streamed
experts. On a machine that holds the checkpoint, a limit that leaves less
than one full session (or the session cache's floor, 5 GiB with q8 caches)
beside the weights engages the cache; otherwise the server's session budget
is lowered to what the limit leaves after the load. The two compose:
`LILY_MEMORY_GB=64 lily --memory-limit-gb 44` is a 64 GB machine whose
user keeps 20 GB for other work. Copy
`tools/bench/expert-usage-qwen38-flash-next.json` next to the checkpoint
as `expert-usage.json` (or point `LILY_EXPERT_USAGE` at it) so the
busiest experts are the resident ones; without it the placement is
uniform. The load prints what it decided, and the bench prints the
cache's lookups and misses per phase.

### Sessions and the context length

The server's session cache comes out of the same memory as the slab. The
plan keeps exactly one full session at `--max-seq` free before it sizes
the slab, and the server's session budget is then exactly that session
(`--cache-bytes` still overrides): its per-token caches, the recurrent
state and three checkpoints of it, the draft head's own caches included
(drafts run under the cache since 2026-10-09, below). The K/V caches are
always q8 here (`--kv-cache bf16`
refuses to start), and requests are served one at a time (`--max-batch`
above 1 refuses to start). More sessions spill to the disk tier and come back from
it, synchronously: the full machine's write ahead of evictions is off here,
since every new session evicts the one before it. Before, the budget was derived separately and fell to its 8 GiB floor
on 64 GB, on top of the plan, which left the system about 5 GB of the 13
the plan keeps for it and for the page cache the experts stream through.

`--max-seq` therefore decides how many experts stay resident:

| `--max-seq` | one session | slots it takes (of 16 441 at 64 GB) |
|---:|---:|---:|
| 131 072 (the default) | 2.7 GB (4.2 GB with bf16 caches) | about 1 050 |
| 262 144 | 4.9 GB (7.9 GB with bf16 caches) | about 1 900 |

On 64 GB, 131 072 is the sensible choice; 262 144 costs about an eighth of
the slots for a context most conversations never reach. The slot counts are
computed from the sizes, not measured; the load prints the real ones. The
bench (`lily-bench`) reserves no session and plans as before, so its slot
counts stay comparable with the measurements below.

## What was built

`ExpertCache` (`src/qwen4exp/expert_cache.rs`) with `ExpertStore` and
`SlotPolicy` (`expert_store.rs`), engaged when the checkpoint does not
fit physical memory (`hw.memsize` against the resident weights plus 5 GB
of scratch and a reserve of 12 GB or a sixth of memory; the n-gram table
is paged and not counted) or on request (`LoadOptions::expert_slots`,
`LILY_EXPERT_SLOTS`). On this 128 GB machine nothing engages and the
footprint is unchanged. Measured footprint with 16 384 slots: 52.9 GB of
process RSS (45.3 GB slab, 3.1 GB resident weights, about 4.5 GB of
scratch, caches and pipelines); a 64 GB machine therefore gets about
43 GB of experts (15 500 slots, 63%) and keeps about 13 GB free.

1. **A slab of expert slots** in the layout of a layer's stacked experts;
   every MoE layer's `MoeWeights` views the slab and carries a `U32 [E]`
   slot table. The decode gathers and the small-m prefill kernels read
   remapped ids (`moe_remap_slots`); the grouped GEMM's block map
   indirects through the table in `moe_build_blocks`. Without a table the
   kernels are unchanged, and with every expert in the slab the benches
   reproduce the resident digests exactly.
2. **A round trip per cached MoE layer.** After the router's top-k the
   pass signals `routed` with a sequence number and waits on `ready` for
   it before the remap; the cache's service thread spins on the signal,
   reads the routed ids from the shared indices buffer, resolves them
   through the policy and signals. With every expert resident this costs
   7% of decode (96.9 against 104.6 tok/s); a blocking wait in the thread
   had cost 30%. A prefill pass is submitted layer by layer instead, each
   part once its layer's `ready` is signalled, because Metal releases a
   queue wait still blocked 4 s after its submission ("Timed-out waits in
   long prefill passes" below).
3. **Placement by usage.** The ranking from `lily-experts` (`expert-usage.
   json` next to the checkpoint, or `LILY_EXPERT_USAGE`; uniform without
   one) pins the top slices; an LRU region (10% by default, at least a
   layer's worth, `LILY_EXPERT_LRU_SHARE`) takes the cold ones, never
   evicting a slot used at the current sequence number. Misses are read
   with positioned reads straight into the slot, each expert's nine
   regions spread over a pool of 16 threads started with the cache
   (rayon) and the service thread, which reads too ("The read threads"
   below).
4. **Drafts under the cache.** The draft head loads as on the full
   machine. Its own experts are resident, not slots, so a speculative
   step's draft pass pays no handshakes; only the verify pass's 48 do, as
   for a plain token. The verify passes encoded ahead for every possible
   accepted count (`spec.rs`, `prepare_variants`) enqueue their cached
   layers' requests while they are encoded, and the variants that are not
   committed hand them back (`QueuedRequests`, dropped with the `Prepared`
   pass, before the matching one is committed: a committed pass runs its
   first layer and that layer's handshake ahead of its parked wait); without
   that the service thread resolved up to 96 requests per step for passes
   that never ran, which was the 14 tok/s of 2026-09-18 ("Speculative
   decoding under the cache" below).
5. **Live usage, promotion and persistence.** The service thread counts
   every resolved (layer, expert), decode-class passes (up to 256 routed
   ids: a step, a verify pass) apart from prefill. A slice's score is the
   loaded ranking, scaled to what 1 000 decoded tokens' worth of lookups
   would accumulate, as a prior, plus 100 per decode lookup and 1 per
   prefill lookup: prefill routes nearly every expert of a layer once per
   chunk whatever the text, so its counts carry little ranking
   information, while a decode lookup is one token's routing and a decode
   miss stalls that token. Every 480 resolutions (about ten decoded
   tokens, or ten prefill chunks; a pass over the slots costs tens of
   microseconds off the GPU's critical path) the 16 lowest-scored
   unprotected pinned slots are compared with the 16 highest-scored
   resident slots of the LRU region, and a cold expert whose score
   exceeds the pinned one's by half again and by two decode lookups takes
   over its pinned role: no data moves and no table changes, only which
   slot a later miss may evict; the promoted slot is protected from
   demotion for 9 600 resolutions (about 200 tokens). The merged scores
   are written as a `lily-experts` ranking every five minutes and when the
   cache is dropped, to `LoadOptions::expert_usage_out` (the server:
   `expert-usage.json` in the disk-cache directory, `~/Library/Caches/
   lily/sessions/` by default; `lily-bench --expert-usage-out`;
   `LILY_EXPERT_USAGE_OUT` overrides both), and the loader prefers that
   file over the ranking next to the checkpoint, so the next load places
   experts by what this machine actually ran. Every score, the prior
   included, halves each 500 decoded tokens' worth of decode lookups
   (`LiveUsage::HALF_LIFE_TOKENS`, since 2026-10-09; "Usage that decays"
   below), so the placement follows recent traffic; `LILY_EXPERT_HALF_LIFE`
   sets another half-life in decoded tokens, 0 none. `LILY_EXPERT_ADAPT=0`
   keeps the counts but never promotes; the load line says which ranking
   was used, the half-life and where the usage goes, and the cache
   statistics report decode-class misses and promotions.

### Measured

8K real-text prompt (`docs/bench/prompts/p0.txt`), 256 generated tokens,
usage ranking from 40 corpus prompts, 16 384 of 24 576 slots (45 GB, two
thirds), digests identical to the resident path in every row. "Cold"
runs under a 60 GB locked balloon, which leaves the checkpoint's pages
mostly out of the page cache, as a 64 GB machine would; "warm" with the
files cached.

| configuration                                   | prefill tok/s | decode tok/s |
|-------------------------------------------------|---------------|--------------|
| resident, no cache (2 drafts / plain)           | 1 898 / 2 117 | 104.6 / 86.3 |
| all slots served through the protocol, 2 drafts | 1 954         | 96.9         |
| two thirds, warm, 2 drafts, serial loads        | 1 370         | 23.3         |
| two thirds, warm, 2 drafts, parallel loads      | 1 757         | 39.7 to 43.6 |
| two thirds, cold, 2 drafts                      | 944           | 14.2         |
| two thirds, cold, 1 draft                       | 919           | 15.1         |
| **two thirds, cold, plain decode**              | **977**       | **54.6**     |

Misses: 30% of prefill's distinct lookups (every layer touches nearly
every expert per 4 096-token chunk, so the cold third is read once per
chunk, 38 GB per 8K prompt, at the SSD's rate) and 8.8% of decode's. So
a 64 GB machine lands near 950 tok/s prefill and 55 tok/s decode on this
prompt, against 2 100 and 86 here: a 2x prefill and 1.6x decode cost,
not a cliff. The same cold run the next day at commit `38d2642`, after
the prefill kernel work of 2026-09-18 (`docs/performance.md`): 929 tok/s
prefill and 64.4 tok/s plain decode, 13.6% of all lookups missing, no
stale reads; against 2 257 and 87 resident. Cold-run figures move with
what the page cache still holds when the run starts. The initial fill of 45 GB takes about 20 s from the page
cache and would take the SSD's 10 to 15 s cold.

### Live usage

Measured the same way (60 GB balloon, `--memory-gb 64`, the shipped
ranking, `p4.txt`, the prompt with the highest decode miss rate of
p3 to p5 under that ranking), 8K prompt and 1 024 decoded tokens, digests
identical to the resident path in every row (cfce735e33882488), 0 stale
reads:

| configuration                                   | decode misses of 492 960 lookups | decode tok/s |
|-------------------------------------------------|----------------------------------|--------------|
| shipped ranking, promotion off                  | 9 478 (1.9%)                     | 51.5         |
| shipped ranking, promotion on (1 059 promotions) | 8 688 (1.8%)                    | 52.5         |
| the counts that run persisted, promotion off    | 6 232 (1.3%)                     | 57.2         |
| the same, promotion on (224 promotions)         | 6 281 (1.3%)                     | 56.6         |

Within one run promotion takes 8% off the decode misses, which is worth
about the noise band in tok/s; the gain is across runs: a process placed
from what the previous one measured misses a third less at decode and
decodes 11% faster, and prefill misses fall too (33 166 against 36 321
over the run). The persisted counts are the merged scores, so they rank
by this machine's routing rather than by the 40 corpus prompts the
shipped file was measured on.

### Speculative decoding under the cache (2026-10-09)

Until today the draft head stayed unloaded under the cache
(`LILY_EXPERT_CACHE_DRAFTS=1` loaded it) because a step with two drafts
decoded at 14 tok/s cold against 55 plain, which the design notes above
attributed to the passes. The cause was elsewhere: the speculative step
encodes the next verify pass for every possible accepted count while the
verify pass runs and commits only the matching one, and encoding a cached
layer enqueues its request at once. The dropped variants left 96 requests
per step in the queue, which the service thread resolved when the next pass
signalled, reading the ids that pass had just routed under the dropped
pass's layer index: 3 739 decode-class lookups per step instead of about
1 150, a miss rate of 10.8% against 2.1% for plain decode, about 400 loads
per step of experts nobody needed, and evictions of ones that were. The
handle `QueuedRequests` now cancels a dropped variant's requests (queued
ones are removed, the one the thread may hold is skipped, checked again
after the signal that ends its wait). The variants have to be dropped
before the matching one is committed: a committed pass runs its first
layer, handshake included, ahead of its parked wait, and a first version
of the fix that dropped them afterwards still served about 17 000 stale
lookups over a 256-token run when the service thread was fast (everything
resident: 127 102 decode-class lookups against 109 984 for the same
tokens with misses to serve), found by a review on 2026-10-09. With the
order fixed, the same three configurations count 106 359 decode-class
lookups each, 159 of the dropped variants' requests reached the thread
after the queue purge and were skipped there, misses are 3 570 (3.4%), and
the rates are 110.7 through the protocol, 82.4 warm and 57.1 cold (the
warm run's prefill was also slower that time, 1 690 against 2 156 tok/s:
page-cache state). The process peaked at 53 GB of RSS under the 64 GB plan
with the draft head loaded, about 1 GB over the plan's 52 GB target, and at
31 GB under the balloon, which pages lily's own buffers out as well as the
checkpoint's: the cold rows carry that cost too.

Whether drafts pay cold on other text and under sampling (the reviews
asked): the same 64 GB plan and balloon, 8K prompts, 256 tokens, plain
against 2 drafts, `--sample` with the checkpoint's sampler defaults:

| prompt                                          | greedy plain | greedy 2 drafts (accepted) | sampled plain | sampled 2 drafts (accepted) |
|-------------------------------------------------|-------------:|---------------------------:|--------------:|----------------------------:|
| `2026-10-01-http-prefill-gap/bench-prompt.txt`  | 39.7         | 47.4 (86.2%)               | 38.8          | 46.9 (74.3%)                |
| `2026-10-01-long-context-prefill/prompt-mixed.txt` | 45.2      | 53.1 (63.3%)               | 49.2          | 50.4 (52.4%)                |
| `mlx-parity/nested/prompts/n32k.txt`            | 51.4         | 59.0 (75.5%)               | 42.2          | 58.1 (80.6%)                |

Drafts came out ahead in all six, by 2 to 38%, including the sampled
mixed prompt at 52% acceptance, where the two are within the noise band;
so the break-even sits below the acceptance any of these prompts showed.
Decode-class misses with drafts ran 2.0 to 4.9% of lookups against 1.4 to
4.0% plain. One run per cell.

Measured at `13f7b54` plus the fix on `docs/bench/2026-10-01-expert-gemm/
prompt-lily.txt` (8K tokens, 256 decoded, greedy, the shipped ranking, no
n-gram preload), "cold" under a 60 GiB locked balloon; every row's token
digest equals the resident path's for its draft count (the fix changes
which requests are served, not what any pass computes; a token digest
shows the drawn tokens agree, not that logits or states are identical,
and no resident 3-draft row was run):

| configuration                               | plain decode | 2 drafts, before | 2 drafts, after | 3 drafts, after |
|---------------------------------------------|-------------:|-----------------:|----------------:|----------------:|
| resident, no cache                          | 88.1         | 118.7            |                 |                 |
| all 24 576 slots through the protocol       | 78.4         | 111.4            | 111.9           |                 |
| 16 246 slots, checkpoint pages warm         | 66.2         | 38.8             | 87.3            | 92.9            |
| 16 246 slots, cold                          | 50.9         | 13.2             | 58.1            | 57.4            |

Decode-class misses over the 256 tokens (the measured run, without the
warm-up): 2 607 plain, 40 148 with two drafts before the fix, 3 782 after
(3.4% of 109 984 lookups: the draft rows of a verify pass route a little
colder than a decoded token); acceptance 79.3%
with two drafts, 78.9% with three. The first two rows also measure the
handshake alone: about 1.4 ms per trunk pass, 11% of a plain token and 6%
of a speculative step, not the 0.1 ms per layer the design estimated. So
drafts are on under the cache as everywhere (`--mtp-drafts 0` turns them
off) and `LILY_EXPERT_CACHE_DRAFTS` is gone. The draft head's 1.5 GB was
always among the plan's resident weights; its session caches now count in
the server's reserve. Cold, the gain is 15% at two drafts and the third
draft returns nothing; warm, 32% and 40%.

### The re-rank at the end of a prefill: built, measured, dropped (2026-10-09)

The replay of the live policy over logged routing (6 prompts of 8K, 128
decoded tokens each) had the slab ranked by the prompt's own prefill
counts halving decode misses, 5.4 against 10.7 per token. So the re-rank
was built: the service thread counted every routed id of a prompt's
prefill passes, repeats included, and at the prompt's first decode-class
request after a prefill of at least 2 048 tokens moved the whole slab to
that prompt's ranking (its counts, ties by the live score). Target experts
already in the LRU region took pinned roles from slots whose experts left
the target, with no data moved; the rest were read by four background
threads into slots taken out of service one at a time, unmapped so that no
miss could take them, and an expert routed before its read landed came in
through an ordinary miss, which the finished read deferred to. Promotion
was biased toward the prompt, with what is persisted left unbiased. It was
measured with `lily-bench --warmup-text` (the warm-up prefills a
different 8K text, so the measured prompt arrives at a cache placed for
another one, as a server's next request does, and its own re-rank and
background reads fall inside the measurement), the 64 GB plan, the
mixed-text 8K prompt, greedy. Token digests were identical with the
re-rank on and off in every pair, plain and with two drafts.

| decoded tokens, cache state | decode misses off | on | change |
|-----------------------------|------------------:|---:|-------:|
| 128, warm                   | 1 347             | 827   | -39% |
| 512, warm                   | 4 871             | 3 306 | -32% |
| 512, cold                   | 4 871             | 3 456 to 3 462 | -29% |
| 512, cold, 2 drafts         | 5 276             | 3 989 to 4 004 | -24% |

| 512 tokens, cold (60 GiB balloon) | off | on |
|-----------------------------------|----:|---:|
| plain decode, tok/s               | 50.1, 50.6 | 53.9, 53.0 |
| 2 drafts, tok/s                   | 58.3, 55.0 | 64.1, 62.2 |

About 2 150 to 2 500 background loads per prompt (about 400 role swaps),
all landed within the first 128 tokens; 16 read threads instead of 4
changed nothing. Prefill moved within the cold runs' noise (the pinned set
the measured prompt starts from is the warm-up prompt's). Promotion does
not undo the re-rank: with the bias at 0, 1, 4 or 16 times the prior's
weight, or promotion off, the 512-token warm misses stay between 3 325 and
3 486 against 4 871 off. The first version re-ranked the pinned set only
(3 432 misses at 512, 933 at 128); taking the LRU region's contents into
the target as well brought the 128-token figure to 827. The replay's
halving is not reached: it counted six prompts, and its policy started
each prompt from a fresh fill.

In agent use the trigger is the question, not the gain: the server log
shows that 13% of requests prefill 2 048 or more new tokens (median 87 to
224), since the session cache supplies the prefix. The re-rank therefore
acts at a session's start and on large tool results; between them the
live promotion is what adapts. "Replaying agent sessions" below measures
that on recorded traffic, where it does not pay, so it was removed again.

### Replaying agent sessions (2026-10-09)

The bench continues the text it prefilled, which is the case most
favourable to a ranking learned from the prefill. Agent traffic is not
that: a long prefill there is mostly a tool result (a file, test output),
and the decode that follows is the model's own reasoning and tool calls.
To measure the cache on real traffic without a 64 GB machine or hours of
cold runs, two tools replay recorded sessions:

- `lily-trace-routing --model <checkpoint> --tokens <file.u32>...
  --out-dir <dir>` prefills each recorded session (little-endian `u32`
  token ids, the whole conversation as the server's disk tier stores it in
  a session's `meta.json`) resident and writes which experts every
  position of every MoE layer routed to, `[layers][tokens][top_k]`. About
  2 000 tokens per second on the M5 Max. Prefill routes a position as the
  decode step would up to rounding. `--predict` also enables the
  cross-layer prediction log ("Cross-layer prediction measurement" below)
  and writes predictor B's top 20 for lookaheads 1 and 2, ordered by
  score, to `<stem>.pred.u16`.
- `lily-cache-replay --usage <ranking> --trace <stem>...` replays them
  through the cache's own policy code (`qwen4exp::cache_replay`:
  `SlotPolicy`, `LiveUsage`, promotion), host only, sweeping the LRU
  share and, with `--prefetch off,1:10,2:10:2,...`, a decode-time
  prefetch (lookahead, top, optional reads per layer). `<stem>.json` gives
  the requests as `[p0, p1, p2]`: `p0..p1` prefilled (the suffix after the
  session's cached prefix), `p1..p2` generated. Sessions replay in order
  through one cache, as one server serves them; `--fresh` starts each from
  the loaded ranking.

The requests come from the chat template: every `<|im_start|>assistant\n`
ends a request's prefill and the matching `<|im_end|>` its generated
span, so a request prefills exactly what the server prefilled (tool
results, the user's message) and decodes what the model sampled. Decoding
is replayed one token per pass; a speculative verify pass also routes its
rejected drafts, which a trace does not hold.

Measured on seven opencode sessions of 2026-10-07 on a client codebase
(34 000 to 119 000 tokens each, 520 000 in all, 201 000 of them
generated with the server's sampling, 320 000 prefilled in 456 requests),
16 246 slots, the shipped ranking, one cache across the sessions in the
order they were served:

| LRU share | pinned | decode misses per token | prefill misses per token | all reads against 10% |
|----------:|-------:|------------------------:|-------------------------:|----------------------:|
| 3.5%      | 15 670 | 12.12 | 3.57 | +4.8%  |
| 5%        | 15 434 | 11.56 | 3.67 | +2.4%  |
| 10%       | 14 621 | 10.66 | 3.98 | 0      |
| 20%       | 12 997 | 10.16 | 4.49 | +1.9%  |
| 30%       | 11 372 | 10.10 | 4.92 | +5.6%  |
| 40%       |  9 748 | 10.28 | 5.29 | +10.1% |

A larger LRU region takes up to 5% off the decode misses and adds up to
32% to prefill's; a decode miss and a prefill miss cost about the same
SSD time (a decode miss waits for one 2.76 MB read, a prefill pass reads
hundreds at the drive's bandwidth), so the total is what to minimise, and
it is flat between 5 and 20% with its low at 10%, the default, which
stays. Replayed in another session order the figures move by about 1%
(10.78 and 4.07 at 10%) and the verdict holds. The re-rank, replayed in
that order with its background loads landing at the measured rate, added
decode misses at every share: 11.12 against 10.78 per token at 10% (+3%),
per session from 0 to +23%, with a 3% gain in prefill misses; 33
re-ranks fired, about 88 000 background loads. A ranking taken
from the prompt helps only where the output continues the prompt (long
document summaries, continuation), not an agent, whose most common use
here does not justify the code; it was removed.

The traces themselves stay out of the repository (they are a client's
code and conversations); only these counts are recorded.

### Usage that decays, the starting ranking, a timed replay (2026-10-09)

Until 2026-10-09 the live counts never decayed: after a few sessions the
pinned set followed everything since the load, and the replay showed that
restarting each session from the shipped ranking (10.09 decode misses per
token) beat one cache carried across all seven (10.66). `lily-cache-replay
--half-lives` sweeps a half-life on every score
(`LiveUsage::with_half_life`); `--fresh --persist` starts each session from
the usage the previous one's cache would have persisted, as a server
restarted between sessions does. `--timed` adds a timeline
(`cache_replay::SsdTiming`): a decoded token's layers take 0.315 ms each
with nothing missing (the warm token under the cache over 48 layers), a
layer's request comes halfway in, and the SSD reads one expert at a time
in 0.42 ms (2.76 MB at about 6.5 GB/s), demand reads before queued
prefetches; a layer waits for its reads. Without prefetch it charges 4.41
ms of stall per token, against the 4.5 ms by which a cold token measured
slower than a warm one on the bench: the model is calibrated where it can
be. A prefetch still queued when its layer has passed is cancelled and its
slot emptied, as a real loader would. Prefill is not timed, and decoding
is one token per pass.

The seven agent sessions, 16 246 slots, one cache across them, per decoded
token:

| half-life (decoded tokens) | LRU 5% misses | LRU 10% misses | stall at 10% | LRU 20% misses |
|---------------------------:|--------------:|---------------:|-------------:|---------------:|
| none                       | 11.56         | 10.66          | 4.41 ms      | 10.16          |
| 64 000                     |               | 10.61          | 4.39 ms      |                |
| 16 000                     |               | 10.10          | 4.17 ms      |                |
| 4 000                      |               | 9.16           | 3.78 ms      |                |
| 1 000                      | 8.48          | 8.38           | 3.46 ms      | 8.69           |
| 500                        | 8.36          | 8.29           | 3.42 ms      | 8.65           |
| 250                        | 8.74          | 8.53           | 3.52 ms      | 8.73           |
| 125                        | 11.35         | 10.53          | 4.35 ms      | 9.99           |
| 60                         | 14.02         | 12.75          | 5.26 ms      | 11.87          |

A half-life of 500 tokens takes 22% off the decode misses and about 1 ms
off a cold token's stall, in every session (-13 to -31%), with prefill
misses within 5% either way; it is the default since. Between 250 and
1 000 the figures are flat; below 125 the cache forgets what it needs
again within a few hundred tokens. Promotions rise from 4 600 to about
250 000 over the 201 000 tokens, which costs nothing measurable: a
promotion swaps two slots' roles, no data moves, and a pass runs every
480 resolutions either way. With decay the 10% LRU share stays: 5% saves
about 8% of the prefill misses and loses 1% at decode, about 3% of all
reads, within what the sessions vary by.

The starting ranking matters less than the decay. What is persisted is the
decayed score, so the file a restart loads reflects the last thousand or
two decoded tokens, not the history since the first load; starting each
session from the previous one's persisted usage gives 8.33 misses at a
500-token half-life, against 8.29 for one cache never restarted, 9.02 at
4 000 tokens and 9.79 without decay.

The half-life is counted in decode lookups, a token's worth being
`top_k` per layer, and the replay decodes one token per pass. Under
speculative decoding a verify pass resolves the distinct experts of its
three rows, about 22.9 per layer, for about 2.6 accepted tokens at agent
traffic's 79% acceptance: about 8.9 lookups per accepted token instead of
10, so 500 tokens' worth lasts about 560 accepted tokens, inside the flat
range. The decay was chosen on the replay alone; no live A/B against
`LILY_EXPERT_HALF_LIFE=0` on recorded traffic has been run.

### The read threads (2026-10-09)

The service thread used to start and join a thread per slice of a
resolution's reads (up to 16, nine per missed expert) every time a layer
missed, about eight times per decoded token. A persistent pool replaced
that. Under the 64 GB plan with the 60 GiB balloon, an 8K mixed-text
prompt, 512 greedy tokens, tokens identical in every run, decode tok/s:

| loader                                              | cold, plain | cold, 2 drafts |
|-----------------------------------------------------|------------:|---------------:|
| threads started per resolution (before)             | 50.4, 49.4  | 57.0, 56.6     |
| rayon pool, service thread reads too (shipped)      | 50.4, 49.7  | 58.4, 57.9     |

Starting threads cost less than expected: plain decode does not move, and
with drafts, whose verify passes miss several experts per layer, the pool
gains 1.5 to 3.5% in this pair. Two runs per cell sit inside the decode
noise band of single runs, but the gain held in three separate A/Bs that
evening (a hand-written pool with the same scheme: 60.3 and 60.1 against
58.2 and 58.5, then 58.5 and 58.4 against 56.9 and 57.7). The initial fill
of 16 246 slots took 4.7 to 6.2 s, against 5.4 to 6.5 s before. How the pool starts its work mattered: a rayon parallel
iterator over the reads measured 1 to 2% slower than the old threads, and
a broadcast to every pool thread no faster, both because the service
thread waits for pool threads to wake before the first read starts; with
the service thread claiming reads itself from the same counter as the
pool jobs (`expert_cache::run_claimed`), the first read starts at once. A
hand-written pool doing the same measured the same.

### Where a miss's time goes (2026-10-10)

`lily-ssd-probe` read one expert in 0.51 ms and k at once in about
0.28 + 0.225 k ms, yet a missed layer cost the server about 0.8 ms at
64 GB (1.39 misses per missed layer) and 1.7 ms at 40 GB (2.86). Measured
with `LILY_EXPERT_TRACE=<stem>` (per resolution: request taken, GPU signal
seen, ids looked up, reads done, `ready` signalled; per region read: start,
end, thread; written as `<stem>.layers.csv` and `<stem>.reads.csv`, the
first 524 288 resolutions and 1 048 576 reads; the trace costs a clock read
per mark and a lock per read, so it perturbs what it measures a little), q4-xl,
8K prompt, 1 024 sampled tokens, plain decode, every run on a fresh APFS
clone of the checkpoint with `LILY_EXPERT_NOCACHE=1`. Per token at 64 GB
(15 736 slots, 19.2 misses and 14.0 missed layers per token) against the
reference with 23 115 slots (0.7 misses):

| ms per token                                   | reference | 64 GB before | 64 GB after |
|------------------------------------------------|----------:|-------------:|------------:|
| holds of missed layers (reads, lookups)        | 0.44      | 9.01         | 8.36        |
| GPU between handshakes, layers 1 to 47         | 12.03     | 13.85        | 12.20       |
| layer 0 to 1 (the parked pass's host release)  | 1.03      | 1.77         | 0.44        |
| token boundary and hit holds                   | 1.55      | 1.61         | 1.58        |
| **total**                                      | **15.05** | **26.24**    | **22.57**   |

The hold itself was close to the probe: 0.64 ms per missed layer against
0.59 predicted. The rest went to three things, each fixed:

- **The GPU slows down while it waits.** Idle at the handshake for half a
  millisecond, the GPU drops its clock and the next layers run slower,
  hits included: 0.08 ms more GPU time after a missed layer at 64 GB, and
  at 40 GB every layer's GPU time went from 0.23 to 0.68 ms (26 ms per
  token). Holding every layer 1 ms with nothing to read, at the reference
  configuration, reproduced it (0.23 to 0.80 ms per layer), so it is the
  idle time, not the reads; reads through a heap bounce buffer instead of
  straight into the slab changed nothing. Now the service thread has a
  warm pass committed on a second Metal context for the length of each
  resolution with misses: one SIMD group polling a host-written word,
  capped at 65 536 polls (about 7 ms of GPU time when nobody releases it,
  `a_warm_pass_nobody_releases_ends_at_its_cap`). It ends about 0.25 ms
  after the release (the GPU sees the host's write late), and that overlap
  costs 0.02 ms per missed layer against the 0.08 (64 GB) to 0.6 ms (40 GB)
  per missed layer it saves. A warm pass running all the
  time instead slowed 64 GB to 41.7 ms per token: two queues with work
  time-slice in milliseconds. `LILY_EXPERT_WARM=0` turns it off. The warm
  thread builds its context while the service thread goes on (resolutions
  before it is ready are not warmed). The word it polls lives in a page
  the service thread's `Warmer` owns, not the warm thread, so a warm pass
  that fails (which turns warming off for the cache's life) or panics
  cannot leave a release writing to freed memory.
- **The pacer overslept.** A paced wait sleeps until an average of past
  pass lengths minus 1.5 ms, and under misses a token takes 18 to 32 ms
  (p10 to p90 at 64 GB): the host woke late for about half of them and the
  parked next pass waited at its first layer. A pass now carries a
  milestone, its last cache handshake (`ComputePass::set_milestone`): the
  paced wait blocks on `ready` reaching it and paces only the rest (a pass
  that ends without reaching it, on an error path, is not paced and does
  not count towards the estimate). Same as
  no pacing at all, which measured 24.49 against 26.24 at 64 GB and 14.45
  against 15.05 at the reference.
- **The reads started late.** The nine reads of a missed expert started
  one by one, 8 microseconds apart as pool threads woke, after 18
  microseconds of the service thread spawning eight jobs; the down
  projection's codes, third in file order, started 61 microseconds in and
  finished last in 78% of single misses. Now one job spawns the others, so
  the service thread reads at once, and every expert's three code regions
  are claimed before any scales or biases (`expert_cache::claimed_read`),
  up to eight experts; from 16 on, and for the initial fill (3.4 against
  3.8 s), each expert's regions together measured faster, so larger
  batches keep that order. In the server the hold per missed layer went
  from 545 to 496 microseconds for one miss, 794 to 729 for two, 1 031 to
  958 for three, 1 265 to 1 203 for four (medians, 64 GB, with the victim
  change below). The probe, A/B pairs on one clone: one expert 0.57 to
  0.44 to 0.48 ms, two 0.93 to 0.77, four 1.44 to 1.25 in the first two
  pairs; later that day, with the drive slower (one expert 0.66 to 0.68 ms
  before), one expert 0.59 to 0.64, two 0.95, four and eight no better.
  Splitting
  the code regions into smaller reads went the other way (pieces of 256,
  128 and 64 KiB: 32 experts at 8.3, 7.0 and 4.9 GB/s against 11.5).
- Smaller: finding a miss's victim scanned every slot, about 9
  microseconds per miss; the LRU region's slots are now kept ordered by the
  eviction rule (`SlotPolicy`, a `BTreeSet`), 1 to 2 microseconds.

Decode ms per token, fresh clone per run, baseline (`ee40c24`) against
all four, alternating:

| configuration                         | before          | after           | digest       |
|---------------------------------------|-----------------|-----------------|--------------|
| reference, 23 115 slots               | 14.58           | 14.36           | equal        |
| 64 GB, 15 736 slots                   | 26.24, 26.48    | 22.83, 22.92    | equal        |
| 64 GB, warm pass off                  |                 | 23.67           | equal        |
| 40 GB, 6 416 slots                    | 82.85, 82.99    | 53.18, 47.2     | see below    |
| 64 GB, 2 drafts, 512 tokens (tok/s)   | 43.1            | 58.3            | equal        |

### Timed-out waits in long prefill passes (2026-10-10, fixed)

At 40 GB a prefill pass of 4 096 tokens waits about 85 ms per layer and
5.6 s in all, and from about 4 s into the pass the GPU stopped waiting for
`ready`: the service thread found `routed` one to six layers ahead of the
request it was serving, so those layers ran with experts that were not
loaded. It happened in every prefill pass measured at 40 GB (57 to 76
timed-out waits per run of four passes), the prefill's distinct lookups
varied run to run (44 094 to 44 663 against 44 561 everywhere else), and
so did the drawn tokens; the 40 GB rows of the table above are not one
token sequence.

**The limit.** Metal's `MTLEvent` page says a wait that times out on an
`MTL4CommandQueue` "unblocks subsequent work in the queue" (on an
`MTLCommandBuffer` it fails the buffer with a timeout error). Neither the
page nor the headers give the limit, and there is no property to read or
raise it and no status that reports it: the commit feedback has no error.
`metal::tests::queue_wait_timeout_probe` measures it without the model: a
pass of (wait `e = k`, work, signal `f = k`) triples, the host raising
`e` on a schedule and a thread polling `f` for when the queue passed each
wait.

- A wait nobody signals is released 4.00 s after the submission that
  encoded it (4.002 to 4.003 s, waits of 1 to 5 s).
- Waits of 140 ms with 4 to 40 ms of GPU work between them: from 4.03 s
  after the submission every wait was released, most at once.
- The clock is per submission, not per wait or per queue. Of two
  submissions on one queue, committed at 0 and at 3.34 s, the first's
  waits were released from 4.03 s and the second's from 7.35 s, its own
  commit plus 4.01 s. The same waits in submissions of one or five, each
  committed at most 1 s before its first wait, held for 8 s.
- Some short waits are spared, by a rule the probe did not pin down:
  waits of 100 ms with 4 ms of work, or 10 ms with 40 ms, held for 7 to
  8 s in one submission, and waits of up to 210 ms without any work for
  8 to 9 s (230 ms and more were released at 4.0 s; mixed schedules
  without work were released 3.4 to 3.75 s after the first long wait).
  The model's prefill waits of 75 to 130 ms were released, so nothing is
  built on this.
- A submission made after released waits waits normally again: the queue
  is not left broken.

**What waits, and for how long.** Every queue-level wait in lily is on a
host-signalled event:

| wait                                    | passes                         | blocked for                                                    | past 4 s?                          |
|-----------------------------------------|--------------------------------|----------------------------------------------------------------|------------------------------------|
| `ready`, each cached MoE layer           | prefill (more than 4 rows)     | the layer's misses: 85 ms a layer at 40 GB, 5.6 s a pass         | yes at 40 GB; 64 GB at 3.8 to 4 s   |
| `ready`, each cached MoE layer           | decode, verify (up to 4 rows), short prefills | a few ms a layer; at most 40 experts a layer, about 9 ms when all miss, 0.45 s a pass | only with a drive or host stalled for seconds |
| `StepSync`, a parked pass's first input  | decode, verify, batched rows   | the pass before it plus the host's per-token callbacks (ms; the token stream's channel is unbounded, so a slow client does not hold them) | only with a stalled host; not watched, below |
| none                                     | the warm pass, blits, vision, everything resident | (the warm pass polls a host-written word, capped at 65 536 polls) | no |

The draft head's experts are resident, so draft passes have no cache
waits; without the cache the only wait is the parked one.

**The fix.** A gated pass (`ComputePass::gate_host_waits`) is submitted in
parts that begin at its waits: the commit submits it up to the first wait,
and a feeder thread submits each later part only once the host has
signalled that part's wait (polling for 0.2 ms, then blocking). The queue
never blocks on a host event, so nothing can time out however long the
reads take; later submissions on the context wait until the feeder has
submitted the last part, which keeps the queue's order. A feeder that
waits 30 s at one gate (`GATE_PATIENCE`; the slowest legitimate wait is
one layer's misses, 0.16 s on average for 8 192 rows at 40 GB) gives up and
faults the context, so a host that never signals fails the request and
reloads the engine instead of hanging it (a queue wait would have been
released after 4 s, with the wrong experts); one that panics faults the
context too, and when no feeder thread can be started the committing
thread feeds the pass itself. `encode_batch`
gates every prefill pass of more than 4 rows (`GATED_PREFILL_ROWS`, a
verify pass's size); without the cache a prefill has no waits and is
submitted as before. A part costs the host's wake-up and commit after
`ready` instead of the GPU seeing the signal (`event_handoff_latency`:
0.21 against 0.09 ms), about 5 ms per 48-layer pass. Decode and verify
passes keep their queue waits, which the table bounds well under the
limit; gating them would cost those 5 ms per token.

**One that happens anyway fails the request.** Every pass's commit
feedback now checks its waits on `ready`, a watched event
(`MetalContext::new_watched_event`, `metal::Watch`): the command buffer
behind a wait must not have started more than 0.1 ms before the host
signalled the event (each `SharedEvent::signal` of a watched event logs its
value and time when it raises the event, the last 4 096), and the event
must not be below the value when that command buffer has completed. A
wait that did not hold records a context fault; the pass's wait reports
it before anything reads the pass's outputs, because a pass with waits
takes its waits' feedback first. The parked pass's `StepSync` wait is not
watched, under the cache or without it: it blocks for milliseconds, and
without the cache passes are submitted exactly as before (no log, no
extra command buffer). An empty command buffer behind the last watched
wait has that feedback arrive as soon as the wait is passed: waiting for
the feedback of the command buffer that follows it, which in a decode
pass runs to the end, cost 0.15 ms per token at the reference. The fault
fails the request (503) and the server drops the engine without spilling
its sessions and loads it again, as after any Metal error. That over
recomputing the pass: the pass read slots the service thread was
rewriting and wrote attention and recurrent state from the wrong experts,
lily keeps no copy of the in-place recurrent state to recompute from, and
a decode-class pass gets there only if the drive or the host stalled for
seconds, so a reload is proportionate. The work behind waits the host released
started 5 microseconds to 3 ms after its signal, 0.1 ms in the median,
never before it (`waits_the_host_releases_are_not_reported_as_timed_out`
prints it); the 0.1 ms margin is for the two clocks (the GPU's timestamps
are converted to the host's), since a false fault costs a reload, and a
real timeout comes seconds after the submission. A real timeout is caught
(`a_real_timed_out_wait_faults_the_context`, ignored, 4.5 s), and a gated
wait holds past 4 s (`a_gated_wait_holds_past_the_queue_wait_limit`,
ignored). A GPU signal after every wait, which the host could
check before it signals, measured 0.22 ms per 48 waits
(`witness_signal_cost_probe`) and was not used. The service thread's
count of timed-out waits stays as a diagnostic.

**Measured**, q4-xl, 8K prompt, 1 024 sampled tokens, a fresh APFS clone
per run with `LILY_EXPERT_NOCACHE=1`, before (`4740cab`) and after
alternating; times in s for the prefill and ms per token for the decode:

| configuration                    | prefill before                | prefill after          | decode before                      | decode after               | digest                          | timed-out waits |
|----------------------------------|-------------------------------|------------------------|------------------------------------|----------------------------|---------------------------------|-----------------|
| 40 GB, 6 416 slots               | 11.13, 11.24                  | 12.03, 12.06           | 52.98, 58.89                       | 55.37, 59.16               | each run its own; `5f98c38bea6bc0e0` | 57, 68; 0, 0    |
| 40 GB, one pass of 8 192         | 6.04                          | 7.72                   | 52.95                              | 55.51                      | `fbeeab98f092b3f4`; `5f98c38bea6bc0e0` | 48; 0           |
| 64 GB, 15 736 slots              | 7.47, 7.39, 7.42, 7.50, 7.55, 7.46 | 7.35, 7.35, 7.34, 7.33 | 22.61, 22.60, 22.60, 22.64, 22.58, 22.65 | 22.69, 22.70, 22.55, 22.64 | `5f98c38bea6bc0e0`              | 0               |
| reference, 23 115 slots          | 4.38, 4.40, 4.45, 4.42, 4.45  | 4.38, 4.46, 4.43, 4.43, 4.44 | 14.42, 14.40, 14.36, 14.38, 14.47 | 14.39, 14.64, 14.39, 14.43, 14.33 | `5f98c38bea6bc0e0`        | 0               |
| resident                         | 3.29, 3.25, 3.34              | 3.32, 3.28, 3.04       | 12.73, 12.66, 12.78                | 12.74, 12.66, 12.80        | `5f98c38bea6bc0e0`              |                 |

At 40 GB the prefill's lookups are now 44 561 and the tokens are the
reference's in every run, also from a single 8 192-token pass of 7.7 s.
The prefill takes 0.8 to 0.9 s longer there because it now waits for every
layer's reads, which the released waits had let the GPU skip; at 64 GB,
where nothing was released, it is 1.5% faster gated (7.34 against 7.46 s),
and at the reference and resident it is unchanged. The 40 GB decode times
are not comparable before and after (different tokens). Runs made in
slow spells of the machine, both binaries alike, are left out: two
reference pairs (18.6 to 19.7 ms per token) and two resident pairs (17.9
to 18.0 and 14.6 to 14.8). The 4-layer checkpoint's
`the_expert_cache_generates_what_the_resident_model_does` checks gated
prefill and queue-waited decode under the cache token for token against
the resident model.

The timing-dependent digest that "Prefill chunks of 8 192 tokens" below
reports for 8 192-token passes under memory pressure is most likely this
as well: passes that slow wait past 4 s. It was not rerun under that
balloon.

### Cross-layer prefetch on agent sessions, and what it costs (2026-10-09)

The same seven sessions with predictor B's picks logged at every position
(`lily-trace-routing --predict`; the routing written alongside matched the
earlier logs bit for bit), replayed at 10% LRU with a decode-time prefetch:
after a token's request for layer `l`, B's best picks for layer `l + d`
that are not resident are read into LRU slots. Reads land instantly in the
replay. Per decoded token:

| prefetch                       | decode misses | layers with a miss | reads | useful |
|--------------------------------|--------------:|-------------------:|------:|-------:|
| none                           | 10.66         | 7.76               |       |        |
| B top 5, one layer ahead       | 8.42          | 6.54               | 4.1   | 2.3    |
| B top 10, one layer ahead      | 5.18          | 4.13               | 15.0  | 5.7    |
| the same, 2 reads per layer    | 5.55          | 4.30               | 13.7  | 5.3    |
| the same, 1 read per layer     | 6.46          | 4.83               | 10.8  | 4.3    |
| B top 20, one layer ahead      | 2.87          | 2.20               | 57.3  | 8.4    |
| B top 10, two layers ahead     | 7.73          | 5.83               | 18.1  | 3.7    |
| B top 20, two layers ahead     | 6.10          | 4.53               | 62.4  | 6.2    |

At 20% LRU the figures move by up to 0.4 per token. Agent traffic keeps
most of what the bench prompts showed (top 10 one ahead: -51% against
-70% there). The misses are spread over the layers evenly (0.13 to 0.30 per
token in every layer, the eight best layers hold a quarter of the useful
reads), so predicting only some layers would not change the ratio below.

A predictor without GPU work was tried as well: transition counts
`P(f at layer l + d | e at layer l)` learned from the routing of the other
six sessions (`cache_replay::Transitions`, `--predictor transitions`),
which the service thread could evaluate at layer `l`'s handshake. It
predicts the popular experts, which are resident: top 10 one ahead reads
1.2 experts per token of which 0.18 are routed to; 96 wide, 73 reads for
2.9 useful, and the wasted reads evict as much as they save (10.28 misses).

B is a GPU computation: per layer, the next layer's MoE read-in and its
router. Its cost, measured with a temporary switch that encodes the routing
log plus one predictor alone (`docs/bench/2026-10-09-prefetch-cost/`, not
in the tree), `lily-bench`, q4, resident, a 1K real-text prompt, 256 greedy
tokens, two runs each, identical tokens in every run:

| encoded in the decode pass | plain step | plain tok/s | 2 drafts tok/s |
|----------------------------|-----------:|------------:|---------------:|
| nothing                    | 10.6 ms    | 92.3        | 104.5          |
| the routing log            | 10.6 to 10.7 | 89.5 to 92.6 | 104.5        |
| log + B one ahead          | 12.0 ms    | 82.2        | 95.7           |
| log + B two ahead          | 12.1 to 12.6 | 77.8 to 80.8 | 95.3         |
| log + A one ahead          | 11.3 ms    | 86.4        |                |

So B one layer ahead costs 1.35 ms per token (8.5% with two drafts), B two
ahead 1.7, A one ahead 0.7. Against that, a useful prefetch saves at most
the 0.4 ms a cold read costs, and only the part of it that lands before
the layer's handshake: B one ahead is known after layer `l`'s MoE and needed
at layer `l + 1`'s router, a mixer's time later (a read takes about as
long as a whole layer), so at best 5.7 x 0.4 = 2.3 ms per token if every
read landed, realistically under its 1.35 ms. B two ahead has the time but
saves at most 3.7 x 0.4 = 1.5 ms for its 1.7. A is known at layer `l`'s
handshake, a layer before it is needed, but predicts worse (on the bench
prompts it was worth about two thirds of B), roughly 1.1 ms for 0.7, before
its ten wasted reads per token compete with the demand misses for the SSD.
Cross-layer prefetch is therefore not built: on this model it would move
cold decode by -5 to +2%, which does not carry an asynchronous loader in
the service thread. It would reopen with a predictor that costs a fraction
of a router pass (fused into a level the pass already has), or a drive fast
enough that a read fits in a mixer's time.

### A whole-token predictor from the hidden state (2026-10-09)

The cross-layer predictors above pay per layer and leave a read less than
a layer's time to land. A predictor run once per token, at the end of the
pass before, avoids both: the deeper a layer, the longer its read has.
Its input has to carry information neither the token id nor the recent
routing does (both measured above), so it is the final hidden state of
position p, the LM head's input, and optionally token p+1's id; its
output is position p+1's routing at every layer, 48 x 512 probabilities.

`lily-trace-routing --hidden` writes every position's final hidden state
(`Qwen4ExpModel::enable_hidden_log`: prefill runs the final read over all
rows, where normally only the last row's feeds the head). Hidden states
can largely be inverted back to tokens, so they stay outside the
repository with the traces. Two checks: the
routing logged alongside matched the earlier logs bit for bit in all seven
sessions, and a 3 000-token prefix dumped with prefill chunks of 4 096 and
1 024 tokens gave identical rows (cosine 1.0000, chunk boundaries
included; neighbouring rows 0.45). `tools/bench/expert_predictor.py`
trains a two-layer MLP (layer norm, 2 560 to 1 024, GELU, 1 024 to
24 576; 28M parameters) on six sessions' positions (about 450 000) for
two epochs, binary cross-entropy (the script's defaults), and predicts the seventh's decoded
positions, for each of the seven; `lily-cache-replay --prefetch 0:TOP
--pred-width 40`
replays the top predictions per layer as reads issued at the token's
start. Top-10 overlap with the routing actually taken: 46 to 51% per
held-out session (B one layer ahead, same token: 79%).

Timed replay of the seven sessions, 500-token half-life, per decoded token:

| prefetch                                    | decode misses | reads | stall    |
|---------------------------------------------|--------------:|------:|---------:|
| none                                        | 8.29          |       | 3.42 ms  |
| B one layer ahead, top 10                   | 3.70          | 9.9   | 3.14 ms  |
| B two layers ahead, top 10                  | 5.48          | 12.5  | 3.11 ms  |
| token id table at token start, top 10       | 8.01          | 9.0   | 3.62 ms  |
| MLP at token start, top 10                  | 7.38          | 2.9   | 3.18 ms  |
| MLP at token start, top 20                  | 6.69          | 7.8   | 3.10 ms  |
| MLP at token start, top 40                  | 5.96          | 18.7  | 3.30 ms  |
| MLP with the next token's id, top 20        | 6.58          | 8.4   | 3.07 ms  |

Timed, B looks worse than its miss count suggests: 4.5 of its useful
reads per token land after their layer asks (a mixer's time against a
0.42 ms read), so it saves 0.3 ms for 1.35 ms of GPU work. The MLP is the
first predictor whose reads mostly land: 0.32 to 0.35 ms saved per token,
about 10% of the stall and 1.7% of a cold token. Weighting the loss 8:1
toward the slices the loaded ranking leaves unpinned changed nothing (3.11
ms; the pinned set drifts under decay), nor did a per-layer read budget
(3.09 to 3.26 ms), nor, on one held-out session, twice the width and four
epochs (overlap +2.4 points, stall 3.10 against 3.12 ms). Wider predicted
sets cut more misses but queue more reads ahead of the demand misses, and
the stall rises again past about 8 reads per token.

What building it would cost: the model at the end of every pass, next to
the LM head that reads the same hidden state (28M parameters, about 56 MB
per token in bf16, 14 MB at 4 bits, a fraction of a millisecond, no new
level on the trunk), and a background loader in the service thread with
in-flight tracking and cancellation (the dropped re-rank had most of it,
`docs/bench/2026-10-09-expert-rerank/`, ignored). Under speculative
decoding a verify pass routes three positions and the model predicts the
first, so the shipped mode would gain less than the replay's 1.7% of a
plain cold token. That is not worth the loader today; it would
reopen with a predictor that sees more than one position back (the draft
head's states, for the verify pass's other rows) or a drive with lower
read latency, where the late reads stop dominating.

### Predictors inside the token, and a drive measured (2026-10-09/10)

The whole-token predictor above sees only the previous position. A pass
knows more as it goes: after layer K, the stream that layer K+1's MoE reads
already says much about the layers after it. Tools for that, all offline:

- `lily-trace-routing --cuts 3,7,15,23` writes, for every position, the
  MoE read-in after each listed layer (`Qwen4ExpModel::enable_trunk_log`,
  `<stem>.trunk.bf16`, `[cuts][tokens][hidden]`); `--router-top W` writes
  each layer's top W router picks with their scores. The log changes no
  logits, draws or routing, and its rows do not depend on the chunking
  (cosine > 0.999).
- `lily-render-chat --model <checkpoint> --out-dir <dir> <chat.json>...`
  renders OpenAI-style chat requests through the server's own template
  (`serve::transcript`) into `<stem>.u32` and `<stem>.json` in the
  disk tier's layout, so public agent trajectories and generated runs
  replay like recorded sessions.
- `lily-cache-replay` replays several prediction planes per position
  (`--planes N`, `<stem>.pred.u16`): the token start's (`--prefetch
  0:TOP`), then one per cut (`0:TOP@3+7+15+23`), each issued when its
  layer ends and predicting the layers after it; a trailing `r` lets a cut
  cancel the queued reads for its layers that it does not pick.
  `--predictor oracle` replays the routing itself, the ceiling under the
  same timing; `--miss-log DIR` writes every decode miss; `--speculate
  DRAFTS:ACCEPT` replays verify passes of the pending token and its
  drafts with a seeded acceptance (token-start predictions cover the
  first row, cuts every row; stall per accepted token). A plane's width
  is explicit (`--pred-width`, 20 for `lily-trace-routing --predict`'s
  two planes): a file whose length does not match planes, layers,
  positions and width is refused.

The predictors are MLPs trained offline (PyTorch) on a 3-million-position
sample of agent traffic, one to two epochs, about nine minutes per model.
Public agent trajectories (`nvidia/Open-SWE-Traces` and
`nebius/SWE-rebench-openhands-trajectories`, CC-BY-4.0) and opencode runs
of the hosted Qwen3.8-Flash on generated tasks render into traces with
`lily-render-chat`. An input of token ids would record which ids
occurred, so it is trained on public data only. On the q4-xl checkpoint
(routing overlaps q4's by only 80%, so predictors are per checkpoint),
replayed on sessions not trained on, timed with the serial drive model
below, per decoded token:

| prefetch | 64 GB stall | 40 GB stall |
|----------|------------:|------------:|
| none | 4.14 ms | 13.21 ms |
| previous hidden state at token start, top 20 | 3.42 ms | 11.51 ms |
| + cuts after layers 3, 7, 15, 23, replacing | 3.09 ms | 10.72 ms |
| the same, top 12 | | 10.60 ms |
| the routing itself (ceiling) | 0.34 ms | 4.19 ms |

About 28% of the room to the ceiling. Measured by top-10 overlap against
the plain whole-token model, on a screening split: a cut after layer 7
+10 points, the routing of the token's first four layers (known on the host
by then) +5.3, twice the width +3.0, the next token's id +2.6, the
previous position's routing +1.4, four previous hidden states +1.3;
predicting two or three positions ahead, which a verify pass's draft rows
would need, -7.5 and -13.5; soft targets, weighting misses and mixing
the inputs did not help. Overlap still rose with data (0.41, 0.48, 0.535
at 0.26M, 0.67M, 2.1M positions). Under speculative decoding (2 drafts,
79% acceptance) the cuts still take 10% off the stall per accepted token
(64 GB 4.32 to 3.84 ms, 40 GB 13.54 to 12.13 ms), the token-start model
alone 3% (it sees only the first row).

What it would cost is not measured. Estimated from the shapes: about 140M
parameters, 290 MB of weights per token in bf16, 85% of them in the
24 576-wide output layers, on the order of a millisecond per token on a
small machine's memory bandwidth; as large as what it saves at 64 GB.
A low-rank output layer, fewer cuts and 4-bit weights are the obvious
cuts, each to be checked against the overlap it costs.

**The drive.** `lily-ssd-probe --model <checkpoint>` reads random experts
through the server's read pool (`expert_cache::read_experts`, the
`load_slots` scheme into heap buffers) with `F_NOCACHE`
(`ExpertStore::set_nocache`) and checks residency with `mincore`: one
expert alone 0.51 ms; k experts at once about 0.28 + 0.225k ms (12.3 GB/s
sustained, 4 450 experts a second); a demand read with D prefetches in
flight 0.62, 0.74, 1.04, 1.89 and 3.59 ms at D = 1, 2, 4, 8, 16: the
drive gives demand reads no priority. Read through the page cache, cold
reads cost the same. `--ssd shared` models that (a 0.33 ms latency, then
0.217 ms of transfer shared equally among the reads in flight, at most
`--prefetch-depth` prefetches in flight; `SharedSsd`); it matches the
probe within 15% except for large demand batches against a deep
prefetch queue, where the drive shares by request, not by expert, and the
model is optimistic. It is the default since the server was brought up
to the drive (below); the serial model (one read at a time, `--read-ms`,
0.42 ms by default, demand first) is `--ssd serial`.

**The server against both.** `lily-bench --tokens-out STEM` writes the
measured run's tokens as a trace; `LILY_EXPERT_NOCACHE=1` turns the page
cache off for the expert reads. Both matter on a machine whose memory
holds the checkpoint: without the knob, a miss on an expert read earlier
in the run is a memory copy (0.15 ms per extra miss measured), which a
64 GB machine's page cache could not hold; small-machine benches on this
machine flatter the drive unless they set it, and start from a cold page
cache (an APFS clone of the checkpoint, `cp -cR`, has none cached).
On q4-xl, 8 192-token prompt, 1 024 sampled tokens, plain decode:

| plan | slots | misses per token | decode | replayed misses | replayed stall (serial) |
|------|------:|-----------------:|-------:|----------------:|------------------------:|
| `--memory-limit-gb 71` | 23 115 | 0.72 | 15.0 to 15.7 ms | | |
| `LILY_MEMORY_GB=64` | 15 736 | 19.24 | 26.3 to 26.6 ms | 19.33 | 8.1 ms |
| `LILY_MEMORY_GB=40` | 6 416 | 113.7 | 82.9 ms | 113.9 | 47.8 ms |

The replay reproduces the cache's misses and promotions to within 1%, but
the server's stall is about 11 and 67 ms per token: about 0.6 ms per
miss at both plans, where the probe reads a layer's misses (1.4 and 2.9
on average) in less. The serial model underestimates the server by about
27%, the shared model more; relative comparisons between predictors hold,
absolute savings scale up with it. The difference was the server's, not
the drive's ("Where a miss's time goes" above): with the
GPU kept clocked, milestone pacing and reads started at once, the same
runs (40 GB now with the reference's tokens, its earlier tokens having
been changed by the timed-out waits fixed in "Timed-out waits in long
prefill passes"):

| plan | decode | stall over the reference | shared model | serial model |
|------|-------:|-------------------------:|-------------:|-------------:|
| `--memory-limit-gb 71` | 15.0 ms | | | |
| `LILY_MEMORY_GB=64` | 23.7 ms | 8.7 ms | 8.4 ms (-3.5%) | 7.8 ms (-10%) |
| `LILY_MEMORY_GB=40` | 57.8 ms | 42.8 ms | 40.0 ms (-6.4%) | 52.0 ms (+22%) |

The shared model now tracks the server at both plans and is the
replay's default; the replay's warm layer time (0.315 ms, 15.1 ms over
48 layers) matches the reference's 15.0 ms.

### The predictors on the calibrated drive (2026-10-10)

The predictors retrained (same shapes and sample; sessions held out
untouched for a final check), replayed on the shared drive model, per
decoded token, held-out sessions only. The three recorded sessions as
served (one cache), the others each from the loaded ranking (`--fresh`):

| prefetch (cuts after 3, 7, 15, 23, replacing) | recorded, 64 GB | recorded, 40 GB | others, 64 GB | others, 40 GB |
|---|---:|---:|---:|---:|
| none | 4.65 ms | 13.04 ms | 4.14 ms | 11.43 ms |
| previous hidden state only, top 20, depth 2 | 3.80 ms | 11.37 ms | 3.98 ms | 11.64 ms |
| cuts, top 20, depth 1 | 3.39 ms | 10.31 ms | 3.55 ms | 10.26 ms |
| cuts, top 20, depth 2 | 3.37 ms | 10.36 ms | 3.63 ms | 10.64 ms |
| cuts, top 20, depth 4 | 3.59 ms | 11.25 ms | 3.95 ms | 11.85 ms |
| cuts, top 20, depth 8 | 4.12 ms | 13.99 ms | 4.77 ms | 15.51 ms |
| cuts, top 10, depth 1 | 3.69 ms | 10.37 ms | 3.60 ms | 10.02 ms |
| the routing itself (ceiling) | 0.33 ms | 2.34 ms | 0.29 ms | 1.88 ms |

Prefetch depth decides it now that wrong reads take bandwidth from
demand reads: one or two prefetches in flight; at eight, prefetching is
worse than none. Only one read in six to eight is useful (top 20). By
source, cuts at depth 1, top 20, 64 / 40 GB: the recorded sessions
-27% / -21%, generated agent runs -24% / -20%, public agent
trajectories -23% / -17%, personal sessions -18% / -13%, the text
probes (prose and code read as if decoded, not agent traffic) -7% / -3%;
about a quarter of the room to the ceiling on agent traffic.

Two cache policies replayed against these predictions, neither kept
(code in `docs/bench/2026-10-10-evict-layer-slots/`, ignored):

- **Eviction by predicted need**: a victim is never an expert the
  latest issued plane predicts for its layer. Under 1% (recorded, 64 GB
  3.37 to 3.36 ms, 40 GB 10.36 to 10.27; with the routing itself as
  prediction about 5%).
- **Slots per layer**: soft per-layer capacities, allocated from
  training sessions by exact per-layer LRU miss curves and a dynamic
  programme. They lose about 5% to the single pool the cache has today,
  whose LRU region floats between layers (no prefetch 4.65 against
  4.92 ms at 64 GB, 13.04 against 13.68 at 40 GB); uniform capacities
  lose 18%.

### Prefill chunks of 8 192 tokens

`LILY_PREFILL_CHUNK=8192` (a per-model value now, `Qwen4ExpModel::
set_prefill_chunk`) halves the cold-expert traffic per prefilled token:
cold under the balloon, 1 161 tok/s prefill and 59.9 tok/s plain decode
against 945 and 54.4 at 4 096, the initial fill 5 s on 16 threads. The
4-layer differential test (`prefill_chunk_8192_matches_4096`) finds
prefill and twelve decode steps bit-identical at both sizes, and on the
full model the two sizes agree exactly under the per-query attention
route; under the tile route they differ by the unions' grouping (the
same rounding class as tile against split, top tokens unchanged, draft
acceptance 67.7% against 71.1% over six prompts, two up and four down).
It is not the default under the cache yet: with the balloon at 60 GB,
the 45 GB slab, the scratch and a 32 GB n-gram preload the machine was
far past its memory, and there the 8 192 runs produced a deterministic
but timing-dependent digest that the warm runs and the 4 096 runs did
not (all of those match the resident path exactly); the drift starts at
token 26 and stays plausible text, so it reads as GPU buffers being
paged under extreme pressure rather than a data race in the cache (a
re-read check on the routed ids found no stale reads). A run on a real
64 GB machine, without the preload, is what settles it. (2026-10-10: more
likely the timed-out waits above, which such slow passes run into; at
40 GB a single 8 192-token pass now reproduces the reference tokens.)

### What would move it further

- **Decode misses** cost a cold region read per resolution (about 0.35
  ms with the nine regions in flight). The live ranking above lowers the
  count run over run; the LRU share is swept on recorded agent sessions
  in "Replaying agent sessions" (10% stays), and a ranking taken from the
  prompt was tried and dropped there.
- **Prefill** is bound by reading the cold third per chunk; a chunk of
  8 192 tokens would halve that per token at twice the activation
  scratch.
- **Prefetching** a token's cold experts from a cross-layer prediction
  was replayed and costed on agent sessions ("Cross-layer prefetch on
  agent sessions" above): the predictor costs about what it saves; a
  whole-token predictor from the previous hidden state saves about 0.35
  ms per cold token for little GPU time ("A whole-token predictor from the
  hidden state"), not enough yet for the loader it needs; predictors that
  also read the trunk after chosen layers save about 25% of the stall in
  replay, at an unmeasured GPU cost ("Predictors inside the token").
- **The handshake** (7% of a 2-draft step at zero misses when first
  measured, 11% of a plain token and 6% of a step on 2026-10-09) is 48
  sequential host-GPU
  exchanges per pass; only fewer cached layers or a GPU-side check that
  skips the wait when nothing is missing would reduce it, and Metal has
  no conditional wait.
- **Cache-aware routing** (not tried, parked 2026-10-10): with 10 of 512
  experts per layer, the last picks often carry little of the gate's
  weight. Where such a low-ranked pick misses and an expert scoring
  nearly as high is resident, computing with the resident one would
  avoid the read (published as cache-conditional routing and expert
  substitution; MoE-CORE, https://arxiv.org/abs/2610.01950: "The main
  configuration executes router-selected experts exactly; an optional
  score-based substitution path handles eligible low-score misses").
  Its rule (Sec. 3.5): with b the best unselected score, a selected
  expert scoring within (1+α)b may be replaced, one scoring higher never
  ("Any high-score miss and any unmatched low-score miss remain on the
  exact H2D path"); candidates are unselected resident experts scoring
  within [(1-α)b, b], matched to the misses in rank order, and the
  substitute keeps the missed expert's gate weight; α = 0.1 there.
  It changes outputs, so it needs a quality evaluation
  beside its speed; a first count of how many misses have a resident
  alternative within a margin needs only `lily-trace-routing
  --router-top` and the replay.

## Cross-layer prediction measurement

Prefetching a decode token's cold experts needs their ids a layer or two
before the router picks them. FATE and HOBBIT take them from the next
layer's router applied to the current layer's input, on the argument
that the residual stream changes little between adjacent layers. This
model has four streams and every block reads its own mixed view of them,
so `lily-experts --predict-out` measures two predictors side by side.
Measured 2026-10-09 (results at the end of this section).

```sh
lily-experts --model <checkpoint> --text <corpus> --prompts 4 \
  --json-out usage.json --predict-out predict.json \
  --global-usage tools/bench/expert-usage-qwen38-flash-next.json
```

Each prompt is prefilled, then `--predict-decode-tokens` (256) greedy
tokens are decoded with the plain decode step (no draft head), stopping
early at an end-of-sequence draw. Both passes log, per MoE layer and
position, the routed top-k and the predictions for that layer
(`Qwen4ExpModel::enable_expert_prediction`):

- **A**: the router of layer `l + d` on layer `l`'s gate input (the
  hyper-connection read layer `l`'s router consumed).
- **B**: layer `l + d`'s own MoE read-in (its hyper-connection read with
  the norm it contains) on the four streams right after layer `l`'s MoE
  inject, then its router. Layer `l + d`'s attention has not run, and
  neither has the n-gram embedding when `l + d` is its layer.

`d` is 1 or 2. Both run the real kernels on their own scratch: the router
GEMM (prefill, the chunk's route) or GEMV (decode), the softmax top-k
kernel with the model's renormalisation, and the same read-in function
the layers call. Lookahead 0 is the plumbing check: A reruns layer `l`'s
router on its own input, B reruns layer `l`'s read-in and router on the
streams its real read consumed, and both must reproduce the logged
routing exactly (`self_check`, and a warning on stdout if not). The
draft head and batched decode across sessions are not logged. Without
the flag nothing is encoded for it and the counts output is unchanged.

Since the replay below, the predicting router selects 20 experts
(`EXPERT_PRED_WIDTH` times top-k; the real routing stays top-10) and
the log keeps their router scores beside them (`expert_pred_scores`,
renormalised over the 20 picks where the model renormalises, so they rank
like the real router's scores but are not their values). The router
top-k kernels take k from the indices tensor and select up to 32 now
(`MOE_ROUTER_MAX_K`; the expert kernels keep their limit of 16); the
first 10 of a 20-wide selection are exactly the 10-wide selection, ties
included (`router_topk_wider_selection_extends_the_narrow_one`). The
summary below scores each prediction's top-10 by score
(`route_predict::top_by_score`, stable on ties, so the kernel's order
decides them), and the lookahead-0 check compares the top-10 by score
with the routed set.

The JSON has a `prefill` and a `decode` section, each with `A` and `B`,
each with lookaheads `"1"` and `"2"`:

- `all_layers` and `per_layer` (`null` for the layers without a source
  layer, which are not counted as misses): `overlap`, the mean of
  `|pred ∩ actual| / k`; `exact`, the share of exact top-k matches;
  `covered`, the share with `actual ⊆ pred`. With predicted sets of size
  k, `covered` equals `exact`.
- `missing_pairs_per_token` and `missing_layers_per_token`: histograms
  over tokens (index = count) of the (layer, expert) pairs routed to but
  not predicted, and of the layers with at least one, over all applicable
  layers of the token.
- `prefetch`, against the global top `--resident-slices` (16 384) of the
  `--global-usage` ranking: `reads_per_token` (histogram of predicted
  slices outside that set, the reads a prefetcher would issue),
  `reads_per_layer`, `cold_actual_per_token`, `cold_covered_per_token`,
  `cold_recall` (covered over actual cold) and `read_precision` (covered
  over reads). The SSD serves about 0.6 expert reads per layer-time, so
  `reads_per_layer` above that cannot be issued ahead in time.

`residency` is the per-prompt test of re-ranking after the prefill: the
share of the prompt's decode lookups inside the top N slices ranked by
(own) the prompt's own prefill counts, (global) the global ranking, and
(mix) both normalised and mixed 50/50, per prompt and averaged. Ties in
the own and mixed rankings go by the global order.

The shipped ranking was counted over 40 prompts of
`docs/bench/prompts/corpus.txt`; prompts from that corpus are in its
sample, which favours the global set. `--skip-prompts 40` starts past
them when the corpus is long enough; another text avoids the question.

### Measured 2026-10-09

Six prompts of 8 192 tokens from `docs/bench/2026-10-01-long-context-prefill/
prompt-mixed.txt` (not the ranking's corpus), 128 greedy tokens decoded after
each (768 in all), against the shipped ranking's top 16 384 slices. The
lookahead-0 self-check of both variants is exact over all 36 864 rows.

| phase   | variant | ahead | overlap | covered | missing layers/token | reads/token | cold recall | read precision |
|---------|---------|------:|--------:|--------:|---------------------:|------------:|------------:|---------------:|
| decode  | A       | 1     | 62.8%   | 1.3%    | 46.4 of 47           | 44.7        | 59.8%       | 38.1%          |
| decode  | A       | 2     | 53.5%   | 0.4%    | 45.8 of 46           | 50.9        | 49.8%       | 27.5%          |
| decode  | B       | 1     | 78.9%   | 10.3%   | 42.1 of 47           | 37.9        | 77.4%       | 58.2%          |
| decode  | B       | 2     | 66.2%   | 2.2%    | 45.0 of 46           | 45.3        | 62.0%       | 38.5%          |

Prefill rows are within a point of decode on overlap and coverage (cold
recall is lower there: 74.4% against 77.4% for B one ahead). So the
papers' 90%+ does not transfer to this model as measured: the
hyper-connection read-in matters (B over A by 16 points of overlap, though
A and B differ in the observation point as well as the read-in), but even
B one layer ahead predicts 8 of a layer's 10 experts on average and the
full set only one token in ten. The "missing layers" column counts every
mispredicted expert, resident ones included; what an optimistic scheme
would pay for is a routed expert that is neither resident nor prefetched,
and against the static top 16 384 that is 28.5 cold slices less the 22.0
covered, about 6.4 (layer, expert) pairs per token, so at most that many
layers to re-run per token: not ruled out, but at about 1 ms per
re-entered layer it would not pay as it stands. Per layer, B's overlap runs
from 45% (layer 1) and 53% (layer 31) to 90 to 92% at layers 40, 41 and 43
(85% at 42), where the full set is right for a third of the tokens. The
log kept only the predicted top-10 then, so a wider predicted set (the
top-20 of the predicted router), a score threshold, or a learned predictor
are not measured here (the log holds the top-20 with scores since; the
replay below uses them). As a prefetcher against the static top
16 384, B one layer ahead would cover 22.0 of a token's 28.5 cold slices
for 37.9 reads, 0.81 per layer against the 0.6 the SSD serves in a layer's
time, so it could not be issued in time without a confidence cut on the
router scores.

Residency is the more promising lever: the share of a prompt's decode
lookups inside the top 16 384 slices is 94.0% ranked by the shipped
counts, 97.5% ranked by the prompt's own prefill counts, 97.0% by the
50/50 mix (mean over the six prompts). The own ranking uses only the
prompt's prefill, so a cache could apply it before decoding; re-ranking
the pinned set when a prompt's prefill ends would cut the static misses
by more than half. It is a comparison of static sets: the live cache has
16 246 slots of which 14 621 are pinned, an LRU region and promotion, so
what re-ranking adds over that policy needs a replay of the same traces
through both.

### Replay of the live policy

The figures above score prefetch and residency against the static top
16 384, which lily never runs: the live cache misses about 10 of a
token's 480 lookups, not the 28.5 cold slices the static set implies.
`--replay-out` replays the cache's own policy code over each prompt's
logged routing, host side, and compares three policies on the same
traces:

```sh
lily-experts --model <checkpoint> --text <corpus> --prompts 6 \
  --predict-decode-tokens 128 --json-out usage.json \
  --predict-out predict.json --replay-out replay.json \
  --global-usage tools/bench/expert-usage-qwen38-flash-next.json
```

What it reproduces (`expert_replay.rs`): the request stream the service
thread sees for a plain run, one request per MoE layer per prefill chunk
(`Qwen4ExpModel::prefill_chunk`, 4 096 unless `LILY_PREFILL_CHUNK` says
otherwise; `lily-experts` prefills through the same chunking) carrying
the union of the chunk's rows' routed ids, then one request per layer
per decoded token with its 10 ids. Each request is resolved as `serve`
resolves it: ids deduplicated, every id counted in `LiveUsage::record`
as decode- or prefill-class by the request's routed-id count
(`DECODE_IDS`), looked up through `SlotPolicy::lookup` at a tick that
rises by one per request (the service's sequence number), and a
promotion pass every `PROMOTE_EVERY` (480) requests. The cache is
`--replay-slots` (16 246, the 64 GB plan) with `--replay-lru-share`
(0.10) clamped as the loader clamps it (`SlotPolicy::lru_share_for`),
placed by `--global-usage`, filled fresh per prompt as a fresh process
would be (`--replay-carry` keeps each policy's slab and usage across
prompts instead). Not reproduced: the server's warm-up pass (two prompt
tokens and two decode steps, 144 small requests before the first
prompt), the draft head's verify passes (the logged decode is plain,
one row per pass), batched decode across sessions, the prompt's chat
template, and timing: a replayed miss is a count, not a stall.

The policies:

- **live**: as shipped.
- **rerank-own**, **rerank-mix**: when the prompt's decode starts, a
  fresh `SlotPolicy` with the same slots and share from a ranking of the
  prompt's own prefill counts with the global ranking as tie-break
  (own), or of the two normalised and mixed 50/50 (mix), the orders the
  residency test uses (`expert_replay::rerank_ranking`); the new
  `LiveUsage` takes that ranking as its prior, and the decode runs live
  from there. The one-off cost is `pinned_reads`, the new pinned slices
  not resident in the slab before, and `lru_fill_reads`, the new LRU
  region's initial contents not resident before (a real re-rank could
  leave the LRU region as it was and skip those).
- **prefetch-V-SET-BUDGET-LAYERS**: live, plus after each decode request
  for layer `l` is resolved, the picks of predictor V (B or A, lookahead
  1) for layer `l + 1` that are not resident are looked up through
  `SlotPolicy::lookup` at layer `l + 1`'s tick, highest score first, so
  they take LRU slots (and evict) exactly as a real load would, and the
  layer's own request then cannot evict them. SET is `top10`, `top20`, or
  `score>=τx`, the picks scoring at least τ times the predictor's mean
  10th-highest score over the decode rows (`--replay-thresholds`, 0.5, 1
  and 2: a higher τ gives a smaller set; `mean_set_size` reports what
  each gave). BUDGET is unlimited, 2 or 1 reads per layer; LAYERS is
  `all` (layers 1 and up; layer 0 has no source layer and is left to the
  live policy) or `deep` (layers 36 and up, `--replay-deep-from`, where
  B's overlap is 85 to 92%). A read is useful when the layer's request
  for that token routes to the slice, wasted otherwise; the request's
  misses are the ones the prefetch did not hide.

`replay.json` holds the configuration (`slots`, the clamped `lru_share`,
`pinned`, `prefill_chunk`, `carry`, the `mean_kth_score` per predictor
and the resulting `thresholds`) and `policies`, one entry per policy with
`prefill` and `decode` over all prompts and `per_prompt`. Each phase
has `tokens`, `requests`, `lookups` (distinct ids per request, summed),
`misses`, `misses_per_token` and `miss_rate`; decode adds
`layers_with_miss_per_token` (mean and histogram over tokens, index =
layers whose request missed: what optimistic execution would re-run),
`mean_misses_per_token_over_prompts`, and for prefetch `reads`,
`useful`, `wasted` (each also per token), `precision` (useful over
reads), `mean_set_size` and `gave_up_full` (prefetches stopped because
every LRU slot was in use at that tick). Re-ranks add `rerank` with the
one-off reads; `promotions` counts the promotion passes' swaps. The
policies run in parallel, spread over the cores; the headline rows are
printed to stdout.

`--predict-raw <dir>` writes, per prompt, `prompt-NNN-tokens.u32`,
`-routing.u32` (`[layers, positions, 10]`), `-pred-ids.u32` and
`-pred-scores.f32` (`[2 variants, 3 lookaheads, layers, positions, 20]`),
little-endian, and `prompt-NNN.json` describing shapes, dtypes and axes,
so later analysis needs no GPU run: about 400 MB per 8K prompt.

## Design: a host-driven expert cache

Only on machines whose physical memory cannot hold the checkpoint (or
under an explicit budget); on this machine nothing changes and the
footprint stays as it is.

1. **A slab of expert slots** replaces the per-layer stacked expert
   tensors: one buffer of `n_slots` slices, each the gate, up and down
   projections of one expert, in the layout the gather kernels and the
   grouped GEMM already read (`[slots * I, h]` codes with their scales and
   biases). Every layer's `MoeWeights` points at the same slab.
2. **A slot table per layer** (`U32 [E]`, `NONE` for cold experts) on the
   GPU, mirrored on the host. A remap kernel after the router turns expert
   ids into slot ids; the downstream kernels are unchanged.
3. **A round trip per MoE layer.** After the router's top-k the pass
   signals a shared event; the gather waits on a second one. The host
   waits for the first, reads the routed ids from the shared indices
   buffer, resolves misses (choose a victim slot, `pread` the nine tensor
   regions of the expert straight into the slot, update the slot table),
   and signals the second. About 0.1 ms per layer when nothing is missing,
   0.5 ms per missing expert.
4. **Placement by usage.** The slots are filled at load from a usage
   ranking (`lily-experts` output shipped with the checkpoint or measured
   at first use); the hot set stays pinned and a small fraction of slots
   serves as an LRU region for cold experts, so prefill's sweep over every
   expert does not evict the hot set.

Estimate for a 64 GB machine at the numbers above: a decode token costs
its 11 ms of resident work plus about 5 ms of round trips and 10 ms of
misses, about 40 tok/s against 90 here; a 4 096-token prefill chunk reads
about 23 GB of cold experts at 5 to 7 GB/s, 4 to 5 s on top of its 2 s of
compute, about 600 tok/s against 2 000. Not a cliff, and prefill can
overlap the reads with the grouped GEMM's expert loop later.

### Replay results 2026-10-09

The same six prompts and 768 decoded tokens as above, replayed through the
real policy code at 16 246 slots (14 621 pinned, 1 625 LRU), fresh fill per
prompt; a miss is a count, not a stall. Per decoded token:

| policy                                   | decode misses | layers with a miss | reads | useful | wasted | one-off reads per prompt |
|------------------------------------------|--------------:|-------------------:|------:|-------:|-------:|-------------------------:|
| live (as shipped)                        | 10.74         | 8.03               |       |        |        |                          |
| re-rank at prefill end, own counts       | 5.41          | 4.48               |       |        |        | 1 654 pinned + 723 LRU   |
| re-rank at prefill end, 50/50 mix        | 6.37          | 5.18               |       |        |        | 896 pinned + 651 LRU     |
| prefetch B top-10, unlimited, all layers | 3.19          | 2.80               | 14.1  | 6.4    | 7.7    |                          |
| prefetch B top-10, 2 per layer           | 3.66          | 3.07               | 13.1  | 6.0    | 7.2    |                          |
| prefetch B top-10, 1 per layer           | 4.84          | 3.85               | 10.6  | 5.0    | 5.7    |                          |
| prefetch B top-20, unlimited             | 1.30          | 1.05               | 48.7  | 7.1    | 41.5   |                          |
| prefetch B score >= 2x mean 10th         | 9.86          | 7.64               | 0.9   | 0.9    | 0.05   |                          |
| prefetch B top-10, deep layers only      | 9.42          | 7.14               | 2.4   | 1.2    | 1.3    |                          |
| prefetch A top-10, unlimited, all layers | 5.50          | 4.65               | 14.4  | 4.1    | 10.3   |                          |

So the live policy misses 10.7 of a token's 480 lookups (2.2%, the bench's
2.1%), on about 8 layers, not the 28.5 the static set implied; what the
reviews asked for changes the picture in both directions. Re-ranking the
pinned set from the prompt's own prefill counts halves the misses for
about 2 400 bulk reads per prompt (6.5 GB, about a second at the SSD's
rate, off the critical path), the mix for about 1 550 reads and 40% fewer
misses. Prefetching B's top-10 one layer ahead takes 70% of the misses
away at 14 reads per token (39 MB, about 6 ms of SSD time per token, half
of them wasted); a budget of two per layer keeps 66% at 13 reads. The
wider sets are not worth their reads (top-20: 49 per token for 88%
fewer misses) and the strict score cut reads almost nothing and saves
almost nothing. Variant A is worth half of B. Deep layers only barely
moves the count: the live policy's misses sit mostly below layer 36. The
layers a token would still miss after unlimited top-10 prefetch are 2.8,
so optimistic execution would re-enter about three layers per token.

Against the cold numbers above (plain decode 19.6 ms per token cold
against 15.1 warm, the difference being the 10.7 misses' SSD reads, about
0.4 ms each, so 5 to 6 ms of a cold token), halving the misses is worth
about 15% of cold decode and prefetching two per layer about 20%, if the
reads land in time; re-rank and prefetch were replayed separately, not
combined. The re-rank was then built, cut the bench's misses by 29 to 39%,
added 3% on recorded agent sessions, and was dropped ("The re-rank at the
end of a prefill" above). Prefetch was then replayed on agent sessions and
its predictor's GPU time measured: the predictor costs about what the
reads save, so it was not built ("Cross-layer prefetch on agent sessions"
above).
