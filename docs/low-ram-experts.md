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
   had cost 30%.
3. **Placement by usage.** The ranking from `lily-experts` (`expert-usage.
   json` next to the checkpoint, or `LILY_EXPERT_USAGE`; uniform without
   one) pins the top slices; an LRU region (10% by default, at least a
   layer's worth, `LILY_EXPERT_LRU_SHARE`) takes the cold ones, never
   evicting a slot used at the current sequence number. Misses are read
   with positioned reads on up to 16 threads, each expert's nine regions
   spread over them, straight into the slot.
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
   experts by what this machine actually ran. `LILY_EXPERT_ADAPT=0` keeps
   the counts but never promotes; the load line says which ranking was
   used and where the usage goes, and the cache statistics report
   decode-class misses and promotions.

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
  decode step would up to rounding.
- `lily-cache-replay --usage <ranking> --trace <stem>...` replays them
  through the cache's own policy code (`qwen4exp::cache_replay`:
  `SlotPolicy`, `LiveUsage`, promotion), host only, sweeping the LRU
  share. `<stem>.json` gives the requests as `[p0, p1, p2]`:
  `p0..p1` prefilled (the suffix after the session's cached prefix),
  `p1..p2` generated. Sessions replay in order through one cache, as one
  server serves them; `--fresh` starts each from the loaded ranking.

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
decode misses at every share: 11.12 against 10.78 per token at 10% (+3%), per session from 0 to +23%, with a 3% gain in prefill
misses; 33 re-ranks fired, about 88 000 background loads. A ranking taken
from the prompt helps only where the output continues the prompt (long
document summaries, continuation), not an agent, whose most common use
here does not justify the code; it was removed.

The traces themselves stay out of the repository (they are a client's
code and conversations); only these counts are recorded.

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
64 GB machine, without the preload, is what settles it.

### What would move it further

- **Decode misses** cost a cold region read per resolution (about 0.35
  ms with the nine regions in flight). The live ranking above lowers the
  count run over run; the LRU share is swept on recorded agent sessions
  in "Replaying agent sessions" (10% stays), and a ranking taken from the
  prompt was tried and dropped there.
- **Prefill** is bound by reading the cold third per chunk; a chunk of
  8 192 tokens would halve that per token at twice the activation
  scratch.
- **The handshake** (7% of a 2-draft step at zero misses when first
  measured, 11% of a plain token and 6% of a step on 2026-10-09) is 48
  sequential host-GPU
  exchanges per pass; only fewer cached layers or a GPU-side check that
  skips the wait when nothing is missing would reduce it, and Metal has
  no conditional wait.

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
