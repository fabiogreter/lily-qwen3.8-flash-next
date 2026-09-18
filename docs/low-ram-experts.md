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
`lily serve --memory-gb 64` (or `LILY_MEMORY_GB=64`, and `lily-bench
--memory-gb`) plans for that much memory instead of the machine's, which
is also how to try the mode here: a 64 GB budget picks 16 441 slots
(42.3 GB) and turns speculative decoding off. Copy
`tools/bench/expert-usage-qwen38-flash-next.json` next to the checkpoint
as `expert-usage.json` (or point `LILY_EXPERT_USAGE` at it) so the
busiest experts are the resident ones; without it the placement is
uniform. The load prints what it decided, and the bench prints the
cache's lookups and misses per phase.

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
4. **No draft head under the cache** unless `LILY_EXPERT_CACHE_DRAFTS` is
   set: a speculative step runs three trunk passes through the cached
   layers (two drafts, one verify) where plain decode runs one, and the
   handshakes and misses scale with passes.

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
not a cliff. The same cold run the next day at commit `8ec67f0`, after
the prefill kernel work of 2026-09-18 (`docs/performance.md`): 929 tok/s
prefill and 64.4 tok/s plain decode, 13.6% of all lookups missing, no
stale reads; against 2 257 and 87 resident. Cold-run figures move with
what the page cache still holds when the run starts. The initial fill of 45 GB takes about 20 s from the page
cache and would take the SSD's 10 to 15 s cold.

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
  ms with the nine regions in flight). A ranking that includes decode-time
  routing, or a larger LRU region (30% measured 6.9% against 8.8% of
  decode lookups missing, at the price of prefill misses), lowers the
  count; both are knobs to sweep on a real 64 GB machine.
- **Prefill** is bound by reading the cold third per chunk; a chunk of
  8 192 tokens would halve that per token at twice the activation
  scratch.
- **The handshake** (7% at zero misses) is 48 sequential host-GPU
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
