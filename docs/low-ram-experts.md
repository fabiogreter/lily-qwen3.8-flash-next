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
