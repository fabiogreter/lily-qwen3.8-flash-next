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
elementwise passes, the GPU sampler (`sample`), the speculative accept
and rollback kernels (`spec`), and the vision tower's position blend, 2-D
rotary and bidirectional attention (`vision`).

### Decode

A decode step is about 920 to 990 dispatches encoded as one concurrent pass
with level barriers, about 709 dependency levels at a 1K prompt. Per token
it reads 4.135 GB of weights plus the 226 MB of GDN recurrent state and 25
to 50 MB of attention caches, so about 4.4 GB. A single long stream reads
at 600 GB/s on this GPU, which would make 7.3 ms; the step measures 10.5 to
10.7 ms at a 1K prompt, about 415 GB/s. The difference is the chain itself:
each level is a true data dependency reading 3 to 24 MB, and a streaming
dispatch reaches only 460 to 485 GB/s at those sizes because every level
pays its own ramp and drain, so the step's floor at its mix of level sizes
is about 9.3 ms and it runs within 10 to 15% of that. Every kernel family
sits within about 10% of the ceiling for its own read size; the
measurements are in [performance.md](performance.md), "Where the decode
step's time goes".

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
That is the entire difference between a 1K and an 8K decode step. Those
numbers were taken with the split attention cut into 256-token splits, 18
threadgroups for one query on a 40-core GPU. Batches of up to four rows
(decode, verify) now take 64-token splits and one threadgroup per four query
heads, 198 threadgroups per query: the kernel drops from 1.64 to 0.49 ms per
decode step at 8K and the combine grows from 0.07 to 0.17 ms, about 8% of
the step (plain decode at 8K measured 77 to 83 tok/s in one pair of runs).
`LILY_QSA_SPLIT` sets the token count (32 to 256) and `LILY_QSA_HEAD_SPLIT=0`
drops the head split. Prefill sub-batches keep the 256-token split.

Block selection is one 512-thread threadgroup per query running a radix
select over the block scores (four 8-bit digits, most significant first)
and then a compaction in ascending block order. Every threadgroup-wide step
is a scan or a simd reduction: the digit is picked by a scan over the 256
bins, the top digit, which the exponent crowds into a few bins, is counted
with one atomic per distinct bin per simdgroup, and each thread owns a run
of consecutive blocks (following the context) so the compaction is one scan
per 32K tokens with each thread placing its own blocks in order. The 512
threads (from 256) only shorten each thread's walk: 16.7 to 7.8 us per
query at 8K and 31.3 to 15.7 at 32K, exact either way; 1 024 threads gave
the gain back to the wider scans. Per decode step the twelve selections had
taken 0.21 ms at 8K (from 0.41 with serial steps) and 0.59 at 32K (from
0.70) before that; per 64-query prefill chunk 18 us at 8K (from 108) and 73
at 32K (from 167).

The split kernel's own latency chains were then cut (the token ids staged
once per split, four K/V rows requested per step, one barrier for the four
heads' softmax sums): split plus combine 48.9 to 42.5 us per layer at 8K
and 86.2 to 76.8 at 32K, bit-identical. What remains in it is the per-token
score and reduction chain: two redesigns without cross-simdgroup staging
measured within 1.6 us of it. The GDN step walks its 128 x 128 state in
register blocks of eight rows (19 to 16.3 us per layer), and the MoE down
gather runs four simdgroups per row pair over alternate routed slots with
the slot sums combined in slot order, bit-identical (23.8 to 22.3 us per
layer, 413 GB/s for its 9.2 MB).

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
0.25 s per chunk. A chunk measures about 1.8 s at an 8K real-text prompt
(2.95 s when the sparse attention still ran per query). **Prefill is
compute- and efficiency-bound, not bandwidth-bound.**

Where a 4 096-token chunk goes at an 8K prompt, per pass in the per-kernel
profile:

| kernel                                       | ms    | share | achieved   |
|----------------------------------------------|------:|------:|------------|
| grouped Q4 expert GEMM                       | 572   | 34%   | about 34 TFLOP/s |
| dense bf16 GEMM (tensor ops)                 | 554   | 33%   | 54 TFLOP/s |
| sparse attention over gathered rows + gather | 134 + 29 | 10% | about 25 TFLOP/s; the gather at 400 to 460 GB/s |
| GDN prefill scan, chunked (scan + WY pass)   | 72 + 31 | 6%  | bound by streaming each chunk's rows |
| hyper-connection mix and inject              | 79    | 5%    |            |
| norms, MoE row gather and combine, convolutions, gates, indexer | 233 | 14% | |

At a 1K prompt, where attention is dense, the grouped expert GEMM and the
dense GEMM dominate alike. The per-query sparse-attention kernel had been the
outlier: past the dense limit the whole chunk ran through the decode-style
split kernel in 256-query sub-batches, score, select, then attend per query,
gathering that query's 512 blocks of K and V with no reuse across queries and
no tensor operations, 40% of the chunk at 2.1 TFLOP/s next to a dense kernel
at 29 on the same head shapes.

Two things addressed it. The rows of a chunk whose causal window still fits
the budget (the prefix up to the dense limit) take the dense kernel, which
is exact for them, and only the rest goes through the indexer. And the
default route past the limit is now **tiled**: `qsa_tile_union` merges the
selections of 16 consecutive queries into one ascending block list with a
query mask per block, `qsa_tile_gather` copies the union's K and V rows
(and each row's query mask) once per KV head into a contiguous device
scratch, and `qsa_attn_rows_nax_h1` runs the dense kernel's tensor-op
flash loop over that scratch, one query head per threadgroup, 128 keys per
slice read straight from device memory, each query masked to its own
selection. The gain rests on how much neighbouring queries' selections
overlap, which the union kernel measures: on real text (this repository's
documentation) a tile's union is 1.9 times one query's 512 blocks at 8K
and 3.2 times at 32K, against 16 for disjoint selections. The first tiled
kernel staged the union's rows into threadgroup memory 32 at a time
instead; taken apart in the harness it was a serial chain of gathered
fetches, dependent tensor ops and barriers per slice, and the gathered-row
kernel halves its time (`docs/performance.md`, item 1). The scratch is
512 MB (`LILY_QSA_ROWS_MB`): at 32K contexts and beyond a 256-query batch
is gathered and attended in groups of six tiles, which measured no
slower than every tile at once and faster at 8K. Measured on
the full model in the per-kernel profile: the sparse attention of a pass
past the dense limit takes 153 + 35 ms (attention + gather) at 8K and
about 300 + 78 at 32K, against 1 200 and 1 480 for the per-query kernel.
`LILY_QSA_ROUTE=split` restores the per-query kernel. Sub-batches under 16 rows (the verify
pass) keep the split kernel.

## The vision tower

Qwen3.8-Flash-Next's image encoder runs on the GPU as its own graph
(`src/qwen4exp/vision.rs`, `tools/reference/VISION.md` "The tower"). Its
input is the preprocessed image: one row of 1 536 pixel values per 16 x 16
patch, `N = gh x gw` rows in block-major order, cast to bf16 on upload. Its
output is one 2 560-wide bf16 row per 2 x 2 block of patches, `N / 4` rows,
which replace the prompt's `<|image_pad|>` rows; the `[N, 1152]` residual
after the last block is exposed for diagnostics. The server-side pixel cap
makes `N` at most 8 192.

What runs, in one serial pass per image: the patch embedding as a dense
bf16 GEMM with the flattened `Conv3d` weight; the learned 48 x 48 position
table resampled bilinearly (align_corners) to the image grid and added;
27 blocks of `x += proj(attn(rope(qkv(LayerNorm1(x)))))` and
`x += fc2(gelu_tanh(fc1(LayerNorm2(x))))` with LayerNorm statistics in f32;
and the merger, a LayerNorm per patch, four consecutive rows read as one
4 608-wide row, `fc1`, the exact erf GELU, `fc2`. Nothing of the text
path's GPU work was reusable except the GEMM, so the kernels are new:
LayerNorm with bias (`norm`), the GELUs (`elementwise`; Metal has no `erf`,
so the exact form uses a 1.5e-7 polynomial), a bias epilogue on the tensor-op
GEMM (`gemm`), and in `vision.metal` the position blend (taps and weights
computed in the kernel from the patch's grid coordinates, no host table), the
2-D rotary over the fused qkv rows (absolute row and column per patch, 18
frequencies each, `rotate_half` pairing over the 72-wide head, in f32), and
full bidirectional attention.

**Attention is a fused online-softmax kernel**, a copy of the text path's
tensor-op flash kernel with the causal limit and the KV cache removed, reading
q, k and v straight out of the fused `[N, 3456]` projection at row stride
3 456 and writing `[N, 1152]`. The alternative, scores through the GEMM per
head with a row softmax between, moves about 0.5 GB per head and layer at
8 192 patches, 230 GB for the tower, which alone is the latency budget; the
fused kernel reads K and V once per query tile and holds nothing but a
32 x 64 score tile. The head dim of 72 is not a multiple of 16, so the QK
reduction extent is dynamic (the PV output width may stay static at a
multiple of 8). Tile shapes 16 x 128, 32 x 128, 32 x 64 and 64 x 64 measure
within 3 % of each other; 32 x 64 is the fastest and 256-key tiles exceed the
32 KB threadgroup memory.

**The bias is added inside the GEMM.** A first version added it in a
separate pass over the bf16 GEMM output, which rounds every Linear output
twice where the reference rounds once, and that alone moved the merged
output's relative L2 error against the f32 reference from 0.077 to 0.063 on
the 333 x 777 image. Everything else rounds where the reference in bf16
rounds: the residual stream is bf16, the LayerNorm and GELU outputs are bf16,
attention probabilities are bf16 with f32 scores and sums.

Measured against the f32 reference goldens (`compare_vision.py`, full
tensors), merged output: 333 x 777 relative L2 0.067, cosine 0.9977, 99.95 %
of the elements within `0.02 + 0.05 |golden|`; 640 x 480 0.053, 0.9986,
99.997 %; 1920 x 1080 0.062, 0.9981, 99.97 %; 3840 x 2160 (capped) 0.090,
0.9960, 99.87 %. The reference tower in bf16 against itself in f32 sits at
0.054 / 0.050 / 0.065 / 0.078 and 99.95 / 99.999 / 99.96 / 99.94 %. The
within-fraction gate is that measured floor minus 0.002 per image rather
than a fixed number, because a fixed one did not separate correct
implementations from wrong ones: the elements outside the tolerance sit in
a handful of tokens (5 of 240, 15 of 1 980) whose massive-activation
channel, near 1e4 in the residual, flips by about 1 000 in the bf16
reference as well, and the reference itself with an f32 residual stream
comes out at 99.88 % on 333 x 777. Block by block, lily's residual tracks
the f32 reference exactly as closely as the bf16 reference does (relative
L2 0.036 against 0.035 after 27 blocks), and the merger kernels reproduce
the reference merger on lily's own input to 0.0005. All four images pass
the measured gate (`tools/reference/VISION.md`, "Comparison 2").

| image | patches | tokens | GPU | host | attention | GEMMs |
|---|---|---|---|---|---|---|
| 333 x 777 | 960 | 240 | 27.1 ms | 27.8 ms | 31 % | 55 % |
| 640 x 480 | 1 200 | 300 | 38.5 ms | 39.1 ms | | |
| 1920 x 1080 | 8 160 | 2 040 | 707 ms | 710 ms | 77 % | 19 % |
| 3840 x 2160, capped | 7 920 | 1 980 | 671 ms | 675 ms | | |

GPU time is the pass span, host time includes the bf16 cast and upload of
the pixel rows and the encoding of about 300 dispatches; the shares are from
the per-kernel profile. At 8 160 patches the GEMMs run 6.9 TFLOP in 133 ms
(52 TFLOP/s) and attention 8.3 TFLOP in 548 ms (15 TFLOP/s): with a head dim
of 72 the softmax bookkeeping per score is 3.5 times larger relative to the
matmul work than at the text path's 256, and the tile shape does not move
it. The tower therefore takes 0.7 s at the cap, under a second but not well
under; with the 1.5 s of language-model prefill the image path stays inside
"a few seconds". What would cut the attention time is a kernel that
amortizes the softmax bookkeeping over more work per threadgroup (two heads
or two query tiles sharing the K and V tiles), not tiling.

Scratch is a `VisionScratch` grown to the largest patch count seen: the
pixel rows, two `[N, 1152]` buffers for the residual and the normed input,
`[N, 3456]` for qkv, two more `[N, 1152]`, `[N, 4304]` for the MLP (the
merger's `fc1` output reuses it) and `[N / 4, 2560]`; 237 MB at 8 160
patches, under the plan's 1 GB. `lily-vision-probe` runs the tower alone over
a golden's `pixel_values`, writes the candidate record for
`compare_vision.py`, and prints the per-kernel profile with
`--kernel-profile`.

**Preprocessing** (`src/qwen4exp/image.rs`) turns the request's PNG or JPEG
bytes into those pixel rows on the host, following the reference processor
(`tools/reference/VISION.md`, "Preprocessing"): decode to 8-bit RGB as
`PIL.Image.convert("RGB")` does (alpha dropped without compositing,
greyscale replicated, a palette looked up, 16-bit samples reduced to their
high byte), `smart_resize` to a multiple of 32 on both sides under the pixel
cap with Python's half-to-even rounding, PIL's `Image.resize(BICUBIC)`, then
`(k - 127.5) / 127.5` in f32 into 1 536-wide rows in block-major patch
order with the two temporal frames identical. The resampler is a
transcription of Pillow 12.3.0's `libImaging/Resample.c`: the Keys cubic
with a = -0.5, the support widened by the downscale factor, coefficients
computed and normalised in f64 and then fixed to 22 fractional bits with
half-away-from-zero rounding, the horizontal pass first and the vertical
second, each accumulating in i32 from a half-unit offset and clamping to
uint8, and a pass skipped altogether when its axis keeps its size. Every
one of those details is load-bearing: against Pillow's own output on the
four test images and seven synthetic sizes (up and down, per axis) lily's
resized bytes are identical, and against the torchvision goldens lily's
`pixel_values` differ on exactly the elements and by exactly the one level
that VISION.md measured between PIL and torchvision (148 of 1.47 M on
333 x 777, 282 of 12.5 M on 1920 x 1080, 26 of 12.2 M on the capped
3840 x 2160, none on 640 x 480), so comparison 1 passes with room and the
tower fed lily's own rows passes comparison 2 on all four images (merged
relative L2 0.070 / 0.053 / 0.062 / 0.087, cosine 0.9976 / 0.9986 / 0.9981 /
0.9963, within-fraction 99.92 / 99.997 / 99.98 / 99.88 %). Two things are
not PIL: 16-bit greyscale PNGs, which Pillow saturates at 255 (I;16 to RGB
goes through a clamp and the image comes out white) where lily keeps the
high byte like the RGB case; and JPEG, decoded by `zune-jpeg` rather than
libjpeg-turbo, which agrees with Pillow on all but about 1 % of greyscale
samples by one level and on 4:2:0 colour differs on 2 to 16 % by one level
and 1 to 10 % by two or three (chroma upsampling). Decoding is the server's
first contact with untrusted binary data, so the format and dimensions are
read off the header first: anything but PNG and JPEG is refused by name, so
is a side over 16 384, an area over 64 megapixels, a zero-sized image or a
frame over the allocation bound, all before either decoder allocates, and
the decoders then run under that bound with truncated data an error. The
whole chain is single-threaded plain loops and costs 3 ms for 333 x 777,
under 1 ms for the identity 640 x 480, 3 to 6 ms for 1920 x 1080 and 34 to
37 ms for the capped 3840 x 2160 (14 to 24 ms uncapped, where only the
vertical pass runs), against 0.7 s for the tower at the cap. The dependency
for this is `png` and `zune-jpeg` directly rather than the `image` facade,
which with only those two formats still pulls colour management the server
never applies.

**Positions.** A prompt with an image numbers its tokens differently from
text (`tools/reference/VISION.md`, "Prompt, tokens, positions"). A counter
runs over the prompt: a text token takes the counter on all three rotary
axes and advances it by one; an image's placeholders take the counter as
their temporal position and the counter plus their row and column in the
merged grid as height and width, in raster order, after which the counter
advances by half the larger grid side. The text after the image therefore
sits below its sequence index, and every token generated afterwards at
sequence index `s` sits at `s + rope_delta` on all axes, with
`rope_delta = max(position) + 1 - seq_len` (the reference's `rope_deltas`,
-216 for the 282-token 333 x 777 prompt). `qwen4exp::positions` computes
this from the tokens and the image spans alone and is pinned exactly to the
four reference goldens (comparison 3). Because positions are a pure function
of the prompt, nothing about them is stored with a session: the snapshot and
the disk tier keep their byte layout and the persistence format tag is
unchanged, the decode state carries only `rope_delta`, set when the prompt is
prefilled and never persisted, and a resumed session's cached keys are valid
whenever its tokens and image spans match the prompt's, which is exactly
what the session cache checks (see "The session cache").
In the kernels the cache slot and the rotary position used to be one number;
they are now two roles. Cache slots, indexer blocks and causal limits stay
the sequence index. The rotary angle takes a `Rope` argument: for decode rows
(the single-token step, verify and draft passes, including the GPU-supplied
positions of the speculative chain) every axis is the sequence index plus the
delta, so the five scalar rotary kernels (`rope_neox`, the fused decode Q and
K prep, the indexer's query prep and block keys) take an `int rope_delta`
added in the kernel, and with delta 0 their arithmetic is what it was, which
keeps the text path byte-identical. For the prefill rows of a prompt with an
image a separate kernel variant (`rope_neox_mrope`, `qsa_prep_q_mrope`,
`qsa_block_keys_mrope`), selected only then, reads a per-token `[rows, 3]`
position buffer and applies the interleaved M-RoPE: 32 pairs over the 64
rotary dims, pair `i` taking the temporal axis when `i % 3 == 0`, height
when `i % 3 == 1` and `i < 33`, width when `i % 3 == 2` and `i < 30`
(`mrope_section` [11, 11, 10]), `rotate_half` pairing `(i, i + 32)`. A block
key takes the position of its first token, which can lie before the chunk,
so the uploaded buffer starts at that token. Frequencies are computed in f32
exactly as the text kernels compute them, so the text and image rows of one
prompt rotate consistently; the bf16 forward goldens carry the text
`inv_freq` in bf16 (VISION.md notes this), which shows up as small logit
gaps in comparison 4, not as argmax disagreement.

**The override and the draft head.** After the embedding gather of a prefill
chunk the images' merged rows replace the placeholder rows of the
`[m, 2560]` bf16 embedding before it is broadcast into the four
hyper-connection streams, which is what the reference's `masked_scatter` on
`inputs_embeds` produces; a span that straddles a 4 096-token chunk boundary
is split at it. The n-gram embedding keeps hashing the placeholder ids, as
the reference does. The draft head builds its input from the embedding of the
next token, so its catch-up during prefill gets the same override shifted by
one row: when row `j`'s token is a placeholder the head is fed the image row,
and its caches are built from the input the trunk saw. Acceptance is exact
equality, so this decides draft quality after an image and nothing else.
Comparison 4 on the four-layer conversion (`lily-vision-probe --forward`,
`compare.py`, and the same check as a model-gated Rust test): the 282-token
image prompt agrees with the reference's argmax at all 16 top-8 positions and
the 3 greedy tokens that follow the prompt (top-8 overlap 7 to 8, worst
shared-id logit gap 0.051),
and the text-only control through the same path is byte-identical to
`lily-probe`'s record; it agrees at 15 of its 16 prompt positions, the
exception being an exact bf16 tie in the reference (two ids at 3.5312, which
torch breaks by the lower id) that the unchanged text path resolves the other
way. Prefill of the 282-token image prompt takes 34 ms on the four layers,
the tower 37 ms.

### How the tower was verified

The reference implementation, transformers' `Qwen4ExpForConditionalGeneration`
in the project venv, runs on the same inputs, so most of the checking is
mechanical. Four comparisons, each living in the tree and each fed at the
server's pixel cap of 2 097 152 pixels, which the goldens record:

1. **Preprocessing.** lily's pixel rows against the reference processor's for
   the same file: grid and resized size exact, values within 0.00984 on at
   least 99.9 % of a seeded 4 096-element sample and 0.0255 at worst
   (`tests/unit/qwen4exp/image.rs`, the `hf_vision_preprocess_*` goldens).
2. **The tower.** lily's output against the reference tower's, fed with
   lily's own preprocessing so that a tower bug cannot hide behind an image
   difference: cosine at least 0.995, relative L2 at most 0.10, and the
   fraction within 0.02 absolute or 5 % relative no more than 0.002 below the
   bf16 reference's own floor for that image (`tests/unit/qwen4exp/vision.rs`,
   `lily-vision-probe`, `compare_vision.py`).
3. **Positions.** lily's token numbering for a prompt with an image against
   the reference's, exactly (`tests/unit/qwen4exp/positions.rs`).
4. **The forward pass** with an image on the four-layer conversion, argmax
   agreement at every recorded position and over the greedy continuation
   (`tests/unit/qwen4exp/probe.rs`, `lily-vision-probe --forward`).

The reference side is `tools/reference/hf_vision_reference.py` (one
subcommand per comparison and the measurements behind the tolerances),
`compare_vision.py` and `vision_golden.py`; the checked facts about the
reference, with line references, are in `tools/reference/VISION.md`.

None of that says whether the answers are any good, so the full model was
also asked about screenshots with known answers, over the API, on
2026-09-17: a login form whose button covers the password field, a pricing
table, a compiler error in a terminal, a bar chart, a settings page with
toggles, a page with two spelling mistakes, two screenshots of one page that
differ in a price, and a follow-up question about an image already in the
cache. With thinking on, the server's default, all eight checks passed. With
thinking off, six: the misses were a spelling mistake in small body text and
the overlap described as a missing field. Latency with thinking off: a
1440 x 900 screenshot is 1 260 image tokens, tower 0.32 s, prefill including
the tower 1.1 s, 1.3 to 2.0 s per answer; a 1920 x 1080 screenshot is 2 040
tokens, tower 0.76 s, prefill 2.6 s, 3.2 s in all; two 1440 x 900 screenshots
are 2 520 tokens and 3.8 s; the follow-up about a cached image took 0.7 s
with no tower pass. The first tower pass after a load is slower (0.68 s at
1 260 tokens) while the GPU ramps.

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

The GDN prefill scan of a batch of 128 rows or more runs in chunked form
on the tensor ops (`gdn_chunk_wy3` + `gdn_chunk_scan` in `gdn.metal`): the
recurrence over each 64-token chunk is written as a WY product. A first
pass, parallel over (chunk, key head), forms `A[i][j] = beta_i (k_i . k_j)
exp(G_i - G_j)` from the chunk's `K K^T` product and the cumulative
log-decay `G`, inverts `I + A` by forward substitution (one column per
lane, the three value heads of a key head at once), and stores per value
head `W = T diag(beta exp(G)) K`, `U = T diag(beta) V` and the decayed
causal `Q K^T`. The second pass walks the chunks of one head in order, one
threadgroup per block of 16 state columns, with the fp32 state block in a
cooperative-tensor accumulator and a bf16 copy of it as the operand of the
two reads: `u~ = U - W S`, `S = exp(G_C) S + K^T diag(exp(G_C - G)) u~`,
`o = exp(G) q S + P u~`. The state is handed to the decode step in the
same fp32 layout as before. Results differ from the token-serial scan by
the bf16 rounding of the state and pseudo-value reads and by the products'
accumulation order: against the per-token CPU reference the outputs are
as close as the serial scan's (0.2% of scale) and the final state within
0.5% (serial: 0.2%); the 4-layer golden gaps went from 0.025 / 0.043 to
0.028 / 0.045. A batch's ragged tail past the last whole chunk continues
from the chunked state through the token-serial scan. The token-serial scan (`gdn_prefill_regscan`, one
simdgroup per head and four value columns, the state in registers) serves
the verify passes and short batches, and records the per-row states the
rollback needs; `LILY_GDN_SCAN_KERNEL=gdn_prefill_regscan` routes every
prefill through it for comparison.

**Rollback needs no recomputation.** The GDN prefill scan records the state
after every row, so the accepted one is copied back by index; the convolution
windows are rewound from the rows' saved inputs; attention caches are
position-indexed and simply get overwritten. The head is then caught up on
all `m` rows with the trunk's draws (the rows past `a` compute values that
later rows overwrite before anything valid reads them) and chains further
drafts from its own residual.

**Drafts follow the request's sampler.** Under greedy decoding the head
proposes with argmax and a verify row confirms its draft when the trunk's
argmax equals it. Under sampling the head draws each proposal from its own
distribution with the request's temperature, top-k, top-p and min-p (no
penalties) and leaves that kept distribution `q` on the GPU next to the
proposal; the verify row then runs speculative sampling (Leviathan et al.)
against the trunk's kept distribution `p`: it accepts the proposal with
probability min(1, p(d) / q(d)) on the row's uniform, and otherwise draws
from the residual max(0, p - q) on a second uniform of the same step. The
row's token is distributed exactly as a plain draw from `p` would be, so
sampling-aware drafting changes no output distribution, and it equals the
proposal exactly when accepted, so the engine's acceptance test is still
equality. The draft draws use their own RNG domain (high bit of the step
index) and never share a uniform with the trunk. Measured on the 8K
synthetic prompt with the server's defaults (temperature 1.0, top-k 20,
top-p 0.95, three seeds): 65 to 67% of proposals accepted against 55 to 65%
with argmax proposals, 2.31 to 2.35 tokens per step against 2.10 to 2.31;
greedy decoding is unchanged to the digest.

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

**Image identity.** An image in a prompt is a run of identical
`<|image_pad|>` tokens, so two requests carrying different screenshots have
identical prompt tokens and, by tokens alone, the caches would happily
answer a question about one with the other's cached state. Tokens stay the
identity, and every lineage (a resident session, a disk entry, a durable
entry) also carries its image spans: position, length, patch grid and a
SHA-256 over the preprocessed f32 pixel rows and the grid. The digest is
over the rows and not the file because two files that preprocess to the
same rows are the same image to the model. Wherever two lineages are
matched (`shared_prefix` in `src/serve/session.rs`: the agreement, the
resume position, the disk lookup and hence the durable boundary), the token
prefix they share is cut back to the start of the first span in either that
the other does not match exactly, a span with no counterpart included. A
fork or a disk restore inherits the spans within the reused prefix, a
release records the served prompt's spans, and a durable entry stores the
spans within its boundary. Because a resume position is never inside a
span (checkpoints sit after the prompt's last image and a boundary is an
agreement, which is cut to a span start when the images differ), a
different image behind a shared preamble resumes from before the image and
a durable entry can end exactly at an image start, from which every later
screenshot behind that preamble resumes. Text-only lineages have no spans
and match as before. The disk meta file carries the spans in an `images`
field that older files lack and load as empty; the cache bytes are the
same, so the format tag does not change.

### Durable prefix entries

Checkpoints sit at `prompt_len - 1` of served prompts, which is always past
the point where two fresh agent runs diverge: both send the same preamble
(system prompt, tool schemas, instructions files) and differ from the user's
message on. The second run therefore shares 20 000 tokens with the first and
can resume from none of them, and every run after it pays the same prefill.

The store already computes how far a prompt agrees with each lineage and
threw the number away. Now `acquire` returns the **agreement**: the longest
common prefix between the prompt and any lineage, resident or on disk,
capped at `prompt_len - 1`. It is never below the reused position. When the
agreement is at least `--durable-min-tokens` (default 1 024) and strictly
beyond where the request resumed, the engine materialises it: it prefills to
the agreement, writes a **durable prefix entry** to the disk tier (the
per-token caches for the shared prefix and a snapshot of the recurrent state
at its end, with the boundary as the entry's live end), drops the snapshot,
and prefills the rest as usual. Two real prompts shared that prefix, so a
third is likely. Within one growing conversation the agreement equals the
reused position and nothing is written turn after turn; a parallel run that
shares only the preamble writes it once, and the third run resumes from it.
The prefill is split at the boundary on purpose: chunks are 4 096 tokens and
the batched kernels are not row-count invariant, so a run resuming at the
boundary must process the remainder in the same chunks the materialising run
did, and it does, because both start a chunk there.

Durable entries live **only on disk**. They are never kept as resident
sessions or as extra checkpoints on a live session, because they are hit far
more rarely than the running conversation's cache and would otherwise
displace it from the GPU budget without anyone noticing. Reading one back
costs the disk read of its prefix, about a second per few gigabytes, against
tens of seconds of prefill. A hit on a durable entry always forks from the
file and leaves it in place, even at its live end, where a hit on an evicted
session would move it back to the GPU and delete the files.

At most 16 durable entries exist at a time; storing one more deletes the
least recently used durable entry, and evicted sessions are never chosen for
that. Otherwise they are ordinary entries: the byte budget trims them and the
TTL expires them, which is what retires a preamble once the client's prompt
has changed. Nothing has to be invalidated by hand. The meta file carries a
`durable` flag that older files lack and load as `false`; the cache bytes are
laid out the same, so the format tag does not change.

Two diagnostics come with it. The request's `timings` object and
`GET /v1/timings` carry `agreement_tokens` on every request and
`durable_prefix_tokens` on the one that wrote an entry; the log line adds
`agreement N` whenever it exceeds the cached count and `durable prefix N
written in Ts`. And when a prompt agreed for at least the threshold beyond
where it could resume, the server prints one `divergence at N` line with the
decoded text either side of the seam and what the cached lineage continued
with. An ordinary hit diverges too, at the user's message, and prints
nothing: the line is for shared text that was not reusable. A client
that renders the same preamble differently between runs, which sets the
ceiling on what any cache can share, shows up in that line rather than in a
capturing proxy.

## The server

One engine thread owns the model's lifetime and runs one generation at a
time; a bounded queue holds the rest. Tokenization and chat-template
rendering run on the connection thread, so they never touch the engine
thread. Detokenization and the output parser run per token inside the token
callback, which executes while the parked next step is already running.

**Images.** Image parts (`image_url` with a `{url, detail}` object or
`input_image` with a string) are accepted in user messages only and their
URL must be a base64 data URI of type `image/png` or `image/jpeg`: the
server makes no outbound request for an image, ever (a fetch from the server
is a request-forgery surface, and the agent clients send data URIs anyway),
and `http(s)`, `file` and other schemes are refused with a 400 that says so,
as are other media types, more than `--max-images` images, and any image
when the tower is not loaded (`--vision off`, or a checkpoint without one;
the front decides this from the flag and the checkpoint's `lily.vision`
block before the engine has loaded). Decoding, resizing and patchifying
(`src/qwen4exp/image.rs`, under `--image-max-pixels` and
`--image-min-pixels`) and the SHA-256 of the result run on the connection
thread with tokenisation, about 35 ms plus the digest for a Retina capture;
decoder errors and limit violations become 400s with the module's message.
A message with an image reaches the chat template as structured content
(`{type: text}` and `{type: image}` items in order) so it writes
`<|vision_start|><|image_pad|><|vision_end|>` where the image sits; a
text-only message is flattened to one string exactly as before, so text
prompts are byte-identical. After tokenisation the single pad is expanded
to `grid_h * grid_w / 4` copies per image, in order, which is what the
reference processor does before tokenising, and the prompt's counts of the
three marker tokens (and `<|video_pad|>`) are checked against the number of
images, so a placeholder typed into message text is refused instead of
tokenised into a fake span or fed to the model bare. The expanded prompt is
what the context limit counts.

On the engine thread the request carries the pixel rows, grids and spans.
After the session lookup the engine sets the state's rope delta from the
prompt's positions for every acquired session, text (delta 0) or image, so
a session restored from disk or forked decodes at the right positions; it
then runs the tower once per image whose span has rows at or beyond the
reused prefix (an image entirely inside it is already in the caches, its
span and digest matched by the lookup), copies each merged output out of
the tower's scratch into its own tensor, and prefills the rest of the prompt
with the whole prompt's per-row positions and the image rows, including the
split at a durable boundary, which works when the boundary falls at or
after a span. The tower's scratch lives with the engine's and is dropped
with it. The log line adds `images N (T tokens), tower Ts` and the `timings`
object `image_tokens` and `vision_ms` (both absent for text); `prefill_ms`
includes the tower time.

The HTTP layer is a small HTTP/1.1 implementation over `std::net`, one
connection per thread with `Connection: close`. It is hand-written because
disconnect detection needs the socket: a general-purpose crate buffered
writes and swallowed the errors, so a departed client kept the GPU busy to
`max_tokens`.

Each request's own numbers leave the engine twice. The `timings` object goes
into the response (next to `usage`, and on the last chunk of a stream) from
values the request already measured for its log line, so the addition is a
`serde_json` serialization and nothing else; and a copy goes into a 32-entry
ring buffer behind `GET /v1/timings`. The ring buffer is the only piece of
frontend state that is not a lock-free atomic: the engine thread pushes one
small entry per request and connection threads clone the buffer to answer,
so the mutex is held for a push or a bounded copy and never across a socket
write or a GPU call. It exists because client libraries drop response fields
they do not know — the AI SDK's openai-compatible provider does, which is why
the opencode plugin in `tools/opencode-plugin-timings/` reads the endpoint
instead of the response.

Sampling runs on the GPU, so that only the token id crosses to the host. Two
kernels per step: a wide one applies the penalties (presence, frequency and
repetition over generated tokens through a per-request count table) and
temperature and reduces the maximum; one threadgroup then selects the top-k
by a two-level bucket search on the distance below the maximum (4 096 bins
over a range that widens when needed, refined once, boundary resolution
2^-24 of the range), applies top-p and min-p over the sorted candidates, and
draws by inverse CDF from a counter-based hash RNG seeded by `seed` and the
step index. Greedy without penalties takes the exact argmax kernel. The
selection is shared by three kernels: the plain draw, the draft head's draw
(which also exports the kept ids and probabilities for the verify pass) and
the verify pass's speculative draw (accept the proposal or draw from the
residual, see "Speculative decoding"). For top-k up to 64 (the server's
default is 20) the selection runs in two phases: 64 threadgroups each
select their slice's top-k under the slice's own maximum (the prepare
kernel's partial maxima cover exactly those slices), and the draw then
selects over the 64 x k union through a candidate map, since the global
top-k with the kernels' fixed tie order lies in that union. A single
threadgroup sweeping the 248 320 logits four times took 170 us per draw
regardless of k; per draw in the profiled model the two phases take about
55 us and 22 us (25 and 19 in isolation), with the 5 us prepare kernel
before them, and the drawn tokens are the same to the digest. Larger k
keeps the single threadgroup (the 1 024-candidate cap 184 us).

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
requests, video input (the template's `<|video_pad|>` is refused), and
fetching images by URL (data URIs only, by design). The engine trait's
persistence methods keep defaults that refuse, so an architecture added
later runs without the disk tier until it implements them.
