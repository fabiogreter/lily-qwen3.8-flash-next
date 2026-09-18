# Performance

What the engine measures, how it was measured, and what is left on the table.
Every number here is either **measured**, and then says under what
conditions, or an **estimate**, and then says what assumption it rests on.

## Method

- **Hardware.** MacBook Pro, M5 Max, 40-core GPU, 128 GB unified memory, on
  mains power, GPU otherwise idle. The benchmark refuses to start on battery.
- **Model.** The full 48-layer Qwen3.8-Flash-Next conversion, 103.1 GB, with
  the draft head, `--ngram-preload` on.
- **Prompts.** 1 024, 8 192 and 32 768 synthetic tokens. The token at
  position `i` is `((i * 2654435761) mod 2^32) mod vocab_size`. It is not
  text, and a model free-running on it takes a different trajectory from one
  running on prose, which matters for draft acceptance.
- **Decoding.** Greedy, 96 generated tokens per run, batch size 1.
- **Repeats.** 3 per cell. A cell reports the median; the range in
  parentheses is the minimum and maximum and is the noise band to hold
  against any difference before reading it as real.
- **Interleaving.** When two builds are compared, the runs are interleaved
  per repeat (repeat 1 of every build, then repeat 2, and so on) with a
  cooldown between runs, so thermal and paging drift over a long series lands
  on every build alike instead of on whichever ran last. A sequential
  back-fill once showed its first cell 30 to 40% faster than everything after
  it. Always interleave; never read a single run's absolute tok/s as a
  change.

### The noise band and what distorts it

- **3 to 4% between identical binaries.** Two runs of the same engine code
  differed by 3 to 4% in decode (80.4 against 77.3 tok/s at 1K, 69.6 against
  67.5 at 8K). Differences inside that band mean nothing.
- **Repeated large model loads distort everything.** This is the trap worth
  stating loudest. Each run loads 73 GB. After several back-to-back loads,
  with 7 to 10 GB of swap in use, runs measured **1.5 to 4 times slower**
  than clean ones, on both binaries of a pair. A port was briefly believed to
  be a large regression for exactly this reason. Check `sysctl vm.swapusage`
  before a series and record it with the run; the timeline tooling writes a
  `.env.txt` with swap, paging counters and the thermal level before every
  run.
- **32K prefill is clock-limited.** Every 32K prefill sags to a busy-clock
  median of 1 340 to 1 560 MHz with dips to about 1 000 MHz, against a full
  1 620 MHz at about 75 W for the 1K and 8K runs. The 32K prefill column is
  5 to 15% clock-limited and noisier than the 8K column. It recovers by the
  next run; there is no cumulative decline across repeats.
- **The 1K prefill column is not a measurement of the engine.** A 0.5 s
  prefill starts right after the 73 GB model load while the GPU is still
  ramping from 338 MHz. Two runs with identical engine code differed by 16%
  (1 946 against 1 637 tok/s). `lily-bench` runs a warm-up prefill at the
  exact shape before the timed one, and the column is still unstable. **Use
  the 8K column for prefill.**
- **A stuck compositor halves everything and looks like a regression.** On
  2026-09-17 the machine ran at about half speed for hours (1K plain decode
  40 to 48 tok/s, 8K prefill 1 265 tok/s) with WindowServer pinned at 100%
  GPU in Activity Monitor long after any run, i.e. a compositor workload
  time-slicing the GPU. The signature that separates it from an engine
  change: per-kernel profile sums stay normal (1K decode kernels 13.3 ms per
  step against 12.5) while the unprofiled loop takes twice as long, and
  `powermetrics` shows the GPU at full clock and 100% active residency at
  half its usual power. A reboot fixed it. Check WindowServer's % GPU before
  trusting any absolute tok/s; paired per-dispatch kernel timings and draft
  acceptance rates stayed valid throughout.
- **The per-kernel profiler inflates what it measures.** In profile mode
  every dispatch becomes its own command buffer, which costs 2 to 3 us each
  and loses intra-level overlap. Against unprofiled spans: decode +14%,
  the 3-row verify pass +55 to 60%, the draft pass about +20%, prefill within
  1 to 4%. Rankings and shares from the profiler are reliable; absolute
  milliseconds of tiny kernels are not. The prefill tables below are
  production numbers, the decode ones are not.

## The numbers

Measured 2026-09-17 at commit `7d2a0ac`, median of three repeats, range in
parentheses, interleaved with the commit before that day's kernel work
(`d7cf7a6`, in the second row of each table) right after a reboot (see the
compositor note above). The README's table is the headline: real prompts
over HTTP; these are the synthetic prompts of the fixed matrix.

### Prefill, tok/s

| commit    | 1K prompt         | 8K prompt         | 32K prompt        |
|-----------|-------------------|-------------------|-------------------|
| `7d2a0ac` | 1 940 (1 612 to 2 073) | 2 136 (1 938 to 2 137) | 1 562 (1 561 to 1 599) |
| `d7cf7a6` | 1 893 (1 088 to 1 954) | 1 942 (1 941 to 2 009) | 1 445 (1 348 to 1 483) |

The 1K column is unreliable and the 32K column is clock-limited, both for the
reasons above. The 8K and 32K gains (10% and 8%) are the pipelined tile
kernel, the four-column GDN scan and the vectorized MoE gather.

### Decode without drafts, tok/s

| commit    | 1K prompt        | 8K prompt        | 32K prompt       |
|-----------|------------------|------------------|------------------|
| `7d2a0ac` | 85.2 (79.5 to 93.7) | 87.6 (86.0 to 87.6) | 83.7 (82.3 to 83.9) |
| `d7cf7a6` | 88.1 (71.1 to 93.5) | 86.9 (86.9 to 87.4) | 82.5 (72.8 to 83.8) |

Unchanged within the noise band: none of that day's commits touch the plain
decode step.

### Decode with 2 drafts per step, tok/s

| commit    | 1K prompt          | 8K prompt          | 32K prompt         |
|-----------|--------------------|--------------------|--------------------|
| `7d2a0ac` | 100.8 (93.3 to 113.2), 64% accepted | 118.1 (117.6 to 118.3), 76% accepted | 82.4 (79.9 to 101.5), 64% accepted |
| `d7cf7a6` | 106.8 (85.1 to 115.1), 70% accepted | 118.4 (118.1 to 118.4), 80% accepted | 93.4 (90.1 to 101.3), 64% accepted |

**The lower acceptance at `7d2a0ac` is the trajectory, not the sampling
rework.** At 8K both commits emit the same 96 tokens (same digest) while
the head's proposals differ (74 proposed and 59 accepted against 76 and
58): the four-column GDN scan's fast-math contraction perturbs the draft
head's near-ties on this random-token prompt. Running `7d2a0ac` with the
single-column scan kept for comparison (`LILY_GDN_SCAN_KERNEL=gdn_prefill_
regscan_c1`) reproduces `d7cf7a6`'s drafts and digests exactly (59/74 at
96 steps, 152/208 at 256). The 1K and 32K speculative cells have ranges of
20 tok/s on both commits; the greedy path itself is unchanged. Over six
real-text prompts of 8K tokens (documentation and code, `lily-bench
--prompt-text`, 256 greedy tokens each) the two scans accept 45.5 / 74.3 /
60.3 / 79.3 / 76.7 / 74.3% (one column) against 63.3 / 82.0 / 54.9 / 67.4 /
82.0 / 76.7% (four columns), 68.4% against 71.1% on average with three
prompts each way: acceptance over 256 tokens swings by 10 to 20 points
with the trajectory, and the four-column scan is not systematically worse.

On the same real text (the first 8K tokens of the corpus, 256 tokens, 2
drafts) the exact speculative sampling accepts 49 / 67 / 62% of sampled
drafts over the sampler's three seeds where argmax drafting under the
target-only rule accepted 66 / 57 / 46%, 2.19 against 2.13 tokens per
step; the per-seed swing is again the trajectory, so three seeds bound
the gain loosely at a few percent on prose and code.

**32K speculative cells are trajectory artifacts in general.** The 32K
sequences with and without drafts diverge from one another at token 1 on
this random token prompt, and an earlier matrix's cell fell into a run of a
repeated special token that the draft head predicts poorly (37% accepted).
The kernels involved do not depend on position, and the 4-layer differential
test of the speculative step at position 33 000 passes. Acceptance on real
code and prose has measured 66% over 86 000 generated tokens and 78 to 98%
in shorter runs, so the synthetic figures are lower bounds.

Speculative decoding changes no output: every emitted token is the trunk's
own draw and the draft count changes only how many rows a pass confirms.
Sampling with temperature accepts fewer drafts than greedy decoding.
Measured with `lily-bench --sample` (the checkpoint's defaults: temperature
1.0, top-k 20, top-p 0.95) on the 8K synthetic prompt over 256 tokens with 2
drafts: 73% accepted greedily (2.46 tokens per step, reproducible to the
digest) against 60%, 55% and 65% for three sampler seeds (2.10 to 2.31
tokens per step) while the head proposed with argmax. With the head drawing
its proposals under the request's sampler and the verify rows running exact
speculative sampling against them (`docs/architecture.md`), the same three
seeds accept 65%, 67% and 65% (2.31 to 2.35 tokens per step), about 5% more
tokens per step on this prompt, on which the head and the trunk disagree
more than on text; the output distribution is unchanged by construction. Each sampled draw (two draft proposals and three verify rows per step
under the defaults) then went from about 175 us on one threadgroup to about
85 us in two phases (64 slices' top-k, then the draw over their union),
about 0.45 ms per speculative step.

Three drafts per step were measured again under the exact scheme (8K
synthetic prompt, 256 tokens, `7d2a0ac`, healthy machine): greedy 113.9
tok/s with two drafts against 100.8 with three (72% against 57% of drafts
accepted), and the sampler's three seeds 111.9 / 101.3 / 106.0 against
96.6 / 106.1 / 98.7 (71 / 61 / 64% against 55 / 64 / 57%). Two drafts
remain the default: the third position is accepted rarely enough that the
extra verify row and draft pass cost more than it returns, on this prompt
and on real text earlier (`README.md`).

### Against mlx-lm

There is **no mlx-lm comparison for Qwen3.8-Flash-Next**: upstream mlx-lm has
no `qwen4_exp` implementation, which is why lily's checkpoint format is its
own. The comparison that exists is on the upstream model, Qwen3.6-35B-A3B,
from `mlx-community/Qwen3.6-35B-A3B-4bit`, where both engines read the same
MLX affine Q4 weights. That path has since been removed from lily, so the
numbers below are a historical measurement of a model the engine no longer
serves, kept because no comparable measurement can replace them.

Measured with mlx 0.32.2 and mlx-lm 0.31.3, median of two rounds, ratios are
lily divided by mlx-lm:

| context | lily decode | mlx decode | ratio | lily prefill | mlx prefill | ratio |
|--------:|------------:|-----------:|------:|-------------:|------------:|------:|
| 1 024   | 193.3 | 148.1 | 1.31x | 4 735 | 4 472 | 1.06x |
| 8 192   | 180.5 | 136.8 | 1.32x | 5 458 | 5 032 | 1.09x |
| 32 768  | 151.1 | 118.2 | 1.28x | 3 928 | 3 964 | 0.99x |
| 131 072 |  92.7 |  75.0 | 1.24x | 1 793 | 2 238 | 0.80x |

Caveats, all of which matter:

- **The work inside a token is not identical.** Stock mlx-lm computes and
  returns a full-vocabulary logprob vector even though the harness discards
  it; lily's production path selects greedily without exposing logprobs. The
  claim is production-engine throughput, not that lily executes the same
  graph more efficiently.
- **The trajectories are not shared.** Each engine free-runs its own greedy
  output. Different tokens route to different experts, so the ratios describe
  free-run throughput, not a controlled same-token workload.
- **Prefill is version-sensitive.** Against mlx 0.31.2 the prefill ratios in
  the same matrix were 1.13x to 1.50x; mlx 0.32.2 closed most of that within
  a day. Decode moved barely. A prefill ratio against any particular mlx
  release ages quickly.
- **Boundaries.** Both engines: batch size 1, greedy, the same synthetic
  prompt ids, a fresh process per arm, an exact-shape warm-up, one measured
  prefill, exactly 64 measured decode steps. Prefill is TTFT-style prompt
  throughput up to the first delivered token. Model loading, tokenization,
  template rendering, HTTP and detokenization are outside the timed regions.
  Each engine repeated its own 64-token digest exactly across rounds; a
  cross-engine digest comparison is informational only.

### MLX engines for this model

No released mlx-lm runs Qwen3.8-Flash-Next (0.31.3 has no `qwen4_exp`; the
port is the open, unmerged mlx-lm pull request 1788). On 2026-09-11 a local
build of that pull request, with two patches it needed (the n-gram hash seed,
and a converter that keeps the n-gram tables at 4 bits, which the stock one
leaves in bf16), was measured against lily's binary of that day, interleaved,
one engine loaded at a time: at 1K, mlx-lm 1 526 tok/s prefill and 29.6
decode against lily's 1 946 and 86.5; at 8K, 1 145 and 25.3 against 1 139 and
72.5. The port has no draft head, so lily's 2-draft figures (116.5 and 82.1)
had no counterpart, and it peaked at 105 GB of memory on the 128 GB machine.

Several third-party MLX engines ship their own `qwen4_exp` and publish
figures for an M5 Max: MTPLX (Apache-2.0) reports 79 tok/s at 9K and 61 at
109K with its MTP path under the model's default sampling, 44 plain, and
810 tok/s prefill at 131K; oMLX reports 58 to 70 tok/s with its speculative
path in its 0.7.0 development builds. None of these were measured with this
harness, so they are cited, not compared; lily's plain decode of 82 to 86 in
the table below exceeds their speculative figures, and its 2-draft rate is
measured greedily, which is worth about 10 % over sampled drafting.

### Against llama.cpp

Measured 2026-09-17 with `tools/bench/http_bench.py`, which drives both
servers over HTTP with the same prompts and reads each engine's own timings.
The other side is Unsloth's llama.cpp fork (build b11007, Unsloth Studio
2026.9.5) serving `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ4_XS, 87 GB, KV
cache f16, four slots over a unified 131 072-token context, flash attention
off because the fork aborts at startup with it on for this model. lily ran
commit 0c9ee63 with `--max-seq 131072`, the disk tier off for the matrix so
that no run could hit a cache.

- **Prompts.** Cut from this repository's documentation and source with the
  model's tokenizer, a fresh region of the corpus for every run, at 1K, 4K,
  16K, 32K and 64K tokens. The 1K column is not reported: it sits inside the
  GPU's clock ramp (see above) and lily's runs there ranged from 631 to
  1 679 tok/s.
- **Generation.** 256 greedy tokens of new text, a task unrelated to the
  prompt. The fork's default speculation is an n-gram drafter that copies
  from the context; on a "continue the document" task it reached full
  acceptance and four times plain decode by copying, which says nothing about
  generating. On new text it gains nothing (11 % acceptance) and is left out.
- **Repeats.** Three, interleaved per repeat, ten seconds between runs;
  medians, ranges in the records under `docs/bench/2026-09-17-vs-unsloth/`.
- **Both models at once do not fit** in 128 GB, so the two sides ran one
  after the other with the other server stopped.

Prefill, tok/s:

| context | lily | llama.cpp | ratio |
|--------:|-----:|----------:|------:|
| 4 096   | 1 300 | 887 | 1.47x |
| 16 384  | 1 546 | 882 | 1.75x |
| 32 768  | 1 499 | 713 | 2.10x |
| 65 536  | 1 382 | 550 | 2.51x |

Decode, tok/s:

| context | lily plain | lily 2 drafts | llama.cpp plain | llama.cpp MTP 2 | ratio, best against best |
|--------:|-----------:|--------------:|----------------:|----------------:|-------------------------:|
| 1 024   | 90.2 | 98.9  | 42.0 | 54.7 | 1.81x |
| 4 096   | 86.0 | 101.7 | 39.1 | 51.8 | 1.96x |
| 16 384  | 85.5 | 102.0 | 31.4 | 44.1 | 2.31x |
| 32 768  | 83.9 | 101.5 | 25.2 | 37.0 | 2.74x |
| 65 536  | 81.8 | 97.7  | 17.0 | 27.0 | 3.62x |

Draft acceptance at two drafts per step was 63 % on lily and 66 % on the
fork, so the head behaves alike in both engines. At three drafts both fell to
50 to 53 % and decoded slower than at two (lily 83 to 94 tok/s, the fork
25 to 51), so three is not worth it on either.

**The fork's MTP needs a patched build.** Unsloth Studio's shipped build
aborts while loading the MTP drafter (`GGML_ASSERT(ggml_can_repeat)` in the
qwen4exp MTP graph; the draft head's `hc_head_norm` tensor is declared with
a shape the trunk's norms moved away from) and then starts a third time
without any drafter while its settings page still says MTP. The bug is
reported on the fork's MTP pull request and in unslothai/unsloth issue
11143 with a one-line fix; the MTP rows above come from a build of the same
source with that fix. Everything else is the shipped build.

**Prompt cache behaviour**, tokens recomputed out of the prompt, from the
harness's `cache` test with lily's disk tier on:

| request | lily | llama.cpp |
|---|---|---|
| the same 16K prefix, first new question | 16 404 (full), then a durable entry | 512 |
| the same prefix, later questions | 1 to 16 | 512 to 516 |
| a conversation growing 575 tokens per turn | about 535, in 0.49 s | about 535, in 0.87 s |
| six 12K conversations interleaved, second round | 25 each, in 0.15 s | 25 each, in 0.31 s |

The fork resumes from a checkpoint below the divergence and recomputes about
512 tokens every time. lily's checkpoint sits at the end of the previous
prompt, so the first divergence recomputes the prompt and writes a durable
prefix entry; from then on the tail alone is recomputed. Both engines keep
more conversations than the fork's four slots, the fork through a host-RAM
cache of evicted slots.

Caveats: the quantizations differ (affine 4-bit, group 64, against IQ4_XS at
4.25 bits per weight); llama.cpp's `prompt_ms` and lily's `prefill_ms` both
include their engine's own bookkeeping around the prefill, and lily's
includes the recurrent-state checkpoint; and the fork ran with flash
attention off, its only working configuration for this model that day.

## Known limits and remaining levers

Ranked by expected gain per unit of effort for one interactive coding agent
with long prompts. The evidence in each is measured; the upside in each is an
estimate with its assumption named.

1. **The sparse-attention prefill kernel.** Measured: 40.2% of every prefill
   chunk past the dense limit, running at 2.1 TFLOP/s next to a dense
   attention kernel at 29 TFLOP/s on the same head shapes, because it
   attends per query and gathers that query's 512 blocks with no reuse and no
   tensor operations. Estimated: 8K prefill from 1 396 to about 1 950 to
   2 100 tok/s and 32K from 1 195 to about 1 700 to 1 900, assuming a
   block-gathered tensor-op kernel reaches a quarter to a half of the dense
   rate and that neighbouring queries' selected blocks overlap enough to
   share, which is **not measured**. Highest effort in this list.
   *Done since* (branch `sparse-prefill-tiles`): the overlap measured 1.9x
   one query's selection per 16-query tile at 8K and 3.2x at 32K on real
   text; the tiled tensor-op route cuts the chunk's sparse attention from
   about 1 200 to 350 ms at 8K and 1 480 to 850 ms at 32K, prefill to about
   1 700 tok/s at 8K and 1 450 at 32K in single profile runs, the dense
   kernel taking the rows below the limit besides (`docs/architecture.md`).
   The one-head tile kernel then had its K/V gathers software-pipelined
   (the slice's V rows and the next slice's K rows fetched into registers
   under the tensor ops): 18 to 25% off the kernel per 8K chunk and 12 to
   15% per 32K chunk in paired profile runs, about 4% and 5% of the chunk.
   The GDN prefill scan then went from one to four value columns per
   simdgroup (the per-token k, q and gate loads shared): 384 to 281 ms per
   8K chunk in paired profile runs, 27% of the kernel and about 3% of the
   chunk. Its results differ from the single-column scan only by fast-math
   contraction, which is enough to move the 8K greedy speculative
   trajectory (the digest 9e5cd959e0842bc7 of the earlier rows became
   74f9aab8ea2e7f87, 151/210 drafts accepted instead of 152/208); the
   4-layer golden gaps shrank slightly. The scan then left the token-serial
   form for prefill batches of 128 rows or more: the chunked WY form on the
   tensor ops (64-token chunks; a parallel pass forms and inverts each
   chunk's `I + A`, a sequential pass per head and block of 16 state columns
   carries the state; `docs/architecture.md`). Per dispatch at the chunk
   shape on the profile transport, 0.84 + 1.9 ms against 4.7; in paired
   kernel profiles per 8K prefill (two chunks) 99 to 109 ms against 171 to
   190, per 1K prefill 26 against 51, per 32K prefill 116 to 125 against
   191: about 4% of the 8K prefill and 3% of the 32K one. The scan pass is
   bound by streaming each chunk's rows into the cores, not by the
   products (with every threadgroup reading one head's cache-resident rows
   it took 0.49 ms, the same as with no products at all); wider column
   blocks read fewer bytes but ran too few threadgroups to hide the
   dependent-product latency (32: 2.2 ms, 64: 2.8, 128 with the state
   copy in device memory: 3.2), narrower ones doubled the bytes (8: 3.8),
   and touching the next chunk's lines ahead, pre-transposed keys and
   eight simdgroups did not help. Its results differ from the serial scan
   by the bf16 rounding of the state and pseudo-value reads (against the
   per-token CPU reference the output error is the serial scan's, the
   final state's 0.5% of scale against 0.2%; the 4-layer golden gaps
   0.028 / 0.045 against 0.025 / 0.043, argmax agreement unchanged), which
   moves every digest; over the six real 8K prompts greedy 2-draft
   acceptance averaged 65.0% against 68.7% with the serial scan (46.0 /
   76.3 / 50.0 / 79.7 / 76.3 / 61.6 against 59.1 / 76.3 / 59.1 / 67.1 /
   83.3 / 67.1), the same kind of per-prompt swing as between the one- and
   four-column serial scans. The MoE input gather (one bf16 element per thread) now moves eight
   per thread on its 16-byte-aligned rows: 1.07 to 0.52 ms per call at the
   chunk shape, about 26 ms per 8K chunk (1.3%). A 128-deep K step for the
   grouped expert GEMM (half the B-tile barriers per FLOP) measured within
   2% of the 64-deep one in paired profiles and was not kept.
   The grouped expert GEMM's row tile was swept afterwards on the 4096-token
   chunk (about 80 routed rows per expert): 32 rows 645 ms, 64 rows 593 to
   622, 96 rows 659, 128 rows 759 (four simdgroups) and 698 (eight), so the
   shipped 64-row tile stays and the tensor work on padded rows, not the
   dequant staging, is what the kernel pays for. The padding then went
   instead: an expert's last tile, which usually holds fewer rows than the
   tile height, runs its product at the smallest of the full, half and
   quarter heights that covers its rows (the block map, the dequant
   staging and the results are unchanged, bit for bit). On the measured
   routing histogram 64-row tiles pad the rows by 47% and the mixed
   heights by 10%; in paired kernel profiles the three grouped GEMMs per
   8K prefill took 573 to 583 ms against 666 to 677 (14% off the kernel,
   about 4.5% of the prefill), per 32K prefill 628 against 710, and with
   the 32-row tile of the 1K prefill 229 to 232 against 246 over four
   interleaved pairs (7%). A 32-query tile for the
   one-head kernel (twice the queries per staged slice; the union grows
   from 1 935 to 2 065 blocks per tile at 8K and 3 676 to 5 610 at 32K in
   the harness, so each query fetches about half the rows) measured slower
   everywhere: 4.68 against 3.89 ms per 256-query dispatch at 8K and 13.1
   against 7.6 at 32K on the profile transport, 410 to 426 against 324 ms
   per 8K chunk in paired kernel profiles. The kernel is bound by its
   per-slice softmax and tensor-op work, not by the gathers, so the next
   lever there is per-slice cost or occupancy, not tile height. The
   occupancy was then probed: the same kernel declaring 12 KB more
   threadgroup memory (32 KB in all) measured 4.45 against 3.90 ms per
   256-query dispatch at 8K and 8.5 against 7.6 at 32K in the harness, so
   at its 20 KB the kernel already runs more than one threadgroup per
   core and a diet below 16 KB would buy one more; with 32 staged rows the
   K/V staging alone is 16 KB (16 rows per slice measured slower earlier,
   and the tensor ops take the slice height in multiples of 16), so that
   diet is not available to this design.
2. **The verify pass's kernel shapes.** Measured: for the same 1.64 GB of
   dense weights a 3-row verify pass spends 8.04 ms in the register-resident
   skinny Q4 GEMM where a decode step spends 3.85 ms in the 2-row GEMV, about
   205 GB/s against 420; the m3 hyper-connection kernels show the same
   pattern. Estimated: +7 to +15% speculative tok/s at every context length
   from closing half that gap, assuming the loss is weight streaming in the
   skinny kernel rather than anything intrinsic to the row count.
   *Partly done since*: the register-A skinny kernels compute two rows per
   simdgroup, load each activation block once and dot the raw codes like
   the decode GEMV; in isolation the 3-row kernel streams 360 to 540 GB/s on
   the model's shapes, in the profiled verify pass it went from 5.80 to 5.46
   ms (the Q8 one from 1.41 to 1.14), about 5% of the pass, so most of the
   gap is elsewhere in the pass. Splitting each stream of the
   hyper-connection down kernels over two simdgroups was measured slower
   (9.2 to 9.8 us at one row, 12.8 to 14.2 at three) and not kept; so were
   three more variants of the single-row pair, timed per dispatch on the
   profile transport against the shipped kernels (which measure 8.7 to 9.3
   us for the down read and 10.0 to 10.6 for the up read, 320 to 360 GB/s):
   the down kernel's five weight blocks per lane requested up front (20.5
   us), an unroll pragma on its loop (11.7), and the up kernel's epilogue
   loads hoisted ahead of its weight walk (13.4). The `fused_read_kernels_
   dispatch_timing` test is the harness for that comparison. Routing
   the decode step's own matvecs through the one-row register-A kernel was
   measured in the profiled 8K step and not kept either: the 192 dense Q4
   projections took 4.08 to 4.12 ms against 3.90 to 4.07 with the packed
   GEMV and the 96 Q8 router matvecs 0.64 against 0.50 to 0.52, although in
   isolation the routers had streamed three times faster through it (the
   `skinny_reg_vs_gemv_timing` test now covers every decode shape).
3. **The sparse-attention occupancy parameter at decode.** Measured: with a
   512-block budget and a 256-token split the decode dispatch is 18
   threadgroups on a 40-core GPU, 134 us per call at 31 GB/s; with block
   selection and scoring that is 2.2 to 2.5 ms of a 12.9 to 13.2 ms step past
   2 051 tokens. Estimated: +10 to +14% plain decode at 8K and beyond, and
   comparable for the verify pass, assuming 4 to 8 times the threadgroups
   brings the kernel near the dense split kernel's rate. The split count is
   already a parameter of the existing kernel, so this is the cheapest item
   to test. *Done since*: 64-token splits and one threadgroup per four query
   heads for batches of up to four rows take the kernel from 1.64 to 0.49 ms
   per decode step at 8K (the combine from 0.07 to 0.17), about 8% of the
   step; the verify pass's from 1.98 to 1.32 ms. The block selection, a
   single threadgroup per query, was then rewritten without serial steps
   (scan-picked radix digits, simd-aggregated counting of the top digit,
   thread-contiguous blocks compacted with one scan): 0.41 to 0.21 ms per
   decode step at 8K, 0.70 to 0.59 at 32K, 0.46 to 0.30 per verify pass at
   8K, and 15.6 to 2.6 ms per 8K prefill (30.1 to 13.1 per 32K prefill).
   At 32K the 32 blocks each thread walks per pass are what remains.
   *Then*: the selection threadgroup went from 256 threads (one per radix
   bin) to 512, the extra threads only shortening each thread's walk: 8K
   decode 16.7 to 7.8 us per query, 32K 31.3 to 15.7, the verify rows
   alike; 1 024 threads gave the gain back to the wider scans. Exact, so
   bit-identical.
4. **Dispatch fusion in the decode graph.** Measured: about 966 dispatches
   per step, with a 1.4 us floor for a trivial dispatch plus barrier, and the
   fused hyper-connection kernels reading 0.68 GB at about 315 GB/s against
   the 575 GB/s probe. Estimated: 0.4 to 0.7 ms per step (4 to 6%) from
   halving the dispatch count, plus 0.6 to 1.0 ms from the down kernel's
   occupancy; the upper bound if every dispatch paid the floor would be
   1.35 ms. *Measured since*: folding the write-gate inject into the
   epilogues of the branch matvec and the MoE combine (96 dispatches fewer
   per step, bit-identical) cost as much in those kernels' epilogues as the
   inject kernel took (6.6 to 6.8 ms over the three kernels against 6.3 to
   6.7 in paired profiles), so the 989 dispatches stay; the floor is not
   what those dispatches pay. Folding same-level matvecs into one dispatch
   (the router, the shared expert's gate|up and the shared gate over the
   FFN input; the attention and indexer projections; the PLE key and value
   projections: 110 dispatches fewer per step, bit-identical through a
   four-set GEMV kernel) measured 92.8 against 93.2 tok/s at 1K over three
   interleaved pairs and within noise at 8K, and was not kept: dispatches
   on one level of the concurrent encoder already overlap, so only levels
   (barriers) cost. *Measured since* (the section below): a step has 709
   levels at 1K, a level holding a tiny kernel between two streaming ones
   costs nothing (the 96 inject levels ablate to 0.00 ms), and what a
   streaming level pays is its own ramp and drain, 2 to 4 us for a read
   of a few megabytes; removing or merging levels of tiny kernels is
   therefore not a lever either. The expert
   gathers with their weight blocks requested up front (every routed
   expert's block per lane in the down kernel, every block of the lane's
   row in the gate/up kernel; bit-identical) measured slower in paired 1K
   profiles, 2.30 against 1.21 ms per step for the down kernel and 2.08
   against 1.90 for gate/up, and 82.2 against 91.6 tok/s unprofiled: the
   registers the prefetch holds cost more occupancy than the stalls it
   hides. Not kept.
5. **Continuous batching across sessions.** Measured: an extra row in a
   verify pass costs about 3 ms on a 12 to 13 ms base, because it adds its
   own experts but shares the 2.8 GB of dense weights. Estimated: x1.5 to
   x1.7 aggregate throughput for two concurrent sessions and x2 to x2.3 for
   three, at unchanged or slightly worse per-session latency, assuming item 2
   is fixed first so the batched rows do not pay the skinny-GEMM penalty.
   This is the only item that changes the engine's shape: per-row cache
   pointers and positions in every attention, GDN and n-gram kernel, per-row
   sampler state, and a scheduler that admits and retires sessions per step.

Two smaller ones, for completeness. The decode loop's pacer sleeps until
1.5 ms before the predicted step end and then polls; `nanosleep` on this
machine overshoots by 25% of the requested time (timer coalescing, capped
at 5 ms), and a pacer that scaled its requests by the measured overshoot
measured 93.1 against 93.1 tok/s at 1K over three interleaved pairs, so
the shipped pacer's own adaptation already covers it. The disk tier writes on the engine thread
at eviction time, measured at 0.10 to 0.37 s for 21 000 to 100 000 tokens and
once 1.17 s; copying to host memory and writing from a background thread
would take that off the request path, but it only bites under a tight cache
budget. And the LM head already runs at the probed peak, so it is not a
lever.

Where bytes could be saved instead of time (3-bit experts, Q4
hyper-connection mixers, 8-bit KV), the quality cost is **not measured**, and
the 4-layer Hugging Face comparison cannot see expert quantization error at
scale. A perplexity or task run on the full model would have to come first.

### Where the decode step's time goes

Measured on 2026-09-18 (branch `decode-explore`) with two tools that time the
production shape, because the per-kernel profile mode (one command buffer
per dispatch) sums to 14.1 ms for a 1K step that runs in 10.7 and cannot
say where the gap to the bandwidth floor sits.

**The level chain.** `lily-bench --gpu-timing` counts the dependency levels
(barriers) a step encodes: 709 at 1K, 733 past the dense limit. A pure
streaming read, one dispatch per barrier-separated level over distinct
regions (`level_size_bandwidth_probe`), reaches 600 GB/s with no barriers
but 479 to 487 GB/s when each level reads 8 MB, 459 at 4 MB, 306 to 404 at
2 MB and 176 at 1 MB: every streaming level pays its own ramp and drain,
about 2 to 4 us. The step's floor is therefore the sum of its levels' bytes
at those per-size ceilings, not the 4.37 GB it reads at 600 GB/s (7.3 ms);
at the mix of level sizes the step has, that floor is about 9.3 ms.

**Marginal costs.** `LILY_ABLATE=group` skips a kernel group's dispatches
(and the levels only it occupies); the step time without it is the group's
cost in the chain. At 1K (GPU span per step, median of 96, the whole step
10.68 ms):

| group ablated | ms saved | bytes per step | implied GB/s |
|---|---|---|---|
| dense GDN projections (36 in, 36 out) | 1.97 | 1.17 GB | 595 (at the probed peak) |
| hyper-connection reads (97 pairs) | 1.97 | 0.68 GB | 345 |
| MoE (router, top-k, gathers, shared) | 3.72 | 1.52 GB | 409 |
| of which the expert gathers | 3.01 | 1.32 GB | 440 |
| of which the top-k level (48 tiny levels) | 0.35 | | 7 us per level |
| of which the shared-expert matvecs | 0.00 | 0.13 GB | overlap the router and gather levels |
| attention layers (12) | 0.90 at 1K, 1.58 at 8K, 2.76 at 32K | | |
| GDN conv + step + out projection | 1.57 | | |
| LM head | 0.57 | 0.36 GB | at peak |
| the 96 inject levels | 0.00 | | a tiny level between streaming kernels is free |

**Kernels in their chain.** The chained harnesses (`*_chain_timing`, one
dispatch per layer over distinct weights, a barrier between, best of
several passes) put each family against the level ceiling of its size:
the expert gate/up gather at 35.4 us for 18.4 MB (521 GB/s, at the 16 MB
ceiling), the down gather 23.7 us for 9.2 MB (390; two routed slots per
simdgroup iteration instead of one measured 22.9 and was not kept), the
hyper-connection pair 16.9 us for 7 MB as two 3.5 MB levels (413 and 428
GB/s each, within 10% of the 4 MB ceiling; variants with the activation
loads or the weight loads removed showed neither is the limiter, and the
kernels were already tuned three ways in an earlier pass), the GDN step
19 us for 6.2 MB of state (taken to 16.3 by the register blocks; splitting
the head over thread groups or column groups measured the same 15 to 16),
the router top-k 4.3 us per level against a 1.35 us floor. The dense
GEMVs read 24 MB levels at the peak. What remains at 1K after the kept
changes is about 0.15 ms in the down gather, 0.15 in the top-k level and
0.1 in the hyper-connection reads: each one to two percent, each needing
its own kernel design.

**Attention past 2K.** The branch costs 0.90 ms at 1K, 1.58 at 8K and
2.76 at 32K. Per layer at 32K in the chain: block scores 12 us, the
selection 31 (now 16), the split kernel about 40 and its combine 5 to 8.
The split kernel's 40 us for 4 MB of K/V is structural: with its V loads
removed it measured 2.5 us less and with its K loads removed 5.4 less, so
the rest is the barriers, the cross-simdgroup staging and the per-token
reductions of the eight-simdgroup design. Cutting its latency chains
(token ids staged once, four rows requested per step, one barrier for the
four heads' softmax sums, two heads staged per barrier pair) took the
split-plus-combine pair from 48.9 to 42.5 us at 8K and 86.2 to 76.8 at
32K, bit-identical. Not kept: a 128-thread threadgroup (3 us faster,
reorders the reductions), a barrier-free one-simdgroup-per-split design
with lanes owning tokens for the scores and dims for the values (82 us),
32-token splits (58 us), and a combine kernel with its statistics
prologue parallelized (within noise, and not bit-identical because
fast-math reassociates the serial sum it replaced).

**Kept, with the paired in-model result** (interleaved base/final runs on
p0.txt, 96 steps): plain
decode at 1K 10.67/10.68/10.65 ms per step (base) against 10.58/10.60/10.60
(final), 91.5 against 92.4 tok/s; at 8K, where the machine changed clock
state mid-batch, 11.39 against 11.18 ms in the first pair and within 0.05
ms in the two slower pairs, 85.7 against 86.9 tok/s in the fast pair;
with two drafts 93.0 against 93.3 tok/s at 1K and 98.1 against 99.5 at
8K, the same 46 of 100 and 52 of 88 drafts accepted. Every digest is
unchanged (plain d66084da31d95a1c and 9d9ad65da00e33ad, speculative
a9de616648498453 and d413fc8311fcbbab). About one percent at 1K and one
to two at 8K, as the chained harnesses predicted; the gains grow with
context (the selection and split kernels scale with it).

**The machine's clock state.** Both streams' measurements drifted between
a fast and a slow GPU state during this work (the 8 MB level probe at 487
against 372 GB/s, a 1K step at 10.7 against 13.9 ms) while the other
worktree ran prefill kernels; every comparison above is paired within one
state, and the probe line is the canary to run before trusting a batch.

## Reproducing this

### One cell

`lily-bench` runs the production cadence: a warm-up prompt at the exact
shape, then a timed prefill and a pipelined decode run with token ids kept on
the GPU.

```sh
target/release/lily-bench \
  --model ~/models/Qwen3.8-Flash-Next-lily-q4 \
  --prompt-len 8192 \
  --decode-steps 96 \
  --ngram-preload \
  --gpu-timing \
  --json-out result.json
```

| flag               | meaning                                                                 |
|--------------------|-------------------------------------------------------------------------|
| `--prompt-len N`   | synthetic prompt length                                                 |
| `--decode-steps N` | generated tokens (96 for the matrix below)                              |
| `--drafts N`       | measure speculative decoding with N drafts per step and report acceptance instead of the one-token loop |
| `--ngram-preload`  | stream the paged n-gram table through the page cache before measuring   |
| `--gpu-timing`     | add command-buffer GPU timestamps and host marks, and print the levels per step |
| `--kernel-profile` | per-kernel GPU times per pass; wall-clock results under this flag are not comparable to a normal run |
| `--json-out PATH`  | write the record                                                        |

`LILY_ABLATE=hc,inject,gdn_proj,gdn_step,attn,head,ple,moe,router,shared,topk,gather,down`
(any subset) skips those kernel groups' dispatches in the decode graph; the
results are garbage, the step time without a group is its marginal cost in
the level chain.

### The matrix

`tools/bench/timeline.sh` runs the fixed matrix (1K / 8K / 32K prompts, 0 and
2 drafts, 96 tokens, 3 repeats, repeats outermost) at one or more commits and
summarizes the records with `tools/bench/summarize.py`.

```sh
tools/bench/timeline.sh --note "what changed"
tools/bench/timeline.sh --commit <old> --commit HEAD --cooldown 20
```

With several `--commit` flags it builds them all first, in worktrees under
`target/timeline/`, and then interleaves them per repeat. `HEAD` or
`worktree` means the working tree, labelled `-dirty` when tracked sources are
modified. It refuses to start on battery power unless told
`--allow-battery`, runs only when invoked, and costs about 10 minutes of GPU
time per commit. Each run's JSON, a `run.json` with the commit, note, power
source and host, and a `.env.txt` with swap, paging counters and the thermal
level land in `docs/bench/<date>-<sha>/`. Those records and the timeline
document `summarize.py` writes are local output, not part of the published
repository.
