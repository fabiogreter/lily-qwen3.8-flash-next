# Performance

What the engine measures, how it was measured, what was done to get there,
what was tried and dropped, and what is left. Every number here is either
**measured**, and then says under what conditions, or an **estimate**, and
then says what assumption it rests on.

## Method

- **Hardware.** MacBook Pro, M5 Max, 40-core GPU, 128 GB unified memory, on
  mains power, GPU otherwise idle. The benchmark refuses to start on battery.
- **Model.** The full 48-layer Qwen3.8-Flash-Next conversion, 103.1 GB, with
  the draft head, `--ngram-preload` on. `lily-bench` preloads the table
  before it measures; the server preloads it in the background after it is
  ready, so an HTTP run starts after the log's `background preload
  finished` line, or its first cells pay cold rows.
- **Prompts.** Real text: the first 1 024, 8 192 or 32 768 tokens of a file
  cut from this repository's documentation and source and two sibling
  repositories (`lily-bench --prompt-text`, `tools/bench/corpus.py`). The
  matrices before 2026-09-18 used synthetic tokens (the token at position
  `i` is `((i * 2654435761) mod 2^32) mod vocab_size`); a model free-running
  on those takes a different trajectory from one running on prose, which
  matters for draft acceptance and expert routing, so the two kinds of
  matrix are not compared with each other.
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
- **The GPU has two clock states under sustained load.** Through a day of
  back-to-back kernel work the machine drifted between a fast and a slow
  state (an 8 MB streaming probe at 487 against 372 GB/s, a 1K decode step
  at 10.7 against 13.9 ms), sometimes between consecutive runs. Every
  paired comparison in this document was taken within one state; the
  `level_size_bandwidth_probe` line is the canary to run before trusting a
  batch, and a pair whose two halves fell into different states was redone.
- **Repeated large model loads distort everything.** This is the trap worth
  stating loudest. Each run loads 73 GB. After several back-to-back loads,
  with 7 to 10 GB of swap in use, runs measured **1.5 to 4 times slower**
  than clean ones, on both binaries of a pair. A port was briefly believed to
  be a large regression for exactly this reason. Check `sysctl vm.swapusage`
  before a series and record it with the run; the timeline tooling writes a
  `.env.txt` with swap, paging counters and the thermal level before every
  run.
- **An idle second costs 0.4 s of residency.** After about 1.5 s without
  GPU work the next submission waits 0.4 to 0.6 s for the model's
  residency set to become resident again (about 6 ms per GB, pinned or
  not; architecture.md, "The Metal 4 transport"). A request after a
  cooldown pays it once, in `session_ms` or `wait_ms`, whatever its
  length; back to back it is gone. A series with cooldowns and one without
  are therefore not comparable at short prompts, and a cooldown of a
  second or less does not reset it. The server keeps the set warm with a
  signal a second while the weights' pin is held (architecture.md, "The
  server"), so an HTTP request within `--pin-hold` of the last one does
  not pay it; `lily-bench` has no such keep-alive.
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
  milliseconds of tiny kernels are not. Where the decode step's time goes
  was therefore measured with two production-shape tools instead (below).
- **Sustained load lowers the GPU clock, and only compute-bound passes
  notice.** After about 90 seconds of continuous GPU load the M5 Max
  settles into a lower GPU clock and stays there until the load stops for
  a minute or two. Measured on 2026-09-18 with the server's profiling
  switches over the 1K-to-64K HTTP series: the 3-row verify pass went from
  18.0 to 21.4 ms of GPU time on the same 1K request with the same drafts
  accepted, in every one of five series (with the display on or off, the
  session cache at 8.6 GB or 1 GB, the n-gram table locked or not, 10 s or
  no pause between requests, the onset moving with the pause length); the
  per-token callbacks measured 0.01 ms and the host's finish segment 2.4
  to 2.6 ms in both states; the profile transport, which serializes
  dispatches and runs cooler, showed the pass's kernel sum unchanged (21.07
  against 21.05 ms); the 8 MB streaming probe read 487 GB/s before and
  after, and plain decode, memory-bound, held 85 to 87 tok/s through the
  same series. A fresh process two minutes later ran the same step at
  full speed. Prefill sags the same way (the 1K prefill from 1 844 to 645
  tok/s after a 10 s pause following a long request; the 32K busy clock at
  1 340 to 1 560 MHz against 1 620 was measured with `powermetrics` on the
  17th). So a single-prompt `lily-bench` run measures the full-clock
  state, a seven-minute HTTP series the sustained one, and the two are not
  compared with each other; the README table reports the sustained
  medians because an agent session is sustained load. A light GPU
  competitor does the same and worse (the 4-layer test model decoding in
  parallel took speculative decode from 115 to 60 tok/s and plain from 92
  to 73), and four CPU-hog processes changed nothing in either loop.
  A `powermetrics` trace under the series then showed what the sag is:
  in the first repeat decode ran at 1 620 MHz drawing 48 to 55 W, in the
  second at 1 234 to 1 241 MHz drawing 24 W, the 4K prefill at a median
  791 MHz and 16 W. Clock and power fell together to half, which is not a
  thermal limiter (that holds power at its cap while the clock falls) but
  the system taking the performance envelope away from a windowless
  process it no longer counts as working for the user. The server now
  holds an `NSProcessInfo` activity assertion, user-initiated and
  latency-critical, for each request (`src/activity.rs`; released between
  requests, so an idle machine still sleeps; `lily-bench` holds one for
  its run). With it the same second repeat ran at 1 489 to 1 551 MHz and
  38 to 42 W, 22.0 to 23.4 ms per speculative step against 25 to 27, and
  the third at 1 540 to 1 620. The sag that remains, from 1 620 to about
  1 500 to 1 550 MHz under sustained load, is the machine's own power
  management; High Power mode is the lever for that, not software.
- **Draft acceptance is a property of one trajectory.** Over 256 greedy
  tokens on one prompt the acceptance rate swings by 10 to 20 points
  between two builds whose logits differ in the last bits, because the two
  trajectories part at some near-tie and then sample different text. A
  change to prefill numerics is judged by its logits against the previous
  build's on the same prompt (`lily-probe`, top-64 of the last prefill
  row) and by acceptance totals over several prompts, never by one cell.

## The numbers

The README's table is the headline: fresh real prompts over HTTP at 4K to
64K, last measured 2026-10-07 for both checkpoints (below, "The README
series, 2026-10-07"): prefill 2 158 to 2 373 tok/s, decode 94 to 103 tok/s
with two drafts, 79 to 87 without on q4 and 68 to 73 on q4-xl. The
2026-10-01 series that preceded it is kept below.

The fixed `lily-bench` matrix that follows was measured 2026-09-18 at
commit `38d2642` against the commit before that day's kernel work
(`37cc34c`), interleaved per repeat, median of three, range in
parentheses, on the first tokens of one real-text prompt
(`docs/bench/prompts/p0.txt`).

### Prefill, tok/s

| commit    | 1K prompt              | 8K prompt              | 32K prompt             |
|-----------|------------------------|------------------------|------------------------|
| `38d2642` | 2 118 (1 808 to 2 132) | 2 257 (2 050 to 2 277) | 2 051 (2 046 to 2 052) |
| `37cc34c` | 1 901 (1 835 to 1 952) | 1 981 (1 926 to 2 110) | 1 747 (1 593 to 1 772) |

11%, 14% and 17%: the chunked GDN scan on the tensor ops, the expert
GEMM's last tiles at half and quarter height, and the sparse attention over
gathered rows, all described below.

### Decode without drafts, tok/s

| commit    | 1K prompt           | 8K prompt           | 32K prompt          |
|-----------|---------------------|---------------------|---------------------|
| `38d2642` | 93.9 (80.9 to 94.7) | 87.1 (83.0 to 88.8) | 85.5 (85.3 to 85.8) |
| `37cc34c` | 93.1 (87.0 to 93.4) | 87.5 (85.2 to 87.6) | 83.3 (81.7 to 83.6) |

Within the band at 1K and 8K, 2.6% at 32K: the block selection over 512
threads, the GDN state walk in register blocks, the split attention kernel's
shorter latency chains and the four-simdgroup down gather are each a
percent or less and grow with context.

### Decode with 2 drafts per step, tok/s

| commit    | 1K prompt                          | 8K prompt                           | 32K prompt                         |
|-----------|------------------------------------|-------------------------------------|------------------------------------|
| `38d2642` | 79.2 (70.2 to 80.4), 33% accepted  | 88.3 (85.0 to 89.7), 48% accepted   | 83.4 (82.8 to 84.5), 44% accepted  |
| `37cc34c` | 93.1 (87.2 to 93.2), 46% accepted  | 100.7 (100.2 to 101.0), 59% accepted | 83.3 (77.7 to 85.7), 46% accepted |

**The lower speculative cells are this prompt's trajectory, not a slower
engine.** The chunked GDN scan and the gathered-row attention change the
order of floating-point operations, so the logits move in their last bits
and the greedy continuation of this prompt parts from the old one at its
second token at 1K and its fourth at 8K; the new continuation happens to be
one the draft head predicts less well. Over the six real-text prompts the
totals are 893 of 1 286 drafts accepted (69.4%) with the chunked scan
against 897 of 1 278 (70.2%) with the serial one at 8K, and 873 of 1 326
(65.8%) against 901 of 1 270 (70.9%) at 1K, where the whole gap is this one
prompt (40.8 against 73.1%); on the full model the last prefill row's top-64
logits differ between the two scans by a mean of 0.12 to 0.39 and at most
1.2 where the top logit is about 17, the top token unchanged on all twelve
prompt-and-length pairs, and the earlier one-column serial scan differs from
the shipped one by 0.09 to 0.46 and at most 1.2 and flips one top token. The
new kernels sit inside the rounding band the old ones already spanned. The
HTTP table in the README, which gives every run a fresh prompt, is the
measurement of speculative throughput; a single-prompt matrix cell is not.

The matrix of the day before (synthetic prompts, `7d2a0ac` against
`d7cf7a6`, right after the reboot that cleared the stuck compositor):
prefill 2 136 against 1 942 tok/s at 8K and 1 562 against 1 445 at 32K
(the pipelined tile kernel, the four-column GDN scan and the vectorized
MoE gather), plain decode unchanged (87.6 against 86.9 at 8K), 2-draft
decode 118.1 against 118.4 at 8K with the same 96 tokens emitted.

### Speculative decoding

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
more than on text; the output distribution is unchanged by construction. On
real text (the first 8K tokens of the corpus, 256 tokens, 2 drafts) the
exact scheme accepts 49 / 67 / 62% of sampled drafts over the sampler's
three seeds where argmax drafting under the target-only rule accepted 66 /
57 / 46%, 2.19 against 2.13 tokens per step; the per-seed swing is again the
trajectory, so three seeds bound the gain loosely at a few percent on prose
and code. Each sampled draw (two draft proposals and three verify rows per
step under the defaults) takes about 85 us in two phases (64 slices' top-k,
then the draw over their union), down from about 175 us on one threadgroup,
about 0.45 ms per speculative step.

Three drafts per step were measured under the exact scheme (8K synthetic
prompt, 256 tokens, healthy machine): greedy 113.9 tok/s with two drafts
against 100.8 with three (72% against 57% of drafts accepted), and the
sampler's three seeds 111.9 / 101.3 / 106.0 against 96.6 / 106.1 / 98.7. Two
drafts remain the default: the third position is accepted rarely enough that
the extra verify row and draft pass cost more than it returns, on this prompt
and on real text over HTTP (below).

Acceptance on real code and prose has measured 66% over 86 000 generated
tokens and 78 to 98% in shorter runs; synthetic-prompt figures are lower
bounds, and the 32K cells of the synthetic matrices were trajectory artifacts
(the sequences with and without drafts diverging at token 1, one cell
falling into a run of a repeated special token that the head predicts
poorly). The kernels involved do not depend on position, and the 4-layer
differential test of the speculative step at position 33 000 passes.

### The README series, 2026-10-07 (q4 and q4-xl)

Same method as the 2026-10-01 series below, `--max-seq 131072`, disk tier
off, one server per checkpoint and draft count, run back to back in one
afternoon rather than interleaved. Builds: q4 and q4-xl without drafts and
q4 with drafts at `a4a23d7`; q4-xl with drafts at `6d45554` (the 4-bit draft
path, the shipped q4-xl checkpoint); prefill rows from the 2-draft servers.
Medians of three:

| tok/s | 4K | 16K | 32K | 64K |
|---|---:|---:|---:|---:|
| prefill q4-xl | 2 219 | 2 370 | 2 373 | 2 180 |
| prefill q4 | 2 158 | 2 208 | 2 240 | 2 205 |
| decode q4-xl, 2 drafts | 100.8 | 100.6 | 93.9 | 94.1 |
| decode q4, 2 drafts | 102.9 | 100.3 | 99.4 | 99.1 |
| decode q4-xl, no drafts | 72.9 | 72.9 | 69.0 | 67.7 |
| decode q4, no drafts | 86.9 | 81.8 | 84.5 | 79.0 |

Acceptance 63 % (q4), 64 % (q4-xl). Two disturbances: a checkpoint
conversion wrote to disk during the q4 run without drafts (one 32K prefill
at 1 306 tok/s, the median moved little), and a CPU-bound build ran during
the first repeats of the q4-xl run with drafts. The q4-xl rows without
drafts were measured on the same trunk before the 4-bit draft path existed,
which does not change a plain decode step. The same afternoon, back to back,
the 4-bit draft path took q4-xl's 2-draft decode from 89.4 to 97.5 tok/s at
4K and 91.1 to 95.6 at 32K.

### The README series, 2026-10-01 (`c096b75`)

The README's method rerun on the server that ships on 2026-10-01 (commit
`c096b75`): `tools/bench/http_bench.py matrix`, fresh real-text prompts
cut from this repository (591 693 corpus tokens that day) at 4K, 16K, 32K
and 64K tokens, 256 greedy tokens of new text, three repeats interleaved
per repeat, medians, range in parentheses. One server with the draft head
as shipped (`--mtp-drafts 2`) gives the prefill and 2-draft rows, a second
started with `--mtp-drafts 0` the plain rows; `--max-seq 262144`, disk tier
off (`--disk-cache-bytes 0`) so no run could hit a cache, each series after
the n-gram preload had finished. On AC, swap 1.78 GB of 3 GB before and
after. Differences from the `38d2642` series: 20 s between runs instead of
10, `--max-seq` 262144 instead of 131072, a corpus grown with the
repository, and the `38d2642` plain rows came from a series before the
activity assertion.

| context | prefill, draft head loaded | prefill, `--mtp-drafts 0` | decode, 2 drafts | decode, no drafts |
|--------:|---------------------------:|--------------------------:|-----------------:|------------------:|
| 4 096   | 2 387 (2 379 to 2 426) | 2 582 (2 568 to 2 587) | 110.0 (108.0 to 110.2) | 85.4 (84.8 to 85.9) |
| 16 384  | 2 534 (2 533 to 2 574) | 2 647 (2 617 to 2 649) | 105.0 (99.6 to 106.1)  | 84.3 (84.1 to 86.7) |
| 32 768  | 2 495 (2 480 to 2 503) | 2 582 (2 579 to 2 598) | 105.2 (104.3 to 106.5) | 83.4 (83.3 to 83.9) |
| 65 536  | 2 335 (2 310 to 2 367) | 2 448 (2 376 to 2 479) | 97.1 (96.5 to 102.9)   | 77.1 (76.4 to 79.7) |

Against `38d2642` (the next section's lily rows):

| context | prefill | decode, 2 drafts | decode, no drafts |
|--------:|--------:|-----------------:|------------------:|
| 4 096   | 2 387 against 1 435 (+66%) | 110.0 against 98.1 (+12%)  | 85.4 against 86.7 (-1%) |
| 16 384  | 2 534 against 1 756 (+44%) | 105.0 against 97.2 (+8%)   | 84.3 against 85.4 (-1%) |
| 32 768  | 2 495 against 1 864 (+34%) | 105.2 against 102.0 (+3%)  | 83.4 against 84.8 (-2%) |
| 65 536  | 2 335 against 1 651 (+41%) | 97.1 against 92.4 (+5%)    | 77.1 against 81.7 (-6%) |

- **Prefill**: the gain is the prefill work since 2026-09-18 ("What was
  done", "Prefill"; this series does not attribute it). The GPU phase is
  90 to 99% of `prefill_ms` (1.55 s of 1.72 s at 4K, 27.7 of 28.1 s at
  64K), the n-gram gather 95 to 110 ms with 0.6 to 4.7 K cold pages per
  run, `wait` 7 to 35 ms. The draft head, caught up during prefill, costs
  3 to 8% of it.
- **Decode with two drafts** accepted 1 717 of 2 686 drafts (64%), the same
  rate as the `38d2642` series, so the gain is shorter steps or a smaller
  clock sag, not more accepted drafts; the longer cooldown may account for
  part of it, which this series does not separate.
- **Plain decode** is within the noise band at 4K to 32K. **The 64K cell is
  6% below `38d2642`'s** (76.4 to 79.7 against 81.7), outside the band; not
  investigated here, and the two series differ in `--max-seq` and corpus
  as listed above. An A/B at 64K against `38d2642` would settle whether the
  engine moved.

Records: `docs/bench/2026-10-01-readme-series/` (not published):
`lily-d2.jsonl`, `lily-d0.jsonl`, `run.log`, `server-d*.log`, `report.txt`,
swap before and after each series.

### Against llama.cpp

Measured with `tools/bench/http_bench.py`, which drives both servers over
HTTP with the same prompts and reads each engine's own timings. The other
side is Unsloth's llama.cpp fork (build b11007, Unsloth Studio 2026.9.5)
serving `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ4_XS, 87 GB, KV cache f16,
four slots over a unified 131 072-token context, flash attention off because
the fork aborts at startup with it on for this model, measured 2026-09-17.
lily's rows are from 2026-09-18 at commit `38d2642` with `--max-seq 131072`
and the disk tier off so that no run could hit a cache (the records under
`docs/bench/2026-09-18-http/`); its rows of 2026-09-17 at commit `0c9ee63`
are kept in the second table for the record.

- **Prompts.** Cut from this repository's documentation and source with the
  model's tokenizer, a fresh region of the corpus for every run, at 1K, 4K,
  16K, 32K and 64K tokens. The 1K prefill column is not reported: it sits
  inside the GPU's clock ramp (see above).
- **Generation.** 256 greedy tokens of new text, a task unrelated to the
  prompt. The fork's default speculation is an n-gram drafter that copies
  from the context; on a "continue the document" task it reached full
  acceptance and four times plain decode by copying, which says nothing about
  generating. On new text it gains nothing (11% acceptance) and is left out.
- **Repeats.** Three, interleaved per repeat, ten seconds between runs;
  medians.
- **Both models at once do not fit** in 128 GB, so the two sides ran one
  after the other with the other server stopped.

Prefill, tok/s:

| context | lily `38d2642` | lily `0c9ee63` | llama.cpp | ratio, current |
|--------:|---------------:|---------------:|----------:|---------------:|
| 4 096   | 1 435 | 1 300 | 887 | 1.62x |
| 16 384  | 1 756 | 1 546 | 882 | 1.99x |
| 32 768  | 1 864 | 1 499 | 713 | 2.61x |
| 65 536  | 1 651 | 1 382 | 550 | 3.00x |

Decode, tok/s:

| context | lily plain | lily 2 drafts | llama.cpp plain | llama.cpp MTP 2 | ratio, best against best |
|--------:|-----------:|--------------:|----------------:|----------------:|-------------------------:|
| 1 024   | 85.8 | 105.8 | 42.0 | 54.7 | 1.93x |
| 4 096   | 86.7 | 98.1 | 39.1 | 51.8 | 1.89x |
| 16 384  | 85.4 | 97.2 | 31.4 | 44.1 | 2.20x |
| 32 768  | 84.8 | 102.0 | 25.2 | 37.0 | 2.76x |
| 65 536  | 81.7 | 92.4 | 17.0 | 27.0 | 3.42x |

lily's prefill column is the server as shipped, with the draft head
loaded and caught up during prefill; without it (`--mtp-drafts 0`) a
series measured 1 471 / 1 807 / 1 811 / 1 710. Its decode rows of
2026-09-17 were 90.2 / 86.0 / 85.5 / 83.9 / 81.8 plain and 98.9 / 101.7 /
102.0 / 101.5 / 97.7 with two drafts. The 2-draft and prefill rows are
medians over a seven-minute series with the activity assertion held (the
series that found the clock sag, below, ran before it existed) and carry
the sustained-load clock sag that remains: the first repeat, at 1 620 MHz,
decoded 113 / 105 / 109 / 109 / 90 tok/s with two drafts (20.4 to 21.5 ms
per speculative step, the 64K cell at 24.2 with the clock already down),
the second and third at 1 490 to 1 620 MHz 22.0 to 23.4 ms per step with
the same drafts accepted. The plain rows, from the series before the
assertion, do not move with the clock, because a plain step is
memory-bound. Draft
acceptance at two drafts per step was 64% on lily over the 2026-09-18
runs (63% on 2026-09-17) and 66% on the fork, so the head behaves alike in
both engines. At three drafts both fell to 50 to 53% and decoded slower than
at two (lily 83 to 94 tok/s, the fork 25 to 51), so three is not worth it on
either.

**The fork's MTP needs a patched build.** Unsloth Studio's shipped build
aborts while loading the MTP drafter (`GGML_ASSERT(ggml_can_repeat)` in the
qwen4exp MTP graph; the draft head's `hc_head_norm` tensor is declared with
a shape the trunk's norms moved away from) and then starts a third time
without any drafter while its settings page still says MTP. The bug is
reported on the fork's MTP pull request and in unslothai/unsloth issue
11143 with a one-line fix; the MTP rows above come from a build of the same
source with that fix. Everything else is the shipped build.

**Prompt cache behaviour**, tokens recomputed out of the prompt, from the
harness's `cache` test with lily's disk tier on (2026-09-17):

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
810 tok/s prefill at 131K. oMLX 0.7.0 (released 2026-09-30) publishes for
Qwen3.8-Flash-Next oQ4e on an M5 Max 128 GB, in the chart of its release
notes, prefill 2 768 / 2 844 / 2 366 tok/s at 4K / 16K / 64K and generation
93.0 / 87.1 / 75.9 tok/s, sampling settings not stated (the benchmarks in
its pull requests used temperature 1.0). lily's 2026-10-01 series at the
same lengths: prefill 2 387 / 2 534 / 2 335 with the draft head (2 582 /
2 647 / 2 448 without), decode 110.0 / 105.0 / 97.1 with two drafts and
85.4 / 84.3 / 77.1 plain, all greedy. None of the third-party figures were
measured with this harness, and the quantizations differ, so they are
cited, not compared: on these numbers oMLX's prefill is ahead by 12 to 16%
at 4K and 16K (7% against lily without the draft head) and level at 64K;
its generation figures are above lily's plain rate at 4K and 16K, just
below it at 64K, and below the 2-draft rate at all three. lily's 2-draft
rate is measured greedily, which is worth about 10% over sampled
drafting.

### Smaller machines

On a machine whose memory cannot hold the checkpoint the engine keeps a
usage-ranked share of the routed experts on the GPU and reads the rest from
the checkpoint as they are routed to (`docs/low-ram-experts.md`). Measured
with the reads cold under a 60 GB locked balloon, as on a 64 GB machine,
on the 8K real-text prompt with 256 generated tokens, digests identical to
the resident path: 929 tok/s prefill and 64.4 tok/s plain
decode at commit `38d2642` (977 and 54.6 at `37cc34c` the day before; the
cold runs' misses depend on what the page cache still holds), against 2 257 and
87 with everything resident. Speculative decoding is off under the cache
because its three trunk passes per step each pay the per-layer handshake
and the misses (14 to 15 tok/s with it on).

## What was done

Grouped by the phase it serves. Each item names the measurement that kept
it; the kernel-level detail is in `docs/architecture.md`.

### Prefill

A 4 096-token chunk reads the weights once (17.4 MB per token) and is bound
by compute and kernel efficiency, not bandwidth. Where a chunk of an 8K
real-text prompt goes, per pass in the per-kernel profile, `37cc34c`
against `38d2642` in one clock state (the dense GEMM, untouched, reads
557 against 554 ms in the two):

| kernel | `37cc34c` ms | `38d2642` ms | share now |
|---|---:|---:|---:|
| grouped Q4 expert GEMM (`gemm_q4_nt_nax_grouped_t64x4`) | 669 | 572 | 34% |
| dense bf16 tensor-op GEMM (`gemm_bf16_nt_nax`) | 558 | 554 | 33% |
| sparse attention past the limit (`qsa_attn_tile_nax_h1`, then `qsa_attn_rows_nax_h1` + `qsa_tile_gather`) | 278 | 134 + 29 | 10% |
| GDN prefill scan (`gdn_prefill_regscan`, then `gdn_chunk_scan` + `gdn_chunk_wy3`) | 176 | 72 + 31 | 6% |
| hyper-connection mix and inject | 81 | 79 | 5% |
| norms, the MoE row gather and combine, convolutions, gates, scores, selection, dequant | 236 | 233 | 14% |
| kernel sum per pass | 1 998 | 1 702 | |

The dense GEMM runs at 54 TFLOP/s, the ceiling this GPU reached on any
shape; the expert GEMM at about 34 by the same accounting in that table,
about 38 since item 0. The kept work, in order of size:

0. **The expert GEMM's B-tile staging** (2026-10-01). In a harness at the
   model's shapes (512 experts, four layers' measured routing profiles,
   `grouped_expert_gemm_timing`) ablations put the cost in the weight
   loads, not the dequantization: filling the tile with the loaded words
   undequantized kept 96% of the kernel's time, dequantizing register
   values without loading them cost 8%, and no tile refresh at all ran at
   51 TFLOP/s. Three changes, each 0 to 9% alone: 128-deep K steps (64
   contiguous bytes of each weight row per step, half the barriers; a
   64-deep step closes a K that is an odd multiple of 64), one uint4 run
   of 32 codes per thread with one scale and bias instead of single code
   words, and the B tile's rows padded by 8 elements (bank spread for the
   tile's columns). Together, bit-identical to the previous kernel
   (kept test-only in `tests/metal/quant_test.metal`), harness ms per
   layer at 4 096-token chunks: gate+up 7.92 to 6.79 (33.9 to 39.5
   TFLOP/s), down 4.17 to 3.60 (32.2 to 37.3); at 8 192 (160 rows per
   expert) 39.0 to 45.1 and 36.1 to 41.2. In the model's profile at a 16K
   real-text prompt per chunk: gate+up 397 to 338 ms (32.4 to 38.1
   TFLOP/s), down 208 to 179 (31.0 to 36.0). lily-bench, ABBA x3: +5.4%
   at 4K, +6.0% at 16K; HTTP with the two items below, fresh prompts, a
   fresh server per session, 3 per build: 4K 2 394 to 2 538 tok/s, 16K
   2 327 to 2 463, 64K 2 249 to 2 395 (+6.0 / +5.8 / +6.5%). In the same
   change set, bit-identical: the MoE
   combine and the gated shared-expert add in one pass (32.7 to 23.6 ms per
   chunk) and the expert offset scan over a threadgroup (2.4 to 0.27 ms).

1. **Sparse attention and indexer scores per query on the tensor ops**
   (2026-10-01). Past 16K the tiled route of item 2 was what grew with
   depth: its union of 16 queries' selections, all masked work beyond one
   query's own blocks, and the copy of the union's rows. At a 64K real-text
   prompt the chunk from 60K to 64K spent 505 + 150 ms in the tiled
   attention and its gather, and 216 ms in the scalar indexer scores, of a
   2 653 ms kernel sum (1 795 for the first chunk). `qsa_attn_gqa_nax` runs
   one threadgroup per (query, KV head) over the query's own blocks and
   tail, the GQA group's 12 heads as the rows of a 16-row tensor-op tile, K
   and V read from the cache into the register operands, no scratch
   (architecture.md, "Prefill"); `qsa_scores_nax` scores a sub-batch as a
   GEMM with the ReLU and head sum in the epilogue. In the same profile the
   last chunk's attention takes 176 ms and its scores 39, the mean
   attention per chunk 161 ms against 480, the mean scores 21 against 114;
   the kernel sum of the last chunk is 2 133 ms, of which only 175 is QSA
   growth over the first chunk (815 before). In the harness (256 queries,
   2 KV heads, 512-block budget) the attention takes 0.67 / 0.69 / 0.71 ms
   at 8K / 32K / 64K against 0.55 + 1.95, 1.19 + 3.30 and 1.51 + 3.67 for
   the tiled gather + attention: flat in depth, about 18 useful TFLOP/s.
   lily-bench, real text (a cargo-registry corpus), interleaved ABBA, three
   repeats, against `3cfdd49`; the 64-step plain decode of each run is the
   clock canary (about 80 tok/s in the full clock state, 64 in the
   sustained-load sag, the code path is unchanged), so pairs are read
   within a state:

   | prompt | before, tok/s | after | change, same clock state |
   |---|---:|---:|---:|
   | 4K | 2 269 / 2 342 | 2 588 / 2 590 | +12% |
   | 16K | 2 212 to 2 224 | 2 399 to 2 422 | +8% (all three pairs) |
   | 32K | 1 879 to 1 963 | 2 252 | +15 to 19% (the other two runs sagged: 1 650 / 1 681) |
   | 64K | 1 812 (full clock), 1 420 / 1 443 (sagged) | 2 110, 1 721 / 1 741 | +16%, +21% |

   Over HTTP (`tools/bench/http_bench.py matrix`, fresh prompts from
   cargo-registry regions no earlier run used, a fresh server per session
   with no disk tier and the n-gram preload awaited, three sessions per
   build alternating, 20 s apart), median (min to max) prefill tok/s and
   GPU s:

   | prompt | before | after | GPU s before / after |
   |---|---:|---:|---:|
   | 4K (one chunk plus a 7-token tail) | 2 236 (2 062 to 2 277) | 2 225 (2 176 to 2 357) | 1.77 / 1.76 |
   | 16K | 2 070 (2 038 to 2 078) | 2 207 (2 145 to 2 290), +7% | 7.53 / 7.04 |
   | 64K | 1 785 (1 757 to 1 825) | 2 255 (2 246 to 2 258), +26% | 36.3 / 28.7 |

   Decode is not changed: batches under 16 rows (decode, verify, the draft
   head's steps) keep the scalar scores and the split kernel.

   Numerics: the attention's online softmax now steps over a query's own
   keys 32 at a time (the tiled route stepped over union rows 128 at a
   time), and the scores accumulate in another order, so a block near the
   512th score can change sides. lily-probe top-64 logits on six real
   prompts (three corpora, 16K and 64K), teacher-forced on `3cfdd49`'s
   greedy path for 9 steps, against the band `3cfdd49`'s own variants span
   (`LILY_QSA_ROUTE=split`, `LILY_PREFILL_CHUNK=8192`): the last prefill
   row's max |dlogit| 1.90 / 0.70 / 1.34 / 1.86 / 0.86 / 1.12 against the
   variants' 1.83 / 0.76 / 2.15 / 2.52 / 1.63 / 1.86; argmax changes only
   at near-ties (reference margins 0.40 and 0.57, and one decode step of the
   mixed 64K prompt), as the split variant's do on two of the six prompts.
   With the scalar scores kept (`LILY_QSA_SCORES=scalar`) the attention
   alone stays at 0.66 to 1.57.
2. **Tiled sparse attention over gathered rows** (replaced by item 1 on
   2026-10-01; kept here for its measurements). Past the dense limit the
   per-query kernel attended to each query's 512 selected blocks with no
   reuse and no tensor operations, 40% of the chunk at 2.1 TFLOP/s. The
   tiled route merges 16 consecutive queries' selections into one block list
   with a query mask per block (on real text the union is 1.9 times one
   query's selection at 8K and 3.2 at 32K, against 16 for disjoint
   selections), and then, after a first tiled kernel that staged the union's
   rows into threadgroup memory 32 at a time, `qsa_tile_gather` copies the
   union's K and V rows once per KV head into a contiguous 512 MB device
   scratch and `qsa_attn_rows_nax_h1` runs the dense kernel's tensor-op loop
   over them, one query head per threadgroup. Per pass past the limit in the
   full model's profile: 1 200 ms (per query) to 290 (staged tiles) to
   153 + 35 (gathered rows) at 8K, 1 480 to about 570 to about 300 + 78 at
   32K. The gathered-row kernel runs at about 25 TFLOP/s, the dense kernel's
   rate; the gather is bandwidth-bound at 400 to 460 GB/s. Its rounding
   differs from the per-query kernel's (a different slice height changes the
   online softmax's order); the 4-layer tile-against-split comparison stays
   within 0.011 on a logit scale of 2.93. The rows of a chunk whose causal
   window fits the budget take the dense kernel, which is exact for them.
3. **The expert GEMM's last tiles.** With about 80 routed rows per expert per
   chunk, 64-row tiles padded the rows by 47% and the tensor work on the
   padding was what the kernel paid for (a row-tile sweep, 32 / 64 / 96 /
   128 rows, had put every other height further behind). An expert's last
   tile now runs its product at the smallest of the full, half and quarter
   heights that covers its rows, bit-identical: padding 10%, the three
   grouped GEMMs per 8K prefill 573 to 583 ms against 666 to 677 in paired
   profiles (14% off the kernel), 628 against 710 per 32K prefill, 229 to
   232 against 246 per 1K prefill.
4. **The GDN prefill scan in chunked form on the tensor ops.** The scan over
   a chunk's rows was a token-serial recurrence, one simdgroup per head; it
   first gained four value columns per simdgroup (384 to 281 ms per 8K chunk
   in the profile), and then left the serial form for batches of 128 rows or
   more: 64-token chunks in WY form, a parallel pass forming and inverting
   each chunk's `I + A` and a sequential pass per head and block of 16 state
   columns carrying the fp32 state. In paired profiles per 8K prefill 99 to
   109 ms against 171 to 190, per 1K 26 against 51, per 32K 116 to 125
   against 191. The scan pass is bound by streaming each chunk's rows into
   the cores, not by the products (with cache-resident operands it took
   0.49 of its 1.9 ms). Against the per-token CPU reference its outputs are
   as close as the serial scan's and its final state within 0.5% of scale
   (serial 0.2%); the 4-layer golden gaps went from 0.025 / 0.043 to 0.028 /
   0.045 with argmax agreement unchanged; on the full model its logits sit
   inside the band the serial variants already span (above). A batch's
   ragged tail continues through the serial scan, which also serves the
   verify passes and records the per-row states the rollback needs.
5. **The block selection without serial steps**: a radix select over the
   block scores with scan-picked digits and one compaction scan, 15.6 to
   2.6 ms per 8K prefill and 30.1 to 13.1 per 32K; then 512 threads per
   query. **The MoE input gather** moving eight bf16 elements per thread on
   its aligned rows, 1.07 to 0.52 ms per call, about 26 ms per 8K chunk.
   **The dense rows below the limit** through the dense kernel.
6. **The n-gram staging off the critical path** (host side, bit-identical;
   `docs/architecture.md`, "Memory layout"). With the table partly cold, the
   state the service is in under interactive use (18 of 32 GB resident),
   staging a fresh prompt's rows cost 0.6 / 1.7 / 3.4 s at 4K / 16K / 64K.
   Two changes: a prompt of more than one chunk stages chunk k+1's rows into
   a second buffer while the GPU runs chunk k, and a prefill batch whose
   1-in-16 residency sample finds at least one page in 32 cold hints every
   row's pages, not only the sample's, so the SSD reads overlap and the
   pager's read-around no longer reads pages nobody asked for (a cold chunk
   of random rows: 1.03 to 0.35 s, 205 000 to 88 000 pages read; a warm
   chunk unchanged at 5.6 ms). Over HTTP, interleaved A/B against `00b6b3d`,
   five server launches per build, fresh prompts from corpus regions no
   other run used, the table put into the same partly cold state before
   every launch (`tools/bench/evict_table.py 0.6`, about 17 GB resident, the
   preload off), medians with min to max:

   | prompt | prefill tok/s before | after | staging exposed, ms, before | after | sampled cold pages before / after |
   |---|---:|---:|---:|---:|---:|
   | 4K | 1 781 (1 697 to 2 152) | 2 133 (2 072 to 2 255) | 593 | 210 | 2 399 / 2 654 |
   | 16K | 1 769 (1 732 to 1 834) | 2 089 (2 052 to 2 103) | 1 663 | 179 | 5 414 / 5 552 |
   | 64K | 1 655 (1 651 to 1 677) | 1 797 (1 760 to 1 832) | 3 447 | 142 | 12 145 / 12 567 |

   GPU time is the same in both builds. What stays exposed is the first
   chunk's staging, now 0.1 to 0.27 s. The `wait_ms` of a multi-chunk
   request grew by 0.1 to 0.2 s at the same time: the first chunk is now
   committed that much sooner after the request arrived, and a stall that
   ends a fixed time after arrival used to pass while the host staged (a
   500 ms pause before the first commit removes it; most likely residency
   coming back after idle, `docs/architecture.md`, "Residency does not
   survive an idle second", not verified). The totals above include it.
   The 4K prompt is one full chunk plus a 7-token tail, so its gain is the
   denser hints alone; a single-chunk agent turn gets that part too.

### Decode

A decode step reads about 4.4 GB for one token (4.135 GB of weights, the
226 MB of GDN state, 25 to 50 MB of attention caches), so it is bandwidth
work, and the engine's job is not to waste bandwidth and not to wait
between steps. At `38d2642` a 1K step takes 10.5 to 10.7 ms in the fast
clock state, about 415 GB/s. What got it there:

1. **The transport.** One command buffer per step, level barriers between
   dependency levels instead of a barrier after every dispatch, the next
   step encoded and committed while the current one runs and parked on a
   shared event until the host has staged its n-gram rows, the accepted
   draft count and the chained positions decided on the GPU and read by
   later dispatches as inline parameters. The host is not on the critical
   path of either loop; the measured GPU idle between a verify and a draft
   pass is 0.013 ms.
2. **The hyper-connection read** fused from six dispatches to two, with the
   RMS normalization folded into the down projection; **the sampler** on the
   GPU with a two-phase top-k (64 slices, then the union) so that only the
   token id crosses to the host.
3. **Sparse attention at decode.** 64-token splits and one threadgroup per
   four query heads (1.64 to 0.49 ms per 8K step for the kernel, the combine
   0.07 to 0.17); the block selection without serial steps (0.41 to 0.21 ms
   per 8K step, 0.70 to 0.59 at 32K) and then over 512 threads per query
   (16.7 to 7.8 us per query at 8K, 31.3 to 15.7 at 32K; 1 024 threads gave
   the gain back); the split kernel's latency chains cut (token ids staged
   once, four rows in flight, one barrier for the four heads' softmax sums):
   split plus combine 48.9 to 42.5 us per layer at 8K and 86.2 to 76.8 at
   32K. All bit-identical.
4. **The GDN step** with the state walked in register blocks of eight rows,
   19 to 16.3 us per layer; **the MoE down gather** with four simdgroups per
   row pair taking alternate routed slots, 23.8 to 22.3 us per layer (385 to
   413 GB/s), the slot sums combined in slot order so the result is
   bit-identical. Both under the step's noise on their own; together with
   item 3 about one percent at 1K and 2.6% at 32K in the matrix above.
5. **The verify pass's skinny GEMMs** computing two rows per simdgroup with
   each activation block loaded once, 5.80 to 5.46 ms per profiled 3-row
   pass (the Q8 one 1.41 to 1.14). Then four weight rows per simdgroup from
   m = 3, each activation block converted and summed once for all four (the
   activation side, not the weight bytes, bounded the two-row kernel past two
   rows), the last partial pass split between the half-simdgroups, and the
   arithmetic pinned in safe math to the earlier kernel's order, bit for bit:
   the model's projections per batched pass 3.72 to 3.31 ms at three rows
   and 4.49 to 3.82 at four, the LM head 0.73 to 0.63 and 0.91 to 0.69
   (`shared_weight_kernels_vs_rows_probe`).

### Speculative decoding

The draft head's proposals verified in one batched trunk pass with
acceptance by exact equality; the accepted count decided on the GPU; the
rollback from recorded per-row GDN states with no recomputation; and, under
sampling, the head drawing its proposals from its own distribution with the
request's sampler and the verify rows running exact speculative sampling
against them (`min(1, p / q)`, the residual `max(0, p - q)`), which changes
no output distribution and accepts 65 to 67% of sampled proposals against 55
to 65% for argmax proposals on the synthetic prompt. Two drafts per step.

### Smaller machines

The expert cache: a slab of expert slots on the GPU, a slot table per layer,
a service thread that resolves each MoE layer's routed experts between the
router and the gather through a shared-event handshake, placement by a
usage ranking with a small LRU region, misses read straight into the slot
on 16 threads. Engaged from physical memory, `--memory-limit-gb` or
`LILY_MEMORY_GB` (a simulated smaller machine); nothing
changes on a machine that fits the checkpoint. `docs/low-ram-experts.md`.

## What was tried and dropped

Every one of these measured slower, or equal, in a paired comparison, and
none is kept behind a knob. The reason is what the measurement said.

### Prefill

| tried | measured | why it lost |
|---|---|---|
| grouped expert GEMM with a 128-deep K step alone (half the B-tile barriers per FLOP) | within 2% of the 64-deep one | kept since 2026-10-01 together with run loads and a padded tile, which is where it pays |
| expert GEMM B tile double-buffered (next step's weights loaded into registers across the product), with one or two tiles | 8.1 to 11.9 against 7.5 to 7.9 ms gate+up per layer; with one tile on the shipped kernel 6.93 against 6.82 | registers held across the product; the product does not overlap the loads |
| expert GEMM warp-specialized: 4 consumer simdgroups on single-simdgroup products, 2 or 4 producer simdgroups streaming the next tile | 11.0 / 8.0 against 7.9 ms (bit-identical) | two producers cannot keep up; four equal the plain kernel |
| expert GEMM grid with a row tile's column blocks on one grid x (oMLX's tile-on-x), or expert-major block order | 4 to 5% slower; equal | not this GPU's lever at 40 cores |
| expert GEMM 128-row tiles with 16/32/64/96/128 heights, 4 or 8 simdgroups; a 48-row height | equal; 48 rows 55% slower | re-reads of an expert's weights already hit the cache |
| gate and up in one kernel with the SwiGLU epilogue (two accumulators on 4 simdgroups, or gate and up on 8 simdgroups exchanging through threadgroup memory; bit-identical) | 8.4 to 8.8 against 7.1 ms for both GEMMs plus `silu_mul` | twice the accumulator registers per thread, or half the threadgroups per core |
| the gate/up GEMM reading token rows through the sorted row map, A staged in threadgroup memory 64 K at a time (bit-identical) | 10.1 against 7.3 ms for gather plus both GEMMs | a 128-deep A tile and the B tile do not fit 32 KB together; staging A costs more than the 27 ms gather per chunk |
| grouped expert GEMM row tiles of 32, 96 and 128 rows | 645 / 659 / 698 to 759 ms against 593 to 622 per chunk | tensor work on padded rows; solved instead by the last-tile heights |
| a 32-query tile for the staged attention kernel | 4.68 against 3.89 ms per 256-query dispatch at 8K, 13.1 against 7.6 at 32K | the union grows faster than the reuse; the kernel was bound per slice, not per gather |
| the staged attention kernel with 8 simdgroups, 16-row slices, two staged slices, two heads per pass | all slower | same: a serial chain of gathered fetches, dependent tensor ops and barriers per slice |
| the staged attention kernel with 12 KB more threadgroup memory (an occupancy probe) | 12 to 14% slower | it already ran more than one threadgroup per core; no occupancy lever there |
| the gathered-row attention with 64-key slices | 2.49 against 1.98 ms | fewer keys per tensor op |
| the gather double-buffered (each group's gather on the previous group's level) | 3.14 against 2.76 and 5.78 against 5.06 ms by the clock | the two groups' rows compete for the cache that the single group fits |
| the gather split over eight threadgroups per tile | no change | bandwidth-bound either way |
| a row scratch of 1.1 GB (every tile at once) or 256 MB (three tiles) instead of 512 MB | equal at 32K and 7% slower at 8K; 60 to 70% slower | a group's rows fitting the cache matters more than dispatch width |
| the per-query attention with 64 keys per step (two score fragments per barrier) | 1.95 against 0.75 ms per 256-query dispatch at 64K | register pressure: the tensor ops take register operands only at 16 or 32 per side, and two steps' operands spill |
| the per-query attention with 16-byte K/V/Q loads (the head dims permuted within each 32-dim chunk) | 1.03 to 1.05 against 0.69 to 0.72 ms (medians, 8K to 64K) | slower despite half the load instructions |
| the per-query attention with the step's V rows requested before the score exchange; Q held in registers across steps | 0.75 to 0.78 and 0.77 to 0.81 against 0.69 to 0.72 ms | registers held across the step cost more occupancy than the latency they hide |
| the per-query attention over 1 024- or 4 096-query sub-batches (fewer dispatches) | 2.6 to 2.8 us per query at every size | a 256-query dispatch already fills the GPU for its duration |
| the chunked GDN scan with 8, 32, 64 or 128 state columns per threadgroup | 3.8 / 2.2 / 2.8 / 3.2 ms against 1.9 (16 columns) | narrower doubles the row bytes, wider runs too few threadgroups to hide the dependent-product latency |
| the chunked GDN scan with eight simdgroups, pre-transposed keys, next-chunk prefetch touches, fp16 operand copies | no gain | the pass streams rows; none of these change the bytes |
| the four-column serial scan kept for batches over 128 rows | 171 to 190 against 99 to 109 ms per 8K prefill | the serial recurrence |
| fusing the elementwise glue (about 9% of a chunk) | not built | the profile shows no dispatch paying a launch floor at these sizes |
| staging ahead only once the GPU signalled that the chunk began (a queue signal after the embedding gather) | the first chunk's extra 0.1 to 0.2 s of wait unchanged | the staging did not delay the pass; the wait is the post-idle wake the old staging hid |
| overlapping the first chunk's gather with the GPU work before it (parking the pass at the n-gram gather) | not built | only the embedding and two GDN layers, about 4% of the chunk, precede the gather in layer 2, and the park would sit inside the GPU span |
| more copy threads for a cold prefill gather (16, 32, 64 with one row in 16 hinted) | 0.69 / 0.55 / 0.50 s against 1.03 (8 threads); every row hinted with 8 threads 0.35 s, with 32 0.44 | the faults, not the thread count, were the cost; the `mincore` calls do not scale across threads |

### Decode

| tried | measured | why it lost |
|---|---|---|
| the write-gate inject folded into the branch matvec's and the MoE combine's epilogues (96 dispatches fewer, bit-identical) | 6.6 to 6.8 ms over the three kernels against 6.3 to 6.7 | the epilogues cost what the inject kernel cost; a tiny level between streaming levels is free |
| same-level matvecs folded into one four-set GEMV dispatch (router, shared expert, attention and indexer projections, PLE keys and values: 110 dispatches fewer, bit-identical) | 92.8 against 93.2 tok/s at 1K over three interleaved pairs, within noise at 8K | dispatches on one level already overlap; only levels cost |
| the router GEMV with the top-k run by its last threadgroup (48 levels fewer, 661 against 709) | 10.55 / 10.61 against 10.54 / 10.54 ms per step; one wrong selection in 2 400 chained dispatches | the top-k level was free (its 0.35 ms ablation figure was an artifact of stale routing hitting cache); the cross-threadgroup handoff is not trustworthy on this GPU |
| expert gathers with every weight block requested up front | 2.30 against 1.21 ms (down) and 2.08 against 1.90 (gate/up) per 1K step, 82 against 92 tok/s | the registers the prefetch holds cost more occupancy than the stalls it hides |
| the down gather with two routed slots per iteration; one row per simdgroup over 32 lanes; five, eight or ten simdgroups per row pair | 22.9 against 23.7 us; 23.6 to 23.9; 22.3 to 22.8 | at the level ceiling for its 9 MB; four simdgroups took the last 1.5 us |
| the hyper-connection down kernel's streams over two simdgroups; its five weight blocks per lane requested up front; an unroll pragma; the up kernel's epilogue loads hoisted; the reads with their activation or weight loads stripped | 9.2 to 14.2 us against 8.7 to 10.6; the stripped variants no faster | within 10% of the 4 MB level ceiling; neither the activations nor the weights are the limiter |
| the decode matvecs through the one-row register-A skinny kernel | 4.08 to 4.12 ms against 3.90 to 4.07 for the 192 Q4 projections; the routers 0.64 against 0.50 | isolation rates (three times faster on the routers) do not survive the chain |
| the split attention kernel with a barrier-free lane-per-token design; 32-token splits; a 128-thread threadgroup; the combine's prologue parallelized | 82 us against 49; 58; 3 us faster but reorders the reductions; noise and not bit-identical | the barriers and staging were not the cost |
| the split attention kernel with eight K rows in flight and the V rows requested before the softmax barriers; one threadgroup per K/V head and split with no cross-simdgroup staging (bit-identical) | 49.5 against 44.4 us; 42.8 against 44.4 at 8K and 43.3 against 44.4 at 32K | register pressure; and 1.6 us was not worth a new kernel, since what remains is the per-token score and reduction chain |
| the GDN step with 16- or 32-row register blocks, or the head split over thread groups or column groups | equal, not bit-identical; 15 to 16 us | no faster than eight rows |
| the pacer's sleeps scaled by the measured 25% `nanosleep` overshoot | 93.1 against 93.1 tok/s | the shipped pacer's own adaptation already covers it |
| three drafts per step | 100.8 against 113.9 tok/s greedy, 96.6 to 106.1 against 101.3 to 111.9 sampled | the third draft is accepted too rarely for its verify row and draft pass |

### Smaller machines

Transparent paging of memory-mapped GPU buffers (the GPU faults random
2.8 MB expert slices at 64 ms each, 16 KB pages one at a time), a blocking
wait in the cache's service thread (30% of decode against 7% spinning), and
speculative decoding under the cache (14 to 15 tok/s against 55 plain).

## Where the decode step's time goes

Measured 2026-09-18 with two tools that time the production shape, because
the per-kernel profile mode (one command buffer per dispatch) sums to 14.1
ms for a 1K step that runs in 10.7 and cannot say where the gap to the
bandwidth floor sits.

**The level chain.** `lily-bench --gpu-timing` counts the dependency levels
(barriers) a step encodes: 709 at 1K, 733 past the dense limit. A pure
streaming read, one dispatch per barrier-separated level over distinct
regions (`level_size_bandwidth_probe`), reaches 600 GB/s with no barriers
but less when each level is its own dispatch behind a barrier:

| bytes per level | GB/s       | at peak | lost per level |
|-----------------|------------|---------|----------------|
| no barriers     | 600        |         |                |
| 8 MB            | 479 to 487 | 13.3 us | about 3 us     |
| 4 MB            | 459        | 6.7 us  | about 2 us     |
| 2 MB            | 306 to 404 | 3.3 us  | 2 to 3 us      |
| 1 MB            | 176        | 1.7 us  | about 4 us     |

Every streaming level loses a roughly fixed 2 to 4 us, so a level's
efficiency is set by how many bytes it amortizes that over. The step's
floor is therefore the sum of its levels' bytes at those per-size
ceilings, not the 4.37 GB it reads at 600 GB/s (7.3 ms); at the mix of
level sizes the step has (3 to 24 MB each), that floor is about 9.3 ms,
and the step runs within 10 to 15% of it.

**What a level pays, and why.** Bandwidth is bytes in flight divided by
latency: RAM delivers its 600 GB/s only while enough load requests are
queued against it, and with a load-to-data latency of a few hundred
nanoseconds that is on the order of 300 KB outstanding at every instant
(an estimate; Apple documents neither figure). Those outstanding loads
land in registers, and a thread can hold only a handful before it runs
out of them (the "every weight block requested up front" variants in the
table above lost occupancy that way), so the 300 KB comes from many
resident threadgroups each with a few loads pending, not from any thread
being deep. The caches are a pass-through for the weights, which a GEMV
touches once; they hold the small reused things, the activation vector a
level reads and the output vector it writes, which the next level reads
back from cache rather than RAM. Three dependent levels, A and C streaming
8 MB of weights each and B a tiny inject kernel between them:

```
time  ------------------------------------------------------------->
A     [ramp][===== stream at ~600 GB/s =====][drain]|
B                                                   [B]|
C                                                      [ramp][===== stream =====][drain]|
                                                    ^barrier ^barrier
bytes in flight
      0 ..rising.. ~300 KB ..steady.. falling.. 0   0  0 ..rising.. ~300 KB ..steady..
```

The bandwidth at any instant is the in-flight line divided by the latency,
so RAM runs at peak only along the steady stretch. At the start of A the
queues are empty: the scheduler places threadgroups on cores, each thread
computes its addresses and issues its first loads, and the request count
climbs toward the steady level. Near the end the threadgroups finish at
different times, the last few run alone with only their own loads
pending, the count falls, and then the outputs are written and made
visible. C's weight reads depend on nothing A produced, but the barrier
orders all memory, so the GPU cannot issue them early: the in-flight count
is forced to zero at every level boundary and each streaming level starts
cold and ends cold. That is the 2 to 4 us per level in the table. B is
free because it reads tens of KB out of cache and finishes inside the
bubble the boundary costs anyway; removing it merges two bubbles into one
that A-to-C paid regardless, which is why the 96 inject levels ablate to
nothing and why folding same-level matvecs into one dispatch gained
nothing. The cost is per cold start of a streaming read, not per barrier.
The per-size ceilings, the free tiny levels, the same-level overlap and the
register-pressure losses are measurements; the queues filling and emptying
are the standard account of those numbers, not something observed.

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
| of which the shared-expert matvecs | 0.00 | 0.13 GB | overlap the router and gather levels |
| attention layers (12) | 0.90 at 1K, 1.58 at 8K, 2.76 at 32K | | |
| GDN conv + step + out projection | 1.57 | | |
| LM head | 0.57 | 0.36 GB | at peak |
| the 96 inject levels | 0.00 | | a tiny level between streaming kernels is free |

The top-k level is not in the table: skipping it "saved" 0.35 ms only
because the stale indices then routed every slot to one expert whose rows
the gathers read from cache (skipping the selection kernel alone "saved"
0.97 ms the same way); the fused-router experiment above measured its real
cost as nothing.

**Kernels in their chain.** The chained harnesses (`*_chain_timing`, one
dispatch per layer over distinct weights, a barrier between, warmed for 300
ms first because the clock ramps, best of several passes) put each family
against the level ceiling of its size: the expert gate/up gather at 35.4 us
for 18.4 MB (521 GB/s, at the 16 MB ceiling), the down gather 22.3 us for
9.2 MB (413), the hyper-connection pair 16.9 us for 7 MB as two 3.5 MB
levels (413 and 428 GB/s each, within 10% of the 4 MB ceiling), the GDN step
16.3 us for 6.2 MB of state, the dense GEMVs' 24 MB levels at the peak.
What remains at 1K is about 0.1 ms in the hyper-connection reads, one
percent.

**Attention past 2K.** The branch costs 0.90 ms at 1K, 1.58 at 8K and 2.76
at 32K. Per layer at 32K in the chain: block scores 12 us, the selection 16,
the split kernel about 40 and its combine 5 to 8. The split kernel's 40 us
for 4 MB of K/V is structural: with its V loads removed it measured 2.5 us
less and with its K loads removed 5.4 less, and the two redesigns in the
table above showed the barriers and staging are not the rest either, so
what remains is its per-token score and reduction chain, which only a
reordering of the arithmetic (rounding moves) could shorten. The kernel
measures the same at 8K and 32K in the chain; the 32K step's extra
attention cost is in the block selection (0.14 to 0.44 ms per step in the
profile) and the block scores.

## What is left, and what is out of scope

**Decode is at its floor for this design.** The step is a chain of about
709 barrier-separated levels, each a true data dependency, each reading 3
to 24 MB, and a streaming dispatch reaches only 460 to 485 GB/s at those
sizes because each one starts with no loads in flight and ends with none
(the section above). Every kernel family sits within about 10% of that
ceiling, every fusion and folding of levels measured a wash because tiny
levels are free and streaming levels pay their cold start regardless, and
the remaining per-kernel items are a percent each. The 2 ms between the
step and the 7.3 ms its bytes would take at 600 GB/s is the ramp and drain
of those levels. Three things would attack it, and none belongs in this
server:

- **Weight streams issued before the barrier.** The weights (4.4 GB of the
  traffic) depend on nothing; only the kilobytes of activations do. A
  persistent kernel whose threadgroups load the next level's weight rows
  and then spin on a device-memory counter until the previous level is done
  would keep the weight stream running through the ramp and drain. It is
  what the megakernel work on other hardware does, and here it would
  recover at most the in-model shortfalls of the hyper-connection reads and
  the gathers, about 0.5 to 0.6 ms (5% at 1K), bounded by how many weight
  bytes the waiting threadgroups can hold. Its failure mode is a GPU hang:
  a waiting threadgroup that occupies a core its producer needs deadlocks
  the GPU, Metal promises no forward progress, and the residency the scheme
  depends on belongs to whatever else the GPU is running, a browser tab or
  the compositor. On a dedicated inference box one could size the grid to
  residency; on the daily-driver laptop this server is meant to run on in
  the background, one cannot, and no measurement changes that.
- **Prefetch touches into the system level cache.** A dependency-free
  dispatch on each level reading the next level's weights so the dependent
  kernel finds them in cache. It rests on the cache keeping a level's bytes
  across a streaming neighbour, which nothing documents and which any other
  application streaming through the cache would undo: a performance that
  varies with what the user is doing, the wrong property for a background
  server.
- **Reordering the split attention kernel's score arithmetic** at long
  context, one to two percent at 32K, at the cost of moving the rounding
  of every decode step; and **continuous batching across sessions**, x1.5
  to x1.7 aggregate throughput for two concurrent sessions by the measured
  3 ms an extra verify row costs, which changes the engine's shape (per-row
  cache pointers and positions in every attention, GDN and n-gram kernel,
  per-row sampler state, a scheduler that admits and retires sessions per
  step) for a server that serves one agent at a time by design.

**Prefill has two levers left, both against ceilings already measured.**
The grouped expert GEMM runs at about 38 TFLOP/s against the dense GEMM's
54 on the same GPU, about 32% of the chunk. The ablations of 2026-10-01
say what separates them: with no B-tile refresh the same kernel runs at 51
to 54, so it is the weight stream, whose loads the tensor-op product does
not overlap (double buffering, register prefetch and producer simdgroups
all measured equal or slower). A kernel that hides them would need either
more resident threadgroups per core or loads that bypass the registers;
neither has a measured design here. The gathered-row attention runs at the dense
kernel's rate and its gather is at bandwidth, so the only lever there is
gathering less, which the tile union already sets. The GDN scan streams
rows at the level ceiling. The elementwise glue is 9% of a chunk in about
180 dispatches none of which pays a launch floor.

**Where bytes could be saved instead of time** (3-bit experts, Q4
hyper-connection mixers, 8-bit KV), the quality cost is **not measured**, and
the 4-layer Hugging Face comparison cannot see expert quantization error at
scale. A perplexity or task run on the full model would have to come first,
and the fork's goal is the model at this precision.

**Finer mixed precision, not tried.** The 8-bit groups (`--q8`) move whole
tensor kinds from 4 to 8 bits at group 64. oMLX's oQ4e conversion of the
same model (`Jundot/Qwen3.8-Flash-Next-oQ4e-mtp`, its 0.7.0 benchmark
checkpoint, read from its `config.json` on 2026-10-07) spends about the same
bytes on the same kinds of tensors but spreads them finer: per-layer bit
widths picked from calibrated sensitivity (attention at 4 to 8 bits by
layer, the GDN projections at 5 to 6), group 128 for most of its 8- and
5-bit tensors, the LM head and embedding at 8. Group 128 for the 8-bit
groups is the cheap step (a few percent of their bytes, a quality check
on the agent replay); 5- and 6-bit tensors need kernel formats lily does
not have; per-layer choices need a sensitivity measurement on the full
model.

**Smaller machines.** A usage ranking that includes decode-time routing, a
larger LRU region (30% measured 6.9% against 8.8% of decode lookups missing,
at the price of prefill misses), and 8 192-token prefill chunks as the
default under the cache (cold: 1 161 against 945 tok/s prefill, 59.9
against 54.4 decode at `37cc34c`; kept opt-in behind `LILY_PREFILL_CHUNK`
because under an extreme balloon it produced a timing-dependent digest that
no other configuration did). All three want a run on a real 64 GB machine
rather than a balloon.

**Two small ones.** The disk tier wrote on the engine thread at eviction
time, 0.10 to 0.37 s for 21 000 to 100 000 tokens and once 1.17 s (0.5 to
1.1 s once the 8.6 GB budget is full, 2026-10-01); the write ahead now does
that between requests on a background thread ([architecture.md](architecture.md),
"The session cache"). And the draft head's proposals leave penalties
out, so a request with penalties drafts from a distribution that is not the
trunk's kept one; the verify rows correct it exactly, at a lower acceptance
that has not been measured.

## Reproducing this

### One cell

`lily-bench` runs the production cadence: a warm-up prompt at the exact
shape, then a timed prefill and a pipelined decode run with token ids kept on
the GPU.

```sh
target/release/lily-bench \
  --model ~/models/Qwen3.8-Flash-Next-lily-q4 \
  --prompt-len 8192 \
  --prompt-text docs/bench/prompts/p0.txt \
  --decode-steps 96 \
  --ngram-preload \
  --gpu-timing \
  --json-out result.json
```

| flag               | meaning                                                                 |
|--------------------|-------------------------------------------------------------------------|
| `--prompt-len N`   | prompt length in tokens                                                 |
| `--prompt-text F`  | the first `N` tokens of this file under the model's tokenizer; without it a synthetic token sequence |
| `--decode-steps N` | generated tokens (96 for the matrix below)                              |
| `--drafts N`       | measure speculative decoding with N drafts per step and report acceptance instead of the one-token loop |
| `--sample`         | draw with the checkpoint's sampler defaults instead of greedily (`--seed`) |
| `--memory-limit-gb G` | keep the load's footprint under G GB (engages the expert cache when the checkpoint does not fit); `LILY_MEMORY_GB=G` plans for a machine with G GB instead (`--memory-gb` before 2026-10-08) |
| `--ngram-preload`  | stream the paged n-gram table through the page cache before measuring   |
| `--gpu-timing`     | add command-buffer GPU timestamps and host marks, and print the levels per step |
| `--kernel-profile` | per-kernel GPU times per pass; wall-clock results under this flag are not comparable to a normal run |
| `--json-out PATH`  | write the record                                                        |

`LILY_ABLATE=hc,inject,gdn_proj,gdn_step,attn,head,ple,moe,router,shared,topk,gather,down`
(any subset) skips those kernel groups' dispatches in the decode graph; the
results are garbage, the step time without a group is its marginal cost in
the level chain. `LILY_GDN_SCAN_KERNEL=gdn_prefill_regscan` routes every
prefill through the token-serial scan, `LILY_QSA_ROUTE=split` through the
decode-style split attention kernel, `LILY_QSA_SCORES=scalar` through the
scalar indexer scores, for comparisons against the shipped kernels.
The timing harnesses (`cargo test --release --lib -- --ignored <name>`):
`level_size_bandwidth_probe`, the `*_chain_timing` tests per kernel family,
`query_attention_timing`, `gdn_prefill_scan_timing`; each takes an env
variable listing the kernel variants to rotate, named in its source.

### The matrix

`tools/bench/timeline.sh` runs the fixed matrix (1K / 8K / 32K prompts, 0 and
2 drafts, 96 tokens, 3 repeats, repeats outermost) at one or more commits and
summarizes the records with `tools/bench/summarize.py`.

```sh
tools/bench/timeline.sh --prompt-text docs/bench/prompts/p0.txt --note "what changed"
tools/bench/timeline.sh --commit <old> --commit HEAD --prompt-text docs/bench/prompts/p0.txt
```

With several `--commit` flags it builds them all first, in worktrees under
`target/timeline/`, and then interleaves them per repeat. `HEAD` or
`worktree` means the working tree, labelled `-dirty` when tracked sources are
modified. `--prompt-text` runs every cell on real text and refuses a commit
whose `lily-bench` lacks the flag, so the cells stay comparable. It refuses
to start on battery power unless told `--allow-battery`, runs only when
invoked, and costs about 10 minutes of GPU time per commit. Each run's JSON,
a `run.json` with the commit, note, power source and host, and a `.env.txt`
with swap, paging counters and the thermal level land in
`docs/bench/<date>-<sha>/`. Those records and the timeline document
`summarize.py` writes are local output, not part of the published
repository.

### Over HTTP

`tools/bench/http_bench.py matrix` drives a running server (lily, or a
llama.cpp one) with fresh real-text prompts at 1K, 4K, 16K, 32K and 64K
tokens, three interleaved repeats, 256 greedy tokens of new text, and
`report` prints the medians from the records; `cache` measures what a prompt
cache gives back. It needs the `tokenizers` package (`uv venv tools/.venv;
uv pip install --python tools/.venv/bin/python tokenizers`). The README's
table comes from it.
