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
- **The per-kernel profiler inflates what it measures.** In profile mode
  every dispatch becomes its own command buffer, which costs 2 to 3 us each
  and loses intra-level overlap. Against unprofiled spans: decode +14%,
  the 3-row verify pass +55 to 60%, the draft pass about +20%, prefill within
  1 to 4%. Rankings and shares from the profiler are reliable; absolute
  milliseconds of tiny kernels are not. The prefill tables below are
  production numbers, the decode ones are not.

## The numbers

Measured, median of three repeats, range in parentheses.

### Prefill, tok/s

| 1K prompt         | 8K prompt         | 32K prompt        |
|-------------------|-------------------|-------------------|
| 1 825 (1 786 to 1 965) | 1 396 (1 393 to 1 396) | 1 195 (1 168 to 1 196) |

The 1K column is unreliable and the 32K column is clock-limited, both for the
reasons above.

### Decode without drafts, tok/s

| 1K prompt        | 8K prompt        | 32K prompt       |
|------------------|------------------|------------------|
| 86.8 (85.3 to 94.0) | 79.7 (78.2 to 79.7) | 75.6 (75.3 to 77.2) |

### Decode with 2 drafts per step, tok/s

| 1K prompt          | 8K prompt          | 32K prompt         |
|--------------------|--------------------|--------------------|
| 116.9 (101.0 to 117.0), 73% accepted | 98.1 (97.3 to 98.1), 64% accepted | 70.9 (69.9 to 71.8), 37% accepted |

**The 32K speculative cell is a trajectory artifact, not a regression.** All
four 32K sequences (with and without drafts, on either side of the change
that produced this row) diverge from one another at token 1 on this random
token prompt, and this one fell into a run of a repeated special token that
the draft head predicts poorly. The kernels involved do not depend on
position, and the 4-layer differential test of the speculative step at
position 33 000 passes. Acceptance on real code and prose has measured 66%
over 86 000 generated tokens and 78 to 98% in shorter runs, so the synthetic
figures are lower bounds.

Speculative decoding changes no output: every emitted token is the trunk's
own draw and the draft count changes only how many rows a pass confirms.
Sampling with temperature accepts fewer drafts than greedy decoding, because
the draft head proposes with argmax; that rate has not been measured as a
number.

### Against mlx-lm

There is **no mlx-lm comparison for Qwen3.8-Flash-Next**: upstream mlx-lm has
no `qwen4_exp` implementation, which is why lily's checkpoint format is its
own. The comparison that exists is on the upstream model, Qwen3.6-35B-A3B,
from `mlx-community/Qwen3.6-35B-A3B-4bit`, where both engines read the same
MLX affine Q4 weights.

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
   (9.2 to 9.8 us at one row, 12.8 to 14.2 at three) and not kept.
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
   step; the verify pass's from 1.98 to 1.32 ms.
4. **Dispatch fusion in the decode graph.** Measured: about 966 dispatches
   per step, with a 1.4 us floor for a trivial dispatch plus barrier, and the
   fused hyper-connection kernels reading 0.68 GB at about 315 GB/s against
   the 575 GB/s probe. Estimated: 0.4 to 0.7 ms per step (4 to 6%) from
   halving the dispatch count, plus 0.6 to 1.0 ms from the down kernel's
   occupancy; the upper bound if every dispatch paid the floor would be
   1.35 ms.
5. **Continuous batching across sessions.** Measured: an extra row in a
   verify pass costs about 3 ms on a 12 to 13 ms base, because it adds its
   own experts but shares the 2.8 GB of dense weights. Estimated: x1.5 to
   x1.7 aggregate throughput for two concurrent sessions and x2 to x2.3 for
   three, at unchanged or slightly worse per-session latency, assuming item 2
   is fixed first so the batched rows do not pay the skinny-GEMM penalty.
   This is the only item that changes the engine's shape: per-row cache
   pointers and positions in every attention, GDN and n-gram kernel, per-row
   sampler state, and a scheduler that admits and retires sessions per step.

Two smaller ones, for completeness. The disk tier writes on the engine thread
at eviction time, measured at 0.10 to 0.37 s for 21 000 to 100 000 tokens and
once 1.17 s; copying to host memory and writing from a background thread
would take that off the request path, but it only bites under a tight cache
budget. And the LM head already runs at the probed peak, so it is not a
lever.

Where bytes could be saved instead of time (3-bit experts, Q4
hyper-connection mixers, 8-bit KV), the quality cost is **not measured**, and
the 4-layer Hugging Face comparison cannot see expert quantization error at
scale. A perplexity or task run on the full model would have to come first.

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
| `--gpu-timing`     | add command-buffer GPU timestamps and host marks                        |
| `--kernel-profile` | per-kernel GPU times per pass; wall-clock results under this flag are not comparable to a normal run |
| `--json-out PATH`  | write the record                                                        |

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
