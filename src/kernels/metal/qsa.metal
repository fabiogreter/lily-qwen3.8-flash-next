// Qwen Sparse Attention (QSA) for Qwen3.8-Flash-Next.
//
// A lightning indexer scores every completed block of `ratio` cached tokens
// against NH small query heads; the best `k_max` blocks plus the incomplete
// tail block form the token set a query attends. Block keys are the mean of
// the block's raw indexer keys, normed and roped at the block start.
#include <metal_stdlib>
using namespace metal;

#define TG 256
#define QSA_SPLIT 256      // largest split (tokens per threadgroup) of the split kernel
#define QSA_HPP 4          // query heads folded per K/V pass
#define QSA_SELECT_TG 512  // threads of the per-query top-k selection (two per radix bin)

// --- Indexer projections --------------------------------------------------------

// One NeoX RoPE pair with the reference's bf16 arithmetic: cos/sin, both
// products and the sum are each rounded to bf16, as torch does for
// `q * cos + rotate_half(q) * sin` on bf16 tensors. The indexer's block
// selection is a hard top-k, so matching this rounding keeps the selected
// set reproducible against the reference where f32 math would flip
// near-tied blocks.
static inline void qsa_rope_pair_bf16(bfloat lo, bfloat hi, float angle,
                                      device bfloat* out_lo, device bfloat* out_hi) {
    const float c = float(bfloat(cos(angle)));
    const float s = float(bfloat(sin(angle)));
    const float l = float(lo);
    const float h = float(hi);
    *out_lo = bfloat(float(bfloat(l * c)) - float(bfloat(h * s)));
    *out_hi = bfloat(float(bfloat(h * c)) + float(bfloat(l * s)));
}

// The rotary position of a sequence index: the index plus `rope_delta` (0
// for text; VISION.md `rope_deltas` for tokens generated after an image).
// Cache slots and block indices stay sequence indices; only the angle takes
// the delta.
static inline float qsa_rope_position(uint seq_index, int rope_delta) {
    return float(int(seq_index) + rope_delta);
}

// The axis rotary pair `j` reads under the interleaved M-RoPE with
// mrope_section [11, 11, 10] (mirrors `kernels::mrope_axis` and
// attention.metal's copy): temporal (0) when j % 3 == 0, height (1) when
// j % 3 == 1 and j < sec_h_end, width (2) when j % 3 == 2 and j < sec_w_end.
static inline uint qsa_mrope_axis(uint j, uint sec_h_end, uint sec_w_end) {
    const uint axis = j % 3;
    if (axis == 1) {
        return j < sec_h_end ? 1u : 0u;
    }
    if (axis == 2) {
        return j < sec_w_end ? 2u : 0u;
    }
    return 0u;
}

// RMSNorm(+1) of one D-wide row held in `value` per thread into `normed`
// (threadgroup), the shared prologue of the indexer's query and block-key
// kernels. D threads; every thread reaches the barriers.
static inline void qsa_norm_row(float value, device const bfloat* w, uint D, float eps,
                                uint tid, uint sg, uint lane,
                                threadgroup float* partial, threadgroup float* inv_rms,
                                threadgroup bfloat* normed) {
    float acc = simd_sum(value * value);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < (D + 31) / 32; ++i) {
            total += partial[i];
        }
        *inv_rms = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normed[tid] = bfloat(value * *inv_rms * (1.0f + float(w[tid])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// q[row, h, :] = rope(rmsnorm(qk[row, h*D..]) * (1 + w)) at position base_pos + row.
// One threadgroup per (row, head); D threads.
kernel void qsa_prep_q_bf16(device const bfloat* qk  [[buffer(0)]],  // [M, (NH+1)*D]
                            device const bfloat* w   [[buffer(1)]],  // [D]
                            device bfloat*       q   [[buffer(2)]],  // [M, NH, D]
                            constant uint&       D   [[buffer(3)]],
                            constant uint&       NH  [[buffer(4)]],
                            constant uint&       rot [[buffer(5)]],
                            constant uint&       base_pos [[buffer(6)]],
                            constant float&      theta [[buffer(7)]],
                            constant float&      eps [[buffer(8)]],
                            constant int&        rope_delta [[buffer(9)]],
                            uint seg  [[threadgroup_position_in_grid]],
                            uint tid  [[thread_index_in_threadgroup]],
                            uint sg   [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;
    threadgroup bfloat normed[TG];

    const uint row = seg / NH;
    const uint h = seg % NH;
    const ulong src = (ulong)row * (NH + 1) * D + (ulong)h * D;
    const ulong dst = (ulong)seg * D;
    const float value = float(qk[src + tid]);
    float acc = simd_sum(value * value);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < (D + 31) / 32; ++i) {
            total += partial[i];
        }
        inv_rms = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normed[tid] = bfloat(value * inv_rms * (1.0f + float(w[tid])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const float angle = qsa_rope_position(base_pos + row, rope_delta)
                          * pow(theta, -2.0f * float(tid) / float(rot));
        qsa_rope_pair_bf16(normed[tid], normed[half_rot + tid], angle,
                           q + dst + tid, q + dst + half_rot + tid);
    } else if (tid >= rot) {
        q[dst + tid] = normed[tid];
    }
}

// qsa_prep_q_bf16 for the prefill rows of a prompt with an image: row `row`
// is sequence index base_pos + row, whose 3-axis position is
// positions[base_pos + row - pos_base] (U32 [rows, 3]); pair `tid` takes the
// axis qsa_mrope_axis gives it. Rows whose axes agree get exactly what
// qsa_prep_q_bf16 computes.
kernel void qsa_prep_q_mrope_bf16(device const bfloat* qk  [[buffer(0)]],  // [M, (NH+1)*D]
                                  device const bfloat* w   [[buffer(1)]],  // [D]
                                  device bfloat*       q   [[buffer(2)]],  // [M, NH, D]
                                  device const uint*   positions [[buffer(3)]],  // [rows, 3]
                                  constant uint&       D   [[buffer(4)]],
                                  constant uint&       NH  [[buffer(5)]],
                                  constant uint&       rot [[buffer(6)]],
                                  constant uint&       base_pos [[buffer(7)]],
                                  constant float&      theta [[buffer(8)]],
                                  constant float&      eps [[buffer(9)]],
                                  constant uint&       pos_base [[buffer(10)]],
                                  constant uint&       sec_h_end [[buffer(11)]],
                                  constant uint&       sec_w_end [[buffer(12)]],
                                  uint seg  [[threadgroup_position_in_grid]],
                                  uint tid  [[thread_index_in_threadgroup]],
                                  uint sg   [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;
    threadgroup bfloat normed[TG];

    const uint row = seg / NH;
    const uint h = seg % NH;
    const ulong src = (ulong)row * (NH + 1) * D + (ulong)h * D;
    const ulong dst = (ulong)seg * D;
    qsa_norm_row(float(qk[src + tid]), w, D, eps, tid, sg, lane, partial, &inv_rms, normed);
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const uint axis = qsa_mrope_axis(tid, sec_h_end, sec_w_end);
        const uint pos = positions[(base_pos + row - pos_base) * 3 + axis];
        const float angle = float(pos) * pow(theta, -2.0f * float(tid) / float(rot));
        qsa_rope_pair_bf16(normed[tid], normed[half_rot + tid], angle,
                           q + dst + tid, q + dst + half_rot + tid);
    } else if (tid >= rot) {
        q[dst + tid] = normed[tid];
    }
}

// cache[base_pos + m, :] = qk[m, NH*D ..]: the raw (unnormed, unroped) indexer key.
kernel void qsa_scatter_keys_bf16(device const bfloat* qk    [[buffer(0)]],
                                  device bfloat*       cache [[buffer(1)]],  // [max_seq, D]
                                  constant uint&       D     [[buffer(2)]],
                                  constant uint&       NH    [[buffer(3)]],
                                  constant uint&       base_pos [[buffer(4)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint m = gid.y;
    cache[(ulong)(base_pos + m) * D + d] = qk[(ulong)m * (NH + 1) * D + (ulong)NH * D + d];
}

// blk[b, :] = rope(rmsnorm(bf16(mean of the block's raw keys)) * (1 + w)) at
// position b * ratio, for b = first_block + threadgroup index. D threads.
kernel void qsa_block_keys_bf16(device const bfloat* cache [[buffer(0)]],  // [max_seq, D]
                                device const bfloat* w     [[buffer(1)]],  // [D]
                                device bfloat*       blk   [[buffer(2)]],  // [max_blocks, D]
                                constant uint&       D     [[buffer(3)]],
                                constant uint&       ratio [[buffer(4)]],
                                constant uint&       first_block [[buffer(5)]],
                                constant uint&       rot   [[buffer(6)]],
                                constant float&      theta [[buffer(7)]],
                                constant float&      eps   [[buffer(8)]],
                                constant uint&       count [[buffer(9)]],  // blocks to build (grid may exceed it)
                                constant int&        rope_delta [[buffer(10)]],
                                uint tg   [[threadgroup_position_in_grid]],
                                uint tid  [[thread_index_in_threadgroup]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;
    threadgroup bfloat normed[TG];

    if (tg >= count) {
        return;  // uniform per threadgroup: no barrier is skipped unevenly
    }
    const uint b = first_block + tg;
    float pooled = 0.0f;
    for (uint i = 0; i < ratio; ++i) {
        pooled += float(cache[(ulong)(b * ratio + i) * D + tid]);
    }
    // The reference pools in f32 and rounds the mean to bf16 before the norm.
    const float value = float(bfloat(pooled / float(ratio)));
    float acc = simd_sum(value * value);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < (D + 31) / 32; ++i) {
            total += partial[i];
        }
        inv_rms = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normed[tid] = bfloat(value * inv_rms * (1.0f + float(w[tid])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong dst = (ulong)b * D;
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const float angle = qsa_rope_position(b * ratio, rope_delta)
                          * pow(theta, -2.0f * float(tid) / float(rot));
        qsa_rope_pair_bf16(normed[tid], normed[half_rot + tid], angle,
                           blk + dst + tid, blk + dst + half_rot + tid);
    } else if (tid >= rot) {
        blk[dst + tid] = normed[tid];
    }
}

// qsa_block_keys_bf16 for the blocks a prompt with an image completes: block
// b is roped at the 3-axis position of its first token, sequence index
// b * ratio, read from positions[b * ratio - pos_base] (VISION.md: the
// indexer's block keys take the cos/sin of each block's first position).
kernel void qsa_block_keys_mrope_bf16(device const bfloat* cache [[buffer(0)]],  // [max_seq, D]
                                      device const bfloat* w     [[buffer(1)]],  // [D]
                                      device bfloat*       blk   [[buffer(2)]],  // [max_blocks, D]
                                      device const uint*   positions [[buffer(3)]],  // [rows, 3]
                                      constant uint&       D     [[buffer(4)]],
                                      constant uint&       ratio [[buffer(5)]],
                                      constant uint&       first_block [[buffer(6)]],
                                      constant uint&       rot   [[buffer(7)]],
                                      constant float&      theta [[buffer(8)]],
                                      constant float&      eps   [[buffer(9)]],
                                      constant uint&       count [[buffer(10)]],
                                      constant uint&       pos_base [[buffer(11)]],
                                      constant uint&       sec_h_end [[buffer(12)]],
                                      constant uint&       sec_w_end [[buffer(13)]],
                                      uint tg   [[threadgroup_position_in_grid]],
                                      uint tid  [[thread_index_in_threadgroup]],
                                      uint sg   [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;
    threadgroup bfloat normed[TG];

    if (tg >= count) {
        return;  // uniform per threadgroup: no barrier is skipped unevenly
    }
    const uint b = first_block + tg;
    float pooled = 0.0f;
    for (uint i = 0; i < ratio; ++i) {
        pooled += float(cache[(ulong)(b * ratio + i) * D + tid]);
    }
    // The reference pools in f32 and rounds the mean to bf16 before the norm.
    const float value = float(bfloat(pooled / float(ratio)));
    qsa_norm_row(value, w, D, eps, tid, sg, lane, partial, &inv_rms, normed);
    const ulong dst = (ulong)b * D;
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const uint axis = qsa_mrope_axis(tid, sec_h_end, sec_w_end);
        const uint pos = positions[(b * ratio - pos_base) * 3 + axis];
        const float angle = float(pos) * pow(theta, -2.0f * float(tid) / float(rot));
        qsa_rope_pair_bf16(normed[tid], normed[half_rot + tid], angle,
                           blk + dst + tid, blk + dst + half_rot + tid);
    } else if (tid >= rot) {
        blk[dst + tid] = normed[tid];
    }
}

// --- Block scoring and selection --------------------------------------------------

// Complete blocks visible to the query at position `pos`.
static inline uint qsa_visible_blocks(uint pos, uint ratio) {
    return (pos + 1) / ratio;
}

// scores[qi, b] = sum_h relu(<q[qi, h], blk[b]>) / sqrt(D) for b < visible
// blocks of query qi (-inf beyond). Grid: threadgroups (ceil(nb_max/TG), QB).
kernel void qsa_scores_f32(device const bfloat* q      [[buffer(0)]],  // [QB, NH, D]
                           device const bfloat* blk    [[buffer(1)]],  // [max_blocks, D]
                           device float*        scores [[buffer(2)]],  // [QB, nb_max]
                           constant uint&       D      [[buffer(3)]],
                           constant uint&       NH     [[buffer(4)]],
                           constant uint&       nb_max [[buffer(5)]],
                           constant uint&       base_pos [[buffer(6)]],
                           constant uint&       ratio  [[buffer(7)]],
                           constant float&      inv_sqrt_d [[buffer(8)]],
                           uint2 tg  [[threadgroup_position_in_grid]],
                           uint  tid [[thread_index_in_threadgroup]]) {
    threadgroup float qs[4 * 128];  // NH * D <= 512

    const uint qi = tg.y;
    for (uint i = tid; i < NH * D; i += TG) {
        qs[i] = float(q[(ulong)qi * NH * D + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint b = tg.x * TG + tid;
    if (b >= nb_max) {
        return;
    }
    const uint nb = qsa_visible_blocks(base_pos + qi, ratio);
    float score = -INFINITY;
    if (b < nb) {
        device const uint4* krow = (device const uint4*)(blk + (ulong)b * D);
        float dots[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint i = 0; i * 8 < D; ++i) {
            const uint4 kq = krow[i];
            const float4 ka = float4(as_type<bfloat4>(kq.xy));
            const float4 kb = float4(as_type<bfloat4>(kq.zw));
            for (uint h = 0; h < NH; ++h) {
                const float4 qa(qs[h * D + i * 8], qs[h * D + i * 8 + 1],
                                qs[h * D + i * 8 + 2], qs[h * D + i * 8 + 3]);
                const float4 qb(qs[h * D + i * 8 + 4], qs[h * D + i * 8 + 5],
                                qs[h * D + i * 8 + 6], qs[h * D + i * 8 + 7]);
                dots[h] += dot(ka, qa) + dot(kb, qb);
            }
        }
        score = 0.0f;
        for (uint h = 0; h < NH; ++h) {
            score += max(dots[h], 0.0f);
        }
        score *= inv_sqrt_d;
    }
    scores[(ulong)qi * nb_max + b] = score;
}

// Non-negative float scores order like their bit patterns.
static inline uint qsa_key(float s) {
    return as_type<uint>(max(s, 0.0f));
}

// Exclusive prefix sum of one value per thread over the whole threadgroup
// (the sum must fit 32 bits); returns this thread's rank and writes the
// total. Consecutive scans need a threadgroup barrier between them (`sums`
// is rewritten by the next call).
template <uint NT>
static inline uint qsa_scan(uint value, threadgroup uint* sums, thread uint& total,
                            uint sg, uint lane) {
    const uint local = simd_prefix_exclusive_sum(value);
    const uint sg_total = simd_sum(value);
    if (lane == 0) {
        sums[sg] = sg_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Lane s holds simdgroup s's total; the scan over the lanes gives every
    // simdgroup's offset and the total in one simd step.
    const uint mine = lane < NT / 32 ? sums[lane] : 0u;
    const uint before = simd_shuffle(simd_prefix_exclusive_sum(mine), sg);
    total = simd_sum(mine);
    return before + local;
}

// Counts one per active lane into hist[bin] with one atomic per distinct
// bin per simdgroup: the top radix digit of the scores (sign and exponent)
// takes a handful of values, so per-lane atomics would serialize on them.
static inline void qsa_hist_add(threadgroup atomic_uint* hist, uint bin, bool active,
                                uint lane) {
    while (true) {
        const uint leader = simd_min(active ? bin : 0xFFFFFFFFu);
        if (leader == 0xFFFFFFFFu) {
            break;
        }
        const bool same = active && bin == leader;
        const ulong votes = (simd_vote::vote_t)simd_ballot(same);
        if (same) {
            active = false;
            if (lane == uint(ctz(votes))) {
                atomic_fetch_add_explicit(&hist[leader], popcount(votes),
                                          memory_order_relaxed);
            }
        }
    }
}

// Largest per-thread block count of the selection (32K tokens at ratio 4
// in one chunk).
#define QSA_SELECT_CACHE_MAX 32

// The selection of one query over `nb` blocks with `CACHE` consecutive
// blocks per thread: a chunk is CACHE * QSA_SELECT_TG blocks, the first
// chunk's keys stay in registers across the passes and later chunks are
// re-read from the score row. See qsa_select_blocks.
template <uint CACHE, uint NT = QSA_SELECT_TG>
static void qsa_select_body(device const float* row, device uint* out, uint nb,
                            uint k_max, threadgroup atomic_uint (*hist)[256],
                            threadgroup uint* sums, threadgroup uint* prefix_s,
                            threadgroup uint* k_rem_s, uint tid, uint sg, uint lane) {
    constexpr uint CHUNK = CACHE * NT;
    const uint t0 = tid * CACHE;
    uint keys[CACHE];
    for (uint i = 0; i < CACHE; ++i) {
        const uint b = t0 + i;
        keys[i] = b < nb ? qsa_key(row[b]) : 0u;
    }
    static_assert(NT >= 256 && NT % 32 == 0, "at least one selection thread per radix bin");
    if (tid < 256) {
        atomic_store_explicit(&hist[0][tid], 0u, memory_order_relaxed);
        atomic_store_explicit(&hist[1][tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Radix select (MSB first) for the k_max-th largest key. The top digit
    // (sign and exponent) crowds a few bins: it is counted with one atomic
    // per distinct bin per simdgroup. Later digits spread over the bins
    // and only the keys under the prefix count.
    uint prefix = 0;
    uint mask = 0;
    uint k_rem = k_max;
    for (uint pass = 0; pass < 4; ++pass) {
        const uint shift = 24 - 8 * pass;
        threadgroup atomic_uint* h = hist[pass & 1];
        for (uint b0 = 0; b0 < nb; b0 += CHUNK) {
            for (uint i = 0; i < CACHE; ++i) {
                const uint b = b0 + t0 + i;
                const uint key = b0 == 0 ? keys[i] : (b < nb ? qsa_key(row[b]) : 0u);
                const bool active = b < nb && (key & mask) == prefix;
                const uint bin = (key >> shift) & 0xFFu;
                if (pass == 0) {
                    qsa_hist_add(h, bin, active, lane);
                } else if (active) {
                    atomic_fetch_add_explicit(&h[bin], 1u, memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Thread t holds bin 255 - t: the scan gives the count above each
        // bin, and exactly one bin straddles the k_rem-th key.
        // Threads past the 256 bins hold nothing (their `c` is 0, so they
        // never straddle the k_rem-th key).
        const uint c = tid < 256 ? atomic_load_explicit(&h[255 - tid], memory_order_relaxed) : 0u;
        uint total;
        const uint above = qsa_scan<NT>(c, sums, total, sg, lane);
        if (tid < 256 && above < k_rem && above + c >= k_rem) {
            *prefix_s = prefix | ((255 - tid) << shift);
            *k_rem_s = k_rem - above;
        }
        if (tid < 256) {
            atomic_store_explicit(&h[tid], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = *prefix_s;
        k_rem = *k_rem_s;
        mask |= 0xFFu << shift;
    }
    const uint threshold = prefix;
    // Elements above the threshold are all taken; exactly k_rem ties fill up.
    const uint ties_to_take = k_rem;

    // Compaction in ascending block order, one scan per chunk: the low half
    // of the scanned value counts a thread's keys above the threshold, the
    // high half its ties (each at most CHUNK).
    uint sel_base = 0;
    uint tie_base = 0;
    for (uint b0 = 0; b0 < nb; b0 += CHUNK) {
        uint n_above = 0;
        uint n_tie = 0;
        for (uint i = 0; i < CACHE; ++i) {
            const uint b = b0 + t0 + i;
            const uint key = b0 == 0 ? keys[i] : (b < nb ? qsa_key(row[b]) : 0u);
            n_above += (b < nb && key > threshold) ? 1u : 0u;
            n_tie += (b < nb && key == threshold) ? 1u : 0u;
        }
        uint totals;
        const uint ranks =
            qsa_scan<NT>((n_tie << 16) | n_above, sums, totals, sg, lane);
        const uint quota = ties_to_take > tie_base ? ties_to_take - tie_base : 0u;
        // Taken keys before this thread's: the scanned counts; within them,
        // the thread walks its blocks in order.
        uint a = sel_base + (ranks & 0xFFFFu);
        uint t = ranks >> 16;
        for (uint i = 0; i < CACHE; ++i) {
            const uint b = b0 + t0 + i;
            const uint key = b0 == 0 ? keys[i] : (b < nb ? qsa_key(row[b]) : 0u);
            if (b < nb) {
                if (key > threshold) {
                    out[a + min(t, quota)] = b;
                    ++a;
                } else if (key == threshold) {
                    if (t < quota) {
                        out[a + t] = b;
                    }
                    ++t;
                }
            }
        }
        sel_base += (totals & 0xFFFFu) + min(totals >> 16, quota);
        tie_base += totals >> 16;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Selects the k_max highest-scoring visible blocks of each query (all of them
// when fewer are visible), written in ascending block order to sel[qi, :] with
// the count in n_sel[qi]. Ties at the k-th score resolve to the lowest block
// ids. One threadgroup of QSA_SELECT_TG threads per query: a radix select
// (four 8-bit digits, most significant first) finds the k_max-th key, then
// one scan per chunk compacts the keys above it and the lowest tied blocks
// (each thread's blocks are consecutive, so it places its own in order
// after the scan). The blocks per thread follow the context so every
// thread holds some; every threadgroup-wide step is a scan or a simd
// reduction and nothing runs serially on one thread. The threads beyond
// the 256 radix bins only shorten each thread's walk: 512 threads halved
// the kernel against 256 (8K decode 16.7 to 7.8 us, 32K 31.3 to 15.7) and
// 1 024 gave it back to the wider scans (15.8 and 16.2).
kernel void qsa_select_blocks(device const float* scores [[buffer(0)]],  // [QB, nb_max]
                              device uint*        sel    [[buffer(1)]],  // [QB, k_max]
                              device uint*        n_sel  [[buffer(2)]],  // [QB]
                              constant uint&      nb_max [[buffer(3)]],
                              constant uint&      base_pos [[buffer(4)]],
                              constant uint&      ratio  [[buffer(5)]],
                              constant uint&      k_max  [[buffer(6)]],
                              uint qi   [[threadgroup_position_in_grid]],
                              uint tid  [[thread_index_in_threadgroup]],
                              uint sg   [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    // Two histograms: the one the next pass counts into is cleared while
    // the current pass picks its digit.
    threadgroup atomic_uint hist[2][256];
    threadgroup uint sums[QSA_SELECT_TG / 32];
    threadgroup uint prefix_s;
    threadgroup uint k_rem_s;

    const uint nb = qsa_visible_blocks(base_pos + qi, ratio);
    device const float* row = scores + (ulong)qi * nb_max;
    device uint* out = sel + (ulong)qi * k_max;

    if (nb <= k_max) {
        for (uint b = tid; b < nb; b += QSA_SELECT_TG) {
            out[b] = b;
        }
        if (tid == 0) {
            n_sel[qi] = nb;
        }
        return;
    }
    if (nb <= 4 * QSA_SELECT_TG) {
        qsa_select_body<4>(row, out, nb, k_max, hist, sums, &prefix_s, &k_rem_s,
                           tid, sg, lane);
    } else if (nb <= 8 * QSA_SELECT_TG) {
        qsa_select_body<8>(row, out, nb, k_max, hist, sums, &prefix_s, &k_rem_s,
                           tid, sg, lane);
    } else if (nb <= 16 * QSA_SELECT_TG) {
        qsa_select_body<16>(row, out, nb, k_max, hist, sums, &prefix_s, &k_rem_s,
                            tid, sg, lane);
    } else {
        qsa_select_body<QSA_SELECT_CACHE_MAX>(row, out, nb, k_max, hist, sums,
                                              &prefix_s, &k_rem_s, tid, sg, lane);
    }
    if (tid == 0) {
        n_sel[qi] = k_max;
    }
}

// --- Sparse attention -----------------------------------------------------------

// Cache position of the i-th attended token of a query: selected blocks
// expand to `ratio` consecutive tokens, then the incomplete tail follows.
static inline uint qsa_token_at(device const uint* sel_row, uint nblk, uint ratio,
                                uint tail_start, uint i) {
    const uint in_blocks = nblk * ratio;
    return i < in_blocks ? sel_row[i / ratio] * ratio + (i % ratio)
                         : tail_start + (i - in_blocks);
}

// Split-K sparse GQA attention over each query's selected tokens. Grid:
// threadgroups (KVH * head_groups, splits, QB), TG threads; each threadgroup
// covers `split` (<= QSA_SPLIT) consecutive attended tokens and, with
// head_groups > 1, only its QSA_HPP query heads of the KV head (a decode
// dispatch of one query otherwise has too few threadgroups to fill the
// GPU). Emits per-split softmax stats and weighted-V partials laid out like
// sdpa_decode_split (head index qi*NQ + hq), for sdpa_decode_combine. D
// must be 256.
// The kernel's latency chains are cut: the split's token ids are staged
// once (one dependent load per thread instead of one per K and per V row),
// each simdgroup requests QSA_KB K (then V) rows before dotting them, the
// four heads' softmax sums share one barrier (separate partial arrays, so no
// write-after-read barrier per head) and the V partials of two heads are
// staged and reduced per barrier pair. The per-token, per-lane and
// per-simdgroup operation order is unchanged from the plain form of the
// kernel, which is bit-identical and measured 6 us per layer slower at 8K
// in the decode chain (docs/performance.md); a variant with the memory
// loads removed showed the rest of the kernel's time is the barriers and
// the cross-simdgroup staging, not the K/V traffic.
#define QSA_KB 4  // K (then V) rows a simdgroup requests per step
kernel void qsa_attn_split_bf16(device const bfloat* q        [[buffer(0)]],  // [QB, NQ, D]
                                device const bfloat* k_cache  [[buffer(1)]],  // [KVH, max_seq, D]
                                device const bfloat* v_cache  [[buffer(2)]],
                                device const uint*   sel      [[buffer(3)]],  // [QB, k_max]
                                device const uint*   n_sel    [[buffer(4)]],  // [QB]
                                device float*        partials [[buffer(5)]],  // [QB*NQ, splits, D]
                                device float*        stats    [[buffer(6)]],  // [QB*NQ, splits, 2]
                                constant uint&       D        [[buffer(7)]],
                                constant uint&       max_seq  [[buffer(8)]],
                                constant uint&       group    [[buffer(9)]],
                                constant uint&       splits   [[buffer(10)]],
                                constant uint&       k_max    [[buffer(11)]],
                                constant uint&       ratio    [[buffer(12)]],
                                constant uint&       base_pos [[buffer(13)]],
                                constant float&      scale    [[buffer(14)]],
                                constant uint&       NQ       [[buffer(15)]],
                                constant uint&       split    [[buffer(16)]],
                                constant uint&       head_groups [[buffer(17)]],
                                uint3 tg  [[threadgroup_position_in_grid]],
                                uint tid  [[thread_index_in_threadgroup]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[QSA_HPP][QSA_SPLIT];
    threadgroup float part[QSA_HPP][(TG / 32)];
    threadgroup float part_sum[QSA_HPP][(TG / 32)];
    threadgroup float red[QSA_HPP];
    threadgroup uint tok[QSA_SPLIT];
    threadgroup float v_stage[2][((TG / 32)) * 256];

    const uint kh = tg.x / head_groups;
    const uint hg = tg.x - kh * head_groups;
    const uint split_idx = tg.y;
    const uint qi = tg.z;
    const uint pos = base_pos + qi;
    const uint nblk = n_sel[qi];
    const uint tail_start = qsa_visible_blocks(pos, ratio) * ratio;
    const uint total = nblk * ratio + (pos + 1 - tail_start);
    const uint slot0 = split_idx * split;
    const uint count = slot0 < total ? min(split, total - slot0) : 0;
    const uint h_begin = hg * QSA_HPP;
    const uint h_end = head_groups == 1 ? group : min(h_begin + QSA_HPP, group);
    device const uint* sel_row = sel + (ulong)qi * k_max;
    device const bfloat* k_head = k_cache + (ulong)kh * max_seq * D;
    device const bfloat* v_head = v_cache + (ulong)kh * max_seq * D;

    if (count == 0) {
        if (tid == 0) {
            for (uint h = h_begin; h < h_end; ++h) {
                const ulong hq = (ulong)qi * NQ + (ulong)kh * group + h;
                stats[(hq * splits + split_idx) * 2] = -INFINITY;
                stats[(hq * splits + split_idx) * 2 + 1] = 0.0f;
            }
        }
        return;
    }
    // The split's cache positions, one dependent load per thread.
    for (uint p = tid; p < count; p += TG) {
        tok[p] = qsa_token_at(sel_row, nblk, ratio, tail_start, slot0 + p);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h0 = h_begin; h0 < h_end; h0 += QSA_HPP) {
        const uint gh = min(uint(QSA_HPP), h_end - h0);
        float4 qa[QSA_HPP];
        float4 qb[QSA_HPP];
        for (uint h = 0; h < gh; ++h) {
            device const bfloat* qh =
                q + ((ulong)qi * NQ + (ulong)kh * group + h0 + h) * D + lane * 8;
            qa[h] = float4(float(qh[0]), float(qh[1]), float(qh[2]), float(qh[3]));
            qb[h] = float4(float(qh[4]), float(qh[5]), float(qh[6]), float(qh[7]));
        }
        float local_max[QSA_HPP];
        for (uint h = 0; h < QSA_HPP; ++h) {
            local_max[h] = -INFINITY;
        }
        // Simdgroup sg takes tokens sg, sg + 8, ... in order, QSA_KB rows
        // requested per step.
        for (uint p0 = sg; p0 < count; p0 += QSA_KB * ((TG / 32))) {
            uint4 kq[QSA_KB];
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                const uint token = p < count ? tok[p] : tok[p0];
                kq[j] = ((device const uint4*)(k_head + (ulong)token * D))[lane];
            }
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                if (p < count) {
                    const float4 ka = float4(as_type<bfloat4>(kq[j].xy));
                    const float4 kb = float4(as_type<bfloat4>(kq[j].zw));
                    for (uint h = 0; h < gh; ++h) {
                        const float s = simd_sum(dot(ka, qa[h]) + dot(kb, qb[h])) * scale;
                        if (lane == 0) {
                            scores[h][p] = s;
                        }
                        local_max[h] = max(local_max[h], s);
                    }
                }
            }
        }
        for (uint h = 0; h < gh; ++h) {
            const float m = simd_max(local_max[h]);
            if (lane == 0) {
                part[h][sg] = m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (uint h = 0; h < gh; ++h) {
                float m = -INFINITY;
                for (uint i = 0; i < (TG / 32); ++i) {
                    m = max(m, part[h][i]);
                }
                red[h] = m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint h = 0; h < gh; ++h) {
            const float chunk_max = red[h];
            float e = 0.0f;
            if (tid < count) {
                e = exp(scores[h][tid] - chunk_max);
                scores[h][tid] = e;
            }
            const float local_sum = simd_sum(e);
            if (lane == 0) {
                part_sum[h][sg] = local_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (uint h = 0; h < gh; ++h) {
                float s = 0.0f;
                for (uint i = 0; i < (TG / 32); ++i) {
                    s += part_sum[h][i];
                }
                const ulong hq = (ulong)qi * NQ + (ulong)kh * group + h0 + h;
                stats[(hq * splits + split_idx) * 2] = red[h];
                stats[(hq * splits + split_idx) * 2 + 1] = s;
            }
        }

        float4 acc0[QSA_HPP];
        float4 acc1[QSA_HPP];
        for (uint h = 0; h < QSA_HPP; ++h) {
            acc0[h] = float4(0.0f);
            acc1[h] = float4(0.0f);
        }
        for (uint p0 = sg; p0 < count; p0 += QSA_KB * ((TG / 32))) {
            uint4 vq[QSA_KB];
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                const uint token = p < count ? tok[p] : tok[p0];
                vq[j] = ((device const uint4*)(v_head + (ulong)token * D))[lane];
            }
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                if (p < count) {
                    const float4 va = float4(as_type<bfloat4>(vq[j].xy));
                    const float4 vb = float4(as_type<bfloat4>(vq[j].zw));
                    for (uint h = 0; h < gh; ++h) {
                        const float wgt = scores[h][p];
                        acc0[h] += wgt * va;
                        acc1[h] += wgt * vb;
                    }
                }
            }
        }
        for (uint hb = 0; hb < gh; hb += 2) {
            const uint nh = min(2u, gh - hb);
            for (uint h = 0; h < nh; ++h) {
                for (uint j = 0; j < 4; ++j) {
                    v_stage[h][sg * D + lane * 8 + j] = acc0[hb + h][j];
                    v_stage[h][sg * D + lane * 8 + 4 + j] = acc1[hb + h][j];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint h = 0; h < nh; ++h) {
                const ulong hq = (ulong)qi * NQ + (ulong)kh * group + h0 + hb + h;
                for (uint d = tid; d < D; d += TG) {
                    float acc = 0.0f;
                    for (uint s = 0; s < (TG / 32); ++s) {
                        acc += v_stage[h][s * D + d];
                    }
                    partials[(hq * splits + split_idx) * D + d] = acc;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// qsa_attn_split_bf16 over a q8 cache (see attention.metal, "8-bit K/V
// cache"): the same body, lane l's 8 dims read as bytes and scaled by
// their group's (l / 4) scale. D must be 256.
kernel void qsa_attn_split_q8(device const bfloat* q        [[buffer(0)]],  // [QB, NQ, D]
                                device const char*   k_cache  [[buffer(1)]],  // [KVH, max_seq, D]
                                device const char*   v_cache  [[buffer(2)]],
                                device const half*   k_scales [[buffer(3)]],  // [KVH, max_seq, D / 32]
                                device const half*   v_scales [[buffer(4)]],
                                device const uint*   sel      [[buffer(5)]],  // [QB, k_max]
                                device const uint*   n_sel    [[buffer(6)]],  // [QB]
                                device float*        partials [[buffer(7)]],  // [QB*NQ, splits, D]
                                device float*        stats    [[buffer(8)]],  // [QB*NQ, splits, 2]
                                constant uint&       D        [[buffer(9)]],
                                constant uint&       max_seq  [[buffer(10)]],
                                constant uint&       group    [[buffer(11)]],
                                constant uint&       splits   [[buffer(12)]],
                                constant uint&       k_max    [[buffer(13)]],
                                constant uint&       ratio    [[buffer(14)]],
                                constant uint&       base_pos [[buffer(15)]],
                                constant float&      scale    [[buffer(16)]],
                                constant uint&       NQ       [[buffer(17)]],
                                constant uint&       split    [[buffer(18)]],
                                constant uint&       head_groups [[buffer(19)]],
                                uint3 tg  [[threadgroup_position_in_grid]],
                                uint tid  [[thread_index_in_threadgroup]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[QSA_HPP][QSA_SPLIT];
    threadgroup float part[QSA_HPP][(TG / 32)];
    threadgroup float part_sum[QSA_HPP][(TG / 32)];
    threadgroup float red[QSA_HPP];
    threadgroup uint tok[QSA_SPLIT];
    threadgroup float v_stage[2][((TG / 32)) * 256];

    const uint kh = tg.x / head_groups;
    const uint hg = tg.x - kh * head_groups;
    const uint split_idx = tg.y;
    const uint qi = tg.z;
    const uint pos = base_pos + qi;
    const uint nblk = n_sel[qi];
    const uint tail_start = qsa_visible_blocks(pos, ratio) * ratio;
    const uint total = nblk * ratio + (pos + 1 - tail_start);
    const uint slot0 = split_idx * split;
    const uint count = slot0 < total ? min(split, total - slot0) : 0;
    const uint h_begin = hg * QSA_HPP;
    const uint h_end = head_groups == 1 ? group : min(h_begin + QSA_HPP, group);
    device const uint* sel_row = sel + (ulong)qi * k_max;
    device const char* k_head = k_cache + (ulong)kh * max_seq * D;
    device const char* v_head = v_cache + (ulong)kh * max_seq * D;
    device const half* ks_head = k_scales + (ulong)kh * max_seq * (D / 32);
    device const half* vs_head = v_scales + (ulong)kh * max_seq * (D / 32);

    if (count == 0) {
        if (tid == 0) {
            for (uint h = h_begin; h < h_end; ++h) {
                const ulong hq = (ulong)qi * NQ + (ulong)kh * group + h;
                stats[(hq * splits + split_idx) * 2] = -INFINITY;
                stats[(hq * splits + split_idx) * 2 + 1] = 0.0f;
            }
        }
        return;
    }
    // The split's cache positions, one dependent load per thread.
    for (uint p = tid; p < count; p += TG) {
        tok[p] = qsa_token_at(sel_row, nblk, ratio, tail_start, slot0 + p);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h0 = h_begin; h0 < h_end; h0 += QSA_HPP) {
        const uint gh = min(uint(QSA_HPP), h_end - h0);
        float4 qa[QSA_HPP];
        float4 qb[QSA_HPP];
        for (uint h = 0; h < gh; ++h) {
            device const bfloat* qh =
                q + ((ulong)qi * NQ + (ulong)kh * group + h0 + h) * D + lane * 8;
            qa[h] = float4(float(qh[0]), float(qh[1]), float(qh[2]), float(qh[3]));
            qb[h] = float4(float(qh[4]), float(qh[5]), float(qh[6]), float(qh[7]));
        }
        float local_max[QSA_HPP];
        for (uint h = 0; h < QSA_HPP; ++h) {
            local_max[h] = -INFINITY;
        }
        // Simdgroup sg takes tokens sg, sg + 8, ... in order, QSA_KB rows
        // requested per step.
        for (uint p0 = sg; p0 < count; p0 += QSA_KB * ((TG / 32))) {
            uint2 kq[QSA_KB];
            float ksc[QSA_KB];
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                const uint token = p < count ? tok[p] : tok[p0];
                kq[j] = ((device const uint2*)(k_head + (ulong)token * D))[lane];
                ksc[j] = float(ks_head[(ulong)token * (D / 32) + lane / 4]);
            }
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                if (p < count) {
                    const float4 ka = float4(as_type<char4>(kq[j].x)) * ksc[j];
                    const float4 kb = float4(as_type<char4>(kq[j].y)) * ksc[j];
                    for (uint h = 0; h < gh; ++h) {
                        const float s = simd_sum(dot(ka, qa[h]) + dot(kb, qb[h])) * scale;
                        if (lane == 0) {
                            scores[h][p] = s;
                        }
                        local_max[h] = max(local_max[h], s);
                    }
                }
            }
        }
        for (uint h = 0; h < gh; ++h) {
            const float m = simd_max(local_max[h]);
            if (lane == 0) {
                part[h][sg] = m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (uint h = 0; h < gh; ++h) {
                float m = -INFINITY;
                for (uint i = 0; i < (TG / 32); ++i) {
                    m = max(m, part[h][i]);
                }
                red[h] = m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint h = 0; h < gh; ++h) {
            const float chunk_max = red[h];
            float e = 0.0f;
            if (tid < count) {
                e = exp(scores[h][tid] - chunk_max);
                scores[h][tid] = e;
            }
            const float local_sum = simd_sum(e);
            if (lane == 0) {
                part_sum[h][sg] = local_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (uint h = 0; h < gh; ++h) {
                float s = 0.0f;
                for (uint i = 0; i < (TG / 32); ++i) {
                    s += part_sum[h][i];
                }
                const ulong hq = (ulong)qi * NQ + (ulong)kh * group + h0 + h;
                stats[(hq * splits + split_idx) * 2] = red[h];
                stats[(hq * splits + split_idx) * 2 + 1] = s;
            }
        }

        float4 acc0[QSA_HPP];
        float4 acc1[QSA_HPP];
        for (uint h = 0; h < QSA_HPP; ++h) {
            acc0[h] = float4(0.0f);
            acc1[h] = float4(0.0f);
        }
        for (uint p0 = sg; p0 < count; p0 += QSA_KB * ((TG / 32))) {
            uint2 vq[QSA_KB];
            float vsc[QSA_KB];
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                const uint token = p < count ? tok[p] : tok[p0];
                vq[j] = ((device const uint2*)(v_head + (ulong)token * D))[lane];
                vsc[j] = float(vs_head[(ulong)token * (D / 32) + lane / 4]);
            }
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < QSA_KB; ++j) {
                const uint p = p0 + j * ((TG / 32));
                if (p < count) {
                    const float4 va = float4(as_type<char4>(vq[j].x)) * vsc[j];
                    const float4 vb = float4(as_type<char4>(vq[j].y)) * vsc[j];
                    for (uint h = 0; h < gh; ++h) {
                        const float wgt = scores[h][p];
                        acc0[h] += wgt * va;
                        acc1[h] += wgt * vb;
                    }
                }
            }
        }
        for (uint hb = 0; hb < gh; hb += 2) {
            const uint nh = min(2u, gh - hb);
            for (uint h = 0; h < nh; ++h) {
                for (uint j = 0; j < 4; ++j) {
                    v_stage[h][sg * D + lane * 8 + j] = acc0[hb + h][j];
                    v_stage[h][sg * D + lane * 8 + 4 + j] = acc1[hb + h][j];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint h = 0; h < nh; ++h) {
                const ulong hq = (ulong)qi * NQ + (ulong)kh * group + h0 + hb + h;
                for (uint d = tid; d < D; d += TG) {
                    float acc = 0.0f;
                    for (uint s = 0; s < (TG / 32); ++s) {
                        acc += v_stage[h][s * D + d];
                    }
                    partials[(hq * splits + split_idx) * D + d] = acc;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}


// --- Tensor-op kernels (Metal 4) --------------------------------------------------
#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#define QSA_TILE_D 256   // attention head dim the tensor-op kernels are written for

// --- Indexer scores on the tensor ops ---------------------------------------------
//
// qsa_scores_f32 as a GEMM: the rows of the indexer queries ([QB, 4, D],
// row = query * 4 + head) against the block keys ([nb, D]), sixteen rows
// (four queries' four heads) by 32 blocks per product with fp32
// accumulation, then ReLU and the sum over the four heads in the epilogue.
// Products of bf16 values are exact in fp32, so only the order of the
// accumulation differs from the scalar kernel (and the reference's matmul
// has an order of its own). Grid: threadgroups (ceil(nb_max /
// QSA_SCORE_BN), ceil(QB / 16)), four simdgroups, each with its own four
// queries over the threadgroup's QSA_SCORE_BN blocks. The head sum relies on
// the destination layout the per-query attention documents: a score row
// (query, head) sits in lanes whose bits 1 and 2 are the head.
#define QSA_SCORE_BN 128

kernel void qsa_scores_nax(device bfloat*      q        [[buffer(0)]],  // [QB, 4, D]
                           device bfloat*      blk      [[buffer(1)]],  // [max_blocks, D]
                           device float*       scores   [[buffer(2)]],  // [QB, nb_max]
                           constant uint&      nb_max   [[buffer(3)]],
                           constant uint&      base_pos [[buffer(4)]],
                           constant uint&      ratio    [[buffer(5)]],
                           constant float&     inv_sqrt_d [[buffer(6)]],
                           constant uint&      QB       [[buffer(7)]],
                           uint2 tg   [[threadgroup_position_in_grid]],
                           uint  sg   [[simdgroup_index_in_threadgroup]],
                           uint  lane [[thread_index_in_simdgroup]]) {
    using namespace mpp::tensor_ops;
    constexpr uint D = 128u;
    const uint q0 = tg.y * 16u + sg * 4u;
    if (q0 >= QB) {
        return;  // per simdgroup: the kernel has no barriers
    }
    const uint rows = min(16u, (QB - q0) * 4u);
    auto tA = tensor(q + (ulong)q0 * 4u * D, dextents<int32_t, 2>(int(D), int(rows)));
    auto tB = tensor(blk, dextents<int32_t, 2>(int(D), int(nb_max)));
    constexpr auto desc = matmul2d_descriptor(16, 32, int(D), false, /*transpose_right=*/true,
                                              false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, metal::execution_simdgroup> op;
    const bool writer = ((lane >> 1) & 3u) == 0u;
    _Pragma("clang loop unroll(full)")
    for (uint n = 0; n < QSA_SCORE_BN / 32u; ++n) {
        const uint cb = tg.x * QSA_SCORE_BN + 32u * n;
        if (cb >= nb_max) {
            break;
        }
        auto bS = tB.slice(0, int(cb));
        auto c = op.template get_destination_cooperative_tensor<decltype(tA), decltype(bS), float>();
        op.run(tA, bS, c);
        _Pragma("clang loop unroll(full)")
        for (uint16_t i = 0; i < c.get_capacity(); ++i) {
            auto ix = c.get_multidimensional_index(i);
            float v = c.is_valid_element(i) ? max(c[i], 0.0f) : 0.0f;
            v += simd_shuffle_xor(v, 2);
            v += simd_shuffle_xor(v, 4);
            const uint row = uint(ix[1]);
            const uint qi = q0 + row / 4u;
            const uint b = cb + uint(ix[0]);
            if (writer && qi < QB && b < nb_max) {
                const uint nb = qsa_visible_blocks(base_pos + qi, ratio);
                scores[(ulong)qi * nb_max + b] = b < nb ? v * inv_sqrt_d : -INFINITY;
            }
        }
    }
}

// --- Per-query tensor-op sparse attention ------------------------------------------
//
// One threadgroup per (query, KV head): the GQA group's query heads (12 of
// them) are the rows of one 16-row tensor-op tile, and the pair of
// simdgroups walks the query's own ascending block list and then its tail,
// BK keys per step, each simdgroup owning half of the head dimension. No
// union, no gathered copy: every K and V row is read from the cache
// straight into the right operand of the tensor op (cooperative tensors),
// and the two halves' partial scores are summed through threadgroup memory,
// the one barrier of a step. The online softmax runs in registers on the
// score fragment.
//
// The element layouts of the cooperative tensors are implementation
// defined; this kernel relies on the ones measured on the M5 (the test
// `gqa_attention_operand_layouts_are_the_assumed_ones` checks them): for a
// lane, `quad = (lane & 1) | ((lane >> 3) & 1) << 1` picks four consecutive
// columns (head dims of an operand, keys of a score row) out of every
// sixteen and `slot = ((lane >> 1) & 3) | ((lane >> 4) & 1) << 2` a key (or
// score row) out of every eight; the lanes sharing a score row differ in
// lane bits 0 and 3.
#define QSA_GQA_DH 128   // head dims per simdgroup
#define QSA_GQA_ROWS 16  // tensor-op rows: the GQA group, padded

// The tensor ops take both operands from registers only at 16 or 32 per
// side: every product is 16 x 32 x 32 (scores of 32 keys over 32 dims, or
// 32 output dims over 32 keys), 32 keys per step. Q is re-read (from L1)
// for every product and each V chunk is loaded right before its product:
// holding Q in registers, requesting the step's V rows before the score
// exchange, and 16-byte loads through a head-dim permutation all measured
// slower (docs/performance.md).
kernel void qsa_attn_gqa_nax(device const bfloat* q        [[buffer(0)]],  // [QB, NQ, D]
                             device const bfloat* k_cache  [[buffer(1)]],  // [KVH, max_seq, D]
                             device const bfloat* v_cache  [[buffer(2)]],
                             device const uint*   sel      [[buffer(3)]],  // [QB, k_max]
                             device const uint*   n_sel    [[buffer(4)]],  // [QB]
                             device bfloat*       out      [[buffer(5)]],  // [QB, NQ, D]
                             constant uint&       max_seq  [[buffer(6)]],
                             constant uint&       group    [[buffer(7)]],
                             constant uint&       k_max    [[buffer(8)]],
                             constant uint&       ratio    [[buffer(9)]],
                             constant uint&       base_pos [[buffer(10)]],
                             constant float&      scale    [[buffer(11)]],
                             constant uint&       NQ       [[buffer(12)]],
                             uint2 tg   [[threadgroup_position_in_grid]],  // (KV head, query)
                             uint  sg   [[simdgroup_index_in_threadgroup]],
                             uint  lane [[thread_index_in_simdgroup]]) {
    // Partial scores of the two halves, double-buffered by step parity so
    // one barrier per step suffices.
    threadgroup float xs[2 * 2 * 16 * 32];
    const uint kh = tg.x;
    const uint qi = tg.y;
    using namespace mpp::tensor_ops;
    constexpr uint BK = 32u;

    const uint pos = base_pos + qi;
    const uint nblk = n_sel[qi];
    const uint tail_start = qsa_visible_blocks(pos, ratio) * ratio;
    const uint in_blocks = nblk * ratio;
    const uint total = in_blocks + (pos + 1 - tail_start);
    device const uint* sel_row = sel + (ulong)qi * k_max;
    const uint d0 = sg * uint(QSA_GQA_DH);
    device const bfloat* k_head = k_cache + (ulong)kh * max_seq * QSA_TILE_D + d0;
    device const bfloat* v_head = v_cache + (ulong)kh * max_seq * QSA_TILE_D + d0;
    const uint quad = (lane & 1u) | (((lane >> 3) & 1u) << 1);
    const uint slot = ((lane >> 1) & 3u) | (((lane >> 4) & 1u) << 2);
    // Query rows slot and slot + 8 (rows past the group read row 0 and are
    // never written).
    device const bfloat* q_row0 =
        q + ((ulong)qi * NQ + kh * group + (slot < group ? slot : 0u)) * QSA_TILE_D + d0;
    device const bfloat* q_row1 =
        q + ((ulong)qi * NQ + kh * group + (slot + 8u < group ? slot + 8u : 0u)) * QSA_TILE_D + d0;
    // Memory dim of the lane's first element of half h of chunk c.
    auto dim = [&](uint c, uint h) { return 32u * c + 16u * h + 4u * quad; };
    // The 8 values of halves 0 and 1 of chunk c of a row.
    auto row8 = [&](device const bfloat* row, uint c, thread bfloat4& a, thread bfloat4& b) {
        a = *(device const bfloat4*)(row + dim(c, 0));
        b = *(device const bfloat4*)(row + dim(c, 1));
    };

    constexpr auto qk_desc = matmul2d_descriptor(
        QSA_GQA_ROWS, 32, 32, false, /*transpose_right=*/true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto pv_desc = matmul2d_descriptor(
        QSA_GQA_ROWS, 32, 32, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
    matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
    using QL = decltype(qk_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>());
    using KR = decltype(qk_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>());
    using PL = decltype(pv_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>());
    using VR = decltype(pv_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>());
    using ST = decltype(qk_op.template get_destination_cooperative_tensor<QL, KR, float>());
    using OT = decltype(pv_op.template get_destination_cooperative_tensor<PL, VR, float>());
    // (Arrays of cooperative tensors are not allowed: named ones.)
    OT o0, o1, o2, o3;
    VR w0, w1, w2, w3;
    ST s0;
    PL p0;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 16u; ++i) {
        o0[i] = 0.0f;
        o1[i] = 0.0f;
        o2[i] = 0.0f;
        o3[i] = 0.0f;
    }
    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.0f, 0.0f};
    uint tok[4];

    auto load_q = [&](uint c, thread QL& qL) {
        bfloat4 a0, a1, b0, b1;
        row8(q_row0, c, a0, a1);
        row8(q_row1, c, b0, b1);
        _Pragma("clang loop unroll(full)")
        for (uint e = 0; e < 4; ++e) {
            qL[e] = a0[e];
            qL[4u + e] = b0[e];
            qL[8u + e] = a1[e];
            qL[12u + e] = b1[e];
        }
    };
    // Scores over head-dim chunk c, accumulated into st.
    auto qk_chunk = [&](uint c, thread ST& st) {
        QL qL;
        KR kR;
        load_q(c, qL);
        _Pragma("clang loop unroll(full)")
        for (uint t = 0; t < 4u; ++t) {
            bfloat4 k0v, k1v;
            row8(k_head + (ulong)tok[t] * QSA_TILE_D, c, k0v, k1v);
            _Pragma("clang loop unroll(full)")
            for (uint e = 0; e < 4; ++e) {
                kR[4u * t + e] = k0v[e];
                kR[16u + 4u * t + e] = k1v[e];
            }
        }
        qk_op.run(qL, kR, st);
    };
    // The V rows of output chunk c.
    auto load_v = [&](uint c, thread VR& vr) {
        _Pragma("clang loop unroll(full)")
        for (uint t3 = 0; t3 < 2u; ++t3) {
            _Pragma("clang loop unroll(full)")
            for (uint t2 = 0; t2 < 2u; ++t2) {
                bfloat4 v0v, v1v;
                row8(v_head + (ulong)tok[t2 + 2u * t3] * QSA_TILE_D, c, v0v, v1v);
                _Pragma("clang loop unroll(full)")
                for (uint e = 0; e < 4; ++e) {
                    vr[16u * t3 + 4u * t2 + e] = v0v[e];
                    vr[16u * t3 + 8u + 4u * t2 + e] = v1v[e];
                }
            }
        }
    };
    auto rescale = [&](thread OT& ot, float a0, float a1) {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            ot[i] *= ((i >> 2) & 1u) ? a1 : a0;
        }
    };
    auto store = [&](uint c, thread OT& ot) {
        _Pragma("clang loop unroll(full)")
        for (uint rr = 0; rr < 2u; ++rr) {
            const uint row = slot + 8u * rr;
            if (row < group) {
                const float inv = 1.0f / l_run[rr];
                device bfloat* o = out + ((ulong)qi * NQ + kh * group + row) * QSA_TILE_D + d0;
                _Pragma("clang loop unroll(full)")
                for (uint h = 0; h < 2u; ++h) {
                    _Pragma("clang loop unroll(full)")
                    for (uint e = 0; e < 4; ++e) {
                        o[dim(c, h) + e] = bfloat(ot[8u * h + 4u * rr + e] * inv);
                    }
                }
            }
        }
    };
    // The cache rows of this lane's keys (key slot + 8t of the step); keys
    // past the list read the query's own row, which is finite.
    auto tokens = [&](uint k0, thread uint* tk) {
        _Pragma("clang loop unroll(full)")
        for (uint t = 0; t < 4u; ++t) {
            const uint j = k0 + slot + 8u * t;
            tk[t] = j < in_blocks ? sel_row[j / ratio] * ratio + (j % ratio)
                                  : (j < total ? tail_start + (j - in_blocks) : pos);
        }
    };

    uint parity = 0;
    uint next[4];
    tokens(0, next);
    for (uint k0 = 0; k0 < total; k0 += BK) {
        _Pragma("clang loop unroll(full)")
        for (uint t = 0; t < 4u; ++t) {
            tok[t] = next[t];
        }
        tokens(k0 + BK, next);
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            s0[i] = 0.0f;
        }
        qk_chunk(0, s0);
        qk_chunk(1, s0);
        qk_chunk(2, s0);
        qk_chunk(3, s0);

        // Sum the two head-dim halves (a + b is b + a: both simdgroups hold
        // the same scores afterwards).
        threadgroup float* mine = xs + (parity * 2u + sg) * (16u * 32u);
        threadgroup float* other = xs + (parity * 2u + (1u - sg)) * (16u * 32u);
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            mine[i * 32u + lane] = s0[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s[16];
        float mx[2] = {m_run[0], m_run[1]};
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            const uint col = 16u * (i >> 3) + 4u * quad + (i & 3u);
            const uint rr = (i >> 2) & 1u;
            const float v = (s0[i] + other[i * 32u + lane]) * scale;
            s[i] = k0 + col < total ? v : -INFINITY;
            mx[rr] = max(mx[rr], s[i]);
        }
        float alpha[2];
        float psum[2] = {0.0f, 0.0f};
        _Pragma("clang loop unroll(full)")
        for (uint rr = 0; rr < 2; ++rr) {
            mx[rr] = max(mx[rr], simd_shuffle_xor(mx[rr], 1));
            mx[rr] = max(mx[rr], simd_shuffle_xor(mx[rr], 8));
            alpha[rr] = mx[rr] == -INFINITY ? 1.0f : exp(m_run[rr] - mx[rr]);
        }
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            const uint rr = (i >> 2) & 1u;
            const float p = s[i] == -INFINITY ? 0.0f : exp(s[i] - mx[rr]);
            p0[i] = bfloat(p);
            psum[rr] += p;
        }
        _Pragma("clang loop unroll(full)")
        for (uint rr = 0; rr < 2; ++rr) {
            psum[rr] += simd_shuffle_xor(psum[rr], 1);
            psum[rr] += simd_shuffle_xor(psum[rr], 8);
            l_run[rr] = l_run[rr] * alpha[rr] + psum[rr];
            m_run[rr] = mx[rr];
        }
        if (alpha[0] != 1.0f || alpha[1] != 1.0f) {
            rescale(o0, alpha[0], alpha[1]);
            rescale(o1, alpha[0], alpha[1]);
            rescale(o2, alpha[0], alpha[1]);
            rescale(o3, alpha[0], alpha[1]);
        }
        load_v(0, w0);
        pv_op.run(p0, w0, o0);
        load_v(1, w1);
        pv_op.run(p0, w1, o1);
        load_v(2, w2);
        pv_op.run(p0, w2, o2);
        load_v(3, w3);
        pv_op.run(p0, w3, o3);
        parity ^= 1u;
    }
    store(0, o0);
    store(1, o1);
    store(2, o2);
    store(3, o3);
}

// qsa_attn_gqa_nax over a q8 cache: the same body, each K and V chunk read
// as bytes and dequantized with its group's scale (a 32-dim chunk is one
// q8 group).
kernel void qsa_attn_gqa_nax_q8(device const bfloat* q        [[buffer(0)]],  // [QB, NQ, D]
                             device const char*   k_cache  [[buffer(1)]],  // [KVH, max_seq, D]
                             device const char*   v_cache  [[buffer(2)]],
                             device const half*   k_scales [[buffer(3)]],  // [KVH, max_seq, D / 32]
                             device const half*   v_scales [[buffer(4)]],
                             device const uint*   sel      [[buffer(5)]],  // [QB, k_max]
                             device const uint*   n_sel    [[buffer(6)]],  // [QB]
                             device bfloat*       out      [[buffer(7)]],  // [QB, NQ, D]
                             constant uint&       max_seq  [[buffer(8)]],
                             constant uint&       group    [[buffer(9)]],
                             constant uint&       k_max    [[buffer(10)]],
                             constant uint&       ratio    [[buffer(11)]],
                             constant uint&       base_pos [[buffer(12)]],
                             constant float&      scale    [[buffer(13)]],
                             constant uint&       NQ       [[buffer(14)]],
                             uint2 tg   [[threadgroup_position_in_grid]],  // (KV head, query)
                             uint  sg   [[simdgroup_index_in_threadgroup]],
                             uint  lane [[thread_index_in_simdgroup]]) {
    // Partial scores of the two halves, double-buffered by step parity so
    // one barrier per step suffices.
    threadgroup float xs[2 * 2 * 16 * 32];
    const uint kh = tg.x;
    const uint qi = tg.y;
    using namespace mpp::tensor_ops;
    constexpr uint BK = 32u;

    const uint pos = base_pos + qi;
    const uint nblk = n_sel[qi];
    const uint tail_start = qsa_visible_blocks(pos, ratio) * ratio;
    const uint in_blocks = nblk * ratio;
    const uint total = in_blocks + (pos + 1 - tail_start);
    device const uint* sel_row = sel + (ulong)qi * k_max;
    const uint d0 = sg * uint(QSA_GQA_DH);
    device const char* k_head = k_cache + (ulong)kh * max_seq * QSA_TILE_D + d0;
    device const char* v_head = v_cache + (ulong)kh * max_seq * QSA_TILE_D + d0;
    // Head-dim chunk c of this simdgroup is group d0 / 32 + c of a row.
    constexpr uint QG = QSA_TILE_D / 32u;
    device const half* ks_head = k_scales + (ulong)kh * max_seq * QG + d0 / 32u;
    device const half* vs_head = v_scales + (ulong)kh * max_seq * QG + d0 / 32u;
    const uint quad = (lane & 1u) | (((lane >> 3) & 1u) << 1);
    const uint slot = ((lane >> 1) & 3u) | (((lane >> 4) & 1u) << 2);
    // Query rows slot and slot + 8 (rows past the group read row 0 and are
    // never written).
    device const bfloat* q_row0 =
        q + ((ulong)qi * NQ + kh * group + (slot < group ? slot : 0u)) * QSA_TILE_D + d0;
    device const bfloat* q_row1 =
        q + ((ulong)qi * NQ + kh * group + (slot + 8u < group ? slot + 8u : 0u)) * QSA_TILE_D + d0;
    // Memory dim of the lane's first element of half h of chunk c.
    auto dim = [&](uint c, uint h) { return 32u * c + 16u * h + 4u * quad; };
    // The 8 values of halves 0 and 1 of chunk c of a row.
    // ...dequantized with the chunk's scale and rounded to bf16, the tensor
    // ops' operand type.
    auto row8 = [&](device const char* row, float sc, uint c, thread bfloat4& a, thread bfloat4& b) {
        a = bfloat4(float4(*(device const char4*)(row + dim(c, 0))) * sc);
        b = bfloat4(float4(*(device const char4*)(row + dim(c, 1))) * sc);
    };

    constexpr auto qk_desc = matmul2d_descriptor(
        QSA_GQA_ROWS, 32, 32, false, /*transpose_right=*/true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto pv_desc = matmul2d_descriptor(
        QSA_GQA_ROWS, 32, 32, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
    matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
    using QL = decltype(qk_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>());
    using KR = decltype(qk_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>());
    using PL = decltype(pv_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>());
    using VR = decltype(pv_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>());
    using ST = decltype(qk_op.template get_destination_cooperative_tensor<QL, KR, float>());
    using OT = decltype(pv_op.template get_destination_cooperative_tensor<PL, VR, float>());
    // (Arrays of cooperative tensors are not allowed: named ones.)
    OT o0, o1, o2, o3;
    VR w0, w1, w2, w3;
    ST s0;
    PL p0;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 16u; ++i) {
        o0[i] = 0.0f;
        o1[i] = 0.0f;
        o2[i] = 0.0f;
        o3[i] = 0.0f;
    }
    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.0f, 0.0f};
    uint tok[4];

    auto load_q = [&](uint c, thread QL& qL) {
        bfloat4 a0, a1, b0, b1;
        a0 = *(device const bfloat4*)(q_row0 + dim(c, 0));
        a1 = *(device const bfloat4*)(q_row0 + dim(c, 1));
        b0 = *(device const bfloat4*)(q_row1 + dim(c, 0));
        b1 = *(device const bfloat4*)(q_row1 + dim(c, 1));
        _Pragma("clang loop unroll(full)")
        for (uint e = 0; e < 4; ++e) {
            qL[e] = a0[e];
            qL[4u + e] = b0[e];
            qL[8u + e] = a1[e];
            qL[12u + e] = b1[e];
        }
    };
    // Scores over head-dim chunk c, accumulated into st.
    auto qk_chunk = [&](uint c, thread ST& st) {
        QL qL;
        KR kR;
        load_q(c, qL);
        _Pragma("clang loop unroll(full)")
        for (uint t = 0; t < 4u; ++t) {
            bfloat4 k0v, k1v;
            row8(k_head + (ulong)tok[t] * QSA_TILE_D,
                 float(ks_head[(ulong)tok[t] * QG + c]), c, k0v, k1v);
            _Pragma("clang loop unroll(full)")
            for (uint e = 0; e < 4; ++e) {
                kR[4u * t + e] = k0v[e];
                kR[16u + 4u * t + e] = k1v[e];
            }
        }
        qk_op.run(qL, kR, st);
    };
    // The V rows of output chunk c.
    auto load_v = [&](uint c, thread VR& vr) {
        _Pragma("clang loop unroll(full)")
        for (uint t3 = 0; t3 < 2u; ++t3) {
            _Pragma("clang loop unroll(full)")
            for (uint t2 = 0; t2 < 2u; ++t2) {
                bfloat4 v0v, v1v;
                const uint vt = tok[t2 + 2u * t3];
                row8(v_head + (ulong)vt * QSA_TILE_D, float(vs_head[(ulong)vt * QG + c]),
                     c, v0v, v1v);
                _Pragma("clang loop unroll(full)")
                for (uint e = 0; e < 4; ++e) {
                    vr[16u * t3 + 4u * t2 + e] = v0v[e];
                    vr[16u * t3 + 8u + 4u * t2 + e] = v1v[e];
                }
            }
        }
    };
    auto rescale = [&](thread OT& ot, float a0, float a1) {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            ot[i] *= ((i >> 2) & 1u) ? a1 : a0;
        }
    };
    auto store = [&](uint c, thread OT& ot) {
        _Pragma("clang loop unroll(full)")
        for (uint rr = 0; rr < 2u; ++rr) {
            const uint row = slot + 8u * rr;
            if (row < group) {
                const float inv = 1.0f / l_run[rr];
                device bfloat* o = out + ((ulong)qi * NQ + kh * group + row) * QSA_TILE_D + d0;
                _Pragma("clang loop unroll(full)")
                for (uint h = 0; h < 2u; ++h) {
                    _Pragma("clang loop unroll(full)")
                    for (uint e = 0; e < 4; ++e) {
                        o[dim(c, h) + e] = bfloat(ot[8u * h + 4u * rr + e] * inv);
                    }
                }
            }
        }
    };
    // The cache rows of this lane's keys (key slot + 8t of the step); keys
    // past the list read the query's own row, which is finite.
    auto tokens = [&](uint k0, thread uint* tk) {
        _Pragma("clang loop unroll(full)")
        for (uint t = 0; t < 4u; ++t) {
            const uint j = k0 + slot + 8u * t;
            tk[t] = j < in_blocks ? sel_row[j / ratio] * ratio + (j % ratio)
                                  : (j < total ? tail_start + (j - in_blocks) : pos);
        }
    };

    uint parity = 0;
    uint next[4];
    tokens(0, next);
    for (uint k0 = 0; k0 < total; k0 += BK) {
        _Pragma("clang loop unroll(full)")
        for (uint t = 0; t < 4u; ++t) {
            tok[t] = next[t];
        }
        tokens(k0 + BK, next);
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            s0[i] = 0.0f;
        }
        qk_chunk(0, s0);
        qk_chunk(1, s0);
        qk_chunk(2, s0);
        qk_chunk(3, s0);

        // Sum the two head-dim halves (a + b is b + a: both simdgroups hold
        // the same scores afterwards).
        threadgroup float* mine = xs + (parity * 2u + sg) * (16u * 32u);
        threadgroup float* other = xs + (parity * 2u + (1u - sg)) * (16u * 32u);
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            mine[i * 32u + lane] = s0[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s[16];
        float mx[2] = {m_run[0], m_run[1]};
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            const uint col = 16u * (i >> 3) + 4u * quad + (i & 3u);
            const uint rr = (i >> 2) & 1u;
            const float v = (s0[i] + other[i * 32u + lane]) * scale;
            s[i] = k0 + col < total ? v : -INFINITY;
            mx[rr] = max(mx[rr], s[i]);
        }
        float alpha[2];
        float psum[2] = {0.0f, 0.0f};
        _Pragma("clang loop unroll(full)")
        for (uint rr = 0; rr < 2; ++rr) {
            mx[rr] = max(mx[rr], simd_shuffle_xor(mx[rr], 1));
            mx[rr] = max(mx[rr], simd_shuffle_xor(mx[rr], 8));
            alpha[rr] = mx[rr] == -INFINITY ? 1.0f : exp(m_run[rr] - mx[rr]);
        }
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 16u; ++i) {
            const uint rr = (i >> 2) & 1u;
            const float p = s[i] == -INFINITY ? 0.0f : exp(s[i] - mx[rr]);
            p0[i] = bfloat(p);
            psum[rr] += p;
        }
        _Pragma("clang loop unroll(full)")
        for (uint rr = 0; rr < 2; ++rr) {
            psum[rr] += simd_shuffle_xor(psum[rr], 1);
            psum[rr] += simd_shuffle_xor(psum[rr], 8);
            l_run[rr] = l_run[rr] * alpha[rr] + psum[rr];
            m_run[rr] = mx[rr];
        }
        if (alpha[0] != 1.0f || alpha[1] != 1.0f) {
            rescale(o0, alpha[0], alpha[1]);
            rescale(o1, alpha[0], alpha[1]);
            rescale(o2, alpha[0], alpha[1]);
            rescale(o3, alpha[0], alpha[1]);
        }
        load_v(0, w0);
        pv_op.run(p0, w0, o0);
        load_v(1, w1);
        pv_op.run(p0, w1, o1);
        load_v(2, w2);
        pv_op.run(p0, w2, o2);
        load_v(3, w3);
        pv_op.run(p0, w3, o3);
        parity ^= 1u;
    }
    store(0, o0);
    store(1, o1);
    store(2, o2);
    store(3, o3);
}



#endif
