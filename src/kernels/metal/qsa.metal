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
#define QSA_UNION_TG 1024  // threads of the per-tile selection union

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
kernel void qsa_attn_split_bf16(device const bfloat* q        [[buffer(0)]],  // [QB, NQ, D]
                                device const bfloat* k_cache  [[buffer(1)]],  // [KVH, max_seq, D]
                                device const bfloat* v_cache  [[buffer(2)]],
                                device const uint*   sel      [[buffer(3)]],  // [QB, k_max]
                                device const uint*   n_sel    [[buffer(4)]],  // [QB]
                                device float*        partials [[buffer(5)]],  // [QB*NQ, splits, D]
                                device float*        stats    [[buffer(6)]],  // [QB*NQ, splits, 2]
                                constant uint&       D        [[buffer(7)]],
                                constant uint&       max_seq  [[buffer(8)]],
                                constant uint&       group    [[buffer(9)]],   // NQ / KVH
                                constant uint&       splits   [[buffer(10)]],
                                constant uint&       k_max    [[buffer(11)]],
                                constant uint&       ratio    [[buffer(12)]],
                                constant uint&       base_pos [[buffer(13)]],
                                constant float&      scale    [[buffer(14)]],
                                constant uint&       NQ       [[buffer(15)]],
                                constant uint&       split    [[buffer(16)]],  // tokens per threadgroup
                                constant uint&       head_groups [[buffer(17)]],  // threadgroups per KV head
                                uint3 tg  [[threadgroup_position_in_grid]],
                                uint tid  [[thread_index_in_threadgroup]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[QSA_HPP][QSA_SPLIT];
    threadgroup float part[QSA_HPP][TG / 32];
    threadgroup float red[QSA_HPP];
    threadgroup float v_stage[(TG / 32) * 256];

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
    // This threadgroup's query heads within the KV head.
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
        for (uint p = sg; p < count; p += TG / 32) {
            const uint token = qsa_token_at(sel_row, nblk, ratio, tail_start, slot0 + p);
            const uint4 kq = ((device const uint4*)(k_head + (ulong)token * D))[lane];
            const float4 ka = float4(as_type<bfloat4>(kq.xy));
            const float4 kb = float4(as_type<bfloat4>(kq.zw));
            for (uint h = 0; h < gh; ++h) {
                const float s = simd_sum(dot(ka, qa[h]) + dot(kb, qb[h])) * scale;
                if (lane == 0) {
                    scores[h][p] = s;
                }
                local_max[h] = max(local_max[h], s);
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
                for (uint i = 0; i < TG / 32; ++i) {
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
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
                part[h][sg] = local_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float s = 0.0f;
                for (uint i = 0; i < TG / 32; ++i) {
                    s += part[h][i];
                }
                const ulong hq = (ulong)qi * NQ + (ulong)kh * group + h0 + h;
                stats[(hq * splits + split_idx) * 2] = chunk_max;
                stats[(hq * splits + split_idx) * 2 + 1] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float4 acc0[QSA_HPP];
        float4 acc1[QSA_HPP];
        for (uint h = 0; h < QSA_HPP; ++h) {
            acc0[h] = float4(0.0f);
            acc1[h] = float4(0.0f);
        }
        for (uint p = sg; p < count; p += TG / 32) {
            const uint token = qsa_token_at(sel_row, nblk, ratio, tail_start, slot0 + p);
            const uint4 vq = ((device const uint4*)(v_head + (ulong)token * D))[lane];
            const float4 va = float4(as_type<bfloat4>(vq.xy));
            const float4 vb = float4(as_type<bfloat4>(vq.zw));
            for (uint h = 0; h < gh; ++h) {
                const float wgt = scores[h][p];
                acc0[h] += wgt * va;
                acc1[h] += wgt * vb;
            }
        }
        for (uint h = 0; h < gh; ++h) {
            for (uint j = 0; j < 4; ++j) {
                v_stage[sg * D + lane * 8 + j] = acc0[h][j];
                v_stage[sg * D + lane * 8 + 4 + j] = acc1[h][j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const ulong hq = (ulong)qi * NQ + (ulong)kh * group + h0 + h;
            for (uint d = tid; d < D; d += TG) {
                float acc = 0.0f;
                for (uint s = 0; s < TG / 32; ++s) {
                    acc += v_stage[s * D + d];
                }
                partials[(hq * splits + split_idx) * D + d] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// --- Tiled sparse attention (Metal 4 tensor ops) ----------------------------------
//
// Prefill past the dense limit. Adjacent queries select largely the same
// blocks, so a tile of QSA_TILE_BQ consecutive queries attends the union of
// its selections through the tensor-op path, each query masked to its own
// selection. `qsa_tile_union` builds the union per tile; the attention kernel
// stages the union's K/V rows into threadgroup memory a tile at a time and
// runs the same flash loop as the dense prefill kernel over them.

#define QSA_TILE_BQ 16
#define QSA_TILE_TAIL_BLOCKS 8  // blocks the tail region of a tile can span (at most 5)

// Per tile of QSA_TILE_BQ queries (query i of tile t at base_pos + t*BQ + i):
// the ascending union of the queries' selected blocks below `vb0`, the block
// count visible to the tile's first query, with a bit per query in
// `union_mask`; `tail_mask[j]` holds the bits for block vb0 + j, which only
// the later queries of the tile see complete. `mask` ([tiles, nb_cap]) is one
// scratch row per tile: zero on entry, zero again on exit. `stats` accumulates
// the union sizes (blocks, tiles) for measurement. One threadgroup of
// QSA_UNION_TG threads per tile.
kernel void qsa_tile_union(device const uint*  sel        [[buffer(0)]],  // [QB, k_max]
                           device const uint*  n_sel      [[buffer(1)]],  // [QB]
                           device atomic_uint* mask       [[buffer(2)]],  // [tiles, nb_cap]
                           device uint*        union_blk  [[buffer(3)]],  // [tiles, cap]
                           device uint*        union_mask [[buffer(4)]],  // [tiles, cap]
                           device uint*        n_union    [[buffer(5)]],  // [tiles]
                           device uint*        tail_mask  [[buffer(6)]],  // [tiles, QSA_TILE_TAIL_BLOCKS]
                           device atomic_uint* stats      [[buffer(7)]],  // [2]
                           constant uint&      QB         [[buffer(8)]],
                           constant uint&      k_max      [[buffer(9)]],
                           constant uint&      ratio      [[buffer(10)]],
                           constant uint&      base_pos   [[buffer(11)]],
                           constant uint&      nb_cap     [[buffer(12)]],
                           constant uint&      cap        [[buffer(13)]],
                           uint tile [[threadgroup_position_in_grid]],
                           uint tid  [[thread_index_in_threadgroup]],
                           uint sg   [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint tail[QSA_TILE_TAIL_BLOCKS];
    threadgroup uint sums[QSA_UNION_TG / 32];

    const uint q0 = tile * QSA_TILE_BQ;
    if (q0 >= QB) {
        return;
    }
    const uint qn = min(uint(QSA_TILE_BQ), QB - q0);
    const uint vb0 = qsa_visible_blocks(base_pos + q0, ratio);
    device atomic_uint* row = mask + (ulong)tile * nb_cap;
    if (tid < QSA_TILE_TAIL_BLOCKS) {
        atomic_store_explicit(&tail[tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter every query's selection into the tile's mask row (bit = query).
    for (uint i = tid; i < qn * k_max; i += QSA_UNION_TG) {
        const uint qi = i / k_max;
        const uint j = i - qi * k_max;
        if (j < n_sel[q0 + qi]) {
            const uint b = sel[(ulong)(q0 + qi) * k_max + j];
            const uint bit = 1u << qi;
            if (b < vb0) {
                atomic_fetch_or_explicit(&row[b], bit, memory_order_relaxed);
            } else if (b - vb0 < QSA_TILE_TAIL_BLOCKS) {
                atomic_fetch_or_explicit(&tail[b - vb0], bit, memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // Compact the non-empty entries in ascending block order and clear them.
    uint base = 0;
    for (uint b0 = 0; b0 < vb0; b0 += QSA_UNION_TG) {
        const uint b = b0 + tid;
        const uint m = b < vb0 ? atomic_load_explicit(&row[b], memory_order_relaxed) : 0u;
        const uint take = m != 0u ? 1u : 0u;
        uint take_total;
        const uint rank = qsa_scan<QSA_UNION_TG>(take, sums, take_total, sg, lane);
        if (take) {
            const uint r = base + rank;
            if (r < cap) {
                union_blk[(ulong)tile * cap + r] = b;
                union_mask[(ulong)tile * cap + r] = m;
            }
            atomic_store_explicit(&row[b], 0u, memory_order_relaxed);
        }
        base += take_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < QSA_TILE_TAIL_BLOCKS) {
        tail_mask[tile * QSA_TILE_TAIL_BLOCKS + tid] =
            atomic_load_explicit(&tail[tid], memory_order_relaxed);
    }
    if (tid == 0) {
        const uint n = min(base, cap);
        n_union[tile] = n;
        atomic_fetch_add_explicit(&stats[0], n, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
    }
}

#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#define QSA_TILE_D 256   // attention head dim the tile kernel is written for
#define QSA_TILE_SG 4    // simdgroups per threadgroup

// Flash attention of one tile of BQ consecutive queries and HPP query heads
// (all within one KV head) over the tile's union of selected blocks plus its
// tail region, every query masked to the tokens it attends. The union's K
// rows are gathered into threadgroup memory BK at a time (block by block,
// `ratio` consecutive cache rows per block), scored against every head with
// the tensor-op matmul, softmaxed under the mask, then the same rows of V are
// staged over K for the P·V accumulation. Grid: threadgroups
// (ceil(M / BQ), NQ / HPP), 32 * QSA_TILE_SG threads.
template <int BQ, int BK, int HPP>
static void qsa_attn_tile_body(device bfloat* q,
                               device bfloat* k_cache,
                               device bfloat* v_cache,
                               device const uint* union_blk,
                               device const uint* union_mask,
                               device const uint* n_union,
                               device const uint* tail_mask,
                               device bfloat* out,
                               uint max_seq,
                               uint base_pos,
                               uint M,
                               uint NQ,
                               uint group,
                               float scale,
                               uint cap,
                               uint ratio,
                               threadgroup uint4* kv4,
                               threadgroup float* s_tile,
                               threadgroup bfloat* p_tile,
                               threadgroup uint* tok_idx,
                               threadgroup uint* tok_mask,
                               threadgroup float* row_max,
                               threadgroup float* row_sum,
                               threadgroup float* row_alpha,
                               uint2 tg,
                               uint tid,
                               uint2 tg_size) {
    constexpr uint THREADS = 32u * uint(QSA_TILE_SG);
    constexpr uint LANES = THREADS / uint(BQ);
    constexpr uint COLS = uint(BK) / LANES;
    constexpr uint ROW4 = uint(QSA_TILE_D) * 2u / 16u;  // uint4 words per K/V row
    static_assert(uint(BQ) * LANES == THREADS,
                  "the threadgroup must divide into BQ equal row groups");
    static_assert(uint(BK) % LANES == 0u, "each lane must own a whole number of key columns");
    static_assert(LANES <= 32u, "a row's lanes must sit inside one simdgroup");
    static_assert(HPP >= 1 && HPP <= 4, "one to four heads per staged tile");
    static_assert(BQ <= 32, "the query mask is one uint");

    using namespace mpp::tensor_ops;

    if (tg_size.x != THREADS) {
        return;
    }
    const uint tile = tg.x;
    const uint q0 = tile * uint(BQ);
    if (q0 >= M) {
        return;
    }
    const uint qn = min(uint(BQ), M - q0);
    const uint h0 = tg.y * uint(HPP);
    const uint kh = h0 / group;
    const uint p0 = base_pos + q0;
    const uint vb0 = qsa_visible_blocks(p0, ratio);
    const uint n_u = min(n_union[tile], cap);
    const uint in_blocks = n_u * ratio;
    const uint tail_start = vb0 * ratio;
    const uint total = in_blocks + (p0 + qn - tail_start);
    device const uint* ublk = union_blk + (ulong)tile * cap;
    device const uint* umask = union_mask + (ulong)tile * cap;
    device const uint* tmask = tail_mask + (ulong)tile * QSA_TILE_TAIL_BLOCKS;
    device const uint4* k_head =
        (device const uint4*)(k_cache + (ulong)kh * max_seq * QSA_TILE_D);
    device const uint4* v_head =
        (device const uint4*)(v_cache + (ulong)kh * max_seq * QSA_TILE_D);
    threadgroup bfloat* kv_tile = (threadgroup bfloat*)kv4;

    if (tid < uint(HPP * BQ)) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
        row_alpha[tid] = 1.0f;
    }

    auto tKV = tensor(kv_tile, dextents<int32_t, 2>(QSA_TILE_D, BK));
    auto tS = tensor(s_tile, dextents<int32_t, 2>(BK, BQ));
    constexpr auto qk_desc = matmul2d_descriptor(
        BQ, BK, QSA_TILE_D, false, /*transpose_right=*/true, false,
        matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(
        BQ, QSA_TILE_D, BK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, metal::execution_simdgroups<QSA_TILE_SG>> qk_op;
    matmul2d<pv_desc, metal::execution_simdgroups<QSA_TILE_SG>> pv_op;

    const array<int, 2> q_strides{1, int(NQ * uint(QSA_TILE_D))};
    auto q_tensor = [&](uint h) {
        return tensor(q + ((ulong)q0 * NQ + h0 + h) * QSA_TILE_D,
                      dextents<int32_t, 2>(QSA_TILE_D, int(qn)), q_strides);
    };
    auto p_tensor = [&](uint h) {
        return tensor(p_tile + h * uint(BQ * BK), dextents<int32_t, 2>(BK, BQ));
    };
    using QT = decltype(q_tensor(0u));
    using PT = decltype(p_tensor(0u));
    using KVT = decltype(tKV);
    auto make_acc = [&]() {
        return pv_op.template get_destination_cooperative_tensor<PT, KVT, float>();
    };
    using AccT = decltype(make_acc());
    // One accumulator per head of the pass, selected at compile time so each
    // stays in registers (the unused ones fold away).
    AccT acc0 = make_acc();
    AccT acc1 = make_acc();
    AccT acc2 = make_acc();
    AccT acc3 = make_acc();
#define QSA_EACH_HEAD(FN)   \
    FN(0u, acc0);           \
    if (HPP > 1) FN(1u, acc1); \
    if (HPP > 2) FN(2u, acc2); \
    if (HPP > 3) FN(3u, acc3);
    auto zero_acc = [&](uint, thread AccT& acc) {
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                acc[i] = 0.0f;
            }
        }
    };
    QSA_EACH_HEAD(zero_acc)
    // Rescales the head's accumulator by its rows' softmax correction and
    // adds this slice's P·V.
    auto pv_step = [&](uint h, thread AccT& acc) {
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                auto ix = acc.get_multidimensional_index(i);
                acc[i] *= row_alpha[h * uint(BQ) + uint(ix[1])];
            }
        }
        auto tP = p_tensor(h);
        pv_op.run(tP, tKV, acc);
    };
    auto store_acc = [&](uint h, thread AccT& acc) {
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                auto ix = acc.get_multidimensional_index(i);
                const uint row = uint(ix[1]);
                const uint col = uint(ix[0]);
                if (row < qn) {
                    out[((ulong)(q0 + row) * NQ + h0 + h) * QSA_TILE_D + col] =
                        bfloat(acc[i] / row_sum[h * uint(BQ) + row]);
                }
            }
        }
    };

    for (uint k0 = 0; k0 < total; k0 += uint(BK)) {
        const uint count = min(uint(BK), total - k0);
        // Which cache rows this slice of the union holds, and who attends them.
        if (tid < uint(BK)) {
            const uint j = k0 + tid;
            uint token = 0u;
            uint m = 0u;
            if (j < in_blocks) {
                const uint u = j / ratio;
                token = ublk[u] * ratio + (j - u * ratio);
                m = umask[u];
            } else if (j < total) {
                // Tail region: a row every query at or past it attends
                // causally, and earlier queries attend only if they selected
                // its (for them complete) block.
                token = tail_start + (j - in_blocks);
                const uint tb = token / ratio - vb0;
                const uint tbits = tb < QSA_TILE_TAIL_BLOCKS ? tmask[tb] : 0u;
                for (uint i = 0; i < qn; ++i) {
                    const uint p = p0 + i;
                    const bool causal = token <= p;
                    const bool own_tail = token >= qsa_visible_blocks(p, ratio) * ratio;
                    if (causal && (own_tail || ((tbits >> i) & 1u))) {
                        m |= 1u << i;
                    }
                }
            }
            tok_idx[tid] = token;
            tok_mask[tid] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage K rows.
        for (uint i = tid; i < uint(BK) * ROW4; i += THREADS) {
            const uint r = i / ROW4;
            const uint c = i - r * ROW4;
            kv4[i] = r < count ? k_head[(ulong)tok_idx[r] * ROW4 + c] : uint4(0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint h = 0; h < uint(HPP); ++h) {
            auto tQ = q_tensor(h);
            auto sT = qk_op.template get_destination_cooperative_tensor<QT, KVT, float>();
            qk_op.run(tQ, tKV, sT);
            sT.store(tS);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Masked online softmax: LANES lanes per query row.
            const uint row = tid / LANES;
            const uint l = tid - row * LANES;
            const uint j0 = l * COLS;
            const uint stat = h * uint(BQ) + row;
            const float prev = row_max[stat];
            float local = prev;
            for (uint j = j0; j < j0 + COLS; ++j) {
                if (j < count && ((tok_mask[j] >> row) & 1u)) {
                    local = max(local, s_tile[row * uint(BK) + j] * scale);
                }
            }
            for (uint off = 1u; off < LANES; off <<= 1) {
                local = max(local, simd_shuffle_xor(local, off));
            }
            const float mx = local;
            // A row that has attended nothing so far keeps its state.
            const float alpha = mx == -INFINITY ? 1.0f : exp(prev - mx);
            float psum = 0.0f;
            for (uint j = j0; j < j0 + COLS; ++j) {
                const bool on = j < count && ((tok_mask[j] >> row) & 1u);
                const float p = on ? exp(s_tile[row * uint(BK) + j] * scale - mx) : 0.0f;
                p_tile[h * uint(BQ * BK) + row * uint(BK) + j] = bfloat(p);
                psum += p;
            }
            for (uint off = 1u; off < LANES; off <<= 1) {
                psum += simd_shuffle_xor(psum, off);
            }
            if (l == 0u) {
                row_sum[stat] = row_sum[stat] * alpha + psum;
                row_max[stat] = mx;
                row_alpha[stat] = alpha;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Stage V rows over K.
        for (uint i = tid; i < uint(BK) * ROW4; i += THREADS) {
            const uint r = i / ROW4;
            const uint c = i - r * ROW4;
            kv4[i] = r < count ? v_head[(ulong)tok_idx[r] * ROW4 + c] : uint4(0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        QSA_EACH_HEAD(pv_step)
        // Finish the tensor ops before the next slice restages kv4/p_tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    QSA_EACH_HEAD(store_acc)
#undef QSA_EACH_HEAD
}

// The one-head tile kernel with its K/V staging software-pipelined: while a
// slice's scores and softmax run on the staged K rows, each thread already
// holds the slice's V rows in registers, and while P.V runs on the staged V
// rows it holds the next slice's K rows, so the gathers' latency overlaps
// the tensor ops instead of being exposed twice per slice. With EARLY the
// next slice's K rows are fetched a whole slice earlier (twice the
// registers; measured no better). Same math and accumulation order as
// qsa_attn_tile_body with HPP = 1.
template <int BQ, int BK, bool EARLY>
static void qsa_attn_tile_pipe_body(device bfloat* q,
                                    device bfloat* k_cache,
                                    device bfloat* v_cache,
                                    device const uint* union_blk,
                                    device const uint* union_mask,
                                    device const uint* n_union,
                                    device const uint* tail_mask,
                                    device bfloat* out,
                                    uint max_seq,
                                    uint base_pos,
                                    uint M,
                                    uint NQ,
                                    uint group,
                                    float scale,
                                    uint cap,
                                    uint ratio,
                                    threadgroup uint4* kv4,
                                    threadgroup float* s_tile,
                                    threadgroup bfloat* p_tile,
                                    threadgroup uint* tok_idx,   // [2][BK]
                                    threadgroup uint* tok_mask,  // [2][BK]
                                    threadgroup float* row_max,
                                    threadgroup float* row_sum,
                                    threadgroup float* row_alpha,
                                    uint2 tg,
                                    uint tid,
                                    uint2 tg_size) {
    constexpr uint THREADS = 32u * uint(QSA_TILE_SG);
    constexpr uint LANES = THREADS / uint(BQ);
    constexpr uint COLS = uint(BK) / LANES;
    constexpr uint ROW4 = uint(QSA_TILE_D) * 2u / 16u;  // uint4 words per K/V row
    constexpr uint WORDS = uint(BK) * ROW4;
    constexpr uint PER_THREAD = WORDS / THREADS;
    static_assert(WORDS % THREADS == 0u, "the staged rows split evenly over the threads");
    static_assert(uint(BQ) * LANES == THREADS,
                  "the threadgroup must divide into BQ equal row groups");
    static_assert(uint(BK) % LANES == 0u, "each lane must own a whole number of key columns");
    static_assert(LANES <= 32u, "a row's lanes must sit inside one simdgroup");
    static_assert(BQ <= 32, "the query mask is one uint");

    using namespace mpp::tensor_ops;

    if (tg_size.x != THREADS) {
        return;
    }
    const uint tile = tg.x;
    const uint q0 = tile * uint(BQ);
    if (q0 >= M) {
        return;
    }
    const uint qn = min(uint(BQ), M - q0);
    const uint h = tg.y;
    const uint kh = h / group;
    const uint p0 = base_pos + q0;
    const uint vb0 = qsa_visible_blocks(p0, ratio);
    const uint n_u = min(n_union[tile], cap);
    const uint in_blocks = n_u * ratio;
    const uint tail_start = vb0 * ratio;
    const uint total = in_blocks + (p0 + qn - tail_start);
    device const uint* ublk = union_blk + (ulong)tile * cap;
    device const uint* umask = union_mask + (ulong)tile * cap;
    device const uint* tmask = tail_mask + (ulong)tile * QSA_TILE_TAIL_BLOCKS;
    device const uint4* k_head =
        (device const uint4*)(k_cache + (ulong)kh * max_seq * QSA_TILE_D);
    device const uint4* v_head =
        (device const uint4*)(v_cache + (ulong)kh * max_seq * QSA_TILE_D);
    threadgroup bfloat* kv_tile = (threadgroup bfloat*)kv4;

    if (tid < uint(BQ)) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
        row_alpha[tid] = 1.0f;
    }

    auto tKV = tensor(kv_tile, dextents<int32_t, 2>(QSA_TILE_D, BK));
    auto tS = tensor(s_tile, dextents<int32_t, 2>(BK, BQ));
    auto tP = tensor(p_tile, dextents<int32_t, 2>(BK, BQ));
    constexpr auto qk_desc = matmul2d_descriptor(
        BQ, BK, QSA_TILE_D, false, /*transpose_right=*/true, false,
        matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(
        BQ, QSA_TILE_D, BK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, metal::execution_simdgroups<QSA_TILE_SG>> qk_op;
    matmul2d<pv_desc, metal::execution_simdgroups<QSA_TILE_SG>> pv_op;

    const array<int, 2> q_strides{1, int(NQ * uint(QSA_TILE_D))};
    auto tQ = tensor(q + ((ulong)q0 * NQ + h) * QSA_TILE_D,
                     dextents<int32_t, 2>(QSA_TILE_D, int(qn)), q_strides);
    using QT = decltype(tQ);
    using PT = decltype(tP);
    using KVT = decltype(tKV);
    auto acc = pv_op.template get_destination_cooperative_tensor<PT, KVT, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0.0f;
        }
    }

    // Which cache rows slice `k0` holds and who attends them, into buffer `buf`.
    auto describe = [&](uint k0, uint buf) {
        if (tid < uint(BK)) {
            const uint j = k0 + tid;
            uint token = 0u;
            uint m = 0u;
            if (j < in_blocks) {
                const uint u = j / ratio;
                token = ublk[u] * ratio + (j - u * ratio);
                m = umask[u];
            } else if (j < total) {
                token = tail_start + (j - in_blocks);
                const uint tb = token / ratio - vb0;
                const uint tbits = tb < QSA_TILE_TAIL_BLOCKS ? tmask[tb] : 0u;
                for (uint i = 0; i < qn; ++i) {
                    const uint p = p0 + i;
                    const bool causal = token <= p;
                    const bool own_tail = token >= qsa_visible_blocks(p, ratio) * ratio;
                    if (causal && (own_tail || ((tbits >> i) & 1u))) {
                        m |= 1u << i;
                    }
                }
            }
            tok_idx[buf * uint(BK) + tid] = token;
            tok_mask[buf * uint(BK) + tid] = m;
        }
    };
    // This thread's words of the slice's rows from `head`, into registers.
    auto fetch = [&](device const uint4* head, uint k0, uint buf, thread uint4* regs) {
        const uint count = min(uint(BK), total - k0);
        for (uint t = 0; t < PER_THREAD; ++t) {
            const uint i = tid + t * THREADS;
            const uint r = i / ROW4;
            const uint c = i - r * ROW4;
            regs[t] = r < count ? head[(ulong)tok_idx[buf * uint(BK) + r] * ROW4 + c] : uint4(0u);
        }
    };
    auto stage = [&](thread const uint4* regs) {
        for (uint t = 0; t < PER_THREAD; ++t) {
            kv4[tid + t * THREADS] = regs[t];
        }
    };

    uint4 regs[PER_THREAD];
    uint4 kregs[EARLY ? PER_THREAD : 1];
    describe(0u, 0u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fetch(k_head, 0u, 0u, regs);
    stage(regs);
    if (EARLY && uint(BK) < total) {
        describe(uint(BK), 1u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k0 = 0; k0 < total; k0 += uint(BK)) {
        const uint buf = (k0 / uint(BK)) & 1u;
        const uint count = min(uint(BK), total - k0);
        const uint next = k0 + uint(BK);
        // kv4 holds this slice's K rows; its V rows go to registers now (and
        // with EARLY the next slice's K rows too).
        fetch(v_head, k0, buf, regs);
        if (EARLY && next < total) {
            fetch(k_head, next, buf ^ 1u, kregs);
        }

        auto sT = qk_op.template get_destination_cooperative_tensor<QT, KVT, float>();
        qk_op.run(tQ, tKV, sT);
        sT.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Masked online softmax: LANES lanes per query row.
        {
            const uint row = tid / LANES;
            const uint l = tid - row * LANES;
            const uint j0 = l * COLS;
            const uint prev_stat = row;
            const float prev = row_max[prev_stat];
            float local = prev;
            for (uint j = j0; j < j0 + COLS; ++j) {
                if (j < count && ((tok_mask[buf * uint(BK) + j] >> row) & 1u)) {
                    local = max(local, s_tile[row * uint(BK) + j] * scale);
                }
            }
            for (uint off = 1u; off < LANES; off <<= 1) {
                local = max(local, simd_shuffle_xor(local, off));
            }
            const float mx = local;
            const float alpha = mx == -INFINITY ? 1.0f : exp(prev - mx);
            float psum = 0.0f;
            for (uint j = j0; j < j0 + COLS; ++j) {
                const bool on = j < count && ((tok_mask[buf * uint(BK) + j] >> row) & 1u);
                const float p = on ? exp(s_tile[row * uint(BK) + j] * scale - mx) : 0.0f;
                p_tile[row * uint(BK) + j] = bfloat(p);
                psum += p;
            }
            for (uint off = 1u; off < LANES; off <<= 1) {
                psum += simd_shuffle_xor(psum, off);
            }
            if (l == 0u) {
                row_sum[prev_stat] = row_sum[prev_stat] * alpha + psum;
                row_max[prev_stat] = mx;
                row_alpha[prev_stat] = alpha;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // V rows over K; without EARLY the next slice's description
        // alongside, then its K rows fetched under the P.V.
        stage(regs);
        if (!EARLY && next < total) {
            describe(next, buf ^ 1u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!EARLY && next < total) {
            fetch(k_head, next, buf ^ 1u, regs);
        }

        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                auto ix = acc.get_multidimensional_index(i);
                acc[i] *= row_alpha[uint(ix[1])];
            }
        }
        pv_op.run(tP, tKV, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (next < total) {
            if (EARLY) {
                stage(kregs);
                // The slice after next, into the buffer this slice used.
                if (next + uint(BK) < total) {
                    describe(next + uint(BK), buf);
                }
            } else {
                stage(regs);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            const uint row = uint(ix[1]);
            const uint col = uint(ix[0]);
            if (row < qn) {
                out[((ulong)(q0 + row) * NQ + h) * QSA_TILE_D + col] =
                    bfloat(acc[i] / row_sum[row]);
            }
        }
    }
}

#define QSA_TILE_PIPE_KERNEL(NAME, BQ, BK, EARLY)                                     \
kernel void NAME(device bfloat*       q          [[buffer(0)]],                       \
                 device bfloat*       k_cache    [[buffer(1)]],                        \
                 device bfloat*       v_cache    [[buffer(2)]],                        \
                 device const uint*   union_blk  [[buffer(3)]],                        \
                 device const uint*   union_mask [[buffer(4)]],                        \
                 device const uint*   n_union    [[buffer(5)]],                        \
                 device const uint*   tail_mask  [[buffer(6)]],                        \
                 device bfloat*       out        [[buffer(7)]],                        \
                 constant uint&       max_seq    [[buffer(8)]],                        \
                 constant uint&       base_pos   [[buffer(9)]],                        \
                 constant uint&       M          [[buffer(10)]],                       \
                 constant uint&       NQ         [[buffer(11)]],                       \
                 constant uint&       group      [[buffer(12)]],                       \
                 constant float&      scale      [[buffer(13)]],                       \
                 constant uint&       cap        [[buffer(14)]],                       \
                 constant uint&       ratio      [[buffer(15)]],                       \
                 uint2 tg      [[threadgroup_position_in_grid]],                       \
                 uint  tid     [[thread_index_in_threadgroup]],                        \
                 uint2 tg_size [[threads_per_threadgroup]]) {                          \
    threadgroup uint4  kv4[(BK) * QSA_TILE_D / 8];                                     \
    threadgroup float  s_tile[(BQ) * (BK)];                                            \
    threadgroup bfloat p_tile[(BQ) * (BK)];                                            \
    threadgroup uint   tok_idx[2 * (BK)], tok_mask[2 * (BK)];                          \
    threadgroup float  row_max[(BQ)], row_sum[(BQ)], row_alpha[(BQ)];                  \
    qsa_attn_tile_pipe_body<(BQ), (BK), (EARLY)>(                                      \
        q, k_cache, v_cache, union_blk, union_mask, n_union, tail_mask, out,          \
        max_seq, base_pos, M, NQ, group, scale, cap, ratio, kv4, s_tile, p_tile,      \
        tok_idx, tok_mask, row_max, row_sum, row_alpha, tg, tid, tg_size);            \
}

#define QSA_TILE_KERNEL(NAME, BQ, BK, HPP)                                            \
kernel void NAME(device bfloat*       q          [[buffer(0)]],  /* [M, NQ, D] */     \
                 device bfloat*       k_cache    [[buffer(1)]],  /* [KVH, max_seq, D] */ \
                 device bfloat*       v_cache    [[buffer(2)]],                        \
                 device const uint*   union_blk  [[buffer(3)]],  /* [tiles, cap] */    \
                 device const uint*   union_mask [[buffer(4)]],  /* [tiles, cap] */    \
                 device const uint*   n_union    [[buffer(5)]],  /* [tiles] */         \
                 device const uint*   tail_mask  [[buffer(6)]],  /* [tiles, 8] */      \
                 device bfloat*       out        [[buffer(7)]],  /* [M, NQ, D] */      \
                 constant uint&       max_seq    [[buffer(8)]],                        \
                 constant uint&       base_pos   [[buffer(9)]],                        \
                 constant uint&       M          [[buffer(10)]],                       \
                 constant uint&       NQ         [[buffer(11)]],                       \
                 constant uint&       group      [[buffer(12)]],                       \
                 constant float&      scale      [[buffer(13)]],                       \
                 constant uint&       cap        [[buffer(14)]],                       \
                 constant uint&       ratio      [[buffer(15)]],                       \
                 uint2 tg      [[threadgroup_position_in_grid]],                       \
                 uint  tid     [[thread_index_in_threadgroup]],                        \
                 uint2 tg_size [[threads_per_threadgroup]]) {                          \
    threadgroup uint4  kv4[(BK) * QSA_TILE_D / 8];                                     \
    threadgroup float  s_tile[(BQ) * (BK)];                                            \
    threadgroup bfloat p_tile[(HPP) * (BQ) * (BK)];                                    \
    threadgroup uint   tok_idx[(BK)], tok_mask[(BK)];                                  \
    threadgroup float  row_max[(HPP) * (BQ)], row_sum[(HPP) * (BQ)], row_alpha[(HPP) * (BQ)]; \
    qsa_attn_tile_body<(BQ), (BK), (HPP)>(                                             \
        q, k_cache, v_cache, union_blk, union_mask, n_union, tail_mask, out,          \
        max_seq, base_pos, M, NQ, group, scale, cap, ratio, kv4, s_tile, p_tile,      \
        tok_idx, tok_mask, row_max, row_sum, row_alpha, tg, tid, tg_size);            \
}

// Threadgroup memory: 16 KB of K/V rows (BK = 32 x 512 B) plus BQ x BK score
// and probability tiles per head, within the 32 KB budget up to four heads.
// The one-head kernel is the pipelined body (qsa_attn_tile_pipe_body); the
// multi-head ones share the staged rows across heads instead. Measured per
// dispatch on the profile transport (256 queries, 24 heads, 1.8x overlap):
// pipelined 3.9 ms against 4.6 to 4.9 for the plain one-head body at 8K and
// 7.8 against 8.5 to 10.6 at 32K; the two-head body 5.1 / 10.4, staging 16
// rows per slice instead of 32 (for occupancy) 6.1 / 12.0, and prefetching
// the next slice's K rows a whole slice earlier no better than the pipeline.
QSA_TILE_PIPE_KERNEL(qsa_attn_tile_nax_h1, QSA_TILE_BQ, 32, false)
QSA_TILE_KERNEL(qsa_attn_tile_nax_h2, QSA_TILE_BQ, 32, 2)
QSA_TILE_KERNEL(qsa_attn_tile_nax_h4, QSA_TILE_BQ, 32, 4)
// The plain one-head body, for comparison by name (`LILY_QSA_TILE_KERNEL`).
QSA_TILE_KERNEL(qsa_attn_tile_nax_h1_plain, QSA_TILE_BQ, 32, 1)

#endif
