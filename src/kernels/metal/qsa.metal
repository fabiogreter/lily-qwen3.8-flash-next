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

// --- Tiled sparse attention (Metal 4 tensor ops) ----------------------------------
//
// Prefill past the dense limit. Adjacent queries select largely the same
// blocks, so a tile of QSA_TILE_BQ consecutive queries attends the union of
// its selections through the tensor-op path, each query masked to its own
// selection. `qsa_tile_union` builds the union per tile; `qsa_tile_gather`
// copies its K/V rows into a contiguous device scratch once per KV head, and
// `qsa_attn_rows_nax_h1` runs the same flash loop as the dense prefill
// kernel over them.

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

// --- Gathered-row tiled attention -----------------------------------------------
//
// The earlier tile kernel staged the union's K/V rows into threadgroup
// memory 32 at a time and paid a chain of gathered fetches and two dependent
// tensor ops per slice, hidden only by the threadgroups a core holds at 20 KB
// each (by parts at 8K: 3.9 ms per 256-query dispatch, 2.1 with no QK op,
// 2.1 with no fetch, 0.34 with no ops). Here a tile's union rows are gathered
// once per KV head into a contiguous device scratch (`qsa_tile_gather`), and
// the attention kernel runs the dense kernel's loop over them: 128-row
// slices read by the tensor ops straight from device memory, 12 KB of
// threadgroup memory, one query head per threadgroup with a tile's heads
// adjacent in the grid so they share the gathered rows in cache. In the
// harness (256 queries, 24 heads): 2.0 + 0.6 ms against 3.9 at 8K and
// 3.6 + 1.2 against 7.6 at 32K; the staged kernel was dropped.

#define QSA_ROWS_BK 128    // keys per slice of the gathered-row kernel
#define QSA_ROWS_TG 256    // threads of the gather kernel
#define QSA_ROWS_SPLIT 8   // threadgroups per (tile, KV head) of the gather kernel

// Gathers tile `tile0 + tg.y`'s union rows for KV head `tg.x` into
// `rows_k`/`rows_v` (`[tiles, KVH, slot, D]`), the query mask of every row
// into `row_mask` (`[tiles, slot]`, KV head 0 only) and the row count into
// `n_rows`; rows past the count up to a slice multiple are zeroed so the
// tensor ops' over-reads stay finite. The rows are strided over
// QSA_ROWS_SPLIT threadgroups (`tg.z`) and their simdgroups, each moving a
// row per iteration with the next one's loads already in flight: the copy
// is latency-bound per simdgroup, not bandwidth-bound.
kernel void qsa_tile_gather(device const uint4*  k_cache    [[buffer(0)]],
                            device const uint4*  v_cache    [[buffer(1)]],
                            device const uint*   union_blk  [[buffer(2)]],
                            device const uint*   union_mask [[buffer(3)]],
                            device const uint*   n_union    [[buffer(4)]],
                            device const uint*   tail_mask  [[buffer(5)]],
                            device uint4*        rows_k     [[buffer(6)]],
                            device uint4*        rows_v     [[buffer(7)]],
                            device uint*         row_mask   [[buffer(8)]],
                            device uint*         n_rows     [[buffer(9)]],
                            constant uint&       max_seq    [[buffer(10)]],
                            constant uint&       base_pos   [[buffer(11)]],
                            constant uint&       M          [[buffer(12)]],
                            constant uint&       cap        [[buffer(13)]],
                            constant uint&       ratio      [[buffer(14)]],
                            constant uint&       slot       [[buffer(15)]],
                            constant uint&       tile0      [[buffer(16)]],
                            constant uint&       kvh        [[buffer(17)]],
                            uint3 tg   [[threadgroup_position_in_grid]],
                            uint  tid  [[thread_index_in_threadgroup]],
                            uint  sg   [[simdgroup_index_in_threadgroup]],
                            uint  lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROW4 = uint(QSA_TILE_D) * 2u / 16u;  // uint4 words per row (32)
    constexpr uint SGS = QSA_ROWS_TG / 32u;
    constexpr uint STRIDE = SGS * uint(QSA_ROWS_SPLIT);
    const uint kh = tg.x;
    const uint local = tg.y;
    const uint tile = tile0 + local;
    const uint q0 = tile * uint(QSA_TILE_BQ);
    if (q0 >= M) {
        return;
    }
    const uint qn = min(uint(QSA_TILE_BQ), M - q0);
    const uint p0 = base_pos + q0;
    const uint vb0 = qsa_visible_blocks(p0, ratio);
    const uint n_u = min(n_union[tile], cap);
    const uint in_blocks = n_u * ratio;
    const uint tail_start = vb0 * ratio;
    const uint total = in_blocks + (p0 + qn - tail_start);
    const uint padded = min((total + uint(QSA_ROWS_BK) - 1u) / uint(QSA_ROWS_BK) * uint(QSA_ROWS_BK), slot);
    device const uint* ublk = union_blk + (ulong)tile * cap;
    device const uint* umask = union_mask + (ulong)tile * cap;
    device const uint* tmask = tail_mask + (ulong)tile * QSA_TILE_TAIL_BLOCKS;
    device const uint4* k_head = k_cache + (ulong)kh * max_seq * ROW4;
    device const uint4* v_head = v_cache + (ulong)kh * max_seq * ROW4;
    const ulong dst0 = ((ulong)local * kvh + kh) * slot;
    device uint4* dk = rows_k + dst0 * ROW4;
    device uint4* dv = rows_v + dst0 * ROW4;
    device uint* dm = row_mask + (ulong)local * slot;
    if (tg.z == 0u && tid == 0u) {
        n_rows[local] = total;
    }
    // The cache row and query mask of gathered row j.
    auto describe = [&](uint j, thread uint& token, thread uint& m) {
        token = 0u;
        m = 0u;
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
    };
    uint j = tg.z * SGS + sg;
    uint token, m;
    describe(j, token, m);
    uint4 kw = j < total ? k_head[(ulong)token * ROW4 + lane] : uint4(0u);
    uint4 vw = j < total ? v_head[(ulong)token * ROW4 + lane] : uint4(0u);
    while (j < padded) {
        const uint jn = j + STRIDE;
        uint token_n = 0u, m_n = 0u;
        uint4 kn = uint4(0u), vn = uint4(0u);
        if (jn < padded) {
            describe(jn, token_n, m_n);
            if (jn < total) {
                kn = k_head[(ulong)token_n * ROW4 + lane];
                vn = v_head[(ulong)token_n * ROW4 + lane];
            }
        }
        dk[(ulong)j * ROW4 + lane] = kw;
        dv[(ulong)j * ROW4 + lane] = vw;
        if (kh == 0u && lane == 0u) {
            dm[j] = m;
        }
        j = jn;
        m = m_n;
        kw = kn;
        vw = vn;
    }
}

// Flash attention of one tile of BQ queries and one query head over the
// tile's gathered rows. Grid: threadgroups (NQ, tiles in the group),
// 32 * QSA_TILE_SG threads.
template <int BQ, int BK>
static void qsa_attn_rows_body(device bfloat* q,
                               device bfloat* rows_k,
                               device bfloat* rows_v,
                               device const uint* row_mask,
                               device const uint* n_rows,
                               device bfloat* out,
                               uint M,
                               uint NQ,
                               uint group,
                               float scale,
                               uint slot,
                               uint kvh,
                               uint tile0,
                               threadgroup float* s_tile,
                               threadgroup bfloat* p_tile,
                               threadgroup float* row_max,
                               threadgroup float* row_sum,
                               threadgroup float* row_alpha,
                               uint2 tg,
                               uint tid,
                               uint2 tg_size) {
    constexpr uint THREADS = 32u * uint(QSA_TILE_SG);
    constexpr uint LANES = THREADS / uint(BQ);
    constexpr uint COLS = uint(BK) / LANES;
    static_assert(uint(BQ) * LANES == THREADS,
                  "the threadgroup must divide into BQ equal row groups");
    static_assert(uint(BK) % LANES == 0u, "each lane must own a whole number of key columns");
    static_assert(LANES <= 32u, "a row's lanes must sit inside one simdgroup");
    static_assert(BQ <= 32, "the query mask is one uint");

    using namespace mpp::tensor_ops;

    if (tg_size.x != THREADS) {
        return;
    }
    const uint h = tg.x;
    const uint local = tg.y;
    const uint tile = tile0 + local;
    const uint q0 = tile * uint(BQ);
    if (q0 >= M) {
        return;
    }
    const uint qn = min(uint(BQ), M - q0);
    const uint kh = h / group;
    const uint total = n_rows[local];
    const ulong src0 = ((ulong)local * kvh + kh) * slot;
    device const uint* mask = row_mask + (ulong)local * slot;

    if (tid < uint(BQ)) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
        row_alpha[tid] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const array<int, 2> q_strides{1, int(NQ * uint(QSA_TILE_D))};
    auto tQ = tensor(q + ((ulong)q0 * NQ + h) * QSA_TILE_D,
                     dextents<int32_t, 2>(QSA_TILE_D, int(qn)), q_strides);
    auto tK = tensor(rows_k + src0 * QSA_TILE_D, dextents<int32_t, 2>(QSA_TILE_D, int(total)));
    auto tV = tensor(rows_v + src0 * QSA_TILE_D, dextents<int32_t, 2>(QSA_TILE_D, int(total)));
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
    using QT = decltype(tQ);
    using PT = decltype(tP);
    using KSlice = decltype(tK.slice(0, 0));
    using VSlice = decltype(tV.slice(0, 0));
    auto acc = pv_op.template get_destination_cooperative_tensor<PT, VSlice, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0.0f;
        }
    }

    const uint row = tid / LANES;
    const uint l = tid - row * LANES;
    const uint j0 = l * COLS;
    for (uint k0 = 0; k0 < total; k0 += uint(BK)) {
        const uint count = min(uint(BK), total - k0);
        auto kS = tK.slice(0, int(k0));
        auto vS = tV.slice(0, int(k0));
        auto sT = qk_op.template get_destination_cooperative_tensor<QT, KSlice, float>();
        qk_op.run(tQ, kS, sT);
        sT.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Masked online softmax: LANES lanes per query row.
        {
            const float prev = row_max[row];
            float local_max = prev;
            bool on[COLS];
            for (uint c = 0; c < COLS; ++c) {
                const uint j = j0 + c;
                on[c] = j < count && ((mask[k0 + j] >> row) & 1u);
                if (on[c]) {
                    local_max = max(local_max, s_tile[row * uint(BK) + j] * scale);
                }
            }
            for (uint off = 1u; off < LANES; off <<= 1) {
                local_max = max(local_max, simd_shuffle_xor(local_max, off));
            }
            const float mx = local_max;
            const float alpha = mx == -INFINITY ? 1.0f : exp(prev - mx);
            float psum = 0.0f;
            for (uint c = 0; c < COLS; ++c) {
                const uint j = j0 + c;
                const float p = on[c] ? exp(s_tile[row * uint(BK) + j] * scale - mx) : 0.0f;
                p_tile[row * uint(BK) + j] = bfloat(p);
                psum += p;
            }
            for (uint off = 1u; off < LANES; off <<= 1) {
                psum += simd_shuffle_xor(psum, off);
            }
            if (l == 0u) {
                row_sum[row] = row_sum[row] * alpha + psum;
                row_max[row] = mx;
                row_alpha[row] = alpha;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                auto ix = acc.get_multidimensional_index(i);
                acc[i] *= row_alpha[uint(ix[1])];
            }
        }
        pv_op.run(tP, vS, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            const uint r = uint(ix[1]);
            const uint col = uint(ix[0]);
            if (r < qn) {
                out[((ulong)(q0 + r) * NQ + h) * QSA_TILE_D + col] =
                    bfloat(acc[i] / row_sum[r]);
            }
        }
    }
}

#define QSA_ROWS_KERNEL(NAME, BQ, BK)                                                 \
kernel void NAME(device bfloat*       q          [[buffer(0)]],                       \
                 device bfloat*       rows_k     [[buffer(1)]],                        \
                 device bfloat*       rows_v     [[buffer(2)]],                        \
                 device const uint*   row_mask   [[buffer(3)]],                        \
                 device const uint*   n_rows     [[buffer(4)]],                        \
                 device bfloat*       out        [[buffer(5)]],                        \
                 constant uint&       M          [[buffer(6)]],                        \
                 constant uint&       NQ         [[buffer(7)]],                        \
                 constant uint&       group      [[buffer(8)]],                        \
                 constant float&      scale      [[buffer(9)]],                        \
                 constant uint&       slot       [[buffer(10)]],                       \
                 constant uint&       kvh        [[buffer(11)]],                       \
                 constant uint&       tile0      [[buffer(12)]],                       \
                 uint2 tg      [[threadgroup_position_in_grid]],                       \
                 uint  tid     [[thread_index_in_threadgroup]],                        \
                 uint2 tg_size [[threads_per_threadgroup]]) {                          \
    threadgroup float  s_tile[(BQ) * (BK)];                                            \
    threadgroup bfloat p_tile[(BQ) * (BK)];                                            \
    threadgroup float  row_max[(BQ)], row_sum[(BQ)], row_alpha[(BQ)];                  \
    qsa_attn_rows_body<(BQ), (BK)>(                                                    \
        q, rows_k, rows_v, row_mask, n_rows, out, M, NQ, group, scale, slot, kvh,     \
        tile0, s_tile, p_tile, row_max, row_sum, row_alpha, tg, tid, tg_size);        \
}

QSA_ROWS_KERNEL(qsa_attn_rows_nax_h1, QSA_TILE_BQ, QSA_ROWS_BK)

#endif
