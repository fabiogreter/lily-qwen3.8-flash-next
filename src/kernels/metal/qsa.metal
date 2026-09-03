// Qwen Sparse Attention (QSA) for Qwen3.8-Flash-Next.
//
// A lightning indexer scores every completed block of `ratio` cached tokens
// against NH small query heads; the best `k_max` blocks plus the incomplete
// tail block form the token set a query attends. Block keys are the mean of
// the block's raw indexer keys, normed and roped at the block start.
#include <metal_stdlib>
using namespace metal;

#define TG 256
#define QSA_SPLIT 256
#define QSA_HPP 4          // query heads folded per K/V pass
#define QSA_SELECT_TG 1024 // threads of the per-query top-k selection

// --- Indexer projections --------------------------------------------------------

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
        const float inv_freq = pow(theta, -2.0f * float(tid) / float(rot));
        const float angle = float(base_pos + row) * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float lo = float(normed[tid]);
        const float hi = float(normed[half_rot + tid]);
        q[dst + tid] = bfloat(lo * c - hi * s);
        q[dst + half_rot + tid] = bfloat(hi * c + lo * s);
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
                                uint tg   [[threadgroup_position_in_grid]],
                                uint tid  [[thread_index_in_threadgroup]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;
    threadgroup bfloat normed[TG];

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
        const float inv_freq = pow(theta, -2.0f * float(tid) / float(rot));
        const float angle = float(b * ratio) * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float lo = float(normed[tid]);
        const float hi = float(normed[half_rot + tid]);
        blk[dst + tid] = bfloat(lo * c - hi * s);
        blk[dst + half_rot + tid] = bfloat(hi * c + lo * s);
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

// Exclusive prefix sum of one flag per thread over the whole threadgroup;
// returns this thread's rank and writes the total to `total`.
static inline uint qsa_scan(uint flag, threadgroup uint* sums, threadgroup uint* total,
                            uint tid, uint sg, uint lane) {
    const uint local = simd_prefix_exclusive_sum(flag);
    const uint sg_total = simd_sum(flag);
    if (lane == 0) {
        sums[sg] = sg_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0;
    for (uint s = 0; s < sg; ++s) {
        before += sums[s];
    }
    if (tid == 0) {
        uint t = 0;
        for (uint s = 0; s < QSA_SELECT_TG / 32; ++s) {
            t += sums[s];
        }
        *total = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return before + local;
}

// Selects the k_max highest-scoring visible blocks of each query (all of them
// when fewer are visible), written in ascending block order to sel[qi, :] with
// the count in n_sel[qi]. Ties at the k-th score resolve to the lowest block
// ids. One threadgroup of QSA_SELECT_TG threads per query.
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
    threadgroup atomic_uint hist[256];
    threadgroup uint sums[QSA_SELECT_TG / 32];
    threadgroup uint total;
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

    // Radix select (MSB first) for the k_max-th largest key.
    if (tid == 0) {
        prefix_s = 0;
        k_rem_s = k_max;
    }
    uint prefix = 0;
    uint mask = 0;
    for (uint pass = 0; pass < 4; ++pass) {
        const uint shift = 24 - 8 * pass;
        for (uint i = tid; i < 256; i += QSA_SELECT_TG) {
            atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = tid; b < nb; b += QSA_SELECT_TG) {
            const uint key = qsa_key(row[b]);
            if ((key & mask) == prefix) {
                atomic_fetch_add_explicit(&hist[(key >> shift) & 0xFFu], 1u,
                                          memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint k_rem = k_rem_s;
            uint cum = 0;
            uint chosen = 0;
            for (int bin = 255; bin >= 0; --bin) {
                const uint c = atomic_load_explicit(&hist[bin], memory_order_relaxed);
                if (cum + c >= k_rem) {
                    chosen = uint(bin);
                    k_rem -= cum;
                    break;
                }
                cum += c;
            }
            prefix_s = prefix | (chosen << shift);
            k_rem_s = k_rem;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = prefix_s;
        mask |= 0xFFu << shift;
    }
    const uint threshold = prefix;
    // Elements above the threshold are all taken; exactly k_rem ties fill up.
    const uint ties_to_take = k_rem_s;

    // Compaction in ascending block order.
    uint sel_base = 0;
    uint tie_base = 0;
    for (uint b0 = 0; b0 < nb; b0 += QSA_SELECT_TG) {
        const uint b = b0 + tid;
        const bool in_range = b < nb;
        const uint key = in_range ? qsa_key(row[b]) : 0u;
        const uint is_tie = (in_range && key == threshold) ? 1u : 0u;
        const uint tie_rank = qsa_scan(is_tie, sums, &total, tid, sg, lane);
        const uint tie_total = total;
        const uint take = (in_range && (key > threshold ||
                                        (is_tie && tie_base + tie_rank < ties_to_take)))
            ? 1u : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint rank = qsa_scan(take, sums, &total, tid, sg, lane);
        const uint take_total = total;
        if (take) {
            out[sel_base + rank] = b;
        }
        sel_base += take_total;
        tie_base += tie_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
// threadgroups (KVH, splits, QB), TG threads. Emits per-split softmax stats
// and weighted-V partials laid out like sdpa_decode_split (head index
// qi*NQ + hq), for sdpa_decode_combine. D must be 256.
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
                                uint3 tg  [[threadgroup_position_in_grid]],
                                uint tid  [[thread_index_in_threadgroup]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[QSA_HPP][QSA_SPLIT];
    threadgroup float part[QSA_HPP][TG / 32];
    threadgroup float red[QSA_HPP];
    threadgroup float v_stage[(TG / 32) * 256];

    const uint kh = tg.x;
    const uint split = tg.y;
    const uint qi = tg.z;
    const uint pos = base_pos + qi;
    const uint nblk = n_sel[qi];
    const uint tail_start = qsa_visible_blocks(pos, ratio) * ratio;
    const uint total = nblk * ratio + (pos + 1 - tail_start);
    const uint slot0 = split * QSA_SPLIT;
    const uint count = slot0 < total ? min(uint(QSA_SPLIT), total - slot0) : 0;
    device const uint* sel_row = sel + (ulong)qi * k_max;
    device const bfloat* k_head = k_cache + (ulong)kh * max_seq * D;
    device const bfloat* v_head = v_cache + (ulong)kh * max_seq * D;

    if (count == 0) {
        if (tid == 0) {
            for (uint h = 0; h < group; ++h) {
                const ulong hq = (ulong)qi * NQ + (ulong)kh * group + h;
                stats[(hq * splits + split) * 2] = -INFINITY;
                stats[(hq * splits + split) * 2 + 1] = 0.0f;
            }
        }
        return;
    }

    for (uint h0 = 0; h0 < group; h0 += QSA_HPP) {
        const uint gh = min(uint(QSA_HPP), group - h0);
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
                stats[(hq * splits + split) * 2] = chunk_max;
                stats[(hq * splits + split) * 2 + 1] = s;
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
                partials[(hq * splits + split) * D + d] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
