// Unit-test-only pair-major and extended-row small-M variants.
#define MOE_GEMV_SMALLM_BODY(MAXR, R4)                                        \
    threadgroup uint tg_pairs[MOE_SMALLM_MAX_S];                              \
    for (uint jj = lane; jj < S; jj += 64) {                                  \
        tg_pairs[jj] = indices[jj];                                           \
    }                                                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    const uint j = tg.y;                                                      \
    const uint e = tg_pairs[j];                                               \
    for (uint jj = 0; jj < j; ++jj) {                                         \
        if (tg_pairs[jj] == e) {                                              \
            return;                                                           \
        }                                                                     \
    }                                                                         \
    uint pair[MAXR];                                                          \
    uint xbase[MAXR];                                                         \
    uint nr = 0;                                                              \
    for (uint jj = j; jj < S; ++jj) {                                         \
        if (tg_pairs[jj] != e || nr >= MAXR) {                                \
            continue;                                                         \
        }                                                                     \
        const uint xr = (XPP != 0) ? jj : jj / TOPK;                          \
        _Pragma("clang loop unroll(full)")                                    \
        for (uint ri = 0; ri < MAXR; ++ri) {                                  \
            if (ri == nr) {                                                   \
                pair[ri] = jj;                                                \
                xbase[ri] = xr * (K / 4);                                     \
            }                                                                 \
        }                                                                     \
        ++nr;                                                                 \
    }                                                                         \
    MOE_GEMV_SMALLM_CORE(MAXR, R4)

#define MOE_GEMV_SMALLM_KERNEL(NAME, MAXR, R4)                                \
    kernel void NAME(device const uint*   codes   [[buffer(0)]],              \
                     device const bfloat* scales  [[buffer(1)]],              \
                     device const bfloat* biases  [[buffer(2)]],              \
                     device const bfloat* x       [[buffer(3)]],              \
                     device const uint*   indices [[buffer(4)]],              \
                     device bfloat*       y       [[buffer(5)]],              \
                     constant uint&       K       [[buffer(6)]],              \
                     constant uint&       GS      [[buffer(7)]],              \
                     constant uint&       N       [[buffer(8)]],              \
                     constant uint&       S       [[buffer(9)]],              \
                     constant uint&       TOPK    [[buffer(10)]],             \
                     constant uint&       XPP     [[buffer(11)]],             \
                     uint2 tg  [[threadgroup_position_in_grid]],              \
                     uint lane [[thread_index_in_threadgroup]]) {             \
        MOE_GEMV_SMALLM_BODY(MAXR, R4)                                        \
    }

MOE_GEMV_SMALLM_KERNEL(moe_gemv_smallm_q4_r8, 8, 4)
MOE_GEMV_SMALLM_KERNEL(moe_gemv_smallm_q4_r16, 16, 2)
MOE_GEMV_SMALLM_KERNEL(moe_gemv_smallm_q4_r8_w, 8, 8)
MOE_GEMV_SMALLM_KERNEL(moe_gemv_smallm_q4_r16_w, 16, 4)
MOE_GEMV_SMALLM_KERNEL(moe_gemv_smallm_q4_r8_n2, 8, 1)
MOE_GEMV_SMALLM_KERNEL(moe_gemv_smallm_q4_r16_n2, 16, 1)

MOE_GEMV_SMALLM_EM_KERNEL(moe_gemv_smallm_q4_em_r16, 16, 2)
MOE_GEMV_SMALLM_EM_KERNEL(moe_gemv_smallm_q4_em_r16_w, 16, 4)
MOE_GEMV_SMALLM_EM_KERNEL(moe_gemv_smallm_q4_em_r8_n2, 8, 1)
MOE_GEMV_SMALLM_EM_KERNEL(moe_gemv_smallm_q4_em_r16_n2, 16, 1)

// Reference router top-k kernels: the repeated-argmax versions the register
// selection replaced (one 256-thread threadgroup, two barriers per round).
// Kept verbatim so the production kernels can be asserted bit-identical.
// Routes logits [E] to K experts; ties select the lowest expert id.
// Scores use full softmax unless renorm restricts it to selected logits.
kernel void moe_router_topk_ref(device const float* logits  [[buffer(0)]],
                            device uint*        indices [[buffer(1)]],
                            device float*       scores  [[buffer(2)]],
                            constant uint&      E       [[buffer(3)]],
                            constant uint&      K       [[buffer(4)]],
                            constant uint&      renorm  [[buffer(5)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint tg_size [[threads_per_threadgroup]],
                            uint sg   [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float probs[MOE_MAX_E];
    threadgroup float red[32];
    threadgroup float sel_val[MOE_MAX_K];
    threadgroup uint  sel_idx[MOE_MAX_K];

    // Full softmax cancels when selected logits are renormalized.
    if (renorm != 0) {
        for (uint i = tid; i < E; i += tg_size) {
            probs[i] = logits[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {

    float local_max = -INFINITY;
    for (uint i = tid; i < E; i += tg_size) {
        local_max = max(local_max, logits[i]);
    }
    local_max = simd_max(local_max);
    if (lane == 0) {
        red[sg] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (uint i = 0; i < (tg_size + 31) / 32; ++i) {
            m = max(m, red[i]);
        }
        red[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gmax = red[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < E; i += tg_size) {
        float e = exp(logits[i] - gmax);
        probs[i] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        red[sg] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < (tg_size + 31) / 32; ++i) {
            s += red[i];
        }
        red[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_sum = 1.0f / red[0];
    for (uint i = tid; i < E; i += tg_size) {
        probs[i] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Repeated argmax masks each selected expert before the next round.
    threadgroup float win_val[32];
    threadgroup uint  win_idx[32];
    for (uint round = 0; round < K; ++round) {
        float best = -INFINITY;
        uint best_i = 0xFFFFFFFF;
        for (uint i = tid; i < E; i += tg_size) {
            float p = probs[i];
            if (p > best || (p == best && i < best_i)) {
                best = p;
                best_i = i;
            }
        }
        for (uint off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(best, off);
            uint oi = simd_shuffle_down(best_i, off);
            if (ov > best || (ov == best && oi < best_i)) {
                best = ov;
                best_i = oi;
            }
        }
        if (lane == 0) {
            win_val[sg] = best;
            win_idx[sg] = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float b = -INFINITY;
            uint bi = 0xFFFFFFFF;
            for (uint i = 0; i < (tg_size + 31) / 32; ++i) {
                if (win_val[i] > b || (win_val[i] == b && win_idx[i] < bi)) {
                    b = win_val[i];
                    bi = win_idx[i];
                }
            }
            sel_val[round] = b;
            sel_idx[round] = bi;
            // NaN cannot collide with a valid -inf logit on the raw path.
            probs[bi] = NAN;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        if (renorm != 0) {
            float mx = sel_val[0];
            float denom = 0.0f;
            for (uint j = 0; j < K; ++j) {
                denom += exp(sel_val[j] - mx);
            }
            for (uint j = 0; j < K; ++j) {
                indices[j] = sel_idx[j];
                scores[j] = exp(sel_val[j] - mx) / denom;
            }
        } else {
            for (uint j = 0; j < K; ++j) {
                indices[j] = sel_idx[j];
                scores[j] = sel_val[j];
            }
        }
    }
}

// Routes each BF16 logits row [E] to K experts.
kernel void moe_router_topk_rows_ref(device const bfloat* logits  [[buffer(0)]],
                                 device uint*         indices [[buffer(1)]],
                                 device float*        scores  [[buffer(2)]],
                                 constant uint&       E       [[buffer(3)]],
                                 constant uint&       K       [[buffer(4)]],
                                 constant uint&       renorm  [[buffer(5)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint tg_size [[threads_per_threadgroup]],
                                 uint sg   [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float probs[MOE_MAX_E];
    threadgroup float red[32];
    threadgroup float sel_val[MOE_MAX_K];
    threadgroup uint  sel_idx[MOE_MAX_K];
    device const bfloat* row_logits = logits + (ulong)row * E;

    if (renorm != 0) {
        for (uint i = tid; i < E; i += tg_size) {
            probs[i] = float(row_logits[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {

    float local_max = -INFINITY;
    for (uint i = tid; i < E; i += tg_size) {
        local_max = max(local_max, float(row_logits[i]));
    }
    local_max = simd_max(local_max);
    if (lane == 0) {
        red[sg] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (uint i = 0; i < (tg_size + 31) / 32; ++i) {
            m = max(m, red[i]);
        }
        red[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gmax = red[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < E; i += tg_size) {
        float e = exp(float(row_logits[i]) - gmax);
        probs[i] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        red[sg] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < (tg_size + 31) / 32; ++i) {
            s += red[i];
        }
        red[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_sum = 1.0f / red[0];
    for (uint i = tid; i < E; i += tg_size) {
        probs[i] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float win_val[32];
    threadgroup uint  win_idx[32];
    for (uint round = 0; round < K; ++round) {
        float best = -INFINITY;
        uint best_i = 0xFFFFFFFF;
        for (uint i = tid; i < E; i += tg_size) {
            float p = probs[i];
            if (p > best || (p == best && i < best_i)) {
                best = p;
                best_i = i;
            }
        }
        for (uint off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(best, off);
            uint oi = simd_shuffle_down(best_i, off);
            if (ov > best || (ov == best && oi < best_i)) {
                best = ov;
                best_i = oi;
            }
        }
        if (lane == 0) {
            win_val[sg] = best;
            win_idx[sg] = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float b = -INFINITY;
            uint bi = 0xFFFFFFFF;
            for (uint i = 0; i < (tg_size + 31) / 32; ++i) {
                if (win_val[i] > b || (win_val[i] == b && win_idx[i] < bi)) {
                    b = win_val[i];
                    bi = win_idx[i];
                }
            }
            sel_val[round] = b;
            sel_idx[round] = bi;
            probs[bi] = NAN;  // collision-free selection mask
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        if (renorm != 0) {
            float mx = sel_val[0];
            float denom = 0.0f;
            for (uint j = 0; j < K; ++j) {
                denom += exp(sel_val[j] - mx);
            }
            for (uint j = 0; j < K; ++j) {
                indices[(ulong)row * K + j] = sel_idx[j];
                scores[(ulong)row * K + j] = exp(sel_val[j] - mx) / denom;
            }
        } else {
            for (uint j = 0; j < K; ++j) {
                indices[(ulong)row * K + j] = sel_idx[j];
                scores[(ulong)row * K + j] = sel_val[j];
            }
        }
    }
}
