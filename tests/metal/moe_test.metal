// Test-only expert-major small-M GEMV chain: the union map kernel and the
// expert-major GEMVs the fused small-m kernels replaced (the reference
// the fused kernels are asserted bit-identical against).
constant uint MOE_SMALLM_MAX_ROWS = 16;

// Reads expert ids and routed rows from the union map.
#define MOE_GEMV_SMALLM_EM_PROLOGUE(MAXR)                                      \
    const uint u = tg.y;                                                       \
    if (u >= min(umap[0], S)) {                                                \
        return;                                                                \
    }                                                                          \
    const uint e = umap[1 + u];                                                \
    const uint nr = min(umap[1 + MOE_SMALLM_MAX_S + u], (uint)MAXR);           \
    device const uint* upairs =                                                \
        umap + 1 + 2 * MOE_SMALLM_MAX_S + u * MOE_SMALLM_MAX_ROWS;             \
    uint pair[MAXR];                                                           \
    uint xbase[MAXR];                                                          \
    _Pragma("clang loop unroll(full)")                                         \
    for (uint ri = 0; ri < MAXR; ++ri) {                                       \
        if (ri < nr) {                                                         \
            const uint jj = upairs[ri];                                        \
            pair[ri] = jj;                                                     \
            xbase[ri] = ((XPP != 0) ? jj : jj / TOPK) * (K / 4);               \
        }                                                                      \
    }

#define MOE_GEMV_SMALLM_EM_BODY(MAXR, R4)                               \
    MOE_GEMV_SMALLM_EM_PROLOGUE(MAXR)                                          \
    MOE_GEMV_SMALLM_CORE(MAXR, R4)

#define MOE_GEMV_SMALLM_CORE(MAXR, R4)                                  \
    const uint sg = lane / 32;                                                 \
    const uint sl = lane % 32;                                                 \
    const uint row0 = (tg.x * 2 + sg) * R4;                                    \
    const uint words = K / 8;                                                  \
    const uint blocks = K / 16;                                                \
    const uint bpg = GS / 16;                                                  \
    const uint groups = K / GS;                                                \
    device const bfloat4* xw = (device const bfloat4*)x;                       \
    device const uint2* wrow[R4];                                              \
    _Pragma("clang loop unroll(full)")                                         \
    for (uint r4 = 0; r4 < R4; ++r4) {                                         \
        wrow[r4] =                                                             \
            (device const uint2*)(codes + ((ulong)e * N + row0 + r4) * words); \
    }                                                                          \
    float acc[MAXR][R4];                                                       \
    _Pragma("clang loop unroll(full)")                                         \
    for (uint ri = 0; ri < MAXR; ++ri) {                                       \
        _Pragma("clang loop unroll(full)")                                     \
        for (uint r4 = 0; r4 < R4; ++r4) {                                     \
            acc[ri][r4] = 0.0f;                                                \
        }                                                                      \
    }                                                                          \
    for (uint i = sl; i < blocks; i += 32) {                                   \
        const uint g = i / bpg;                                                \
        uint2 w2[R4];                                                          \
        float s[R4];                                                           \
        float b[R4];                                                           \
        _Pragma("clang loop unroll(full)")                                     \
        for (uint r4 = 0; r4 < R4; ++r4) {                                     \
            const ulong grow = (ulong)e * N + row0 + r4;                       \
            w2[r4] = wrow[r4][i];                                              \
            s[r4] = float(scales[grow * groups + g]);                          \
            b[r4] = float(biases[grow * groups + g]);                          \
        }                                                                      \
        _Pragma("clang loop unroll(full)")                                     \
        for (uint ri = 0; ri < MAXR; ++ri) {                                   \
            if (ri >= nr) {                                                    \
                break;                                                         \
            }                                                                  \
            device const bfloat4* xv = xw + xbase[ri] + 4 * i;                 \
            const float4 x0 = float4(xv[0]);                                   \
            const float4 x1 = float4(xv[1]);                                   \
            const float4 x2 = float4(xv[2]);                                   \
            const float4 x3 = float4(xv[3]);                                   \
            _Pragma("clang loop unroll(full)")                                 \
            for (uint r4 = 0; r4 < R4; ++r4) {                                 \
                float2 d = qdot_word_masked(w2[r4].x, x0, x1);                 \
                d += qdot_word_masked(w2[r4].y, x2, x3);                       \
                acc[ri][r4] += s[r4] * d.x + b[r4] * d.y;                      \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    _Pragma("clang loop unroll(full)")                                         \
    for (uint ri = 0; ri < MAXR; ++ri) {                                       \
        if (ri >= nr) {                                                        \
            break; /* nr is threadgroup-uniform, so reductions stay whole */   \
        }                                                                      \
        _Pragma("clang loop unroll(full)")                                     \
        for (uint r4 = 0; r4 < R4; ++r4) {                                     \
            const float r = simd_sum(acc[ri][r4]);                             \
            if (sl == 0) {                                                     \
                y[(ulong)pair[ri] * N + row0 + r4] = bfloat(r);                \
            }                                                                  \
        }                                                                      \
    }

// Builds an expert-major union map; S must not exceed MOE_SMALLM_MAX_S.
// One pair per thread avoids a macOS 27 AGX compiler crash.
kernel void moe_union_experts(device const uint* indices [[buffer(0)]],
                              device uint*       umap    [[buffer(1)]],
                              constant uint&     S       [[buffer(2)]],
                              uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint pairs[MOE_SMALLM_MAX_S];
    threadgroup uint first[MOE_SMALLM_MAX_S];
    threadgroup uint slot[MOE_SMALLM_MAX_S];
    if (tid < S) {
        pairs[tid] = indices[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < S) {
        uint f = 1;
        for (uint jj = 0; jj < tid; ++jj) {
            if (pairs[jj] == pairs[tid]) {
                f = 0;
                break;
            }
        }
        first[tid] = f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint u = 0;
        for (uint j = 0; j < S; ++j) {
            slot[j] = u;
            u += first[j];
        }
        umap[0] = u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < S && first[tid] != 0) {
        const uint u = slot[tid];
        const uint e = pairs[tid];
        uint nr = 0;
        for (uint jj = tid; jj < S && nr < MOE_SMALLM_MAX_ROWS; ++jj) {
            if (pairs[jj] == e) {
                umap[1 + 2 * MOE_SMALLM_MAX_S + u * MOE_SMALLM_MAX_ROWS + nr] = jj;
                ++nr;
            }
        }
        umap[1 + u] = e;
        umap[1 + MOE_SMALLM_MAX_S + u] = nr;
    }
}

#define MOE_GEMV_SMALLM_EM_KERNEL(NAME, MAXR, R4)                              \
    kernel void NAME(device const uint*   codes   [[buffer(0)]],               \
                     device const bfloat* scales  [[buffer(1)]],               \
                     device const bfloat* biases  [[buffer(2)]],               \
                     device const bfloat* x       [[buffer(3)]],               \
                     device const uint*   umap    [[buffer(4)]],               \
                     device bfloat*       y       [[buffer(5)]],               \
                     constant uint&       K       [[buffer(6)]],               \
                     constant uint&       GS      [[buffer(7)]],               \
                     constant uint&       N       [[buffer(8)]],               \
                     constant uint&       S       [[buffer(9)]],               \
                     constant uint&       TOPK    [[buffer(10)]],              \
                     constant uint&       XPP     [[buffer(11)]],              \
                     uint2 tg  [[threadgroup_position_in_grid]],               \
                     uint lane [[thread_index_in_threadgroup]]) {              \
        MOE_GEMV_SMALLM_EM_BODY(MAXR, R4)                                    \
    }


MOE_GEMV_SMALLM_EM_KERNEL(moe_gemv_smallm_q4_em_r8, 8, 4)
MOE_GEMV_SMALLM_EM_KERNEL(moe_gemv_smallm_q4_em_r8_w, 8, 8)

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

// Pair-parallelism variants of the fused down kernel (microbench only; the
// production instantiation uses MOE_SMALLM_DOWN_PSG).
MOE_SMALLM_DOWN_COMBINE_KERNEL(moe_smallm_q4_down_combine_p1, 4, 1)
MOE_SMALLM_DOWN_COMBINE_KERNEL(moe_smallm_q4_down_combine_p4, 4, 4)
