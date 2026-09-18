// Gated DeltaNet recurrence, causal depthwise conv1d, and gated RMSNorm.
// The gated delta step is adapted from MLX-LM; see NOTICE.
#include <metal_stdlib>
using namespace metal;

#define DIM 128

// Output-gate activation selector shared by the gated norms: 0 = SiLU
// (Qwen3.5), 1 = sigmoid (Qwen3.8-Flash-Next's `output_gate_type`).
#define GDN_GATE_SILU 0u
#define GDN_GATE_SIGMOID 1u

static inline float gate_activation(float gate, uint gate_act) {
    const float sig = 1.0f / (1.0f + exp(-gate));
    return gate_act == GDN_GATE_SIGMOID ? sig : gate * sig;
}

// One GDN decode step per value head; each thread owns one state column.
// Value head h uses key head h / vpk; state is FP32 [H, DIM, DIM].
template <typename StateT>
static inline float gdn_step_body(device const bfloat* q,
                                 device const bfloat* k,
                                 device const bfloat* v,
                                 device const bfloat* a,
                                 device const bfloat* b,
                                 device const float* a_log,
                                 device const bfloat* dt_bias,
                                 device StateT* state,
                                 float scale,
                                 uint vpk,
                                 uint h,
                                 uint tid,
                                 uint sg,
                                 uint lane,
                                 threadgroup float* q_norm,
                                 threadgroup float* k_norm,
                                 threadgroup float* part_q,
                                 threadgroup float* part_k,
                                 threadgroup float* gates) {

    const uint hk = h / vpk;
    float qv = float(q[hk * DIM + tid]);
    float kv = float(k[hk * DIM + tid]);
    float sq = simd_sum(qv * qv);
    float sk = simd_sum(kv * kv);
    if (lane == 0) {
        part_q[sg] = sq;
        part_k[sg] = sk;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tq = 0.0f;
        float tk = 0.0f;
        for (uint i = 0; i < DIM / 32; ++i) {
            tq += part_q[i];
            tk += part_k[i];
        }
        // Keep zero rows finite.
        gates[0] = 1.0f / max(sqrt(tq), 1e-12f);
        gates[1] = 1.0f / max(sqrt(tk), 1e-12f);
        // Stable softplus.
        float x = float(a[h]) + float(dt_bias[h]);
        float softplus = x > 20.0f ? x : log(1.0f + exp(x));
        gates[2] = exp(-exp(a_log[h]) * softplus);
        gates[3] = 1.0f / (1.0f + exp(-float(b[h])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    q_norm[tid] = qv * gates[0] * scale;
    k_norm[tid] = kv * gates[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float decay = gates[2];
    float beta = gates[3];
    device StateT* st = state + (ulong)h * DIM * DIM;

    // Predict from the fully decayed state before applying the delta update.
    float kv_pred = 0.0f;
    for (uint ki = 0; ki < DIM; ++ki) {
        kv_pred += k_norm[ki] * st[ki * DIM + tid] * decay;
    }
    float v_new = (float(v[h * DIM + tid]) - kv_pred) * beta;
    float o = 0.0f;
    for (uint ki = 0; ki < DIM; ++ki) {
        float updated = st[ki * DIM + tid] * decay + k_norm[ki] * v_new;
        st[ki * DIM + tid] = StateT(updated);
        o += q_norm[ki] * updated;
    }
    return o;
}

#define GDN_STEP_GATED_WRAPPER(NAME, STATE_T)                                 \
kernel void NAME(device const bfloat* q       [[buffer(0)]],                  \
                 device const bfloat* k       [[buffer(1)]],                  \
                 device const bfloat* v       [[buffer(2)]],                  \
                 device const bfloat* a       [[buffer(3)]],                  \
                 device const bfloat* b       [[buffer(4)]],                  \
                 device const float*  a_log   [[buffer(5)]],                  \
                 device const bfloat* dt_bias [[buffer(6)]],                  \
                 device STATE_T*      state   [[buffer(7)]],                  \
                 device const bfloat* z       [[buffer(8)]],                  \
                 device const float*  norm_w  [[buffer(9)]],                  \
                 device bfloat*       out     [[buffer(10)]],                 \
                 constant float&      scale   [[buffer(11)]],                 \
                 constant uint&       vpk     [[buffer(12)]],                 \
                 constant float&      eps     [[buffer(13)]],                 \
                 constant uint&       gate_act [[buffer(14)]],                \
                 uint h    [[threadgroup_position_in_grid]],                  \
                 uint tid  [[thread_index_in_threadgroup]],                   \
                 uint sg   [[simdgroup_index_in_threadgroup]],                \
                 uint lane [[thread_index_in_simdgroup]]) {                   \
    threadgroup float q_norm[DIM];                                            \
    threadgroup float k_norm[DIM];                                            \
    threadgroup float part_q[DIM / 32];                                       \
    threadgroup float part_k[DIM / 32];                                       \
    threadgroup float gates[4];                                               \
    threadgroup bfloat raw[DIM];                                              \
    threadgroup float norm_part[DIM / 32];                                    \
    threadgroup float inv_rms = 0.0f;                                         \
    const float o = gdn_step_body(q, k, v, a, b, a_log, dt_bias, state,      \
                                  scale, vpk, h, tid, sg, lane, q_norm,       \
                                  k_norm, part_q, part_k, gates);             \
    raw[tid] = bfloat(o);                                                     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    const float rv = float(raw[tid]);                                         \
    const float ss = simd_sum(rv * rv);                                       \
    if (lane == 0) {                                                          \
        norm_part[sg] = ss;                                                   \
    }                                                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    if (tid == 0) {                                                           \
        float total = 0.0f;                                                   \
        for (uint i = 0; i < DIM / 32; ++i) {                                \
            total += norm_part[i];                                            \
        }                                                                     \
        inv_rms = rsqrt(total / float(DIM) + eps);                            \
    }                                                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    const float gate = float(z[h * DIM + tid]);                               \
    const float act = gate_activation(gate, gate_act);                        \
    out[h * DIM + tid] = bfloat(norm_w[tid] * rv * inv_rms * act);            \
}

GDN_STEP_GATED_WRAPPER(gdn_step_gated, float)
#undef GDN_STEP_GATED_WRAPPER

// Normalizes q/k rows for the register scan.
kernel void gdn_qk_l2norm(device const bfloat* qkv   [[buffer(0)]],  // [M, C]
                          device bfloat*       qk    [[buffer(1)]],  // [M, 2*HK*DIM]
                          constant float&      scale [[buffer(2)]],
                          constant uint&       HK    [[buffer(3)]],
                          constant uint&       H     [[buffer(4)]],
                          uint3 gid  [[thread_position_in_grid]],
                          uint  lane [[thread_index_in_simdgroup]]) {
    const uint hk = gid.y;
    const ulong row = (ulong)gid.z * (2 * HK + H) * DIM;
    const ulong out_row = (ulong)gid.z * 2 * HK * DIM;
    float qv[DIM / 32], kv[DIM / 32];
    float sq = 0.0f, sk = 0.0f;
    for (uint i = 0; i < DIM / 32; ++i) {
        uint d = lane + 32 * i;
        qv[i] = float(qkv[row + hk * DIM + d]);
        kv[i] = float(qkv[row + (HK + hk) * DIM + d]);
        sq += qv[i] * qv[i];
        sk += kv[i] * kv[i];
    }
    float inv_q = 1.0f / max(sqrt(simd_sum(sq)), 1e-12f);
    float inv_k = 1.0f / max(sqrt(simd_sum(sk)), 1e-12f);
    for (uint i = 0; i < DIM / 32; ++i) {
        uint d = lane + 32 * i;
        qk[out_row + hk * DIM + d] = bfloat(qv[i] * inv_q * scale);
        qk[out_row + (HK + hk) * DIM + d] = bfloat(kv[i] * inv_k);
    }
}

// Computes FP32 decay and beta gates.
kernel void gdn_gates(device const bfloat* a       [[buffer(0)]],  // [M, H]
                      device const bfloat* b       [[buffer(1)]],  // [M, H]
                      device const float*  a_log   [[buffer(2)]],  // [H]
                      device const bfloat* dt_bias [[buffer(3)]],  // [H]
                      device float*        decay   [[buffer(4)]],  // [M, H]
                      device float*        beta    [[buffer(5)]],  // [M, H]
                      constant uint&       H       [[buffer(6)]],
                      uint gid [[thread_position_in_grid]]) {
    uint h = gid % H;
    float x = float(a[gid]) + float(dt_bias[h]);
    float softplus = x > 20.0f ? x : log(1.0f + exp(x));
    decay[gid] = exp(-exp(a_log[h]) * softplus);
    beta[gid] = 1.0f / (1.0f + exp(-float(b[gid])));
}

// Register-resident prefill scan over tokens, one simdgroup per value column.
// After token t < mid_count the running state is also written to `mid` slot t
// (`[mid_count, H, DIM, DIM]`), so a caller that later rejects tokens t+1..
// can roll the recurrence back without rerunning it.
template <typename StateT>
static inline void gdn_prefill_regscan_body(device const bfloat* qkv,
                                            device const bfloat* qk,
                                            device const float* decay,
                                            device const float* beta,
                                            device StateT* state,
                                            device bfloat* out,
                                            device StateT* mid,
                                            uint M,
                                            uint H,
                                            uint vpk,
                                            uint mid_count,
                                            uint2 tg,
                                            uint sg,
                                            uint lane) {
    const uint NK = DIM / 32;  // state rows per lane
    const uint h = tg.x;
    const uint dv = tg.y * 4 + sg;
    const uint HK = H / vpk;
    const uint hk = h / vpk;
    const uint C = (2 * HK + H) * DIM;

    // Each lane owns NK rows of one state column.
    device StateT* st = state + ((ulong)h * DIM + NK * lane) * DIM + dv;
    float s[NK];
    for (uint i = 0; i < NK; ++i) {
        s[i] = st[i * DIM];
    }

    for (uint t = 0; t < M; ++t) {
        device const bfloat* qrow =
            qk + (ulong)t * 2 * HK * DIM + hk * DIM + NK * lane;
        device const bfloat* krow = qrow + HK * DIM;
        float g = decay[t * H + h];
        float kh[NK];
        float kv = 0.0f;
        for (uint i = 0; i < NK; ++i) {
            kh[i] = float(krow[i]);
            s[i] *= g;
            kv += kh[i] * s[i];
        }
        kv = simd_sum(kv);
        float v_new = (float(qkv[(ulong)t * C + (2 * HK + h) * DIM + dv]) - kv)
            * beta[t * H + h];
        float o = 0.0f;
        for (uint i = 0; i < NK; ++i) {
            s[i] += kh[i] * v_new;
            o += float(qrow[i]) * s[i];
        }
        o = simd_sum(o);
        if (lane == 0) {
            out[((ulong)t * H + h) * DIM + dv] = bfloat(o);
        }
        if (t < mid_count) {
            device StateT* md = mid + (ulong)t * H * DIM * DIM
                + ((ulong)h * DIM + NK * lane) * DIM + dv;
            for (uint i = 0; i < NK; ++i) {
                md[i * DIM] = StateT(s[i]);
            }
        }
    }
    for (uint i = 0; i < NK; ++i) {
        st[i * DIM] = StateT(s[i]);
    }
}

// One token's operands of the scan for a lane: its NK k and q values and the
// decay and beta gates.
static inline void gdn_scan_load(device const bfloat* qk, device const float* decay,
                                 device const float* beta, uint t, uint H, uint HK, uint hk,
                                 uint h, uint lane0, thread float* kh, thread float* qh,
                                 thread float& g, thread float& b) {
    device const bfloat* qrow = qk + (ulong)t * 2 * HK * DIM + hk * DIM + lane0;
    device const bfloat* krow = qrow + HK * DIM;
    for (uint i = 0; i < DIM / 32; ++i) {
        kh[i] = float(krow[i]);
        qh[i] = float(qrow[i]);
    }
    g = decay[t * H + h];
    b = beta[t * H + h];
}

// The scan with CS value columns per simdgroup (dv .. dv + CS - 1), sharing
// each token's k, q and gate loads across the columns (the single-column
// scan reloads them per column: 128 simdgroups per head read the same
// rows), and with PF the next token's operands loaded a token ahead. Per
// column the arithmetic and its order are the single-column scan's; the
// results differ from it only by the compiler's fast-math contraction of
// the interleaved column updates (the 4-layer golden probes moved by less
// than a bf16 ulp of their logit gap).
template <typename StateT, int CS, bool PF>
static inline void gdn_prefill_regscan_cols_body(device const bfloat* qkv,
                                                 device const bfloat* qk,
                                                 device const float* decay,
                                                 device const float* beta,
                                                 device StateT* state,
                                                 device bfloat* out,
                                                 device StateT* mid,
                                                 uint M,
                                                 uint H,
                                                 uint vpk,
                                                 uint mid_count,
                                                 uint2 tg,
                                                 uint sg,
                                                 uint lane) {
    const uint NK = DIM / 32;
    const uint h = tg.x;
    const uint dv = (tg.y * 4 + sg) * uint(CS);
    const uint HK = H / vpk;
    const uint hk = h / vpk;
    const uint C = (2 * HK + H) * DIM;

    device StateT* st = state + ((ulong)h * DIM + NK * lane) * DIM + dv;
    float s[CS][DIM / 32];
    for (uint c = 0; c < uint(CS); ++c) {
        for (uint i = 0; i < NK; ++i) {
            s[c][i] = st[i * DIM + c];
        }
    }
    float kh[DIM / 32], qh[DIM / 32], g = 0.0f, b = 0.0f, v[CS];
    if (PF && M > 0) {
        gdn_scan_load(qk, decay, beta, 0u, H, HK, hk, h, NK * lane, kh, qh, g, b);
        for (uint c = 0; c < uint(CS); ++c) {
            v[c] = float(qkv[(2 * HK + h) * DIM + dv + c]);
        }
    }
    for (uint t = 0; t < M; ++t) {
        float kn[DIM / 32], qn[DIM / 32], gn = 0.0f, bn = 0.0f, vn[CS];
        if (PF) {
            if (t + 1 < M) {
                gdn_scan_load(qk, decay, beta, t + 1, H, HK, hk, h, NK * lane, kn, qn, gn, bn);
                for (uint c = 0; c < uint(CS); ++c) {
                    vn[c] = float(qkv[(ulong)(t + 1) * C + (2 * HK + h) * DIM + dv + c]);
                }
            }
        } else {
            gdn_scan_load(qk, decay, beta, t, H, HK, hk, h, NK * lane, kh, qh, g, b);
            for (uint c = 0; c < uint(CS); ++c) {
                v[c] = float(qkv[(ulong)t * C + (2 * HK + h) * DIM + dv + c]);
            }
        }
        float kv[CS];
        for (uint c = 0; c < uint(CS); ++c) {
            kv[c] = 0.0f;
        }
        for (uint i = 0; i < NK; ++i) {
            for (uint c = 0; c < uint(CS); ++c) {
                s[c][i] *= g;
                kv[c] += kh[i] * s[c][i];
            }
        }
        float vnew[CS];
        for (uint c = 0; c < uint(CS); ++c) {
            vnew[c] = (v[c] - simd_sum(kv[c])) * b;
        }
        float o[CS];
        for (uint c = 0; c < uint(CS); ++c) {
            o[c] = 0.0f;
        }
        for (uint i = 0; i < NK; ++i) {
            for (uint c = 0; c < uint(CS); ++c) {
                s[c][i] += kh[i] * vnew[c];
                o[c] += qh[i] * s[c][i];
            }
        }
        for (uint c = 0; c < uint(CS); ++c) {
            const float oc = simd_sum(o[c]);
            if (lane == 0) {
                out[((ulong)t * H + h) * DIM + dv + c] = bfloat(oc);
            }
        }
        if (t < mid_count) {
            device StateT* md = mid + (ulong)t * H * DIM * DIM
                + ((ulong)h * DIM + NK * lane) * DIM + dv;
            for (uint c = 0; c < uint(CS); ++c) {
                for (uint i = 0; i < NK; ++i) {
                    md[i * DIM + c] = StateT(s[c][i]);
                }
            }
        }
        if (PF) {
            for (uint i = 0; i < NK; ++i) {
                kh[i] = kn[i];
                qh[i] = qn[i];
            }
            g = gn;
            b = bn;
            for (uint c = 0; c < uint(CS); ++c) {
                v[c] = vn[c];
            }
        }
    }
    for (uint c = 0; c < uint(CS); ++c) {
        for (uint i = 0; i < NK; ++i) {
            st[i * DIM + c] = StateT(s[c][i]);
        }
    }
}

#define GDN_REGSCAN_WRAPPER(NAME, STATE_T)                                    \
kernel void NAME(device const bfloat* qkv   [[buffer(0)]],                    \
                 device const bfloat* qk    [[buffer(1)]],                    \
                 device const float*  decay [[buffer(2)]],                    \
                 device const float*  beta  [[buffer(3)]],                    \
                 device STATE_T*      state [[buffer(4)]],                    \
                 device bfloat*       out   [[buffer(5)]],                    \
                 device STATE_T*      mid   [[buffer(6)]],                    \
                 constant uint&       M     [[buffer(7)]],                    \
                 constant uint&       H     [[buffer(8)]],                    \
                 constant uint&       vpk   [[buffer(9)]],                    \
                 constant uint&       mid_count [[buffer(10)]],               \
                 uint2 tg   [[threadgroup_position_in_grid]],                 \
                 uint  sg   [[simdgroup_index_in_threadgroup]],               \
                 uint  lane [[thread_index_in_simdgroup]]) {                  \
    gdn_prefill_regscan_body(qkv, qk, decay, beta, state, out, mid, M, H, vpk,\
                             mid_count, tg, sg, lane);                        \
}

// The single-column scan, kept for comparison by name (`LILY_GDN_SCAN_KERNEL`).
GDN_REGSCAN_WRAPPER(gdn_prefill_regscan_c1, float)
#undef GDN_REGSCAN_WRAPPER

#define GDN_REGSCAN_VARIANT(NAME, CS, PF)                                      \
kernel void NAME(device const bfloat* qkv   [[buffer(0)]],                    \
                 device const bfloat* qk    [[buffer(1)]],                    \
                 device const float*  decay [[buffer(2)]],                    \
                 device const float*  beta  [[buffer(3)]],                    \
                 device float*        state [[buffer(4)]],                    \
                 device bfloat*       out   [[buffer(5)]],                    \
                 device float*        mid   [[buffer(6)]],                    \
                 constant uint&       M     [[buffer(7)]],                    \
                 constant uint&       H     [[buffer(8)]],                    \
                 constant uint&       vpk   [[buffer(9)]],                    \
                 constant uint&       mid_count [[buffer(10)]],               \
                 uint2 tg   [[threadgroup_position_in_grid]],                 \
                 uint  sg   [[simdgroup_index_in_threadgroup]],               \
                 uint  lane [[thread_index_in_simdgroup]]) {                  \
    gdn_prefill_regscan_cols_body<float, CS, PF>(qkv, qk, decay, beta, state, \
                                                 out, mid, M, H, vpk, mid_count,\
                                                 tg, sg, lane);               \
}
// The shipped scan: four value columns per simdgroup. Measured per dispatch
// on the profile transport at the chunk shape (4096 tokens, 48 heads), two
// rotated runs: one column 7.8 to 8.4 ms, two 5.7 to 6.7, four 5.2 to 6.6,
// eight 7.5 to 9.2, sixteen 23 to 25 (registers); loading the next token's
// operands a token ahead was slower at every width (the recurrence's own
// latency, not the loads', is the chain).
GDN_REGSCAN_VARIANT(gdn_prefill_regscan, 4, false)
GDN_REGSCAN_VARIANT(gdn_prefill_regscan_c2, 2, false)
#undef GDN_REGSCAN_VARIANT

// Causal depthwise conv1d + SiLU; window stores KD-1 inputs oldest first.
kernel void conv1d_step_bf16(device bfloat*       window [[buffer(0)]],  // [C, KD-1]
                             device const bfloat* x      [[buffer(1)]],  // [C]
                             device const bfloat* w      [[buffer(2)]],  // [KD, C]
                             device bfloat*       out    [[buffer(3)]],  // [C]
                             constant uint&       C      [[buffer(4)]],
                             constant uint&       KD     [[buffer(5)]],
                             uint c [[thread_position_in_grid]]) {
    uint taps = KD - 1;
    float acc = 0.0f;
    for (uint t = 0; t < taps; ++t) {
        acc += float(window[c * taps + t]) * float(w[t * C + c]);
    }
    float xc = float(x[c]);
    acc += xc * float(w[taps * C + c]);
    out[c] = bfloat(acc / (1.0f + exp(-acc)));
    for (uint t = 0; t + 1 < taps; ++t) {
        window[c * taps + t] = window[c * taps + t + 1];
    }
    window[c * taps + taps - 1] = bfloat(xc);
}

// Rewinds a conv window: `win_out` becomes the window after only the first
// `n` rows of `x` followed `win_in` (the last S inputs of win_in ++ x[0..n]).
// Shared by the GDN conv (S = KD-1) and the dilated PLE conv (S = (KD-1)*DIL).
kernel void conv_window_rollback_bf16(device const bfloat* win_in  [[buffer(0)]],  // [C, S]
                                      device const bfloat* x       [[buffer(1)]],  // [M, C]
                                      device bfloat*       win_out [[buffer(2)]],  // [C, S]
                                      constant uint&       C       [[buffer(3)]],
                                      constant uint&       S       [[buffer(4)]],
                                      constant uint&       n       [[buffer(5)]],
                                      uint c [[thread_position_in_grid]]) {
    for (uint s = 0; s < S; ++s) {
        const uint i = n + s;  // index into win_in ++ x[0..n]
        win_out[c * S + s] = i < S ? win_in[c * S + i] : x[(ulong)(i - S) * C + c];
    }
}

// Tiled conv1d prefill uses separate input/output windows to avoid races.
#define CONV_TILE 64  // must match CONV1D_PREFILL_TILE in kernels/gdn.rs

kernel void conv1d_prefill_bf16(device const bfloat* win_in  [[buffer(0)]],  // [C, KD-1]
                                device bfloat*       win_out [[buffer(1)]],  // [C, KD-1]
                                device const bfloat* x       [[buffer(2)]],  // [M, C]
                                device const bfloat* w       [[buffer(3)]],  // [KD, C]
                                device bfloat*       out     [[buffer(4)]],  // [M, C]
                                constant uint&       C       [[buffer(5)]],
                                constant uint&       KD      [[buffer(6)]],
                                constant uint&       M       [[buffer(7)]],
                                uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x;
    uint m0 = gid.y * CONV_TILE;
    uint m1 = min(m0 + CONV_TILE, M);
    uint taps = KD - 1;
    // Prime the register window with inputs preceding this tile.
    float win[8];  // supports KD <= 9; Qwen3.5 uses KD = 4
    for (uint t = 0; t < taps; ++t) {
        uint i = m0 + t;
        win[t] = i >= taps ? float(x[(ulong)(i - taps) * C + c])
                           : float(win_in[c * taps + i]);
    }
    for (uint m = m0; m < m1; ++m) {
        float acc = 0.0f;
        for (uint t = 0; t < taps; ++t) {
            acc += win[t] * float(w[t * C + c]);
        }
        float xc = float(x[(ulong)m * C + c]);
        acc += xc * float(w[taps * C + c]);
        out[(ulong)m * C + c] = bfloat(acc / (1.0f + exp(-acc)));
        for (uint t = 0; t + 1 < taps; ++t) {
            win[t] = win[t + 1];
        }
        win[taps - 1] = xc;
    }
    // The final tile writes the post-chunk window.
    if (m1 == M) {
        for (uint t = 0; t < taps; ++t) {
            win_out[c * taps + t] = bfloat(win[t]);
        }
    }
}

// Gated RMSNorm, one threadgroup per row.
#define TG 256

kernel void gated_rmsnorm_bf16(device const bfloat* x    [[buffer(0)]],  // [rows, D]
                               device const bfloat* gate [[buffer(1)]],  // [rows, D]
                               device const float*  w    [[buffer(2)]],  // [D]
                               device bfloat*       out  [[buffer(3)]],  // [rows, D]
                               constant uint&       D    [[buffer(4)]],
                               constant float&      eps  [[buffer(5)]],
                               constant uint&       gate_act [[buffer(6)]],
                               uint row  [[threadgroup_position_in_grid]],
                               uint tid  [[thread_index_in_threadgroup]],
                               uint sg   [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;

    float acc = 0.0f;
    for (uint i = tid; i < D; i += TG) {
        float v = float(x[row * D + i]);
        acc += v * v;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < TG / 32; ++i) {
            total += partial[i];
        }
        inv_rms = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < D; i += TG) {
        float g = float(gate[row * D + i]);
        float s = gate_activation(g, gate_act);
        out[row * D + i] = bfloat(w[i] * float(x[row * D + i]) * inv_rms * s);
    }
}

// ---------------------------------------------------------------------------
// Chunked prefill scan on the tensor ops (Metal 4). The recurrence over a
// chunk of GDN_CHUNK tokens is written in its WY form: with the local
// cumulative log-decay G_i, beta_i and the unit-norm keys,
//
//     A[i][j] = beta_i (k_i . k_j) exp(G_i - G_j)      (j < i)
//     T       = (I + A)^-1
//     W       = T diag(beta exp(G)) K          U = T diag(beta) V
//     u~      = U - W S_0                                  (pseudo values)
//     o_t     = exp(G_t) q_t S_0 + sum_{i<=t} (q_t . k_i) exp(G_t - G_i) u~_i
//     S_C     = exp(G_C) S_0 + sum_i exp(G_C - G_i) k_i u~_i^T
//
// The WY pass (`gdn_chunk_wy3`, or `gdn_chunk_wy` for more than three
// value heads per key head) does the part that needs no state, for every
// chunk at once: A, T by forward substitution, W, U and the decayed causal
// Q K^T product P. M is a multiple of GDN_CHUNK: the products over a
// chunk's tokens take that dimension as a dynamic extent, and an extent
// short of the chunk read rows past it (the tensor ops pad the reduction
// dimension, not the tile edges), which in the unit tests' exactly-sized
// buffers picked up other allocations' words and, under concurrent tests,
// NaNs; the dispatch hands a ragged tail to the token-serial scan instead. `gdn_chunk_scan` walks the chunks of one head in order
// for a block of GDN_CHUNK_NB state columns, carrying the fp32 state in a
// cooperative tensor and reading it through a bf16 copy. The results
// differ from the token-serial scan by the bf16 rounding of the state and
// pseudo-value reads and by the accumulation order of the tensor products.
//
// Measured at the chunk shape (4 096 tokens, 48 value heads over 16 key
// heads) on the profile transport: the two passes 0.86 + 1.75 ms against
// the serial scan's 4.7. The scan pass is bound by streaming each chunk's
// W, K, Q and P rows into the cores, not by the products: with every
// threadgroup reading one head's (cache-resident) rows it took 0.49 ms,
// the same as with no products at all. Wider column blocks (32, 64, 128
// with the state copy in device memory) read fewer bytes but ran with too
// few threadgroups to cover the dependent-product latency (2.2, 2.8 and
// 3.2 ms); narrower ones (8) doubled the bytes (3.8 ms); touching the
// next chunk's lines ahead did not help (2.2 to 2.5 ms), nor did
// pre-transposed keys for the state update or eight simdgroups.
#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#define GDN_CHUNK 64        // tokens per chunk
#define GDN_CHUNK_SG 4      // simdgroups per `gdn_chunk_scan` threadgroup
#define GDN_CHUNK_NB 16     // state columns per `gdn_chunk_scan` threadgroup

static inline float gdn_log_decay(float a, float a_log, float dt_bias) {
    const float x = a + dt_bias;
    const float softplus = x > 20.0f ? x : log(1.0f + exp(x));
    return -exp(a_log) * softplus;
}

// The WY pass with one threadgroup per (chunk, key head), the value heads
// of the key head in turn, one column of T per lane in registers.
static inline void gdn_chunk_wy_body(device bfloat* qkv,
                                     device bfloat* qk,
                                     device const bfloat* a,
                                     device const bfloat* b,
                                     device const float* a_log,
                                     device const bfloat* dt_bias,
                                     device bfloat* w_out,
                                     device bfloat* u_out,
                                     device bfloat* p_out,
                                     device float* g_out,
                                     uint M,
                                     uint H,
                                     uint vpk,
                                     threadgroup float* A,   // [C, C]: A, then T_w | T_u as bf16
                                     threadgroup float* G,   // [C]
                                     threadgroup float* Bt,  // [C]
                                     threadgroup float* g_total,
                                     uint2 tg,
                                     uint tid,
                                     uint sg,
                                     uint lane) {
    using namespace mpp::tensor_ops;
    constexpr int C = GDN_CHUNK;
    constexpr int SG = 4;

    const uint t0 = tg.x * uint(C);
    if (t0 >= M) {
        return;
    }
    const uint n = min(uint(C), M - t0);
    const uint hk = tg.y;
    const uint HK = H / vpk;
    const uint CW = (2 * HK + H) * uint(DIM);

    const array<int, 2> qk_strides{1, int(2 * HK * uint(DIM))};
    auto tK = tensor(qk + (ulong)t0 * 2 * HK * DIM + (HK + hk) * DIM,
                     dextents<int32_t, 2>(DIM, int(n)), qk_strides);
    auto tQ = tensor(qk + (ulong)t0 * 2 * HK * DIM + hk * DIM,
                     dextents<int32_t, 2>(DIM, int(n)), qk_strides);
    using KT = decltype(tK);

    // Products of chunk rows: [n x DIM] . [n x DIM]^T.
    constexpr auto kk_desc = matmul2d_descriptor(
        C, C, DIM, false, /*transpose_right=*/true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<kk_desc, metal::execution_simdgroups<SG>> kk_op;
    // T_w . K and T_u . V: [C x C] . [n x DIM] over the chunk's tokens.
    constexpr auto tw_desc = matmul2d_descriptor(
        C, DIM, static_cast<int>(metal::dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<tw_desc, metal::execution_simdgroups<SG>> tw_op;

    threadgroup bfloat* Tw = (threadgroup bfloat*)A;
    threadgroup bfloat* Tu = Tw + C * C;
    auto tTw = tensor(Tw, dextents<int32_t, 2>(C, C));
    auto tTu = tensor(Tu, dextents<int32_t, 2>(C, C));
    using TT = decltype(tTw);
    const array<int, 2> wu_strides{1, int(H * uint(DIM))};

    for (uint v = 0; v < vpk; ++v) {
        const uint h = hk * vpk + v;

        // Local cumulative log-decay and beta; tokens past the chunk's end
        // carry no decay and no write.
        {
            float lg = 0.0f;
            float bt = 0.0f;
            if (tid < n) {
                const ulong r = (ulong)(t0 + tid) * H + h;
                lg = gdn_log_decay(float(a[r]), a_log[h], float(dt_bias[h]));
                bt = 1.0f / (1.0f + exp(-float(b[r])));
            }
            const float pre = simd_prefix_inclusive_sum(lg);
            if (sg == 0 && lane == 31) {
                *g_total = pre;
            }
            if (tid < uint(C)) {
                Bt[tid] = bt;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < uint(C)) {
                G[tid] = pre + (sg == 1 ? *g_total : 0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A from K K^T (recomputed per value head, cheaper than holding it).
        {
            auto kk = kk_op.template get_destination_cooperative_tensor<KT, KT, float>();
            kk_op.run(tK, tK, kk);
            for (uint16_t i = 0; i < kk.get_capacity(); ++i) {
                if (kk.is_valid_element(i)) {
                    const auto ix = kk.get_multidimensional_index(i);
                    const uint j = uint(ix[0]);
                    const uint r = uint(ix[1]);
                    A[r * C + j] = (j < r && r < n) ? Bt[r] * kk[i] * exp(G[r] - G[j]) : 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // T = (I + A)^-1 by forward substitution, one column per lane, the
        // column in registers, four accumulation chains.
        float t[C];
        if (tid < uint(C)) {
            const uint c = tid;
#pragma clang loop unroll(full)
            for (int i = 0; i < C; ++i) {
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma clang loop unroll(full)
                for (int j = 0; j + 3 < i; j += 4) {
                    s0 = fma(A[i * C + j], t[j], s0);
                    s1 = fma(A[i * C + j + 1], t[j + 1], s1);
                    s2 = fma(A[i * C + j + 2], t[j + 2], s2);
                    s3 = fma(A[i * C + j + 3], t[j + 3], s3);
                }
#pragma clang loop unroll(full)
                for (int j = i & ~3; j < i; ++j) {
                    s0 = fma(A[i * C + j], t[j], s0);
                }
                const float s = (s0 + s1) + (s2 + s3);
                t[i] = uint(i) == c ? 1.0f : (uint(i) < c ? 0.0f : -s);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < uint(C)) {
            const uint c = tid;
            const float bw = Bt[c] * exp(G[c]);
            const float bu = Bt[c];
#pragma clang loop unroll(full)
            for (int i = 0; i < C; ++i) {
                Tw[i * C + c] = bfloat(t[i] * bw);
                Tu[i * C + c] = bfloat(t[i] * bu);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // W = T_w K and U = T_u V, rows of this chunk, stored as bf16 tiles
        // (the accumulation stays fp32; a bf16 destination rounds once).
        {
            auto acc = tw_op.template get_destination_cooperative_tensor<TT, KT, bfloat>();
            tw_op.run(tTw, tK, acc);
            auto tWo = tensor(w_out + ((ulong)t0 * H + h) * DIM, dextents<int32_t, 2>(DIM, int(n)),
                              wu_strides);
            acc.store(tWo);
        }
        {
            const array<int, 2> v_strides{1, int(CW)};
            auto tV = tensor(qkv + (ulong)t0 * CW + (2 * HK + h) * DIM,
                             dextents<int32_t, 2>(DIM, int(n)), v_strides);
            using VT = decltype(tV);
            auto acc = tw_op.template get_destination_cooperative_tensor<TT, VT, bfloat>();
            tw_op.run(tTu, tV, acc);
            auto tUo = tensor(u_out + ((ulong)t0 * H + h) * DIM, dextents<int32_t, 2>(DIM, int(n)),
                              wu_strides);
            acc.store(tUo);
        }

        // P = causal Q K^T with the decay between the two tokens, through a
        // staged tile (over T_w, which the products above are done with),
        // written out as whole rows.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup bfloat* Pt = Tw;
        {
            auto qkp = kk_op.template get_destination_cooperative_tensor<KT, KT, float>();
            kk_op.run(tQ, tK, qkp);
            for (uint16_t i = 0; i < qkp.get_capacity(); ++i) {
                if (qkp.is_valid_element(i)) {
                    const auto ix = qkp.get_multidimensional_index(i);
                    const uint j = uint(ix[0]);
                    const uint r = uint(ix[1]);
                    if (r < n) {
                        Pt[r * C + j] = bfloat(j <= r ? qkp[i] * exp(G[r] - G[j]) : 0.0f);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const threadgroup uint4* src = (const threadgroup uint4*)Pt;
            for (uint i = tid; i < uint(C) * (C / 8); i += 32u * SG) {
                const uint r = i / (C / 8);
                if (r < n) {
                    ((device uint4*)(p_out + ((ulong)(t0 + r) * H + h) * C))[i - r * (C / 8)] = src[i];
                }
            }
        }
        if (tid < n) {
            g_out[(ulong)(t0 + tid) * H + h] = G[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gdn_chunk_wy(device bfloat*       qkv     [[buffer(0)]],  // [M, (2*HK+H)*DIM]
                         device bfloat*       qk      [[buffer(1)]],  // [M, 2*HK*DIM] normalized
                         device const bfloat* a       [[buffer(2)]],  // [M, H]
                         device const bfloat* b       [[buffer(3)]],  // [M, H]
                         device const float*  a_log   [[buffer(4)]],  // [H]
                         device const bfloat* dt_bias [[buffer(5)]],  // [H]
                         device bfloat*       w_out   [[buffer(6)]],  // [M, H, DIM]
                         device bfloat*       u_out   [[buffer(7)]],  // [M, H, DIM]
                         device bfloat*       p_out   [[buffer(8)]],  // [M, H, GDN_CHUNK]
                         device float*        g_out   [[buffer(9)]],  // [M, H] local cumsum
                         constant uint&       M       [[buffer(10)]],
                         constant uint&       H       [[buffer(11)]],
                         constant uint&       vpk     [[buffer(12)]],
                         uint2 tg   [[threadgroup_position_in_grid]],
                         uint  tid  [[thread_index_in_threadgroup]],
                         uint  sg   [[simdgroup_index_in_threadgroup]],
                         uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup float A[GDN_CHUNK * GDN_CHUNK];
    threadgroup float G[GDN_CHUNK], Bt[GDN_CHUNK];
    threadgroup float g_total;
    gdn_chunk_wy_body(qkv, qk, a, b, a_log, dt_bias, w_out, u_out, p_out, g_out, M, H, vpk, A,
                      G, Bt, &g_total, tg, tid, sg, lane);
}

// The WY pass with the three value heads of a key head solved at once:
// 256 threads, thread (v, c) owns column c of head v's T. The three A
// matrices are held as fp16 (|A| <= 1) so they fit beside the tiles; the
// K K^T and Q K^T products are computed once. Measured 0.86 ms against
// 1.13 for the per-head pass at the chunk shape.
static inline void gdn_chunk_wy3_body(device bfloat* qkv,
                                      device bfloat* qk,
                                      device const bfloat* a,
                                      device const bfloat* b,
                                      device const float* a_log,
                                      device const bfloat* dt_bias,
                                      device bfloat* w_out,
                                      device bfloat* u_out,
                                      device bfloat* p_out,
                                      device float* g_out,
                                      uint M,
                                      uint H,
                                      uint vpk,
                                      threadgroup half* A3,    // [3][C][C]: A; then T_w | T_u | P tile
                                      threadgroup float* G,    // [3][C]
                                      threadgroup float* Bt,   // [3][C]
                                      threadgroup float* g_total,  // [3]
                                      uint2 tg,
                                      uint tid,
                                      uint sg,
                                      uint lane) {
    using namespace mpp::tensor_ops;
    constexpr int C = GDN_CHUNK;
    constexpr int SG = 8;
    constexpr uint THREADS = 32u * SG;

    const uint t0 = tg.x * uint(C);
    if (t0 >= M) {
        return;
    }
    const uint n = min(uint(C), M - t0);
    const uint hk = tg.y;
    const uint HK = H / vpk;
    const uint CW = (2 * HK + H) * uint(DIM);
    const uint v = tid / uint(C);   // value head of this thread (3 = idle)
    const uint c = tid % uint(C);

    const array<int, 2> qk_strides{1, int(2 * HK * uint(DIM))};
    auto tK = tensor(qk + (ulong)t0 * 2 * HK * DIM + (HK + hk) * DIM,
                     dextents<int32_t, 2>(DIM, int(n)), qk_strides);
    auto tQ = tensor(qk + (ulong)t0 * 2 * HK * DIM + hk * DIM,
                     dextents<int32_t, 2>(DIM, int(n)), qk_strides);
    using KT = decltype(tK);

    constexpr auto kk_desc = matmul2d_descriptor(
        C, C, DIM, false, /*transpose_right=*/true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<kk_desc, metal::execution_simdgroups<SG>> kk_op;
    constexpr auto tw_desc = matmul2d_descriptor(
        C, DIM, static_cast<int>(metal::dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<tw_desc, metal::execution_simdgroups<SG>> tw_op;

    threadgroup bfloat* Tw = (threadgroup bfloat*)A3;
    threadgroup bfloat* Tu = Tw + C * C;
    threadgroup bfloat* Pt = Tu + C * C;
    auto tTw = tensor(Tw, dextents<int32_t, 2>(C, C));
    auto tTu = tensor(Tu, dextents<int32_t, 2>(C, C));
    using TT = decltype(tTw);

    // Gates: thread (v, i) for i < n; the prefix sum runs per head over its
    // two simdgroups.
    {
        float lg = 0.0f;
        float bt = 0.0f;
        if (v < vpk && c < n) {
            const ulong r = (ulong)(t0 + c) * H + hk * vpk + v;
            lg = gdn_log_decay(float(a[r]), a_log[hk * vpk + v], float(dt_bias[hk * vpk + v]));
            bt = 1.0f / (1.0f + exp(-float(b[r])));
        }
        const float pre = simd_prefix_inclusive_sum(lg);
        if ((sg & 1u) == 0u && lane == 31 && v < 3) {
            g_total[v] = pre;
        }
        if (v < 3) {
            Bt[v * C + c] = bt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (v < 3) {
            G[v * C + c] = pre + ((sg & 1u) ? g_total[v] : 0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto kk = kk_op.template get_destination_cooperative_tensor<KT, KT, float>();
    kk_op.run(tK, tK, kk);
    for (uint16_t i = 0; i < kk.get_capacity(); ++i) {
        if (kk.is_valid_element(i)) {
            const auto ix = kk.get_multidimensional_index(i);
            const uint j = uint(ix[0]);
            const uint r = uint(ix[1]);
            for (uint vv = 0; vv < 3; ++vv) {
                A3[(vv * C + r) * C + j] = half(
                    (j < r && r < n) ? Bt[vv * C + r] * kk[i] * exp(G[vv * C + r] - G[vv * C + j]) : 0.0f);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float t[C];
    if (v < 3) {
        const threadgroup half* Av = A3 + v * C * C;
#pragma clang loop unroll(full)
        for (int i = 0; i < C; ++i) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma clang loop unroll(full)
            for (int j = 0; j + 3 < i; j += 4) {
                s0 = fma(float(Av[i * C + j]), t[j], s0);
                s1 = fma(float(Av[i * C + j + 1]), t[j + 1], s1);
                s2 = fma(float(Av[i * C + j + 2]), t[j + 2], s2);
                s3 = fma(float(Av[i * C + j + 3]), t[j + 3], s3);
            }
#pragma clang loop unroll(full)
            for (int j = i & ~3; j < i; ++j) {
                s0 = fma(float(Av[i * C + j]), t[j], s0);
            }
            const float sum = (s0 + s1) + (s2 + s3);
            t[i] = uint(i) == c ? 1.0f : (uint(i) < c ? 0.0f : -sum);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto qkp = kk_op.template get_destination_cooperative_tensor<KT, KT, float>();
    kk_op.run(tQ, tK, qkp);

    const array<int, 2> wu_strides{1, int(H * uint(DIM))};
    for (uint vv = 0; vv < vpk; ++vv) {
        const uint h = hk * vpk + vv;
        if (v == vv) {
            const float bw = Bt[v * C + c] * exp(G[v * C + c]);
            const float bu = Bt[v * C + c];
#pragma clang loop unroll(full)
            for (int i = 0; i < C; ++i) {
                Tw[i * C + c] = bfloat(t[i] * bw);
                Tu[i * C + c] = bfloat(t[i] * bu);
            }
        }
        for (uint16_t i = 0; i < qkp.get_capacity(); ++i) {
            if (qkp.is_valid_element(i)) {
                const auto ix = qkp.get_multidimensional_index(i);
                const uint j = uint(ix[0]);
                const uint r = uint(ix[1]);
                if (r < n) {
                    Pt[r * C + j] = bfloat(j <= r ? qkp[i] * exp(G[vv * C + r] - G[vv * C + j]) : 0.0f);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto acc = tw_op.template get_destination_cooperative_tensor<TT, KT, bfloat>();
            tw_op.run(tTw, tK, acc);
            auto tWo = tensor(w_out + ((ulong)t0 * H + h) * DIM, dextents<int32_t, 2>(DIM, int(n)),
                              wu_strides);
            acc.store(tWo);
        }
        {
            const array<int, 2> v_strides{1, int(CW)};
            auto tV = tensor(qkv + (ulong)t0 * CW + (2 * HK + h) * DIM,
                             dextents<int32_t, 2>(DIM, int(n)), v_strides);
            using VT = decltype(tV);
            auto acc = tw_op.template get_destination_cooperative_tensor<TT, VT, bfloat>();
            tw_op.run(tTu, tV, acc);
            auto tUo = tensor(u_out + ((ulong)t0 * H + h) * DIM, dextents<int32_t, 2>(DIM, int(n)),
                              wu_strides);
            acc.store(tUo);
        }
        {
            const threadgroup uint4* src = (const threadgroup uint4*)Pt;
            for (uint i = tid; i < uint(C) * (C / 8); i += THREADS) {
                const uint r = i / (C / 8);
                if (r < n) {
                    ((device uint4*)(p_out + ((ulong)(t0 + r) * H + h) * C))[i - r * (C / 8)] = src[i];
                }
            }
        }
        if (tid < n) {
            g_out[(ulong)(t0 + tid) * H + h] = G[vv * C + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gdn_chunk_wy3(device bfloat*       qkv     [[buffer(0)]],                            \
                 device bfloat*       qk      [[buffer(1)]],                            \
                 device const bfloat* a       [[buffer(2)]],                            \
                 device const bfloat* b       [[buffer(3)]],                            \
                 device const float*  a_log   [[buffer(4)]],                            \
                 device const bfloat* dt_bias [[buffer(5)]],                            \
                 device bfloat*       w_out   [[buffer(6)]],                            \
                 device bfloat*       u_out   [[buffer(7)]],                            \
                 device bfloat*       p_out   [[buffer(8)]],                            \
                 device float*        g_out   [[buffer(9)]],                            \
                 constant uint&       M       [[buffer(10)]],                           \
                 constant uint&       H       [[buffer(11)]],                           \
                 constant uint&       vpk     [[buffer(12)]],                           \
                 uint2 tg   [[threadgroup_position_in_grid]],                           \
                 uint  tid  [[thread_index_in_threadgroup]],                            \
                 uint  sg   [[simdgroup_index_in_threadgroup]],                         \
                 uint  lane [[thread_index_in_simdgroup]]) {                            \
    threadgroup half A3[3 * GDN_CHUNK * GDN_CHUNK];                                     \
    threadgroup float G[3 * GDN_CHUNK], Bt[3 * GDN_CHUNK];                              \
    threadgroup float g_total[3];                                                       \
    gdn_chunk_wy3_body(qkv, qk, a, b, a_log, dt_bias, w_out, u_out, p_out,             \
                       g_out, M, H, vpk, A3, G, Bt, g_total, tg, tid, sg, lane);        \
}


// One threadgroup per (block of NB state columns, value head), the chunks
// in order. The fp32 state block lives in the accumulator of the state
// update; its bf16 copy `Sb` feeds the two reads of it. The three products
// with a chunk's rows on the left share one descriptor (dynamic K,
// accumulate), so the output's two terms land in one accumulator with the
// exp(G_t) row scale applied between them.
template <int NB, int SG, typename ST>
static inline void gdn_chunk_scan_body(device bfloat* qk,
                                       device bfloat* w,
                                       device bfloat* u,
                                       device bfloat* p,
                                       device const float* g,
                                       device float* state,
                                       device bfloat* out,
                                       uint M,
                                       uint H,
                                       uint vpk,
                                       threadgroup ST* Sb,   // [DIM, NB]
                                       threadgroup ST* Ut,   // [C, NB]
                                       threadgroup ST* Ut2,  // [C, NB]
                                       uint2 tg) {
    using namespace mpp::tensor_ops;
    constexpr int C = GDN_CHUNK;

    const uint cb = tg.x;
    const uint h = tg.y;
    const uint HK = H / vpk;
    const uint hk = h / vpk;
    const uint col0 = cb * uint(NB);

    auto tSb = tensor(Sb, dextents<int32_t, 2>(NB, DIM));
    auto tUt = tensor(Ut, dextents<int32_t, 2>(NB, C));
    auto tUt2 = tensor(Ut2, dextents<int32_t, 2>(NB, C));
    const array<int, 2> qk_strides{1, int(2 * HK * uint(DIM))};
    const array<int, 2> wu_strides{1, int(H * uint(DIM))};
    const array<int, 2> p_strides{1, int(H * uint(C))};

    // [n x K] . [K x NB] accumulated: W S, Q S and P u~.
    constexpr auto row_desc = matmul2d_descriptor(
        C, NB, static_cast<int>(metal::dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<row_desc, metal::execution_simdgroups<SG>> row_op;
    // [n x DIM]^T . [n x NB] accumulated onto the state block.
    constexpr auto ks_desc = matmul2d_descriptor(
        DIM, NB, static_cast<int>(metal::dynamic_extent), /*transpose_left=*/true,
        false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<ks_desc, metal::execution_simdgroups<SG>> ks_op;

    auto tK0 = tensor(qk + (HK + hk) * DIM, dextents<int32_t, 2>(DIM, int(min(uint(C), M))),
                      qk_strides);
    using RT = decltype(tK0);   // any chunk-row operand
    using SBT = decltype(tSb);  // any threadgroup operand
    auto S = ks_op.template get_destination_cooperative_tensor<RT, SBT, float>();
    device float* st = state + ((ulong)h * DIM) * DIM + col0;
    for (uint16_t i = 0; i < S.get_capacity(); ++i) {
        if (S.is_valid_element(i)) {
            const auto ix = S.get_multidimensional_index(i);
            const float sv = st[(ulong)uint(ix[1]) * DIM + uint(ix[0])];
            S[i] = sv;
            Sb[uint(ix[1]) * NB + uint(ix[0])] = ST(sv);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t0 = 0; t0 < M; t0 += uint(C)) {
        const uint n = min(uint(C), M - t0);
        device const float* gc = g + (ulong)t0 * H + h;
        const float gC = gc[(ulong)(n - 1) * H];
        auto tW = tensor(w + ((ulong)t0 * H + h) * DIM, dextents<int32_t, 2>(DIM, int(n)),
                         wu_strides);
        auto tK = tensor(qk + (ulong)t0 * 2 * HK * DIM + (HK + hk) * DIM,
                         dextents<int32_t, 2>(DIM, int(n)), qk_strides);
        auto tQ = tensor(qk + (ulong)t0 * 2 * HK * DIM + hk * DIM,
                         dextents<int32_t, 2>(DIM, int(n)), qk_strides);
        auto tP = tensor(p + ((ulong)t0 * H + h) * C, dextents<int32_t, 2>(C, int(n)),
                         p_strides);

        // u~ = U - W S_0, plain and decayed to the chunk's end.
        auto acc = row_op.template get_destination_cooperative_tensor<RT, SBT, float>();
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                acc[i] = 0.0f;
            }
        }
        row_op.run(tW, tSb, acc);
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                const auto ix = acc.get_multidimensional_index(i);
                const uint col = uint(ix[0]);
                const uint r = uint(ix[1]);
                if (r < n) {
                    const float uv = float(u[((ulong)(t0 + r) * H + h) * DIM + col0 + col]) - acc[i];
                    Ut[r * NB + col] = ST(uv);
                    Ut2[r * NB + col] = ST(uv * exp(gC - gc[(ulong)r * H]));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = exp(G_C) S_0 + K^T u~'.
        {
            const float gamma = exp(gC);
            for (uint16_t i = 0; i < S.get_capacity(); ++i) {
                if (S.is_valid_element(i)) {
                    S[i] *= gamma;
                }
            }
            ks_op.run(tK, tUt2, S);
        }

        // o = exp(G_t) q_t S_0 + P u~.
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                acc[i] = 0.0f;
            }
        }
        row_op.run(tQ, tSb, acc);
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                const uint r = uint(acc.get_multidimensional_index(i)[1]);
                acc[i] *= r < n ? exp(gc[(ulong)r * H]) : 0.0f;
            }
        }
        row_op.run(tP, tUt, acc);
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                const auto ix = acc.get_multidimensional_index(i);
                const uint col = uint(ix[0]);
                const uint r = uint(ix[1]);
                if (r < n) {
                    out[((ulong)(t0 + r) * H + h) * DIM + col0 + col] = bfloat(acc[i]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The next chunk reads the updated state.
        for (uint16_t i = 0; i < S.get_capacity(); ++i) {
            if (S.is_valid_element(i)) {
                const auto ix = S.get_multidimensional_index(i);
                Sb[uint(ix[1]) * NB + uint(ix[0])] = ST(S[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint16_t i = 0; i < S.get_capacity(); ++i) {
        if (S.is_valid_element(i)) {
            const auto ix = S.get_multidimensional_index(i);
            st[(ulong)uint(ix[1]) * DIM + uint(ix[0])] = S[i];
        }
    }
}

#define GDN_CHUNK_SCAN_KERNEL(NAME, ST)                                        \
kernel void NAME(device bfloat*       qk    [[buffer(0)]],  /* [M, 2*HK*DIM] */     \
                 device bfloat*       w     [[buffer(1)]],  /* [M, H, DIM] */       \
                 device bfloat*       u     [[buffer(2)]],  /* [M, H, DIM] */       \
                 device bfloat*       p     [[buffer(3)]],  /* [M, H, GDN_CHUNK] */ \
                 device const float*  g     [[buffer(4)]],  /* [M, H] */            \
                 device float*        state [[buffer(5)]],  /* [H, DIM, DIM] */     \
                 device bfloat*       out   [[buffer(6)]],  /* [M, H, DIM] */       \
                 constant uint&       M     [[buffer(7)]],                          \
                 constant uint&       H     [[buffer(8)]],                          \
                 constant uint&       vpk   [[buffer(9)]],                          \
                 uint2 tg [[threadgroup_position_in_grid]]) {                       \
    threadgroup ST Sb[DIM * GDN_CHUNK_NB];                                         \
    threadgroup ST Ut[GDN_CHUNK * GDN_CHUNK_NB];                                   \
    threadgroup ST Ut2[GDN_CHUNK * GDN_CHUNK_NB];                                  \
    gdn_chunk_scan_body<GDN_CHUNK_NB, GDN_CHUNK_SG, ST>(qk, w, u, p, g, state, out, M, H, \
                                                        vpk, Sb, Ut, Ut2, tg);     \
}
// The state and pseudo-value copies are bf16: fp16 copies measured the
// same speed with the state's error against the CPU reference 0.0011
// instead of 0.0013 (scale 0.27) and the same acceptance on real text,
// and would cap the state's range at 65 504.
GDN_CHUNK_SCAN_KERNEL(gdn_chunk_scan, bfloat)
#undef GDN_CHUNK_SCAN_KERNEL
#endif  // __METAL_VERSION__ >= 400
