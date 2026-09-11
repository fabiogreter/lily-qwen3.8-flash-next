// Hyper-connection (gated residual) kernels for Qwen3.8-Flash-Next.
// The residual stream is G copies of the hidden width H laid out [rows, G*H].
#include <metal_stdlib>
using namespace metal;

#define TG 256

// Grouped RMSNorm: each of the G segments of a row is normalized on its own
// and scaled by gain w_bias + w[g*H + i]. One threadgroup per (row, g); the
// buffer is addressed as [rows*G, H].
kernel void rmsnorm_grouped_bf16(device const bfloat* x   [[buffer(0)]],
                                 device const bfloat* w   [[buffer(1)]],
                                 device bfloat*       out [[buffer(2)]],
                                 constant uint&       H   [[buffer(3)]],
                                 constant uint&       G   [[buffer(4)]],
                                 constant float&      eps [[buffer(5)]],
                                 constant float&      w_bias [[buffer(6)]],
                                 uint seg  [[threadgroup_position_in_grid]],
                                 uint tid  [[thread_index_in_threadgroup]],
                                 uint sg   [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;

    const ulong base = (ulong)seg * H;
    const uint w0 = (seg % G) * H;
    float acc = 0.0f;
    for (uint i = tid; i < H; i += TG) {
        float v = float(x[base + i]);
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
        inv_rms = rsqrt(total / float(H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < H; i += TG) {
        float gain = w_bias + float(w[w0 + i]);
        out[base + i] = bfloat(float(x[base + i]) * inv_rms * gain);
    }
}

// out = silu(x * inv_scale): the low-rank read gate's activation, whose input
// is divided by the stream count.
kernel void silu_scaled_bf16(device const bfloat* x   [[buffer(0)]],
                             device bfloat*       out [[buffer(1)]],
                             constant float&      inv_scale [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    float v = float(x[gid]) * inv_scale;
    out[gid] = bfloat(v / (1.0f + exp(-v)));
}

// mixed[r, i] = mean_g sigmoid(up[r, g*H+i]) * hn[r, g*H+i]: the read gate
// collapses the G normalized streams into one hidden-width input.
kernel void hc_mix_bf16(device const bfloat* up    [[buffer(0)]],  // [rows, G*H] gate logits
                        device const bfloat* hn    [[buffer(1)]],  // [rows, G*H] normed streams
                        device bfloat*       mixed [[buffer(2)]],  // [rows, H]
                        constant uint&       H     [[buffer(3)]],
                        constant uint&       G     [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const ulong row = (ulong)gid.y * G * H;
    float acc = 0.0f;
    for (uint g = 0; g < G; ++g) {
        const float logit = float(up[row + g * H + i]);
        const float gate = 1.0f / (1.0f + exp(-logit));
        acc += gate * float(hn[row + g * H + i]);
    }
    mixed[(ulong)gid.y * H + i] = bfloat(acc / float(G));
}

// hyper[r, g*H+i] += branch[r, i] * 2*sigmoid(inj[r, g] * inv_g): the write
// gate injects the block output into every stream with its own scalar weight.
kernel void hc_inject_bf16(device bfloat*       hyper  [[buffer(0)]],  // [rows, G*H]
                           device const bfloat* branch [[buffer(1)]],  // [rows, H]
                           device const bfloat* inj    [[buffer(2)]],  // [rows, G] logits
                           constant uint&       H      [[buffer(3)]],
                           constant uint&       G      [[buffer(4)]],
                           constant float&      inv_g  [[buffer(5)]],
                           uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint g = col / H;
    const uint i = col % H;
    const ulong r = gid.y;
    const float logit = float(inj[r * G + g]) * inv_g;
    const float weight = 2.0f / (1.0f + exp(-logit));
    const ulong at = r * G * H + col;
    hyper[at] = bfloat(float(hyper[at]) + float(branch[r * H + i]) * weight);
}

// hyper[r, g*H+i] = x[r, i] for every g: the stream initialization
// (`hidden_states.repeat(1, 1, G)`).
kernel void hc_broadcast_bf16(device const bfloat* x     [[buffer(0)]],  // [rows, H]
                              device bfloat*       hyper [[buffer(1)]],  // [rows, G*H]
                              constant uint&       H     [[buffer(2)]],
                              constant uint&       G     [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const ulong r = gid.y;
    hyper[r * G * H + col] = x[r * H + col % H];
}

// ---------------------------------------------------------------------------
// Fused single-row read gate (the decode graph). The six dispatches of the
// unfused read (grouped norm, down GEMV, inject GEMV, scaled SiLU, up GEMV,
// mix) become two: `hc_read_down_q8` and `hc_read_up_mix_q8_g<G>`. Both are
// laid out for the DRAM-streamed, latency-bound regime of a single-token
// step: many simdgroups in flight, no threadgroup prologue before the first
// weight load, per-stream state in registers.

// Streams per read gate the fused kernels are instantiated for.
#define HC_MAX_G 8

// The four Q8 codes of a word as floats.
static inline float4 hc_unpack_q8(uint word) {
    return float4(float(word & 0xFF), float((word >> 8) & 0xFF),
                  float((word >> 16) & 0xFF), float((word >> 24) & 0xFF));
}

// quant.metal's dot_word_q8 on unpacked codes and an activation in registers.
static inline float hc_dot_q8(float4 q, float s, float b, float4 x) {
    return s * dot(q, x) + b * dot(x, float4(1.0f));
}

// quant.metal's dot_word_q8 with the activation already in registers.
static inline float hc_dot_word_q8(uint word, float s, float b, float4 x) {
    return hc_dot_q8(hc_unpack_q8(word), s, b, x);
}

// down[r] = sum_k dequant(down_w[r, k]) * hn[k] for r < R, and inj[r - R]
// likewise over inject_w for R <= r < R + NI, with
// hn[k] = hyper[k] * inv_rms[g(k)] * (w_bias + norm_w[k]) (rmsnorm_grouped).
// One threadgroup per output row, one simdgroup per stream g (32*G threads):
// simdgroup g dots its stream's uint4 blocks against hyper * gain in f32 and
// sums hyper^2 in the same loop, so the stream RMS costs no prologue; the row
// is sum_g inv_rms[g] * part[g]. Unlike the unfused path, hn is not rounded
// to bf16 before the dot, so down/inj differ from
// gemv_q8_bf16(rmsnorm_grouped_bf16(hyper)) at bf16 rounding level (they are
// the more precise of the two). Threadgroup 0 writes inv_rms[G] for the up
// kernel. Requires H % 16 == 0 (a block never straddles streams) and
// GS % 16 == 0.
kernel void hc_read_down_q8(device const bfloat* hyper    [[buffer(0)]],  // [G*H]
                            device const bfloat* norm_w   [[buffer(1)]],  // [G*H]
                            device const uint*   d_codes  [[buffer(2)]],  // [R, G*H/4]
                            device const bfloat* d_scales [[buffer(3)]],  // [R, G*H/GS]
                            device const bfloat* d_biases [[buffer(4)]],
                            device const uint*   i_codes  [[buffer(5)]],  // [NI, G*H/4]
                            device const bfloat* i_scales [[buffer(6)]],
                            device const bfloat* i_biases [[buffer(7)]],
                            device bfloat*       down     [[buffer(8)]],  // [R]
                            device bfloat*       inj      [[buffer(9)]],  // [NI]
                            device float*        inv_rms  [[buffer(10)]], // [G]
                            constant uint&       H        [[buffer(11)]],
                            constant uint&       G        [[buffer(12)]],
                            constant uint&       R        [[buffer(13)]],
                            constant uint&       NI       [[buffer(14)]],
                            constant uint&       GS       [[buffer(15)]],
                            constant float&      eps      [[buffer(16)]],
                            constant float&      w_bias   [[buffer(17)]],
                            uint row  [[threadgroup_position_in_grid]],
                            uint tid  [[thread_index_in_threadgroup]],
                            uint g    [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[HC_MAX_G];
    threadgroup float rms[HC_MAX_G];
    if (row >= R + NI) {
        return;  // whole threadgroup
    }
    const bool is_inj = row >= R;
    device const uint* codes = is_inj ? i_codes : d_codes;
    device const bfloat* scales = is_inj ? i_scales : d_scales;
    device const bfloat* biases = is_inj ? i_biases : d_biases;
    const uint r = is_inj ? row - R : row;

    const uint K = G * H;
    const uint words = K / 4;
    const uint bpg = GS / 16;
    const uint groups = K / GS;
    const uint stream_blocks = H / 16;
    device const uint4* wrow = (device const uint4*)(codes + (ulong)r * words);
    device const bfloat4* xv = (device const bfloat4*)hyper;
    device const bfloat4* nv = (device const bfloat4*)norm_w;
    float acc_w = 0.0f;
    float acc_x2 = 0.0f;
    const uint i_end = (g + 1) * stream_blocks;
    for (uint i = g * stream_blocks + lane; i < i_end; i += 32) {
        const uint q = i / bpg;
        const float s = float(scales[r * groups + q]);
        const float b = float(biases[r * groups + q]);
        const uint4 w4 = wrow[i];
        const uint ws[4] = {w4.x, w4.y, w4.z, w4.w};
        for (uint j = 0; j < 4; ++j) {
            const float4 x = float4(xv[4 * i + j]);
            const float4 xn = x * (w_bias + float4(nv[4 * i + j]));
            acc_w += hc_dot_word_q8(ws[j], s, b, xn);
            acc_x2 += dot(x, x);
        }
    }
    acc_w = simd_sum(acc_w);
    acc_x2 = simd_sum(acc_x2);
    if (lane == 0) {
        const float inv = rsqrt(acc_x2 / float(H) + eps);
        part[g] = acc_w * inv;
        rms[g] = inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < G; ++s) {
            total += part[s];
        }
        if (is_inj) {
            inj[r] = bfloat(total);
        } else {
            down[r] = bfloat(total);
        }
    }
    if (row == 0 && tid < G) {
        inv_rms[tid] = rms[tid];
    }
}

// mixed[i] = mean_g sigmoid(up[g*H + i]) * hn[g*H + i], with
// up[g*H + i] = sum_k dequant(up_w[g*H + i, k]) * act[k],
// act = silu(down * inv_g) (silu_scaled_bf16's expression, computed per lane
// for the 16 elements of its own block) and hn recomputed from hyper,
// inv_rms and the norm weight as rmsnorm_grouped_bf16 does. One simdgroup
// per output column i owning its GG rows of `up_w`; each row's loop is
// GEMV_Q8_BODY's (lanes over uint4 blocks, simd_sum) and the gate epilogue
// hc_mix_bf16's, so given the same `down` and `inv_rms` the result is
// bit-identical to silu_scaled + gemv_q8 + hc_mix. GG is compile time so the
// per-stream arrays stay in registers (a runtime count spilled them and cost
// a third of the kernel). The threadgroup size sets the columns per
// threadgroup (threads / 32).
template <uint GG>
static inline void hc_read_up_mix_q8_body(device const uint* u_codes,
                                          device const bfloat* u_scales,
                                          device const bfloat* u_biases,
                                          device const bfloat* down,
                                          device const bfloat* hyper,
                                          device const bfloat* norm_w,
                                          device const float* inv_rms,
                                          device bfloat* mixed,
                                          uint H, uint R, uint GS, float inv_g, float w_bias,
                                          uint col, uint lane) {
    const uint words = R / 4;
    const uint blocks = words / 4;
    const uint bpg = GS / 16;
    const uint groups = R / GS;
    device const bfloat4* dv = (device const bfloat4*)down;

    float sum[GG];
    for (uint g = 0; g < GG; ++g) {
        sum[g] = 0.0f;
    }
    for (uint i = lane; i < blocks; i += 32) {
        const uint q = i / bpg;
        uint4 w4[GG];
        for (uint g = 0; g < GG; ++g) {
            const ulong row = (ulong)g * H + col;
            w4[g] = ((device const uint4*)(u_codes + row * words))[i];
        }
        float4 act[4];
        for (uint j = 0; j < 4; ++j) {
            const float4 v = float4(dv[4 * i + j]) * inv_g;
            act[j] = float4(bfloat4(v / (1.0f + exp(-v))));
        }
        for (uint g = 0; g < GG; ++g) {
            const ulong row = (ulong)g * H + col;
            const float s = float(u_scales[row * groups + q]);
            const float b = float(u_biases[row * groups + q]);
            sum[g] += hc_dot_word_q8(w4[g].x, s, b, act[0])
                + hc_dot_word_q8(w4[g].y, s, b, act[1])
                + hc_dot_word_q8(w4[g].z, s, b, act[2])
                + hc_dot_word_q8(w4[g].w, s, b, act[3]);
        }
    }
    float acc = 0.0f;
    for (uint g = 0; g < GG; ++g) {
        const float logit = float(bfloat(simd_sum(sum[g])));
        const float gate = 1.0f / (1.0f + exp(-logit));
        const uint e = g * H + col;
        const float hn = float(bfloat(float(hyper[e]) * inv_rms[g] * (w_bias + float(norm_w[e]))));
        acc += gate * hn;
    }
    if (lane == 0) {
        mixed[col] = bfloat(acc / float(GG));
    }
}

#define HC_READ_UP_MIX_KERNEL(GG)                                                    \
    kernel void hc_read_up_mix_q8_g##GG(                                             \
                     device const uint*   u_codes  [[buffer(0)]],  /* [G*H, R/4] */  \
                     device const bfloat* u_scales [[buffer(1)]],  /* [G*H, R/GS] */ \
                     device const bfloat* u_biases [[buffer(2)]],                    \
                     device const bfloat* down     [[buffer(3)]],  /* [R] */         \
                     device const bfloat* hyper    [[buffer(4)]],  /* [G*H] */       \
                     device const bfloat* norm_w   [[buffer(5)]],  /* [G*H] */       \
                     device const float*  inv_rms  [[buffer(6)]],  /* [G] */         \
                     device bfloat*       mixed    [[buffer(7)]],  /* [H] */         \
                     constant uint&       H        [[buffer(8)]],                    \
                     constant uint&       R        [[buffer(9)]],                    \
                     constant uint&       GS       [[buffer(10)]],                   \
                     constant float&      inv_g    [[buffer(11)]],                   \
                     constant float&      w_bias   [[buffer(12)]],                   \
                     uint tg     [[threadgroup_position_in_grid]],                   \
                     uint tgsize [[threads_per_threadgroup]],                        \
                     uint sg     [[simdgroup_index_in_threadgroup]],                 \
                     uint lane   [[thread_index_in_simdgroup]]) {                    \
        const uint col = tg * (tgsize / 32) + sg;                                    \
        if (col >= H) {                                                              \
            return;  /* whole simdgroup */                                           \
        }                                                                            \
        hc_read_up_mix_q8_body<GG>(u_codes, u_scales, u_biases, down, hyper, norm_w, \
                                   inv_rms, mixed, H, R, GS, inv_g, w_bias, col,     \
                                   lane);                                            \
    }

HC_READ_UP_MIX_KERNEL(1)
HC_READ_UP_MIX_KERNEL(2)
HC_READ_UP_MIX_KERNEL(3)
HC_READ_UP_MIX_KERNEL(4)
HC_READ_UP_MIX_KERNEL(5)
HC_READ_UP_MIX_KERNEL(6)
HC_READ_UP_MIX_KERNEL(7)
HC_READ_UP_MIX_KERNEL(8)

// ---------------------------------------------------------------------------
// Fused small-batch read gate (the verify pass and the draft head, MB <= 4
// rows). Same mapping, math and per-lane operation order as the single-row
// kernels above, so row m of the output is bit-identical to hc_read_down_q8 /
// hc_read_up_mix_q8 run on that row alone (up to the compiler's fast-math
// scheduling of the longer unrolled bodies; see the tests). Each lane loads
// its weight block, scale and bias once and loops the MB rows in registers;
// the skinny GEMMs this replaces already read the weights once per row block
// too, so the win is the dispatch count (six kernels to two) and the fused
// norm, not weight traffic. MB is a template parameter for the same reason GG
// is: the per-row accumulators must stay in registers. One difference from
// the single-row pair: the down kernel also writes the scaled SiLU of `down`
// (`act`, silu_scaled_bf16's expression on the bf16 logit), and the up kernel
// reads it instead of recomputing it per column, which for MB rows is the
// up kernel's dominant ALU cost (m=4: 22 to 12 us in the microbench). The
// value is the one the single-row up kernel computes in registers.

// Rows per dispatch the small-batch kernels are instantiated for.
#define HC_MAX_MB 4

// hc_read_down_q8 over MB rows for one output row `r` of one matrix
// ([N, G*H] Q8): hyper is [MB, G*H], out [MB, N], act [MB, N] (down rows
// only), inv_rms [MB, G]. `part`/`rms` are threadgroup scratch of
// MB * HC_MAX_G floats each.
template <uint MB>
static inline void hc_read_down_q8_rows_body(device const bfloat* hyper,
                                             device const bfloat* norm_w,
                                             device const uint* codes,
                                             device const bfloat* scales,
                                             device const bfloat* biases,
                                             device bfloat* out,
                                             device bfloat* act,
                                             device float* inv_rms,
                                             uint H, uint G, uint N, uint GS, float eps,
                                             float w_bias, float inv_g, uint r, bool is_down,
                                             uint tid, uint g, uint lane,
                                             threadgroup float* part, threadgroup float* rms) {
    const uint K = G * H;
    const uint words = K / 4;
    const uint bpg = GS / 16;
    const uint groups = K / GS;
    const uint stream_blocks = H / 16;
    device const uint4* wrow = (device const uint4*)(codes + (ulong)r * words);
    device const bfloat4* nv = (device const bfloat4*)norm_w;
    float acc_w[MB];
    float acc_x2[MB];
    for (uint m = 0; m < MB; ++m) {
        acc_w[m] = 0.0f;
        acc_x2[m] = 0.0f;
    }
    const uint i_end = (g + 1) * stream_blocks;
    for (uint i = g * stream_blocks + lane; i < i_end; i += 32) {
        const uint q = i / bpg;
        const float s = float(scales[r * groups + q]);
        const float b = float(biases[r * groups + q]);
        const uint4 w4 = wrow[i];
        const uint ws[4] = {w4.x, w4.y, w4.z, w4.w};
        for (uint j = 0; j < 4; ++j) {
            // Unpacked once per word, not once per row: at MB rows the
            // kernel is ALU-bound and the unpack is half of the dot's ops.
            const float4 qf = hc_unpack_q8(ws[j]);
            const float4 gain = w_bias + float4(nv[4 * i + j]);
            for (uint m = 0; m < MB; ++m) {
                device const bfloat4* xv = (device const bfloat4*)(hyper + (ulong)m * K);
                const float4 x = float4(xv[4 * i + j]);
                const float4 xn = x * gain;
                acc_w[m] += hc_dot_q8(qf, s, b, xn);
                acc_x2[m] += dot(x, x);
            }
        }
    }
    for (uint m = 0; m < MB; ++m) {
        acc_w[m] = simd_sum(acc_w[m]);
        acc_x2[m] = simd_sum(acc_x2[m]);
    }
    if (lane == 0) {
        for (uint m = 0; m < MB; ++m) {
            const float inv = rsqrt(acc_x2[m] / float(H) + eps);
            part[m * HC_MAX_G + g] = acc_w[m] * inv;
            rms[m * HC_MAX_G + g] = inv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint m = 0; m < MB; ++m) {
            float total = 0.0f;
            for (uint s = 0; s < G; ++s) {
                total += part[m * HC_MAX_G + s];
            }
            const bfloat logit = bfloat(total);
            out[(ulong)m * N + r] = logit;
            if (is_down) {
                // silu_scaled_bf16 on the bf16 logit.
                const float v = float(logit) * inv_g;
                act[(ulong)m * N + r] = bfloat(v / (1.0f + exp(-v)));
            }
        }
    }
    // The down row 0 threadgroup publishes the stream RMS for the up kernel.
    if (is_down && r == 0 && tid < MB * G) {
        inv_rms[tid] = rms[(tid / G) * HC_MAX_G + tid % G];
    }
}

#define HC_READ_DOWN_ROWS_KERNEL(MB)                                                 \
    kernel void hc_read_down_q8_m##MB(                                               \
                     device const bfloat* hyper    [[buffer(0)]],  /* [MB, G*H] */   \
                     device const bfloat* norm_w   [[buffer(1)]],  /* [G*H] */       \
                     device const uint*   d_codes  [[buffer(2)]],  /* [R, G*H/4] */  \
                     device const bfloat* d_scales [[buffer(3)]],  /* [R, G*H/GS] */ \
                     device const bfloat* d_biases [[buffer(4)]],                    \
                     device const uint*   i_codes  [[buffer(5)]],  /* [NI, G*H/4] */ \
                     device const bfloat* i_scales [[buffer(6)]],                    \
                     device const bfloat* i_biases [[buffer(7)]],                    \
                     device bfloat*       down     [[buffer(8)]],  /* [MB, R] */     \
                     device bfloat*       inj      [[buffer(9)]],  /* [MB, NI] */    \
                     device float*        inv_rms  [[buffer(10)]], /* [MB, G] */     \
                     device bfloat*       act      [[buffer(11)]], /* [MB, R] */     \
                     constant uint&       H        [[buffer(12)]],                   \
                     constant uint&       G        [[buffer(13)]],                   \
                     constant uint&       R        [[buffer(14)]],                   \
                     constant uint&       NI       [[buffer(15)]],                   \
                     constant uint&       GS       [[buffer(16)]],                   \
                     constant float&      eps      [[buffer(17)]],                   \
                     constant float&      w_bias   [[buffer(18)]],                   \
                     constant float&      inv_g    [[buffer(19)]],                   \
                     uint row  [[threadgroup_position_in_grid]],                     \
                     uint tid  [[thread_index_in_threadgroup]],                      \
                     uint g    [[simdgroup_index_in_threadgroup]],                   \
                     uint lane [[thread_index_in_simdgroup]]) {                      \
        threadgroup float part[MB * HC_MAX_G];                                       \
        threadgroup float rms[MB * HC_MAX_G];                                        \
        if (row >= R + NI) {                                                         \
            return;  /* whole threadgroup */                                         \
        }                                                                            \
        if (row >= R) {                                                              \
            hc_read_down_q8_rows_body<MB>(hyper, norm_w, i_codes, i_scales, i_biases, \
                                          inj, act, inv_rms, H, G, NI, GS, eps,      \
                                          w_bias, inv_g, row - R, false, tid, g,     \
                                          lane, part, rms);                          \
        } else {                                                                     \
            hc_read_down_q8_rows_body<MB>(hyper, norm_w, d_codes, d_scales, d_biases, \
                                          down, act, inv_rms, H, G, R, GS, eps,      \
                                          w_bias, inv_g, row, true, tid, g, lane,    \
                                          part, rms);                                \
        }                                                                            \
    }

HC_READ_DOWN_ROWS_KERNEL(1)
HC_READ_DOWN_ROWS_KERNEL(2)
HC_READ_DOWN_ROWS_KERNEL(3)
HC_READ_DOWN_ROWS_KERNEL(4)

// hc_read_up_mix_q8_body over MB rows with the activation precomputed: act
// is [MB, R] (silu(down / G) in bf16, from hc_read_down_q8_m<MB>), hyper
// [MB, GG*H], inv_rms [MB, GG], mixed [MB, H]. The GG weight blocks of the
// column are loaded once per lane and dotted against every row's activation.
template <uint GG, uint MB>
static inline void hc_read_up_mix_q8_rows_body(device const uint* u_codes,
                                               device const bfloat* u_scales,
                                               device const bfloat* u_biases,
                                               device const bfloat* act,
                                               device const bfloat* hyper,
                                               device const bfloat* norm_w,
                                               device const float* inv_rms,
                                               device bfloat* mixed,
                                               uint H, uint R, uint GS, float w_bias,
                                               uint col, uint lane) {
    const uint K = GG * H;
    const uint words = R / 4;
    const uint blocks = words / 4;
    const uint bpg = GS / 16;
    const uint groups = R / GS;

    float sum[GG * MB];
    for (uint e = 0; e < GG * MB; ++e) {
        sum[e] = 0.0f;
    }
    for (uint i = lane; i < blocks; i += 32) {
        const uint q = i / bpg;
        for (uint g = 0; g < GG; ++g) {
            const ulong row = (ulong)g * H + col;
            const uint4 w4 = ((device const uint4*)(u_codes + row * words))[i];
            const float s = float(u_scales[row * groups + q]);
            const float b = float(u_biases[row * groups + q]);
            // Unpacked once per stream, not once per row: at MB rows the
            // kernel is ALU-bound and the unpack is half of the dot's ops.
            // The activation reloads per stream hit L1 (MB * R bf16).
            const float4 q0 = hc_unpack_q8(w4.x);
            const float4 q1 = hc_unpack_q8(w4.y);
            const float4 q2 = hc_unpack_q8(w4.z);
            const float4 q3 = hc_unpack_q8(w4.w);
            for (uint m = 0; m < MB; ++m) {
                device const bfloat4* av = (device const bfloat4*)(act + (ulong)m * R);
                sum[g * MB + m] += hc_dot_q8(q0, s, b, float4(av[4 * i]))
                    + hc_dot_q8(q1, s, b, float4(av[4 * i + 1]))
                    + hc_dot_q8(q2, s, b, float4(av[4 * i + 2]))
                    + hc_dot_q8(q3, s, b, float4(av[4 * i + 3]));
            }
        }
    }
    for (uint m = 0; m < MB; ++m) {
        float acc = 0.0f;
        for (uint g = 0; g < GG; ++g) {
            const float logit = float(bfloat(simd_sum(sum[g * MB + m])));
            const float gate = 1.0f / (1.0f + exp(-logit));
            const uint e = g * H + col;
            const float hn = float(bfloat(float(hyper[(ulong)m * K + e]) * inv_rms[m * GG + g]
                                          * (w_bias + float(norm_w[e]))));
            acc += gate * hn;
        }
        if (lane == 0) {
            mixed[(ulong)m * H + col] = bfloat(acc / float(GG));
        }
    }
}

#define HC_READ_UP_MIX_ROWS_KERNEL(GG, MB)                                           \
    kernel void hc_read_up_mix_q8_g##GG##_m##MB(                                     \
                     device const uint*   u_codes  [[buffer(0)]],  /* [G*H, R/4] */  \
                     device const bfloat* u_scales [[buffer(1)]],  /* [G*H, R/GS] */ \
                     device const bfloat* u_biases [[buffer(2)]],                    \
                     device const bfloat* act      [[buffer(3)]],  /* [MB, R] */     \
                     device const bfloat* hyper    [[buffer(4)]],  /* [MB, G*H] */   \
                     device const bfloat* norm_w   [[buffer(5)]],  /* [G*H] */       \
                     device const float*  inv_rms  [[buffer(6)]],  /* [MB, G] */     \
                     device bfloat*       mixed    [[buffer(7)]],  /* [MB, H] */     \
                     constant uint&       H        [[buffer(8)]],                    \
                     constant uint&       R        [[buffer(9)]],                    \
                     constant uint&       GS       [[buffer(10)]],                   \
                     constant float&      w_bias   [[buffer(11)]],                   \
                     uint tg     [[threadgroup_position_in_grid]],                   \
                     uint tgsize [[threads_per_threadgroup]],                        \
                     uint sg     [[simdgroup_index_in_threadgroup]],                 \
                     uint lane   [[thread_index_in_simdgroup]]) {                    \
        const uint col = tg * (tgsize / 32) + sg;                                    \
        if (col >= H) {                                                              \
            return;  /* whole simdgroup */                                           \
        }                                                                            \
        hc_read_up_mix_q8_rows_body<GG, MB>(u_codes, u_scales, u_biases, act, hyper,  \
                                            norm_w, inv_rms, mixed, H, R, GS, w_bias, \
                                            col, lane);                              \
    }

#define HC_READ_UP_MIX_ROWS_KERNELS(GG) \
    HC_READ_UP_MIX_ROWS_KERNEL(GG, 1)   \
    HC_READ_UP_MIX_ROWS_KERNEL(GG, 2)   \
    HC_READ_UP_MIX_ROWS_KERNEL(GG, 3)   \
    HC_READ_UP_MIX_ROWS_KERNEL(GG, 4)

HC_READ_UP_MIX_ROWS_KERNELS(1)
HC_READ_UP_MIX_ROWS_KERNELS(2)
HC_READ_UP_MIX_ROWS_KERNELS(3)
HC_READ_UP_MIX_ROWS_KERNELS(4)
HC_READ_UP_MIX_ROWS_KERNELS(5)
HC_READ_UP_MIX_ROWS_KERNELS(6)
HC_READ_UP_MIX_ROWS_KERNELS(7)
HC_READ_UP_MIX_ROWS_KERNELS(8)
