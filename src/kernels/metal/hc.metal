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

// quant.metal's dot_word_q8 with the activation already in registers.
static inline float hc_dot_word_q8(uint word, float s, float b, float4 x) {
    float4 q = float4(float(word & 0xFF), float((word >> 8) & 0xFF),
                      float((word >> 16) & 0xFF), float((word >> 24) & 0xFF));
    return s * dot(q, x) + b * dot(x, float4(1.0f));
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
