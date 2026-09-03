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
