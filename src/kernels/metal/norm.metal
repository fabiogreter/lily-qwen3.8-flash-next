// Row-wise weighted RMSNorm with fp32 reduction.
#include <metal_stdlib>
using namespace metal;

#define TG 256

// H=2048 fused residual add + RMSNorm; rounded residuals are written to x.
constant constexpr uint RESIDUAL_H = 2048;

kernel void add_rmsnorm_bf16(
                             device bfloat*       x   [[buffer(0)]],
                             device const bfloat* b   [[buffer(1)]],
                             device const bfloat* w   [[buffer(2)]],
                             device bfloat*       out [[buffer(3)]],
                             constant float&      eps [[buffer(4)]],
                             constant float&      w_bias [[buffer(5)]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;
    bfloat cached[8];

    float acc = 0.0f;
    for (uint j = 0; j < 8; ++j) {
        const uint i = tid + j * TG;
        const bfloat y = bfloat(float(x[row * RESIDUAL_H + i]) +
                                float(b[row * RESIDUAL_H + i]));
        cached[j] = y;
        x[row * RESIDUAL_H + i] = y;
        const float v = float(y);
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
        inv_rms = rsqrt(total / float(RESIDUAL_H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 8; ++j) {
        const uint i = tid + j * TG;
        const float gain = w_bias + float(w[i]);
        out[row * RESIDUAL_H + i] = bfloat(float(cached[j]) * inv_rms * gain);
    }
}

// RMSNorm with gain w_bias + w and FP32 accumulation.
kernel void rmsnorm_bf16(device const bfloat* x   [[buffer(0)]],
                         device const bfloat* w   [[buffer(1)]],
                         device bfloat*       out [[buffer(2)]],
                         constant uint&       H   [[buffer(3)]],
                         constant float&      eps [[buffer(4)]],
                         constant float&      w_bias [[buffer(5)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint sg   [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;

    float acc = 0.0f;
    for (uint i = tid; i < H; i += TG) {
        float v = float(x[row * H + i]);
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
        float gain = w_bias + float(w[i]);
        out[row * H + i] = bfloat(float(x[row * H + i]) * inv_rms * gain);
    }
}

// LayerNorm with weight and bias over the last dimension, statistics in
// fp32 in two passes (mean, then the sum of squared deviations): the vision
// tower's pre-merger rows carry values up to 1e4 with a std of 1e2, where
// E[x^2] - mean^2 in fp32 loses digits. One threadgroup per row.
kernel void layernorm_bf16(device const bfloat* x   [[buffer(0)]],
                           device const bfloat* w   [[buffer(1)]],
                           device const bfloat* b   [[buffer(2)]],
                           device bfloat*       out [[buffer(3)]],
                           constant uint&       H   [[buffer(4)]],
                           constant float&      eps [[buffer(5)]],
                           uint row  [[threadgroup_position_in_grid]],
                           uint tid  [[thread_index_in_threadgroup]],
                           uint sg   [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float stat;

    const ulong base = (ulong)row * H;
    float acc = 0.0f;
    for (uint i = tid; i < H; i += TG) {
        acc += float(x[base + i]);
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
        stat = total / float(H);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = stat;

    float sq = 0.0f;
    for (uint i = tid; i < H; i += TG) {
        const float d = float(x[base + i]) - mean;
        sq += d * d;
    }
    sq = simd_sum(sq);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        partial[sg] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < TG / 32; ++i) {
            total += partial[i];
        }
        stat = rsqrt(total / float(H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_std = stat;

    for (uint i = tid; i < H; i += TG) {
        const float v = (float(x[base + i]) - mean) * inv_std;
        out[base + i] = bfloat(v * float(w[i]) + float(b[i]));
    }
}
