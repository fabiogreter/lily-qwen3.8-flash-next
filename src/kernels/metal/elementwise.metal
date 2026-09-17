// Elementwise, embedding, and argmax kernels.
#include <metal_stdlib>
using namespace metal;

// Word copy for buffer-to-buffer state restores inside a compute pass.
kernel void copy_u32(device const uint* src [[buffer(0)]],
                     device uint*       dst [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    dst[gid] = src[gid];
}

// XOR of every word of `src` into out[0] (one atomic per simdgroup): a
// read-bandwidth probe that touches every byte.
kernel void checksum_u32(device const uint4* src   [[buffer(0)]],
                         device atomic_uint* out   [[buffer(1)]],
                         constant uint&      n4    [[buffer(2)]],
                         uint gid  [[thread_position_in_grid]],
                         uint grid [[threads_per_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint acc = 0u;
    for (uint i = gid; i < n4; i += grid) {
        const uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    acc = simd_xor(acc);
    if (lane == 0) {
        atomic_fetch_xor_explicit(out, acc, memory_order_relaxed);
    }
}

kernel void add_bf16(device const bfloat* a   [[buffer(0)]],
                     device const bfloat* b   [[buffer(1)]],
                     device bfloat*       out [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    out[gid] = bfloat(float(a[gid]) + float(b[gid]));
}

// SwiGLU: out = silu(gate) * up.
kernel void silu_mul_bf16(device const bfloat* gate [[buffer(0)]],
                          device const bfloat* up   [[buffer(1)]],
                          device bfloat*       out  [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    float g = float(gate[gid]);
    float s = g / (1.0f + exp(-g));
    out[gid] = bfloat(s * float(up[gid]));
}

// out[r, c] = silu(gu[r, c]) * gu[r, N + c] over a [m, 2N] gate|up stack
// (silu_mul_bf16 without splitting the stack first).
kernel void silu_mul_gu_rows_bf16(device const bfloat* gu  [[buffer(0)]],
                                  device bfloat*       out [[buffer(1)]],
                                  constant uint&       N   [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint r = gid.y;
    if (c >= N) {
        return;
    }
    const ulong base = (ulong)r * 2 * N;
    float g = float(gu[base + c]);
    float s = g / (1.0f + exp(-g));
    out[(ulong)r * N + c] = bfloat(s * float(gu[base + N + c]));
}

// Attention output gate: out = x * sigmoid(gate).
kernel void sigmoid_mul_bf16(device const bfloat* gate [[buffer(0)]],
                             device const bfloat* x    [[buffer(1)]],
                             device bfloat*       out  [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    float g = float(gate[gid]);
    out[gid] = bfloat(float(x[gid]) / (1.0f + exp(-g)));
}

// Splits [m, n_total] into up to four contiguous column segments.
kernel void split_cols_bf16(device const bfloat* src [[buffer(0)]],
                            device bfloat*       d0  [[buffer(1)]],
                            device bfloat*       d1  [[buffer(2)]],
                            device bfloat*       d2  [[buffer(3)]],
                            device bfloat*       d3  [[buffer(4)]],
                            constant uint4&      w   [[buffer(5)]],
                            constant uint&  n_total  [[buffer(6)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x;
    const uint row = gid.y;
    const bfloat v = src[(ulong)row * n_total + col];
    if (col < w.x) {
        d0[(ulong)row * w.x + col] = v;
        return;
    }
    col -= w.x;
    if (col < w.y) {
        d1[(ulong)row * w.y + col] = v;
        return;
    }
    col -= w.y;
    if (col < w.z) {
        d2[(ulong)row * w.z + col] = v;
        return;
    }
    col -= w.z;
    if (col < w.w) {
        d3[(ulong)row * w.w + col] = v;
    }
}

// Embedding lookup: copies row `row` of a [rows, H] bf16 table.
kernel void gather_row_bf16(device const bfloat* table [[buffer(0)]],
                            device bfloat*       out   [[buffer(1)]],
                            constant uint&       row   [[buffer(2)]],
                            constant uint&       H     [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    out[gid] = table[(ulong)row * H + gid];
}

// Batched embedding lookup: out[i, :] = table[ids[i], :].
kernel void gather_rows_bf16(device const bfloat* table [[buffer(0)]],
                             device const uint*   ids   [[buffer(1)]],
                             device bfloat*       out   [[buffer(2)]],
                             constant uint&       H     [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    uint i = gid / H;
    uint d = gid % H;
    out[gid] = table[(ulong)ids[i] * H + d];
}

// gather_rows_bf16 by eight elements (one uint4) per thread, for rows whose
// width is a multiple of 8 and 16-byte aligned buffers.
kernel void gather_rows_bf16_x8(device const uint4* table [[buffer(0)]],
                                device const uint*  ids   [[buffer(1)]],
                                device uint4*       out   [[buffer(2)]],
                                constant uint&      W     [[buffer(3)]],  // H / 8
                                uint gid [[thread_position_in_grid]]) {
    const uint i = gid / W;
    const uint w = gid - i * W;
    out[gid] = table[(ulong)ids[i] * W + w];
}

#define ARGMAX_TG 256

struct ArgMaxPair {
    float v;
    uint  i;
};

static inline void argmax_merge(thread float& bv, thread uint& bi, float v, uint i) {
    if (v > bv || (v == bv && i < bi)) {
        bv = v;
        bi = i;
    }
}

kernel void argmax_f32_partial(device const float* x        [[buffer(0)]],
                               device ArgMaxPair*  partials [[buffer(1)]],
                               constant uint&      n        [[buffer(2)]],
                               constant uint&      chunk    [[buffer(3)]],
                               uint g    [[threadgroup_position_in_grid]],
                               uint tid  [[thread_index_in_threadgroup]],
                               uint sg   [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part_v[ARGMAX_TG / 32];
    threadgroup uint  part_i[ARGMAX_TG / 32];

    float bv = -INFINITY;
    uint  bi = 0;
    uint end = min((g + 1) * chunk, n);
    for (uint i = g * chunk + tid; i < end; i += ARGMAX_TG) {
        argmax_merge(bv, bi, x[i], i);
    }
    for (uint off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(bv, off);
        uint  oi = simd_shuffle_down(bi, off);
        argmax_merge(bv, bi, ov, oi);
    }
    if (lane == 0) {
        part_v[sg] = bv;
        part_i[sg] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint s = 1; s < ARGMAX_TG / 32; ++s) {
            argmax_merge(bv, bi, part_v[s], part_i[s]);
        }
        partials[g].v = bv;
        partials[g].i = bi;
    }
}

// Reduces partial argmax results to one index.
kernel void argmax_f32_final(device const ArgMaxPair* partials [[buffer(0)]],
                             device uint*             out      [[buffer(1)]],
                             constant uint&           groups   [[buffer(2)]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part_v[ARGMAX_TG / 32];
    threadgroup uint  part_i[ARGMAX_TG / 32];

    float bv = -INFINITY;
    uint  bi = 0;
    for (uint g = tid; g < groups; g += ARGMAX_TG) {
        argmax_merge(bv, bi, partials[g].v, partials[g].i);
    }
    for (uint off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(bv, off);
        uint  oi = simd_shuffle_down(bi, off);
        argmax_merge(bv, bi, ov, oi);
    }
    if (lane == 0) {
        part_v[sg] = bv;
        part_i[sg] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint s = 1; s < ARGMAX_TG / 32; ++s) {
            argmax_merge(bv, bi, part_v[s], part_i[s]);
        }
        out[0] = bi;
    }
}

// erf for the exact GELU: Metal's standard library has none. Abramowitz and
// Stegun 7.1.26, absolute error below 1.5e-7, three orders under a bf16 ulp
// of any GELU output.
static inline float erf_f32(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float a = abs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float poly = t * (0.254829592f
        + t * (-0.284496736f
        + t * (1.421413741f
        + t * (-1.453152027f + t * 1.061405429f))));
    return sign * (1.0f - poly * exp(-a * a));
}

// x = gelu(x) in place. `erf_form` 0 is the tanh approximation
// (`gelu_pytorch_tanh`, the vision tower's blocks), 1 the exact erf form
// (`nn.GELU()`, its merger); the activation runs in fp32.
kernel void gelu_bf16(device bfloat*  x        [[buffer(0)]],
                      constant uint&  erf_form [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    const float v = float(x[gid]);
    float g;
    if (erf_form != 0u) {
        g = 0.5f * v * (1.0f + erf_f32(v * 0.70710678118654752f));
    } else {
        // 0.5 x (1 + tanh(u)) written as x * sigmoid(2u): the same function
        // without the `1 + tanh` cancellation in the negative tail.
        const float u = 0.79788456080286536f * (v + 0.044715f * v * v * v);
        g = v / (1.0f + exp(-2.0f * u));
    }
    x[gid] = bfloat(g);
}
