// The vision tower's own kernels (docs/vision-support-plan.md item 3,
// tools/reference/VISION.md "The tower"): the position-embedding blend, the
// 2-D rotary over the fused qkv rows, and full bidirectional attention over
// every patch of one image. LayerNorm lives in norm.metal, the bias and GELU
// passes in elementwise.metal, the projections are the dense bf16 GEMM.
#include <metal_stdlib>
using namespace metal;

// Decodes a block-major patch row into its grid coordinates: row
// r = ((bh * (gw / 2) + bw) * 2 + ih) * 2 + iw holds the patch at grid
// position (2 bh + ih, 2 bw + iw). VISION.md "Patchify".
static inline uint2 vision_grid_coords(uint r, uint gw) {
    const uint blocks_w = gw / 2;
    const uint iw = r % 2;
    const uint ih = (r / 2) % 2;
    const uint bw = (r / 4) % blocks_w;
    const uint bh = r / (4 * blocks_w);
    return uint2(bh * 2 + ih, bw * 2 + iw);
}

// One axis of the align_corners bilinear resample of the `side` x `side`
// position table to an axis of length `n`: taps floor(src) and floor(src) + 1
// clamped into the table, weights 1 - |src - tap|. `src = i * (side - 1) /
// max(n - 1, 1)` in float32, the closed form of linspace(0, side-1, n)[i]
// that vision_utils._interpolation_axis_taps_weights uses.
static inline void vision_pos_axis_taps(uint i, uint n, uint side,
                                        thread uint2& taps, thread float2& w) {
    const float src = float(i) * float(side - 1) / float(max(n - 1u, 1u));
    const float f = floor(src);
    const int t0 = int(f);
    taps = uint2(uint(clamp(t0, 0, int(side) - 1)), uint(clamp(t0 + 1, 0, int(side) - 1)));
    const float d0 = abs(src - f);
    const float d1 = abs(src - f - 1.0f);
    w = float2(max(1.0f - d0, 0.0f), max(1.0f - d1, 0.0f));
}

// x[r, c] = bf16(x[r, c] + bf16(sum_k w_k * table[tap_k, c])) over the
// patch-embedding output x [N, H] (the GEMM with its bias epilogue): the
// resampled position embedding, blended in float32 from the bf16 table and
// cast to bf16 before the add, as the reference does
// (`pos_embeds.to(hidden_states.dtype)`). Taps and weights depend only on the
// patch's grid coordinates, so every thread of a row recomputes them; that is
// cheaper than a host-side table and needs no extra buffer. Grid (H, N).
kernel void vision_patch_pos_bf16(device bfloat*       x     [[buffer(0)]],
                                  device const bfloat* table [[buffer(1)]],
                                  constant uint&       H     [[buffer(2)]],
                                  constant uint&       gh    [[buffer(3)]],
                                  constant uint&       gw    [[buffer(4)]],
                                  constant uint&       side  [[buffer(5)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint r = gid.y;
    if (c >= H) {
        return;
    }
    const uint2 rc = vision_grid_coords(r, gw);
    uint2 ht, wt;
    float2 hw, ww;
    vision_pos_axis_taps(rc.x, gh, side, ht, hw);
    vision_pos_axis_taps(rc.y, gw, side, wt, ww);
    float blend = 0.0f;
    blend += hw.x * ww.x * float(table[(ulong)(ht.x * side + wt.x) * H + c]);
    blend += hw.x * ww.y * float(table[(ulong)(ht.x * side + wt.y) * H + c]);
    blend += hw.y * ww.x * float(table[(ulong)(ht.y * side + wt.x) * H + c]);
    blend += hw.y * ww.y * float(table[(ulong)(ht.y * side + wt.y) * H + c]);
    const ulong idx = (ulong)r * H + c;
    x[idx] = bfloat(float(x[idx]) + float(bfloat(blend)));
}

// The 2-D rotary in place over the qkv Linear's output [N, 3H] (q | k | v,
// each `heads` heads of `D`). Each thread owns one rotate_half pair
// (i, i + D/2) of one head of one row: for q and k the rotation in float32
// with angle
//   rot[i] = (i < D/4 ? row : col) * inv_freq[i % (D/4)],
//   inv_freq[j] = theta^(-2 j / (D/2)),
// which is `emb = cat(rot, rot)` over the head applied with rotate_half
// pairing (i, i + D/2) (VISION.md "Rotary"); v is left alone. Grid
// (H, N). Precise sin/cos: angles reach a few hundred radians.
kernel void vision_qkv_rope_bf16(device bfloat*       qkv   [[buffer(0)]],
                                 constant uint&       H     [[buffer(1)]],
                                 constant uint&       D     [[buffer(2)]],
                                 constant uint&       gw    [[buffer(3)]],
                                 constant float&      theta [[buffer(4)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x;
    const uint r = gid.y;
    const uint half_h = H / 2;
    if (j >= 2 * half_h) {
        return;
    }
    const uint which = j / half_h;          // 0 q, 1 k
    const uint within = j % half_h;
    const uint half_d = D / 2;
    const uint head = within / half_d;
    const uint i = within % half_d;
    const uint col0 = which * H + head * D + i;
    const uint col1 = col0 + half_d;
    const ulong base = (ulong)r * 3 * H;
    const float x0 = float(qkv[base + col0]);
    const float x1 = float(qkv[base + col1]);
    const uint2 rc = vision_grid_coords(r, gw);
    const uint quarter = half_d / 2;
    const uint fi = i % quarter;
    const float pos = float(i < quarter ? rc.x : rc.y);
    const float inv_freq =
        1.0f / precise::pow(theta, float(2 * fi) / float(half_d));
    const float ang = pos * inv_freq;
    const float c = precise::cos(ang);
    const float s = precise::sin(ang);
    qkv[base + col0] = bfloat(x0 * c - x1 * s);
    qkv[base + col1] = bfloat(x1 * c + x0 * s);
}

// Full bidirectional attention over the N patches of one image, one query
// tile and head per threadgroup, online softmax in float32: the tensor-op
// flash kernel of attention.metal without the causal limit and the KV cache,
// reading q, k and v straight out of the fused qkv rows (row stride 3H) and
// writing out [N, H]. Copied rather than shared so the text path's kernel
// keeps its numerics untouched.
#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

template <int D, int BQ, int BK, int SG>
static void vision_attn_body(device bfloat* qkv,
                             device bfloat* out,
                             uint N,
                             uint H,
                             float scale,
                             threadgroup float* s_tile,
                             threadgroup bfloat* p_tile,
                             threadgroup float* row_max,
                             threadgroup float* row_sum,
                             threadgroup float* row_alpha,
                             uint2 tg,
                             uint tid,
                             uint2 tg_size) {
    constexpr uint LANES = uint(32 * SG) / uint(BQ);
    static_assert(BQ * int(LANES) == 32 * SG,
                  "the threadgroup must divide into BQ equal row groups");
    static_assert(BK % int(LANES) == 0,
                  "each lane must own a whole number of key columns");
    static_assert(LANES <= 32, "a row's lanes must sit inside one simdgroup");

    using namespace mpp::tensor_ops;

    if (tg_size.x != uint(32 * SG)) {
        return;
    }
    const int q0 = int(tg.x) * BQ;
    const uint head = tg.y;
    const int len = int(N);
    const int stride = int(3 * H);

    if (tid < uint(BQ)) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const array<int, 2> row_strides{1, stride};
    device bfloat* q_head = qkv + (ulong)q0 * stride + head * D;
    device bfloat* k_head = qkv + H + head * D;
    device bfloat* v_head = qkv + 2 * H + head * D;
    auto tQ = tensor(q_head, dextents<int32_t, 2>(D, min(BQ, len - q0)), row_strides);
    auto tK = tensor(k_head, dextents<int32_t, 2>(D, len), row_strides);
    auto tV = tensor(v_head, dextents<int32_t, 2>(D, len), row_strides);
    auto tS = tensor(s_tile, dextents<int32_t, 2>(BK, BQ));
    auto tP = tensor(p_tile, dextents<int32_t, 2>(BK, BQ));

    // A static reduction extent must be a multiple of 16 and the head dim
    // (72) is not, so QK takes it as a dynamic extent from the operands; a
    // static output width only needs a multiple of 8, so PV keeps D static.
    constexpr auto qk_desc = matmul2d_descriptor(
        BQ, BK, static_cast<int>(metal::dynamic_extent), false,
        /*transpose_right=*/true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(
        BQ, D, BK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, metal::execution_simdgroups<SG>> qk_op;
    matmul2d<pv_desc, metal::execution_simdgroups<SG>> pv_op;

    using KSlice = decltype(tK.slice(0, 0));
    using VSlice = decltype(tV.slice(0, 0));

    auto acc = pv_op.template get_destination_cooperative_tensor<decltype(tP), VSlice, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0.0f;
        }
    }

    const uint row = tid / LANES;
    const uint lane = tid % LANES;
    constexpr uint cols = uint(BK) / LANES;
    const uint j0 = lane * cols;

    for (int k0 = 0; k0 < len; k0 += BK) {
        auto kSlice = tK.slice(0, k0);
        auto vSlice = tV.slice(0, k0);
        auto sT = qk_op.template get_destination_cooperative_tensor<decltype(tQ), KSlice, float>();
        qk_op.run(tQ, kSlice, sT);
        sT.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Every key of the image is visible: only the ragged last tile masks.
        const int limit = min(len - k0, BK);
        const float prev_max = row_max[row];
        float local = prev_max;
        for (uint j = j0; j < j0 + cols; ++j) {
            if (int(j) < limit) {
                local = max(local, s_tile[row * BK + j] * scale);
            }
        }
        for (uint off = 1u; off < LANES; off <<= 1) {
            local = max(local, simd_shuffle_xor(local, off));
        }
        const float mx = local;
        const float alpha = exp(prev_max - mx);
        float psum = 0.0f;
        for (uint j = j0; j < j0 + cols; ++j) {
            const float p = int(j) < limit
                ? exp(s_tile[row * BK + j] * scale - mx)
                : 0.0f;
            p_tile[row * BK + j] = bfloat(p);
            psum += p;
        }
        for (uint off = 1u; off < LANES; off <<= 1) {
            psum += simd_shuffle_xor(psum, off);
        }
        if (lane == 0u) {
            row_sum[row] = row_sum[row] * alpha + psum;
            row_max[row] = mx;
            row_alpha[row] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                auto ix = acc.get_multidimensional_index(i);
                acc[i] *= row_alpha[ix[1]];
            }
        }
        pv_op.run(tP, vSlice, acc);
        // Finish tensor ops before reusing p_tile and s_tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            const int r = ix[1], col = ix[0];
            if (q0 + r < len) {
                out[(ulong)(q0 + r) * H + head * D + col] = bfloat(acc[i] / row_sum[r]);
            }
        }
    }
}

#define VISION_ATTN_KERNEL(NAME, D, BQ, BK, SG)                                          \
kernel void NAME(device bfloat* qkv   [[buffer(0)]],                                     \
                 device bfloat* out   [[buffer(1)]],                                     \
                 constant uint& N     [[buffer(2)]],                                     \
                 constant uint& H     [[buffer(3)]],                                     \
                 constant float& scale [[buffer(4)]],                                    \
                 uint2 tg  [[threadgroup_position_in_grid]],                             \
                 uint  tid [[thread_index_in_threadgroup]],                              \
                 uint2 tg_size [[threads_per_threadgroup]]) {                            \
    threadgroup float  s_tile[(BQ) * (BK)];                                              \
    threadgroup bfloat p_tile[(BQ) * (BK)];                                              \
    threadgroup float  row_max[(BQ)], row_sum[(BQ)], row_alpha[(BQ)];                    \
    vision_attn_body<(D), (BQ), (BK), (SG)>(                                             \
        qkv, out, N, H, scale, s_tile, p_tile, row_max, row_sum, row_alpha,              \
        tg, tid, tg_size);                                                               \
}

// Head dim 72 (16 heads of 1152): 32-query tiles over 64-key tiles. Measured
// at 8 160 patches against 16 x 128, 32 x 128 and 64 x 64 tiles, this is the
// fastest by a few percent (12 KB of threadgroup memory instead of 24 lets two
// threadgroups share a core); 256-key tiles exceed the 32 KB limit.
VISION_ATTN_KERNEL(vision_attn_nax_d72, 72, 32, 64, 4)
#endif
