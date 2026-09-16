// BF16 C[M,N] = A[M,K] * B[N,K]^T with FP32 accumulation.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint GEMM_BM = 64;
constant constexpr uint GEMM_BN = 64;
constant constexpr uint GEMM_BK = 32;

// Metal 4 tensor GEMM; tensor slices bounds-check ragged edge tiles.
#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

constant constexpr int NAX_BM = 64;
constant constexpr int NAX_BN = 64;
constant constexpr int NAX_SIMDGROUPS = 4;

kernel void gemm_bf16_nt_nax(device bfloat* a   [[buffer(0)]],
                             device bfloat* b   [[buffer(1)]],
                             device bfloat* c   [[buffer(2)]],
                             constant uint& K   [[buffer(3)]],
                             constant uint& N   [[buffer(4)]],
                             constant uint& M   [[buffer(5)]],
                             uint2 tgid [[threadgroup_position_in_grid]]) {
    using namespace mpp::tensor_ops;
    constexpr auto desc = matmul2d_descriptor(
        NAX_BM, NAX_BN, static_cast<int>(metal::dynamic_extent),
        /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, metal::execution_simdgroups<NAX_SIMDGROUPS>> op;

    auto ta = metal::tensor(a, metal::dextents<int32_t, 2>(K, M));
    auto tb = metal::tensor(b, metal::dextents<int32_t, 2>(K, N));
    auto tc = metal::tensor(c, metal::dextents<int32_t, 2>(N, M));
    auto tile_a = ta.slice(0, int(tgid.y) * NAX_BM);
    auto tile_b = tb.slice(0, int(tgid.x) * NAX_BN);
    auto tile_c = tc.slice(int(tgid.x) * NAX_BN, int(tgid.y) * NAX_BM);

    auto acc = op.get_destination_cooperative_tensor<decltype(tile_a),
                                                     decltype(tile_b), bfloat>();
    op.run(tile_a, tile_b, acc);
    acc.store(tile_c);
}

// The same GEMM with a bias epilogue: C = bf16(A B^T + bias[col]) rounded
// once, as a Linear layer's output is. The accumulator stays f32 until the
// bias is in; a separate bias pass over a bf16 C would round twice.
kernel void gemm_bf16_nt_bias_nax(device bfloat* a          [[buffer(0)]],
                                  device bfloat* b          [[buffer(1)]],
                                  device bfloat* c          [[buffer(2)]],
                                  device const bfloat* bias [[buffer(3)]],
                                  constant uint& K          [[buffer(4)]],
                                  constant uint& N          [[buffer(5)]],
                                  constant uint& M          [[buffer(6)]],
                                  uint2 tgid [[threadgroup_position_in_grid]]) {
    using namespace mpp::tensor_ops;
    constexpr auto desc = matmul2d_descriptor(
        NAX_BM, NAX_BN, static_cast<int>(metal::dynamic_extent),
        /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, metal::execution_simdgroups<NAX_SIMDGROUPS>> op;

    auto ta = metal::tensor(a, metal::dextents<int32_t, 2>(K, M));
    auto tb = metal::tensor(b, metal::dextents<int32_t, 2>(K, N));
    auto tile_a = ta.slice(0, int(tgid.y) * NAX_BM);
    auto tile_b = tb.slice(0, int(tgid.x) * NAX_BN);

    auto acc = op.get_destination_cooperative_tensor<decltype(tile_a),
                                                     decltype(tile_b), float>();
    op.run(tile_a, tile_b, acc);
    const int col0 = int(tgid.x) * NAX_BN;
    const int row0 = int(tgid.y) * NAX_BM;
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            const int col = col0 + ix[0], row = row0 + ix[1];
            if (col < int(N) && row < int(M)) {
                c[(ulong)row * N + col] = bfloat(acc[i] + float(bias[col]));
            }
        }
    }
}
#endif
