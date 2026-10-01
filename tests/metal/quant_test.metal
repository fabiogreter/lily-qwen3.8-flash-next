// Test-only: the grouped Q4 expert GEMM as it shipped before 2026-10-01
// (64-deep K steps, one 4-byte code word per thread and step, an unpadded
// B tile), kept as the reference the shipped kernel is asserted
// bit-identical against. Concatenated after quant.metal.
#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Grouped Q4 GEMM dequantizes B tiles to BF16 and accumulates in FP32.
constant constexpr uint REF_BN = 64;
constant constexpr uint REF_BK = 64;

// One block of the grouped GEMM at row-tile height BM: the B tile is
// dequantized per K step and the product accumulated over the block's rows.
template <uint BM, int SG, typename AT, typename BT>
static void ref_grouped_rows(device const uint* codes,
                                        device const bfloat* scales,
                                        device const bfloat* biases,
                                        thread AT& ta,
                                        thread BT& tb,
                                        device bfloat* c,
                                        uint K, uint N, uint GS,
                                        uint m0, uint b_row0, uint n0, uint m_end,
                                        threadgroup bfloat* b_tile,
                                        uint tid) {
    using namespace mpp::tensor_ops;
    const uint words = K / 8;
    const uint groups = K / GS;
    constexpr auto desc = matmul2d_descriptor(
        BM, REF_BN, REF_BK, /*transpose_left=*/false,
        /*transpose_right=*/true, /*relaxed_precision=*/false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, metal::execution_simdgroups<SG>> op;

    using ASlice = decltype(ta.slice(0, 0));
    auto acc = op.template get_destination_cooperative_tensor<ASlice, BT, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0.0f;
        }
    }

    for (uint k0 = 0; k0 < K; k0 += REF_BK) {
        // Dequantize one 64x64 B tile across 32*SG threads.
        for (uint i = tid; i < REF_BN * REF_BK / 8; i += 32 * SG) {
            uint r = i / (REF_BK / 8);
            uint wcol = i % (REF_BK / 8);
            uint k = k0 + wcol * 8;
            ulong row = b_row0 + r;
            uint word = codes[row * words + k / 8];
            uint g = k / GS;
            float s = float(scales[row * groups + g]);
            float b = float(biases[row * groups + g]);
            store_word_q4_tg(word, s, b,
                             (threadgroup bfloat4*)(b_tile + r * REF_BK), wcol);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a_slice = ta.slice(int(k0), int(m0));
        op.run(a_slice, tb, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            uint row = m0 + uint(ix[1]);
            uint col = n0 + uint(ix[0]);
            if (row < m_end) {
                c[(ulong)row * N + col] = bfloat(acc[i]);
            }
        }
    }
}

// BM must match the block-map row tile. An expert's last tile usually
// holds fewer rows than BM (about 80 routed rows per expert per 4 096-token
// chunk against 64-row tiles); such a block runs the product at the
// smallest height of BM, BM/2 and BM/4 that covers its rows instead of
// paying tensor work on padding.
template <uint BM, int SG>
static void ref_grouped_body(device const uint* codes,
                                        device const bfloat* scales,
                                        device const bfloat* biases,
                                        device bfloat* a,
                                        device bfloat* c,
                                        device const uint4* blocks,
                                        uint K, uint N, uint GS,
                                        threadgroup bfloat* b_tile,
                                        uint bid, uint tid) {
    const uint4 blk = blocks[bid];
    if (blk.x >= blk.w) {
        return;  // sentinel entry from the GPU-built block map
    }
    const uint m0 = blk.x, b_row0 = blk.y, n0 = blk.z, m_end = blk.w;
    const uint rows = m_end - m0;

    auto ta = metal::tensor(a, metal::dextents<int32_t, 2>(K, int(m_end)));
    auto tb = metal::tensor(b_tile, metal::dextents<int32_t, 2>(REF_BK, REF_BN));
    if (rows <= BM / 4) {
        ref_grouped_rows<BM / 4, SG>(codes, scales, biases, ta, tb, c, K, N, GS,
                                                m0, b_row0, n0, m_end, b_tile, tid);
    } else if (rows <= BM / 2) {
        ref_grouped_rows<BM / 2, SG>(codes, scales, biases, ta, tb, c, K, N, GS,
                                                m0, b_row0, n0, m_end, b_tile, tid);
    } else {
        ref_grouped_rows<BM, SG>(codes, scales, biases, ta, tb, c, K, N, GS,
                                            m0, b_row0, n0, m_end, b_tile, tid);
    }
}

#define REF_GROUPED(NAME, BM, SG)                                   \
    kernel void NAME(device const uint*   codes  [[buffer(0)]],                \
                     device const bfloat* scales [[buffer(1)]],                \
                     device const bfloat* biases [[buffer(2)]],                \
                     device bfloat*       a      [[buffer(3)]],                \
                     device bfloat*       c      [[buffer(4)]],                \
                     device const uint4*  blocks [[buffer(5)]],                \
                     constant uint&       K      [[buffer(6)]],                \
                     constant uint&       N      [[buffer(7)]],                \
                     constant uint&       GS     [[buffer(8)]],                \
                     uint bid [[threadgroup_position_in_grid]],                \
                     uint tid [[thread_index_in_threadgroup]]) {               \
        threadgroup bfloat b_tile[REF_BN * REF_BK];                          \
        ref_grouped_body<BM, SG>(codes, scales, biases, a, c,       \
                                            blocks, K, N, GS, b_tile, bid,     \
                                            tid);                              \
    }

// Dispatch instantiated tiles with 32*SG threads.
REF_GROUPED(gemm_q4_nt_nax_grouped_ref_t32x4, 32, 4)
REF_GROUPED(gemm_q4_nt_nax_grouped_ref_t64x4, 64, 4)
#endif

