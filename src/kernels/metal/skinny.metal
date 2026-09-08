// Small-M Q4 GEMM with FP32 accumulation and BF16-rounded weights.
// Staged-A and register-A variants cover different M/N regimes.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint SKINNY_KC = 256;
constant constexpr uint SKINNY_SG = 4;

// Stages an A tile with zero padding; vec_ok enables aligned 16-byte loads.
template <uint MB, uint KC>
static inline void stage_a_chunk(device const bfloat* a,
                                 threadgroup bfloat* a_tile, uint K, uint m0,
                                 uint m_rem, uint k0, uint kc, bool vec_ok,
                                 uint tid) {
    if (vec_ok) {
        for (uint i = tid; i < MB * (KC / 8); i += 32 * SKINNY_SG) {
            uint r = i / (KC / 8);
            uint col = (i % (KC / 8)) * 8;
            uint4 v = uint4(0);
            if (r < m_rem && col < kc) {
                v = *(device const uint4*)(a + (ulong)(m0 + r) * K + k0 + col);
            }
            ((threadgroup uint4*)a_tile)[i] = v;
        }
    } else {
        for (uint i = tid; i < MB * KC; i += 32 * SKINNY_SG) {
            uint r = i / KC;
            uint col = i % KC;
            a_tile[i] = (r < m_rem && col < kc)
                            ? a[(ulong)(m0 + r) * K + k0 + col]
                            : bfloat(0.0f);
        }
    }
}

// Accumulates one eight-value weight word against each staged A row.
template <uint MB, uint KC>
static inline void accumulate_word(float4 wlo, float4 whi,
                                   threadgroup const bfloat* a_tile, uint col,
                                   thread float (&acc)[MB]) {
    threadgroup const bfloat4* xv =
        (threadgroup const bfloat4*)a_tile + col / 4;
    for (uint i = 0; i < MB; ++i) {
        float4 xlo = float4(xv[i * (KC / 4)]);
        float4 xhi = float4(xv[i * (KC / 4) + 1]);
        acc[i] += dot(wlo, xlo) + dot(whi, xhi);
    }
}

// Reduces and stores one output column; control flow is simdgroup-uniform.
template <uint MB, typename CT>
static inline void store_column(thread float (&acc)[MB], device CT* c,
                                uint m0, uint m_rem, uint N, uint row,
                                uint lane) {
    for (uint i = 0; i < MB; ++i) {
        float sum = simd_sum(acc[i]);
        if (lane == 0 && i < m_rem) {
            c[(ulong)(m0 + i) * N + row] = CT(sum);
        }
    }
}

// Host requires K and GS divisible by 8 so code words do not cross groups.
template <uint MB, uint KC, typename CT>
static void gemm_skinny_q4_body(device const uint* codes,
                                device const bfloat* scales,
                                device const bfloat* biases,
                                device const bfloat* a, device CT* c,
                                uint K, uint N, uint GS, uint M,
                                threadgroup bfloat* a_tile, uint2 tg, uint tid,
                                uint simd_id, uint lane) {
    const uint row = tg.x * SKINNY_SG + simd_id;
    const uint m0 = tg.y * MB;
    const uint m_rem = min(M - m0, MB);
    const uint words = K / 8;
    const uint groups = K / GS;
    const bool a_vec = ((ulong)a & 15) == 0;

    float acc[MB];
    for (uint i = 0; i < MB; ++i) {
        acc[i] = 0.0f;
    }

    for (uint k0 = 0; k0 < K; k0 += KC) {
        const uint kc = min(KC, K - k0);
        stage_a_chunk<MB, KC>(a, a_tile, K, m0, m_rem, k0, kc, a_vec, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < N) {
            for (uint col = lane * 8; col < kc; col += 256) {
                uint word = codes[(ulong)row * words + (k0 + col) / 8];
                uint g = (k0 + col) / GS;
                float s = float(scales[row * groups + g]);
                float b = float(biases[row * groups + g]);
                float4 qlo =
                    float4(float((word >> 0) & 0xF), float((word >> 4) & 0xF),
                           float((word >> 8) & 0xF), float((word >> 12) & 0xF));
                float4 qhi =
                    float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                           float((word >> 24) & 0xF), float((word >> 28) & 0xF));
                accumulate_word<MB, KC>(float4(bfloat4(qlo * s + b)),
                                        float4(bfloat4(qhi * s + b)), a_tile,
                                        col, acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < N) {
        store_column<MB, CT>(acc, c, m0, m_rem, N, row, lane);
    }
}

// 8-bit staged body: a code word holds four elements (low byte first), so a
// lane consumes two words per eight-element step. Dequantized weights are
// bf16-rounded like the Q4 path and the dequant+GEMM fallback it replaces.
template <uint MB, uint KC, typename CT>
static void gemm_skinny_q8_body(device const uint* codes,
                                device const bfloat* scales,
                                device const bfloat* biases,
                                device const bfloat* a, device CT* c,
                                uint K, uint N, uint GS, uint M,
                                threadgroup bfloat* a_tile, uint2 tg, uint tid,
                                uint simd_id, uint lane) {
    const uint row = tg.x * SKINNY_SG + simd_id;
    const uint m0 = tg.y * MB;
    const uint m_rem = min(M - m0, MB);
    const uint words = K / 4;
    const uint groups = K / GS;
    const bool a_vec = ((ulong)a & 15) == 0;

    float acc[MB];
    for (uint i = 0; i < MB; ++i) {
        acc[i] = 0.0f;
    }

    for (uint k0 = 0; k0 < K; k0 += KC) {
        const uint kc = min(KC, K - k0);
        stage_a_chunk<MB, KC>(a, a_tile, K, m0, m_rem, k0, kc, a_vec, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < N) {
            for (uint col = lane * 8; col < kc; col += 256) {
                const uint wi = (ulong)row * words + (k0 + col) / 4;
                const uint w0 = codes[wi];
                const uint w1 = codes[wi + 1];
                uint g = (k0 + col) / GS;
                float s = float(scales[row * groups + g]);
                float b = float(biases[row * groups + g]);
                float4 qlo = float4(float(w0 & 0xFF), float((w0 >> 8) & 0xFF),
                                    float((w0 >> 16) & 0xFF), float((w0 >> 24) & 0xFF));
                float4 qhi = float4(float(w1 & 0xFF), float((w1 >> 8) & 0xFF),
                                    float((w1 >> 16) & 0xFF), float((w1 >> 24) & 0xFF));
                accumulate_word<MB, KC>(float4(bfloat4(qlo * s + b)),
                                        float4(bfloat4(qhi * s + b)), a_tile,
                                        col, acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < N) {
        store_column<MB, CT>(acc, c, m0, m_rem, N, row, lane);
    }
}

// Register-A body requires M == MB and K/GS divisible by 32.
// Each uint4 weight block stays within one quantization group.
template <uint MB, typename CT>
static void gemm_skinny_q4_reg_body(device const uint* codes,
                                    device const bfloat* scales,
                                    device const bfloat* biases,
                                    device const bfloat* a, device CT* c,
                                    uint K, uint N, uint GS, uint tg,
                                    uint tg_size, uint simd_id, uint lane) {
    const uint row = tg * (tg_size / 32) + simd_id;
    if (row >= N) {
        return;
    }
    const uint words = K / 8;
    const uint blocks = words / 4;
    const uint bpg = GS / 32;
    const uint groups = K / GS;
    const bool a_vec = ((ulong)a & 15) == 0;
    device const uint4* wrow = (device const uint4*)(codes + (ulong)row * words);

    float acc[MB];
    for (uint i = 0; i < MB; ++i) {
        acc[i] = 0.0f;
    }

    for (uint blk = lane; blk < blocks; blk += 32) {
        const uint g = blk / bpg;
        const float s = float(scales[row * groups + g]);
        const float b = float(biases[row * groups + g]);
        const uint4 w4 = wrow[blk];
        float4 wq[8];
        for (uint wi = 0; wi < 4; ++wi) {
            const uint word = w4[wi];
            float4 qlo =
                float4(float((word >> 0) & 0xF), float((word >> 4) & 0xF),
                       float((word >> 8) & 0xF), float((word >> 12) & 0xF));
            float4 qhi =
                float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                       float((word >> 24) & 0xF), float((word >> 28) & 0xF));
            wq[2 * wi] = float4(bfloat4(qlo * s + b));
            wq[2 * wi + 1] = float4(bfloat4(qhi * s + b));
        }
        for (uint i = 0; i < MB; ++i) {
            device const bfloat* arow = a + (ulong)i * K + blk * 32;
            if (a_vec) {
                device const uint4* av = (device const uint4*)arow;
                for (uint wi = 0; wi < 4; ++wi) {
                    const uint4 x = av[wi];
                    acc[i] += dot(wq[2 * wi], float4(as_type<bfloat4>(x.xy))) +
                              dot(wq[2 * wi + 1], float4(as_type<bfloat4>(x.zw)));
                }
            } else {
                for (uint wi = 0; wi < 4; ++wi) {
                    const float4 xlo =
                        float4(arow[wi * 8], arow[wi * 8 + 1],
                               arow[wi * 8 + 2], arow[wi * 8 + 3]);
                    const float4 xhi =
                        float4(arow[wi * 8 + 4], arow[wi * 8 + 5],
                               arow[wi * 8 + 6], arow[wi * 8 + 7]);
                    acc[i] += dot(wq[2 * wi], xlo) + dot(wq[2 * wi + 1], xhi);
                }
            }
        }
    }

    for (uint i = 0; i < MB; ++i) {
        const float sum = simd_sum(acc[i]);
        if (lane == 0) {
            c[(ulong)i * N + row] = CT(sum);
        }
    }
}

// 8-bit register-A body: a uint4 block holds 16 elements; requires M == MB,
// K % 16 == 0 and GS % 16 == 0 (blocks never straddle a quant group).
template <uint MB, typename CT>
static void gemm_skinny_q8_reg_body(device const uint* codes,
                                    device const bfloat* scales,
                                    device const bfloat* biases,
                                    device const bfloat* a, device CT* c,
                                    uint K, uint N, uint GS, uint tg,
                                    uint tg_size, uint simd_id, uint lane) {
    const uint row = tg * (tg_size / 32) + simd_id;
    if (row >= N) {
        return;
    }
    const uint words = K / 4;
    const uint blocks = words / 4;
    const uint bpg = GS / 16;
    const uint groups = K / GS;
    const bool a_vec = ((ulong)a & 15) == 0;
    device const uint4* wrow = (device const uint4*)(codes + (ulong)row * words);

    float acc[MB];
    for (uint i = 0; i < MB; ++i) {
        acc[i] = 0.0f;
    }

    for (uint blk = lane; blk < blocks; blk += 32) {
        const uint g = blk / bpg;
        const float s = float(scales[row * groups + g]);
        const float b = float(biases[row * groups + g]);
        const uint4 w4 = wrow[blk];
        float4 wq[4];
        for (uint wi = 0; wi < 4; ++wi) {
            const uint word = w4[wi];
            float4 q = float4(float(word & 0xFF), float((word >> 8) & 0xFF),
                              float((word >> 16) & 0xFF), float((word >> 24) & 0xFF));
            wq[wi] = float4(bfloat4(q * s + b));
        }
        for (uint i = 0; i < MB; ++i) {
            device const bfloat* arow = a + (ulong)i * K + blk * 16;
            if (a_vec) {
                device const uint4* av = (device const uint4*)arow;
                const uint4 x0 = av[0];
                const uint4 x1 = av[1];
                acc[i] += dot(wq[0], float4(as_type<bfloat4>(x0.xy))) +
                          dot(wq[1], float4(as_type<bfloat4>(x0.zw))) +
                          dot(wq[2], float4(as_type<bfloat4>(x1.xy))) +
                          dot(wq[3], float4(as_type<bfloat4>(x1.zw)));
            } else {
                for (uint wi = 0; wi < 4; ++wi) {
                    const float4 x = float4(arow[wi * 4], arow[wi * 4 + 1],
                                            arow[wi * 4 + 2], arow[wi * 4 + 3]);
                    acc[i] += dot(wq[wi], x);
                }
            }
        }
    }

    for (uint i = 0; i < MB; ++i) {
        const float sum = simd_sum(acc[i]);
        if (lane == 0) {
            c[(ulong)i * N + row] = CT(sum);
        }
    }
}

// Staged variants use 32*SKINNY_SG threads and one M block. CT is the output
// element type: bf16 for activations, f32 for logits.
#define GEMM_SKINNY_Q4(NAME, MB, KC, CT)                                       \
    kernel void NAME(device const uint*   codes  [[buffer(0)]],                \
                     device const bfloat* scales [[buffer(1)]],                \
                     device const bfloat* biases [[buffer(2)]],                \
                     device const bfloat* a      [[buffer(3)]],                \
                     device CT*           c      [[buffer(4)]],                \
                     constant uint&       K      [[buffer(5)]],                \
                     constant uint&       N      [[buffer(6)]],                \
                     constant uint&       GS     [[buffer(7)]],                \
                     constant uint&       M      [[buffer(8)]],                \
                     uint2 tg      [[threadgroup_position_in_grid]],           \
                     uint  tid     [[thread_index_in_threadgroup]],            \
                     uint  simd_id [[simdgroup_index_in_threadgroup]],         \
                     uint  lane    [[thread_index_in_simdgroup]]) {            \
        threadgroup bfloat a_tile[MB * KC];                                    \
        gemm_skinny_q4_body<MB, KC, CT>(codes, scales, biases, a, c, K, N, GS, \
                                        M, a_tile, tg, tid, simd_id, lane);    \
    }

GEMM_SKINNY_Q4(gemm_skinny_q4_bf16_m8, 8, SKINNY_KC, bfloat)
GEMM_SKINNY_Q4(gemm_skinny_q4_bf16_m16, 16, SKINNY_KC, bfloat)
GEMM_SKINNY_Q4(gemm_skinny_q4_f32_m8, 8, SKINNY_KC, float)
GEMM_SKINNY_Q4(gemm_skinny_q4_f32_m16, 16, SKINNY_KC, float)

#define GEMM_SKINNY_Q8(NAME, MB, KC, CT)                                       \
    kernel void NAME(device const uint*   codes  [[buffer(0)]],                \
                     device const bfloat* scales [[buffer(1)]],                \
                     device const bfloat* biases [[buffer(2)]],                \
                     device const bfloat* a      [[buffer(3)]],                \
                     device CT*           c      [[buffer(4)]],                \
                     constant uint&       K      [[buffer(5)]],                \
                     constant uint&       N      [[buffer(6)]],                \
                     constant uint&       GS     [[buffer(7)]],                \
                     constant uint&       M      [[buffer(8)]],                \
                     uint2 tg      [[threadgroup_position_in_grid]],           \
                     uint  tid     [[thread_index_in_threadgroup]],            \
                     uint  simd_id [[simdgroup_index_in_threadgroup]],         \
                     uint  lane    [[thread_index_in_simdgroup]]) {            \
        threadgroup bfloat a_tile[MB * KC];                                    \
        gemm_skinny_q8_body<MB, KC, CT>(codes, scales, biases, a, c, K, N, GS, \
                                        M, a_tile, tg, tid, simd_id, lane);    \
    }

GEMM_SKINNY_Q8(gemm_skinny_q8_bf16_m8, 8, SKINNY_KC, bfloat)
GEMM_SKINNY_Q8(gemm_skinny_q8_bf16_m16, 16, SKINNY_KC, bfloat)
GEMM_SKINNY_Q8(gemm_skinny_q8_f32_m8, 8, SKINNY_KC, float)
GEMM_SKINNY_Q8(gemm_skinny_q8_f32_m16, 16, SKINNY_KC, float)

#define GEMM_SKINNY_Q8_REG(NAME, MB, CT)                                       \
    kernel void NAME(device const uint*   codes  [[buffer(0)]],                \
                     device const bfloat* scales [[buffer(1)]],                \
                     device const bfloat* biases [[buffer(2)]],                \
                     device const bfloat* a      [[buffer(3)]],                \
                     device CT*           c      [[buffer(4)]],                \
                     constant uint&       K      [[buffer(5)]],                \
                     constant uint&       N      [[buffer(6)]],                \
                     constant uint&       GS     [[buffer(7)]],                \
                     uint tg      [[threadgroup_position_in_grid]],            \
                     uint tg_size [[threads_per_threadgroup]],                 \
                     uint simd_id [[simdgroup_index_in_threadgroup]],          \
                     uint lane    [[thread_index_in_simdgroup]]) {             \
        gemm_skinny_q8_reg_body<MB, CT>(codes, scales, biases, a, c, K, N, GS, \
                                        tg, tg_size, simd_id, lane);           \
    }

GEMM_SKINNY_Q8_REG(gemm_skinny_q8_bf16_reg_m1, 1, bfloat)
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_bf16_reg_m2, 2, bfloat)
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_bf16_reg_m3, 3, bfloat)
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_bf16_reg_m4, 4, bfloat)
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_bf16_reg_m5, 5, bfloat)
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_bf16_reg_m6, 6, bfloat)
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_bf16_reg_m7, 7, bfloat)
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_bf16_reg_m8, 8, bfloat)

// Register-A variants require M to match the kernel suffix.
#define GEMM_SKINNY_Q4_REG(NAME, MB, CT)                                       \
    kernel void NAME(device const uint*   codes  [[buffer(0)]],                \
                     device const bfloat* scales [[buffer(1)]],                \
                     device const bfloat* biases [[buffer(2)]],                \
                     device const bfloat* a      [[buffer(3)]],                \
                     device CT*           c      [[buffer(4)]],                \
                     constant uint&       K      [[buffer(5)]],                \
                     constant uint&       N      [[buffer(6)]],                \
                     constant uint&       GS     [[buffer(7)]],                \
                     uint tg      [[threadgroup_position_in_grid]],            \
                     uint tg_size [[threads_per_threadgroup]],                 \
                     uint simd_id [[simdgroup_index_in_threadgroup]],          \
                     uint lane    [[thread_index_in_simdgroup]]) {             \
        gemm_skinny_q4_reg_body<MB, CT>(codes, scales, biases, a, c, K, N, GS, \
                                        tg, tg_size, simd_id, lane);           \
    }

GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m1, 1, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m2, 2, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m3, 3, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m4, 4, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m5, 5, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m6, 6, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m7, 7, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m8, 8, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m1, 1, float)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m2, 2, float)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m3, 3, float)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m4, 4, float)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m5, 5, float)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m6, 6, float)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m7, 7, float)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m8, 8, float)
