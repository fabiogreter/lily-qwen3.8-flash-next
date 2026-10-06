// Unit-test-only staged-A variants with explicit K-chunk sizes.
GEMM_SKINNY_Q4(gemm_skinny_q4_bf16_m8_kc512, 8, 512, bfloat)
GEMM_SKINNY_Q4(gemm_skinny_q4_bf16_m8_kc1024, 8, 1024, bfloat)

// The register-A Q4 kernels as shipped before the explicit-order rewrite
// (main 0446016), kept verbatim (helpers renamed) as the bit-identity
// reference of `gemm_skinny_q4_reg_body`. Their numerics are whatever the
// fast-math compiler made of this source; the rewrite reproduces them
// operation by operation, and the tests compare the two bit for bit.
constant constexpr uint SKINNY_REG_ROWS_PREVIOUS = 2;

static inline void skinny_load_x8_previous(device const bfloat* p, bool vec,
                                           thread float4& lo, thread float4& hi) {
    if (vec) {
        const uint4 v = *(device const uint4*)p;
        lo = float4(as_type<bfloat4>(v.xy));
        hi = float4(as_type<bfloat4>(v.zw));
    } else {
        lo = float4(p[0], p[1], p[2], p[3]);
        hi = float4(p[4], p[5], p[6], p[7]);
    }
}

static inline void skinny_unpack_q4_previous(uint word, thread float4& lo, thread float4& hi) {
    lo = float4(float((word >> 0) & 0xF), float((word >> 4) & 0xF),
                float((word >> 8) & 0xF), float((word >> 12) & 0xF));
    hi = float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                float((word >> 24) & 0xF), float((word >> 28) & 0xF));
}

template <uint MB, typename CT>
static void gemm_skinny_q4_reg_body_previous(device const uint* codes,
                                             device const bfloat* scales,
                                             device const bfloat* biases,
                                             device const bfloat* a, device CT* c,
                                             uint K, uint N, uint GS, uint tg,
                                             uint tg_size, uint simd_id, uint lane) {
    const uint row0 = (tg * (tg_size / 32) + simd_id) * SKINNY_REG_ROWS_PREVIOUS;
    if (row0 >= N) {
        return;  // whole simdgroup
    }
    const bool has1 = row0 + 1 < N;
    const uint row1 = has1 ? row0 + 1 : row0;
    const uint words = K / 8;
    const uint blocks = words / 4;
    const uint bpg = GS / 32;
    const uint groups = K / GS;
    const bool a_vec = ((ulong)a & 15) == 0;
    device const uint4* w0 = (device const uint4*)(codes + (ulong)row0 * words);
    device const uint4* w1 = (device const uint4*)(codes + (ulong)row1 * words);

    float acc0[MB];
    float acc1[MB];
    for (uint i = 0; i < MB; ++i) {
        acc0[i] = 0.0f;
        acc1[i] = 0.0f;
    }

    for (uint blk = lane; blk < blocks; blk += 32) {
        const uint g = blk / bpg;
        const float s0 = float(scales[row0 * groups + g]);
        const float b0 = float(biases[row0 * groups + g]);
        const float s1 = float(scales[row1 * groups + g]);
        const float b1 = float(biases[row1 * groups + g]);
        const uint4 v0 = w0[blk];
        const uint4 v1 = w1[blk];
        float qx0[MB];
        float qx1[MB];
        float xs[MB];
        for (uint i = 0; i < MB; ++i) {
            qx0[i] = 0.0f;
            qx1[i] = 0.0f;
            xs[i] = 0.0f;
        }
        for (uint wi = 0; wi < 4; ++wi) {
            float4 q0lo, q0hi, q1lo, q1hi;
            skinny_unpack_q4_previous(v0[wi], q0lo, q0hi);
            skinny_unpack_q4_previous(v1[wi], q1lo, q1hi);
            for (uint i = 0; i < MB; ++i) {
                float4 xlo, xhi;
                skinny_load_x8_previous(a + (ulong)i * K + blk * 32 + wi * 8, a_vec, xlo, xhi);
                xs[i] += dot(xlo, float4(1.0f)) + dot(xhi, float4(1.0f));
                qx0[i] += dot(q0lo, xlo) + dot(q0hi, xhi);
                qx1[i] += dot(q1lo, xlo) + dot(q1hi, xhi);
            }
        }
        for (uint i = 0; i < MB; ++i) {
            acc0[i] += s0 * qx0[i] + b0 * xs[i];
            acc1[i] += s1 * qx1[i] + b1 * xs[i];
        }
    }

    for (uint i = 0; i < MB; ++i) {
        const float sum0 = simd_sum(acc0[i]);
        const float sum1 = simd_sum(acc1[i]);
        if (lane == 0) {
            c[(ulong)i * N + row0] = CT(sum0);
            if (has1) {
                c[(ulong)i * N + row1] = CT(sum1);
            }
        }
    }
}

#define GEMM_SKINNY_Q4_REG_PREVIOUS(NAME, MB, CT)                              \
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
        gemm_skinny_q4_reg_body_previous<MB, CT>(codes, scales, biases, a, c,  \
                                                 K, N, GS, tg, tg_size,        \
                                                 simd_id, lane);               \
    }

GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_bf16_reg_m1_previous, 1, bfloat)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_bf16_reg_m2_previous, 2, bfloat)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_bf16_reg_m3_previous, 3, bfloat)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_bf16_reg_m4_previous, 4, bfloat)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_bf16_reg_m5_previous, 5, bfloat)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_bf16_reg_m6_previous, 6, bfloat)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_bf16_reg_m7_previous, 7, bfloat)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_bf16_reg_m8_previous, 8, bfloat)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_f32_reg_m1_previous, 1, float)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_f32_reg_m2_previous, 2, float)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_f32_reg_m3_previous, 3, float)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_f32_reg_m4_previous, 4, float)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_f32_reg_m5_previous, 5, float)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_f32_reg_m6_previous, 6, float)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_f32_reg_m7_previous, 7, float)
GEMM_SKINNY_Q4_REG_PREVIOUS(gemm_skinny_q4_f32_reg_m8_previous, 8, float)
