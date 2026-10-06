// Small-M Q4/Q8 GEMMs with FP32 accumulation. The staged-A variants round
// dequantized weights to bf16 like the dequant + GEMM fallback they replace;
// the register-A variants (m <= 8) dot the raw codes like the decode GEMV.
// The two cover different M/N regimes.

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
                // 64-bit like the Q4 body: a Q8 LM head (`--q8-dense`)
                // is the widest weight this kernel sees.
                const ulong wi = (ulong)row * words + (k0 + col) / 4;
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

// Register-A bodies: lanes stride over the uint4 weight blocks (32 Q4 or 16
// Q8 elements) of a few consecutive weight rows, and each lane dots its
// activation blocks against those rows' raw codes; scale and bias are
// applied once per block as s * dot(q, x) + b * sum(x) (quant.metal's
// dot_word_q4), so the weights are never dequantized element by element and
// never rounded to bf16. Requires M == MB, and K and GS multiples of the
// block (a block never straddles a quant group).

// Weight rows per Q8 register-A simdgroup.
constant constexpr uint SKINNY_REG_ROWS = 2;

// Eight bf16 activations at `p` as two float4 (16-byte load when aligned).
static inline void skinny_load_x8(device const bfloat* p, bool vec,
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

static inline float4 skinny_unpack_q8(uint word) {
    return float4(float(word & 0xFF), float((word >> 8) & 0xFF),
                  float((word >> 16) & 0xFF), float((word >> 24) & 0xFF));
}

// Q4 register-A. Lane L of a simdgroup owns blocks L, L + 32, L + 64, ... of
// its rows, and a row's result is the simd_sum of its 32 lane partials. The
// arithmetic of every output is spelled out operation by operation and
// compiled in safe math (no reassociation, no contraction), so the compiler
// cannot reorder it: per element the code times the activation (exact, a
// 4-bit integer times a bf16), summed left to right in fours; per code word
// qx = (qx + dot_lo) + dot_hi, and xs (the block's activation sum) likewise;
// per block the partial becomes fma(b, xs, fma(s, qx, acc)) on even rows and
// fma(s, qx, fma(b, xs, acc)) on odd rows (the latter on both at m = 1). That
// is exactly what the fast-math compiler made of the earlier two-row kernel,
// which tests/metal/skinny_test.metal keeps as the bit-identity reference
// (`*_previous`). A row's result depends only on its index parity and m, so
// fused stacks and their slices at even row offsets stay bit-identical.
//
// The activation side of a block (loading and converting MB x 32 bf16 values
// and summing them) does not depend on the weight row; past m = 2 it, not the
// weight bytes, bounded the two-row kernel (an ALU-bound loop at 1.2x to 2.8x
// the m = 1 time). A simdgroup therefore takes RR rows to share it: 2 up to
// m = 2 (bandwidth-bound already; at m = 2 four rows sped up the big
// projections but slowed the batched step through its small ones, 16.1
// against 15.6 ms per two-row step) and 4 from m = 3. Eight rows share more
// and win on back-to-back dispatches, but in the model's passes, where a
// barrier follows nearly every projection, their fewer and longer
// simdgroups lose more to occupancy and the dispatch tail than they save,
// and past m = 5 their partials outgrow the registers. The code word loop stays rolled past
// RR x MB = 8 (skinny_q4_block): unrolled, the body outgrows the instruction
// cache and the kernel runs 1.3 to 5x slower.
//
// When at most 16 lanes have a block left for the last pass (K = 2560 is 80
// blocks, 2.5 passes), the two half-simdgroups split that pass: both take the
// same block, each for half of the rows, and the upper half's partials travel
// to and from their owning lanes by shuffle, so the last pass costs half a
// pass of ALU time instead of a full one.

#define SKINNY_UNROLL _Pragma("clang loop unroll(full)")

// Four products summed left to right; each product is exact, so the fma is
// that sum's rounding, not an extra fusion.
static inline float skinny_dot4(float4 q, float4 x) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    float d = q.x * x.x;
    d = fma(q.y, x.y, d);
    d = fma(q.z, x.z, d);
    return fma(q.w, x.w, d);
}

static inline float skinny_sum4(float4 x) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    return ((x.x + x.y) + x.z) + x.w;
}

// Folds a block into a lane partial; `sb` selects the order (see above).
static inline float skinny_fold(bool sb, float s, float qx, float b, float xs, float acc) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    return sb ? fma(s, qx, fma(b, xs, acc)) : fma(b, xs, fma(s, qx, acc));
}

// Eight bf16 activations at `p` as raw bits (16-byte load when aligned).
static inline uint4 skinny_load_raw8(device const bfloat* p, bool vec) {
    if (vec) {
        return *(device const uint4*)p;
    }
    device const ushort* u = (device const ushort*)p;
    return uint4(uint(u[0]) | (uint(u[1]) << 16), uint(u[2]) | (uint(u[3]) << 16),
                 uint(u[4]) | (uint(u[5]) << 16), uint(u[6]) | (uint(u[7]) << 16));
}

// Adds code word `wi` of block `blk` to qx[r][i] (row r's codes dotted with
// activation row i) and xs[i] (activation row i summed), for NR rows whose
// code words start at woff[r]. Half a word (four elements) at a time, which
// keeps four activation floats per row live instead of eight.
template <uint MB, uint NR>
static inline void skinny_q4_word(device const uint* codes, device const bfloat* a,
                                  thread const uint* woff, uint K, uint blk, uint wi,
                                  bool a_vec, thread float (&qx)[NR][MB],
                                  thread float (&xs)[MB]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    // A word's nibbles as bytes: wa holds codes 0, 2, 4, 6 and wb codes
    // 1, 3, 5, 7, so each code is one byte-to-float conversion.
    uint wa[NR];
    uint wb[NR];
    SKINNY_UNROLL for (uint r = 0; r < NR; ++r) {
        const uint wv = codes[woff[r] + blk * 4 + wi];
        wa[r] = wv & 0x0F0F0F0Fu;
        wb[r] = (wv >> 4) & 0x0F0F0F0Fu;
    }
    uint4 raw[MB];
    SKINNY_UNROLL for (uint i = 0; i < MB; ++i) {
        raw[i] = skinny_load_raw8(a + (i * K + blk * 32 + wi * 8), a_vec);
    }
    SKINNY_UNROLL for (uint h = 0; h < 2; ++h) {
        float4 x[MB];
        SKINNY_UNROLL for (uint i = 0; i < MB; ++i) {
            x[i] = float4(as_type<bfloat4>(h == 0 ? raw[i].xy : raw[i].zw));
            xs[i] = xs[i] + skinny_sum4(x[i]);
        }
        SKINNY_UNROLL for (uint r = 0; r < NR; ++r) {
            const uchar4 ba = as_type<uchar4>(wa[r]);
            const uchar4 bb = as_type<uchar4>(wb[r]);
            const float4 q = h == 0
                ? float4(float(ba.x), float(bb.x), float(ba.y), float(bb.y))
                : float4(float(ba.z), float(bb.z), float(ba.w), float(bb.w));
            SKINNY_UNROLL for (uint i = 0; i < MB; ++i) {
                qx[r][i] = qx[r][i] + skinny_dot4(q, x[i]);
            }
        }
    }
}

// qx and xs over block `blk` (skinny_q4_word over its four code words). The
// word loop is unrolled only while the body is small (NR x MB <= 8): there
// every load of the block issues up front, which the latency-bound low-m
// dispatches need, and past it the unrolled body outgrows the instruction
// cache.
template <uint MB, uint NR>
static inline void skinny_q4_block(device const uint* codes, device const bfloat* a,
                                   thread const uint* woff, uint K, uint blk, bool a_vec,
                                   thread float (&qx)[NR][MB], thread float (&xs)[MB]) {
    SKINNY_UNROLL for (uint i = 0; i < MB; ++i) {
        xs[i] = 0.0f;
        SKINNY_UNROLL for (uint r = 0; r < NR; ++r) {
            qx[r][i] = 0.0f;
        }
    }
    if (NR * MB <= 8) {
        SKINNY_UNROLL for (uint wi = 0; wi < 4; ++wi) {
            skinny_q4_word<MB, NR>(codes, a, woff, K, blk, wi, a_vec, qx, xs);
        }
    } else {
        _Pragma("clang loop unroll(disable)") for (uint wi = 0; wi < 4; ++wi) {
            skinny_q4_word<MB, NR>(codes, a, woff, K, blk, wi, a_vec, qx, xs);
        }
    }
}

// Requires N * K / 8 < 2^32 (32-bit code word offsets; the host checks).
template <uint MB, uint RR, typename CT>
static void gemm_skinny_q4_reg_body(device const uint* codes,
                                    device const bfloat* scales,
                                    device const bfloat* biases,
                                    device const bfloat* a, device CT* c,
                                    uint K, uint N, uint GS, uint tg,
                                    uint tg_size, uint simd_id, uint lane) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    static_assert(RR % 2 == 0, "the split last pass gives each half-simdgroup RR / 2 rows");
    const uint row0 = (tg * (tg_size / 32) + simd_id) * RR;
    if (row0 >= N) {
        return;  // whole simdgroup
    }
    const uint words = K / 8;
    const uint blocks = words / 4;
    const uint bpg = GS / 32;
    const uint groups = K / GS;
    const bool a_vec = ((ulong)a & 15) == 0;
    // Rows past N repeat the last row (the loads stay in bounds) and are not
    // stored.
    uint rows[RR];
    uint woff[RR];
    SKINNY_UNROLL for (uint r = 0; r < RR; ++r) {
        rows[r] = min(row0 + r, N - 1);
        woff[r] = rows[r] * words;
    }

    float acc[RR][MB];
    SKINNY_UNROLL for (uint r = 0; r < RR; ++r) {
        SKINNY_UNROLL for (uint i = 0; i < MB; ++i) {
            acc[r][i] = 0.0f;
        }
    }

    const uint rem = blocks % 32;
    const bool split = rem != 0 && rem <= 16;
    const uint full = split ? blocks - rem : blocks;
    for (uint blk = lane; blk < full; blk += 32) {
        const uint g = blk / bpg;
        float qx[RR][MB];
        float xs[MB];
        skinny_q4_block<MB, RR>(codes, a, woff, K, blk, a_vec, qx, xs);
        SKINNY_UNROLL for (uint r = 0; r < RR; ++r) {
            const float s = float(scales[rows[r] * groups + g]);
            const float b = float(biases[rows[r] * groups + g]);
            const bool sb = MB == 1 || (r & 1) != 0;
            SKINNY_UNROLL for (uint i = 0; i < MB; ++i) {
                acc[r][i] = skinny_fold(sb, s, qx[r][i], b, xs[i], acc[r][i]);
            }
        }
    }
    if (split) {
        // Lanes l and l + 16 both take block full + l % 16: the lower half
        // for rows [0, RR/2), the upper half for rows [RR/2, RR) on behalf of
        // lane l, which owns all of them. Lanes without a block compute the
        // pass's first block and discard it, keeping the shuffles uniform.
        constexpr uint HR = RR / 2;
        const uint hv = lane / 16;
        const bool active = lane % 16 < rem;
        const uint blk = active ? full + lane % 16 : full;
        const uint g = blk / bpg;
        uint hrows[HR];
        uint hoff[HR];
        SKINNY_UNROLL for (uint r = 0; r < HR; ++r) {
            hrows[r] = hv == 0 ? rows[r] : rows[HR + r];
            hoff[r] = hv == 0 ? woff[r] : woff[HR + r];
        }
        float qx[HR][MB];
        float xs[MB];
        skinny_q4_block<MB, HR>(codes, a, hoff, K, blk, a_vec, qx, xs);
        SKINNY_UNROLL for (uint r = 0; r < HR; ++r) {
            const float s = float(scales[hrows[r] * groups + g]);
            const float b = float(biases[hrows[r] * groups + g]);
            const bool sb = MB == 1 || ((hv * HR + r) & 1) != 0;
            SKINNY_UNROLL for (uint i = 0; i < MB; ++i) {
                const float up = simd_shuffle_up(acc[HR + r][i], 16);
                const float in = hv == 0 ? acc[r][i] : up;
                const float out = active ? skinny_fold(sb, s, qx[r][i], b, xs[i], in) : in;
                const float back = simd_shuffle_down(out, 16);
                if (hv == 0 && active) {
                    acc[r][i] = out;
                    acc[HR + r][i] = back;
                }
            }
        }
    }

    SKINNY_UNROLL for (uint i = 0; i < MB; ++i) {
        SKINNY_UNROLL for (uint r = 0; r < RR; ++r) {
            const float sum = simd_sum(acc[r][i]);
            if (lane == 0 && row0 + r < N) {
                c[(ulong)i * N + row0 + r] = CT(sum);
            }
        }
    }
}

// 8-bit register-A body: a uint4 block holds 16 elements (four words of
// four); requires M == MB, K % 16 == 0 and GS % 16 == 0.
template <uint MB, typename CT>
static void gemm_skinny_q8_reg_body(device const uint* codes,
                                    device const bfloat* scales,
                                    device const bfloat* biases,
                                    device const bfloat* a, device CT* c,
                                    uint K, uint N, uint GS, uint tg,
                                    uint tg_size, uint simd_id, uint lane) {
    const uint row0 = (tg * (tg_size / 32) + simd_id) * SKINNY_REG_ROWS;
    if (row0 >= N) {
        return;  // whole simdgroup
    }
    const bool has1 = row0 + 1 < N;
    const uint row1 = has1 ? row0 + 1 : row0;
    const uint words = K / 4;
    const uint blocks = words / 4;
    const uint bpg = GS / 16;
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
        // Two words (eight elements) per step: one 16-byte activation load.
        for (uint wp = 0; wp < 2; ++wp) {
            const float4 q0lo = skinny_unpack_q8(v0[2 * wp]);
            const float4 q0hi = skinny_unpack_q8(v0[2 * wp + 1]);
            const float4 q1lo = skinny_unpack_q8(v1[2 * wp]);
            const float4 q1hi = skinny_unpack_q8(v1[2 * wp + 1]);
            for (uint i = 0; i < MB; ++i) {
                float4 xlo, xhi;
                skinny_load_x8(a + (ulong)i * K + blk * 16 + wp * 8, a_vec, xlo, xhi);
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
// f32 output: the `--q8-dense` LM head's logits (skinny.rs routes only wide
// outputs here, so the 8-bit routers keep their staged f32 kernels).
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_f32_reg_m1, 1, float)
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_f32_reg_m2, 2, float)
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_f32_reg_m3, 3, float)
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_f32_reg_m4, 4, float)
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_f32_reg_m5, 5, float)
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_f32_reg_m6, 6, float)
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_f32_reg_m7, 7, float)
GEMM_SKINNY_Q8_REG(gemm_skinny_q8_f32_reg_m8, 8, float)

// Register-A variants require M to match the kernel suffix; RR (weight rows
// per simdgroup) must match Q4_REG_ROWS_PER_SG in skinny.rs.
#define GEMM_SKINNY_Q4_REG(NAME, MB, RR, CT)                                   \
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
        gemm_skinny_q4_reg_body<MB, RR, CT>(codes, scales, biases, a, c, K, N, \
                                            GS, tg, tg_size, simd_id, lane);   \
    }

GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m1, 1, 2, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m2, 2, 2, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m3, 3, 4, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m4, 4, 4, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m5, 5, 4, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m6, 6, 4, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m7, 7, 4, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_bf16_reg_m8, 8, 4, bfloat)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m1, 1, 2, float)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m2, 2, 2, float)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m3, 3, 4, float)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m4, 4, 4, float)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m5, 5, 4, float)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m6, 6, 4, float)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m7, 7, 4, float)
GEMM_SKINNY_Q4_REG(gemm_skinny_q4_f32_reg_m8, 8, 4, float)
