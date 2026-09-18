// Unit-test-only entry point for the unfused recurrent-step oracle. The shared
// body comes from gdn.metal, which is concatenated before this file by gdn.rs
// under cfg(test).
#define GDN_STEP_WRAPPER(NAME, STATE_T)                                       \
kernel void NAME(device const bfloat* q       [[buffer(0)]],                  \
                 device const bfloat* k       [[buffer(1)]],                  \
                 device const bfloat* v       [[buffer(2)]],                  \
                 device const bfloat* a       [[buffer(3)]],                  \
                 device const bfloat* b       [[buffer(4)]],                  \
                 device const float*  a_log   [[buffer(5)]],                  \
                 device const bfloat* dt_bias [[buffer(6)]],                  \
                 device STATE_T*      state   [[buffer(7)]],                  \
                 device bfloat*       out     [[buffer(8)]],                  \
                 constant float&      scale   [[buffer(9)]],                  \
                 constant uint&       vpk     [[buffer(10)]],                 \
                 uint h    [[threadgroup_position_in_grid]],                  \
                 uint tid  [[thread_index_in_threadgroup]],                   \
                 uint sg   [[simdgroup_index_in_threadgroup]],                \
                 uint lane [[thread_index_in_simdgroup]]) {                   \
    threadgroup float q_norm[DIM];                                            \
    threadgroup float k_norm[DIM];                                            \
    threadgroup float part_q[DIM / 32];                                       \
    threadgroup float part_k[DIM / 32];                                       \
    threadgroup float gates[4];                                               \
    const float o = gdn_step_body(q, k, v, a, b, a_log, dt_bias, state,      \
                                  scale, vpk, h, tid, sg, lane, q_norm,       \
                                  k_norm, part_q, part_k, gates);             \
    out[h * DIM + tid] = bfloat(o);                                           \
}

GDN_STEP_WRAPPER(gdn_step, float)
#undef GDN_STEP_WRAPPER

// Streaming reference for the decode step's state traffic
// (`gdn_state_stream_timing`): a threadgroup of C * RG threads per (head,
// column group); thread (rg, c) reads its DIM/RG rows of column c in blocks
// of eight loads, scales them and writes them back. No prologue, no
// reductions: the bandwidth a given grid shape reaches on the step's access
// pattern.
kernel void gdn_state_stream(device float*   state [[buffer(0)]],
                             device float*   out   [[buffer(1)]],
                             constant uint&  C     [[buffer(2)]],
                             constant uint&  RG    [[buffer(3)]],
                             constant float& decay [[buffer(4)]],
                             uint tg  [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    const uint groups_per_head = DIM / C;
    const uint h = tg / groups_per_head;
    const uint cg = tg % groups_per_head;
    const uint col = cg * C + tid % C;
    const uint rg = tid / C;
    const uint rows = DIM / RG;
    device float* st = state + (ulong)h * DIM * DIM + (ulong)(rg * rows) * DIM + col;
    float acc = 0.0f;
    for (uint r0 = 0; r0 < rows; r0 += 8) {
        float sv[8];
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < 8; ++j) {
            sv[j] = st[(r0 + j) * DIM];
        }
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < 8; ++j) {
            const float u = sv[j] * decay;
            st[(r0 + j) * DIM] = u;
            acc += u;
        }
    }
    if (acc == 12345.678f) {
        out[col] = acc;
    }
}
