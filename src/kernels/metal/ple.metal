// Per-Layer (n-gram) Embedding kernels for Qwen3.8-Flash-Next: the quantized
// table gather (ids are hashed on the host), the per-stream gate, and the dilated
// depthwise conv that adds local context. Streams are laid out [rows, G*H].
#include <metal_stdlib>
using namespace metal;

#define TG 256
#define PLE_MAX_CONTEXT 16  // (kernel-1)*dilation entries the conv window holds

// Dequantizes one Q4 code word to eight BF16 values.
static inline void ple_store_word_q4(uint word, float s, float b,
                                     device bfloat4* out, uint w) {
    float4 lo = float4(float((word >> 0) & 0xF), float((word >> 4) & 0xF),
                       float((word >> 8) & 0xF), float((word >> 12) & 0xF));
    float4 hi = float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                       float((word >> 24) & 0xF), float((word >> 28) & 0xF));
    out[2 * w] = bfloat4(lo * s + b);
    out[2 * w + 1] = bfloat4(hi * s + b);
}

// out[r, j*K .. (j+1)*K] = table[ids[r, j]]: one thread per (code word, head, row).
kernel void ple_gather_q4_bf16(device const uint*   codes  [[buffer(0)]],
                               device const bfloat* scales [[buffer(1)]],
                               device const bfloat* biases [[buffer(2)]],
                               device const uint*   ids    [[buffer(3)]],  // [M, NH]
                               device bfloat*       out    [[buffer(4)]],  // [M, NH*K]
                               constant uint&       K      [[buffer(5)]],
                               constant uint&       GS     [[buffer(6)]],
                               constant uint&       NH     [[buffer(7)]],
                               uint3 gid [[thread_position_in_grid]]) {
    const uint words = K / 8;
    const uint w = gid.x;
    const uint j = gid.y;
    const uint r = gid.z;
    if (w >= words) {
        return;
    }
    const ulong row = ids[(ulong)r * NH + j];
    const uint groups = K / GS;
    const uint g = w / (GS / 8);
    const float s = float(scales[row * groups + g]);
    const float b = float(biases[row * groups + g]);
    const uint word = codes[row * words + w];
    device bfloat* dst = out + ((ulong)r * NH + j) * K;
    ple_store_word_q4(word, s, b, (device bfloat4*)dst, w);
}

// gate_g = <key[r, g], query[r, g]> / sqrt(H), signed-sqrt'ed, then
// gated[r, g*H + i] = sigmoid(gate_g) * value[r, i]. One threadgroup per (row, g).
kernel void ple_gate_value_bf16(device const bfloat* key   [[buffer(0)]],  // [rows, G*H] normed
                                device const bfloat* query [[buffer(1)]],  // [rows, G*H] normed
                                device const bfloat* value [[buffer(2)]],  // [rows, H]
                                device bfloat*       gated [[buffer(3)]],  // [rows, G*H]
                                constant uint&       H     [[buffer(4)]],
                                constant uint&       G     [[buffer(5)]],
                                constant float&      inv_sqrt_h [[buffer(6)]],
                                uint seg  [[threadgroup_position_in_grid]],
                                uint tid  [[thread_index_in_threadgroup]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float gate_s;

    const ulong base = (ulong)seg * H;
    const ulong vrow = (ulong)(seg / G) * H;
    float acc = 0.0f;
    for (uint i = tid; i < H; i += TG) {
        acc += float(key[base + i]) * float(query[base + i]);
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
        const float gate = total * inv_sqrt_h;
        const float mag = sqrt(max(abs(gate), 1e-6f));
        const float signed_gate = gate < 0.0f ? -mag : mag;
        gate_s = 1.0f / (1.0f + exp(-signed_gate));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = gate_s;
    for (uint i = tid; i < H; i += TG) {
        gated[base + i] = bfloat(s * float(value[vrow + i]));
    }
}

// Dilated causal depthwise conv over x[M, C] (kernel KD, dilation DIL), then
// hyper[m, c] += base[m, c] + silu(conv). The window holds the S = (KD-1)*DIL
// inputs before the chunk, oldest first; win_out receives the post-chunk window.
kernel void ple_conv1d_prefill_bf16(device const bfloat* win_in  [[buffer(0)]],  // [C, S]
                                    device bfloat*       win_out [[buffer(1)]],  // [C, S]
                                    device const bfloat* x       [[buffer(2)]],  // [M, C]
                                    device const bfloat* w       [[buffer(3)]],  // [KD, C]
                                    device const bfloat* base    [[buffer(4)]],  // [M, C]
                                    device bfloat*       hyper   [[buffer(5)]],  // [M, C]
                                    constant uint&       C       [[buffer(6)]],
                                    constant uint&       KD      [[buffer(7)]],
                                    constant uint&       DIL     [[buffer(8)]],
                                    constant uint&       M       [[buffer(9)]],
                                    uint c [[thread_position_in_grid]]) {
    const uint S = (KD - 1) * DIL;
    float win[PLE_MAX_CONTEXT];
    for (uint s = 0; s < S; ++s) {
        win[s] = float(win_in[c * S + s]);
    }
    for (uint m = 0; m < M; ++m) {
        const float xc = float(x[(ulong)m * C + c]);
        float acc = xc * float(w[(KD - 1) * C + c]);
        for (uint t = 0; t + 1 < KD; ++t) {
            // Tap t looks back (KD-1-t)*DIL tokens; the window's newest entry
            // is one token back.
            acc += win[S - (KD - 1 - t) * DIL] * float(w[t * C + c]);
        }
        const float conv = acc / (1.0f + exp(-acc));
        const ulong at = (ulong)m * C + c;
        hyper[at] = bfloat(float(hyper[at]) + float(base[at]) + conv);
        for (uint s = 0; s + 1 < S; ++s) {
            win[s] = win[s + 1];
        }
        win[S - 1] = xc;
    }
    for (uint s = 0; s < S; ++s) {
        win_out[c * S + s] = bfloat(win[s]);
    }
}

// Single-token variant of the above; the window is shifted in place.
kernel void ple_conv1d_step_bf16(device bfloat*       window [[buffer(0)]],  // [C, S]
                                 device const bfloat* x      [[buffer(1)]],  // [C]
                                 device const bfloat* w      [[buffer(2)]],  // [KD, C]
                                 device const bfloat* base   [[buffer(3)]],  // [C]
                                 device bfloat*       hyper  [[buffer(4)]],  // [C]
                                 constant uint&       C      [[buffer(5)]],
                                 constant uint&       KD     [[buffer(6)]],
                                 constant uint&       DIL    [[buffer(7)]],
                                 uint c [[thread_position_in_grid]]) {
    const uint S = (KD - 1) * DIL;
    const float xc = float(x[c]);
    float acc = xc * float(w[(KD - 1) * C + c]);
    for (uint t = 0; t + 1 < KD; ++t) {
        acc += float(window[c * S + S - (KD - 1 - t) * DIL]) * float(w[t * C + c]);
    }
    const float conv = acc / (1.0f + exp(-acc));
    hyper[c] = bfloat(float(hyper[c]) + float(base[c]) + conv);
    for (uint s = 0; s + 1 < S; ++s) {
        window[c * S + s] = window[c * S + s + 1];
    }
    window[c * S + S - 1] = bfloat(xc);
}
