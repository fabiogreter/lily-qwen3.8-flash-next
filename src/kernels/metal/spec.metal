#include <metal_stdlib>
using namespace metal;

// Speculative-step control on the GPU: how many drafts the trunk confirmed,
// and every value the draft pass needs that depends on it, written as words
// for later dispatches of the same pass to read as inline arguments.
//
// Slot layout (each slot is `stride` words apart):
//   0            accepted count a (leading rows whose draw equals the draft)
//   1            a + 1 (rows kept by the rollback)
//   2 + 3i       position of chain row i: pos0 + a + 1 + i
//   3 + 3i       indexer block that row completes: position / ratio
//   4 + 3i       1 when it completes a block ((position + 1) % ratio == 0), else 0
kernel void spec_accept(device const uint* draws  [[buffer(0)]],  // [m]: the trunk's draw per verified row
                        device const uint* ids    [[buffer(1)]],  // [m]: the verified tokens (pending, drafts)
                        device uint*       ctrl   [[buffer(2)]],
                        constant uint&     m      [[buffer(3)]],
                        constant uint&     pos0   [[buffer(4)]],
                        constant uint&     ratio  [[buffer(5)]],
                        constant uint&     chain  [[buffer(6)]],  // chain rows to describe
                        constant uint&     stride [[buffer(7)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid != 0) {
        return;
    }
    uint a = 0;
    while (a + 1 < m && draws[a] == ids[a + 1]) {
        ++a;
    }
    ctrl[0] = a;
    ctrl[stride] = a + 1;
    for (uint i = 0; i < chain; ++i) {
        const uint p = pos0 + a + 1 + i;
        ctrl[(2 + 3 * i) * stride] = p;
        ctrl[(3 + 3 * i) * stride] = p / ratio;
        ctrl[(4 + 3 * i) * stride] = ((p + 1) % ratio == 0) ? 1u : 0u;
    }
}

// dst[0..words] = src[row, 0..words] for a row index the GPU may supply;
// nothing is written when row >= rows.
kernel void copy_row_u32(device const uint* src   [[buffer(0)]],  // [rows, words]
                         device uint*       dst   [[buffer(1)]],  // [words]
                         constant uint&     words [[buffer(2)]],
                         constant uint&     rows  [[buffer(3)]],
                         constant uint&     row   [[buffer(4)]],
                         uint gid [[thread_position_in_grid]]) {
    if (row >= rows || gid >= words) {
        return;
    }
    dst[gid] = src[(ulong)row * words + gid];
}
