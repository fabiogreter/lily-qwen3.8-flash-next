// Unit-test-only primitive used to validate the elementwise binary-op harness.
kernel void mul_bf16(device const bfloat* a   [[buffer(0)]],
                     device const bfloat* b   [[buffer(1)]],
                     device bfloat*       out [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    out[gid] = bfloat(float(a[gid]) * float(b[gid]));
}

// Memory-bandwidth probes (ignored timing test `memory_bandwidth_probe`):
// coalesced 16-byte streams over a buffer far larger than the system cache.
kernel void bw_fill_u4(device uint4* dst [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    uint h = gid * 2654435761u;
    dst[gid] = uint4(h, h ^ 0x9e3779b9u, h * 3u + 1u, ~h);
}

kernel void bw_read_u4(device const uint4* src [[buffer(0)]],
                       device uint*        out [[buffer(1)]],
                       constant uint&      per_thread [[buffer(2)]],
                       uint gid [[thread_position_in_grid]],
                       uint gsz [[threads_per_grid]]) {
    uint4 acc = uint4(0);
    for (uint i = 0; i < per_thread; ++i) acc ^= src[gid + i * gsz];
    uint r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    // Practically never true; keeps the loads observable.
    if (r == 0x12345678u) out[gid & 1023u] = r;
}

kernel void bw_copy_u4(device const uint4* src [[buffer(0)]],
                       device uint4*       dst [[buffer(1)]],
                       constant uint&      per_thread [[buffer(2)]],
                       uint gid [[thread_position_in_grid]],
                       uint gsz [[threads_per_grid]]) {
    for (uint i = 0; i < per_thread; ++i) dst[gid + i * gsz] = src[gid + i * gsz];
}
