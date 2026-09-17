// Token sampling on the GPU: penalties, temperature, top-k by a two-level
// bucket search on the distance below the maximum logit (boundary resolution
// 2^-24 of the searched range), top-p / min-p over the sorted candidates, and
// an inverse-CDF draw from a counter-based hash RNG. Only the chosen id ever
// crosses to the host, which keeps the pipelined decode loop free of logits
// round-trips.
//
// Two kernels: a wide one applies penalties and temperature and reduces the
// maximum, then one threadgroup of SAMPLE_TG threads does the selection and
// the draw. Everything is deterministic given the logits, the histogram and
// (seed, step); ties at the top-k boundary are broken by a fixed thread order.
#include <metal_stdlib>
using namespace metal;

#define SAMPLE_TG 1024
#define SAMPLE_PREP_TG 256
#define SAMPLE_PREP_GROUPS 64
// Largest candidate set: top-k requests above this are capped (the mass of
// tokens ranked past 1 024 is negligible at any temperature a chat model runs
// at, and the cap is documented in the API).
#define SAMPLE_K_CAP 1024
#define SAMPLE_BINS 4096
// The largest k the two-phase selection serves (its merge is over
// groups * k candidates; the host sets the group count).
#define SAMPLE_LOCAL_K_MAX 64

// Monotonic float -> uint key: greater floats give greater keys (used to
// order the candidate set).
static inline uint sample_key(float v) {
    const uint u = as_type<uint>(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exclusive prefix sum of `flag` over the threadgroup; `*total` receives the sum.
static inline uint sample_scan(uint flag, threadgroup uint* sums, threadgroup uint* total,
                               uint tid, uint sg, uint lane) {
    const uint local = simd_prefix_exclusive_sum(flag);
    const uint sg_total = simd_sum(flag);
    if (lane == 0) {
        sums[sg] = sg_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0;
    for (uint s = 0; s < sg; ++s) {
        before += sums[s];
    }
    if (tid == 0) {
        uint t = 0;
        for (uint s = 0; s < SAMPLE_TG / 32; ++s) {
            t += sums[s];
        }
        *total = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return before + local;
}

// Bucket of a distance-below-max `d` at resolution `inv_w` (bins per unit).
static inline uint sample_bin(float d, float inv_w) {
    return min(uint(d * inv_w), uint(SAMPLE_BINS - 1));
}

// Finds the bin in which the cumulative count (from bin 0 upwards) reaches
// `k`. Writes the bin and the count before it to threadgroup scalars and the
// grand total to `*total`. Each of the SAMPLE_TG threads owns 4 bins.
static inline void sample_locate(threadgroup atomic_uint* hist, uint k,
                                 threadgroup uint* sums, threadgroup uint* total,
                                 threadgroup uint* out_bin, threadgroup uint* out_before,
                                 uint tid, uint sg, uint lane) {
    uint c[4];
    uint mine = 0;
    for (uint j = 0; j < 4; ++j) {
        c[j] = atomic_load_explicit(&hist[tid * 4 + j], memory_order_relaxed);
        mine += c[j];
    }
    const uint before = sample_scan(mine, sums, total, tid, sg, lane);
    if (before < k && before + mine >= k) {
        uint cum = before;
        for (uint j = 0; j < 4; ++j) {
            if (cum + c[j] >= k) {
                *out_bin = tid * 4 + j;
                *out_before = cum;
                break;
            }
            cum += c[j];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// splitmix64 finalizer over (seed, step): the same (seed, step) always draws
// the same uniform, so a request with `seed` replays exactly.
static inline float sample_uniform(uint seed_lo, uint seed_hi, uint step) {
    ulong z = (ulong(seed_hi) << 32) | ulong(seed_lo);
    z += ulong(step) * 0x9E3779B97F4A7C15ul;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    z ^= z >> 31;
    // 24 significant bits keep the draw strictly below 1.0 as a float.
    return float(uint(z >> 40)) * (1.0f / 16777216.0f);
}

// Penalties follow the OpenAI / vLLM conventions over generated tokens only:
// repetition divides positive logits (multiplies negative ones) for tokens
// already emitted, presence subtracts a flat amount, frequency subtracts per
// occurrence. Writes the adjusted logits and one running maximum per
// threadgroup.
kernel void sample_prepare_f32(device const float* logits    [[buffer(0)]],  // [V]
                               device float*       adjusted  [[buffer(1)]],  // [V]
                               device const uint*  counts    [[buffer(2)]],  // [V]
                               device float*       maxima    [[buffer(3)]],  // [SAMPLE_PREP_GROUPS]
                               constant float&     temperature [[buffer(4)]],
                               constant float&     presence  [[buffer(5)]],
                               constant float&     frequency [[buffer(6)]],
                               constant float&     repetition [[buffer(7)]],
                               constant uint&      V         [[buffer(8)]],
                               constant uint&      use_penalties [[buffer(9)]],
                               uint gid  [[threadgroup_position_in_grid]],
                               uint tid  [[thread_index_in_threadgroup]],
                               uint sg   [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[SAMPLE_PREP_TG / 32];
    const float inv_t = 1.0f / temperature;
    const uint chunk = (V + SAMPLE_PREP_GROUPS - 1) / SAMPLE_PREP_GROUPS;
    const uint begin = gid * chunk;
    const uint end = min(begin + chunk, V);
    float local_max = -INFINITY;
    for (uint i = begin + tid; i < end; i += SAMPLE_PREP_TG) {
        float l = logits[i];
        if (use_penalties) {
            const uint c = counts[i];
            if (c > 0u) {
                l = l > 0.0f ? l / repetition : l * repetition;
                l -= presence + frequency * float(c);
            }
        }
        l *= inv_t;
        adjusted[i] = l;
        local_max = max(local_max, l);
    }
    local_max = simd_max(local_max);
    if (lane == 0) {
        partial[sg] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = partial[0];
        for (uint s = 1; s < SAMPLE_PREP_TG / 32; ++s) {
            m = max(m, partial[s]);
        }
        maxima[gid] = m;
    }
}

// Inclusive prefix sum of one float per thread over the threadgroup; the
// total is written to `*total`. Callers separate consecutive scans by a
// barrier (`sg_sums` is rewritten).
static inline float sample_scan_f(float v, threadgroup float* sg_sums, thread float& total,
                                  uint sg, uint lane) {
    const float local = simd_prefix_inclusive_sum(v);
    if (lane == 31) {
        sg_sums[sg] = local;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float before = 0.0f;
    for (uint s = 0; s < sg; ++s) {
        before += sg_sums[s];
    }
    total = before;
    for (uint s = sg; s < SAMPLE_TG / 32; ++s) {
        total += sg_sums[s];
    }
    return before + local;
}

// An element of the selection's input: `adjusted[i]` directly, or through
// `map` (the merged local candidates of the two-phase path, a padding entry
// of ~0 reading as -inf).
template <bool mapped>
static inline float sample_at(device const float* adjusted, device const uint* map, uint i) {
    if (!mapped) {
        return adjusted[i];
    }
    const uint id = map[i];
    return id == 0xFFFFFFFFu ? -INFINITY : adjusted[id];
}

// The vocabulary id behind candidate slot `i`.
static inline uint sample_id(device const uint* map, bool mapped, uint i) {
    return mapped ? map[i] : i;
}

// The maximum of the adjusted logits, from the prepare kernel's partials.
static inline float sample_top(device const float* maxima) {
    float top = -INFINITY;
    for (uint g = 0; g < SAMPLE_PREP_GROUPS; ++g) {
        top = max(top, maxima[g]);
    }
    return top;
}

// The kept candidate set of `adjusted` under the reference `top` (its
// maximum, or the slice's for the local phase; the search buckets the
// distance below it): its top-k, sorted descending, then
// truncated by top-p / min-p to a prefix of `n_keep` ids. On return
// `cand_id[0..n_keep)` holds the ids, `cum[i]` the inclusive prefix of the
// unnormalised probabilities exp(adjusted - top), and each thread `tid <
// n_keep` its candidate's `p` and prefix `c`. `cum[n_keep - 1]` is the kept
// mass. Shared by the plain draw, the draft draw and the speculative
// verify draw.
template <bool mapped>
static void sample_select(device const float* adjusted, device const uint* map, float top,
                          float top_p, float min_p, uint top_k, uint V,
                          threadgroup atomic_uint* hist, threadgroup uint* sums,
                          threadgroup uint* total, threadgroup uint* bin_s,
                          threadgroup uint* before_s, threadgroup uint* cand_key,
                          threadgroup uint* cand_id, threadgroup float* cum,
                          threadgroup float* sg_sums, threadgroup uint* n_keep_s,
                          thread float& p, thread float& c, thread uint& n_keep,
                          uint tid, uint sg, uint lane) {
    uint k = min(min(top_k == 0u ? uint(SAMPLE_K_CAP) : top_k, uint(SAMPLE_K_CAP)), V);

    // 2. Coarse search: bucket the distance below the maximum over a range
    //    (widened when fewer than k candidates fall inside it). Elements far
    //    below the maximum never touch an atomic, and the near ones spread
    //    over 4 096 bins, so the histogram has no hot spot.
    float range = 32.0f;
    float inv_w = float(SAMPLE_BINS) / range;
    for (uint attempt = 0; attempt < 4; ++attempt) {
        for (uint i = tid; i < SAMPLE_BINS; i += SAMPLE_TG) {
            atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        }
        if (tid == 0) {
            *bin_s = SAMPLE_BINS;
            *before_s = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Four elements per iteration so their loads overlap (the sweeps
        // over V are latency-bound on one threadgroup otherwise).
        for (uint i = tid; i < V; i += 4 * SAMPLE_TG) {
            float d[4];
            for (uint u = 0; u < 4; ++u) {
                const uint j = i + u * SAMPLE_TG;
                d[u] = j < V ? top - sample_at<mapped>(adjusted, map, j) : INFINITY;
            }
            for (uint u = 0; u < 4; ++u) {
                if (d[u] < range) {
                    atomic_fetch_add_explicit(&hist[sample_bin(d[u], inv_w)], 1u,
                                              memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sample_locate(hist, k, sums, total, bin_s, before_s, tid, sg, lane);
        if (*total >= k) {
            break;
        }
        if (attempt == 3) {
            // Fewer than k finite candidates in a huge range: take them all.
            k = *total;
            sample_locate(hist, k, sums, total, bin_s, before_s, tid, sg, lane);
            break;
        }
        range *= 8.0f;
        inv_w = float(SAMPLE_BINS) / range;
    }
    const uint coarse_bin = *bin_s;
    const uint before_coarse = *before_s;
    const float lo = float(coarse_bin) / inv_w;
    const float inv_w2 = inv_w * float(SAMPLE_BINS);

    // 3. Refine the boundary bucket at 4 096x finer resolution.
    for (uint i = tid; i < SAMPLE_BINS; i += SAMPLE_TG) {
        atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < V; i += 4 * SAMPLE_TG) {
        float d[4];
        for (uint u = 0; u < 4; ++u) {
            const uint j = i + u * SAMPLE_TG;
            d[u] = j < V ? top - sample_at<mapped>(adjusted, map, j) : INFINITY;
        }
        for (uint u = 0; u < 4; ++u) {
            if (d[u] < range && sample_bin(d[u], inv_w) == coarse_bin) {
                atomic_fetch_add_explicit(&hist[sample_bin(d[u] - lo, inv_w2)], 1u,
                                          memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sample_locate(hist, k - before_coarse, sums, total, bin_s, before_s, tid, sg, lane);
    const uint fine_bin = *bin_s;
    const uint n_definite = before_coarse + *before_s;

    // 4. Compaction. Each thread counts its definite takes and its boundary
    //    ties, a scan assigns disjoint slots, and a second sweep writes them.
    //    Ties are taken in thread order until the count is met: deterministic,
    //    and irrelevant for the draw since tied logits share a probability.
    uint my_def = 0, my_tie = 0;
    for (uint i = tid; i < V; i += 4 * SAMPLE_TG) {
        float d[4];
        for (uint u = 0; u < 4; ++u) {
            const uint j = i + u * SAMPLE_TG;
            d[u] = j < V ? top - sample_at<mapped>(adjusted, map, j) : INFINITY;
        }
        for (uint u = 0; u < 4; ++u) {
            if (d[u] < range) {
                const uint b = sample_bin(d[u], inv_w);
                if (b < coarse_bin) {
                    ++my_def;
                } else if (b == coarse_bin) {
                    const uint f = sample_bin(d[u] - lo, inv_w2);
                    my_def += f < fine_bin ? 1u : 0u;
                    my_tie += f == fine_bin ? 1u : 0u;
                }
            }
        }
    }
    const uint def_base = sample_scan(my_def, sums, total, tid, sg, lane);
    const uint tie_base = sample_scan(my_tie, sums, total, tid, sg, lane);
    uint def_at = def_base;
    uint tie_at = n_definite + tie_base;
    for (uint i = tid; i < V; i += 4 * SAMPLE_TG) {
        float val[4];
        for (uint u = 0; u < 4; ++u) {
            const uint j = i + u * SAMPLE_TG;
            val[u] = j < V ? sample_at<mapped>(adjusted, map, j) : -INFINITY;
        }
        // In ascending element order within the thread, as before.
        for (uint u = 0; u < 4; ++u) {
            const uint j = i + u * SAMPLE_TG;
            const float d = top - val[u];
            if (d < range) {
                const uint b = sample_bin(d, inv_w);
                uint cls = 0;
                if (b < coarse_bin) {
                    cls = 1;
                } else if (b == coarse_bin) {
                    const uint f = sample_bin(d - lo, inv_w2);
                    cls = f < fine_bin ? 1u : (f == fine_bin ? 2u : 0u);
                }
                if (cls == 1) {
                    cand_key[def_at] = sample_key(val[u]);
                    cand_id[def_at] = j;
                    ++def_at;
                } else if (cls == 2 && tie_at < k) {
                    cand_key[tie_at] = sample_key(val[u]);
                    cand_id[tie_at] = j;
                    ++tie_at;
                }
            }
        }
    }
    const uint n = k;
    uint sort_size = 1;
    while (sort_size < n) {
        sort_size <<= 1;
    }
    for (uint i = n + tid; i < sort_size; i += SAMPLE_TG) {
        cand_key[i] = 0u;
        cand_id[i] = 0xFFFFFFFFu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. Bitonic sort of the candidates, descending by key. Up to 32
    //    candidates sort inside simdgroup 0 with lane shuffles, without
    //    threadgroup barriers per round.
    if (sort_size <= 32u) {
        if (sg == 0) {
            uint key = lane < sort_size ? cand_key[lane] : 0u;
            uint id = lane < sort_size ? cand_id[lane] : 0xFFFFFFFFu;
            for (uint size = 2; size <= sort_size; size <<= 1) {
                for (uint stride = size >> 1; stride > 0; stride >>= 1) {
                    const uint partner = lane ^ stride;
                    const uint okey = simd_shuffle_xor(key, stride);
                    const uint oid = simd_shuffle_xor(id, stride);
                    const bool descending = (lane & size) == 0;
                    const bool lower = lane < partner;
                    // The lower lane keeps the larger key of a descending
                    // pair, the smaller of an ascending one.
                    const bool take_other = lower == descending ? okey > key : okey < key;
                    if (take_other) {
                        key = okey;
                        id = oid;
                    }
                }
            }
            if (lane < sort_size) {
                cand_key[lane] = key;
                cand_id[lane] = id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else
    for (uint size = 2; size <= sort_size; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            const uint partner = tid ^ stride;
            if (tid < sort_size && partner > tid) {
                const bool descending = (tid & size) == 0;
                const uint a = cand_key[tid];
                const uint b = cand_key[partner];
                const bool swap = descending ? (a < b) : (a > b);
                if (swap) {
                    const uint ia = cand_id[tid];
                    cand_key[tid] = b;
                    cand_key[partner] = a;
                    cand_id[tid] = cand_id[partner];
                    cand_id[partner] = ia;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // 6. Softmax over the candidates and inclusive prefix sums of probability.
    const float top_val = sample_at<mapped>(adjusted, map, cand_id[0]);
    p = 0.0f;
    if (tid < n) {
        p = exp(sample_at<mapped>(adjusted, map, cand_id[tid]) - top_val);
    }
    float mass;
    c = sample_scan_f(p, sg_sums, mass, sg, lane);
    if (tid < SAMPLE_K_CAP) {
        cum[tid] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total_mass = cum[n - 1];

    // 7. Nucleus / min-p truncation: both keep a prefix of the sorted list.
    const bool keep = tid < n
        && (tid == 0 || ((c - p) < top_p * total_mass && p >= min_p));
    sample_scan(keep ? 1u : 0u, sums, n_keep_s, tid, sg, lane);
    n_keep = *n_keep_s;
}

// Inverse-CDF draw over the inclusive prefix sums `prefix[0..n_keep)`: the
// thread whose interval [prefix[tid - 1], prefix[tid]) holds `target`
// returns its slot (the last one also takes anything at or past its
// start), the others SAMPLE_K_CAP. The intervals tile the mass exactly.
static inline uint sample_pick(float target, threadgroup const float* prefix, uint n_keep,
                               uint tid) {
    if (tid < n_keep) {
        const float lo_c = tid == 0 ? 0.0f : prefix[tid - 1];
        const float c = prefix[tid];
        const bool last = tid == n_keep - 1;
        if ((lo_c <= target && target < c) || (last && target >= lo_c)) {
            return tid;
        }
    }
    return SAMPLE_K_CAP;
}

#define SAMPLE_TG_STATE                                                        \
    threadgroup atomic_uint hist[SAMPLE_BINS];                                 \
    threadgroup uint sums[SAMPLE_TG / 32];                                     \
    threadgroup uint total;                                                    \
    threadgroup uint bin_s;                                                    \
    threadgroup uint before_s;                                                 \
    threadgroup uint cand_key[SAMPLE_K_CAP];                                   \
    threadgroup uint cand_id[SAMPLE_K_CAP];                                    \
    threadgroup float cum[SAMPLE_K_CAP];                                       \
    threadgroup float sg_sums[SAMPLE_TG / 32];                                 \
    threadgroup uint n_keep_s;

#define SAMPLE_SELECT_ARGS                                                     \
    hist, sums, &total, &bin_s, &before_s, cand_key, cand_id, cum, sg_sums,    \
    &n_keep_s

// The selection through the map or directly, on the runtime flag, each an
// instantiation of its own so the direct path carries no per-element branch.
#define SAMPLE_SELECT(MAPPED, ADJ, MAP, MAXIMA, TOP_P, MIN_P, TOP_K, V, P, C, N_KEEP) \
    if (MAPPED) {                                                              \
        sample_select<true>(ADJ, MAP, sample_top(MAXIMA), TOP_P, MIN_P, TOP_K, \
                            V, SAMPLE_SELECT_ARGS, P, C, N_KEEP, tid, sg, lane); \
    } else {                                                                   \
        sample_select<false>(ADJ, MAP, sample_top(MAXIMA), TOP_P, MIN_P, TOP_K, \
                             V, SAMPLE_SELECT_ARGS, P, C, N_KEEP, tid, sg, lane); \
    }

// Selects the top-k of `adjusted`, truncates by top-p / min-p, draws one id
// into `out[0]` and bumps its `counts` entry.
kernel void sample_f32(device const float* adjusted  [[buffer(0)]],  // [V]
                       device const float* maxima    [[buffer(1)]],  // [SAMPLE_PREP_GROUPS]
                       device uint*        counts    [[buffer(2)]],  // [V]
                       device uint*        out       [[buffer(3)]],  // [1]
                       device const uint*  map       [[buffer(4)]],  // [V] or unused
                       constant float&     top_p     [[buffer(5)]],
                       constant float&     min_p     [[buffer(6)]],
                       constant uint&      top_k     [[buffer(7)]],
                       constant uint&      V         [[buffer(8)]],
                       constant uint&      seed_lo   [[buffer(9)]],
                       constant uint&      seed_hi   [[buffer(10)]],
                       constant uint&      step      [[buffer(11)]],
                       constant uint&      use_penalties [[buffer(12)]],
                       constant uint&      mapped    [[buffer(13)]],
                       uint tid  [[thread_index_in_threadgroup]],
                       uint sg   [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    SAMPLE_TG_STATE
    float p, c;
    uint n_keep;
    SAMPLE_SELECT(mapped != 0u, adjusted, map, maxima, top_p, min_p, top_k, V, p, c, n_keep)

    // 8. Inverse-CDF draw within the kept mass.
    const float target = sample_uniform(seed_lo, seed_hi, step) * cum[n_keep - 1];
    if (sample_pick(target, cum, n_keep, tid) == tid) {
        const uint chosen = sample_id(map, mapped != 0u, cand_id[tid]);
        out[0] = chosen;
        if (use_penalties) {
            counts[chosen] += 1u;
        }
    }
}

// The draft head's draw: as sample_f32, and the kept distribution it drew
// from goes to q_ids / q_probs (normalised over the kept mass) with its size
// in q_n[0], for the verify pass's speculative draw against this proposal.
kernel void sample_draft_f32(device const float* adjusted  [[buffer(0)]],  // [V]
                             device const float* maxima    [[buffer(1)]],  // [SAMPLE_PREP_GROUPS]
                             device uint*        counts    [[buffer(2)]],  // [V]
                             device uint*        out       [[buffer(3)]],  // [1]
                             device uint*        q_ids     [[buffer(4)]],  // [SAMPLE_K_CAP]
                             device float*       q_probs   [[buffer(5)]],  // [SAMPLE_K_CAP]
                             device uint*        q_n       [[buffer(6)]],  // [1]
                             device const uint*  map       [[buffer(7)]],  // [V] or unused
                             constant float&     top_p     [[buffer(8)]],
                             constant float&     min_p     [[buffer(9)]],
                             constant uint&      top_k     [[buffer(10)]],
                             constant uint&      V         [[buffer(11)]],
                             constant uint&      seed_lo   [[buffer(12)]],
                             constant uint&      seed_hi   [[buffer(13)]],
                             constant uint&      step      [[buffer(14)]],
                             constant uint&      use_penalties [[buffer(15)]],
                             constant uint&      mapped    [[buffer(16)]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    SAMPLE_TG_STATE
    float p, c;
    uint n_keep;
    const bool via = mapped != 0u;
    SAMPLE_SELECT(via, adjusted, map, maxima, top_p, min_p, top_k, V, p, c, n_keep)
    const float mass = cum[n_keep - 1];
    if (tid < n_keep) {
        q_ids[tid] = sample_id(map, via, cand_id[tid]);
        q_probs[tid] = p / mass;
    }
    if (tid == 0) {
        q_n[0] = n_keep;
    }
    const float target = sample_uniform(seed_lo, seed_hi, step) * mass;
    if (sample_pick(target, cum, n_keep, tid) == tid) {
        const uint chosen = sample_id(map, via, cand_id[tid]);
        out[0] = chosen;
        if (use_penalties) {
            counts[chosen] += 1u;
        }
    }
}

// Speculative sampling (Leviathan et al.): the trunk's kept distribution p
// against the draft's exported q and its proposal d = draft[0]. Accepts d
// with probability min(1, p(d) / q(d)) on the (seed, step) uniform,
// otherwise draws from the residual max(0, p - q) normalised, on a second
// uniform of the same step (its high bit set). The output is distributed
// exactly as a plain draw from p, and equals d exactly when accepted (the
// residual has no mass on a rejected d). Falls back to a plain draw from p
// when the residual has no mass, which only rounding can produce.
kernel void sample_spec_f32(device const float* adjusted  [[buffer(0)]],  // [V]
                            device const float* maxima    [[buffer(1)]],  // [SAMPLE_PREP_GROUPS]
                            device uint*        counts    [[buffer(2)]],  // [V]
                            device uint*        out       [[buffer(3)]],  // [1]
                            device const uint*  q_ids     [[buffer(4)]],  // [SAMPLE_K_CAP]
                            device const float* q_probs   [[buffer(5)]],  // [SAMPLE_K_CAP]
                            device const uint*  q_n       [[buffer(6)]],  // [1]
                            device const uint*  draft     [[buffer(7)]],  // [1]
                            device const uint*  map       [[buffer(8)]],  // [V] or unused
                            constant float&     top_p     [[buffer(9)]],
                            constant float&     min_p     [[buffer(10)]],
                            constant uint&      top_k     [[buffer(11)]],
                            constant uint&      V         [[buffer(12)]],
                            constant uint&      seed_lo   [[buffer(13)]],
                            constant uint&      seed_hi   [[buffer(14)]],
                            constant uint&      step      [[buffer(15)]],
                            constant uint&      use_penalties [[buffer(16)]],
                            constant uint&      mapped    [[buffer(17)]],
                            uint tid  [[thread_index_in_threadgroup]],
                            uint sg   [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    SAMPLE_TG_STATE
    threadgroup float p_d_s;
    threadgroup float q_d_s;
    float p, c;
    uint n_keep;
    const bool via = mapped != 0u;
    SAMPLE_SELECT(via, adjusted, map, maxima, top_p, min_p, top_k, V, p, c, n_keep)
    const float mass = cum[n_keep - 1];
    const uint d = draft[0];
    const uint qn = min(q_n[0], uint(SAMPLE_K_CAP));

    // p(d) from the kept set (0 outside it) and q(d) from the draft's set.
    if (tid == 0) {
        p_d_s = 0.0f;
        q_d_s = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < n_keep && sample_id(map, via, cand_id[tid]) == d) {
        p_d_s = p / mass;
    }
    if (tid < qn && q_ids[tid] == d) {
        q_d_s = q_probs[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float p_d = p_d_s;
    const float q_d = q_d_s;
    const float u1 = sample_uniform(seed_lo, seed_hi, step);
    if (u1 * q_d < p_d) {
        if (tid == 0) {
            out[0] = d;
            if (use_penalties) {
                counts[d] += 1u;
            }
        }
        return;
    }

    // Rejected: the residual over the kept set (q's mass elsewhere only
    // lowers the residual, which is zero outside p's support anyway).
    float q_i = 0.0f;
    if (tid < n_keep) {
        const uint id = sample_id(map, via, cand_id[tid]);
        for (uint j = 0; j < qn; ++j) {
            if (q_ids[j] == id) {
                q_i = q_probs[j];
                break;
            }
        }
    }
    const float r = tid < n_keep ? max(0.0f, p / mass - q_i) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r_total;
    const float rc = sample_scan_f(r, sg_sums, r_total, sg, lane);
    // The residual's prefix sums, in the sorted keys' storage (free now).
    threadgroup float* rcum = (threadgroup float*)cand_key;
    if (tid < SAMPLE_K_CAP) {
        rcum[tid] = rc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float u2 = sample_uniform(seed_lo, seed_hi, step | 0x80000000u);
    const uint slot = r_total > 0.0f ? sample_pick(u2 * r_total, rcum, n_keep, tid)
                                     : sample_pick(u2 * mass, cum, n_keep, tid);
    if (slot == tid) {
        const uint chosen = sample_id(map, via, cand_id[tid]);
        out[0] = chosen;
        if (use_penalties) {
            counts[chosen] += 1u;
        }
    }
}

// First phase of the two-phase selection for small k: threadgroup g takes
// its slice of `adjusted` (V / groups elements, the last one the
// remainder), selects the slice's top-k under the global maximum and writes
// the ids to local_ids[g * k ..], padded with ~0. The global top-k (ties by
// the kernels' fixed order) lies in the union of the slices' top-k, so
// sample_f32 and its siblings then select over that union through their
// `map`: groups x k elements instead of V.
kernel void sample_local_topk_f32(device const float* adjusted  [[buffer(0)]],  // [V]
                                  device const float* maxima    [[buffer(1)]],  // [SAMPLE_PREP_GROUPS]
                                  device uint*        local_ids [[buffer(2)]],  // [groups, k]
                                  constant uint&      top_k     [[buffer(3)]],
                                  constant uint&      V         [[buffer(4)]],
                                  constant uint&      groups    [[buffer(5)]],
                                  uint g    [[threadgroup_position_in_grid]],
                                  uint tid  [[thread_index_in_threadgroup]],
                                  uint sg   [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    SAMPLE_TG_STATE
    const uint chunk = (V + groups - 1) / groups;
    const uint begin = g * chunk;
    const uint count = begin < V ? min(chunk, V - begin) : 0u;
    device uint* ids = local_ids + g * top_k;
    if (count == 0u) {
        if (tid < top_k) {
            ids[tid] = 0xFFFFFFFFu;
        }
        return;
    }
    // The slice's own maximum as the reference: the prepare kernel's
    // partials cover exactly these slices when the group counts agree, and
    // the slice's top-k is within the first search range of it, where the
    // global maximum can be many ranges above a whole slice.
    const float top = groups == SAMPLE_PREP_GROUPS ? maxima[g] : sample_top(maxima);
    float p, c;
    uint n_keep;
    sample_select<false>(adjusted + begin, local_ids, top, 1.0f, 0.0f, top_k, count,
                         SAMPLE_SELECT_ARGS, p, c, n_keep, tid, sg, lane);
    if (tid < top_k) {
        ids[tid] = tid < n_keep ? begin + cand_id[tid] : 0xFFFFFFFFu;
    }
}
