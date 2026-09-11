const TEST_SOURCE: &str = concat!(
    include_str!("../../../src/kernels/metal/elementwise.metal"),
    "\n",
    include_str!("../../metal/elementwise_test.metal")
);

fn mul_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
) -> Result<()> {
    binary_op_source(ctx, pass, "mul_bf16", DType::BF16, a, b, out, TEST_SOURCE)
}

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;

#[test]
fn elementwise_ops_match_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(3);
    let n = 1000;
    let a: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
    let (ra, rb) = (cpu_ref::round_bf16(&a), cpu_ref::round_bf16(&b));

    let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[n]).expect("a");
    let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[n]).expect("b");

    let pair = |f: &dyn Fn(f32, f32) -> f32| -> Vec<f32> {
        ra.iter().zip(&rb).map(|(&x, &y)| f(x, y)).collect()
    };
    let cases = [
        ("add", pair(&|x, y| x + y)),
        ("mul", pair(&|x, y| x * y)),
        ("silu_mul", pair(&|x, y| cpu_ref::silu(x) * y)),
    ];
    for (name, expected) in cases {
        let out = Tensor::zeros(&ctx, &[n], DType::BF16).expect("out");
        let pass = ctx.begin().expect("pass");
        match name {
            "add" => add_bf16(&ctx, &pass, &ta, &tb, &out).expect("add"),
            "mul" => mul_bf16(&ctx, &pass, &ta, &tb, &out).expect("mul"),
            _ => silu_mul_bf16(&ctx, &pass, &ta, &tb, &out).expect("silu_mul"),
        }
        pass.commit_wait().expect("commit");
        cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
    }
}

#[test]
fn argmax_f32_stages_match_first_max_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(8);
    let n = 4097;
    let mut x: Vec<f32> = (0..n).map(|_| rng.gen_range(-8.0f32..8.0)).collect();
    x[513] = 9.5;
    x[3501] = 9.5;
    let expected = 513u32;

    let tx = Tensor::from_f32(&ctx, &x, &[n]).expect("x");
    let partials =
        Tensor::zeros(&ctx, &[ARGMAX_GROUPS * 2], DType::F32).expect("partials");
    let out = Tensor::zeros(&ctx, &[1], DType::U32).expect("out");

    let partial_pass = ctx.begin().expect("partial pass");
    argmax_f32_partial(&ctx, &partial_pass, &tx, &partials)
        .expect("argmax_f32_partial");
    partial_pass.commit_wait().expect("partial commit");

    let chunk = n.div_ceil(ARGMAX_GROUPS);
    let bytes = partials.raw_bytes();
    for group in 0..ARGMAX_GROUPS {
        let start = group * chunk;
        let end = ((group + 1) * chunk).min(n);
        let (expected_value, expected_index) = x[start..end].iter().enumerate().fold(
            (f32::NEG_INFINITY, 0usize),
            |(best_value, best_index), (offset, &value)| {
                let index = start + offset;
                if value > best_value {
                    (value, index)
                } else {
                    (best_value, best_index)
                }
            },
        );
        let base = group * 8;
        let actual_value = f32::from_ne_bytes(
            bytes[base..base + 4].try_into().expect("partial value"),
        );
        let actual_index = u32::from_ne_bytes(
            bytes[base + 4..base + 8].try_into().expect("partial index"),
        );
        assert_eq!(actual_value, expected_value, "group {group} value");
        assert_eq!(actual_index as usize, expected_index, "group {group} index");
    }

    let final_pass = ctx.begin().expect("final pass");
    argmax_f32_final(&ctx, &final_pass, &partials, &out).expect("argmax_f32_final");
    final_pass.commit_wait().expect("final commit");
    assert_eq!(out.to_u32().expect("read staged result")[0], expected);

    out.zero_fill();
    let combined_pass = ctx.begin().expect("combined pass");
    argmax_f32(&ctx, &combined_pass, &tx, &partials, &out).expect("argmax_f32");
    combined_pass.commit_wait().expect("combined commit");
    assert_eq!(out.to_u32().expect("read combined result")[0], expected);
}

#[test]
fn copy_words_copies_any_dtype_and_rejects_mismatch() {
    let ctx = MetalContext::new().expect("metal context");
    let src: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5 - 7.0).collect();
    let t_src = Tensor::from_f32(&ctx, &src, &[10, 100]).expect("src");
    let t_dst = Tensor::zeros(&ctx, &[1000], DType::F32).expect("dst");
    let pass = ctx.begin().expect("pass");
    copy_words(&ctx, &pass, &t_src, &t_dst).expect("copy");
    pass.commit_wait().expect("commit");
    assert_eq!(t_dst.to_f32().expect("read"), src);

    let ids: Vec<u32> = (0..6).collect();
    let t_ids = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ids), &[6], DType::U32).expect("ids");
    let t_out = Tensor::zeros(&ctx, &[6], DType::U32).expect("out");
    let pass = ctx.begin().expect("pass");
    copy_words(&ctx, &pass, &t_ids, &t_out).expect("copy u32");
    pass.commit_wait().expect("commit");
    assert_eq!(t_out.to_u32().expect("read"), ids);

    let short = Tensor::zeros(&ctx, &[999], DType::F32).expect("short");
    let pass = ctx.begin().expect("pass");
    assert!(copy_words(&ctx, &pass, &t_src, &short).is_err());
}

/// Raw unified-memory bandwidth from the GPU: a 4 GiB coalesced read and a
/// 4 GiB copy, best of several passes. The number to hold the weight-streaming
/// kernels against (the decode step's effective GB/s is bytes moved / GPU ms).
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn memory_bandwidth_probe() {
    let ctx = MetalContext::new().expect("metal context");
    let bytes = 4usize << 30;
    let u4 = bytes / 16;
    let per_thread = 16usize;
    let threads = u4 / per_thread;
    let src = Tensor::zeros(&ctx, &[bytes / 4], DType::U32).expect("src");
    let dst = Tensor::zeros(&ctx, &[bytes / 4], DType::U32).expect("dst");
    let out = Tensor::zeros(&ctx, &[1024], DType::U32).expect("out");
    let grid = Grid::Threads { grid: (threads, 1, 1), threadgroup: (256, 1, 1) };
    let per = u32_bytes(per_thread);
    {
        let fill = ctx.pipeline("bw_fill_u4", TEST_SOURCE, MslVersion::V3_1).expect("fill");
        let pass = ctx.begin().expect("pass");
        pass.dispatch_at(&fill, &[src.binding()], &[], Grid::Threads { grid: (u4, 1, 1), threadgroup: (256, 1, 1) }).expect("dispatch");
        pass.commit_wait().expect("fill");
    }
    let read = ctx.pipeline("bw_read_u4", TEST_SOURCE, MslVersion::V3_1).expect("read");
    let copy = ctx.pipeline("bw_copy_u4", TEST_SOURCE, MslVersion::V3_1).expect("copy");
    for (name, kernel, traffic) in [("read", &read, bytes), ("copy", &copy, 2 * bytes)] {
        let mut times = Vec::new();
        for _ in 0..5 {
            let pass = ctx.begin().expect("pass");
            let second: &Tensor = if name == "read" { &out } else { &dst };
            pass.dispatch_at(kernel, &[src.binding(), second.binding()], &[&per[..]], grid).expect("dispatch");
            let done = pass.commit().expect("commit").wait_retain().expect("wait");
            let t = done.timing().expect("timing");
            times.push(t.gpu_end_secs - t.gpu_start_secs);
        }
        times.sort_by(f64::total_cmp);
        let best = times[0];
        let median = times[times.len() / 2];
        eprintln!(
            "bandwidth {name}: {:.0} GB/s best, {:.0} GB/s median ({} bytes of traffic, {:.1} ms best)",
            traffic as f64 / best / 1e9,
            traffic as f64 / median / 1e9,
            traffic,
            best * 1e3
        );
    }
}

/// `silu_mul_rows_bf16` over a `[m, 2n]` gate|up stack equals `silu_mul_bf16`
/// on the split halves bit for bit (the small-m MoE path's shared expert
/// reads the fused stack GEMM output in place instead of splitting it).
#[test]
fn silu_mul_rows_matches_split_silu_mul_bitwise() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(97);
    for (m, n) in [(1usize, 640usize), (3, 640), (8, 96), (16, 1280)] {
        let gu: Vec<f32> =
            (0..m * 2 * n).map(|_| rng.gen_range(-4.0f32..4.0)).collect();
        let gu_ref = &gu;
        let (gate, up): (Vec<f32>, Vec<f32>) = (0..m)
            .flat_map(|r| {
                (0..n).map(move |c| (gu_ref[r * 2 * n + c], gu_ref[r * 2 * n + n + c]))
            })
            .unzip();
        let t_gu = Tensor::from_f32_as_bf16(&ctx, &gu, &[m, 2 * n]).expect("gu");
        let t_gate = Tensor::from_f32_as_bf16(&ctx, &gate, &[m, n]).expect("gate");
        let t_up = Tensor::from_f32_as_bf16(&ctx, &up, &[m, n]).expect("up");
        let rows_out = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("rows out");
        let split_out = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("split out");
        let pass = ctx.begin().expect("pass");
        silu_mul_rows_bf16(&ctx, &pass, &t_gu, &rows_out).expect("rows");
        silu_mul_bf16(&ctx, &pass, &t_gate, &t_up, &split_out).expect("split");
        pass.commit_wait().expect("commit");
        let bits = |t: &Tensor| -> Vec<u32> {
            t.to_f32().expect("read").iter().map(|x| x.to_bits()).collect()
        };
        assert_eq!(bits(&rows_out), bits(&split_out), "m={m} n={n}");
    }
}
