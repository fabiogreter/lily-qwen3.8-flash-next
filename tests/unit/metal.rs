use std::time::{Duration, Instant};

/// Host seconds on Metal's GPU timestamp clock (mach_absolute_time).
fn mach_secs() -> f64 {
    #[repr(C)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }
    unsafe extern "C" {
        fn mach_absolute_time() -> u64;
        fn mach_timebase_info(info: *mut Timebase) -> i32;
    }
    let mut tb = Timebase { numer: 0, denom: 0 };
    let ticks = unsafe {
        mach_timebase_info(&mut tb);
        mach_absolute_time()
    };
    ticks as f64 * tb.numer as f64 / tb.denom as f64 * 1e-9
}

use super::*;
use crate::kernels::elementwise::copy_words;
use crate::tensor::DType;

/// A pass committed with a pending event wait must not run past the wait
/// until the host signals: its input is written only after commit.
#[test]
fn event_wait_holds_committed_pass_until_signaled() {
    let ctx = MetalContext::new().expect("metal context");
    let n = 4096usize;
    let x = Tensor::zeros(&ctx, &[n], DType::U32).expect("x");
    let out = Tensor::zeros(&ctx, &[n], DType::U32).expect("out");
    let event = ctx.new_shared_event().expect("event");
    assert_eq!(event.signaled_value(), 0);

    let pass = ctx.begin_concurrent().expect("pass");
    pass.wait_event(&event, 1).expect("wait");
    copy_words(&ctx, &pass, &x, &out).expect("copy");
    let pending = pass.commit().expect("commit");

    // The GPU is parked at the wait; write the input it will copy, then release it.
    std::thread::sleep(Duration::from_millis(20));
    let data: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    x.write_bytes(bytemuck::cast_slice(&data)).expect("write");
    event.signal(1);
    pending.wait().expect("wait");
    assert_eq!(out.to_u32().expect("read"), data);
    assert_eq!(event.signaled_value(), 1);
}

/// Handoff cost of "GPU pass B depends on something the host does after pass
/// A completes": today's protocol (commit B after A completes) against
/// pre-committing B behind an event wait and signaling. Reports the GPU idle
/// time between A's end and B's payload for each.
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn event_handoff_latency() {
    let ctx = MetalContext::new().expect("metal context");
    let words = 8usize << 20; // 32 MiB per copy, ~0.1 ms on the GPU
    let src = Tensor::zeros(&ctx, &[words], DType::U32).expect("src");
    let mid = Tensor::zeros(&ctx, &[words], DType::U32).expect("mid");
    let dst = Tensor::zeros(&ctx, &[words], DType::U32).expect("dst");
    let event = ctx.new_shared_event().expect("event");
    let iterations = 200u64;

    // Protocol X: commit B once A has completed (the current decode loop).
    let mut x_gap = Vec::new();
    let mut b_pure = Vec::new();
    for _ in 0..iterations {
        let a = ctx.begin_concurrent().expect("a");
        copy_words(&ctx, &a, &src, &mid).expect("copy a");
        let a_done = a.commit().expect("commit a").wait_retain().expect("wait a");
        let b = ctx.begin_concurrent().expect("b");
        copy_words(&ctx, &b, &mid, &dst).expect("copy b");
        let b_done = b.commit().expect("commit b").wait_retain().expect("wait b");
        let (ta, tb) = (a_done.timing().expect("ta"), b_done.timing().expect("tb"));
        x_gap.push(tb.gpu_start_secs - ta.gpu_end_secs);
        b_pure.push(tb.gpu_end_secs - tb.gpu_start_secs);
    }

    // Protocol Y: commit B right after A with a wait; signal once A is done
    // and the host has done its (here: no) work. B's GPU span includes the
    // stall, so the idle time is B.end - A.end minus B's pure duration.
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let b_pure_med = median(&mut b_pure);
    let mut y_gap = Vec::new();
    let mut host_signal = Vec::new();
    for i in 1..=iterations {
        let a = ctx.begin_concurrent().expect("a");
        copy_words(&ctx, &a, &src, &mid).expect("copy a");
        let a_pending = a.commit().expect("commit a");
        let b = ctx.begin_concurrent().expect("b");
        b.wait_event(&event, i).expect("wait");
        copy_words(&ctx, &b, &mid, &dst).expect("copy b");
        let b_pending = b.commit().expect("commit b");
        let a_done = a_pending.wait_retain().expect("wait a");
        let woke = Instant::now();
        event.signal(i);
        host_signal.push(woke.elapsed().as_secs_f64());
        let b_done = b_pending.wait_retain().expect("wait b");
        let (ta, tb) = (a_done.timing().expect("ta"), b_done.timing().expect("tb"));
        y_gap.push(tb.gpu_end_secs - ta.gpu_end_secs - b_pure_med);
    }
    eprintln!(
        "handoff idle: commit-after-completion {:.3} ms, event pre-commit {:.3} ms (B payload {:.3} ms, host signal call {:.1} us)",
        median(&mut x_gap) * 1e3,
        median(&mut y_gap) * 1e3,
        b_pure_med * 1e3,
        median(&mut host_signal) * 1e6
    );

    // What does the wait itself cost? Three placements of a wait inside B,
    // each preceded by a cover copy (so B = cover + wait + payload), against
    // B = cover + payload with no wait at all.
    let cover = Tensor::zeros(&ctx, &[words], DType::U32).expect("cover");
    let mut base = Vec::new();
    for _ in 0..iterations {
        let b = ctx.begin_concurrent().expect("b");
        copy_words(&ctx, &b, &src, &cover).expect("cover");
        copy_words(&ctx, &b, &mid, &dst).expect("copy b");
        let done = b.commit().expect("commit").wait_retain().expect("wait");
        let t = done.timing().expect("t");
        base.push(t.gpu_end_secs - t.gpu_start_secs);
    }
    let base_med = median(&mut base);
    let mut value = iterations;
    for (name, delay_ms) in [("satisfied before commit", None), ("signaled right after commit", Some(0.0)), ("signaled 1 ms after commit", Some(1.0))] {
        let mut spans = Vec::new();
        for _ in 0..iterations {
            value += 1;
            let b = ctx.begin_concurrent().expect("b");
            copy_words(&ctx, &b, &src, &cover).expect("cover");
            b.wait_event(&event, value).expect("wait");
            copy_words(&ctx, &b, &mid, &dst).expect("copy b");
            if delay_ms.is_none() {
                event.signal(value);
            }
            let pending = b.commit().expect("commit");
            if let Some(ms) = delay_ms {
                if ms > 0.0 {
                    std::thread::sleep(Duration::from_micros((ms * 1e3) as u64));
                }
                event.signal(value);
            }
            let done = pending.wait_retain().expect("wait");
            let t = done.timing().expect("t");
            spans.push(t.gpu_end_secs - t.gpu_start_secs);
        }
        eprintln!(
            "wait placement, {name}: B span {:.3} ms vs {:.3} ms without a wait (+{:.3} ms)",
            median(&mut spans) * 1e3,
            base_med * 1e3,
            (median(&mut spans) - base_med) * 1e3
        );
    }

    // Does a big encoder after the wait cost extra? 800 tiny dispatches (a
    // decode step's worth) after a satisfied wait versus in one encoder.
    let small = Tensor::zeros(&ctx, &[1024], DType::U32).expect("small");
    let small2 = Tensor::zeros(&ctx, &[1024], DType::U32).expect("small2");
    for (name, with_wait) in [("one encoder", false), ("satisfied wait, then 800 dispatches", true)] {
        let mut spans = Vec::new();
        for _ in 0..50 {
            value += 1;
            let b = ctx.begin_concurrent().expect("b");
            copy_words(&ctx, &b, &src, &cover).expect("cover");
            if with_wait {
                event.signal(value);
                b.wait_event(&event, value).expect("wait");
            }
            for _ in 0..800 {
                copy_words(&ctx, &b, &small, &small2).expect("tiny");
                b.level_barrier(&[&small2]).expect("barrier");
            }
            let done = b.commit().expect("commit").wait_retain().expect("wait");
            let t = done.timing().expect("t");
            spans.push(t.gpu_end_secs - t.gpu_start_secs);
        }
        eprintln!("big second encoder, {name}: span {:.3} ms", median(&mut spans) * 1e3);
    }

    // Signal while the GPU is still busy before the wait: 16 cover copies
    // (~2 ms), the signal 0.5 ms after commit. If this stalls, the wait is
    // evaluated when the command buffer is scheduled, not when it is reached.
    let mut long_base = Vec::new();
    for _ in 0..50 {
        let b = ctx.begin_concurrent().expect("b");
        for _ in 0..16 {
            copy_words(&ctx, &b, &src, &cover).expect("cover");
        }
        copy_words(&ctx, &b, &mid, &dst).expect("copy b");
        let done = b.commit().expect("commit").wait_retain().expect("wait");
        let t = done.timing().expect("t");
        long_base.push(t.gpu_end_secs - t.gpu_start_secs);
    }
    let long_base_med = median(&mut long_base);
    for delay_us in [100u64, 500, 1200, 1500, 1700, 1850, 2000, 2200] {
        let mut spans = Vec::new();
        let mut leads = Vec::new();
        for _ in 0..50 {
            value += 1;
            let b = ctx.begin_concurrent().expect("b");
            for _ in 0..16 {
                copy_words(&ctx, &b, &src, &cover).expect("cover");
            }
            b.wait_event(&event, value).expect("wait");
            copy_words(&ctx, &b, &mid, &dst).expect("copy b");
            let pending = b.commit().expect("commit");
            std::thread::sleep(Duration::from_micros(delay_us));
            event.signal(value);
            let signaled = mach_secs();
            let done = pending.wait_retain().expect("wait");
            let t = done.timing().expect("t");
            spans.push(t.gpu_end_secs - t.gpu_start_secs);
            // Lead of the signal over the GPU's arrival at the wait (cover after start).
            leads.push(t.gpu_start_secs + long_base_med - b_pure_med - signaled);
        }
        eprintln!(
            "signal during a {:.2} ms cover, {delay_us} us after commit: lead {:+.3} ms -> B span {:.3} ms (+{:.3} ms)",
            long_base_med * 1e3,
            median(&mut leads) * 1e3,
            median(&mut spans) * 1e3,
            (median(&mut spans) - long_base_med) * 1e3
        );
    }
}

/// The Metal 4 semantics the transport leans on: the argument table's state
/// is captured per dispatch (one table serves every dispatch of a pass), and
/// inline params bound by address from the arena read back correctly for
/// every type kernels take (`uint`, `float`, `uint4`).
#[test]
fn argument_table_binds_per_dispatch_and_params_read_back() {
    const SRC: &str = "
        kernel void combine(device const uint* a [[buffer(0)]], device uint* out [[buffer(1)]],
                            constant uint& k [[buffer(2)]], constant uint4& v [[buffer(3)]],
                            constant float& f [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
            out[gid] = a[gid] * k + v.x + v.w + uint(f);
        }";
    let ctx = MetalContext::new().expect("metal context");
    let kernel = ctx.pipeline("combine", SRC, MslVersion::V3_1).expect("kernel");
    let n = 1000usize;
    let a1 = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&(0..n as u32).collect::<Vec<_>>()), &[n], DType::U32).expect("a1");
    let a2 = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&(0..n as u32).map(|i| 3 * i).collect::<Vec<_>>()), &[n], DType::U32).expect("a2");
    let o1 = Tensor::zeros(&ctx, &[n], DType::U32).expect("o1");
    let o2 = Tensor::zeros(&ctx, &[n], DType::U32).expect("o2");
    let grid = Grid::Threads { grid: (n, 1, 1), threadgroup: (256, 1, 1) };
    let u4 = |x: u32, w: u32| -> [u8; 16] {
        let mut b = [0u8; 16];
        b[..4].copy_from_slice(&x.to_ne_bytes());
        b[12..].copy_from_slice(&w.to_ne_bytes());
        b
    };
    for concurrent in [false, true] {
        let pass = if concurrent { ctx.begin_concurrent() } else { ctx.begin() }.expect("pass");
        pass.dispatch_at(&kernel, &[a1.binding(), o1.binding()], &[&3u32.to_ne_bytes(), &u4(1, 5), &2.0f32.to_ne_bytes()], grid).expect("d1");
        pass.dispatch_at(&kernel, &[a2.binding(), o2.binding()], &[&7u32.to_ne_bytes(), &u4(2, 9), &4.0f32.to_ne_bytes()], grid).expect("d2");
        pass.level_barrier(&[]).expect("barrier");
        // In place, reading the first dispatch's result.
        pass.dispatch_at(&kernel, &[o1.binding(), o1.binding()], &[&1u32.to_ne_bytes(), &u4(100, 0), &0.0f32.to_ne_bytes()], grid).expect("d3");
        pass.commit_wait().expect("run");
        let out1 = o1.to_u32().expect("o1");
        let out2 = o2.to_u32().expect("o2");
        for i in 0..n as u32 {
            assert_eq!(out1[i as usize], i * 3 + 1 + 5 + 2 + 100, "o1[{i}] (concurrent={concurrent})");
            assert_eq!(out2[i as usize], 3 * i * 7 + 2 + 9 + 4, "o2[{i}] (concurrent={concurrent})");
        }
    }
}

/// A serial pass orders every dispatch after the one before it (Metal 4
/// encoders are concurrent unless told otherwise), and a concurrent pass
/// does so at its level barriers.
#[test]
fn passes_order_dependent_dispatches() {
    const SRC: &str = "
        kernel void bump(device uint* x [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
            x[gid] = x[gid] + 1;
        }";
    let ctx = MetalContext::new().expect("metal context");
    let kernel = ctx.pipeline("bump", SRC, MslVersion::V3_1).expect("kernel");
    let n = 64usize;
    let grid = Grid::Threads { grid: (n, 1, 1), threadgroup: (64, 1, 1) };
    let steps = 300usize;
    for concurrent in [false, true] {
        let x = Tensor::zeros(&ctx, &[n], DType::U32).expect("x");
        let pass = if concurrent { ctx.begin_concurrent() } else { ctx.begin() }.expect("pass");
        for _ in 0..steps {
            pass.dispatch_at(&kernel, &[x.binding()], &[], grid).expect("bump");
            if concurrent {
                pass.level_barrier(&[&x]).expect("barrier");
            }
        }
        pass.commit_wait().expect("run");
        assert!(x.to_u32().expect("x").iter().all(|&v| v == steps as u32), "concurrent={concurrent}");
    }
    // Across passes too: the next pass consumes the previous pass's writes.
    let x = Tensor::zeros(&ctx, &[n], DType::U32).expect("x");
    let mut pending = Vec::new();
    for _ in 0..20 {
        let pass = ctx.begin_concurrent().expect("pass");
        pass.dispatch_at(&kernel, &[x.binding()], &[], grid).expect("bump");
        pending.push(pass.commit().expect("commit"));
    }
    for p in pending {
        p.wait().expect("wait");
    }
    assert!(x.to_u32().expect("x").iter().all(|&v| v == 20));
}

/// Profile mode: every dispatch (and buffer copy) of a pass lands in the
/// recorded breakdown under its kernel name with a positive GPU time, in order,
/// under the pass's label; results are unchanged, and the pass's span covers
/// its dispatches.
#[test]
fn profile_mode_records_every_dispatch() {
    const SRC: &str = "
        kernel void bump(device uint* x [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
            x[gid] = x[gid] + 1;
        }";
    let ctx = MetalContext::new_with_profile(true).expect("metal context");
    assert!(ctx.profiling());
    let kernel = ctx.pipeline("bump", SRC, MslVersion::V3_1).expect("kernel");
    assert_eq!(kernel.name, "bump");
    let n = 4096usize;
    let grid = Grid::Threads { grid: (n, 1, 1), threadgroup: (256, 1, 1) };
    let x = Tensor::zeros(&ctx, &[n], DType::U32).expect("x");
    let y = Tensor::zeros(&ctx, &[n], DType::U32).expect("y");
    let label = "unit-test-profile";
    for concurrent in [false, true] {
        let pass = if concurrent { ctx.begin_concurrent() } else { ctx.begin() }.expect("pass");
        pass.set_label(label);
        for _ in 0..3 {
            pass.dispatch_at(&kernel, &[x.binding()], &[], grid).expect("bump");
            pass.level_barrier(&[&x]).expect("barrier");
        }
        copy_words(&ctx, &pass, &x, &y).expect("copy kernel");
        let ((xb, xo), (yb, yo)) = (x.binding(), y.binding());
        pass.copy_buffer(xb, xo, yb, yo, n * 4).expect("copy");
        pass.commit_wait().expect("run");
    }
    assert!(y.to_u32().expect("y").iter().all(|&v| v == 6));

    let recorded: Vec<_> = profile::take().into_iter().filter(|p| p.label == label).collect();
    assert_eq!(recorded.len(), 2, "one profile per committed pass");
    for pass in &recorded {
        let names: Vec<&str> = pass.kernels.iter().map(|k| k.name).collect();
        assert_eq!(names, ["bump", "bump", "bump", "copy_u32", "copy_buffer"]);
        let sum: f64 = pass.kernels.iter().map(|k| k.gpu_secs).sum();
        for k in &pass.kernels {
            assert!(k.gpu_secs > 0.0 && k.gpu_secs.is_finite(), "{}: {} s", k.name, k.gpu_secs);
        }
        assert!(pass.span_secs >= sum, "span {} s covers the kernels {} s", pass.span_secs, sum);
    }
    assert!(profile::take().iter().all(|p| p.label != label), "take drains");

    // Off by default: nothing is recorded, the same work still runs.
    let plain = MetalContext::new_with_profile(false).expect("metal context");
    assert!(!plain.profiling());
    let kernel = plain.pipeline("bump", SRC, MslVersion::V3_1).expect("kernel");
    let z = Tensor::zeros(&plain, &[n], DType::U32).expect("z");
    let pass = plain.begin_concurrent().expect("pass");
    pass.set_label(label);
    pass.dispatch_at(&kernel, &[z.binding()], &[], grid).expect("bump");
    pass.commit_wait().expect("run");
    assert!(z.to_u32().expect("z").iter().all(|&v| v == 1));
    assert!(profile::take().iter().all(|p| p.label != label));
}

/// Buffers allocated after earlier passes were submitted are resident for
/// the next one, and dropped buffers leave the residency set.
#[test]
fn residency_follows_buffer_lifetimes() {
    let ctx = MetalContext::new().expect("metal context");
    let n = 4096usize;
    let src = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&(0..n as u32).collect::<Vec<_>>()), &[n], DType::U32).expect("src");
    let before = ctx.resident_allocations();
    let first = ctx.begin().expect("pass");
    let scratch = Tensor::zeros(&ctx, &[n], DType::U32).expect("scratch");
    copy_words(&ctx, &first, &src, &scratch).expect("copy");
    let pending = first.commit().expect("commit");
    // Allocated with a pass in flight; used by the next pass.
    let late = Tensor::zeros(&ctx, &[n], DType::U32).expect("late");
    assert_eq!(ctx.resident_allocations(), before + 2);
    let second = ctx.begin().expect("pass");
    copy_words(&ctx, &second, &scratch, &late).expect("copy");
    pending.wait().expect("wait");
    second.commit_wait().expect("run");
    assert_eq!(late.to_u32().expect("late"), (0..n as u32).collect::<Vec<_>>());
    drop(scratch);
    drop(late);
    assert_eq!(ctx.resident_allocations(), before);
}

/// An encoded pass that is never committed (a pre-encoded speculative
/// variant that lost) returns its command memory without touching the GPU,
/// and a pending pass dropped without a wait completes first.
#[test]
fn uncommitted_and_unawaited_passes_release_cleanly() {
    let ctx = MetalContext::new().expect("metal context");
    let n = 4096usize;
    let src = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&(0..n as u32).collect::<Vec<_>>()), &[n], DType::U32).expect("src");
    let dst = Tensor::zeros(&ctx, &[n], DType::U32).expect("dst");
    for _ in 0..8 {
        let pass = ctx.begin_concurrent().expect("pass");
        copy_words(&ctx, &pass, &src, &dst).expect("copy");
        let encoded = pass.end().expect("end").detach();
        drop(encoded);
    }
    assert!(dst.to_u32().expect("dst").iter().all(|&v| v == 0));
    let pass = ctx.begin_concurrent().expect("pass");
    copy_words(&ctx, &pass, &src, &dst).expect("copy");
    drop(pass.commit().expect("commit"));
    assert_eq!(dst.to_u32().expect("dst"), (0..n as u32).collect::<Vec<_>>());
}

/// Costs of two Metal 4 bookkeeping paths the transport pays for: how long
/// after the fence a pass's commit feedback arrives (what `timing()` waits
/// for), and what committing the residency set costs with thousands of
/// allocations in it (paid on the first submission after an allocation and
/// on every buffer drop).
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn feedback_latency_and_residency_commit_cost() {
    let ctx = MetalContext::new().expect("metal context");
    let n = 4096usize;
    let src = Tensor::zeros(&ctx, &[n], DType::U32).expect("src");
    let dst = Tensor::zeros(&ctx, &[n], DType::U32).expect("dst");
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let mut latency = Vec::new();
    for _ in 0..50 {
        let pass = ctx.begin_concurrent().expect("pass");
        copy_words(&ctx, &pass, &src, &dst).expect("copy");
        let done = pass.commit().expect("commit").wait_retain().expect("wait");
        let t0 = Instant::now();
        done.timing().expect("timing");
        latency.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!("commit feedback arrives {:.3} ms (median) after the fence, max {:.3} ms", median(&mut latency), latency.iter().cloned().fold(0.0, f64::max));

    // Residency: grow the set to a model-sized population, then time the
    // allocate -> first submission path and the drop path.
    let mut population = Vec::new();
    for _ in 0..4000 {
        population.push(Tensor::zeros(&ctx, &[64], DType::U32).expect("buf"));
    }
    let pass = ctx.begin_concurrent().expect("pass");
    copy_words(&ctx, &pass, &src, &dst).expect("copy");
    pass.commit_wait().expect("run");
    let mut add_then_submit = Vec::new();
    let mut plain_submit = Vec::new();
    let mut drop_cost = Vec::new();
    for i in 0..40 {
        let extra = Tensor::zeros(&ctx, &[64], DType::U32).expect("extra");
        let pass = ctx.begin_concurrent().expect("pass");
        copy_words(&ctx, &pass, &src, &dst).expect("copy");
        let encoded = pass.end().expect("end");
        let t0 = Instant::now();
        let pending = encoded.commit().expect("commit");
        add_then_submit.push(t0.elapsed().as_secs_f64() * 1e3);
        pending.wait().expect("wait");
        let pass = ctx.begin_concurrent().expect("pass");
        copy_words(&ctx, &pass, &src, &dst).expect("copy");
        let encoded = pass.end().expect("end");
        let t0 = Instant::now();
        let pending = encoded.commit().expect("commit");
        plain_submit.push(t0.elapsed().as_secs_f64() * 1e3);
        pending.wait().expect("wait");
        let t0 = Instant::now();
        drop(extra);
        drop_cost.push(t0.elapsed().as_secs_f64() * 1e3);
        if i == 0 {
            eprintln!("residency set holds {} allocations", ctx.resident_allocations());
        }
    }
    eprintln!(
        "submit after an allocation {:.3} ms, plain submit {:.3} ms, buffer drop (remove + commit) {:.3} ms (medians, ~4000 resident allocations)",
        median(&mut add_then_submit),
        median(&mut plain_submit),
        median(&mut drop_cost)
    );
}

/// Wake-up latency of the blocking fence wait against polling the event, for
/// a pass of about a millisecond: what every `commit_wait` (prefill chunks,
/// blits) pays on top of the GPU time.
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn blocking_wait_latency() {
    let ctx = MetalContext::new().expect("metal context");
    let words = 64usize << 20; // 256 MiB copy, ~1 ms
    let src = Tensor::zeros(&ctx, &[words], DType::U32).expect("src");
    let dst = Tensor::zeros(&ctx, &[words], DType::U32).expect("dst");
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let mut blocking = Vec::new();
    let mut spinning = Vec::new();
    let mut gpu = Vec::new();
    for i in 0..60 {
        let pass = ctx.begin_concurrent().expect("pass");
        copy_words(&ctx, &pass, &src, &dst).expect("copy");
        let encoded = pass.end().expect("end");
        let t0 = Instant::now();
        let pending = encoded.commit().expect("commit");
        let done = if i % 2 == 0 {
            pending.wait_retain().expect("wait")
        } else {
            let mut pacer = Pacer::default();
            pacer.begin(Instant::now());
            pacer.end(Instant::now(), false); // estimate 0: poll from the start
            pending.wait_retain_paced(&mut pacer).expect("wait")
        };
        let wall = t0.elapsed().as_secs_f64() * 1e3;
        let t = done.timing().expect("timing");
        gpu.push((t.gpu_end_secs - t.gpu_start_secs) * 1e3);
        if i % 2 == 0 { blocking.push(wall) } else { spinning.push(wall) }
    }
    eprintln!(
        "commit-to-return for a {:.3} ms pass: blocking wait {:.3} ms, polling {:.3} ms",
        median(&mut gpu),
        median(&mut blocking),
        median(&mut spinning)
    );
}

#[test]
fn injected_fault_fails_the_next_submission_and_is_reported() {
    let ctx = MetalContext::new().expect("metal context");
    assert!(ctx.fault().is_none());
    ctx.inject_fault("injected for the test");
    assert_eq!(ctx.fault().as_deref(), Some("injected for the test"));
    // The fault is checked before anything reaches the queue, so even an
    // empty pass fails with it, and it stays: the context is done for.
    let pass = ctx.begin().expect("pass");
    let error = pass.commit_wait().expect_err("a faulted context must not submit");
    assert!(format!("{error:#}").contains("injected for the test"), "{error:#}");
    assert!(ctx.begin().expect("pass").commit_wait().is_err());
    assert_eq!(ctx.fault().as_deref(), Some("injected for the test"));
}
