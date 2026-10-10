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
    for (name, delay_ms) in [
        ("satisfied before commit", None),
        ("signaled right after commit", Some(0.0)),
        ("signaled 1 ms after commit", Some(1.0)),
    ] {
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
    for (name, with_wait) in
        [("one encoder", false), ("satisfied wait, then 800 dispatches", true)]
    {
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
        eprintln!(
            "big second encoder, {name}: span {:.3} ms",
            median(&mut spans) * 1e3
        );
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
    let a1 = Tensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&(0..n as u32).collect::<Vec<_>>()),
        &[n],
        DType::U32,
    )
    .expect("a1");
    let a2 = Tensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&(0..n as u32).map(|i| 3 * i).collect::<Vec<_>>()),
        &[n],
        DType::U32,
    )
    .expect("a2");
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
        let pass = if concurrent { ctx.begin_concurrent() } else { ctx.begin() }
            .expect("pass");
        pass.dispatch_at(
            &kernel,
            &[a1.binding(), o1.binding()],
            &[&3u32.to_ne_bytes(), &u4(1, 5), &2.0f32.to_ne_bytes()],
            grid,
        )
        .expect("d1");
        pass.dispatch_at(
            &kernel,
            &[a2.binding(), o2.binding()],
            &[&7u32.to_ne_bytes(), &u4(2, 9), &4.0f32.to_ne_bytes()],
            grid,
        )
        .expect("d2");
        pass.level_barrier(&[]).expect("barrier");
        // In place, reading the first dispatch's result.
        pass.dispatch_at(
            &kernel,
            &[o1.binding(), o1.binding()],
            &[&1u32.to_ne_bytes(), &u4(100, 0), &0.0f32.to_ne_bytes()],
            grid,
        )
        .expect("d3");
        pass.commit_wait().expect("run");
        let out1 = o1.to_u32().expect("o1");
        let out2 = o2.to_u32().expect("o2");
        for i in 0..n as u32 {
            assert_eq!(
                out1[i as usize],
                i * 3 + 1 + 5 + 2 + 100,
                "o1[{i}] (concurrent={concurrent})"
            );
            assert_eq!(
                out2[i as usize],
                3 * i * 7 + 2 + 9 + 4,
                "o2[{i}] (concurrent={concurrent})"
            );
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
        let pass = if concurrent { ctx.begin_concurrent() } else { ctx.begin() }
            .expect("pass");
        for _ in 0..steps {
            pass.dispatch_at(&kernel, &[x.binding()], &[], grid).expect("bump");
            if concurrent {
                pass.level_barrier(&[&x]).expect("barrier");
            }
        }
        pass.commit_wait().expect("run");
        assert!(
            x.to_u32().expect("x").iter().all(|&v| v == steps as u32),
            "concurrent={concurrent}"
        );
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
        let pass = if concurrent { ctx.begin_concurrent() } else { ctx.begin() }
            .expect("pass");
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

    let recorded: Vec<_> =
        profile::take().into_iter().filter(|p| p.label == label).collect();
    assert_eq!(recorded.len(), 2, "one profile per committed pass");
    for pass in &recorded {
        let names: Vec<&str> = pass.kernels.iter().map(|k| k.name).collect();
        assert_eq!(names, ["bump", "bump", "bump", "copy_u32", "copy_buffer"]);
        let sum: f64 = pass.kernels.iter().map(|k| k.gpu_secs).sum();
        for k in &pass.kernels {
            assert!(
                k.gpu_secs > 0.0 && k.gpu_secs.is_finite(),
                "{}: {} s",
                k.name,
                k.gpu_secs
            );
        }
        assert!(
            pass.span_secs >= sum,
            "span {} s covers the kernels {} s",
            pass.span_secs,
            sum
        );
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
    let src = Tensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&(0..n as u32).collect::<Vec<_>>()),
        &[n],
        DType::U32,
    )
    .expect("src");
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

/// A recording returns the buffers allocated while it was open that are
/// still alive, nothing from before or after it, and only one runs at a time.
#[test]
fn buffer_recordings_return_the_live_buffers_allocated_meanwhile() {
    let ctx = MetalContext::new().expect("metal context");
    let before = ctx.new_buffer(64).expect("before");
    let recording = ctx.record_buffers().expect("recording");
    assert!(ctx.record_buffers().is_err(), "one recording at a time");
    let kept = Tensor::zeros(&ctx, &[1024], DType::U32).expect("kept");
    let temporary = ctx.new_buffer(4096).expect("temporary");
    let bytes = ctx.new_buffer_with_bytes(&[7u8; 100]).expect("bytes");
    drop(temporary);
    let recorded = recording.finish();
    let after = ctx.new_buffer(64).expect("after");
    let addresses: Vec<u64> = recorded.iter().map(|b| b.address()).collect();
    assert_eq!(addresses, vec![kept.binding().0.address(), bytes.address()]);
    assert!(!addresses.contains(&before.address()));
    assert!(!addresses.contains(&after.address()));
    // A recording dropped without `finish` (an error path) ends too.
    drop(ctx.record_buffers().expect("again"));
    let _unrecorded = ctx.new_buffer(64).expect("unrecorded");
    assert!(ctx.record_buffers().expect("free again").finish().is_empty());
}

/// An encoded pass that is never committed (a pre-encoded speculative
/// variant that lost) returns its command memory without touching the GPU,
/// and a pending pass dropped without a wait completes first.
#[test]
fn uncommitted_and_unawaited_passes_release_cleanly() {
    let ctx = MetalContext::new().expect("metal context");
    let n = 4096usize;
    let src = Tensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&(0..n as u32).collect::<Vec<_>>()),
        &[n],
        DType::U32,
    )
    .expect("src");
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
    eprintln!(
        "commit feedback arrives {:.3} ms (median) after the fence, max {:.3} ms",
        median(&mut latency),
        latency.iter().cloned().fold(0.0, f64::max)
    );

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

/// Read bandwidth of the GPU over a file-mapped buffer, cold and warm:
/// `LILY_MAP_PROBE_FILE` names a large file (evict its pages first, e.g.
/// with a locked memory balloon), and the test maps it, reads every byte
/// twice through the checksum kernel and prints GB/s for each pass.
/// `cargo test --release -- --ignored --nocapture mapped_buffer_read_bandwidth`.
#[test]
#[ignore = "bandwidth probe; needs LILY_MAP_PROBE_FILE"]
fn mapped_buffer_read_bandwidth() {
    let path = std::env::var("LILY_MAP_PROBE_FILE").expect("LILY_MAP_PROBE_FILE");
    let len = std::fs::metadata(&path).expect("file").len() as usize;
    let ctx = MetalContext::new().expect("metal context");
    let (buf, inner) =
        ctx.new_buffer_mapped(std::path::Path::new(&path), 0, len).expect("map");
    assert_eq!(inner, 0);
    let src = Tensor::from_buffer(buf, &[len / 4], DType::U32).expect("tensor");
    let out = Tensor::zeros(&ctx, &[1], DType::U32).expect("out");
    for pass_name in ["cold", "warm", "warm"] {
        let t0 = Instant::now();
        let pass = ctx.begin().expect("pass");
        crate::kernels::elementwise::checksum_words(&ctx, &pass, &src, &out)
            .expect("checksum");
        pass.commit_wait().expect("commit");
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "{pass_name}: {:.2} GB in {:.3} s = {:.1} GB/s (checksum {:08x})",
            len as f64 / 1e9,
            secs,
            len as f64 / secs / 1e9,
            out.to_u32().expect("out")[0]
        );
    }
}

/// Like `mapped_buffer_read_bandwidth`, but reading 64 slices of
/// `LILY_MAP_PROBE_SLICE` bytes (default 2.8 MB, one expert's three
/// projections) at pseudo-random offsets: the access pattern of expert
/// weights served from disk. Prints GB/s cold and warm.
#[test]
#[ignore = "bandwidth probe; needs LILY_MAP_PROBE_FILE"]
fn mapped_buffer_random_slice_bandwidth() {
    let path = std::env::var("LILY_MAP_PROBE_FILE").expect("LILY_MAP_PROBE_FILE");
    let slice: usize = std::env::var("LILY_MAP_PROBE_SLICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_800_000);
    let slice = slice.next_multiple_of(16);
    let len = std::fs::metadata(&path).expect("file").len() as usize;
    let ctx = MetalContext::new().expect("metal context");
    let (buf, _) =
        ctx.new_buffer_mapped(std::path::Path::new(&path), 0, len).expect("map");
    let src = Tensor::from_buffer(buf, &[len / 4], DType::U32).expect("tensor");
    let out = Tensor::zeros(&ctx, &[1], DType::U32).expect("out");
    let n = 64usize;
    let mut seed = 0x9e3779b97f4a7c15u64;
    let offsets: Vec<usize> = (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed as usize) % ((len - slice) / 16)) * 16
        })
        .collect();
    for pass_name in ["cold", "warm"] {
        let t0 = Instant::now();
        let pass = ctx.begin().expect("pass");
        for &off in &offsets {
            let view = src.view(off / 4, &[slice / 4]).expect("view");
            crate::kernels::elementwise::checksum_words(&ctx, &pass, &view, &out)
                .expect("checksum");
        }
        pass.commit_wait().expect("commit");
        let secs = t0.elapsed().as_secs_f64();
        let bytes = (n * slice) as f64;
        eprintln!(
            "{pass_name}: {n} slices of {:.2} MB in {:.3} s = {:.2} GB/s, {:.1} ms per slice",
            slice as f64 / 1e6,
            secs,
            bytes / secs / 1e9,
            secs * 1e3 / n as f64
        );
    }
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

/// What the first submission after an idle gap costs, and what of it scales
/// with the resident set: a model-sized population of resident buffers
/// (`LILY_IDLE_PROBE_GB`, default 1, zero-filled, `LILY_IDLE_PROBE_MLOCK=1`
/// locks them like the pinned weights), then per gap in `LILY_IDLE_PROBE_GAPS`
/// (seconds, default `1,3,10,30`) a sleep and one ~1 ms copy pass, followed at
/// once by the same pass as the back-to-back reference. Prints the commit to
/// GPU start delay, the GPU span and the commit-to-return wall time of both.
/// `LILY_IDLE_PROBE_MODE`: `plain` (default); `wake`, which commits a
/// one-dispatch pass right after the sleep and the measured pass
/// `LILY_IDLE_PROBE_LEAD_MS` (default 100) later, the way a request-arrival
/// wake overlaps host work; `signal`, the same with only a queue-level fence
/// signal (no command buffer) as the wake; `request`, which calls
/// `requestResidency` on the set at the end of the sleep, then waits the lead,
/// then commits; `keepalive` (see the loop). `LILY_IDLE_PROBE_HEAP=1` puts
/// the population in 1 GiB heaps instead of buffers.
/// `cargo test --release --lib -- --ignored --nocapture idle_first_submission`.
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn idle_first_submission_probe() {
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_owned());
    let gb: usize = env("LILY_IDLE_PROBE_GB", "1").parse().expect("gb");
    let lock = env("LILY_IDLE_PROBE_MLOCK", "0") == "1";
    let mode = env("LILY_IDLE_PROBE_MODE", "plain");
    let lead_ms: u64 = env("LILY_IDLE_PROBE_LEAD_MS", "100").parse().expect("lead");
    let gaps: Vec<f64> = env("LILY_IDLE_PROBE_GAPS", "1,3,10,30")
        .split(',')
        .map(|s| s.trim().parse().expect("gap"))
        .collect();
    let ctx = MetalContext::new().expect("metal context");
    let mut population = Vec::new();
    // `LILY_IDLE_PROBE_HEAP=1`: the population as 1 GiB heaps in the set
    // (one buffer placed in each and zero-filled) instead of buffers.
    let mut heaps = Vec::new();
    if env("LILY_IDLE_PROBE_HEAP", "0") == "1" {
        use objc2_metal::{MTLHeap, MTLHeapDescriptor, MTLStorageMode};
        for _ in 0..gb {
            let desc = MTLHeapDescriptor::new();
            desc.setSize(1 << 30);
            desc.setStorageMode(MTLStorageMode::Shared);
            let heap = ctx.device.newHeapWithDescriptor(&desc).expect("heap");
            let buf = heap
                .newBufferWithLength_options(
                    1 << 30,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("heap buffer");
            unsafe {
                core::ptr::write_bytes(buf.contents().as_ptr().cast::<u8>(), 0, 1 << 30)
            };
            ctx.residency
                .set
                .addAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&*heap));
            heaps.push((heap, buf));
        }
        ctx.residency.dirty.set(true);
    }
    for _ in 0..if heaps.is_empty() { gb } else { 0 } {
        let t = Tensor::zeros(&ctx, &[1usize << 28], DType::U32).expect("1 GiB");
        if lock {
            let (buf, _) = t.binding();
            // SAFETY: the buffer's own mapping, of its length.
            let rc = unsafe { libc::mlock(buf.contents().as_ptr(), buf.length()) };
            assert_eq!(rc, 0, "mlock: {}", std::io::Error::last_os_error());
        }
        population.push(t);
    }
    let words = 64usize << 20; // 256 MiB copy, ~1 ms
    let src = Tensor::zeros(&ctx, &[words], DType::U32).expect("src");
    let dst = Tensor::zeros(&ctx, &[words], DType::U32).expect("dst");
    let tiny_src = Tensor::zeros(&ctx, &[64], DType::U32).expect("tiny");
    let tiny_dst = Tensor::zeros(&ctx, &[64], DType::U32).expect("tiny");
    // (start delay, GPU span, wall) in ms of one measured pass.
    let run = |big: bool| -> (f64, f64, f64) {
        let pass = ctx.begin_concurrent().expect("pass");
        if big {
            copy_words(&ctx, &pass, &src, &dst).expect("copy");
        } else {
            copy_words(&ctx, &pass, &tiny_src, &tiny_dst).expect("copy");
        }
        let encoded = pass.end().expect("end");
        let t0 = Instant::now();
        let committed = mach_secs();
        let done = encoded.commit().expect("commit").wait_retain().expect("wait");
        let wall = t0.elapsed().as_secs_f64() * 1e3;
        let t = done.timing().expect("timing");
        (
            (t.gpu_start_secs - committed) * 1e3,
            (t.gpu_end_secs - t.gpu_start_secs) * 1e3,
            wall,
        )
    };
    for _ in 0..5 {
        run(true);
    }
    eprintln!(
        "idle probe: {gb} GiB resident{}, {} allocations, mode {mode}, lead {lead_ms} ms",
        if lock { " (mlocked)" } else { "" },
        ctx.resident_allocations()
    );
    eprintln!(
        "gap_s\tmode\tdelay_ms\tgpu_ms\twall_ms\tpre_ms\tref_delay\tref_gpu\tref_wall"
    );
    for &gap in &gaps {
        // `keepalive`: a bare fence signal every `LILY_IDLE_PROBE_TICK_MS`
        // (default 1000) through the gap, to see whether that alone keeps
        // the set resident. A measurement only; the server does not do this.
        if mode == "keepalive" {
            let tick: u64 =
                env("LILY_IDLE_PROBE_TICK_MS", "1000").parse().expect("tick");
            let end = Instant::now() + Duration::from_secs_f64(gap);
            while Instant::now() < end {
                ctx.wake().expect("tick");
                std::thread::sleep(
                    Duration::from_millis(tick).min(end - Instant::now()),
                );
            }
        } else {
            std::thread::sleep(Duration::from_secs_f64(gap));
        }
        let mut pre = 0.0;
        let mut pending = None;
        match mode.as_str() {
            "wake" => {
                let t0 = Instant::now();
                let pass = ctx.begin_concurrent().expect("pass");
                copy_words(&ctx, &pass, &tiny_src, &tiny_dst).expect("copy");
                pending = Some(pass.commit().expect("commit"));
                pre = t0.elapsed().as_secs_f64() * 1e3;
                std::thread::sleep(Duration::from_millis(lead_ms));
            }
            "signal" => {
                let t0 = Instant::now();
                let feedback = Feedback::new(0, None, Vec::new());
                ctx.submit(&[], &feedback).expect("fence signal");
                pre = t0.elapsed().as_secs_f64() * 1e3;
                std::thread::sleep(Duration::from_millis(lead_ms));
            }
            "request" => {
                let t0 = Instant::now();
                ctx.residency.set.requestResidency();
                pre = t0.elapsed().as_secs_f64() * 1e3;
                std::thread::sleep(Duration::from_millis(lead_ms));
            }
            _ => {}
        }
        let (d, g, w) = run(true);
        if let Some(p) = pending {
            p.wait().expect("wake pass");
        }
        let (rd, rg, rw) = run(true);
        eprintln!(
            "{gap}\t{mode}\t{d:.2}\t{g:.3}\t{w:.2}\t{pre:.3}\t{rd:.2}\t{rg:.3}\t{rw:.2}"
        );
    }
    drop(population);
    for (heap, _) in &heaps {
        ctx.residency
            .set
            .removeAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&**heap));
    }
    ctx.residency.set.commit();
}

/// A wake is a bare fence signal nobody waits for: passes before and after
/// it complete and wait normally, and a faulted context refuses it.
#[test]
fn wake_is_a_fence_signal_that_later_passes_wait_past() {
    let ctx = MetalContext::new().expect("metal context");
    let n = 4096usize;
    let data: Vec<u32> = (0..n as u32).collect();
    let src = Tensor::zeros(&ctx, &[n], DType::U32).expect("src");
    src.write_bytes(bytemuck::cast_slice(&data)).expect("write");
    let dst = Tensor::zeros(&ctx, &[n], DType::U32).expect("dst");
    let pass = ctx.begin_concurrent().expect("pass");
    copy_words(&ctx, &pass, &src, &dst).expect("copy");
    let pending = pass.commit().expect("commit");
    ctx.wake().expect("wake behind a pending pass");
    ctx.wake().expect("wake twice");
    pending.wait().expect("the pass before the wakes completes");
    let pass = ctx.begin_concurrent().expect("pass");
    copy_words(&ctx, &pass, &src, &dst).expect("copy");
    pass.commit_wait().expect("the pass after the wakes completes");
    assert_eq!(dst.to_u32().expect("read"), data);
    ctx.inject_fault("injected for the test");
    assert!(ctx.wake().is_err());
}

/// Waits the host releases, early or late, while the queue is blocked on
/// them or before it gets there, are never taken for timed-out ones. Prints
/// how long after the host's signal the work behind them started, the
/// figure `EARLY_START_MARGIN_SECS` is set against.
#[test]
fn waits_the_host_releases_are_not_reported_as_timed_out() {
    let ctx = MetalContext::new().expect("metal context");
    let words = 1usize << 16;
    let a = Tensor::zeros(&ctx, &[words], DType::U32).expect("a");
    let b = Tensor::zeros(&ctx, &[words], DType::U32).expect("b");
    let event = ctx.new_watched_event().expect("event");
    let mut margins = Vec::new();
    for i in 1..=60u64 {
        // Work ahead of the wait, in a pass of its own: the waiting pass
        // starts with the wait, so its GPU start is the work behind it (the
        // witness command buffer).
        let busy = ctx.begin_concurrent().expect("pass");
        copy_words(&ctx, &busy, &a, &b).expect("copy");
        let busy = busy.commit().expect("commit");
        let pass = ctx.begin_concurrent().expect("pass");
        pass.wait_event(&event, i).expect("wait");
        copy_words(&ctx, &pass, &b, &a).expect("copy");
        let delay = Duration::from_micros([0, 50, 300, 1500][i as usize % 4]);
        if i % 5 == 0 {
            event.signal(i);
        }
        let pending = pass.commit().expect("commit");
        std::thread::sleep(delay);
        event.signal(i);
        busy.wait().expect("busy pass");
        let done =
            pending.wait_retain().expect("a released wait reported as timed out");
        let start = done.timing().expect("timing").gpu_start_secs;
        margins.push(start - event.signalled_at(i).expect("the signal is logged"));
    }
    assert!(ctx.fault().is_none(), "{:?}", ctx.fault());
    margins.sort_by(f64::total_cmp);
    println!(
        "work behind a released wait started {:.3} to {:.3} ms after the host's signal (median {:.3})",
        margins[0] * 1e3,
        margins[margins.len() - 1] * 1e3,
        margins[margins.len() / 2] * 1e3
    );
    // Causality says above zero; half the margin is left for the clocks.
    assert!(
        margins[0] > -EARLY_START_MARGIN_SECS / 2.0,
        "work started {:.3} ms before the signal",
        -margins[0] * 1e3
    );
}

/// A wait the GPU went past without the host's signal (what a timed-out
/// queue wait does) faults the context, and the pass's wait reports it.
#[test]
fn a_wait_passed_before_the_host_signal_faults_the_context() {
    let ctx = MetalContext::new().expect("metal context");
    let words = 1usize << 16;
    let a = Tensor::zeros(&ctx, &[words], DType::U32).expect("a");
    let b = Tensor::zeros(&ctx, &[words], DType::U32).expect("b");
    let event = ctx.new_watched_event().expect("event");
    let pass = ctx.begin().expect("pass");
    pass.wait_event(&event, 1).expect("wait");
    copy_words(&ctx, &pass, &a, &b).expect("copy");
    // Released behind the host's back, and the host's own signal logged
    // as made later than the GPU starts the work (its feedback is checked
    // after the host signalled).
    event.signal_unlogged(1);
    event
        .log
        .as_ref()
        .expect("watched")
        .lock()
        .expect("log")
        .push(1, host_secs() + 10.0);
    let pending = pass.commit().expect("commit");
    let err = pending.wait().expect_err("the early release goes unnoticed");
    assert!(format!("{err:#}").contains("queue wait timed out"), "{err:#}");
    assert!(ctx.fault().is_some());
}

/// Waits on events that are not watched (the parked pass's, and every wait
/// without the expert cache) are submitted as before: no witness command
/// buffer, no check; a wait on a watched event gets both.
#[test]
fn only_waits_on_watched_events_are_witnessed() {
    let ctx = MetalContext::new().expect("metal context");
    let words = 1usize << 12;
    let a = Tensor::zeros(&ctx, &[words], DType::U32).expect("a");
    let b = Tensor::zeros(&ctx, &[words], DType::U32).expect("b");
    let plain = ctx.new_shared_event().expect("event");
    let watched = ctx.new_watched_event().expect("event");
    let cmds = |events: &[&SharedEvent]| {
        let pass = ctx.begin().expect("pass");
        for (k, event) in events.iter().enumerate() {
            pass.wait_event(event, k as u64 + 1).expect("wait");
            copy_words(&ctx, &pass, &a, &b).expect("copy");
        }
        let encoded = pass.end().expect("end");
        let cmds = encoded.program.iter().filter(|i| matches!(i, Item::Cmd(_))).count();
        (cmds, watches_of(&encoded.program).len())
    };
    assert_eq!(cmds(&[&plain]), (1, 0));
    assert_eq!(cmds(&[&watched]), (2, 1));
    assert_eq!(cmds(&[&plain, &watched]), (3, 1));
    assert_eq!(cmds(&[&watched, &plain]), (3, 1));
    // An unwatched event logs nothing, so no wait on it can be judged.
    plain.signal(5);
    assert_eq!(plain.signalled_at(5), None);
}

/// A real timed-out wait (about 4 s, see `queue_wait_timeout_probe`) is
/// caught: the work behind it completes while the event is still below the
/// awaited value.
#[test]
#[ignore = "takes 4.5 s and depends on Metal's undocumented wait limit"]
fn a_real_timed_out_wait_faults_the_context() {
    let ctx = MetalContext::new().expect("metal context");
    let words = 1usize << 16;
    let a = Tensor::zeros(&ctx, &[words], DType::U32).expect("a");
    let b = Tensor::zeros(&ctx, &[words], DType::U32).expect("b");
    let event = ctx.new_watched_event().expect("event");
    let pass = ctx.begin().expect("pass");
    pass.wait_event(&event, 1).expect("wait");
    copy_words(&ctx, &pass, &a, &b).expect("copy");
    let pending = pass.commit().expect("commit");
    std::thread::sleep(Duration::from_millis(4500));
    event.signal(1);
    let err = pending.wait().expect_err("the timed-out wait goes unnoticed");
    assert!(format!("{err:#}").contains("queue wait timed out"), "{err:#}");
}

#[test]
fn timed_out_wait_rule() {
    // Below the value once the work behind it completed: never waited.
    assert!(wait_released_early(3, 4, None, 10.0).is_some());
    // Started well before the host's signal: released early.
    assert!(wait_released_early(4, 4, Some(10.0), 9.999).is_some());
    // Started after it, within the margin before it, or the signal time
    // unknown: held (or unknowable).
    assert!(wait_released_early(4, 4, Some(10.0), 10.0001).is_none());
    assert!(wait_released_early(4, 4, Some(10.0), 10.0 - 50e-6).is_none());
    assert!(wait_released_early(9, 4, None, 1.0).is_none());
    // The event below the value is a timeout whatever the clocks say.
    assert!(wait_released_early(3, 4, Some(10.0), 10.0001).is_some());
    // No GPU timestamp (a failed command buffer reports zeros).
    assert!(wait_released_early(4, 4, Some(10.0), 0.0).is_none());
}

#[test]
fn signal_log_finds_the_first_signal_at_or_above_a_value() {
    let mut log = SignalLog::default();
    assert_eq!(log.signalled_at(1), None);
    log.push(2, 1.0);
    log.push(5, 2.0);
    // At or below the last value: the event did not move, nothing logged.
    log.push(5, 2.5);
    log.push(4, 3.0);
    assert_eq!(log.entries.len(), 2);
    assert_eq!(log.signalled_at(1), Some(1.0));
    assert_eq!(log.signalled_at(3), Some(2.0));
    assert_eq!(log.signalled_at(5), Some(2.0));
    assert_eq!(log.signalled_at(6), None);
    for v in 6..6 + SignalLog::CAPACITY as u64 {
        log.push(v, v as f64);
    }
    assert_eq!(log.entries.len(), SignalLog::CAPACITY);
    // The first signal at or above 3 is gone: unknown, not the oldest kept.
    assert_eq!(log.signalled_at(3), None);
    assert_eq!(log.signalled_at(7), Some(7.0));
}

/// A gated pass's parts are submitted only once the host signalled their
/// waits, in order, and a pass committed after it waits for its last part.
#[test]
fn gated_pass_submits_each_part_after_its_signal() {
    let ctx = MetalContext::new().expect("metal context");
    let n = 4096usize;
    let x: Vec<Tensor> =
        (0..4).map(|_| Tensor::zeros(&ctx, &[n], DType::U32).expect("x")).collect();
    let src: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    x[0].write_bytes(bytemuck::cast_slice(&src)).expect("write");
    let event = ctx.new_watched_event().expect("event");
    let reached = ctx.new_shared_event().expect("reached");
    let pass = ctx.begin().expect("pass");
    pass.gate_host_waits();
    for k in 0..3u64 {
        if k > 0 {
            pass.signal_event(&reached, k).expect("signal");
            pass.wait_event(&event, k).expect("wait");
        }
        copy_words(&ctx, &pass, &x[k as usize], &x[k as usize + 1]).expect("copy");
    }
    let signaller = gate_signaller(&ctx, &event, &reached, 2);
    let pending = pass.commit().expect("commit");
    // Submitted behind the gated pass: clears its last output, after it.
    let zeros = Tensor::zeros(&ctx, &[n], DType::U32).expect("zeros");
    let later = ctx.begin().expect("later");
    copy_words(&ctx, &later, &zeros, &x[3]).expect("copy");
    let later = later.commit().expect("commit later");
    pending.wait().expect("gated pass");
    later.wait().expect("later pass");
    let signalled = signaller.join().expect("signaller");
    assert_eq!(x[2].to_u32().expect("read"), src);
    assert!(x[3].to_u32().expect("read").iter().all(|&v| v == 0));
    assert!(ctx.fault().is_none(), "{:?}", ctx.fault());
    assert_parts_after(&ctx, &signalled);
}

/// A thread that, for each of `parts` gates, waits until the GPU has
/// signalled `reached` to the gate's value, checks that nothing behind the
/// gate was submitted 20 ms later, and signals `event`; returns the host
/// time just before each signal.
fn gate_signaller(
    ctx: &MetalContext,
    event: &SharedEvent,
    reached: &SharedEvent,
    parts: u64,
) -> std::thread::JoinHandle<Vec<f64>> {
    let (event, reached, feeding) =
        (event.clone(), reached.clone(), ctx.feeding.clone());
    std::thread::spawn(move || {
        let mut signalled = Vec::new();
        for k in 1..=parts {
            assert!(reached.wait_until(k, 5000), "part {k} never ran");
            std::thread::sleep(Duration::from_millis(20));
            let submitted = feeding.parts.lock().expect("parts").len() as u64;
            assert_eq!(submitted, k - 1, "a part was submitted before its gate opened");
            signalled.push(host_secs());
            event.signal(k);
        }
        signalled
    })
}

/// Every part of the gated pass on `ctx` was submitted once, in order, after
/// its gate opened (`signalled`: the host time just before each signal).
fn assert_parts_after(ctx: &MetalContext, signalled: &[f64]) {
    let parts = ctx.feeding.parts.lock().expect("parts").clone();
    assert_eq!(parts.len(), signalled.len(), "{parts:?}");
    for (k, ((value, at), signal)) in parts.iter().zip(signalled).enumerate() {
        assert_eq!(*value, k as u64 + 1);
        assert!(
            at >= signal,
            "part {value} submitted {:.3} ms before its signal",
            (signal - at) * 1e3
        );
    }
}

/// Before the gate is opened nothing behind it was submitted, so nothing
/// ran.
#[test]
fn gated_pass_holds_the_work_behind_a_wait() {
    let ctx = MetalContext::new().expect("metal context");
    let n = 4096usize;
    let a = Tensor::zeros(&ctx, &[n], DType::U32).expect("a");
    let b = Tensor::zeros(&ctx, &[n], DType::U32).expect("b");
    let data: Vec<u32> = (1..=n as u32).collect();
    a.write_bytes(bytemuck::cast_slice(&data)).expect("write");
    let event = ctx.new_shared_event().expect("event");
    let pass = ctx.begin().expect("pass");
    pass.gate_host_waits();
    pass.wait_event(&event, 1).expect("wait");
    copy_words(&ctx, &pass, &a, &b).expect("copy");
    let pending = pass.commit().expect("commit");
    std::thread::sleep(Duration::from_millis(30));
    assert!(ctx.feeding.parts.lock().expect("parts").is_empty());
    assert!(b.to_u32().expect("read").iter().all(|&v| v == 0));
    let signalled = host_secs();
    event.signal(1);
    pending.wait().expect("gated pass");
    assert_eq!(b.to_u32().expect("read"), data);
    assert_parts_after(&ctx, &[signalled]);
}

/// Without a feeder thread the committing thread feeds the pass itself:
/// the commit returns once the last part is submitted, and the pass is the
/// same.
#[test]
fn gated_pass_is_fed_inline_when_no_feeder_thread_starts() {
    let ctx = MetalContext::new().expect("metal context");
    ctx.fail_feeder_spawn.set(true);
    let n = 4096usize;
    let x: Vec<Tensor> =
        (0..4).map(|_| Tensor::zeros(&ctx, &[n], DType::U32).expect("x")).collect();
    let src: Vec<u32> = (0..n as u32).map(|i| i ^ 0x5a5a_5a5a).collect();
    x[0].write_bytes(bytemuck::cast_slice(&src)).expect("write");
    let event = ctx.new_watched_event().expect("event");
    let reached = ctx.new_shared_event().expect("reached");
    let pass = ctx.begin().expect("pass");
    pass.gate_host_waits();
    for k in 0..3u64 {
        if k > 0 {
            pass.signal_event(&reached, k).expect("signal");
            pass.wait_event(&event, k).expect("wait");
        }
        copy_words(&ctx, &pass, &x[k as usize], &x[k as usize + 1]).expect("copy");
    }
    let signaller = gate_signaller(&ctx, &event, &reached, 2);
    let pending = pass.commit().expect("commit");
    assert!(pending.feeder.is_none());
    assert_eq!(event.signaled_value(), 2, "the commit returned before the last part");
    pending.wait().expect("gated pass");
    let signalled = signaller.join().expect("signaller");
    assert_eq!(x[3].to_u32().expect("read"), src);
    assert!(ctx.fault().is_none(), "{:?}", ctx.fault());
    assert_parts_after(&ctx, &signalled);
}

/// A gate the host never opens faults the context after the feeder's
/// patience, and everything waiting on the context returns: a pass
/// committed meanwhile (waiting for the feeder), and the gated pass's drop.
#[test]
fn a_gate_never_opened_faults_the_context_after_the_patience() {
    let mut ctx = MetalContext::new().expect("metal context");
    ctx.gate_patience = Duration::from_millis(300);
    let n = 4096usize;
    let a = Tensor::zeros(&ctx, &[n], DType::U32).expect("a");
    let b = Tensor::zeros(&ctx, &[n], DType::U32).expect("b");
    let event = ctx.new_watched_event().expect("event");
    let pass = ctx.begin().expect("pass");
    pass.gate_host_waits();
    copy_words(&ctx, &pass, &a, &b).expect("copy");
    pass.wait_event(&event, 1).expect("wait");
    copy_words(&ctx, &pass, &b, &a).expect("copy");
    let t0 = Instant::now();
    let pending = pass.commit().expect("commit");
    let later = ctx.begin().expect("later");
    copy_words(&ctx, &later, &a, &b).expect("copy");
    let err = later.commit().map(|_| ()).expect_err("a pass behind a stuck gate");
    assert!(format!("{err:#}").contains("gave up"), "{err:#}");
    drop(pending);
    assert!(t0.elapsed() < Duration::from_secs(3), "{:?}", t0.elapsed());
    assert!(ctx.fault().is_some_and(|f| f.contains("gave up")), "{:?}", ctx.fault());
    // Nothing was left blocked on the queue: the event is not needed.
    assert_eq!(event.signaled_value(), 0);
}

/// A gated pass holds past Metal's queue wait limit (about 4 s,
/// `queue_wait_timeout_probe`): nothing behind the wait is on the queue
/// until the host signals.
#[test]
#[ignore = "takes 4.5 s"]
fn a_gated_wait_holds_past_the_queue_wait_limit() {
    let ctx = MetalContext::new().expect("metal context");
    let n = 1usize << 16;
    let a = Tensor::zeros(&ctx, &[n], DType::U32).expect("a");
    let b = Tensor::zeros(&ctx, &[n], DType::U32).expect("b");
    let data: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(7)).collect();
    a.write_bytes(bytemuck::cast_slice(&data)).expect("write");
    let event = ctx.new_watched_event().expect("event");
    let pass = ctx.begin().expect("pass");
    pass.gate_host_waits();
    pass.wait_event(&event, 1).expect("wait");
    copy_words(&ctx, &pass, &a, &b).expect("copy");
    let pending = pass.commit().expect("commit");
    std::thread::sleep(Duration::from_millis(4500));
    assert!(b.to_u32().expect("read").iter().all(|&v| v == 0));
    event.signal(1);
    pending.wait().expect("a gated wait past 4 s");
    assert_eq!(b.to_u32().expect("read"), data);
    assert!(ctx.fault().is_none(), "{:?}", ctx.fault());
}

/// A paced wait whose pass ends without reaching its milestone is not
/// paced: no sleep on the pacer's estimate, and the estimate is unchanged.
#[test]
fn a_milestone_never_reached_skips_pacing() {
    let ctx = MetalContext::new().expect("metal context");
    let n = 4096usize;
    let a = Tensor::zeros(&ctx, &[n], DType::U32).expect("a");
    let b = Tensor::zeros(&ctx, &[n], DType::U32).expect("b");
    let milestone = ctx.new_shared_event().expect("event");
    let mut pacer = Pacer::default();
    let t = Instant::now();
    pacer.begin(t);
    pacer.end(t + Duration::from_secs(2), false);
    let pass = ctx.begin().expect("pass");
    copy_words(&ctx, &pass, &a, &b).expect("copy");
    pass.set_milestone(&milestone, 1);
    let t0 = Instant::now();
    pass.commit().expect("commit").wait_paced(&mut pacer).expect("wait");
    assert!(t0.elapsed() < Duration::from_secs(1), "{:?}", t0.elapsed());
    assert_eq!(pacer.ema_secs, Some(2.0));
}

/// How long a queue-level wait holds when nothing signals it in time: `n`
/// (wait `e = k`, signal `f = k`) pairs, the host raising `e` to `k` at
/// `k * LILY_WAIT_PROBE_STEP_MS` (the first after
/// `LILY_WAIT_PROBE_FIRST_MS`, default one step). A host thread polls `f`
/// and records when the GPU passed each wait; a wait passed before the host
/// signalled it timed out. The pairs go in passes of `LILY_WAIT_PROBE_GROUP`
/// (default all in one), each committed `LILY_WAIT_PROBE_LEAD_MS` before its
/// first wait is signalled (default: all at once at the start).
/// `LILY_WAIT_PROBE_MB`: a copy of this many MiB after each wait, so each
/// wait sits between two command buffers that do work, as in a pass.
/// `cargo test --release --lib -- --ignored --nocapture queue_wait_timeout`.
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn queue_wait_timeout_probe() {
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_owned());
    let step_ms: u64 = env("LILY_WAIT_PROBE_STEP_MS", "250").parse().expect("step");
    let first_ms: u64 =
        env("LILY_WAIT_PROBE_FIRST_MS", &step_ms.to_string()).parse().expect("first");
    // `LILY_WAIT_PROBE_SCHEDULE`: the host's intervals instead, as
    // comma-separated `<count>x<ms>` runs (`20x100,10x300`).
    let schedule: Option<Vec<u64>> =
        std::env::var("LILY_WAIT_PROBE_SCHEDULE").ok().map(|v| {
            v.split(',')
                .flat_map(|run| {
                    let (count, ms) = run.split_once('x').expect("<count>x<ms>");
                    let ms: u64 = ms.parse().expect("ms");
                    std::iter::repeat_n(ms, count.parse().expect("count"))
                })
                .collect()
        });
    let n: u64 = match &schedule {
        Some(steps) => steps.len() as u64,
        None => env("LILY_WAIT_PROBE_N", "40").parse().expect("n"),
    };
    let group: u64 =
        env("LILY_WAIT_PROBE_GROUP", &n.to_string()).parse().expect("group");
    let lead_ms: Option<u64> =
        std::env::var("LILY_WAIT_PROBE_LEAD_MS").ok().map(|v| v.parse().expect("lead"));
    let mb: usize = env("LILY_WAIT_PROBE_MB", "0").parse().expect("mb");
    // `LILY_WAIT_PROBE_COPIES`: that many copies after each wait.
    let copies: usize = env("LILY_WAIT_PROBE_COPIES", "1").parse().expect("copies");
    let ctx = MetalContext::new().expect("metal context");
    let e = ctx.new_shared_event().expect("event");
    let f = ctx.new_shared_event().expect("event");
    let bufs = (mb > 0).then(|| {
        let words = mb << 18;
        (
            Tensor::zeros(&ctx, &[words], DType::U32).expect("src"),
            Tensor::zeros(&ctx, &[words], DType::U32).expect("dst"),
        )
    });
    // Host time at which `e` is raised to `k`.
    let signal_at = |k: u64| {
        Duration::from_millis(match &schedule {
            Some(steps) => steps[..k as usize].iter().sum(),
            None => first_ms + step_ms * (k - 1),
        })
    };
    let t0 = Instant::now();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let poller = {
        let (f, stop) = (f.clone(), stop.clone());
        std::thread::spawn(move || {
            let mut seen = vec![None; n as usize + 1];
            let mut last = 0u64;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let v = f.signaled_value().min(n);
                if v > last {
                    let t = t0.elapsed();
                    for k in last + 1..=v {
                        seen[k as usize] = Some(t);
                    }
                    last = v;
                }
                std::thread::sleep(Duration::from_micros(20));
            }
            seen
        })
    };
    // `LILY_WAIT_PROBE_GATED=1`: each pass is gated (submitted in parts as
    // the host signals), the fix for long cached prefill passes.
    let gated = env("LILY_WAIT_PROBE_GATED", "0") == "1";
    let encode = |from: u64, to: u64| {
        let pass = ctx.begin().expect("pass");
        if gated {
            pass.gate_host_waits();
        }
        for k in from..=to {
            pass.wait_event(&e, k).expect("wait");
            if let Some((src, dst)) = &bufs {
                for _ in 0..copies {
                    copy_words(&ctx, &pass, src, dst).expect("copy");
                }
            }
            pass.signal_event(&f, k).expect("signal");
        }
        pass.commit().expect("commit")
    };
    let spin_until = |at: Duration| {
        while t0.elapsed() < at {
            std::hint::spin_loop();
        }
    };
    let mut pending = Vec::new();
    let next_group = std::cell::Cell::new(1u64);
    let commit_due = |k_signalled: u64, pending: &mut Vec<_>| {
        // Commits every group whose commit time has come.
        while next_group.get() <= n {
            let from = next_group.get();
            let at = match lead_ms {
                None => Duration::ZERO,
                Some(lead) => {
                    signal_at(from).saturating_sub(Duration::from_millis(lead))
                }
            };
            if at > t0.elapsed() && from > k_signalled + 1 {
                break;
            }
            let to = (from + group - 1).min(n);
            pending.push((from, t0.elapsed(), encode(from, to)));
            next_group.set(to + 1);
        }
    };
    commit_due(0, &mut pending);
    let mut host = Vec::new();
    for k in 1..=n {
        let at = signal_at(k);
        loop {
            commit_due(k - 1, &mut pending);
            let now = t0.elapsed();
            if now >= at {
                break;
            }
            let wake = match lead_ms {
                Some(lead) if next_group.get() <= n => signal_at(next_group.get())
                    .saturating_sub(Duration::from_millis(lead))
                    .min(at),
                _ => at,
            };
            spin_until(wake.max(now));
        }
        e.signal(k);
        host.push(t0.elapsed());
    }
    let commits: Vec<(u64, Duration)> =
        pending.iter().map(|(k, t, _)| (*k, *t)).collect();
    for (_, _, p) in pending {
        if let Err(err) = p.wait() {
            println!("pass wait error: {err:#}");
        }
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let seen = poller.join().expect("poller");
    let mut timed_out = 0;
    let mut prev_gpu = Duration::ZERO;
    for k in 1..=n {
        if let Some((_, t)) = commits.iter().find(|(c, _)| *c == k) {
            println!(
                "        commit of the pass from k {k} at {:7.3} s",
                t.as_secs_f64()
            );
        }
        let gpu = seen[k as usize].unwrap_or_default();
        let early = gpu < host[k as usize - 1];
        timed_out += usize::from(early);
        println!(
            "k {k:>3}: host signal {:7.3} s, GPU passed {:7.3} s (+{:6.3} after the previous){}",
            host[k as usize - 1].as_secs_f64(),
            gpu.as_secs_f64(),
            gpu.saturating_sub(prev_gpu).as_secs_f64(),
            if early { "  <- timed out" } else { "" }
        );
        prev_gpu = gpu;
    }
    println!("timed out: {timed_out} of {n}; context fault: {:?}", ctx.fault());
}

/// Cost of one more queue-level signal right after each of a pass's 48
/// satisfied waits (a witness the host could check before it signals).
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn witness_signal_cost_probe() {
    let ctx = MetalContext::new().expect("metal context");
    let words = 16usize << 10;
    let a = Tensor::zeros(&ctx, &[words], DType::U32).expect("a");
    let b = Tensor::zeros(&ctx, &[words], DType::U32).expect("b");
    let (r, q, w) = (
        ctx.new_shared_event().expect("r"),
        ctx.new_shared_event().expect("q"),
        ctx.new_shared_event().expect("w"),
    );
    q.signal(u64::MAX / 2);
    let mut seq = 0u64;
    let mut run = |witness: bool| -> f64 {
        let pass = ctx.begin_concurrent().expect("pass");
        for _ in 0..48 {
            seq += 1;
            copy_words(&ctx, &pass, &a, &b).expect("copy");
            pass.signal_event(&r, seq).expect("signal");
            pass.wait_event(&q, seq).expect("wait");
            if witness {
                pass.signal_event(&w, seq).expect("witness");
            }
            copy_words(&ctx, &pass, &b, &a).expect("copy");
        }
        let t = pass.commit().expect("commit").wait_retain().expect("wait");
        let t = t.timing().expect("timing");
        t.gpu_end_secs - t.gpu_start_secs
    };
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    for round in 0..3 {
        let (mut plain, mut with) = (Vec::new(), Vec::new());
        for _ in 0..200 {
            plain.push(run(false));
            with.push(run(true));
        }
        eprintln!(
            "round {round}: pass of 48 waits {:.3} ms, with a witness signal after each {:.3} ms",
            median(plain) * 1e3,
            median(with) * 1e3
        );
    }
}
