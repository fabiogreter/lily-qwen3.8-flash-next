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
