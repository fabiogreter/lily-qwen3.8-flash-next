//! Metal 4 command-model probe (ignored timing test): does the new command
//! queue submit faster, and does a queue-level event wait avoid the penalty
//! the classic in-buffer wait pays? Same kernel, same buffers, same host-side
//! method (wall time from commit until a done event the GPU raises last).

use std::ptr::NonNull;
use std::time::{Duration, Instant};

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4ComputeCommandEncoder, MTL4VisibilityOptions,
    MTLAllocation, MTLBuffer, MTLDevice, MTLResidencySet, MTLResidencySetDescriptor, MTLSize,
    MTLStages,
};

use super::*;
use crate::tensor::DType;

const SRC: &str = "kernel void probe_copy(device const uint* a [[buffer(0)]], device uint* b [[buffer(1)]], uint gid [[thread_position_in_grid]]) { b[gid] = a[gid]; }";
const TINY: usize = 1024;
const COVER: usize = 8 << 20; // 32 MiB copy, ~0.13 ms
const ITER: usize = 20;

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn spin_until(event: &SharedEvent, value: u64) -> bool {
    let deadline = Instant::now() + Duration::from_millis(200);
    while event.signaled_value() < value {
        if Instant::now() > deadline {
            return false;
        }
        std::hint::spin_loop();
    }
    true
}

#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn metal4_command_model_probe() {
    let ctx = MetalContext::new().expect("metal context");
    let device = ctx.device();
    let kernel = ctx.pipeline("probe_copy", SRC, MslVersion::V3_1).expect("kernel");
    let a = Tensor::zeros(&ctx, &[TINY], DType::U32).expect("a");
    let b = Tensor::zeros(&ctx, &[TINY], DType::U32).expect("b");
    let src = Tensor::zeros(&ctx, &[COVER], DType::U32).expect("src");
    let dst = Tensor::zeros(&ctx, &[COVER], DType::U32).expect("dst");
    let done = ctx.new_shared_event().expect("done");
    let gate = ctx.new_shared_event().expect("gate");
    let mut value = 0u64;
    let mut gate_value = 0u64;
    let tiny_grid = Grid::Threads { grid: (TINY, 1, 1), threadgroup: (256, 1, 1) };
    let cover_grid = Grid::Threads { grid: (COVER, 1, 1), threadgroup: (256, 1, 1) };

    // ---- Metal 4 setup: queue, allocator, residency, argument table.
    let queue4 = device.newMTL4CommandQueue().expect("MTL4 queue");
    let allocator = device.newCommandAllocator().expect("allocator");
    let residency = device.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).expect("residency set");
    for t in [&a, &b, &src, &dst] {
        residency.addAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(t.buffer()));
    }
    residency.commit();
    residency.requestResidency();
    queue4.addResidencySet(&residency);
    let table_desc = MTL4ArgumentTableDescriptor::new();
    table_desc.setMaxBufferBindCount(2);
    let table = device.newArgumentTableWithDescriptor_error(&table_desc).expect("argument table");
    let commit4 = |cmd: &ProtocolObject<dyn MTL4CommandBuffer>| {
        let mut list = [NonNull::from(cmd)];
        unsafe { queue4.commit_count(NonNull::from(&mut list[0]), 1) };
    };
    // Encodes `dispatches` tiny copies (barrier between them) into a fresh MTL4
    // command buffer, plus `cover` big copies before them when asked.
    let encode4 = |cover: usize, dispatches: usize| -> objc2::rc::Retained<ProtocolObject<dyn MTL4CommandBuffer>> {
        let cmd = device.newCommandBuffer().expect("MTL4 command buffer");
        cmd.beginCommandBufferWithAllocator(&allocator);
        let enc = cmd.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(&kernel.pipeline);
        enc.setArgumentTable(Some(&table));
        let tg = MTLSize { width: 256, height: 1, depth: 1 };
        for _ in 0..cover {
            unsafe {
                table.setAddress_atIndex(src.buffer().gpuAddress(), 0);
                table.setAddress_atIndex(dst.buffer().gpuAddress(), 1);
            }
            enc.dispatchThreads_threadsPerThreadgroup(MTLSize { width: COVER, height: 1, depth: 1 }, tg);
            enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(MTLStages::Dispatch, MTLStages::Dispatch, MTL4VisibilityOptions::None);
        }
        unsafe {
            table.setAddress_atIndex(a.buffer().gpuAddress(), 0);
            table.setAddress_atIndex(b.buffer().gpuAddress(), 1);
        }
        for _ in 0..dispatches {
            enc.dispatchThreads_threadsPerThreadgroup(MTLSize { width: TINY, height: 1, depth: 1 }, tg);
            enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(MTLStages::Dispatch, MTLStages::Dispatch, MTL4VisibilityOptions::None);
        }
        enc.endEncoding();
        cmd.endCommandBuffer();
        cmd
    };
    let encode3 = |cover: usize, dispatches: usize, wait: Option<u64>, signal: u64| -> EncodedPass<'_> {
        let pass = ctx.begin_concurrent().expect("pass");
        for _ in 0..cover {
            pass.dispatch_at(&kernel, &[src.binding(), dst.binding()], &[], cover_grid).expect("cover");
            pass.level_barrier(&[]).expect("barrier");
        }
        if let Some(v) = wait {
            pass.wait_event(&gate, v).expect("wait");
        }
        for _ in 0..dispatches {
            pass.dispatch_at(&kernel, &[a.binding(), b.binding()], &[], tiny_grid).expect("tiny");
            pass.level_barrier(&[]).expect("barrier");
        }
        pass.signal_done(&done, signal).expect("signal");
        pass.end().expect("end")
    };

    // ---- 1. Submission + completion wall time, 1 and 800 dispatches.
    for dispatches in [1usize, 800] {
        let mut classic = Vec::new();
        for _ in 0..ITER {
            value += 1;
            let encoded = encode3(0, dispatches, None, value);
            let t0 = Instant::now();
            let _pending = encoded.commit().expect("commit");
            assert!(spin_until(&done, value), "classic pass never signaled");
            classic.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let mut m4 = Vec::new();
        for _ in 0..ITER {
            value += 1;
            let cmd = encode4(0, dispatches);
            let t0 = Instant::now();
            commit4(&cmd);
            queue4.signalEvent_value(done.as_event(), value);
            assert!(spin_until(&done, value), "MTL4 pass never signaled");
            m4.push(t0.elapsed().as_secs_f64() * 1e3);
            allocator.reset();
        }
        eprintln!("commit-to-done, {dispatches} dispatches: classic {:.3} ms, Metal 4 {:.3} ms", median(&mut classic), median(&mut m4));
    }

    // ---- 2. Host encode time for 800 dispatches.
    {
        let t0 = Instant::now();
        for _ in 0..5 {
            value += 1;
            drop(encode3(0, 800, None, value));
        }
        let classic = t0.elapsed().as_secs_f64() * 1e3 / 5.0;
        let t0 = Instant::now();
        for _ in 0..5 {
            let cmd = encode4(0, 800);
            drop(cmd);
            allocator.reset();
        }
        let m4 = t0.elapsed().as_secs_f64() * 1e3 / 5.0;
        eprintln!("host encode, 800 dispatches: classic {classic:.2} ms, Metal 4 {m4:.2} ms");
    }

    // ---- 3. A cover pass, then a dependent 800-dispatch pass behind a wait
    // that is unsatisfied at submission and signaled 0.5 ms after. Penalty =
    // wall - wall without the wait - 0.5 ms.
    let sleep = Duration::from_micros(500);
    let mut base3 = Vec::new();
    let mut wait3 = Vec::new();
    for with_wait in [false, true] {
        for _ in 0..ITER {
            value += 1;
            gate_value += 1;
            let a_pass = encode3(1, 0, None, value);
            value += 1;
            let b_pass = encode3(0, 800, with_wait.then_some(gate_value), value);
            let t0 = Instant::now();
            let _pa = a_pass.commit().expect("commit a");
            let _pb = b_pass.commit().expect("commit b");
            if with_wait {
                std::thread::sleep(sleep);
                gate.signal(gate_value);
            }
            assert!(spin_until(&done, value), "classic dependent pass never signaled");
            let wall = t0.elapsed().as_secs_f64() * 1e3;
            if with_wait { wait3.push(wall) } else { base3.push(wall) }
        }
    }
    let mut base4 = Vec::new();
    let mut wait4 = Vec::new();
    for with_wait in [false, true] {
        for _ in 0..ITER {
            value += 1;
            gate_value += 1;
            let a_cmd = encode4(1, 0);
            let b_cmd = encode4(0, 800);
            let t0 = Instant::now();
            commit4(&a_cmd);
            if with_wait {
                queue4.waitForEvent_value(gate.as_event(), gate_value);
            }
            commit4(&b_cmd);
            queue4.signalEvent_value(done.as_event(), value);
            if with_wait {
                std::thread::sleep(sleep);
                gate.signal(gate_value);
            }
            assert!(spin_until(&done, value), "MTL4 dependent pass never signaled");
            let wall = t0.elapsed().as_secs_f64() * 1e3;
            if with_wait { wait4.push(wall) } else { base4.push(wall) }
            allocator.reset();
        }
    }
    let (b3, w3, b4, w4) = (median(&mut base3), median(&mut wait3), median(&mut base4), median(&mut wait4));
    eprintln!(
        "dependent pass behind an unsatisfied wait (signal +0.5 ms): classic {:.3} ms -> {:.3} ms (penalty {:.3}), Metal 4 {:.3} ms -> {:.3} ms (penalty {:.3})",
        b3, w3, w3 - b3 - 0.5, b4, w4, w4 - b4 - 0.5
    );

    // Not probed again: committing an already committed Metal 4 command
    // buffer a second time aborts the process (IOGPUMetal4CommandQueue
    // asserts `allocator && storage`), so command buffers are single-use and
    // per-step host encoding cannot be skipped that way.
}
