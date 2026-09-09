//! Metal device context: pipeline compilation/caching, buffer allocation, and
//! serial or concurrent compute passes that batch kernel dispatches into one
//! command buffer.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, ensure};
use core::ffi::c_void;
use core::ptr::NonNull;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType,
    MTLEvent, MTLLanguageVersion, MTLLibrary, MTLResourceOptions, MTLSharedEvent, MTLSize,
};

use crate::tensor::Tensor;

pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub struct Kernel {
    pub pipeline: Pipeline,
}

impl Kernel {
    /// The SIMD width this pipeline executes at. Apple's own porting guidance is to
    /// read it from the pipeline rather than assume 32, and a simdgroup-scoped
    /// intrinsic has to be issued by exactly one simdgroup's worth of threads for the
    /// call to be convergent.
    pub fn thread_execution_width(&self) -> usize {
        self.pipeline.threadExecutionWidth()
    }

    /// Bytes of statically reserved threadgroup memory reported by the pipeline.
    pub fn static_threadgroup_memory_length(&self) -> usize {
        self.pipeline.staticThreadgroupMemoryLength()
    }
}

/// The MSL language version a kernel source requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MslVersion {
    /// Metal 3.1 (macOS 14+): the baseline for every kernel (`bfloat`).
    V3_1,
    /// Metal 4.0 (macOS 26+): tensor ops / MetalPerformancePrimitives (the
    /// neural-accelerator matmul path). Compilation fails on older systems,
    /// so callers must keep a 3.1 fallback.
    V4_0,
    /// Metal 4.1 (macOS 27+). objc2-metal 0.3.2 predates the named constant,
    /// but MTLLanguageVersion is an open integer wrapper and the SDK value is
    /// `(4 << 16) + 1`.
    V4_1,
}

impl MslVersion {
    fn language_version(self) -> MTLLanguageVersion {
        match self {
            MslVersion::V3_1 => MTLLanguageVersion::Version3_1,
            MslVersion::V4_0 => MTLLanguageVersion::Version4_0,
            MslVersion::V4_1 => MTLLanguageVersion((4 << 16) + 1),
        }
    }
}

/// Largest `params` slice any dispatch passes. Bounds the stack array
/// [`ComputePass::dispatch_at`] validates into; exceeding it is an error rather
/// than a silent heap fallback, so the bound stays honest.
const MAX_KERNEL_PARAMS: usize = 16;

pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Keyed by function name and MSL version.
    pipelines: Mutex<HashMap<(&'static str, MslVersion), Pipeline>>,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| anyhow!("no Metal device found"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("failed to create command queue"))?;
        let ctx = Self { device, queue, pipelines: Mutex::new(HashMap::new()) };
        // The production GEMM paths use native Metal tensor units.
        let family = ctx.apple_gpu_family();
        ensure!(
            family >= 10,
            "lily needs an Apple GPU with native tensor units (family 10, M5 and \
             later); this device reports family {family}"
        );
        Ok(ctx)
    }

    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Compiles MSL source through the framework's compiler (the same
    /// front-end an offline `xcrun metal` uses), reporting diagnostics on
    /// failure.
    pub fn compile_library(
        &self,
        source: &str,
        version: MslVersion,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(version.language_version());
        self.device
            .newLibraryWithSource_options_error(
                &NSString::from_str(source),
                Some(&options),
            )
            .map_err(|e| anyhow!("failed to compile Metal library: {e:?}"))
    }

    /// Creates the compute pipeline for `fn_name` from `library`.
    pub fn pipeline_from_library(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        fn_name: &str,
    ) -> Result<Pipeline> {
        let function =
            library.newFunctionWithName(&NSString::from_str(fn_name)).ok_or_else(
                || anyhow!("kernel function '{fn_name}' not found in source"),
            )?;
        self.device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| anyhow!("failed to create pipeline for {fn_name}: {e:?}"))
    }

    /// Returns the compute pipeline for `fn_name`, compiling `source` on first
    /// use. Kernels are compiled from source at runtime, so there is no offline
    /// `.metal` -> `.metallib` step in the build.
    pub fn pipeline(
        &self,
        fn_name: &'static str,
        source: &str,
        version: MslVersion,
    ) -> Result<Kernel> {
        let mut cache = self
            .pipelines
            .lock()
            .map_err(|e| anyhow!("pipeline cache lock poisoned: {e}"))?;
        if let Some(p) = cache.get(&(fn_name, version)) {
            return Ok(Kernel { pipeline: p.clone() });
        }

        let library = self
            .compile_library(source, version)
            .with_context(|| format!("for {fn_name}"))?;
        let pipeline = self.pipeline_from_library(&library, fn_name)?;
        cache.insert((fn_name, version), pipeline.clone());
        Ok(Kernel { pipeline })
    }

    /// Allocates a zero-initialized shared-storage buffer.
    pub fn new_buffer(&self, len: usize) -> Result<Buffer> {
        let buf = self
            .device
            .newBufferWithLength_options(
                len.max(1),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| anyhow!("failed to allocate {len}-byte buffer"))?;
        // Metal does not guarantee new buffer contents; callers rely on zeros.
        unsafe { core::ptr::write_bytes(buf.contents().as_ptr().cast::<u8>(), 0, len) };
        Ok(buf)
    }

    pub fn new_buffer_with_bytes(&self, bytes: &[u8]) -> Result<Buffer> {
        let buf = self.new_buffer(bytes.len())?;
        // SAFETY: the buffer was just allocated with exactly `bytes.len()` bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buf.contents().as_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        Ok(buf)
    }

    /// Starts a compute pass. Dispatches encoded on one pass execute serially
    /// The serial compute pass used by prefill and utility kernels.
    pub fn begin(&self) -> Result<ComputePass<'_>> {
        self.begin_pass(false)
    }

    fn begin_pass(&self, concurrent: bool) -> Result<ComputePass<'_>> {
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("failed to create command buffer"))?;
        let encoder = new_encoder(&cmd, concurrent)?;
        Ok(ComputePass {
            _ctx: std::marker::PhantomData,
            cmd,
            encoder: RefCell::new(encoder),
            ended: Cell::new(false),
            concurrent,
            done: RefCell::new(None),
        })
    }

    /// A host/GPU synchronization point for [`ComputePass::wait_event`]: the
    /// GPU blocks mid-pass until the host has signaled at least the awaited
    /// value. Lets a pass be committed before all of its inputs exist.
    pub fn new_shared_event(&self) -> Result<SharedEvent> {
        let event = self.device.newSharedEvent().ok_or_else(|| anyhow!("failed to create shared event"))?;
        Ok(SharedEvent { event })
    }

    /// A compute pass whose single encoder uses `MTLDispatchType::Concurrent` —
    /// Metal drops the serial encoder's implicit ordering, so dispatches may
    /// overlap and the caller owns every hazard via
    /// [`ComputePass::memory_barrier`] at dependency boundaries. Used by the
    /// concurrent decode path.
    pub fn begin_concurrent(&self) -> Result<ComputePass<'_>> {
        self.begin_pass(true)
    }

    /// Highest supported Apple GPU family number (10 for M5-class with
    /// native neural accelerators, 9 for M3/M4-class where Metal-4 tensor
    /// ops run via MPP emulation, 0 if none report). Metal exposes no
    /// direct "native tensor units" query; the family number is the
    /// architecture-policy signal.
    pub fn apple_gpu_family(&self) -> i64 {
        (1..=10i64)
            .rev()
            .find(|&n| {
                self.device.supportsFamily(objc2_metal::MTLGPUFamily(n as isize + 1000))
            })
            .unwrap_or(0)
    }
}

fn new_encoder(
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
    concurrent: bool,
) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
    if concurrent {
        let desc = MTLComputePassDescriptor::computePassDescriptor();
        desc.setDispatchType(MTLDispatchType::Concurrent);
        cmd.computeCommandEncoderWithDescriptor(&desc)
            .ok_or_else(|| anyhow!("failed to create concurrent compute encoder"))
    } else {
        cmd.computeCommandEncoder().ok_or_else(|| anyhow!("failed to create compute command encoder"))
    }
}

/// A monotonically signaled counter shared by host and GPU
/// (`MTLSharedEvent`). The GPU waits for `signaled_value >= v` where a pass
/// asked for it; the host raises the value once the awaited inputs are in
/// place. Waits are released in order, so callers use increasing values.
#[derive(Clone)]
pub struct SharedEvent {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
}

impl SharedEvent {
    /// Raises the counter to `value` (never lower it: a pass may be waiting
    /// for the current value).
    pub fn signal(&self, value: u64) {
        self.event.setSignaledValue(value);
    }

    pub fn signaled_value(&self) -> u64 {
        self.event.signaledValue()
    }

    /// Releases every pending and future wait, for error paths that would
    /// otherwise leave a committed pass blocked on the GPU forever.
    pub fn release_all(&self) {
        self.event.setSignaledValue(u64::MAX);
    }

    fn as_event(&self) -> &ProtocolObject<dyn MTLEvent> {
        ProtocolObject::from_ref(&*self.event)
    }
}

pub struct ComputePass<'a> {
    _ctx: std::marker::PhantomData<&'a MetalContext>,
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: RefCell<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    ended: Cell<bool>,
    concurrent: bool,
    /// The event and value [`Self::signal_done`] encoded, if any.
    done: RefCell<Option<(SharedEvent, u64)>>,
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        // Metal asserts if an encoder is released mid-encoding; close it so an
        // error path (`?` between dispatches) surfaces the error instead of
        // trapping in the destructor. The unfinished command buffer is simply
        // never committed.
        if !self.ended.get() {
            self.encoder.borrow().endEncoding();
        }
    }
}

/// How to map work items onto the GPU for one dispatch.
#[derive(Clone, Copy)]
pub enum Grid {
    /// `dispatchThreads` with a non-uniform grid of exactly this many threads.
    Threads { grid: (usize, usize, usize), threadgroup: (usize, usize, usize) },
    /// `dispatchThreadgroups`: kernels that reduce within a threadgroup need an
    /// exact threadgroup count (e.g. one per output row).
    Threadgroups { groups: (usize, usize, usize), threadgroup: (usize, usize, usize) },
}

impl<'a> ComputePass<'a> {
    /// Encodes one kernel dispatch with a byte offset per buffer binding — used
    /// to bind slices of a larger buffer (e.g. the q/k/v thirds of the fused
    /// GDN projection). Offsets must be 4-byte aligned per Metal's rules.
    pub fn dispatch_at(
        &self,
        kernel: &Kernel,
        buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
        params: &[&[u8]],
        grid: Grid,
    ) -> Result<()> {
        // Validate params before any encoder exists so an error can't leave an
        // encoder open -- but into a stack array, not a Vec. This runs once per
        // dispatch and a decode step submits hundreds of them, so the collect
        // would be a malloc/free pair on the latency-sensitive host path.
        ensure!(
            params.len() <= MAX_KERNEL_PARAMS,
            "{} kernel params exceeds MAX_KERNEL_PARAMS ({MAX_KERNEL_PARAMS})",
            params.len()
        );
        let mut param_buf =
            [(NonNull::<c_void>::dangling(), 0usize); MAX_KERNEL_PARAMS];
        for (slot, bytes) in param_buf.iter_mut().zip(params) {
            let ptr = NonNull::new(bytes.as_ptr().cast::<c_void>().cast_mut())
                .context("empty kernel param")?;
            *slot = (ptr, bytes.len());
        }
        let param_ptrs = param_buf.get(..params.len()).context("param slice")?;

        encode_dispatch(&self.encoder.borrow(), kernel, buffers, param_ptrs, grid);
        Ok(())
    }

    /// Orders one dependency level before the next on a concurrent encoder.
    pub fn level_barrier(&self, _written: &[&Tensor]) -> Result<()> {
        self.memory_barrier()
    }

    /// Makes the GPU block here until the host has signaled `event` to at
    /// least `value`. Everything encoded before completes first (the wait sits
    /// between two encoders), so this also acts as a full barrier. The pass
    /// can then be committed before the inputs read after this point exist;
    /// the host writes them and signals. A pass committed with an unsatisfied
    /// wait holds the whole queue, so every path after commit must signal
    /// (or [`SharedEvent::release_all`]).
    pub fn wait_event(&self, event: &SharedEvent, value: u64) -> Result<()> {
        self.encoder.borrow().endEncoding();
        self.cmd.encodeWaitForEvent_value(event.as_event(), value);
        let encoder = new_encoder(&self.cmd, self.concurrent)?;
        *self.encoder.borrow_mut() = encoder;
        Ok(())
    }

    /// Makes the GPU raise `event` to `value` once everything encoded before
    /// this point has completed (the host can wait on the event, or poll
    /// [`SharedEvent::signaled_value`], instead of on the whole pass).
    pub fn signal_event(&self, event: &SharedEvent, value: u64) -> Result<()> {
        self.encoder.borrow().endEncoding();
        self.cmd.encodeSignalEvent_value(event.as_event(), value);
        let encoder = new_encoder(&self.cmd, self.concurrent)?;
        *self.encoder.borrow_mut() = encoder;
        Ok(())
    }

    /// Orders all prior dispatches' buffer writes before every subsequent
    /// dispatch — the dependency-level boundary for [`MetalContext::
    /// begin_concurrent`] passes. Serial encoders safely ignore it.
    pub fn memory_barrier(&self) -> Result<()> {
        self.encoder.borrow().memoryBarrierWithScope(MTLBarrierScope::Buffers);
        Ok(())
    }

    /// Ends encoding and submits without blocking; pair with
    /// [`PendingPass::wait`]. Two passes in flight let the host encode step
    /// N+1 while the GPU runs step N (the queue executes them in submit
    /// order).
    pub fn commit(self) -> Result<PendingPass<'a>> {
        self.end()?.commit()
    }

    /// Ends encoding without submitting. The host may still write shared
    /// buffers the encoded kernels read (a decode step's per-token inputs)
    /// before [`EncodedPass::commit`] hands the work to the GPU.
    pub fn end(self) -> Result<EncodedPass<'a>> {
        self.encoder.borrow().endEncoding();
        self.ended.set(true);
        Ok(EncodedPass { _ctx: std::marker::PhantomData, cmd: self.cmd.clone(), done: self.done.borrow().clone() })
    }

    /// Encodes, as the pass's last command, a GPU signal of `event` to
    /// `value`, and remembers it: [`PendingPass::wait_paced`] then polls the
    /// event (which the GPU writes directly) instead of blocking on the
    /// command buffer, whose completion the host only learns of ~0.1 ms
    /// later. Call after everything else is encoded.
    pub fn signal_done(&self, event: &SharedEvent, value: u64) -> Result<()> {
        self.signal_event(event, value)?;
        *self.done.borrow_mut() = Some((event.clone(), value));
        Ok(())
    }

    /// Ends encoding, submits the command buffer, and blocks until the GPU
    /// finishes.
    pub fn commit_wait(self) -> Result<()> {
        self.encoder.borrow().endEncoding();
        self.ended.set(true);
        self.cmd.commit();
        self.cmd.waitUntilCompleted();
        Ok(())
    }
}

/// A fully encoded, not yet submitted command buffer.
pub struct EncodedPass<'a> {
    _ctx: std::marker::PhantomData<&'a MetalContext>,
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    done: Option<(SharedEvent, u64)>,
}

impl<'a> EncodedPass<'a> {
    /// Submits without blocking.
    pub fn commit(self) -> Result<PendingPass<'a>> {
        self.cmd.commit();
        Ok(PendingPass { _ctx: std::marker::PhantomData, cmd: self.cmd, done: self.done })
    }

    /// Drops the context lifetime so the pass can be kept inside long-lived
    /// state (passes encoded ahead for several possible outcomes). The
    /// context must outlive it; [`DetachedPass::attach`] restores the tie.
    pub fn detach(self) -> DetachedPass {
        DetachedPass { cmd: self.cmd, done: self.done }
    }
}

/// An [`EncodedPass`] held without its context lifetime; see
/// [`EncodedPass::detach`].
pub struct DetachedPass {
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    done: Option<(SharedEvent, u64)>,
}

impl DetachedPass {
    pub fn attach<'a>(self, _ctx: &'a MetalContext) -> EncodedPass<'a> {
        EncodedPass { _ctx: std::marker::PhantomData, cmd: self.cmd, done: self.done }
    }
}

/// One buffer-to-buffer copy for [`MetalContext::blit_copy`], in bytes.
pub struct BlitCopy<'t> {
    pub src: &'t Tensor,
    pub src_offset: usize,
    pub dst: &'t Tensor,
    pub dst_offset: usize,
    pub len: usize,
}

impl MetalContext {
    /// Copies byte ranges between buffers on the GPU's blit engine and waits.
    /// Used for session forks and recurrent-state checkpoints, where a few
    /// hundred megabytes move at memory speed instead of through the host.
    pub fn blit_copy(&self, copies: &[BlitCopy<'_>]) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("failed to create command buffer"))?;
        let blit = cmd
            .blitCommandEncoder()
            .ok_or_else(|| anyhow!("failed to create blit command encoder"))?;
        for copy in copies {
            let (src, src_base) = copy.src.binding();
            let (dst, dst_base) = copy.dst.binding();
            ensure!(
                copy.src_offset + copy.len <= copy.src.byte_len()
                    && copy.dst_offset + copy.len <= copy.dst.byte_len(),
                "blit copy out of range"
            );
            if copy.len == 0 {
                continue;
            }
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src,
                    src_base + copy.src_offset,
                    dst,
                    dst_base + copy.dst_offset,
                    copy.len,
                );
            }
        }
        blit.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }

    /// Metal's advice on how much this process may keep resident on the GPU
    /// (the wired limit in practice), in bytes.
    pub fn recommended_working_set(&self) -> usize {
        self.device.recommendedMaxWorkingSetSize() as usize
    }

    /// Bytes currently allocated on the device by this process.
    pub fn current_allocated(&self) -> usize {
        self.device.currentAllocatedSize()
    }
}

/// A committed-but-unawaited pass. Holding one while encoding the next pass
/// is the decode pipelining primitive; hosts must not read buffers the
/// pending pass writes until [`Self::wait`] returns.
pub struct PendingPass<'a> {
    _ctx: std::marker::PhantomData<&'a MetalContext>,
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    done: Option<(SharedEvent, u64)>,
}

#[derive(Clone, Copy, Debug)]
pub struct PassTiming {
    pub gpu_start_secs: f64,
    pub gpu_end_secs: f64,
}

pub struct CompletedPass {
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl CompletedPass {
    pub fn timing(&self) -> Result<PassTiming> {
        // A pass observed through its done signal may not have been marked
        // completed by the driver yet; the timestamps need that.
        self.cmd.waitUntilCompleted();
        let timing = PassTiming {
            gpu_start_secs: self.cmd.GPUStartTime(),
            gpu_end_secs: self.cmd.GPUEndTime(),
        };
        ensure!(
            timing.gpu_start_secs.is_finite()
                && timing.gpu_end_secs.is_finite()
                && timing.gpu_start_secs > 0.0
                && timing.gpu_end_secs > timing.gpu_start_secs,
            "invalid Metal GPU timestamps: {timing:?}"
        );
        Ok(timing)
    }
}

impl PendingPass<'_> {
    /// Blocks until the GPU finishes this pass.
    pub fn wait(self) -> Result<()> {
        self.wait_retain().map(|_| ())
    }

    /// Blocks until completion while retaining the completed command buffer.
    /// Benchmarks query its GPU clock only after their cadence timer ends.
    pub fn wait_retain(self) -> Result<CompletedPass> {
        self.cmd.waitUntilCompleted();
        self.finished()
    }

    /// Like [`Self::wait`], paced: sleeps until `pacer` predicts the pass is
    /// about to finish, then polls the pass's done signal (see
    /// [`ComputePass::signal_done`]) and returns within microseconds of it.
    /// A blocked thread would learn of the completion ~0.1 ms later. Without
    /// a done signal this is a plain blocking wait. Polling is capped so a
    /// wrong prediction cannot pin a core; the pacer learns from the outcome.
    pub fn wait_paced(self, pacer: &mut Pacer) -> Result<()> {
        self.wait_retain_paced(pacer).map(|_| ())
    }

    pub fn wait_retain_paced(self, pacer: &mut Pacer) -> Result<CompletedPass> {
        let Some((event, value)) = self.done.clone() else {
            let done = self.wait_retain()?;
            pacer.end(Instant::now(), false);
            return Ok(done);
        };
        if let Some(wake_at) = pacer.wake_at() {
            let now = Instant::now();
            if wake_at > now {
                std::thread::sleep(wake_at - now);
            }
        }
        let overslept = event.signaled_value() >= value;
        if !overslept {
            let spin_until = Instant::now() + SPIN_CAP;
            while event.signaled_value() < value {
                if Instant::now() > spin_until {
                    self.cmd.waitUntilCompleted();
                    break;
                }
                std::hint::spin_loop();
            }
        }
        pacer.end(Instant::now(), overslept);
        self.finished()
    }

    fn finished(self) -> Result<CompletedPass> {
        if self.cmd.status() == MTLCommandBufferStatus::Error {
            anyhow::bail!("Metal command buffer failed: {:?}", self.cmd.error());
        }
        Ok(CompletedPass { cmd: self.cmd })
    }
}

/// Longest a paced wait polls before falling back to a blocking wait.
const SPIN_CAP: Duration = Duration::from_millis(20);

/// Predicts when a repeating GPU pass completes so the host can sleep until
/// shortly before and then poll ([`PendingPass::wait_spinning`]). Tracks an
/// exponential average of the interval between `begin` and `end` marks.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pacer {
    began: Option<Instant>,
    ema_secs: Option<f64>,
}

impl Pacer {
    /// How far ahead of the predicted completion to wake: covers sleep
    /// overshoot and prediction error at the cost of that much polling.
    pub const MARGIN: Duration = Duration::from_micros(1500);

    /// Marks the start of an interval (the pass will run from about now).
    pub fn begin(&mut self, now: Instant) {
        self.began = Some(now);
    }

    /// When to wake for the pass begun last: `None` until an estimate exists.
    pub fn wake_at(&self) -> Option<Instant> {
        let (began, ema) = (self.began?, self.ema_secs?);
        let predicted = began + Duration::from_secs_f64(ema);
        Some(predicted.checked_sub(Self::MARGIN).unwrap_or(began))
    }

    /// Marks completion and folds the interval into the estimate. When the
    /// waiter `overslept` (the pass was already done on wake-up) the true
    /// completion time is unknown, so the estimate only shrinks; it grows
    /// again from observed completions. This keeps a too-long estimate from
    /// feeding on the late wake-ups it causes.
    pub fn end(&mut self, now: Instant, overslept: bool) {
        if let Some(began) = self.began.take() {
            let secs = now.duration_since(began).as_secs_f64();
            self.ema_secs = Some(match (self.ema_secs, overslept) {
                (Some(ema), true) => (0.85 * ema).min(secs),
                (Some(ema), false) => 0.7 * ema + 0.3 * secs,
                (None, _) => secs,
            });
        }
    }
}

fn encode_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    kernel: &Kernel,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
    params: &[(NonNull<c_void>, usize)],
    grid: Grid,
) {
    encoder.setComputePipelineState(&kernel.pipeline);
    for (i, (buf, offset)) in buffers.iter().enumerate() {
        unsafe { encoder.setBuffer_offset_atIndex(Some(buf), *offset, i) };
    }
    for (i, (ptr, len)) in params.iter().enumerate() {
        unsafe { encoder.setBytes_length_atIndex(*ptr, *len, buffers.len() + i) };
    }
    let size =
        |(w, h, d): (usize, usize, usize)| MTLSize { width: w, height: h, depth: d };
    match grid {
        Grid::Threads { grid, threadgroup } => {
            encoder
                .dispatchThreads_threadsPerThreadgroup(size(grid), size(threadgroup));
        }
        Grid::Threadgroups { groups, threadgroup } => {
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                size(groups),
                size(threadgroup),
            );
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/metal.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/unit/metal4.rs"]
mod tests_metal4;
