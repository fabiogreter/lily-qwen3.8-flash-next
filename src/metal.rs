//! Metal device context on the Metal 4 command model: pipeline
//! compilation/caching, buffer allocation with explicit residency, and serial
//! or concurrent compute passes that batch kernel dispatches into command
//! buffers submitted through an `MTL4CommandQueue`.
//!
//! What Metal 4 changes for the engine, and how this module absorbs it so the
//! rest of the crate keeps the `ComputePass` API it always had:
//!
//! - **No implicit ordering or hazard tracking.** Dispatches in one encoder
//!   run concurrently unless a barrier says otherwise, and consecutive command
//!   buffers on a queue may overlap. Serial passes therefore place a barrier
//!   after every dispatch, concurrent passes keep their explicit level
//!   barriers, and every command buffer opens with a queue-stage barrier so a
//!   pass sees the writes of the pass before it. All barriers flush caches
//!   (`MTL4VisibilityOptions::Device`); the execution-only variant is not a
//!   memory barrier.
//! - **No `setBuffer`/`setBytes`.** Kernel arguments bind through an argument
//!   table by GPU address. Inline params are copied into a per-pass arena
//!   buffer and bound by address; kernel binding indices are unchanged.
//! - **No residency tracking.** Every buffer the context allocates joins one
//!   residency set attached to the queue and leaves it when dropped; the set
//!   is re-committed before the next submission when membership changed.
//! - **No completion API on command buffers.** Each submission is followed by
//!   a queue-level signal on the context's fence event; waiting on a pass is
//!   waiting on that counter. Errors and GPU timestamps arrive through the
//!   commit feedback handler.
//! - **Command memory is explicit.** Passes encode into a command allocator
//!   from a pool; an allocator returns to the pool once its pass completed
//!   (or was dropped un-committed). Command buffers are single-use.
//! - **Per-kernel profile mode** (`LILY_KERNEL_PROFILE=1` or
//!   [`MetalContext::new_with_profile`]): every dispatch is closed into its own
//!   command buffer so the commit feedback's GPU start/end times apply to that
//!   one kernel; see [`profile`]. Off by default, and then free.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use block2::RcBlock;
use core::ptr::NonNull;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_foundation::{NSArray, NSError, NSString};
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4CommandQueueError, MTL4CommitFeedback, MTL4CommitOptions,
    MTL4ComputeCommandEncoder, MTL4VisibilityOptions, MTLAllocation, MTLBuffer,
    MTLCompileOptions, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLEvent, MTLLanguageVersion, MTLLibrary, MTLResidencySet, MTLResidencySetDescriptor,
    MTLResourceOptions, MTLSharedEvent, MTLSize, MTLStages,
};

use crate::tensor::Tensor;

pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// A device buffer handle. Cloning shares the allocation; the allocation
/// leaves the residency set when the last handle drops. Like the context
/// that allocated it, a buffer stays on the thread it was created on (Metal
/// buffers are not `Send` in these bindings, and never were here).
pub type Buffer = Rc<GpuBuffer>;

/// The commit feedback callback Metal invokes per command buffer.
type FeedbackHandler = RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTL4CommitFeedback>>)>;

pub struct Kernel {
    pub pipeline: Pipeline,
    /// The kernel function's name (the per-kernel profile's key).
    pub name: &'static str,
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

/// Metal's buffer-argument limit; also the argument table's size.
const MAX_BUFFER_BINDINGS: usize = 31;

/// Alignment of every inline param in the arena. Metal asks for 4 on Apple
/// GPUs; 16 also satisfies the widest param type kernels take (`uint4`).
const PARAM_ALIGN: usize = 16;

/// First arena chunk; a pass that outgrows it gets a larger one appended.
const ARENA_CHUNK: usize = 1 << 20;

/// The stages compute passes run in. Copies go through the compute encoder
/// too, so barriers cover both.
fn compute_stages() -> MTLStages {
    MTLStages::Dispatch | MTLStages::Blit
}

/// Per-kernel GPU timing. In profile mode ([`MetalContext::new_with_profile`],
/// or `LILY_KERNEL_PROFILE=1` at [`MetalContext::new`]) every dispatch of a
/// pass is closed into its own command buffer, so the commit feedback's GPU
/// start/end times measure that one kernel. Each completed pass's breakdown
/// is recorded process-wide until [`take`] drains it; nothing is recorded
/// otherwise. The mode changes the work's shape (one command buffer per
/// dispatch instead of one per pass, no encoder barriers), so wall time and
/// pass spans under it are not production numbers; the per-kernel times are.
pub mod profile {
    use std::sync::Mutex;

    /// Label of passes nobody named through [`super::ComputePass::set_label`].
    pub const UNLABELED: &str = "(unlabeled)";

    /// One dispatch's GPU time.
    #[derive(Clone, Debug)]
    pub struct KernelSample {
        pub name: &'static str,
        pub gpu_secs: f64,
    }

    /// One completed pass: its label, its dispatches in encode order (copies
    /// appear as `copy_buffer`), and its GPU span from the first command
    /// buffer's start to the last one's end. The span minus the summed
    /// samples is time between command buffers: the profile mode's own cost
    /// plus whatever gaps exist without it (a parked pass's wait included).
    #[derive(Clone, Debug)]
    pub struct PassProfile {
        pub label: &'static str,
        pub span_secs: f64,
        pub kernels: Vec<KernelSample>,
    }

    static PASSES: Mutex<Vec<PassProfile>> = Mutex::new(Vec::new());

    pub(super) fn record(pass: PassProfile) {
        if let Ok(mut passes) = PASSES.lock() {
            passes.push(pass);
        }
    }

    /// Drains every pass profile recorded so far (in completion order).
    pub fn take() -> Vec<PassProfile> {
        PASSES.lock().map(|mut passes| std::mem::take(&mut *passes)).unwrap_or_default()
    }
}

pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    residency: Rc<Residency>,
    pool: Rc<Pool>,
    /// Completion counter: every submission ends with a queue-level signal
    /// of the next value; waiting on a pass waits on its value.
    fence: SharedEvent,
    fence_value: AtomicU64,
    /// Serializes submissions so fence values are signaled in order.
    submit: Mutex<()>,
    /// First GPU error the commit feedback reported, surfaced by the next
    /// wait or commit.
    fault: Arc<Fault>,
    /// Keyed by function name and MSL version.
    pipelines: Mutex<HashMap<(&'static str, MslVersion), Pipeline>>,
    /// Per-kernel profile mode; see [`profile`].
    profile: bool,
}

impl MetalContext {
    /// A context on the system default device; the per-kernel profile mode
    /// is on when `LILY_KERNEL_PROFILE` is set to anything but `0`.
    pub fn new() -> Result<Self> {
        let profile = std::env::var_os("LILY_KERNEL_PROFILE").is_some_and(|v| !v.is_empty() && v != "0");
        Self::new_with_profile(profile)
    }

    /// Like [`Self::new`], with the per-kernel profile mode chosen explicitly.
    pub fn new_with_profile(profile: bool) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| anyhow!("no Metal device found"))?;
        let queue = device
            .newMTL4CommandQueue()
            .ok_or_else(|| anyhow!("failed to create a Metal 4 command queue (macOS 26 or later is required)"))?;
        let residency = Rc::new(Residency::new(&device)?);
        queue.addResidencySet(&residency.set);
        let fence = SharedEvent::new(&device)?;
        let pool = Rc::new(Pool { device: device.clone(), residency: residency.clone(), free: RefCell::new(Vec::new()) });
        let ctx = Self {
            device,
            queue,
            residency,
            pool,
            fence,
            fence_value: AtomicU64::new(0),
            submit: Mutex::new(()),
            fault: Arc::new(Fault::default()),
            pipelines: Mutex::new(HashMap::new()),
            profile,
        };
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

    /// Whether the per-kernel profile mode is on; see [`profile`].
    pub fn profiling(&self) -> bool {
        self.profile
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
            return Ok(Kernel { pipeline: p.clone(), name: fn_name });
        }

        let library = self
            .compile_library(source, version)
            .with_context(|| format!("for {fn_name}"))?;
        let pipeline = self.pipeline_from_library(&library, fn_name)?;
        cache.insert((fn_name, version), pipeline.clone());
        Ok(Kernel { pipeline, name: fn_name })
    }

    /// Allocates a zero-initialized shared-storage buffer and makes it
    /// resident for the queue.
    pub fn new_buffer(&self, len: usize) -> Result<Buffer> {
        self.residency.new_buffer(&self.device, len)
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

    /// Number of allocations currently in the queue's residency set (every
    /// live buffer this context allocated, plus the pool's arenas).
    pub fn resident_allocations(&self) -> usize {
        self.residency.set.allocationCount()
    }

    /// Starts a serial compute pass: dispatches execute in encode order (a
    /// barrier follows each one). Used by prefill and utility kernels.
    pub fn begin(&self) -> Result<ComputePass<'_>> {
        self.begin_pass(false)
    }

    /// A compute pass whose dispatches may overlap; the caller owns every
    /// hazard via [`ComputePass::memory_barrier`] at dependency boundaries.
    /// Used by the concurrent decode path.
    pub fn begin_concurrent(&self) -> Result<ComputePass<'_>> {
        self.begin_pass(true)
    }

    fn begin_pass(&self, concurrent: bool) -> Result<ComputePass<'_>> {
        let slot = self.pool.take()?;
        Ok(ComputePass {
            ctx: self,
            program: RefCell::new(Vec::new()),
            open: RefCell::new(None),
            concurrent,
            done: RefCell::new(None),
            label: Cell::new(profile::UNLABELED),
            names: RefCell::new(Vec::new()),
            segment_kernel: Cell::new(None),
            slot: RefCell::new(slot),
        })
    }

    /// A host/GPU synchronization point for [`ComputePass::wait_event`]: the
    /// GPU blocks between two command buffers until the host has signaled at
    /// least the awaited value. Lets a pass be committed before all of its
    /// inputs exist.
    pub fn new_shared_event(&self) -> Result<SharedEvent> {
        SharedEvent::new(&self.device)
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

    /// Metal's advice on how much this process may keep resident on the GPU
    /// (the wired limit in practice), in bytes.
    pub fn recommended_working_set(&self) -> usize {
        self.device.recommendedMaxWorkingSetSize() as usize
    }

    /// Bytes currently allocated on the device by this process.
    pub fn current_allocated(&self) -> usize {
        self.device.currentAllocatedSize()
    }

    /// The first GPU error the commit feedback reported on this context, if
    /// any. A fault is permanent: every later submission and wait fails with
    /// it, because the queue's state after a timeout or hang (and whatever
    /// its last command buffers left in memory) is not trustworthy. The
    /// owner recovers by replacing the context and everything allocated
    /// from it.
    pub fn fault(&self) -> Option<String> {
        self.fault.message()
    }

    /// Records a fault as if the commit feedback had reported `message`
    /// (tests and the server's hidden fault-injection flag): the next
    /// submission or wait fails exactly as after a real GPU error.
    pub fn inject_fault(&self, message: &str) {
        self.fault.record(message.to_owned());
    }

    /// Copies byte ranges between buffers on the GPU and waits. Used for
    /// session forks and recurrent-state checkpoints, where a few hundred
    /// megabytes move at memory speed instead of through the host.
    pub fn blit_copy(&self, copies: &[BlitCopy<'_>]) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        let pass = self.begin()?;
        for copy in copies {
            ensure!(
                copy.src_offset + copy.len <= copy.src.byte_len()
                    && copy.dst_offset + copy.len <= copy.dst.byte_len(),
                "blit copy out of range"
            );
            if copy.len == 0 {
                continue;
            }
            let (src, src_base) = copy.src.binding();
            let (dst, dst_base) = copy.dst.binding();
            pass.copy_buffer(src, src_base + copy.src_offset, dst, dst_base + copy.dst_offset, copy.len)?;
        }
        pass.commit_wait()
    }

    /// Submits a pass's program in order and signals the fence after it.
    /// Returns the fence value to wait for.
    fn submit(&self, program: &[Item], feedback: &Arc<Feedback>) -> Result<u64> {
        self.fault.check()?;
        let _guard = self.submit.lock().map_err(|e| anyhow!("submit lock poisoned: {e}"))?;
        self.residency.flush();
        let mut index = 0usize;
        for item in program {
            match item {
                Item::Cmd(cmd) => {
                    let options = MTL4CommitOptions::new();
                    let handler = feedback.handler(&self.fault, index);
                    index += 1;
                    // SAFETY: the block pointer is valid for the call; the
                    // options object copies the block.
                    unsafe { options.addFeedbackHandler(RcBlock::as_ptr(&handler)) };
                    let mut ptr = NonNull::from(&**cmd);
                    // SAFETY: one valid command buffer pointer, count 1.
                    unsafe { self.queue.commit_count_options(NonNull::from(&mut ptr), 1, &options) };
                }
                Item::Wait(event, value) => self.queue.waitForEvent_value(event.as_event(), *value),
                Item::Signal(event, value) => self.queue.signalEvent_value(event.as_event(), *value),
            }
        }
        let value = self.fence_value.fetch_add(1, Ordering::AcqRel) + 1;
        self.queue.signalEvent_value(self.fence.as_event(), value);
        Ok(value)
    }

    /// Blocks until the fence reaches `value`, bailing early on a reported
    /// GPU fault.
    fn wait_fence(&self, value: u64) -> Result<()> {
        while !self.fence.event.waitUntilSignaledValue_timeoutMS(value, 1000) {
            self.fault.check()?;
        }
        self.fault.check()
    }
}

/// The context's residency set plus the bookkeeping to re-commit it lazily.
struct Residency {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    /// Membership changed since the last commit.
    dirty: Cell<bool>,
}

impl Residency {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let desc = MTLResidencySetDescriptor::new();
        // SAFETY: plain property set on a fresh descriptor.
        unsafe { desc.setInitialCapacity(4096) };
        let set = device
            .newResidencySetWithDescriptor_error(&desc)
            .map_err(|e| anyhow!("failed to create residency set: {e:?}"))?;
        set.commit();
        set.requestResidency();
        Ok(Self { set, dirty: Cell::new(false) })
    }

    fn new_buffer(self: &Rc<Self>, device: &ProtocolObject<dyn MTLDevice>, len: usize) -> Result<Buffer> {
        let raw = device
            .newBufferWithLength_options(len.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("failed to allocate {len}-byte buffer"))?;
        // Metal does not guarantee new buffer contents; callers rely on zeros.
        unsafe { core::ptr::write_bytes(raw.contents().as_ptr().cast::<u8>(), 0, len) };
        let address = raw.gpuAddress();
        self.set.addAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&*raw));
        self.dirty.set(true);
        Ok(Rc::new(GpuBuffer { raw, address, residency: self.clone() }))
    }

    fn remove(&self, raw: &ProtocolObject<dyn MTLBuffer>) {
        self.set.removeAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(raw));
        // Commit now so the memory is released promptly (the set retains
        // members until commit); additions wait for the next submission.
        self.set.commit();
        self.dirty.set(false);
    }

    /// Commits pending additions; called before every submission.
    fn flush(&self) {
        if self.dirty.replace(false) {
            self.set.commit();
        }
    }
}

/// A device buffer that is resident for the context's queue while alive.
/// Derefs to the underlying `MTLBuffer`.
pub struct GpuBuffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    address: u64,
    residency: Rc<Residency>,
}

impl GpuBuffer {
    /// The buffer's GPU virtual address (what argument tables bind).
    pub fn address(&self) -> u64 {
        self.address
    }
}

impl Deref for GpuBuffer {
    type Target = ProtocolObject<dyn MTLBuffer>;
    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        self.residency.remove(&self.raw);
    }
}

/// Command allocators (plus the params arena and argument table that live
/// with them) recycled across passes.
struct Pool {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    residency: Rc<Residency>,
    free: RefCell<Vec<SlotInner>>,
}

impl Pool {
    fn take(self: &Rc<Self>) -> Result<Slot> {
        let recycled = self.free.borrow_mut().pop();
        let inner = match recycled {
            Some(inner) => inner,
            None => {
                let allocator = self
                    .device
                    .newCommandAllocator()
                    .ok_or_else(|| anyhow!("failed to create a Metal 4 command allocator"))?;
                let desc = MTL4ArgumentTableDescriptor::new();
                desc.setMaxBufferBindCount(MAX_BUFFER_BINDINGS);
                let table = self
                    .device
                    .newArgumentTableWithDescriptor_error(&desc)
                    .map_err(|e| anyhow!("failed to create argument table: {e:?}"))?;
                SlotInner { allocator, table, arena: Arena::default() }
            }
        };
        Ok(Slot { inner: Some(inner), pool: self.clone() })
    }

    fn recycle(&self, mut inner: SlotInner) {
        inner.allocator.reset();
        inner.arena.reset();
        self.free.borrow_mut().push(inner);
    }
}

struct SlotInner {
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    arena: Arena,
}

/// A pool slot on loan to a pass; returns to the pool on drop. Holders that
/// committed work must not drop it before the GPU finished (the allocator
/// is reset on return).
struct Slot {
    inner: Option<SlotInner>,
    pool: Rc<Pool>,
}

impl Slot {
    fn get(&mut self) -> &mut SlotInner {
        self.inner.as_mut().expect("slot in use")
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            self.pool.recycle(inner);
        }
    }
}

/// Bump allocator for inline kernel params, in shared buffers the GPU reads
/// by address. Chunks stay allocated across passes; the cursor rewinds.
#[derive(Default)]
struct Arena {
    chunks: Vec<Buffer>,
    chunk: usize,
    cursor: usize,
}

impl Arena {
    fn reset(&mut self) {
        self.chunk = 0;
        self.cursor = 0;
    }

    fn push(&mut self, pool: &Pool, bytes: &[u8]) -> Result<u64> {
        let len = bytes.len().max(1);
        loop {
            if let Some(chunk) = self.chunks.get(self.chunk) {
                let start = self.cursor.next_multiple_of(PARAM_ALIGN);
                if start + len <= chunk.length() {
                    // SAFETY: shared-storage buffer; [start, start+len) is
                    // within it and unused by any encoded dispatch of this
                    // arena's current lease.
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            chunk.contents().as_ptr().cast::<u8>().add(start),
                            bytes.len(),
                        );
                    }
                    self.cursor = start + len;
                    return Ok(chunk.address() + start as u64);
                }
                self.chunk += 1;
                self.cursor = 0;
                continue;
            }
            let size = ARENA_CHUNK.max(len.next_power_of_two()) << self.chunks.len().min(8);
            let chunk = pool.residency.new_buffer(&pool.device, size)?;
            self.chunks.push(chunk);
        }
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
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let event = device.newSharedEvent().ok_or_else(|| anyhow!("failed to create shared event"))?;
        Ok(Self { event })
    }

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

/// First GPU error reported through commit feedback, context-wide.
#[derive(Default)]
struct Fault {
    message: Mutex<Option<String>>,
}

impl Fault {
    fn record(&self, message: String) {
        if let Ok(mut slot) = self.message.lock() {
            slot.get_or_insert(message);
        }
    }

    fn check(&self) -> Result<()> {
        match self.message.lock() {
            Ok(slot) => match &*slot {
                Some(message) => bail!("Metal command buffer failed: {message}"),
                None => Ok(()),
            },
            Err(e) => bail!("fault lock poisoned: {e}"),
        }
    }

    fn message(&self) -> Option<String> {
        self.message.lock().ok().and_then(|slot| slot.clone())
    }
}

/// One line naming a Metal error: domain and code, what the code means for
/// the Metal 4 command-queue domain, the system's description, and the
/// underlying errors Metal attaches (the driver's own diagnosis, which the
/// generic "operation couldn't be completed" text hides).
fn describe_error(error: &NSError) -> String {
    let domain = error.domain().to_string();
    let code = error.code();
    let mut text = format!("{domain} error {code}");
    if domain == "MTL4CommandQueueErrorDomain" {
        let meaning = match MTL4CommandQueueError(code) {
            MTL4CommandQueueError::Timeout => Some("Timeout: the workload took longer to execute than the system allows"),
            MTL4CommandQueueError::NotPermitted => Some("NotPermitted: the process has no access to a GPU device"),
            MTL4CommandQueueError::OutOfMemory => Some("OutOfMemory: the GPU lacks the memory to execute a command buffer"),
            MTL4CommandQueueError::DeviceRemoved => Some("DeviceRemoved: the GPU was removed before the command buffer completed"),
            MTL4CommandQueueError::AccessRevoked => {
                Some("AccessRevoked: the system revoked GPU access after too many timeouts or hangs")
            }
            MTL4CommandQueueError::Internal => Some("Internal: a problem inside the Metal framework"),
            _ => None,
        };
        if let Some(meaning) = meaning {
            text.push_str(&format!(" ({meaning})"));
        }
    }
    text.push_str(&format!(": {}", error.localizedDescription()));
    let info = error.userInfo();
    let one_line = |e: &NSError| format!("{} error {}: {}", e.domain(), e.code(), e.localizedDescription());
    let mut underlying: Vec<String> = Vec::new();
    // The key strings are what `NSUnderlyingErrorKey` and
    // `NSMultipleUnderlyingErrorsKey` hold (reading the extern statics
    // needs `unsafe`; the literals do not).
    if let Some(object) = info.objectForKey(&NSString::from_str("NSUnderlyingError"))
        && let Some(e) = object.downcast_ref::<NSError>()
    {
        underlying.push(one_line(e));
    }
    if let Some(object) = info.objectForKey(&NSString::from_str("NSMultipleUnderlyingErrorsKey"))
        && let Some(list) = object.downcast_ref::<NSArray<AnyObject>>()
    {
        for item in list.iter() {
            if let Some(e) = item.downcast_ref::<NSError>() {
                let line = one_line(e);
                if !underlying.contains(&line) {
                    underlying.push(line);
                }
            }
        }
    }
    if !underlying.is_empty() {
        text.push_str(&format!("; underlying: {}", underlying.join(", ")));
    }
    text
}

/// Commit feedback for one pass: how many of its command buffers reported,
/// their GPU time span, and any error. In profile mode also each command
/// buffer's own span, recorded as a [`profile::PassProfile`] once the last
/// handler has run (handlers may arrive out of order).
struct Feedback {
    expected: usize,
    profile: Option<ProfileInfo>,
    state: Mutex<FeedbackState>,
}

/// What the profile needs besides the timestamps: the pass's label and the
/// kernel of each command buffer, by submission index.
struct ProfileInfo {
    label: &'static str,
    names: Vec<&'static str>,
}

#[derive(Default)]
struct FeedbackState {
    received: usize,
    error: Option<String>,
    gpu_start_secs: f64,
    gpu_end_secs: f64,
    /// Profile mode only: (start, end) per command buffer, by submission index.
    spans: Vec<(f64, f64)>,
}

impl Feedback {
    fn new(expected: usize, profile: Option<ProfileInfo>) -> Arc<Self> {
        let spans = if profile.is_some() { vec![(0.0, 0.0); expected] } else { Vec::new() };
        Arc::new(Self { expected, profile, state: Mutex::new(FeedbackState { spans, ..FeedbackState::default() }) })
    }

    fn handler(self: &Arc<Self>, fault: &Arc<Fault>, index: usize) -> FeedbackHandler {
        let feedback = self.clone();
        let fault = fault.clone();
        RcBlock::new(move |fb: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
            // SAFETY: Metal hands a valid feedback object for the callback.
            let fb = unsafe { fb.as_ref() };
            let error = fb.error().map(|e| describe_error(&e));
            let (start, end) = (fb.GPUStartTime(), fb.GPUEndTime());
            if let Some(message) = &error {
                fault.record(message.clone());
            }
            if let Ok(mut state) = feedback.state.lock() {
                if state.received == 0 || start < state.gpu_start_secs {
                    state.gpu_start_secs = start;
                }
                if end > state.gpu_end_secs {
                    state.gpu_end_secs = end;
                }
                if state.error.is_none() {
                    state.error = error;
                }
                if let Some(span) = state.spans.get_mut(index) {
                    *span = (start, end);
                }
                state.received += 1;
                if state.received == feedback.expected
                    && let Some(info) = &feedback.profile
                {
                    let kernels = info
                        .names
                        .iter()
                        .zip(&state.spans)
                        .map(|(name, (start, end))| profile::KernelSample { name, gpu_secs: end - start })
                        .collect();
                    profile::record(profile::PassProfile {
                        label: info.label,
                        span_secs: state.gpu_end_secs - state.gpu_start_secs,
                        kernels,
                    });
                }
            }
        })
    }

    /// Waits for every command buffer's feedback (it arrives shortly after
    /// the fence) and returns the recorded span.
    fn complete(&self) -> Result<FeedbackState> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            {
                let state = self.state.lock().map_err(|e| anyhow!("feedback lock poisoned: {e}"))?;
                if state.received >= self.expected {
                    return Ok(FeedbackState {
                        received: state.received,
                        error: state.error.clone(),
                        gpu_start_secs: state.gpu_start_secs,
                        gpu_end_secs: state.gpu_end_secs,
                        spans: Vec::new(),
                    });
                }
            }
            ensure!(Instant::now() < deadline, "commit feedback did not arrive");
            std::thread::sleep(Duration::from_micros(50));
        }
    }
}

/// One element of a pass's submission program.
enum Item {
    Cmd(Retained<ProtocolObject<dyn MTL4CommandBuffer>>),
    Wait(SharedEvent, u64),
    Signal(SharedEvent, u64),
}

struct OpenSegment {
    cmd: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
}

pub struct ComputePass<'a> {
    ctx: &'a MetalContext,
    /// Closed command buffers and the queue operations between them.
    program: RefCell<Vec<Item>>,
    /// The command buffer being encoded, opened lazily by the first command
    /// after a queue operation.
    open: RefCell<Option<OpenSegment>>,
    concurrent: bool,
    /// The event and value [`Self::signal_done`] encoded, if any.
    done: RefCell<Option<(SharedEvent, u64)>>,
    /// Name for the per-kernel profile ([`Self::set_label`]).
    label: Cell<&'static str>,
    /// Profile mode only: the kernel of each closed command buffer, in
    /// program order (one dispatch per command buffer).
    names: RefCell<Vec<&'static str>>,
    /// Profile mode only: the kernel dispatched into the open segment.
    segment_kernel: Cell<Option<&'static str>>,
    // Declared last: command buffers must drop before their allocator is
    // recycled.
    slot: RefCell<Slot>,
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        // An error path (`?` between dispatches) leaves a segment open; close
        // it so the encoder and command buffer are released cleanly. Nothing
        // is committed.
        if let Some(seg) = self.open.borrow_mut().take() {
            seg.encoder.endEncoding();
            seg.cmd.endCommandBuffer();
        }
    }
}

/// One inline kernel argument (a `constant T&` in MSL) of a dispatch.
#[derive(Clone, Copy)]
pub enum Param<'p> {
    /// A host constant, copied into the pass's params arena.
    Bytes(&'p [u8]),
    /// A host `uint` constant.
    U32(u32),
    /// A host `float` constant.
    F32(f32),
    /// The word at this byte offset of a resident buffer, read by the GPU
    /// when the dispatch runs (so a kernel earlier in the pass may set it).
    Gpu(&'p GpuBuffer, usize),
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
    /// Runs `f` on the open segment's encoder and argument table, opening a
    /// command buffer first if none is.
    fn with_encoder<R>(
        &self,
        f: impl FnOnce(&ProtocolObject<dyn MTL4ComputeCommandEncoder>, &mut SlotInner, &Pool) -> Result<R>,
    ) -> Result<R> {
        let mut open = self.open.borrow_mut();
        let mut slot = self.slot.borrow_mut();
        let inner = slot.get();
        if open.is_none() {
            let cmd = self
                .ctx
                .device
                .newCommandBuffer()
                .ok_or_else(|| anyhow!("failed to create a Metal 4 command buffer"))?;
            cmd.beginCommandBufferWithAllocator(&inner.allocator);
            let encoder = cmd
                .computeCommandEncoder()
                .ok_or_else(|| anyhow!("failed to create a Metal 4 compute encoder"))?;
            // Consume everything earlier submissions wrote: Metal 4 does not
            // order command buffers' memory effects on its own.
            encoder.barrierAfterQueueStages_beforeStages_visibilityOptions(
                compute_stages(),
                compute_stages(),
                MTL4VisibilityOptions::Device,
            );
            encoder.setArgumentTable(Some(&inner.table));
            *open = Some(OpenSegment { cmd, encoder });
        }
        let seg = open.as_ref().expect("segment open");
        f(&seg.encoder, inner, &self.ctx.pool)
    }

    /// Closes the open command buffer (if any) into the program.
    fn close_segment(&self) {
        if let Some(seg) = self.open.borrow_mut().take() {
            seg.encoder.endEncoding();
            seg.cmd.endCommandBuffer();
            self.program.borrow_mut().push(Item::Cmd(seg.cmd));
            if self.ctx.profile {
                self.names.borrow_mut().push(self.segment_kernel.take().unwrap_or("(no dispatch)"));
            }
        }
    }

    /// Profile mode: the open segment holds exactly the dispatch `name`;
    /// close it so the command buffer's GPU span is that dispatch's.
    fn close_profiled_dispatch(&self, name: &'static str) {
        if self.ctx.profile {
            self.segment_kernel.set(Some(name));
            self.close_segment();
        }
    }

    /// Names the pass in the per-kernel profile ([`profile`]); no effect on
    /// what it executes.
    pub fn set_label(&self, label: &'static str) {
        self.label.set(label);
    }

    /// Encodes one kernel dispatch with a byte offset per buffer binding — used
    /// to bind slices of a larger buffer (e.g. the q/k/v thirds of the fused
    /// GDN projection). Offsets must be 4-byte aligned per Metal's rules.
    /// Every param is a host constant; see [`Self::dispatch_with`] for params
    /// the GPU supplies.
    pub fn dispatch_at(
        &self,
        kernel: &Kernel,
        buffers: &[(&GpuBuffer, usize)],
        params: &[&[u8]],
        grid: Grid,
    ) -> Result<()> {
        ensure!(
            params.len() <= MAX_KERNEL_PARAMS,
            "{} kernel params exceeds MAX_KERNEL_PARAMS ({MAX_KERNEL_PARAMS})",
            params.len()
        );
        let mut args = [Param::U32(0); MAX_KERNEL_PARAMS];
        for (arg, bytes) in args.iter_mut().zip(params) {
            *arg = Param::Bytes(bytes);
        }
        self.dispatch_with(kernel, buffers, &args[..params.len()], grid)
    }

    /// Like [`Self::dispatch_at`], with each param either a host constant or
    /// the address of a word in a resident buffer, read when the dispatch
    /// runs: a kernel earlier in the pass (ordered by the pass's barriers) may
    /// have written it. Params bind at indices `buffers.len()..`.
    pub fn dispatch_with(
        &self,
        kernel: &Kernel,
        buffers: &[(&GpuBuffer, usize)],
        params: &[Param<'_>],
        grid: Grid,
    ) -> Result<()> {
        ensure!(
            params.len() <= MAX_KERNEL_PARAMS,
            "{} kernel params exceeds MAX_KERNEL_PARAMS ({MAX_KERNEL_PARAMS})",
            params.len()
        );
        ensure!(
            buffers.len() + params.len() <= MAX_BUFFER_BINDINGS,
            "{} buffer bindings exceeds Metal's limit of {MAX_BUFFER_BINDINGS}",
            buffers.len() + params.len()
        );
        let serial = !self.concurrent;
        self.with_encoder(|encoder, inner, pool| {
            encoder.setComputePipelineState(&kernel.pipeline);
            for (i, (buf, offset)) in buffers.iter().enumerate() {
                // SAFETY: the address lies within a resident buffer.
                unsafe { inner.table.setAddress_atIndex(buf.address() + *offset as u64, i) };
            }
            for (i, param) in params.iter().enumerate() {
                let address = match param {
                    Param::Bytes(bytes) => {
                        ensure!(!bytes.is_empty(), "empty kernel param");
                        inner.arena.push(pool, bytes)?
                    }
                    Param::U32(v) => inner.arena.push(pool, &v.to_ne_bytes())?,
                    Param::F32(v) => inner.arena.push(pool, &v.to_ne_bytes())?,
                    Param::Gpu(buf, offset) => {
                        ensure!(offset.is_multiple_of(4) && *offset + 4 <= buf.length(), "GPU param at byte {offset} is not a word of its buffer");
                        buf.address() + *offset as u64
                    }
                };
                // SAFETY: the address lies within a resident arena chunk or buffer.
                unsafe { inner.table.setAddress_atIndex(address, buffers.len() + i) };
            }
            let size = |(w, h, d): (usize, usize, usize)| MTLSize { width: w, height: h, depth: d };
            match grid {
                Grid::Threads { grid, threadgroup } => {
                    encoder.dispatchThreads_threadsPerThreadgroup(size(grid), size(threadgroup));
                }
                Grid::Threadgroups { groups, threadgroup } => {
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threadgroup));
                }
            }
            if serial {
                encoder_barrier(encoder);
            }
            Ok(())
        })?;
        self.close_profiled_dispatch(kernel.name);
        Ok(())
    }

    /// Copies `len` bytes between buffers (any alignment), ordered like a
    /// dispatch of this pass.
    pub fn copy_buffer(&self, src: &GpuBuffer, src_offset: usize, dst: &GpuBuffer, dst_offset: usize, len: usize) -> Result<()> {
        let serial = !self.concurrent;
        self.with_encoder(|encoder, _inner, _pool| {
            // SAFETY: ranges validated by the caller against the buffer sizes.
            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src, src_offset, dst, dst_offset, len,
                );
            }
            if serial {
                encoder_barrier(encoder);
            }
            Ok(())
        })?;
        self.close_profiled_dispatch("copy_buffer");
        Ok(())
    }

    /// Orders one dependency level before the next on a concurrent encoder.
    pub fn level_barrier(&self, _written: &[&Tensor]) -> Result<()> {
        self.memory_barrier()
    }

    /// Makes the GPU block here until the host has signaled `event` to at
    /// least `value`. Everything encoded before completes first (the wait
    /// sits between two command buffers), so this also acts as a full
    /// barrier. The pass can then be committed before the inputs read after
    /// this point exist; the host writes them and signals. A pass committed
    /// with an unsatisfied wait holds the whole queue, so every path after
    /// commit must signal (or [`SharedEvent::release_all`]).
    pub fn wait_event(&self, event: &SharedEvent, value: u64) -> Result<()> {
        self.close_segment();
        self.program.borrow_mut().push(Item::Wait(event.clone(), value));
        Ok(())
    }

    /// Makes the GPU raise `event` to `value` once everything encoded before
    /// this point has completed (the host can wait on the event, or poll
    /// [`SharedEvent::signaled_value`], instead of on the whole pass).
    pub fn signal_event(&self, event: &SharedEvent, value: u64) -> Result<()> {
        self.close_segment();
        self.program.borrow_mut().push(Item::Signal(event.clone(), value));
        Ok(())
    }

    /// Orders all prior dispatches' buffer writes before every subsequent
    /// dispatch — the dependency-level boundary for [`MetalContext::
    /// begin_concurrent`] passes. Serial passes already order every dispatch.
    pub fn memory_barrier(&self) -> Result<()> {
        // In profile mode every dispatch sits in its own command buffer, each
        // opening with a queue-stage barrier that already orders it after all
        // earlier work; an encoder barrier here would only open an otherwise
        // empty command buffer.
        if self.concurrent && !self.ctx.profile {
            self.with_encoder(|encoder, _inner, _pool| {
                encoder_barrier(encoder);
                Ok(())
            })?;
        }
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
        self.close_segment();
        let program = std::mem::take(&mut *self.program.borrow_mut());
        let done = self.done.borrow().clone();
        let slot = self.slot.replace(Slot { inner: None, pool: self.ctx.pool.clone() });
        let names = std::mem::take(&mut *self.names.borrow_mut());
        Ok(EncodedPass { ctx: self.ctx, program, done, slot, label: self.label.get(), names })
    }

    /// Encodes, as the pass's last command, a GPU signal of `event` to
    /// `value`, and remembers it: [`PendingPass::wait_paced`] then polls the
    /// event (which the GPU writes directly) instead of blocking on the
    /// fence. Call after everything else is encoded.
    pub fn signal_done(&self, event: &SharedEvent, value: u64) -> Result<()> {
        self.signal_event(event, value)?;
        *self.done.borrow_mut() = Some((event.clone(), value));
        Ok(())
    }

    /// Ends encoding, submits, and blocks until the GPU finishes.
    pub fn commit_wait(self) -> Result<()> {
        self.commit()?.wait()
    }
}

/// A full barrier between dispatches of one encoder, caches flushed.
fn encoder_barrier(encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>) {
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        compute_stages(),
        compute_stages(),
        MTL4VisibilityOptions::Device,
    );
}

/// A fully encoded, not yet submitted pass.
pub struct EncodedPass<'a> {
    ctx: &'a MetalContext,
    program: Vec<Item>,
    done: Option<(SharedEvent, u64)>,
    slot: Slot,
    label: &'static str,
    /// Profile mode only: one kernel name per `Item::Cmd`.
    names: Vec<&'static str>,
}

impl<'a> EncodedPass<'a> {
    /// Submits without blocking.
    pub fn commit(self) -> Result<PendingPass<'a>> {
        let EncodedPass { ctx, program, done, slot, label, names } = self;
        let cmds = program.iter().filter(|item| matches!(item, Item::Cmd(_))).count();
        let profile = if ctx.profile {
            ensure!(names.len() == cmds, "profile: {} kernel names for {cmds} command buffers", names.len());
            Some(ProfileInfo { label, names })
        } else {
            None
        };
        let feedback = Feedback::new(cmds, profile);
        let fence_value = ctx.submit(&program, &feedback)?;
        Ok(PendingPass { ctx, program, fence_value, done, feedback, slot: Some(slot) })
    }

    /// Drops the context lifetime so the pass can be kept inside long-lived
    /// state (passes encoded ahead for several possible outcomes). The
    /// context must outlive it; [`DetachedPass::attach`] restores the tie.
    pub fn detach(self) -> DetachedPass {
        DetachedPass { program: self.program, done: self.done, slot: self.slot, label: self.label, names: self.names }
    }
}

/// An [`EncodedPass`] held without its context lifetime; see
/// [`EncodedPass::detach`]. Dropping it un-committed returns its command
/// memory to the pool.
pub struct DetachedPass {
    program: Vec<Item>,
    done: Option<(SharedEvent, u64)>,
    slot: Slot,
    label: &'static str,
    names: Vec<&'static str>,
}

impl DetachedPass {
    pub fn attach<'a>(self, ctx: &'a MetalContext) -> EncodedPass<'a> {
        EncodedPass { ctx, program: self.program, done: self.done, slot: self.slot, label: self.label, names: self.names }
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

/// A committed-but-unawaited pass. Holding one while encoding the next pass
/// is the decode pipelining primitive; hosts must not read buffers the
/// pending pass writes until [`Self::wait`] returns. Dropping it without
/// waiting blocks until the GPU is done (its command memory is recycled).
pub struct PendingPass<'a> {
    ctx: &'a MetalContext,
    /// Kept alive until completion.
    program: Vec<Item>,
    fence_value: u64,
    done: Option<(SharedEvent, u64)>,
    feedback: Arc<Feedback>,
    slot: Option<Slot>,
}

impl Drop for PendingPass<'_> {
    fn drop(&mut self) {
        if self.slot.is_some() {
            let _ = self.ctx.wait_fence(self.fence_value);
            self.program.clear();
            self.slot = None;
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PassTiming {
    pub gpu_start_secs: f64,
    pub gpu_end_secs: f64,
}

pub struct CompletedPass {
    feedback: Arc<Feedback>,
}

impl CompletedPass {
    /// The pass's GPU time span, from commit feedback (which arrives shortly
    /// after the pass's fence signal; this waits for it).
    pub fn timing(&self) -> Result<PassTiming> {
        let state = self.feedback.complete()?;
        if let Some(error) = state.error {
            bail!("Metal command buffer failed: {error}");
        }
        let timing = PassTiming { gpu_start_secs: state.gpu_start_secs, gpu_end_secs: state.gpu_end_secs };
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

    /// Blocks until completion while retaining the pass's feedback.
    /// Benchmarks query its GPU clock only after their cadence timer ends.
    pub fn wait_retain(mut self) -> Result<CompletedPass> {
        self.ctx.wait_fence(self.fence_value)?;
        self.finished()
    }

    /// Like [`Self::wait`], paced: sleeps until `pacer` predicts the pass is
    /// about to finish, then polls the pass's done signal (see
    /// [`ComputePass::signal_done`], falling back to the fence) and returns
    /// within microseconds of it. A blocked thread would learn of the
    /// completion later. Polling is capped so a wrong prediction cannot pin
    /// a core; the pacer learns from the outcome.
    pub fn wait_paced(self, pacer: &mut Pacer) -> Result<()> {
        self.wait_retain_paced(pacer).map(|_| ())
    }

    pub fn wait_retain_paced(mut self, pacer: &mut Pacer) -> Result<CompletedPass> {
        let (event, value) = match &self.done {
            Some((event, value)) => (event.clone(), *value),
            None => (self.ctx.fence.clone(), self.fence_value),
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
                    self.ctx.wait_fence(self.fence_value)?;
                    break;
                }
                std::hint::spin_loop();
            }
        }
        pacer.end(Instant::now(), overslept);
        // The done event may lead the fence by the trailing signal; the
        // command memory is only safe to reuse after the fence.
        if self.done.is_some() {
            self.ctx.wait_fence(self.fence_value)?;
        }
        self.finished()
    }

    fn finished(&mut self) -> Result<CompletedPass> {
        // Completed: release the command memory to the pool.
        self.program.clear();
        self.slot = None;
        self.ctx.fault.check()?;
        if self.ctx.profile {
            // The pass's breakdown is recorded by its last feedback handler,
            // shortly after the fence; wait for it so `profile::take` after a
            // wait includes this pass.
            self.feedback.complete()?;
        }
        Ok(CompletedPass { feedback: self.feedback.clone() })
    }
}

/// Longest a paced wait polls before falling back to a blocking wait.
const SPIN_CAP: Duration = Duration::from_millis(20);

/// Predicts when a repeating GPU pass completes so the host can sleep until
/// shortly before and then poll ([`PendingPass::wait_paced`]). Tracks an
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

#[cfg(test)]
#[path = "../tests/unit/metal.rs"]
mod tests;
