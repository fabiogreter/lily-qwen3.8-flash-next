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
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use block2::RcBlock;
use core::ffi::c_void;
use core::ptr::NonNull;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_foundation::{NSArray, NSError, NSString};
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator,
    MTL4CommandBuffer, MTL4CommandEncoder, MTL4CommandQueue, MTL4CommandQueueError,
    MTL4CommitFeedback, MTL4CommitOptions, MTL4ComputeCommandEncoder,
    MTL4VisibilityOptions, MTLAllocation, MTLBuffer, MTLCompileOptions,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent,
    MTLLanguageVersion, MTLLibrary, MTLResidencySet, MTLResidencySetDescriptor,
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

    static LEVELS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    pub(super) fn count_level() {
        LEVELS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Dependency levels (barriers) encoded so far, process-wide: the
    /// difference across one pass's encoding is that pass's level count.
    pub fn levels() -> usize {
        LEVELS.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// lily's own GPU busy time, process-wide: what the dashboard's estimate of
/// lily's GPU power divides the machine's GPU power by (`serve::machine`,
/// docs/dashboard.md § 6). Every command buffer lily commits goes through
/// [`super::submit_items`] with a commit feedback handler, so every pass is
/// counted, whichever context it ran on (the model, the drafts, vision, the
/// expert cache).
///
/// **What is counted.** Each command buffer's GPU span (`GPUStartTime` to
/// `GPUEndTime`, wall time on the host clock, stalls inside the buffer
/// included). Command buffers of one pass and consecutive passes can overlap,
/// and their feedback arrives out of order, so nothing can be added as it
/// arrives: the handler that completes a pass merges the pass's own spans
/// into their union ([`pass_union`]) and adds that union to a small set of
/// recent intervals ([`Recent`]), counting only the time the set did not
/// cover yet. The set forgets intervals that ended more than
/// [`HORIZON_SECS`] before the newest end it saw, which keeps it small; a pass
/// whose union reaches back past that horizon (its feedback completed more
/// than that after an overlapping pass's) could be counted twice where they
/// overlap. Passes on one queue run one after the other with an overlap of a
/// command buffer at most, so that is not expected to happen.
///
/// **When it is counted.** A pass's time is booked when its last feedback
/// arrives, not as it runs: a long prefill pass lands whole in the window its
/// feedback completes in. A pass that never completes (a fault, a pass
/// dropped before all its command buffers were committed) is not counted.
/// Spans with no usable timestamps (not finite, not after 0, ending before
/// they start) are left out.
///
/// The cost on the feedback path (a vector per pass, a sort and a lock held
/// for a merge into a set of about a second of intervals, on Metal's
/// completion thread) is not measured.
pub mod busy {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// How far back from the newest end seen the set keeps intervals, in
    /// seconds.
    pub const HORIZON_SECS: f64 = 1.0;

    /// The busy time counted so far, in nanoseconds.
    static TOTAL_NS: AtomicU64 = AtomicU64::new(0);
    static RECENT: Mutex<Recent> = Mutex::new(Recent::new());

    /// The union of a pass's command-buffer spans `(start, end)` in seconds,
    /// sorted and disjoint (touching spans merged); spans without usable
    /// timestamps are dropped.
    pub fn pass_union(spans: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let mut valid: Vec<(f64, f64)> = spans
            .iter()
            .copied()
            .filter(|&(start, end)| {
                start.is_finite() && end.is_finite() && start > 0.0 && end > start
            })
            .collect();
        valid.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut union: Vec<(f64, f64)> = Vec::with_capacity(valid.len());
        for (start, end) in valid {
            match union.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => union.push((start, end)),
            }
        }
        union
    }

    /// The intervals counted within the last [`HORIZON_SECS`], sorted and
    /// disjoint, and the newest end seen.
    #[derive(Debug, Default)]
    pub struct Recent {
        intervals: Vec<(f64, f64)>,
        newest_end: f64,
    }

    impl Recent {
        pub const fn new() -> Self {
            Self { intervals: Vec::new(), newest_end: 0.0 }
        }

        /// Adds a pass's union (from [`pass_union`]: sorted, disjoint,
        /// valid) and returns the seconds of it the set did not cover yet;
        /// then forgets what ended more than [`HORIZON_SECS`] before the
        /// newest end.
        pub fn add(&mut self, union: &[(f64, f64)]) -> f64 {
            let mut added = 0.0;
            for &(start, end) in union {
                let mut covered = 0.0;
                let (mut lo, mut hi) = (start, end);
                self.intervals.retain(|&(a, b)| {
                    if b < start || a > end {
                        return true;
                    }
                    covered += b.min(end) - a.max(start);
                    lo = lo.min(a);
                    hi = hi.max(b);
                    false
                });
                added += (end - start - covered).max(0.0);
                let at = self.intervals.partition_point(|&(a, _)| a < lo);
                self.intervals.insert(at, (lo, hi));
                self.newest_end = self.newest_end.max(end);
            }
            let horizon = self.newest_end - HORIZON_SECS;
            self.intervals.retain(|&(_, b)| b >= horizon);
            added
        }

        /// The intervals the set holds (tests).
        pub fn intervals(&self) -> &[(f64, f64)] {
            &self.intervals
        }
    }

    /// Counts a completed pass's command-buffer spans.
    pub(super) fn record(spans: &[(f64, f64)]) {
        let union = pass_union(spans);
        if union.is_empty() {
            return;
        }
        let added = RECENT.lock().unwrap_or_else(|e| e.into_inner()).add(&union);
        let ns = (added * 1e9).round();
        if ns >= 1.0 {
            TOTAL_NS.fetch_add(ns as u64, Ordering::Relaxed);
        }
    }

    /// lily's GPU busy time since the process started, in nanoseconds.
    pub fn total_ns() -> u64 {
        TOTAL_NS.load(Ordering::Relaxed)
    }
}

/// lily's GPU busy time since the process started, in nanoseconds; see
/// [`busy`].
pub fn gpu_busy_ns() -> u64 {
    busy::total_ns()
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
    /// Serializes submissions so fence values are signaled in order (shared
    /// with a gated pass's feeder thread).
    submit: Arc<Mutex<()>>,
    /// Set while a gated pass still has parts to submit: later submissions
    /// wait so the queue keeps program order ([`ComputePass::gate_host_waits`]).
    feeding: Arc<Feeding>,
    /// How long a gated pass's feeder waits at one gate before it faults
    /// the context ([`GATE_PATIENCE`]; tests shorten it).
    gate_patience: Duration,
    /// Tests: the next gated passes find no feeder thread.
    #[cfg(test)]
    fail_feeder_spawn: Cell<bool>,
    /// First GPU error the commit feedback reported, surfaced by the next
    /// wait or commit.
    fault: Arc<Fault>,
    /// Keyed by function name and MSL version.
    pipelines: Mutex<HashMap<(&'static str, MslVersion), Pipeline>>,
    /// Pipelines built so far and the nanoseconds it took (library compile
    /// plus pipeline creation), for the warm-up log line.
    built: AtomicU64,
    build_nanos: AtomicU64,
    /// Per-kernel profile mode; see [`profile`].
    profile: bool,
    /// The buffers [`Self::new_buffer`] allocated while a
    /// [`BufferRecording`] is open.
    recorded: RefCell<Option<Vec<Weak<GpuBuffer>>>>,
}

impl MetalContext {
    /// A context on the system default device; the per-kernel profile mode
    /// is on when `LILY_KERNEL_PROFILE` is set to anything but `0`.
    pub fn new() -> Result<Self> {
        let profile = std::env::var_os("LILY_KERNEL_PROFILE")
            .is_some_and(|v| !v.is_empty() && v != "0");
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
        let fence = SharedEvent::new(&device, false)?;
        let pool = Rc::new(Pool {
            device: device.clone(),
            residency: residency.clone(),
            free: RefCell::new(Vec::new()),
        });
        let ctx = Self {
            device,
            queue,
            residency,
            pool,
            fence,
            fence_value: AtomicU64::new(0),
            submit: Arc::new(Mutex::new(())),
            feeding: Arc::default(),
            gate_patience: GATE_PATIENCE,
            #[cfg(test)]
            fail_feeder_spawn: Cell::new(false),
            fault: Arc::new(Fault::default()),
            pipelines: Mutex::new(HashMap::new()),
            built: AtomicU64::new(0),
            build_nanos: AtomicU64::new(0),
            profile,
            recorded: RefCell::new(None),
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

        let started = Instant::now();
        let library = self
            .compile_library(source, version)
            .with_context(|| format!("for {fn_name}"))?;
        let pipeline = self.pipeline_from_library(&library, fn_name)?;
        self.built.fetch_add(1, Ordering::Relaxed);
        self.build_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        cache.insert((fn_name, version), pipeline.clone());
        Ok(Kernel { pipeline, name: fn_name })
    }

    /// Pipelines [`Self::pipeline`] has built on this context and the
    /// seconds that took. Metal keeps compiled libraries in a per-user
    /// cache across processes and in memory within one, so after the first
    /// run a build costs well under a millisecond; a cold one (a changed
    /// source, a new OS) up to about 150 ms per source file.
    pub fn build_stats(&self) -> (u64, f64) {
        (
            self.built.load(Ordering::Relaxed),
            self.build_nanos.load(Ordering::Relaxed) as f64 * 1e-9,
        )
    }

    /// Allocates a zero-initialized shared-storage buffer and makes it
    /// resident for the queue.
    pub fn new_buffer(&self, len: usize) -> Result<Buffer> {
        let buf = self.residency.new_buffer(&self.device, len)?;
        if let Some(recorded) = self.recorded.borrow_mut().as_mut() {
            recorded.push(Rc::downgrade(&buf));
        }
        Ok(buf)
    }

    /// Starts recording the buffers [`Self::new_buffer`] allocates (and so
    /// [`Self::new_buffer_with_bytes`] and every tensor constructor), until
    /// [`BufferRecording::finish`] returns the ones still alive. Mapped
    /// buffers are not recorded (their pages belong to a file), nor are the
    /// pool's own arenas. The model load records its weights this way, for
    /// the server's `--pin-weights`. One recording at a time.
    pub fn record_buffers(&self) -> Result<BufferRecording<'_>> {
        let mut recorded = self.recorded.borrow_mut();
        ensure!(recorded.is_none(), "a buffer recording is already open");
        *recorded = Some(Vec::new());
        Ok(BufferRecording { ctx: self })
    }

    /// A read-only buffer over `len` bytes of `path` at `offset`, mapped
    /// from the file instead of copied: the pages are the file's page-cache
    /// pages, which the OS can drop under memory pressure and read back on
    /// demand (weights served from disk on machines the model does not fit).
    /// Returns the buffer, which starts at the page boundary at or below
    /// `offset`, and the byte offset of the requested range inside it.
    pub fn new_buffer_mapped(
        &self,
        path: &std::path::Path,
        offset: u64,
        len: usize,
    ) -> Result<(Buffer, usize)> {
        self.residency.new_buffer_mapped(&self.device, path, offset, len)
    }

    /// A shared buffer over `len` bytes of host memory at `ptr`, not copied
    /// and not freed by Metal: the caller keeps the memory, so it can
    /// outlive the buffer and this context (and be written by threads that
    /// never see either). Not recorded ([`Self::record_buffers`]).
    ///
    /// # Safety
    ///
    /// `ptr` is page-aligned, `len` a nonzero multiple of the page size, and
    /// the memory stays allocated until the buffer is dropped and every pass
    /// that used it has completed.
    pub unsafe fn new_buffer_no_copy(
        &self,
        ptr: NonNull<c_void>,
        len: usize,
    ) -> Result<Buffer> {
        // SAFETY: as documented.
        unsafe { self.residency.new_buffer_no_copy(&self.device, ptr, len) }
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
            milestone: RefCell::new(None),
            gated: Cell::new(false),
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
        SharedEvent::new(&self.device, false)
    }

    /// Like [`Self::new_shared_event`], for an event whose waits can block
    /// long enough to time out (the expert cache's `ready`): its host
    /// signals are logged, and every pass's commit feedback checks that the
    /// work behind each wait on it held until the host's signal ([`Watch`]).
    /// Waits on other events are submitted and checked as before.
    pub fn new_watched_event(&self) -> Result<SharedEvent> {
        SharedEvent::new(&self.device, true)
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

    /// Starts the GPU's return from idle without waiting for it: submits a
    /// queue-level signal of the fence and nothing else, so no command memory
    /// is held and nobody has to wait for it. After about 1.5 s without work
    /// the first submission waits for the queue's residency set to become
    /// resident again, about 6 ms per GB (0.4 to 0.6 s for the full model,
    /// `mlock` or not; `tests/unit/metal.rs::idle_first_submission_probe`);
    /// a wake issued while the host is still preparing a pass overlaps that
    /// with the host work, and costs about 0.03 ms when the GPU is awake.
    pub fn wake(&self) -> Result<()> {
        self.submit(&[], &Feedback::new(0, None, Vec::new())).map(|_| ())
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
            pass.copy_buffer(
                src,
                src_base + copy.src_offset,
                dst,
                dst_base + copy.dst_offset,
                copy.len,
            )?;
        }
        pass.commit_wait()
    }

    /// Submits a pass's program in order and signals the fence after it.
    /// Returns the fence value to wait for.
    ///
    /// A queue-level wait ([`ComputePass::wait_event`]) that blocks too long
    /// is released by Metal: on an `MTL4CommandQueue` a wait that times out
    /// "unblocks subsequent work in the queue", silently, and the limit is
    /// not documented. Measured (`tests/unit/metal.rs`,
    /// `queue_wait_timeout_probe`): a wait still blocked about 4.0 s after
    /// the submission that encoded it is released, and so is every wait of
    /// that submission after it, while a submission made later gets its own
    /// 4 s. Probes whose waits each blocked for less than about 0.1 s held
    /// for 8 s, but in a long prefill under the expert cache waits of 75 to
    /// 130 ms were released, so nothing relies on that. No wait may block
    /// for seconds after its submission: a pass whose waits can add up to
    /// that is gated ([`ComputePass::gate_host_waits`]), and every pass's
    /// commit feedback checks that the work behind each wait on a watched
    /// event ([`Self::new_watched_event`]) started only after the host
    /// signalled it ([`Watch`]); a wait that did not is recorded as a fault.
    /// The other waits (a parked pass's [`ComputePass::wait_event`] on its
    /// first input) block for a pass and the host's per-token work, and are
    /// left unwatched as before.
    fn submit(&self, program: &[Item], feedback: &Arc<Feedback>) -> Result<u64> {
        // A feeder that stopped at a fault ends its feeding: check after.
        self.feeding.wait_idle(&self.fault)?;
        self.fault.check()?;
        let _guard =
            self.submit.lock().map_err(|e| anyhow!("submit lock poisoned: {e}"))?;
        self.residency.flush();
        submit_items(&self.queue, program, feedback, &self.fault, 0);
        let value = self.fence_value.fetch_add(1, Ordering::AcqRel) + 1;
        self.queue.signalEvent_value(self.fence.as_event(), value);
        Ok(value)
    }

    /// Submits a gated pass ([`ComputePass::gate_host_waits`]): `program` up
    /// to its first wait now, the rest from a feeder thread, each part that
    /// begins at a wait only once the host has signalled that wait's event
    /// to its value. Returns the fence value the last part signals, the
    /// items submitted so far, and the feeder (which returns the others).
    /// When no feeder thread can be started this thread feeds the pass
    /// before returning (the waits' events are signalled by another thread,
    /// [`ComputePass::gate_host_waits`]), and there is no feeder.
    fn submit_gated(
        &self,
        mut program: Vec<Item>,
        feedback: &Arc<Feedback>,
    ) -> Result<(u64, Vec<Item>, Option<FeederHandle>)> {
        self.feeding.wait_idle(&self.fault)?;
        self.fault.check()?;
        let first = program
            .iter()
            .position(|item| matches!(item, Item::Wait(..)))
            .ok_or_else(|| anyhow!("a gated pass without a wait"))?;
        let rest = program.split_off(first);
        let (next_cmd, value) = {
            let _guard =
                self.submit.lock().map_err(|e| anyhow!("submit lock poisoned: {e}"))?;
            self.residency.flush();
            let next_cmd =
                submit_items(&self.queue, &program, feedback, &self.fault, 0);
            // Reserved now and signalled by the last part: every later
            // submission waits for the feeder, so fence values still reach
            // the queue in order.
            let value = self.fence_value.fetch_add(1, Ordering::AcqRel) + 1;
            self.feeding.begin();
            (next_cmd, value)
        };
        let feeder = Feeder {
            queue: SendQueue(self.queue.clone()),
            submit: self.submit.clone(),
            feeding: self.feeding.clone(),
            fault: self.fault.clone(),
            fence: self.fence.clone(),
            fence_value: value,
            feedback: feedback.clone(),
            next_cmd,
            items: SendItems(rest),
            patience: self.gate_patience,
        };
        // The feeder reaches its thread through a channel, not the closure,
        // so it is still here when the thread cannot be started.
        let (tx, rx) = std::sync::mpsc::channel::<Feeder>();
        let feeder = match self.spawn_feeder(rx) {
            Ok(handle) => match tx.send(feeder) {
                Ok(()) => return Ok((value, program, Some(FeederHandle(handle)))),
                Err(std::sync::mpsc::SendError(feeder)) => feeder,
            },
            Err(e) => {
                eprintln!(
                    "metal: could not start a gated pass's feeder ({e}); feeding it from the committing thread"
                );
                feeder
            }
        };
        // Errors are the context's fault now, which the pass's wait reports;
        // the items submitted must be kept until then either way.
        let (items, _) = feeder.run();
        program.extend(items.0);
        Ok((value, program, None))
    }

    /// Starts a gated pass's feeder thread, which takes its [`Feeder`] from
    /// `rx`.
    fn spawn_feeder(
        &self,
        rx: std::sync::mpsc::Receiver<Feeder>,
    ) -> std::io::Result<std::thread::JoinHandle<FeederResult>> {
        #[cfg(test)]
        if self.fail_feeder_spawn.get() {
            return Err(std::io::Error::other("injected for the test"));
        }
        std::thread::Builder::new()
            .name("lily-gated-pass".into())
            .spawn(move || rx.recv().ok().map(Feeder::run))
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

/// An open [`MetalContext::record_buffers`]; dropping it without
/// [`Self::finish`] (an error path) ends the recording and forgets it.
pub struct BufferRecording<'a> {
    ctx: &'a MetalContext,
}

impl BufferRecording<'_> {
    /// Ends the recording and returns the recorded buffers that are still
    /// alive, in allocation order (temporaries dropped meanwhile are gone).
    pub fn finish(self) -> Vec<Buffer> {
        let recorded = self.ctx.recorded.borrow_mut().take().unwrap_or_default();
        recorded.iter().filter_map(Weak::upgrade).collect()
    }
}

impl Drop for BufferRecording<'_> {
    fn drop(&mut self) {
        self.ctx.recorded.borrow_mut().take();
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

    fn new_buffer(
        self: &Rc<Self>,
        device: &ProtocolObject<dyn MTLDevice>,
        len: usize,
    ) -> Result<Buffer> {
        let raw = device
            .newBufferWithLength_options(
                len.max(1),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| anyhow!("failed to allocate {len}-byte buffer"))?;
        // Metal does not guarantee new buffer contents; callers rely on zeros.
        unsafe { core::ptr::write_bytes(raw.contents().as_ptr().cast::<u8>(), 0, len) };
        let address = raw.gpuAddress();
        self.set.addAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&*raw));
        self.dirty.set(true);
        Ok(Rc::new(GpuBuffer { raw, address, residency: self.clone() }))
    }

    fn new_buffer_mapped(
        self: &Rc<Self>,
        device: &ProtocolObject<dyn MTLDevice>,
        path: &std::path::Path,
        offset: u64,
        len: usize,
    ) -> Result<(Buffer, usize)> {
        use std::os::fd::AsRawFd;
        ensure!(len > 0, "mapped buffer of zero bytes");
        // SAFETY: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let base = offset - offset % page;
        let inner = usize::try_from(offset - base)?;
        let map_len = (inner + len).next_multiple_of(page as usize);
        let file = std::fs::File::open(path)
            .with_context(|| format!("opening {} for mapping", path.display()))?;
        // SAFETY: a private read-only mapping of a file we opened; the
        // kernel validates the range.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                base as libc::off_t,
            )
        };
        ensure!(
            ptr != libc::MAP_FAILED,
            "mapping {} bytes of {} at {base}: {}",
            map_len,
            path.display(),
            std::io::Error::last_os_error()
        );
        let ptr = NonNull::new(ptr).expect("mmap returned null");
        // Metal unmaps the file when it releases the buffer.
        let unmap = RcBlock::new(|p: NonNull<c_void>, len: usize| {
            // SAFETY: the mapping handed to Metal, of this length.
            unsafe { libc::munmap(p.as_ptr(), len) };
        });
        // SAFETY: page-aligned mapping of map_len bytes; the deallocator
        // block outlives its use through Metal's copy of it.
        let raw = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                ptr,
                map_len,
                MTLResourceOptions::StorageModeShared,
                Some(&unmap),
            )
        };
        let Some(raw) = raw else {
            // SAFETY: the mapping we just made, no buffer holds it.
            unsafe { libc::munmap(ptr.as_ptr(), map_len) };
            bail!("Metal refused a {map_len}-byte mapped buffer of {}", path.display());
        };
        let address = raw.gpuAddress();
        self.set.addAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&*raw));
        self.dirty.set(true);
        Ok((Rc::new(GpuBuffer { raw, address, residency: self.clone() }), inner))
    }

    /// # Safety
    ///
    /// As [`MetalContext::new_buffer_no_copy`].
    unsafe fn new_buffer_no_copy(
        self: &Rc<Self>,
        device: &ProtocolObject<dyn MTLDevice>,
        ptr: NonNull<c_void>,
        len: usize,
    ) -> Result<Buffer> {
        // SAFETY: as documented; no deallocator, the memory is the caller's.
        let raw = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                ptr,
                len,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
        .ok_or_else(|| anyhow!("Metal refused a {len}-byte no-copy buffer"))?;
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
                let allocator = self.device.newCommandAllocator().ok_or_else(|| {
                    anyhow!("failed to create a Metal 4 command allocator")
                })?;
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
            let size =
                ARENA_CHUNK.max(len.next_power_of_two()) << self.chunks.len().min(8);
            let chunk = pool.residency.new_buffer(&pool.device, size)?;
            self.chunks.push(chunk);
        }
    }
}

/// A monotonically signaled counter shared by host and GPU
/// (`MTLSharedEvent`). The GPU waits for `signaled_value >= v` where a pass
/// asked for it; the host raises the value once the awaited inputs are in
/// place. Waits are released in order, so callers use increasing values.
///
/// A watched event's host signals ([`MetalContext::new_watched_event`]) are
/// logged with their time, so the commit feedback can tell a queue wait the
/// host released from one that timed out ([`MetalContext::submit`] explains
/// the timeout).
#[derive(Clone)]
pub struct SharedEvent {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// The host's signals, for a watched event.
    log: Option<Arc<Mutex<SignalLog>>>,
}

/// The last [`SignalLog::CAPACITY`] host signals that raised an event:
/// (value, host seconds on the [`host_secs`] clock, taken just before the
/// signal), values rising. A signal at or below the last one logged is not
/// logged: Metal ignores a lower value, and the event reached an equal one
/// at the earlier time. The log's lock is held across the signal itself, so
/// with several signalling threads the log's order is still the event's and
/// every logged time precedes its signal (lily has one host signaller per
/// event: the expert cache's service thread for `ready`, and error paths'
/// [`SharedEvent::release_all`] once it has stopped).
#[derive(Default)]
struct SignalLog {
    entries: std::collections::VecDeque<(u64, f64)>,
}

impl SignalLog {
    /// Far more signals than happen between a wait's command buffer
    /// completing and its feedback being checked (48 per decoded token).
    const CAPACITY: usize = 4096;

    fn push(&mut self, value: u64, at: f64) {
        if self.entries.back().is_some_and(|&(last, _)| value <= last) {
            return;
        }
        if self.entries.len() == Self::CAPACITY {
            self.entries.pop_front();
        }
        self.entries.push_back((value, at));
    }

    /// When the host first raised the event to `value` or above, if the log
    /// still holds that signal. `None` also when the oldest entry kept is
    /// already at or above `value` but an older one may have been the first:
    /// the time is then unknown.
    fn signalled_at(&self, value: u64) -> Option<f64> {
        let i = self.entries.partition_point(|&(v, _)| v < value);
        if i == 0 && self.entries.len() == Self::CAPACITY {
            return None;
        }
        self.entries.get(i).map(|&(_, at)| at)
    }
}

// SAFETY: MTLSharedEvent is documented as safe to signal and wait on from
// any thread (that is what a shared event is for); the handle holds nothing
// else.
unsafe impl Send for SharedEvent {}
unsafe impl Sync for SharedEvent {}

impl SharedEvent {
    fn new(device: &ProtocolObject<dyn MTLDevice>, watched: bool) -> Result<Self> {
        let event = device
            .newSharedEvent()
            .ok_or_else(|| anyhow!("failed to create shared event"))?;
        Ok(Self { event, log: watched.then(Arc::default) })
    }

    /// Raises the counter to `value` (never lower it: a pass may be waiting
    /// for the current value).
    pub fn signal(&self, value: u64) {
        match &self.log {
            Some(log) => {
                let mut log = log.lock().unwrap_or_else(|e| e.into_inner());
                log.push(value, host_secs());
                self.event.setSignaledValue(value);
            }
            None => self.event.setSignaledValue(value),
        }
    }

    /// Whether the event's waits are checked ([`MetalContext::new_watched_event`]).
    fn watched(&self) -> bool {
        self.log.is_some()
    }

    /// When the host first signalled `value` or above, while the log holds
    /// it; `None` for a value only the GPU signalled, one too long ago, or
    /// an event that is not watched.
    fn signalled_at(&self, value: u64) -> Option<f64> {
        let log = self.log.as_ref()?.lock().unwrap_or_else(|e| e.into_inner());
        log.signalled_at(value)
    }

    /// Tests: raises the counter without logging it, the way a queue wait
    /// that times out lets the GPU go on without the host's signal.
    #[cfg(test)]
    fn signal_unlogged(&self, value: u64) {
        self.event.setSignaledValue(value);
    }

    pub fn signaled_value(&self) -> u64 {
        self.event.signaledValue()
    }

    /// Blocks until the counter reaches `value` or `timeout_ms` passes;
    /// true when it did.
    pub fn wait_until(&self, value: u64, timeout_ms: u64) -> bool {
        self.event.waitUntilSignaledValue_timeoutMS(value, timeout_ms)
    }

    /// Releases every pending and future wait, for error paths that would
    /// otherwise leave a committed pass blocked on the GPU forever.
    pub fn release_all(&self) {
        self.signal(u64::MAX);
    }

    fn as_event(&self) -> &ProtocolObject<dyn MTLEvent> {
        ProtocolObject::from_ref(&*self.event)
    }
}

/// Commits `items` to `queue` in order (command buffers with a feedback
/// handler each, numbered from `index`, and the queue operations between
/// them). Returns the next command buffer's index. The caller holds the
/// context's submit lock.
fn submit_items(
    queue: &ProtocolObject<dyn MTL4CommandQueue>,
    items: &[Item],
    feedback: &Arc<Feedback>,
    fault: &Arc<Fault>,
    mut index: usize,
) -> usize {
    for item in items {
        match item {
            Item::Cmd(cmd) => {
                let options = MTL4CommitOptions::new();
                let handler = feedback.handler(fault, index);
                index += 1;
                // SAFETY: the block pointer is valid for the call; the
                // options object copies the block.
                unsafe { options.addFeedbackHandler(RcBlock::as_ptr(&handler)) };
                let mut ptr = NonNull::from(&**cmd);
                // SAFETY: one valid command buffer pointer, count 1.
                unsafe {
                    queue.commit_count_options(NonNull::from(&mut ptr), 1, &options)
                };
            }
            Item::Wait(event, value) => {
                queue.waitForEvent_value(event.as_event(), *value)
            }
            Item::Signal(event, value) => {
                queue.signalEvent_value(event.as_event(), *value)
            }
        }
    }
    index
}

/// Whether a gated pass's feeder is still submitting; see
/// [`MetalContext::submit_gated`].
#[derive(Default)]
struct Feeding {
    busy: Mutex<bool>,
    idle: std::sync::Condvar,
    /// Tests: per part a feeder submitted, the value of the wait it begins
    /// at and the host time just before its submission.
    #[cfg(test)]
    parts: Mutex<Vec<(u64, f64)>>,
}

impl Feeding {
    fn begin(&self) {
        *self.busy.lock().unwrap_or_else(|e| e.into_inner()) = true;
    }

    fn end(&self) {
        *self.busy.lock().unwrap_or_else(|e| e.into_inner()) = false;
        self.idle.notify_all();
    }

    /// Blocks until no feeder is submitting; a fault ends the wait (a
    /// feeder stops at one, but the check costs nothing).
    fn wait_idle(&self, fault: &Fault) -> Result<()> {
        let mut busy = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        while *busy {
            fault.check()?;
            busy = self
                .idle
                .wait_timeout(busy, Duration::from_millis(50))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        Ok(())
    }
}

/// The context's queue, moved to a feeder thread. SAFETY of the `Send`:
/// Metal command queues are thread-safe, and the feeder only commits and
/// encodes queue operations, under the context's submit lock.
struct SendQueue(Retained<ProtocolObject<dyn MTL4CommandQueue>>);
unsafe impl Send for SendQueue {}

/// A pass's remaining program, moved to its feeder thread and back. SAFETY
/// of the `Send`: the command buffers' encoding has ended, so all that is
/// left to do with them is to commit them, which Metal allows from any
/// thread, and to drop them; the events are `Send` already.
struct SendItems(Vec<Item>);
unsafe impl Send for SendItems {}

/// What a feeder thread returns: the items to keep until completion and how
/// feeding went (`None` if the thread never got its [`Feeder`]).
type FeederResult = Option<(SendItems, Result<()>)>;

/// The thread that submits a gated pass's parts; joined by the pass's wait
/// (or drop), which takes the items back to keep them until completion.
struct FeederHandle(std::thread::JoinHandle<FeederResult>);

/// How long a feeder polls a host event before it blocks on it: a hit-only
/// resolution of the expert cache signals within tens of microseconds, and
/// a blocked thread wakes some 10 to 20 microseconds late.
const GATE_SPIN: Duration = Duration::from_micros(200);

/// After this long at one gate a feeder gives up and faults the context: the
/// request fails and the engine is reloaded instead of hanging. Ungated, a
/// queue wait this long would have been released by Metal after 4 s
/// ([`MetalContext::submit`]). The longest legitimate wait is one layer's
/// misses in a prefill pass: on average 85 ms a layer for 4 096 rows at
/// 40 GB, 160 ms for 8 192 rows (7.7 s a pass); allowing the slowest layer
/// twice the average (not measured), 0.3 s, and 3 s with a drive ten times
/// slower. 30 s leaves ten times that.
const GATE_PATIENCE: Duration = Duration::from_secs(30);

/// A gated pass's feeder: the rest of the program, every part of which
/// begins at a wait, submitted part by part as the host signals.
struct Feeder {
    queue: SendQueue,
    submit: Arc<Mutex<()>>,
    feeding: Arc<Feeding>,
    fault: Arc<Fault>,
    fence: SharedEvent,
    fence_value: u64,
    feedback: Arc<Feedback>,
    next_cmd: usize,
    items: SendItems,
    patience: Duration,
}

/// Ends a feeder's [`Feeding`] however it stops. One that panics faults
/// the context first, as one that fails does: the rest of its pass was not
/// submitted, so the pass's fence and the host events it would have reached
/// never come, and a fault makes every wait on the context end with an
/// error instead.
struct EndFeeding {
    feeding: Arc<Feeding>,
    fault: Arc<Fault>,
}

impl Drop for EndFeeding {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.fault.record("a gated pass's feeder panicked".to_owned());
        }
        self.feeding.end();
    }
}

impl Feeder {
    /// Submits the rest of the pass; returns its items, to be kept until
    /// the pass completes, and how it went (a failure is the context's
    /// fault too).
    fn run(self) -> (SendItems, Result<()>) {
        let _end =
            EndFeeding { feeding: self.feeding.clone(), fault: self.fault.clone() };
        let result = self.feed();
        if let Err(e) = &result {
            self.fault.record(format!("a gated pass stopped feeding: {e:#}"));
        }
        (self.items, result)
    }

    fn feed(&self) -> Result<()> {
        let items = &self.items.0;
        let mut index = self.next_cmd;
        let mut start = 0;
        while start < items.len() {
            if let Item::Wait(event, value) = &items[start] {
                self.gate(event, *value)?;
            }
            let end = items[start + 1..]
                .iter()
                .position(|item| matches!(item, Item::Wait(..)))
                .map_or(items.len(), |p| start + 1 + p);
            self.fault.check()?;
            let _guard =
                self.submit.lock().map_err(|e| anyhow!("submit lock poisoned: {e}"))?;
            #[cfg(test)]
            if let Item::Wait(_, value) = &items[start] {
                self.feeding
                    .parts
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push((*value, host_secs()));
            }
            index = submit_items(
                &self.queue.0,
                &items[start..end],
                &self.feedback,
                &self.fault,
                index,
            );
            if end == items.len() {
                self.queue.0.signalEvent_value(self.fence.as_event(), self.fence_value);
            }
            start = end;
        }
        Ok(())
    }

    /// Blocks until the host has raised `event` to `value`; fails (with the
    /// context faulted) after the feeder's patience ([`GATE_PATIENCE`]).
    fn gate(&self, event: &SharedEvent, value: u64) -> Result<()> {
        let began = Instant::now();
        while event.signaled_value() < value {
            if began.elapsed() < GATE_SPIN {
                std::hint::spin_loop();
                continue;
            }
            while !event.wait_until(value, 50) {
                self.fault.check()?;
                if began.elapsed() > self.patience {
                    let message = format!(
                        "a gated pass gave up after {:.1} s waiting for a host event to reach {value}",
                        began.elapsed().as_secs_f64()
                    );
                    eprintln!("metal: {message}");
                    self.fault.record(message.clone());
                    bail!(message);
                }
            }
            break;
        }
        Ok(())
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
            MTL4CommandQueueError::Timeout => Some(
                "Timeout: the workload took longer to execute than the system allows",
            ),
            MTL4CommandQueueError::NotPermitted => {
                Some("NotPermitted: the process has no access to a GPU device")
            }
            MTL4CommandQueueError::OutOfMemory => Some(
                "OutOfMemory: the GPU lacks the memory to execute a command buffer",
            ),
            MTL4CommandQueueError::DeviceRemoved => Some(
                "DeviceRemoved: the GPU was removed before the command buffer completed",
            ),
            MTL4CommandQueueError::AccessRevoked => Some(
                "AccessRevoked: the system revoked GPU access after too many timeouts or hangs",
            ),
            MTL4CommandQueueError::Internal => {
                Some("Internal: a problem inside the Metal framework")
            }
            _ => None,
        };
        if let Some(meaning) = meaning {
            text.push_str(&format!(" ({meaning})"));
        }
    }
    text.push_str(&format!(": {}", error.localizedDescription()));
    let info = error.userInfo();
    let one_line = |e: &NSError| {
        format!("{} error {}: {}", e.domain(), e.code(), e.localizedDescription())
    };
    let mut underlying: Vec<String> = Vec::new();
    // The key strings are what `NSUnderlyingErrorKey` and
    // `NSMultipleUnderlyingErrorsKey` hold (reading the extern statics
    // needs `unsafe`; the literals do not).
    if let Some(object) = info.objectForKey(&NSString::from_str("NSUnderlyingError"))
        && let Some(e) = object.downcast_ref::<NSError>()
    {
        underlying.push(one_line(e));
    }
    if let Some(object) =
        info.objectForKey(&NSString::from_str("NSMultipleUnderlyingErrorsKey"))
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
/// their GPU time span, and any error. Also each command buffer's own span:
/// once the last handler has run (handlers may arrive out of order) their
/// union is counted as lily's GPU busy time ([`busy`]), and in profile mode
/// they are recorded as a [`profile::PassProfile`] as well.
struct Feedback {
    expected: usize,
    profile: Option<ProfileInfo>,
    /// The pass's waits on host events, by the command buffer behind each.
    watches: Vec<Watch>,
    state: Mutex<FeedbackState>,
}

/// A wait of a pass and the command buffer right behind it: that command
/// buffer's GPU start time tells whether the wait held until the host's
/// signal or timed out ([`MetalContext::submit`]).
struct Watch {
    /// Submission index of the command buffer after the wait.
    cmd: usize,
    event: SharedEvent,
    value: u64,
}

/// The waits of `program` on watched events
/// ([`MetalContext::new_watched_event`]) that a command buffer follows, in
/// order.
fn watches_of(program: &[Item]) -> Vec<Watch> {
    let mut watches = Vec::new();
    let mut waits: Vec<(SharedEvent, u64)> = Vec::new();
    let mut cmd = 0usize;
    for item in program {
        match item {
            Item::Wait(event, value) if event.watched() => {
                waits.push((event.clone(), *value))
            }
            Item::Wait(..) => {}
            Item::Cmd(_) => {
                watches.extend(waits.drain(..).map(|(event, value)| Watch {
                    cmd,
                    event,
                    value,
                }));
                cmd += 1;
            }
            Item::Signal(..) => {}
        }
    }
    watches
}

/// Why the work behind a wait for `value` cannot have waited for the host:
/// the event was still below `value` when that work had completed
/// (`now`, read after completion), or the GPU started it at `gpu_start`
/// more than [`EARLY_START_MARGIN_SECS`] before the host's signal
/// (`signalled_at`, taken just before the host raised the event, so a held
/// wait always starts after it). `None` when the wait held, or when the
/// host's signal time is no longer known.
fn wait_released_early(
    now: u64,
    value: u64,
    signalled_at: Option<f64>,
    gpu_start: f64,
) -> Option<String> {
    if now < value {
        return Some(format!(
            "the work behind a wait for {value} ran before the host signalled it (the event is at {now})"
        ));
    }
    match signalled_at {
        Some(at) if gpu_start > 0.0 && gpu_start < at - EARLY_START_MARGIN_SECS => {
            Some(format!(
                "the work behind a wait for {value} started {:.3} ms before the host signalled it",
                (at - gpu_start) * 1e3
            ))
        }
        _ => None,
    }
}

/// How much earlier than the host's signal the work behind a wait must have
/// started to count as released early. Both times are `mach_absolute_time`,
/// but the GPU's is converted from the GPU's own clock, and a false
/// positive costs a failed request and an engine reload. Work behind a held
/// wait started 5 microseconds to 3 ms after the signal, 0.1 ms in the
/// median (`waits_the_host_releases_are_not_reported_as_timed_out` prints
/// it): never before it, so the margin is only for disagreement between the
/// clocks. A timed-out wait is released seconds after its submission, so
/// this only spares a release that landed within 0.1 ms of the host's own
/// signal, which the service thread's check of `routed` still counts.
const EARLY_START_MARGIN_SECS: f64 = 100e-6;

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
    /// (start, end) per command buffer, by submission index: the pass's
    /// busy time ([`busy`]) and, in profile mode, its per-kernel profile.
    spans: Vec<(f64, f64)>,
    /// Watches checked so far.
    watched: usize,
}

impl Feedback {
    fn new(
        expected: usize,
        profile: Option<ProfileInfo>,
        watches: Vec<Watch>,
    ) -> Arc<Self> {
        // A span never reported stays (0, 0), which the busy count skips.
        let spans = vec![(0.0, 0.0); expected];
        Arc::new(Self {
            expected,
            profile,
            watches,
            state: Mutex::new(FeedbackState { spans, ..FeedbackState::default() }),
        })
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
            let watched = feedback.check_watches(index, start, &fault);
            if let Ok(mut state) = feedback.state.lock() {
                state.watched += watched;
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
                if state.received == feedback.expected {
                    busy::record(&state.spans);
                }
                if state.received == feedback.expected
                    && let Some(info) = &feedback.profile
                {
                    let kernels = info
                        .names
                        .iter()
                        .zip(&state.spans)
                        .map(|(name, (start, end))| profile::KernelSample {
                            name,
                            gpu_secs: end - start,
                        })
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

    /// Checks the watches on command buffer `index`, which the GPU started
    /// at `start`: a wait that did not hold is recorded as a fault (the
    /// pass ran on inputs that were not ready, and so may every pass after
    /// it that read what this one wrote). Returns how many were checked.
    fn check_watches(&self, index: usize, start: f64, fault: &Fault) -> usize {
        let from = self.watches.partition_point(|w| w.cmd < index);
        let mut checked = 0;
        for w in self.watches[from..].iter().take_while(|w| w.cmd == index) {
            checked += 1;
            let now = w.event.signaled_value();
            if let Some(why) =
                wait_released_early(now, w.value, w.event.signalled_at(w.value), start)
            {
                let message = format!(
                    "a GPU queue wait timed out: {why}, so the pass computed with inputs that were not ready \
                     (Metal 4 releases a queue wait that blocks too long)"
                );
                if fault.message().is_none() {
                    eprintln!("metal: {message}");
                }
                fault.record(message);
            }
        }
        checked
    }

    /// Blocks until every watch was checked: the feedback of the command
    /// buffer behind a pass's last wait usually arrived long before the
    /// pass's fence (the rest of the pass ran after it).
    fn watches_checked(&self) -> Result<()> {
        if self.watches.is_empty() {
            return Ok(());
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            {
                let state = self
                    .state
                    .lock()
                    .map_err(|e| anyhow!("feedback lock poisoned: {e}"))?;
                if state.watched >= self.watches.len() {
                    return Ok(());
                }
            }
            ensure!(Instant::now() < deadline, "commit feedback did not arrive");
            std::thread::yield_now();
        }
    }

    /// Waits for every command buffer's feedback (it arrives shortly after
    /// the fence) and returns the recorded span.
    fn complete(&self) -> Result<FeedbackState> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            {
                let state = self
                    .state
                    .lock()
                    .map_err(|e| anyhow!("feedback lock poisoned: {e}"))?;
                if state.received >= self.expected {
                    return Ok(FeedbackState {
                        received: state.received,
                        error: state.error.clone(),
                        gpu_start_secs: state.gpu_start_secs,
                        gpu_end_secs: state.gpu_end_secs,
                        spans: Vec::new(),
                        watched: state.watched,
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
    /// The last host-signalled wait of the pass ([`Self::set_milestone`]).
    milestone: RefCell<Option<(SharedEvent, u64)>>,
    /// Submitted in parts at its waits ([`Self::gate_host_waits`]).
    gated: Cell<bool>,
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
        f: impl FnOnce(
            &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
            &mut SlotInner,
            &Pool,
        ) -> Result<R>,
    ) -> Result<R> {
        let mut open = self.open.borrow_mut();
        let mut slot = self.slot.borrow_mut();
        let inner = slot.get();
        if open.is_none() {
            let cmd =
                self.ctx.device.newCommandBuffer().ok_or_else(|| {
                    anyhow!("failed to create a Metal 4 command buffer")
                })?;
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

    /// Puts an empty command buffer right behind the pass's last wait on a
    /// watched event ([`MetalContext::new_watched_event`]; a pass without
    /// one, as every pass without the expert cache, is left as it was): its
    /// commit feedback, whose GPU start time shows whether the wait held
    /// ([`Watch`]), then arrives as soon as the wait is passed instead of
    /// after the command buffer that would follow it, which in a decode pass
    /// runs to the end of the pass (the wait would cost a feedback's latency,
    /// about 0.08 ms, per pass). Earlier waits' command buffers complete
    /// before the last wait starts. Not in profile mode, whose waits take
    /// every command buffer's feedback anyway.
    fn witness_last_wait(&self) -> Result<()> {
        if self.ctx.profile {
            return Ok(());
        }
        let mut program = self.program.borrow_mut();
        let Some(last) = program
            .iter()
            .rposition(|item| matches!(item, Item::Wait(event, _) if event.watched()))
        else {
            return Ok(());
        };
        let mut slot = self.slot.borrow_mut();
        let cmd = self
            .ctx
            .device
            .newCommandBuffer()
            .ok_or_else(|| anyhow!("failed to create a Metal 4 command buffer"))?;
        cmd.beginCommandBufferWithAllocator(&slot.get().allocator);
        cmd.endCommandBuffer();
        program.insert(last + 1, Item::Cmd(cmd));
        Ok(())
    }

    /// Closes the open command buffer (if any) into the program.
    fn close_segment(&self) {
        if let Some(seg) = self.open.borrow_mut().take() {
            seg.encoder.endEncoding();
            seg.cmd.endCommandBuffer();
            self.program.borrow_mut().push(Item::Cmd(seg.cmd));
            if self.ctx.profile {
                self.names
                    .borrow_mut()
                    .push(self.segment_kernel.take().unwrap_or("(no dispatch)"));
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
                unsafe {
                    inner.table.setAddress_atIndex(buf.address() + *offset as u64, i)
                };
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
                        ensure!(
                            offset.is_multiple_of(4) && *offset + 4 <= buf.length(),
                            "GPU param at byte {offset} is not a word of its buffer"
                        );
                        buf.address() + *offset as u64
                    }
                };
                // SAFETY: the address lies within a resident arena chunk or buffer.
                unsafe { inner.table.setAddress_atIndex(address, buffers.len() + i) };
            }
            let size = |(w, h, d): (usize, usize, usize)| MTLSize {
                width: w,
                height: h,
                depth: d,
            };
            match grid {
                Grid::Threads { grid, threadgroup } => {
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        size(grid),
                        size(threadgroup),
                    );
                }
                Grid::Threadgroups { groups, threadgroup } => {
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        size(groups),
                        size(threadgroup),
                    );
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
    pub fn copy_buffer(
        &self,
        src: &GpuBuffer,
        src_offset: usize,
        dst: &GpuBuffer,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
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
        profile::count_level();
        self.memory_barrier()
    }

    /// Makes the GPU block here until the host has signaled `event` to at
    /// least `value`. Everything encoded before completes first (the wait
    /// sits between two command buffers), so this also acts as a full
    /// barrier. The pass can then be committed before the inputs read after
    /// this point exist; the host writes them and signals. A pass committed
    /// with an unsatisfied wait holds the whole queue, so every path after
    /// commit must signal (or [`SharedEvent::release_all`]). Metal releases
    /// a wait still blocked about 4 s after its submission
    /// ([`MetalContext::submit`]); a pass whose waits can take that long is
    /// gated ([`Self::gate_host_waits`]).
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
        self.witness_last_wait()?;
        let program = std::mem::take(&mut *self.program.borrow_mut());
        let done = self.done.borrow().clone();
        let milestone = self.milestone.borrow_mut().take();
        let slot = self.slot.replace(Slot { inner: None, pool: self.ctx.pool.clone() });
        let names = std::mem::take(&mut *self.names.borrow_mut());
        Ok(EncodedPass {
            ctx: self.ctx,
            program,
            done,
            milestone,
            gated: self.gated.get(),
            slot,
            label: self.label.get(),
            names,
        })
    }

    /// Encodes, as the pass's last command, a GPU signal of `event` to
    /// `value`, and remembers it: [`PendingPass::wait_polled`] then polls the
    /// event (which the GPU writes directly) instead of blocking on the
    /// fence. Call after everything else is encoded.
    pub fn signal_done(&self, event: &SharedEvent, value: u64) -> Result<()> {
        self.signal_event(event, value)?;
        *self.done.borrow_mut() = Some((event.clone(), value));
        Ok(())
    }

    /// Submits the pass in parts at its waits ([`Self::wait_event`]):
    /// [`EncodedPass::commit`] submits it up to its first wait, and a feeder
    /// thread submits each later part, which begins at a wait, only once the
    /// host has signalled that wait's event to its value. The queue then
    /// never blocks on a host event, so no wait can time out
    /// ([`MetalContext::submit`]), for passes whose waits can add up to
    /// seconds (a long prefill under the expert cache: each layer waits for
    /// its misses to be read). Each part costs about 0.1 ms more than a
    /// queue wait would (the host submits it after the signal instead of
    /// the GPU seeing the signal), and later submissions on the context
    /// wait until the feeder has submitted the last part. The waits' events
    /// must be signalled by another thread than the one that commits on the
    /// context. Without waits the pass is submitted as usual.
    pub fn gate_host_waits(&self) {
        self.gated.set(true);
    }

    /// Marks `event` reaching `value` as the pass's milestone: a wait the
    /// pass encoded ([`Self::wait_event`]) that the host satisfies, after
    /// which little of the pass is left. A polled wait
    /// ([`PendingPass::wait_polled`]) then blocks until the host has
    /// signalled it and polls only the rest, so a pass whose length varies
    /// with host work it waits for (the expert cache's misses) does not keep
    /// a core polling through that work. Called again, the later milestone
    /// replaces the earlier one.
    pub fn set_milestone(&self, event: &SharedEvent, value: u64) {
        *self.milestone.borrow_mut() = Some((event.clone(), value));
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
    milestone: Option<(SharedEvent, u64)>,
    gated: bool,
    slot: Slot,
    label: &'static str,
    /// Profile mode only: one kernel name per `Item::Cmd`.
    names: Vec<&'static str>,
}

impl<'a> EncodedPass<'a> {
    /// Submits without blocking.
    pub fn commit(self) -> Result<PendingPass<'a>> {
        let EncodedPass { ctx, program, done, milestone, gated, slot, label, names } =
            self;
        let cmds = program.iter().filter(|item| matches!(item, Item::Cmd(_))).count();
        let profile = if ctx.profile {
            ensure!(
                names.len() == cmds,
                "profile: {} kernel names for {cmds} command buffers",
                names.len()
            );
            Some(ProfileInfo { label, names })
        } else {
            None
        };
        let feedback = Feedback::new(cmds, profile, watches_of(&program));
        let gated = gated && program.iter().any(|item| matches!(item, Item::Wait(..)));
        let (fence_value, program, feeder) = if gated {
            ctx.submit_gated(program, &feedback)?
        } else {
            (ctx.submit(&program, &feedback)?, program, None)
        };
        Ok(PendingPass {
            ctx,
            program,
            fence_value,
            done,
            milestone,
            feedback,
            feeder,
            slot: Some(slot),
        })
    }

    /// Drops the context lifetime so the pass can be kept inside long-lived
    /// state (passes encoded ahead for several possible outcomes). The
    /// context must outlive it; [`DetachedPass::attach`] restores the tie.
    pub fn detach(self) -> DetachedPass {
        DetachedPass {
            program: self.program,
            done: self.done,
            milestone: self.milestone,
            gated: self.gated,
            slot: self.slot,
            label: self.label,
            names: self.names,
        }
    }
}

/// An [`EncodedPass`] held without its context lifetime; see
/// [`EncodedPass::detach`]. Dropping it un-committed returns its command
/// memory to the pool.
pub struct DetachedPass {
    program: Vec<Item>,
    done: Option<(SharedEvent, u64)>,
    milestone: Option<(SharedEvent, u64)>,
    gated: bool,
    slot: Slot,
    label: &'static str,
    names: Vec<&'static str>,
}

impl DetachedPass {
    pub fn attach<'a>(self, ctx: &'a MetalContext) -> EncodedPass<'a> {
        EncodedPass {
            ctx,
            program: self.program,
            done: self.done,
            milestone: self.milestone,
            gated: self.gated,
            slot: self.slot,
            label: self.label,
            names: self.names,
        }
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
    milestone: Option<(SharedEvent, u64)>,
    feedback: Arc<Feedback>,
    /// A gated pass's feeder, until it submitted the last part.
    feeder: Option<FeederHandle>,
    slot: Option<Slot>,
}

impl Drop for PendingPass<'_> {
    fn drop(&mut self) {
        if self.slot.is_some() {
            // The feeder submits every part whatever happens to this pass
            // (a part not submitted would leave whoever waits for the host
            // events it signals waiting), unless the context faults.
            let _ = self.join_feeder();
            // TODO: after a context fault wait_fence returns early, so the
            // slot's command memory goes back to the pool while parts may
            // still run on the GPU; harmless today only because any fault
            // reloads the engine (predates gated passes).
            let _ = self.ctx.wait_fence(self.fence_value);
            self.program.clear();
            self.slot = None;
        }
    }
}

/// A pass's GPU execution span, in seconds on the host clock of
/// [`host_secs`] (Metal's `GPUStartTime`/`GPUEndTime` count
/// `mach_absolute_time`), so host and GPU marks can be subtracted.
#[derive(Clone, Copy, Debug)]
pub struct PassTiming {
    pub gpu_start_secs: f64,
    pub gpu_end_secs: f64,
}

/// Host time in seconds on the clock [`PassTiming`] uses
/// (`mach_absolute_time`).
pub fn host_secs() -> f64 {
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
    // SAFETY: plain libSystem calls with a valid out-pointer.
    let ticks = unsafe {
        mach_timebase_info(&mut tb);
        mach_absolute_time()
    };
    ticks as f64 * tb.numer as f64 / tb.denom as f64 * 1e-9
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
        let timing = PassTiming {
            gpu_start_secs: state.gpu_start_secs,
            gpu_end_secs: state.gpu_end_secs,
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

    /// Blocks until completion while retaining the pass's feedback.
    /// Benchmarks query its GPU clock only after their cadence timer ends.
    pub fn wait_retain(mut self) -> Result<CompletedPass> {
        self.join_feeder()?;
        self.ctx.wait_fence(self.fence_value)?;
        self.finished()
    }

    /// Like [`Self::wait`], polled: checks the pass's done signal (see
    /// [`ComputePass::signal_done`], falling back to the fence) every
    /// [`POLL_INTERVAL`] and returns within about that much of it, where a
    /// thread blocked on the fence learns of the completion about 0.09 ms
    /// later. Polling replaced a pacer that slept until a predicted
    /// completion: when the window server's frames made pass lengths
    /// irregular, its estimate overslept and left the GPU idle, and
    /// speculative decoding under screen use ran at 33 to 123 tok/s
    /// (median 84) against 87 to 123 (median 113) blocking
    /// (docs/performance.md, "Waiting for the GPU").
    pub fn wait_polled(self) -> Result<()> {
        self.wait_retain_polled().map(|_| ())
    }

    /// [`Self::wait_polled`], keeping the pass's feedback like
    /// [`Self::wait_retain`].
    pub fn wait_retain_polled(mut self) -> Result<CompletedPass> {
        self.join_feeder()?;
        let (event, value) = match &self.done {
            Some((event, value)) => (event.clone(), *value),
            None => (self.ctx.fence.clone(), self.fence_value),
        };
        if let Some((milestone, reached)) = &self.milestone {
            // Blocks: the host signals the milestone only after work of its
            // own of unpredictable length, so polling would only burn the
            // core for that long. The check of the pass's end keeps a
            // milestone that is never signalled (an error path that released
            // the pass's waits) from holding the thread forever.
            while !milestone.wait_until(*reached, MILESTONE_POLL_MS) {
                if event.signaled_value() >= value {
                    break;
                }
                self.ctx.fault.check()?;
            }
        }
        // The fault check is an uncontended lock, cheap next to the sleep;
        // it ends the wait on a failed command buffer as the fence wait does.
        while event.signaled_value() < value {
            self.ctx.fault.check()?;
            std::thread::sleep(POLL_INTERVAL);
        }
        // The done event may lead the fence by the trailing signal; the
        // command memory is only safe to reuse after the fence.
        if self.done.is_some() {
            self.ctx.wait_fence(self.fence_value)?;
        }
        self.finished()
    }

    /// Waits until a gated pass's feeder submitted its last part and keeps
    /// the items it submitted until completion.
    fn join_feeder(&mut self) -> Result<()> {
        let Some(feeder) = self.feeder.take() else { return Ok(()) };
        match feeder.0.join() {
            Ok(Some((items, result))) => {
                self.program.extend(items.0);
                result
            }
            Ok(None) => bail!("a gated pass's feeder never got its pass"),
            Err(_) => {
                let message = "a gated pass's feeder panicked";
                self.ctx.fault.record(message.to_owned());
                bail!(message)
            }
        }
    }

    fn finished(&mut self) -> Result<CompletedPass> {
        // Completed: release the command memory to the pool.
        self.program.clear();
        self.slot = None;
        // A wait that timed out is recorded by the feedback of the command
        // buffer behind it, which ran before the pass's end; make sure it
        // was looked at before the pass's outputs are.
        self.feedback.watches_checked()?;
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

/// How often a polled wait blocked on a milestone checks whether the pass
/// ended without it.
const MILESTONE_POLL_MS: u64 = 50;

/// How long a polled wait ([`PendingPass::wait_polled`]) sleeps between
/// checks of the pass's done signal. Measured on an M5 Max (2026-10-19), a
/// requested 20 us sleep lasts 27 us (the kernel's timer leeway and the
/// wake-up add about a third) and a loop of them costs 3.9% of a core, so
/// a completion is seen about 15 us after it on average and 30 us at most;
/// 10 us cost 6.9% for
/// about 6 us less latency, and a blocking fence wait wakes about 90 us
/// after the signal.
pub const POLL_INTERVAL: Duration = Duration::from_micros(20);

#[cfg(test)]
#[path = "../tests/unit/metal.rs"]
mod tests;
