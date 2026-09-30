//! Diagnostics behind the per-request timings: where a prefill's wall time
//! went, how the paged n-gram table's gathers fared, and what the system's
//! virtual memory did meanwhile.
//!
//! The engine counters are cumulative and per thread: the code that does the
//! work adds to them ([`record`]), and the server reads them before and after
//! a phase and subtracts ([`Counters::since`]). Only the engine thread runs a
//! request, so its counters are exactly the request's, with no locking and no
//! plumbing through the model API. Updating them costs a thread-local read
//! and write.
//!
//! The system side ([`vm_counters`], [`pressure_level`], [`task_memory`]) is
//! a handful of Mach and sysctl calls, a few microseconds each, sampled at
//! phase boundaries only. They are declared here rather than taken from the
//! `libc` crate, whose Mach bindings are deprecated.

use std::cell::Cell;

/// What the paged n-gram table's gathers did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Gather {
    /// Staged batches (one per decode step, verify step or prefill chunk).
    pub batches: u64,
    pub rows: u64,
    /// Rows whose pages were checked with `mincore` right before the copy:
    /// all of them in decode and verify batches, every 16th in prefill
    /// batches, where checking every page would cost more than the copy.
    pub checked_rows: u64,
    /// Pages the checked rows' bytes lie on, counted per row (a row spans
    /// three to six pages across its codes, scales and biases; a page two
    /// rows share counts twice).
    pub pages: u64,
    /// Of those, the ones that were not resident: each is a read from the
    /// SSD that the copy then waits for.
    pub cold_pages: u64,
    /// Wall time inside `PagedTable::gather`, residency checks included.
    pub secs: f64,
}

/// Where the prefill loop's time went, summed over its chunks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Prefill {
    pub chunks: u64,
    /// Growing the state's caches and the prefill scratch (allocation and
    /// zeroing, plus the copy of the old caches when the state grows).
    pub alloc_secs: f64,
    /// Hashing the chunk's n-gram ids and staging their rows.
    pub ngram_secs: f64,
    /// Encoding the chunk's pass on the host.
    pub encode_secs: f64,
    /// The passes' GPU execution, from the commit feedback's timestamps.
    pub gpu_secs: f64,
    /// Commit to completion minus the GPU span: submission latency, the GPU
    /// busy with other work, residency being established. A pass without
    /// usable timestamps counts here whole.
    pub wait_secs: f64,
}

/// The engine thread's cumulative counters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Counters {
    pub gather: Gather,
    pub prefill: Prefill,
}

const ZERO: Counters = Counters {
    gather: Gather {
        batches: 0,
        rows: 0,
        checked_rows: 0,
        pages: 0,
        cold_pages: 0,
        secs: 0.0,
    },
    prefill: Prefill {
        chunks: 0,
        alloc_secs: 0.0,
        ngram_secs: 0.0,
        encode_secs: 0.0,
        gpu_secs: 0.0,
        wait_secs: 0.0,
    },
};

thread_local! {
    static COUNTERS: Cell<Counters> = const { Cell::new(ZERO) };
}

/// This thread's counters so far.
pub fn counters() -> Counters {
    COUNTERS.with(Cell::get)
}

/// Adds to this thread's counters.
pub fn record(f: impl FnOnce(&mut Counters)) {
    COUNTERS.with(|cell| {
        let mut counters = cell.get();
        f(&mut counters);
        cell.set(counters);
    });
}

impl Counters {
    /// What was added since `earlier` (an earlier [`counters`] of the same
    /// thread).
    pub fn since(self, earlier: Counters) -> Counters {
        let (g, e) = (self.gather, earlier.gather);
        let (p, q) = (self.prefill, earlier.prefill);
        Counters {
            gather: Gather {
                batches: g.batches.saturating_sub(e.batches),
                rows: g.rows.saturating_sub(e.rows),
                checked_rows: g.checked_rows.saturating_sub(e.checked_rows),
                pages: g.pages.saturating_sub(e.pages),
                cold_pages: g.cold_pages.saturating_sub(e.cold_pages),
                secs: (g.secs - e.secs).max(0.0),
            },
            prefill: Prefill {
                chunks: p.chunks.saturating_sub(q.chunks),
                alloc_secs: (p.alloc_secs - q.alloc_secs).max(0.0),
                ngram_secs: (p.ngram_secs - q.ngram_secs).max(0.0),
                encode_secs: (p.encode_secs - q.encode_secs).max(0.0),
                gpu_secs: (p.gpu_secs - q.gpu_secs).max(0.0),
                wait_secs: (p.wait_secs - q.wait_secs).max(0.0),
            },
        }
    }
}

/// System-wide paging counters (`host_statistics64`, `HOST_VM_INFO64`), in
/// pages of the kernel's page size (16 KB on Apple Silicon). Cumulative since
/// boot; the difference of two samples is what happened in between.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct VmCounters {
    /// Pages read in from files (a cold n-gram row is one of these).
    pub pageins: u64,
    /// Dirty file pages written out to make room.
    pub pageouts: u64,
    /// Compressed pages read back from the swap files.
    pub swapins: u64,
    /// Compressed pages written to the swap files.
    pub swapouts: u64,
    /// Pages the compressor took in to free memory.
    pub compressions: u64,
    /// Compressed pages that were touched again and expanded.
    pub decompressions: u64,
}

impl VmCounters {
    pub fn since(self, earlier: VmCounters) -> VmCounters {
        VmCounters {
            pageins: self.pageins.saturating_sub(earlier.pageins),
            pageouts: self.pageouts.saturating_sub(earlier.pageouts),
            swapins: self.swapins.saturating_sub(earlier.swapins),
            swapouts: self.swapouts.saturating_sub(earlier.swapouts),
            compressions: self.compressions.saturating_sub(earlier.compressions),
            decompressions: self.decompressions.saturating_sub(earlier.decompressions),
        }
    }
}

/// The system's paging counters now, `None` when the call fails.
pub fn vm_counters() -> Option<VmCounters> {
    let mut info = sys::VmStatistics64::default();
    let mut count = sys::VmStatistics64::COUNT;
    // SAFETY: `info` is a correctly laid out `vm_statistics64` of `count`
    // 32-bit words; the kernel writes at most that many.
    let rc = unsafe {
        sys::host_statistics64(
            sys::host(),
            sys::HOST_VM_INFO64,
            (&raw mut info).cast(),
            &mut count,
        )
    };
    (rc == 0).then_some(VmCounters {
        pageins: info.pageins,
        pageouts: info.pageouts,
        swapins: info.swapins,
        swapouts: info.swapouts,
        compressions: info.compressions,
        decompressions: info.decompressions,
    })
}

/// The memorystatus pressure level (`kern.memorystatus_vm_pressure_level`):
/// 1 normal, 2 warning, 4 critical. `None` when the sysctl fails.
pub fn pressure_level() -> Option<u32> {
    let mut level: std::ffi::c_int = 0;
    let mut len = std::mem::size_of::<std::ffi::c_int>();
    // SAFETY: a NUL-terminated name and an int-sized output buffer.
    let rc = unsafe {
        sys::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            (&raw mut level).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && level >= 0).then_some(level as u32)
}

/// How much memory this process may lock with `mlock`: the smaller of
/// `vm.user_wire_limit` (per process) and `vm.global_user_wire_limit` (every
/// process's user wirings plus what the system has wired already, e.g.
/// 116.8 GB each on a 128 GB machine). `None` when neither can be read.
pub fn user_wire_limit() -> Option<u64> {
    let per_process = sysctl_u64(c"vm.user_wire_limit");
    let global = sysctl_u64(c"vm.global_user_wire_limit");
    match (per_process, global) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// A 32- or 64-bit unsigned sysctl, `None` when it cannot be read.
fn sysctl_u64(name: &std::ffi::CStr) -> Option<u64> {
    let mut value: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: a NUL-terminated name and an 8-byte output buffer of `len`.
    let rc = unsafe {
        sys::sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    match (rc, len) {
        (0, 8) => Some(value),
        // A 32-bit value lands in the low half (little-endian).
        (0, 4) => Some(value & 0xffff_ffff),
        _ => None,
    }
}

/// The name of a [`pressure_level`].
pub fn pressure_name(level: u32) -> &'static str {
    match level {
        1 => "normal",
        2 => "warning",
        4 => "critical",
        _ => "unknown",
    }
}

/// This process's memory as the system accounts it (`task_info`,
/// `TASK_VM_INFO`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TaskMemory {
    /// What memory pressure and the jetsam limits charge the process with
    /// (Activity Monitor's "Memory"): its anonymous memory, its GPU buffers
    /// and its wired pages, not the page cache it maps.
    pub phys_footprint: u64,
    /// Bytes of the process's memory currently held by the compressor.
    pub compressed: u64,
}

/// This process's footprint now, `None` when the call fails.
pub fn task_memory() -> Option<TaskMemory> {
    let mut info = sys::TaskVmInfo::default();
    let mut count = sys::TaskVmInfo::COUNT;
    // SAFETY: `info` is the revision-1 `task_vm_info` layout, `count` words
    // long; the kernel fills at most that many.
    let rc = unsafe {
        sys::task_info(
            sys::mach_task_self_,
            sys::TASK_VM_INFO,
            (&raw mut info).cast(),
            &mut count,
        )
    };
    (rc == 0 && count >= sys::TaskVmInfo::COUNT).then_some(TaskMemory {
        phys_footprint: info.phys_footprint,
        compressed: info.compressed,
    })
}

/// The Mach and sysctl calls above. Layouts are the macOS SDK's
/// `vm_statistics64` and `task_vm_info` (checked against the headers with
/// `offsetof`: `pageins` at 32, `swapouts` at 120; `compressed` at 120,
/// `phys_footprint` at 144).
mod sys {
    use std::ffi::{c_char, c_int, c_void};
    use std::sync::OnceLock;

    pub const HOST_VM_INFO64: c_int = 4;
    pub const TASK_VM_INFO: c_int = 22;

    #[repr(C)]
    #[derive(Default)]
    pub struct VmStatistics64 {
        pub free_count: u32,
        pub active_count: u32,
        pub inactive_count: u32,
        pub wire_count: u32,
        pub zero_fill_count: u64,
        pub reactivations: u64,
        pub pageins: u64,
        pub pageouts: u64,
        pub faults: u64,
        pub cow_faults: u64,
        pub lookups: u64,
        pub hits: u64,
        pub purges: u64,
        pub purgeable_count: u32,
        pub speculative_count: u32,
        pub decompressions: u64,
        pub compressions: u64,
        pub swapins: u64,
        pub swapouts: u64,
        pub compressor_page_count: u32,
        pub throttled_count: u32,
        pub external_page_count: u32,
        pub internal_page_count: u32,
        pub total_uncompressed_pages_in_compressor: u64,
    }

    impl VmStatistics64 {
        /// `HOST_VM_INFO64_COUNT` of this revision: 152 bytes in 32-bit words.
        pub const COUNT: u32 = (std::mem::size_of::<Self>() / 4) as u32;
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct TaskVmInfo {
        pub virtual_size: u64,
        pub region_count: i32,
        pub page_size: i32,
        pub resident_size: u64,
        pub resident_size_peak: u64,
        pub device: u64,
        pub device_peak: u64,
        pub internal: u64,
        pub internal_peak: u64,
        pub external: u64,
        pub external_peak: u64,
        pub reusable: u64,
        pub reusable_peak: u64,
        pub purgeable_volatile_pmap: u64,
        pub purgeable_volatile_resident: u64,
        pub purgeable_volatile_virtual: u64,
        pub compressed: u64,
        pub compressed_peak: u64,
        pub compressed_lifetime: u64,
        pub phys_footprint: u64,
    }

    impl TaskVmInfo {
        /// `TASK_VM_INFO_REV1_COUNT`: through `phys_footprint`, 152 bytes.
        pub const COUNT: u32 = (std::mem::size_of::<Self>() / 4) as u32;
    }

    unsafe extern "C" {
        pub static mach_task_self_: u32;
        fn mach_host_self() -> u32;
        pub fn host_statistics64(
            host: u32,
            flavor: c_int,
            info: *mut c_int,
            count: *mut u32,
        ) -> c_int;
        pub fn task_info(
            task: u32,
            flavor: c_int,
            info: *mut c_int,
            count: *mut u32,
        ) -> c_int;
        pub fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    /// The host port, taken once: every `mach_host_self` call adds a send
    /// right to it.
    pub fn host() -> u32 {
        static HOST: OnceLock<u32> = OnceLock::new();
        // SAFETY: no preconditions.
        *HOST.get_or_init(|| unsafe { mach_host_self() })
    }
}

#[cfg(test)]
#[path = "../tests/unit/stats.rs"]
mod tests;
