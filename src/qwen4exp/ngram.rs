//! The hashed n-gram embedding table: host-side hashing (the host owns the
//! token history anyway) and two storage backends for the 32 GB table.
//!
//! * [`NgramTable::Resident`] keeps the quantized table in GPU memory and
//!   gathers rows by hashed id, as phase 1 did.
//! * [`NgramTable::Paged`] leaves the table in the checkpoint files, memory
//!   maps them, and copies the 16 rows a token needs into a small staging
//!   buffer, which the same gather kernel then dequantizes with sequential
//!   ids. The rows live in the page cache, evictable (or pinned on request),
//!   instead of in wired GPU memory: on the 128 GB machine this frees 32 GB
//!   for KV and session caches at no quality cost. A warm row is a memcpy.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context as _, Result, ensure};

use crate::metal::MetalContext;
use crate::safetensors::{Checkpoint, SafetensorsDType};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

/// The Hugging Face `Qwen4ExpTextNGramEmbedding` hashing.
pub struct NgramHasher {
    multipliers: Vec<u64>,
    sizes: Vec<u64>,
    offsets: Vec<u64>,
    heads_per_ngram: usize,
    eos: u32,
}

impl NgramHasher {
    pub fn new(
        multipliers: &[u64],
        sizes: &[u64],
        offsets: &[u64],
        heads_per_ngram: usize,
        eos: u32,
    ) -> Result<Self> {
        ensure!(multipliers.len() >= 3, "need unigram, bigram and trigram multipliers");
        ensure!(
            sizes.len() == offsets.len() && !sizes.is_empty(),
            "sizes/offsets mismatch"
        );
        ensure!(
            heads_per_ngram > 0 && sizes.len() == 2 * heads_per_ngram,
            "head count"
        );
        ensure!(
            sizes.iter().zip(offsets).all(|(&s, &o)| s + o <= u32::MAX as u64),
            "table ids must fit u32"
        );
        Ok(Self {
            multipliers: multipliers.to_vec(),
            sizes: sizes.to_vec(),
            offsets: offsets.to_vec(),
            heads_per_ngram,
            eos,
        })
    }

    pub fn heads(&self) -> usize {
        self.sizes.len()
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// Row ids for `tokens` given the two tokens before them (oldest first);
    /// head `j` hashes the current token with `j / heads_per_ngram + 1`
    /// predecessors, and a predecessor beyond an eos is replaced by eos so
    /// segments never mix. Appends `tokens.len() * heads()` ids to `out`.
    pub fn ids(&self, tokens: &[u32], hist: [u32; 2], out: &mut Vec<u32>) {
        let eos = self.eos;
        let m = &self.multipliers;
        for (r, &t0) in tokens.iter().enumerate() {
            let p1 = if r >= 1 { tokens[r - 1] } else { hist[1] };
            let p2 = match r {
                0 => hist[0],
                1 => hist[1],
                _ => tokens[r - 2],
            };
            let s1 = if p1 == eos { eos } else { p1 };
            let s2 = if p1 == eos || p2 == eos { eos } else { p2 };
            let bigram =
                (t0 as u64).wrapping_mul(m[0]) ^ (s1 as u64).wrapping_mul(m[1]);
            let trigram = bigram ^ (s2 as u64).wrapping_mul(m[2]);
            for (j, (&size, &offset)) in
                self.sizes.iter().zip(&self.offsets).enumerate()
            {
                let mixed =
                    if j / self.heads_per_ngram >= 1 { trigram } else { bigram };
                out.push((mixed % size) as u32 + offset as u32);
            }
        }
    }

    /// The history after feeding `tokens`.
    pub fn advance(hist: [u32; 2], tokens: &[u32]) -> [u32; 2] {
        match tokens {
            [] => hist,
            [t] => [hist[1], *t],
            [.., a, b] => [*a, *b],
        }
    }
}

/// A read-only shared mapping of one shard file.
struct Mapping {
    ptr: *const u8,
    len: usize,
    /// The file, which the preload reads through the page cache the mapping
    /// shares.
    path: PathBuf,
}

// SAFETY: the mapping is immutable file-backed memory; concurrent reads from
// any thread are fine, and it is only unmapped when the last owner drops.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    fn open(path: &Path) -> Result<Self> {
        use std::os::unix::io::AsRawFd as _;
        let file =
            File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let len = file.metadata().context("file metadata")?.len() as usize;
        ensure!(len > 0, "{} is empty", path.display());
        // SAFETY: a fresh read-only shared mapping of the whole file.
        let ptr = unsafe {
            sys::mmap(
                std::ptr::null_mut(),
                len,
                sys::PROT_READ,
                sys::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        ensure!(
            ptr != sys::MAP_FAILED,
            "mmap of {} failed: {}",
            path.display(),
            std::io::Error::last_os_error()
        );
        Ok(Self { ptr: ptr.cast::<u8>().cast_const(), len, path: path.to_path_buf() })
    }

    fn slice(&self, offset: usize, len: usize) -> &[u8] {
        assert!(offset + len <= self.len, "mapping read out of range");
        // SAFETY: in-bounds view of the mapping, which outlives `self`.
        unsafe { core::slice::from_raw_parts(self.ptr.add(offset), len) }
    }

    /// Hints that `[offset, offset+len)` will be read soon (asynchronous
    /// read-ahead for pages not yet resident).
    fn will_need(&self, offset: usize, len: usize) {
        let page = sys::page_size();
        let start = offset & !(page - 1);
        let end = (offset + len).min(self.len);
        // SAFETY: page-aligned range inside the mapping; advice only.
        unsafe {
            sys::madvise(
                self.ptr.add(start).cast_mut().cast(),
                end - start,
                sys::MADV_WILLNEED,
            );
        }
    }

    /// Checks which pages of `[offset, offset+len)` are resident and asks
    /// for read-ahead on the range when any is not. Returns the pages the
    /// range lies on and how many of them were not resident. A failed
    /// `mincore` counts every page as cold and hints anyway: the answer is
    /// diagnostics and a hint, never a reason to fail a gather.
    fn prefetch(&self, offset: usize, len: usize) -> (u64, u64) {
        let page = sys::page_size();
        let start = offset & !(page - 1);
        let end = (offset + len).min(self.len);
        let pages = (end - start).div_ceil(page);
        // A row's slice of one region spans one or two pages; wider ranges
        // are not gathered row by row and just take the hint.
        let mut vec = [0u8; 4];
        let cold = if pages <= vec.len() {
            // SAFETY: page-aligned range inside the mapping; `vec` has one
            // byte for each of its pages.
            let rc = unsafe {
                sys::mincore(
                    self.ptr.add(start).cast_mut().cast(),
                    end - start,
                    vec.as_mut_ptr().cast(),
                )
            };
            if rc == 0 {
                vec[..pages].iter().filter(|&&b| b & 1 == 0).count()
            } else {
                pages
            }
        } else {
            pages
        };
        if cold > 0 {
            self.will_need(offset, len);
        }
        (pages as u64, cold as u64)
    }

    /// Whether every page of `[offset, offset+len)` is resident; `false`
    /// when `mincore` fails.
    fn all_resident(&self, offset: usize, len: usize) -> bool {
        self.resident(offset, len).is_ok_and(|bytes| {
            let page = sys::page_size();
            let start = offset & !(page - 1);
            bytes >= (offset + len).min(self.len) - start
        })
    }

    /// Bytes of `[offset, offset+len)` currently resident in memory.
    fn resident(&self, offset: usize, len: usize) -> Result<usize> {
        let page = sys::page_size();
        let start = offset & !(page - 1);
        let end = (offset + len).min(self.len).div_ceil(page) * page;
        let end = end.min(self.len.div_ceil(page) * page);
        let pages = (end - start) / page;
        let mut vec = vec![0u8; pages];
        // SAFETY: page-aligned range inside the mapping; `vec` has one byte per page.
        let rc = unsafe {
            sys::mincore(
                self.ptr.add(start).cast_mut().cast(),
                end - start,
                vec.as_mut_ptr().cast(),
            )
        };
        ensure!(rc == 0, "mincore failed: {}", std::io::Error::last_os_error());
        Ok(vec.iter().filter(|&&b| b & 1 != 0).count() * page)
    }

    /// Pins `[offset, offset+len)` in memory.
    fn lock(&self, offset: usize, len: usize) -> Result<()> {
        let page = sys::page_size();
        let start = offset & !(page - 1);
        let end = (offset + len).min(self.len);
        // SAFETY: page-aligned range inside the mapping.
        let rc =
            unsafe { sys::mlock(self.ptr.add(start).cast_mut().cast(), end - start) };
        ensure!(rc == 0, "mlock failed: {}", std::io::Error::last_os_error());
        Ok(())
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: unmapping what `open` mapped.
        unsafe {
            sys::munmap(self.ptr.cast_mut().cast(), self.len);
        }
    }
}

/// The handful of libSystem calls the mapping needs, declared here to avoid a
/// dependency on the `libc` crate.
mod sys {
    use std::ffi::{c_int, c_void};

    pub const PROT_READ: c_int = 1;
    pub const MAP_SHARED: c_int = 1;
    pub const MADV_WILLNEED: c_int = 3;
    pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
    const QOS_CLASS_BACKGROUND: u32 = 0x09;
    const IOPOL_TYPE_DISK: c_int = 0;
    const IOPOL_SCOPE_THREAD: c_int = 1;
    const IOPOL_THROTTLE: c_int = 3;

    unsafe extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
        pub fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
        pub fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> c_int;
        pub fn mlock(addr: *mut c_void, len: usize) -> c_int;
        fn getpagesize() -> c_int;
        fn pthread_set_qos_class_self_np(qos: u32, relative_priority: c_int) -> c_int;
        fn setiopolicy_np(iotype: c_int, scope: c_int, policy: c_int) -> c_int;
    }

    /// Puts the calling thread at background QoS (efficiency cores, lowest
    /// scheduling priority) and throttles its disk I/O, which then yields
    /// to any other I/O on the device. Best effort; returns whether both
    /// took.
    pub fn lower_thread_priority() -> bool {
        // SAFETY: both calls only change the calling thread's own policy.
        unsafe {
            pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0) == 0
                && setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_THREAD, IOPOL_THROTTLE)
                    == 0
        }
    }

    pub fn page_size() -> usize {
        // SAFETY: no preconditions.
        (unsafe { getpagesize() }) as usize
    }
}

/// One tensor's bytes: the mapping holding it and its offset there.
struct Region {
    map: Arc<Mapping>,
    start: usize,
}

/// One contiguous row range of the table (its codes, scales and biases may
/// sit in different shard files).
struct ShardRegion {
    codes: Region,
    scales: Region,
    biases: Region,
    rows: usize,
}

/// The table left in the checkpoint files, memory-mapped and read row by
/// row on demand. Warm rows are memcpys from the page cache; cold rows get a
/// read-ahead hint for their pages first so their faults overlap.
pub struct PagedTable {
    regions: Vec<ShardRegion>,
    /// `row_starts[i]` is the first global row of region `i`; one extra entry
    /// holds the total.
    row_starts: Vec<usize>,
    words: usize,
    groups: usize,
    group_size: usize,
}

/// Batches at least this large are copied by several threads (prefill);
/// smaller ones (decode, verify) are copied inline.
const PARALLEL_ROWS: usize = 256;

/// Prefill batches check the residency of every this-many-th row only. A
/// `mincore` call costs about 0.4 us and the calls do not scale across the
/// copy threads, so checking all of a 4 096-token chunk's ~200 000 pages
/// took 80 ms against the 2 ms of the copy; one row in 16 costs about 5 ms
/// per chunk (0.2 % of its GPU time) and still counts thousands of pages.
/// Decode and verify batches check every row.
const PREFILL_CHECK_EVERY: usize = 16;

/// What the residency checks of one gather saw.
#[derive(Default)]
struct Residency {
    rows: u64,
    pages: u64,
    cold: u64,
}

impl PagedTable {
    pub const BITS: usize = 4;

    /// Opens the shards named `bases[i]` (each with `.weight`, `.scales` and
    /// `.biases`) in row order.
    pub fn open(
        ckpt: &Checkpoint,
        bases: &[String],
        group_size: usize,
    ) -> Result<Self> {
        ensure!(!bases.is_empty(), "n-gram table has no shards");
        let mut regions = Vec::with_capacity(bases.len());
        let mut row_starts = vec![0usize];
        let mut maps: Vec<(std::path::PathBuf, Arc<Mapping>)> = Vec::new();
        let mut words = None;
        let mut groups = None;
        for base in bases {
            let codes = ckpt
                .meta(&format!("{base}.weight"))
                .with_context(|| format!("{base}.weight missing"))?;
            let scales = ckpt
                .meta(&format!("{base}.scales"))
                .with_context(|| format!("{base}.scales missing"))?;
            let biases = ckpt
                .meta(&format!("{base}.biases"))
                .with_context(|| format!("{base}.biases missing"))?;
            ensure!(
                matches!(codes.dtype, SafetensorsDType::U32)
                    && matches!(scales.dtype, SafetensorsDType::BF16)
                    && matches!(biases.dtype, SafetensorsDType::BF16),
                "{base}: unexpected dtypes for an affine table"
            );
            ensure!(
                codes.shape.len() == 2
                    && scales.shape.len() == 2
                    && biases.shape == scales.shape,
                "{base}: table shard shapes must be [rows, k]"
            );
            let rows = codes.shape[0];
            ensure!(
                scales.shape[0] == rows,
                "{base}: scale rows differ from code rows"
            );
            let w = codes.shape[1];
            let g = scales.shape[1];
            ensure!(words.is_none_or(|x| x == w), "{base}: code width differs");
            ensure!(groups.is_none_or(|x| x == g), "{base}: group count differs");
            words = Some(w);
            groups = Some(g);
            let mut open = |meta: &crate::safetensors::TensorMeta| -> Result<Region> {
                let map = match maps.iter().find(|(p, _)| p == meta.shard()) {
                    Some((_, m)) => m.clone(),
                    None => {
                        let m = Arc::new(Mapping::open(meta.shard())?);
                        maps.push((meta.shard().to_path_buf(), m.clone()));
                        m
                    }
                };
                ensure!(
                    meta.start() as usize + meta.byte_len() <= map.len,
                    "{base}: tensor beyond its shard file"
                );
                Ok(Region { map, start: meta.start() as usize })
            };
            regions.push(ShardRegion {
                codes: open(codes)?,
                scales: open(scales)?,
                biases: open(biases)?,
                rows,
            });
            row_starts.push(row_starts.last().copied().unwrap_or(0) + rows);
        }
        let words = words.context("no shards")?;
        let groups = groups.context("no shards")?;
        ensure!(
            words * (32 / Self::BITS) == groups * group_size,
            "table width {} does not match {groups} groups of {group_size}",
            words * (32 / Self::BITS)
        );
        Ok(Self { regions, row_starts, words, groups, group_size })
    }

    pub fn rows(&self) -> usize {
        *self.row_starts.last().unwrap_or(&0)
    }

    /// Row width in elements.
    pub fn width(&self) -> usize {
        self.words * (32 / Self::BITS)
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    fn codes_bytes(&self) -> usize {
        self.words * 4
    }

    fn group_bytes(&self) -> usize {
        self.groups * 2
    }

    /// Bytes of the table on disk.
    pub fn bytes(&self) -> u64 {
        self.regions
            .iter()
            .map(|r| (r.rows * (self.codes_bytes() + 2 * self.group_bytes())) as u64)
            .sum()
    }

    /// The three byte ranges of `row`: (region, offsets of codes, scales, biases).
    fn locate(&self, row: usize) -> Result<(&ShardRegion, usize, usize, usize)> {
        let region_index = self.row_starts.partition_point(|&s| s <= row);
        ensure!(
            region_index >= 1 && row < self.rows(),
            "n-gram row {row} out of range"
        );
        let region = &self.regions[region_index - 1];
        let local = row - self.row_starts[region_index - 1];
        Ok((
            region,
            region.codes.start + local * self.codes_bytes(),
            region.scales.start + local * self.group_bytes(),
            region.biases.start + local * self.group_bytes(),
        ))
    }

    /// Copies the rows `ids` into row-major `codes`, `scales` and `biases`.
    /// The pages of every `check_every`-th row are checked with `mincore`
    /// first and the cold ones get a read-ahead hint, so their SSD reads
    /// overlap instead of faulting one after another during the copy
    /// (`madvise` on a resident page is a wasted system call, which the
    /// check saves). Returns the rows checked, their pages and how many of
    /// those were cold.
    fn stage_rows(
        &self,
        ids: &[u32],
        codes: &mut [u8],
        scales: &mut [u8],
        biases: &mut [u8],
        check_every: usize,
    ) -> Result<Residency> {
        let (cb, gb) = (self.codes_bytes(), self.group_bytes());
        let mut seen = Residency::default();
        for &id in ids.iter().step_by(check_every) {
            let (region, c, s, b) = self.locate(id as usize)?;
            seen.rows += 1;
            for (map, offset, len) in [
                (&region.codes.map, c, cb),
                (&region.scales.map, s, gb),
                (&region.biases.map, b, gb),
            ] {
                let (pages, cold) = map.prefetch(offset, len);
                seen.pages += pages;
                seen.cold += cold;
            }
        }
        for (i, &id) in ids.iter().enumerate() {
            let (region, c, s, b) = self.locate(id as usize)?;
            codes[i * cb..(i + 1) * cb].copy_from_slice(region.codes.map.slice(c, cb));
            scales[i * gb..(i + 1) * gb]
                .copy_from_slice(region.scales.map.slice(s, gb));
            biases[i * gb..(i + 1) * gb]
                .copy_from_slice(region.biases.map.slice(b, gb));
        }
        Ok(seen)
    }

    /// Reads the rows `ids` into row-major `codes`, `scales` and `biases`
    /// (each exactly `ids.len()` rows wide), and adds the batch to the
    /// calling thread's gather counters ([`crate::stats`]).
    pub fn gather(
        &self,
        ids: &[u32],
        codes: &mut [u8],
        scales: &mut [u8],
        biases: &mut [u8],
    ) -> Result<()> {
        let started = std::time::Instant::now();
        let (cb, gb) = (self.codes_bytes(), self.group_bytes());
        ensure!(
            codes.len() == ids.len() * cb
                && scales.len() == ids.len() * gb
                && biases.len() == ids.len() * gb,
            "staging buffers do not match {} rows",
            ids.len()
        );
        let seen = if ids.len() < PARALLEL_ROWS {
            self.stage_rows(ids, codes, scales, biases, 1)?
        } else {
            self.stage_parallel(ids, codes, scales, biases)?
        };
        let secs = started.elapsed().as_secs_f64();
        crate::stats::record(|c| {
            c.gather.batches += 1;
            c.gather.rows += ids.len() as u64;
            c.gather.checked_rows += seen.rows;
            c.gather.pages += seen.pages;
            c.gather.cold_pages += seen.cold;
            c.gather.secs += secs;
        });
        Ok(())
    }

    /// [`Self::stage_rows`] split over up to 8 threads (prefill batches),
    /// checking every [`PREFILL_CHECK_EVERY`]-th row.
    fn stage_parallel(
        &self,
        ids: &[u32],
        codes: &mut [u8],
        scales: &mut [u8],
        biases: &mut [u8],
    ) -> Result<Residency> {
        let (cb, gb) = (self.codes_bytes(), self.group_bytes());
        let threads = 8usize.min(ids.len() / 64).max(1);
        let per = ids.len().div_ceil(threads);
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            let mut rest = (ids, codes, scales, biases);
            loop {
                let n = per.min(rest.0.len());
                if n == 0 {
                    break;
                }
                let (ids_a, ids_b) = rest.0.split_at(n);
                let (c_a, c_b) = rest.1.split_at_mut(n * cb);
                let (s_a, s_b) = rest.2.split_at_mut(n * gb);
                let (b_a, b_b) = rest.3.split_at_mut(n * gb);
                rest = (ids_b, c_b, s_b, b_b);
                handles.push(scope.spawn(move || {
                    self.stage_rows(ids_a, c_a, s_a, b_a, PREFILL_CHECK_EVERY)
                }));
            }
            let mut seen = Residency::default();
            for handle in handles {
                let part = handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("n-gram copy thread panicked"))??;
                seen.rows += part.rows;
                seen.pages += part.pages;
                seen.cold += part.cold;
            }
            Ok(seen)
        })
    }

    /// Reads the whole table once so its pages are resident, and pins them
    /// with `mlock` when `lock` is set (each tensor right after it was
    /// read). Reads go through `pread` in [`PRELOAD_CHUNK`] pieces, which
    /// fills the page cache the mapping shares at the SSD's sequential rate;
    /// touching the mapping page by page was fault-bound (on a cold 4 GB
    /// file 4 GB/s from one thread against 12 GB/s for the reads, and at
    /// background QoS 1.9 to 4 GB/s). A piece that `mincore` finds resident
    /// already is skipped, so a warm table costs only the checks. `stop`
    /// ends it between pieces. Returns what was read and what is resident
    /// afterwards (from `mincore`), which is what the log should show.
    pub fn preload(&self, lock: bool, stop: &AtomicBool) -> Result<Preloaded> {
        use std::os::unix::fs::FileExt as _;
        let (cb, gb) = (self.codes_bytes(), self.group_bytes());
        let mut buf = vec![0u8; PRELOAD_CHUNK];
        let mut read = 0u64;
        let mut stopped = false;
        'parts: for region in &self.regions {
            for (part, len) in [
                (&region.codes, region.rows * cb),
                (&region.scales, region.rows * gb),
                (&region.biases, region.rows * gb),
            ] {
                let file = File::open(&part.map.path).with_context(|| {
                    format!("opening {} for the preload", part.map.path.display())
                })?;
                let mut offset = part.start;
                let end = part.start + len;
                while offset < end {
                    if stop.load(Ordering::Relaxed) {
                        stopped = true;
                        break 'parts;
                    }
                    let n = PRELOAD_CHUNK.min(end - offset);
                    if !part.map.all_resident(offset, n) {
                        file.read_exact_at(&mut buf[..n], offset as u64).with_context(
                            || format!("reading {}", part.map.path.display()),
                        )?;
                        read += n as u64;
                    }
                    offset += n;
                }
                if lock {
                    part.map.lock(part.start, len)?;
                }
            }
        }
        Ok(Preloaded { read, resident: self.resident_bytes()?, stopped })
    }

    /// Runs [`Self::preload`] on a thread of its own at background QoS with
    /// throttled I/O, so it does not compete with the requests being served
    /// meanwhile; a row they need before the preload reaches it just faults
    /// in (the table is read-only). `done` gets the outcome and the seconds
    /// it took, on that thread. The handle stops the preload when dropped.
    pub fn preload_in_background(
        self: &Arc<Self>,
        lock: bool,
        done: impl FnOnce(Result<Preloaded>, f64) + Send + 'static,
    ) -> Result<BackgroundPreload> {
        let stop = Arc::new(AtomicBool::new(false));
        let table = self.clone();
        let flag = stop.clone();
        let thread = std::thread::Builder::new()
            .name("lily-ngram-preload".into())
            .spawn(move || {
                let started = Instant::now();
                if !sys::lower_thread_priority() {
                    eprintln!("n-gram preload: could not lower its thread priority");
                }
                let outcome = table.preload(lock, &flag);
                done(outcome, started.elapsed().as_secs_f64());
            })
            .context("spawning the n-gram preload thread")?;
        Ok(BackgroundPreload { stop, thread: Some(thread) })
    }

    /// Bytes of the table currently resident in memory.
    pub fn resident_bytes(&self) -> Result<u64> {
        let (cb, gb) = (self.codes_bytes(), self.group_bytes());
        let mut total = 0u64;
        for region in &self.regions {
            for (part, len) in [
                (&region.codes, region.rows * cb),
                (&region.scales, region.rows * gb),
                (&region.biases, region.rows * gb),
            ] {
                total += part.map.resident(part.start, len)? as u64;
            }
        }
        Ok(total)
    }
}

/// Bytes one preload read request covers (see [`PagedTable::preload`]).
const PRELOAD_CHUNK: usize = 8 << 20;

/// What [`PagedTable::preload`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Preloaded {
    /// Bytes read from the files (pieces not already resident).
    pub read: u64,
    /// Bytes of the table resident when it ended.
    pub resident: u64,
    /// Whether it was stopped before reaching the end.
    pub stopped: bool,
}

/// A preload running on its own thread ([`PagedTable::preload_in_background`]).
/// The thread holds the table, and with it the mappings, until it ends.
pub struct BackgroundPreload {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl BackgroundPreload {
    /// Asks the preload to stop at its next piece and waits for the thread,
    /// so the table's mappings are released with the model that owns them.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for BackgroundPreload {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Where the table lives.
pub enum NgramTable {
    Resident(Box<QuantWeights>),
    Paged(Arc<PagedTable>),
}

impl NgramTable {
    pub fn is_paged(&self) -> bool {
        matches!(self, Self::Paged(_))
    }
}

/// Staging for gathered rows in the same affine layout as the table, plus the
/// sequential ids the gather kernel reads them with.
pub struct StagedRows {
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    seq_ids: Tensor,
    group_size: usize,
    capacity: usize,
}

impl StagedRows {
    pub fn new(
        ctx: &MetalContext,
        table: &PagedTable,
        capacity: usize,
    ) -> Result<Self> {
        ensure!(capacity > 0, "staging capacity must be nonzero");
        let seq: Vec<u32> = (0..capacity as u32).collect();
        Ok(Self {
            codes: Tensor::zeros(ctx, &[capacity, table.words], DType::U32)?,
            scales: Tensor::zeros(ctx, &[capacity, table.groups], DType::BF16)?,
            biases: Tensor::zeros(ctx, &[capacity, table.groups], DType::BF16)?,
            seq_ids: Tensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&seq),
                &[capacity],
                DType::U32,
            )?,
            group_size: table.group_size,
            capacity,
        })
    }

    /// Reads rows `ids` from `table` into the staging buffers. The GPU must
    /// not be reading them (the previous step has completed).
    pub fn fill(&self, table: &PagedTable, ids: &[u32]) -> Result<()> {
        ensure!(
            ids.len() <= self.capacity,
            "{} rows exceed staging capacity {}",
            ids.len(),
            self.capacity
        );
        let n = ids.len();
        let cb = table.codes_bytes();
        let gb = table.group_bytes();
        // SAFETY: shared-storage buffers owned by this struct; the caller
        // guarantees the GPU is idle on them, and the ranges are in bounds.
        let (codes, scales, biases) = unsafe {
            (
                core::slice::from_raw_parts_mut(self.codes.contents_ptr(), n * cb),
                core::slice::from_raw_parts_mut(self.scales.contents_ptr(), n * gb),
                core::slice::from_raw_parts_mut(self.biases.contents_ptr(), n * gb),
            )
        };
        table.gather(ids, codes, scales, biases)
    }

    /// The first `rows` staged rows as a table the gather kernel can read
    /// with `seq_ids(rows)`.
    pub fn as_table(&self, rows: usize) -> Result<QuantWeights> {
        let words = self.codes.shape()[1];
        let groups = self.scales.shape()[1];
        Ok(QuantWeights {
            codes: self.codes.view(0, &[rows, words])?,
            scales: self.scales.view(0, &[rows, groups])?,
            biases: self.biases.view(0, &[rows, groups])?,
            group_size: self.group_size,
            bits: PagedTable::BITS,
        })
    }

    pub fn seq_ids(&self, rows: usize) -> Result<Tensor> {
        self.seq_ids.view(0, &[rows])
    }
}

/// Prefix of the checkpoint names of the table shards under one PLE module.
pub fn shard_bases(prefix: &str, shards: usize) -> Vec<String> {
    (0..shards)
        .map(|i| format!("{prefix}ple_embedding.ngram_embedding.shard_{i}"))
        .collect()
}

/// Which storage the loader should use for the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NgramStorage {
    /// Rows stay in the checkpoint files and are read on demand (default).
    #[default]
    Paged,
    /// The whole table is uploaded to GPU memory.
    Resident,
}

impl std::str::FromStr for NgramStorage {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "paged" => Ok(Self::Paged),
            "resident" => Ok(Self::Resident),
            other => anyhow::bail!(
                "unknown n-gram table storage {other:?}; use paged or resident"
            ),
        }
    }
}

#[allow(dead_code)]
fn _assert_path_in_scope(_: &Path) {}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/ngram.rs"]
mod tests;
