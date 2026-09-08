//! The hashed n-gram embedding table: host-side hashing (the host owns the
//! token history anyway) and two storage backends for the 32 GB table.
//!
//! * [`NgramTable::Resident`] keeps the quantized table in GPU memory and
//!   gathers rows by hashed id, as phase 1 did.
//! * [`NgramTable::Paged`] leaves the table in the checkpoint files and reads
//!   the 16 rows a token needs with `pread` into a small staging buffer, which
//!   the same gather kernel then dequantizes with sequential ids. The rows live
//!   in the page cache, evictable, instead of in wired GPU memory: on the
//!   128 GB machine this frees 32 GB for KV and session caches at no quality
//!   cost. A warm page-cache read is a few microseconds per row.

use std::fs::File;
use std::os::unix::fs::FileExt as _;
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::{self, Sender};

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
        ensure!(heads_per_ngram > 0 && sizes.len() == 2 * heads_per_ngram, "head count");
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
            let bigram = (t0 as u64).wrapping_mul(m[0]) ^ (s1 as u64).wrapping_mul(m[1]);
            let trigram = bigram ^ (s2 as u64).wrapping_mul(m[2]);
            for (j, (&size, &offset)) in self.sizes.iter().zip(&self.offsets).enumerate() {
                let mixed = if j / self.heads_per_ngram >= 1 { trigram } else { bigram };
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

/// One tensor's bytes: the file holding it and its offset there.
struct Region {
    file: Arc<File>,
    start: u64,
}

/// One contiguous row range of the table (its codes, scales and biases may
/// sit in different shard files).
struct ShardRegion {
    codes: Region,
    scales: Region,
    biases: Region,
    rows: usize,
}

/// The table's layout and files, shared with the reader threads.
struct TableInner {
    regions: Vec<ShardRegion>,
    /// `row_starts[i]` is the first global row of region `i`; one extra entry
    /// holds the total.
    row_starts: Vec<usize>,
    words: usize,
    groups: usize,
    group_size: usize,
}

/// A raw destination pointer handed to a reader thread. The batch that owns
/// the memory blocks until every task reports back, so the pointer outlives
/// its use.
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}

struct Task {
    ids: Vec<u32>,
    codes: SendPtr,
    scales: SendPtr,
    biases: SendPtr,
    done: Sender<Result<()>>,
}

/// The table left in the checkpoint files, read row by row on demand. A small
/// persistent pool of reader threads hides the latency of cold rows (a token
/// needs 16 rows from random places in 32 GB; one cold SSD read is ~100 us,
/// so serially a cold token costs milliseconds, in parallel a fraction).
pub struct PagedTable {
    inner: Arc<TableInner>,
    workers: Vec<Sender<Task>>,
}

const READER_THREADS: usize = 8;

impl PagedTable {
    pub const BITS: usize = 4;

    /// Opens the shards named `bases[i]` (each with `.weight`, `.scales` and
    /// `.biases`) in row order.
    pub fn open(ckpt: &Checkpoint, bases: &[String], group_size: usize) -> Result<Self> {
        ensure!(!bases.is_empty(), "n-gram table has no shards");
        let mut regions = Vec::with_capacity(bases.len());
        let mut row_starts = vec![0usize];
        let mut files: Vec<(std::path::PathBuf, Arc<File>)> = Vec::new();
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
                codes.shape.len() == 2 && scales.shape.len() == 2 && biases.shape == scales.shape,
                "{base}: table shard shapes must be [rows, k]"
            );
            let rows = codes.shape[0];
            ensure!(scales.shape[0] == rows, "{base}: scale rows differ from code rows");
            let w = codes.shape[1];
            let g = scales.shape[1];
            ensure!(words.is_none_or(|x| x == w), "{base}: code width differs");
            ensure!(groups.is_none_or(|x| x == g), "{base}: group count differs");
            words = Some(w);
            groups = Some(g);
            let mut open = |meta: &crate::safetensors::TensorMeta| -> Result<Region> {
                let file = match files.iter().find(|(p, _)| p == meta.shard()) {
                    Some((_, f)) => f.clone(),
                    None => {
                        let f = Arc::new(
                            File::open(meta.shard())
                                .with_context(|| format!("opening {}", meta.shard().display()))?,
                        );
                        files.push((meta.shard().to_path_buf(), f.clone()));
                        f
                    }
                };
                Ok(Region { file, start: meta.start() })
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
        let inner = Arc::new(TableInner { regions, row_starts, words, groups, group_size });
        let mut workers = Vec::with_capacity(READER_THREADS);
        for i in 0..READER_THREADS {
            let (tx, rx) = mpsc::channel::<Task>();
            let table = inner.clone();
            std::thread::Builder::new()
                .name(format!("ngram-reader-{i}"))
                .spawn(move || {
                    while let Ok(task) = rx.recv() {
                        let n = task.ids.len();
                        let (cb, gb) = (table.codes_bytes(), table.group_bytes());
                        // SAFETY: the batch owner sized these ranges for `n`
                        // rows and waits for `done` before touching them.
                        let result = unsafe {
                            table.read_rows(
                                &task.ids,
                                core::slice::from_raw_parts_mut(task.codes.0, n * cb),
                                core::slice::from_raw_parts_mut(task.scales.0, n * gb),
                                core::slice::from_raw_parts_mut(task.biases.0, n * gb),
                            )
                        };
                        let _ = task.done.send(result);
                    }
                })
                .context("spawning n-gram reader thread")?;
            workers.push(tx);
        }
        Ok(Self { inner, workers })
    }

    pub fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// Row width in elements.
    pub fn width(&self) -> usize {
        self.inner.words * (32 / Self::BITS)
    }

    pub fn group_size(&self) -> usize {
        self.inner.group_size
    }

    fn codes_bytes(&self) -> usize {
        self.inner.codes_bytes()
    }

    fn group_bytes(&self) -> usize {
        self.inner.group_bytes()
    }

    /// Reads the rows `ids` into row-major `codes`, `scales` and `biases`
    /// (each exactly `ids.len()` rows wide), spread over the reader threads.
    pub fn gather(
        &self,
        ids: &[u32],
        codes: &mut [u8],
        scales: &mut [u8],
        biases: &mut [u8],
    ) -> Result<()> {
        let (cb, gb) = (self.codes_bytes(), self.group_bytes());
        ensure!(
            codes.len() == ids.len() * cb && scales.len() == ids.len() * gb && biases.len() == ids.len() * gb,
            "staging buffers do not match {} rows",
            ids.len()
        );
        if ids.len() <= 2 {
            return self.inner.read_rows(ids, codes, scales, biases);
        }
        let parts = ids.len().min(self.workers.len());
        let per = ids.len().div_ceil(parts);
        let (done_tx, done_rx) = mpsc::channel();
        let mut sent = 0;
        let mut offset = 0;
        for (w, chunk) in ids.chunks(per).enumerate() {
            let n = chunk.len();
            let task = Task {
                ids: chunk.to_vec(),
                // SAFETY: disjoint sub-ranges of the caller's buffers.
                codes: SendPtr(unsafe { codes.as_mut_ptr().add(offset * cb) }),
                scales: SendPtr(unsafe { scales.as_mut_ptr().add(offset * gb) }),
                biases: SendPtr(unsafe { biases.as_mut_ptr().add(offset * gb) }),
                done: done_tx.clone(),
            };
            self.workers[w % self.workers.len()]
                .send(task)
                .map_err(|_| anyhow::anyhow!("n-gram reader thread is gone"))?;
            sent += 1;
            offset += n;
        }
        drop(done_tx);
        let mut first_error = None;
        for _ in 0..sent {
            match done_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    first_error.get_or_insert(e);
                }
                Err(_) => {
                    first_error.get_or_insert(anyhow::anyhow!("n-gram reader thread is gone"));
                }
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Streams every table byte through the page cache once so the first
    /// requests do not pay cold random reads. Costs the table's size in
    /// evictable page-cache memory and a few seconds of sequential I/O.
    pub fn preload(&self) -> Result<u64> {
        self.inner.preload()
    }
}

impl TableInner {
    fn rows(&self) -> usize {
        *self.row_starts.last().unwrap_or(&0)
    }

    fn codes_bytes(&self) -> usize {
        self.words * 4
    }

    fn group_bytes(&self) -> usize {
        self.groups * 2
    }

    fn read_rows(&self, ids: &[u32], codes: &mut [u8], scales: &mut [u8], biases: &mut [u8]) -> Result<()> {
        let (cb, gb) = (self.codes_bytes(), self.group_bytes());
        for (i, &id) in ids.iter().enumerate() {
            self.read_row(
                id as usize,
                &mut codes[i * cb..(i + 1) * cb],
                &mut scales[i * gb..(i + 1) * gb],
                &mut biases[i * gb..(i + 1) * gb],
            )?;
        }
        Ok(())
    }

    fn read_row(&self, row: usize, codes: &mut [u8], scales: &mut [u8], biases: &mut [u8]) -> Result<()> {
        let region_index = self.row_starts.partition_point(|&s| s <= row);
        ensure!(region_index >= 1 && row < self.rows(), "n-gram row {row} out of range");
        let region = &self.regions[region_index - 1];
        let local = (row - self.row_starts[region_index - 1]) as u64;
        region
            .codes
            .file
            .read_exact_at(codes, region.codes.start + local * self.codes_bytes() as u64)
            .context("reading n-gram codes")?;
        region
            .scales
            .file
            .read_exact_at(scales, region.scales.start + local * self.group_bytes() as u64)
            .context("reading n-gram scales")?;
        region
            .biases
            .file
            .read_exact_at(biases, region.biases.start + local * self.group_bytes() as u64)
            .context("reading n-gram biases")?;
        Ok(())
    }

    /// Streams every table byte through the page cache once so the first
    /// requests do not pay cold random reads. Costs the table's size in
    /// evictable page-cache memory and a few seconds of sequential I/O.
    fn preload(&self) -> Result<u64> {
        let mut buf = vec![0u8; 8 << 20];
        let mut total = 0u64;
        for region in &self.regions {
            for (part, len) in [
                (&region.codes, region.rows * self.codes_bytes()),
                (&region.scales, region.rows * self.group_bytes()),
                (&region.biases, region.rows * self.group_bytes()),
            ] {
                let mut done = 0usize;
                while done < len {
                    let n = buf.len().min(len - done);
                    part.file.read_exact_at(&mut buf[..n], part.start + done as u64)?;
                    done += n;
                }
                total += len as u64;
            }
        }
        Ok(total)
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
    pub fn new(ctx: &MetalContext, table: &PagedTable, capacity: usize) -> Result<Self> {
        ensure!(capacity > 0, "staging capacity must be nonzero");
        let seq: Vec<u32> = (0..capacity as u32).collect();
        Ok(Self {
            codes: Tensor::zeros(ctx, &[capacity, table.inner.words], DType::U32)?,
            scales: Tensor::zeros(ctx, &[capacity, table.inner.groups], DType::BF16)?,
            biases: Tensor::zeros(ctx, &[capacity, table.inner.groups], DType::BF16)?,
            seq_ids: Tensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&seq),
                &[capacity],
                DType::U32,
            )?,
            group_size: table.inner.group_size,
            capacity,
        })
    }

    /// Reads rows `ids` from `table` into the staging buffers. The GPU must
    /// not be reading them (the previous step has completed).
    pub fn fill(&self, table: &PagedTable, ids: &[u32]) -> Result<()> {
        ensure!(ids.len() <= self.capacity, "{} rows exceed staging capacity {}", ids.len(), self.capacity);
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
            other => anyhow::bail!("unknown n-gram table storage {other:?}; use paged or resident"),
        }
    }
}

#[allow(dead_code)]
fn _assert_path_in_scope(_: &Path) {}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/ngram.rs"]
mod tests;
