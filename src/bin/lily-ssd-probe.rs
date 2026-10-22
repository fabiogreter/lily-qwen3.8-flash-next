//! Times lily's expert read path against the drive, without loading the
//! model, to calibrate the timed replay's SSD model
//! (`cache_replay::SsdTiming`, `lily-cache-replay --timed`): the checkpoint's
//! expert shards are opened by `ExpertStore` (config and safetensors headers
//! only, no weights, no Metal) and read the way the cache's service thread
//! reads a resolution's misses, `expert_cache::read_experts`: nine positioned
//! reads per expert over the read pool (`LOAD_THREADS` threads plus the
//! calling one, `run_claimed`), here into plain heap buffers touched before
//! any timing instead of the slab.
//!
//! The reads must come from the drive. The probe sets `F_NOCACHE` on the
//! store's descriptors (`--buffered` keeps the server's cached reads), draws
//! experts uniformly at random over every (layer, expert) without drawing
//! one twice in a phase (one permutation runs on across phases, so the next
//! phase reads experts the previous ones did not, until all have been read),
//! and checks with `mincore` how many of the experts' pages were resident
//! before the run, after a re-read of experts just read (the cache check),
//! and after the run.
//!
//! The phases:
//!
//! 1. Demand batches: `k` experts per `read_experts` call, a call every
//!    `--period-ms`, for `k` in 1, 2, 4, 8, 16, 32 (interleaved): wall time
//!    per batch and per expert.
//! 2. Contention: a background prefetcher of `D` lanes, each its own thread
//!    with its own pool of eight (so an expert's nine regions are read at
//!    once, as the read pool would), keeps `D` experts in flight; it runs
//!    alone first, then demand batches of `k` in 1, 4, 8 go through the read
//!    pool every `--period-ms` while it runs. `D` in 1, 2, 4, 8, 16.
//! 3. Sustained: every reader of the pool back to back for `--sustain-secs`
//!    on fresh experts: one `run_claimed` over the regions of every expert
//!    in turn, as `read_experts` would, each thread into its own region
//!    buffer, so no batch boundary idles a reader.
//!
//! Prints a table; `--json-out` writes every number, raw batch times
//! included.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, ensure};
use clap::Parser;
use lily::qwen4exp::config::Qwen4ExpConfig;
use lily::qwen4exp::expert_cache::{
    LOAD_THREADS, read_experts, read_pool, run_claimed,
};
use lily::qwen4exp::expert_store::{ExpertStore, REGIONS};
use lily::safetensors::Checkpoint;
use serde::Serialize;

#[derive(Parser)]
#[command(
    name = "lily-ssd-probe",
    about = "Times the expert read path against the SSD"
)]
struct Cli {
    /// The checkpoint directory (`config.json` and the safetensors shards).
    #[arg(long)]
    model: PathBuf,
    /// Write every number as JSON here.
    #[arg(long)]
    json_out: Option<PathBuf>,
    /// A smoke run: few batches, short windows.
    #[arg(long)]
    quick: bool,
    /// Read through the page cache as the server does (no `F_NOCACHE`):
    /// cold only where `mincore` says the pages were not resident.
    #[arg(long)]
    buffered: bool,
    /// Seed of the expert draws.
    #[arg(long, default_value_t = 0x55d_5eed)]
    seed: u64,
    /// One demand batch is started this often (back to back when a batch
    /// takes longer), as decode's misses come a few layers apart.
    #[arg(long, default_value_t = 5.0)]
    period_ms: f64,
    /// Phase 1 batches per `k` (default 300, `--quick` 8).
    #[arg(long)]
    reps: Option<usize>,
    /// Phase 2 demand batches per (`D`, `k`) (default 150, `--quick` 6).
    #[arg(long)]
    cell_reps: Option<usize>,
    /// Phase 3 length, seconds (default 5, `--quick` 0.5).
    #[arg(long)]
    sustain_secs: Option<f64>,
}

/// Experts per demand batch in phase 1.
const DEMAND_KS: [usize; 6] = [1, 2, 4, 8, 16, 32];
/// Background experts in flight in phase 2.
const BACKGROUND_DS: [usize; 5] = [1, 2, 4, 8, 16];
/// Experts per demand batch in phase 2.
const CONTENTION_KS: [usize; 3] = [1, 4, 8];

/// SplitMix64, for the seeded draws.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`), by rejection, without modulo bias.
    fn below(&mut self, n: u64) -> u64 {
        let zone = u64::MAX - u64::MAX % n;
        loop {
            let x = self.next_u64();
            if x < zone {
                return x % n;
            }
        }
    }
}

/// Uniform draws of (layer, expert) without repeats within a phase: a
/// shuffled permutation of every pair, read on across phases and shuffled
/// anew when it runs out; a pair the current phase already drew is skipped.
/// Once a phase has drawn every pair it repeats them, and counts it.
struct Sampler {
    experts: usize,
    perm: Vec<u32>,
    cursor: usize,
    rng: Rng,
    seen: Vec<bool>,
    drawn: usize,
    repeats: u64,
}

impl Sampler {
    fn new(layers: usize, experts: usize, seed: u64) -> Self {
        let total = layers * experts;
        let mut sampler = Self {
            experts,
            perm: (0..total as u32).collect(),
            cursor: 0,
            rng: Rng(seed),
            seen: vec![false; total],
            drawn: 0,
            repeats: 0,
        };
        sampler.shuffle();
        sampler
    }

    fn shuffle(&mut self) {
        for i in (1..self.perm.len()).rev() {
            let j = self.rng.below(i as u64 + 1) as usize;
            self.perm.swap(i, j);
        }
        self.cursor = 0;
    }

    /// Starts a phase: every pair may be drawn once again.
    fn begin_phase(&mut self) {
        self.seen.fill(false);
        self.drawn = 0;
        self.repeats = 0;
    }

    /// Draws made in this phase after it had drawn every pair.
    fn repeats(&self) -> u64 {
        self.repeats
    }

    fn next(&mut self) -> (usize, usize) {
        loop {
            if self.cursor == self.perm.len() {
                self.shuffle();
            }
            let id = self.perm[self.cursor] as usize;
            self.cursor += 1;
            let pair = (id / self.experts, id % self.experts);
            if self.drawn == self.perm.len() {
                self.repeats += 1;
                return pair;
            }
            if !self.seen[id] {
                self.seen[id] = true;
                self.drawn += 1;
                return pair;
            }
        }
    }
}

/// The nearest-rank `p`-th percentile (0 < `p` <= 100) of `sorted`
/// (ascending, not empty): the smallest sample at or above `p`% of them.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// A distribution of times, ms.
#[derive(Serialize, Clone, Copy, Debug, PartialEq)]
struct Summary {
    n: usize,
    mean: f64,
    p50: f64,
    p90: f64,
    max: f64,
    min: f64,
}

/// `None` for no samples.
fn summarize(samples: &[f64]) -> Option<Summary> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(Summary {
        n: sorted.len(),
        mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
        p50: percentile(&sorted, 50.0),
        p90: percentile(&sorted, 90.0),
        max: sorted[sorted.len() - 1],
        min: sorted[0],
    })
}

/// Waits until `next`, then moves it one `period` on from the later of it
/// and now.
fn pace(next: &mut Instant, period: Duration) {
    let now = Instant::now();
    if *next > now {
        std::thread::sleep(*next - now);
    }
    *next = (*next).max(Instant::now()) + period;
}

/// `n` buffers of `bytes`, every page written, so no read is charged a
/// page fault on its destination.
fn buffers(n: usize, bytes: usize) -> Vec<Vec<u8>> {
    (0..n).map(|_| vec![0xA5u8; bytes]).collect()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// How much of some experts sat in the unified buffer cache.
#[derive(Serialize, Clone, Copy, Debug, Default)]
struct Residency {
    experts: usize,
    /// Experts with at least one resident page.
    touched: usize,
    pages: u64,
    resident_pages: u64,
}

/// A read-only mapping of a whole shard, unmapped on drop.
struct Mapping {
    addr: *mut libc::c_void,
    len: usize,
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: a mapping this probe made and nothing else references.
        unsafe { libc::munmap(self.addr, self.len) };
    }
}

/// Counts the resident pages of `experts` with `mincore` over temporary
/// mappings of their shards, unmapped before it returns, so no mapping of
/// the shards exists while reads are timed. Mapping does not fault pages in.
fn residency(
    store: &ExpertStore,
    experts: impl IntoIterator<Item = (usize, usize)>,
) -> Result<Residency> {
    const MINCORE_INCORE: u8 = 1;
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    ensure!(page > 0, "page size unknown");
    let mut maps: HashMap<usize, Mapping> = HashMap::new();
    let mut vec: Vec<u8> = Vec::new();
    let mut out = Residency::default();
    for (layer, expert) in experts {
        let mut resident = 0;
        for region in store.slice(layer, expert).regions {
            let map = match maps.entry(region.shard) {
                std::collections::hash_map::Entry::Occupied(m) => m.into_mut(),
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(map_shard(store.shard_path(region.shard))?)
                }
            };
            let start = region.offset / page * page;
            let end = (region.offset + region.len as u64).div_ceil(page) * page;
            let end = end.min(map.len as u64);
            let pages = ((end - start) / page) as usize;
            vec.clear();
            vec.resize(pages, 0);
            // SAFETY: `start..end` lies inside the mapping; `vec` holds one
            // byte per page.
            let rc = unsafe {
                libc::mincore(
                    (map.addr as *mut u8).add(start as usize) as _,
                    (end - start) as usize,
                    vec.as_mut_ptr() as _,
                )
            };
            ensure!(rc == 0, "mincore: {}", std::io::Error::last_os_error());
            out.pages += pages as u64;
            resident += vec.iter().filter(|&&v| v & MINCORE_INCORE != 0).count() as u64;
        }
        out.experts += 1;
        out.resident_pages += resident;
        out.touched += usize::from(resident > 0);
    }
    Ok(out)
}

fn map_shard(path: &Path) -> Result<Mapping> {
    use std::os::fd::AsRawFd as _;
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let len = file.metadata()?.len() as usize;
    // SAFETY: a fresh shared read-only mapping of a file we opened; the
    // descriptor may close once mapped.
    let addr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    ensure!(
        addr != libc::MAP_FAILED,
        "mapping {}: {}",
        path.display(),
        std::io::Error::last_os_error()
    );
    Ok(Mapping { addr, len })
}

impl Residency {
    fn describe(&self) -> String {
        format!(
            "{} of {} experts have resident pages ({:.2}% of {} pages)",
            self.touched,
            self.experts,
            100.0 * self.resident_pages as f64 / self.pages.max(1) as f64,
            self.pages
        )
    }
}

#[derive(Serialize)]
struct DemandCell {
    k: usize,
    /// Wall time per batch, ms.
    batch: Option<Summary>,
    /// The same divided by `k`.
    per_expert: Option<Summary>,
    /// `k` experts' bytes over the median batch time.
    gb_per_s_at_p50: f64,
    batch_ms: Vec<f64>,
}

impl DemandCell {
    fn new(k: usize, batch_ms: Vec<f64>, bytes: usize) -> Self {
        let per: Vec<f64> = batch_ms.iter().map(|t| t / k as f64).collect();
        let batch = summarize(&batch_ms);
        let gb_per_s_at_p50 =
            batch.map_or(0.0, |s| (k * bytes) as f64 / (s.p50 * 1e-3) / 1e9);
        Self { k, batch, per_expert: summarize(&per), gb_per_s_at_p50, batch_ms }
    }
}

#[derive(Serialize)]
struct CacheCheck {
    experts: usize,
    first: Option<Summary>,
    reread: Option<Summary>,
    resident_after: Residency,
    first_ms: Vec<f64>,
    reread_ms: Vec<f64>,
}

/// The background's work in one window.
#[derive(Serialize, Clone, Copy)]
struct Background {
    secs: f64,
    experts: usize,
    experts_per_s: f64,
    gb_per_s: f64,
    /// One expert's nine reads, ms.
    latency: Option<Summary>,
}

impl Background {
    /// From each lane's `(end, ms)` records, the experts that ended in
    /// `from..to` (seconds on the probe's clock).
    fn window(records: &[(f64, f64)], from: f64, to: f64, bytes: usize) -> Self {
        let lat: Vec<f64> = records
            .iter()
            .filter(|&&(end, _)| end >= from && end < to)
            .map(|&(_, ms)| ms)
            .collect();
        let secs = to - from;
        let rate = lat.len() as f64 / secs;
        Self {
            secs,
            experts: lat.len(),
            experts_per_s: rate,
            gb_per_s: rate * bytes as f64 / 1e9,
            latency: summarize(&lat),
        }
    }
}

#[derive(Serialize)]
struct Contention {
    d: usize,
    alone: Background,
    with_demand: Background,
    demand: Vec<DemandCell>,
    repeats: u64,
}

#[derive(Serialize)]
struct Sustained {
    readers: usize,
    secs: f64,
    bytes: u64,
    experts: f64,
    experts_per_s: f64,
    gb_per_s: f64,
    /// Ran out of distinct experts before the time was up.
    exhausted: bool,
}

#[derive(Serialize)]
struct Report {
    model: PathBuf,
    layers: usize,
    experts: usize,
    slice_bytes: usize,
    region_lens: [usize; REGIONS],
    read_threads: usize,
    nocache: bool,
    seed: u64,
    quick: bool,
    period_ms: f64,
    /// Another lily process was running (it may read the SSD), from the
    /// instance lock's record; `None`: the record could not be read.
    other_lily_running: Option<bool>,
    resident_before: Residency,
    demand: Vec<DemandCell>,
    demand_repeats: u64,
    cache_check: CacheCheck,
    contention: Vec<Contention>,
    sustained: Sustained,
    resident_after: Residency,
    total_secs: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let started = Instant::now();
    let reps = cli.reps.unwrap_or(if cli.quick { 8 } else { 300 });
    let cell_reps = cli.cell_reps.unwrap_or(if cli.quick { 6 } else { 150 });
    let sustain = cli.sustain_secs.unwrap_or(if cli.quick { 0.5 } else { 5.0 });
    let check_n = if cli.quick { 8 } else { 32 };
    let (warmup, alone_secs) = if cli.quick { (0.05, 0.15) } else { (0.2, 0.6) };
    ensure!(cli.period_ms >= 0.0 && sustain > 0.0, "negative period or empty sustain");
    let period = Duration::from_secs_f64(cli.period_ms * 1e-3);

    // Another lily (the service) may be reading the same drive.
    let other_lily_running = other_lily_running();
    if other_lily_running != Some(false) {
        eprintln!(
            "warning: {}",
            match other_lily_running {
                Some(_) => "another lily is running; its reads share the drive",
                None => "could not read the instance lock's record for another lily",
            }
        );
    }

    let config = Qwen4ExpConfig::from_model_dir(&cli.model)?;
    let ckpt = Checkpoint::open(&cli.model)?;
    let store = ExpertStore::open(&ckpt, &config)?;
    drop(ckpt);
    store.set_nocache(!cli.buffered)?;
    let (layers, experts) = (store.layers(), store.experts());
    let bytes = store.slice_bytes();
    let all = || (0..layers).flat_map(|l| (0..experts).map(move |e| (l, e)));
    println!(
        "lily-ssd-probe: {} ({layers} layers x {experts} experts, {:.2} MB per expert in {REGIONS} reads), {}, read pool {LOAD_THREADS} threads + caller",
        cli.model.display(),
        bytes as f64 / 1e6,
        if cli.buffered { "buffered (page cache on)" } else { "F_NOCACHE" },
    );
    let resident_before = residency(&store, all())?;
    println!("page cache before: {}", resident_before.describe());

    let pool = read_pool()?;
    let sampler = Mutex::new(Sampler::new(layers, experts, cli.seed));
    let draw = || sampler.lock().unwrap_or_else(|e| e.into_inner()).next();
    let begin = || sampler.lock().unwrap_or_else(|e| e.into_inner()).begin_phase();
    let repeats = || sampler.lock().unwrap_or_else(|e| e.into_inner()).repeats();
    let k_max = DEMAND_KS.iter().chain(&CONTENTION_KS).copied().max().unwrap_or(1);
    let mut bufs = buffers(k_max, bytes);
    let mut batch: Vec<(usize, usize)> = Vec::with_capacity(k_max);

    // 1. Demand batches, the `k` values interleaved and rotated per round.
    begin();
    let mut times: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); DEMAND_KS.len()];
    let mut next = Instant::now();
    for rep in 0..reps {
        for j in 0..DEMAND_KS.len() {
            let i = (j + rep) % DEMAND_KS.len();
            batch.clear();
            batch.extend((0..DEMAND_KS[i]).map(|_| draw()));
            pace(&mut next, period);
            let t = Instant::now();
            read_experts(&pool, &store, &batch, &mut bufs)?;
            times[i].push(ms(t.elapsed()));
        }
    }
    let demand: Vec<DemandCell> = DEMAND_KS
        .iter()
        .zip(times)
        .map(|(&k, t)| DemandCell::new(k, t, bytes))
        .collect();
    let demand_repeats = repeats();
    println!(
        "\n1. demand batches: one read_experts call per batch, one every {} ms",
        cli.period_ms
    );
    print_demand(&demand, "");
    warn_repeats(demand_repeats);

    // The cache check: fresh experts read one at a time, then again.
    begin();
    let picks: Vec<(usize, usize)> = (0..check_n).map(|_| draw()).collect();
    let mut pass = || -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(picks.len());
        for &pick in &picks {
            pace(&mut next, period);
            let t = Instant::now();
            read_experts(&pool, &store, &[pick], &mut bufs)?;
            out.push(ms(t.elapsed()));
        }
        Ok(out)
    };
    let first_ms = pass()?;
    let reread_ms = pass()?;
    let cache_check = CacheCheck {
        experts: picks.len(),
        first: summarize(&first_ms),
        reread: summarize(&reread_ms),
        resident_after: residency(&store, picks.iter().copied())?,
        first_ms,
        reread_ms,
    };
    println!("\ncache check: {} experts read one at a time, then again", check_n);
    println!("            {}", Columns::HEADER);
    println!("  first     {}", Columns(cache_check.first));
    println!("  re-read   {}", Columns(cache_check.reread));
    println!("  after the re-read: {}", cache_check.resident_after.describe());

    // 2. Contention.
    println!(
        "\n2. contention: D background experts in flight (own threads, nine reads each at once), demand every {} ms through the read pool",
        cli.period_ms
    );
    let mut contention = Vec::new();
    for d in BACKGROUND_DS {
        begin();
        let origin = Instant::now();
        let stop = AtomicBool::new(false);
        let ready = std::sync::Barrier::new(d + 1);
        let mut cell_times: Vec<Vec<f64>> =
            vec![Vec::with_capacity(cell_reps); CONTENTION_KS.len()];
        let (lanes, window) = std::thread::scope(|s| -> Result<_> {
            let handles: Vec<_> = (0..d)
                .map(|_| {
                    s.spawn(|| -> Result<Vec<(f64, f64)>> {
                        let lane = rayon::ThreadPoolBuilder::new()
                            .num_threads(REGIONS - 1)
                            .build();
                        let mut buf = buffers(1, bytes);
                        let mut records = Vec::with_capacity(1 << 14);
                        ready.wait();
                        let lane = lane?;
                        while !stop.load(Ordering::Relaxed) {
                            let pick = draw();
                            let t = Instant::now();
                            read_experts(&lane, &store, &[pick], &mut buf)?;
                            let end = Instant::now();
                            records.push((
                                end.duration_since(origin).as_secs_f64(),
                                ms(end - t),
                            ));
                        }
                        Ok(records)
                    })
                })
                .collect();
            ready.wait();
            std::thread::sleep(Duration::from_secs_f64(warmup));
            let a0 = origin.elapsed().as_secs_f64();
            std::thread::sleep(Duration::from_secs_f64(alone_secs));
            let a1 = origin.elapsed().as_secs_f64();
            let mut next = Instant::now();
            let mut demand = || -> Result<()> {
                for rep in 0..cell_reps {
                    for j in 0..CONTENTION_KS.len() {
                        let i = (j + rep) % CONTENTION_KS.len();
                        batch.clear();
                        batch.extend((0..CONTENTION_KS[i]).map(|_| draw()));
                        pace(&mut next, period);
                        let t = Instant::now();
                        read_experts(&pool, &store, &batch, &mut bufs)?;
                        cell_times[i].push(ms(t.elapsed()));
                    }
                }
                Ok(())
            };
            let outcome = demand();
            let w1 = origin.elapsed().as_secs_f64();
            stop.store(true, Ordering::Relaxed);
            let mut records = Vec::new();
            for h in handles {
                let lane =
                    h.join().map_err(|_| anyhow::anyhow!("a lane panicked"))??;
                records.extend(lane);
            }
            outcome?;
            Ok((records, (a0, a1, w1)))
        })?;
        let (a0, a1, w1) = window;
        let row = Contention {
            d,
            alone: Background::window(&lanes, a0, a1, bytes),
            with_demand: Background::window(&lanes, a1, w1, bytes),
            demand: CONTENTION_KS
                .iter()
                .zip(cell_times)
                .map(|(&k, t)| DemandCell::new(k, t, bytes))
                .collect(),
            repeats: repeats(),
        };
        print_contention(&row);
        contention.push(row);
    }

    // 3. Sustained: every pool reader back to back on fresh experts, each
    // thread into its own region buffer, until the time is up.
    begin();
    let list: Vec<(usize, usize)> = (0..layers * experts).map(|_| draw()).collect();
    let max_region = store.region_lens().into_iter().max().unwrap_or(0);
    let mut thread_bufs = buffers(LOAD_THREADS + 1, max_region);
    let bases: Vec<usize> =
        thread_bufs.iter_mut().map(|b| b.as_mut_ptr() as usize).collect();
    let lens = store.region_lens();
    let read_bytes = AtomicU64::new(0);
    let last_end = AtomicU64::new(0);
    let t0 = Instant::now();
    let deadline = t0 + Duration::from_secs_f64(sustain);
    run_claimed(&pool, list.len() * REGIONS, |k| {
        if Instant::now() >= deadline {
            return Ok(());
        }
        let (i, r) = (k / REGIONS, k % REGIONS);
        let me = rayon::current_thread_index().unwrap_or(LOAD_THREADS);
        // SAFETY: buffer `me` is `max_region` long and used only by the
        // thread with that pool index (the caller has none).
        let dst =
            unsafe { core::slice::from_raw_parts_mut(bases[me] as *mut u8, lens[r]) };
        store.read_region(list[i].0, list[i].1, r, dst)?;
        read_bytes.fetch_add(lens[r] as u64, Ordering::Relaxed);
        last_end.fetch_max(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    })?;
    let secs = last_end.load(Ordering::Relaxed) as f64 * 1e-9;
    let total = read_bytes.load(Ordering::Relaxed);
    let sustained = Sustained {
        readers: LOAD_THREADS + 1,
        secs,
        bytes: total,
        experts: total as f64 / bytes as f64,
        experts_per_s: total as f64 / bytes as f64 / secs,
        gb_per_s: total as f64 / secs / 1e9,
        exhausted: secs < sustain * 0.99,
    };
    drop(thread_bufs);
    println!(
        "\n3. sustained: {} readers for {:.2} s: {:.0} experts, {:.0} experts/s ({:.3} ms each), {:.2} GB/s{}",
        sustained.readers,
        sustained.secs,
        sustained.experts,
        sustained.experts_per_s,
        1e3 / sustained.experts_per_s,
        sustained.gb_per_s,
        if sustained.exhausted {
            " (every expert read before the time was up)"
        } else {
            ""
        }
    );

    let resident_after = residency(&store, all())?;
    println!("\npage cache after: {}", resident_after.describe());
    let report = Report {
        model: cli.model.clone(),
        layers,
        experts,
        slice_bytes: bytes,
        region_lens: store.region_lens(),
        read_threads: LOAD_THREADS,
        nocache: !cli.buffered,
        seed: cli.seed,
        quick: cli.quick,
        period_ms: cli.period_ms,
        other_lily_running,
        resident_before,
        demand,
        demand_repeats,
        cache_check,
        contention,
        sustained,
        resident_after,
        total_secs: started.elapsed().as_secs_f64(),
    };
    println!("total {:.1} s", report.total_secs);
    if let Some(path) = &cli.json_out {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("writing {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

/// Whether another lily runs, judged from the record the instance lock's
/// holder writes into the lock file (`lily::instance::Holder`) without
/// touching the lock: even a shared lock held for a moment would make a
/// lily starting in that moment (`LOCK_EX | LOCK_NB`) refuse with exit 75.
/// The record outlives its process (the file is never removed), so the
/// recorded pid must still be alive (`kill(pid, 0)`) and, as far as the
/// probe can see, still run the recorded binary (`proc_pidpath`; a reused
/// pid runs another one). No file or no record: no lily runs (a holder
/// writes its record right after locking). `None` when the file cannot be
/// read.
fn other_lily_running() -> Option<bool> {
    let text = match std::fs::read_to_string(lily::instance::default_path()) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some(false),
        Err(_) => return None,
    };
    let Some(holder) = lily::instance::Holder::parse(&text) else {
        return Some(false);
    };
    let Ok(pid) = libc::pid_t::try_from(holder.pid) else {
        return Some(false);
    };
    // SAFETY: signal 0 only checks that the process exists.
    if unsafe { libc::kill(pid, 0) } != 0
        && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
    {
        return Some(false);
    }
    // Alive (or not ours to signal): compare its binary when both are known.
    let mut path = vec![0u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
    // SAFETY: the buffer is `path.len()` bytes long.
    let n =
        unsafe { libc::proc_pidpath(pid, path.as_mut_ptr().cast(), path.len() as u32) };
    if n <= 0 || holder.binary.is_empty() {
        return Some(true);
    }
    let running = PathBuf::from(String::from_utf8_lossy(&path[..n as usize]).as_ref());
    let recorded = PathBuf::from(&holder.binary);
    let real = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_owned());
    Some(real(&running) == real(&recorded))
}

/// A summary's columns, `-` for none.
struct Columns(Option<Summary>);

impl Columns {
    const HEADER: &'static str = "     p50      p90      max     mean";
}

impl std::fmt::Display for Columns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(s) => {
                write!(f, "{:8.3} {:8.3} {:8.3} {:8.3}", s.p50, s.p90, s.max, s.mean)
            }
            None => write!(f, "{:>8} {:>8} {:>8} {:>8}", "-", "-", "-", "-"),
        }
    }
}

fn print_demand(cells: &[DemandCell], indent: &str) {
    println!(
        "{indent}   k  batches  batch ms: {}  | per expert ms: {}  GB/s@p50",
        Columns::HEADER,
        Columns::HEADER
    );
    for c in cells {
        println!(
            "{indent}{:4} {:8}            {}  |                {}  {:7.2}",
            c.k,
            c.batch_ms.len(),
            Columns(c.batch),
            Columns(c.per_expert),
            c.gb_per_s_at_p50
        );
    }
}

fn print_contention(row: &Contention) {
    let bg = |b: &Background| {
        format!(
            "{:6.0} experts/s {:5.2} GB/s, expert ms p50 {} p90 {}",
            b.experts_per_s,
            b.gb_per_s,
            b.latency.map_or("-".into(), |s| format!("{:.2}", s.p50)),
            b.latency.map_or("-".into(), |s| format!("{:.2}", s.p90)),
        )
    };
    println!("  D = {}", row.d);
    println!("    background alone:       {}", bg(&row.alone));
    println!("    background with demand: {}", bg(&row.with_demand));
    print_demand(&row.demand, "    ");
    warn_repeats(row.repeats);
}

fn warn_repeats(repeats: u64) {
    if repeats > 0 {
        println!(
            "    note: {repeats} draws repeated an expert already read in this phase (the checkpoint has too few experts for the phase)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_phase_draws_every_pair_once_before_repeating() {
        let mut s = Sampler::new(3, 5, 7);
        for phase in 0..4 {
            s.begin_phase();
            let mut seen = std::collections::HashSet::new();
            for _ in 0..15 {
                let (l, e) = s.next();
                assert!(l < 3 && e < 5);
                assert!(seen.insert((l, e)), "phase {phase} drew ({l}, {e}) twice");
            }
            assert_eq!(s.repeats(), 0);
            // A phase with every pair drawn repeats, and says so.
            if phase == 3 {
                s.next();
                assert_eq!(s.repeats(), 1);
            }
        }
    }

    #[test]
    fn phases_continue_the_permutation() {
        // Two phases of half the pairs each read disjoint halves.
        let mut s = Sampler::new(4, 4, 1);
        s.begin_phase();
        let first: std::collections::HashSet<_> = (0..8).map(|_| s.next()).collect();
        s.begin_phase();
        let second: std::collections::HashSet<_> = (0..8).map(|_| s.next()).collect();
        assert_eq!(first.len(), 8);
        assert!(first.is_disjoint(&second));
    }

    #[test]
    fn draws_are_roughly_uniform() {
        let mut s = Sampler::new(2, 8, 3);
        let mut first = [0u32; 16];
        for _ in 0..4000 {
            s.begin_phase();
            s.next();
            let (l, e) = s.next();
            first[l * 8 + e] += 1;
        }
        // 250 expected per pair.
        assert!(first.iter().all(|&n| (150..350).contains(&n)), "{first:?}");
    }

    #[test]
    fn nearest_rank_percentiles() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&v, 50.0), 5.0);
        assert_eq!(percentile(&v, 90.0), 9.0);
        assert_eq!(percentile(&v, 91.0), 10.0);
        assert_eq!(percentile(&v, 100.0), 10.0);
        assert_eq!(percentile(&v, 0.1), 1.0);
        assert_eq!(percentile(&[4.0], 50.0), 4.0);
        let s = summarize(&[3.0, 1.0, 2.0, 10.0]).expect("summary");
        assert_eq!(
            (s.n, s.p50, s.p90, s.max, s.min, s.mean),
            (4, 2.0, 10.0, 10.0, 1.0, 4.0)
        );
        assert!(summarize(&[]).is_none());
    }

    #[test]
    fn background_windows_count_the_experts_that_ended_inside() {
        let records = [(0.5, 2.0), (1.0, 4.0), (1.5, 6.0), (2.0, 8.0)];
        let b = Background::window(&records, 1.0, 2.0, 1_000_000_000);
        assert_eq!(b.experts, 2);
        assert_eq!(b.experts_per_s, 2.0);
        assert_eq!(b.gb_per_s, 2.0);
        assert_eq!(b.latency.map(|s| s.p50), Some(4.0));
    }
}
