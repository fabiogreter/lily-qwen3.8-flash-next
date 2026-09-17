//! The on-disk session tier: evicted sessions are written to files and read
//! back when a later prompt shares their prefix, so a long conversation that
//! fell out of GPU memory resumes in the time it takes to read a few
//! gigabytes instead of the minute it takes to recompute them.
//!
//! One directory per session under `<root>/<format>/`: `meta.json` (tokens,
//! checkpoint positions, sizes, last use), `prefix.bin` (the per-token caches
//! for every token, as the model's `write_prefix` lays them out) and one
//! `ckpt-<pos>.bin` per resumable position (the model's snapshot layout, the
//! live end included). Entries are evicted least-recently-used under a byte
//! budget and expire after a maximum age since their last use (so a quiet
//! machine does not keep 100 GB of stale sessions); the index is rebuilt from
//! the meta files at startup, so the tier survives restarts. Entries whose
//! format tag differs are left alone (they belong to another model or layout)
//! but never read.
//!
//! A **durable** entry is a materialised prefix boundary rather than an
//! evicted session (see the session module). It is written the same way,
//! with the boundary as its live end, and differs only in lifecycle: a hit
//! never consumes it, and at most [`DURABLE_MAX_ENTRIES`] exist at a time,
//! the least recently used going first, because every shared prefix an agent
//! client ever sent is a candidate and only the ones still being hit matter.
//! Otherwise it ages and trims like any entry, which is what retires a
//! preamble once the client's prompt has changed.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, ensure};
use serde::{Deserialize, Serialize};

use super::session::CachedImage;

/// Free space to leave on the volume after a write.
const FREE_SPACE_MARGIN: u64 = 8 << 30;
const META: &str = "meta.json";
const PREFIX: &str = "prefix.bin";
/// Most durable prefix entries kept at a time; the least recently used goes
/// when one more is stored. Sixteen covers a handful of agent clients and
/// projects each with a preamble or two, while keeping what a run of
/// nondeterministic prompts could pile up bounded.
pub const DURABLE_MAX_ENTRIES: usize = 16;

#[derive(Serialize, Deserialize, Clone)]
struct Meta {
    format: String,
    tokens: Vec<u32>,
    /// Ascending positions with a checkpoint file; always includes `tokens.len()`.
    checkpoints: Vec<usize>,
    bytes: u64,
    /// Seconds since the epoch of the last store or hit.
    last_used: u64,
    cache_key: Option<String>,
    /// A materialised prefix boundary rather than an evicted session. Meta
    /// files written before the field existed load as `false`; the format
    /// tag stays, since the cache bytes are laid out the same.
    #[serde(default)]
    durable: bool,
    /// The images among `tokens` (position, length, grid, pixel digest),
    /// which a lookup matches alongside the tokens. Meta files written
    /// before the field existed load as text-only lineages, which is what
    /// they are; the cache bytes are the same, so the format tag stays.
    #[serde(default)]
    images: Vec<CachedImage>,
}

/// One persisted session (its token list stays in memory for prefix matching).
pub struct DiskEntry {
    pub id: String,
    pub tokens: Vec<u32>,
    pub checkpoints: Vec<usize>,
    pub bytes: u64,
    pub last_used: u64,
    pub cache_key: Option<String>,
    /// Whether this is a durable prefix entry (never consumed by a hit).
    pub durable: bool,
    /// The images among `tokens`, matched alongside them on lookup.
    pub images: Vec<CachedImage>,
}

/// What a store call describes about the entry it is about to write.
#[derive(Clone, Copy)]
struct NewEntry<'a> {
    tokens: &'a [u32],
    images: &'a [CachedImage],
    cache_key: Option<&'a str>,
    checkpoints: &'a [usize],
    durable: bool,
}

pub struct DiskStore {
    dir: PathBuf,
    format: String,
    budget: u64,
    /// Seconds an entry may go unused before it is deleted (0: never).
    max_age: u64,
    entries: Vec<DiskEntry>,
    next_id: u64,
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Sanitizes a format tag into a directory name.
fn dir_name(format: &str) -> String {
    format
        .chars()
        .map(
            |c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c } else { '_' },
        )
        .collect()
}

impl DiskStore {
    /// Opens (creating) the tier for `format` under `root` with `budget`
    /// bytes and a `max_age` in seconds (0 for none), indexing the entries
    /// already there, expiring the stale ones and trimming to budget.
    pub fn open(root: &Path, format: &str, budget: u64, max_age: u64) -> Result<Self> {
        let dir = root.join(dir_name(format));
        fs::create_dir_all(&dir)
            .with_context(|| format!("creating {}", dir.display()))?;
        let mut entries = Vec::new();
        let mut next_id = 1u64;
        for entry in
            fs::read_dir(&dir).with_context(|| format!("listing {}", dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let id = entry.file_name().to_string_lossy().into_owned();
            match Self::load_meta(&path, format) {
                Ok(meta) => {
                    if let Some(n) =
                        id.strip_prefix("s").and_then(|n| n.parse::<u64>().ok())
                    {
                        next_id = next_id.max(n + 1);
                    }
                    entries.push(DiskEntry {
                        id,
                        tokens: meta.tokens,
                        checkpoints: meta.checkpoints,
                        bytes: meta.bytes,
                        last_used: meta.last_used,
                        cache_key: meta.cache_key,
                        durable: meta.durable,
                        images: meta.images,
                    });
                }
                Err(error) => {
                    eprintln!("session cache: dropping {}: {error:#}", path.display());
                    let _ = fs::remove_dir_all(&path);
                }
            }
        }
        let mut store =
            Self { dir, format: format.to_owned(), budget, max_age, entries, next_id };
        store.expire();
        store.trim(0);
        Ok(store)
    }

    /// Deletes entries unused for longer than the maximum age. Called on
    /// open, on every store and before every lookup, so the tier shrinks on
    /// its own even when nothing new is written.
    pub fn expire(&mut self) {
        if self.max_age == 0 {
            return;
        }
        let now = now_secs();
        let stale: Vec<String> = self
            .entries
            .iter()
            .filter(|e| now.saturating_sub(e.last_used) > self.max_age)
            .map(|e| e.id.clone())
            .collect();
        for id in stale {
            self.remove(&id);
        }
    }

    pub fn max_age_secs(&self) -> u64 {
        self.max_age
    }

    fn load_meta(path: &Path, format: &str) -> Result<Meta> {
        let meta: Meta = serde_json::from_slice(&fs::read(path.join(META))?)
            .context("parsing meta.json")?;
        ensure!(meta.format == format, "format {:?} != {format:?}", meta.format);
        ensure!(path.join(PREFIX).is_file(), "prefix.bin missing");
        ensure!(
            !meta.tokens.is_empty() && meta.checkpoints.contains(&meta.tokens.len()),
            "no live-end checkpoint"
        );
        for &pos in &meta.checkpoints {
            ensure!(
                path.join(format!("ckpt-{pos}.bin")).is_file(),
                "ckpt-{pos}.bin missing"
            );
        }
        Ok(meta)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget
    }

    pub fn used_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes).sum()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[DiskEntry] {
        &self.entries
    }

    /// Durable prefix entries currently held (at most [`DURABLE_MAX_ENTRIES`]).
    pub fn durable_len(&self) -> usize {
        self.entries.iter().filter(|e| e.durable).count()
    }

    /// Writes a session: `prefix` streams the per-token caches for all of
    /// `tokens`, `checkpoint(pos, w)` streams the snapshot at each position in
    /// `checkpoints` (which must include `tokens.len()`), `images` are the
    /// spans among `tokens`. Returns the new id, or `None` when the entry
    /// does not fit the budget or the volume.
    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &mut self,
        tokens: &[u32],
        images: &[CachedImage],
        cache_key: Option<&str>,
        checkpoints: &[usize],
        prefix: &mut dyn FnMut(&mut dyn Write) -> Result<()>,
        checkpoint: &mut dyn FnMut(usize, &mut dyn Write) -> Result<()>,
    ) -> Result<Option<String>> {
        self.store_with(
            false,
            tokens,
            images,
            cache_key,
            checkpoints,
            prefix,
            checkpoint,
        )
    }

    /// [`Self::store`] for a durable prefix entry: `tokens` is the shared
    /// prefix and its length the one checkpoint. Storing the entry that takes
    /// the count past [`DURABLE_MAX_ENTRIES`] deletes the least recently used
    /// durable entry; evicted sessions are never touched for it.
    #[allow(clippy::too_many_arguments)]
    pub fn store_durable(
        &mut self,
        tokens: &[u32],
        images: &[CachedImage],
        cache_key: Option<&str>,
        checkpoints: &[usize],
        prefix: &mut dyn FnMut(&mut dyn Write) -> Result<()>,
        checkpoint: &mut dyn FnMut(usize, &mut dyn Write) -> Result<()>,
    ) -> Result<Option<String>> {
        self.store_with(
            true,
            tokens,
            images,
            cache_key,
            checkpoints,
            prefix,
            checkpoint,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn store_with(
        &mut self,
        durable: bool,
        tokens: &[u32],
        images: &[CachedImage],
        cache_key: Option<&str>,
        checkpoints: &[usize],
        prefix: &mut dyn FnMut(&mut dyn Write) -> Result<()>,
        checkpoint: &mut dyn FnMut(usize, &mut dyn Write) -> Result<()>,
    ) -> Result<Option<String>> {
        ensure!(!tokens.is_empty(), "empty session");
        ensure!(
            checkpoints.contains(&tokens.len()),
            "the live end must be a checkpoint"
        );
        ensure!(
            images.iter().all(|i| i.end() <= tokens.len()),
            "an image span extends past the session's tokens"
        );
        self.expire();
        let id = format!("s{}", self.next_id);
        let path = self.dir.join(&id);
        let new = NewEntry { tokens, images, cache_key, checkpoints, durable };
        let result = self.write_entry(&path, &new, prefix, checkpoint);
        match result {
            Ok(Some(entry)) => {
                self.next_id += 1;
                self.entries.push(entry);
                // The cap first, so the new entry (the most recent) is what
                // stays when the budget trim below has to choose.
                if durable {
                    self.trim_durable();
                }
                self.trim(0);
                Ok(Some(id))
            }
            Ok(None) => {
                let _ = fs::remove_dir_all(&path);
                Ok(None)
            }
            Err(error) => {
                let _ = fs::remove_dir_all(&path);
                Err(error)
            }
        }
    }

    fn write_entry(
        &mut self,
        path: &Path,
        new: &NewEntry<'_>,
        prefix: &mut dyn FnMut(&mut dyn Write) -> Result<()>,
        checkpoint: &mut dyn FnMut(usize, &mut dyn Write) -> Result<()>,
    ) -> Result<Option<DiskEntry>> {
        let NewEntry { tokens, images, cache_key, checkpoints, durable } = *new;
        fs::create_dir_all(path)?;
        let mut bytes = 0u64;
        {
            let mut w =
                BufWriter::with_capacity(8 << 20, File::create(path.join(PREFIX))?);
            prefix(&mut w)?;
            w.flush()?;
            bytes += w.get_ref().metadata()?.len();
        }
        let mut positions: Vec<usize> = checkpoints.to_vec();
        positions.sort_unstable();
        positions.dedup();
        for &pos in &positions {
            let mut w = BufWriter::with_capacity(
                8 << 20,
                File::create(path.join(format!("ckpt-{pos}.bin")))?,
            );
            checkpoint(pos, &mut w)?;
            w.flush()?;
            bytes += w.get_ref().metadata()?.len();
        }
        if bytes > self.budget {
            return Ok(None);
        }
        // Make room, then check the volume can spare it.
        self.trim(bytes);
        if let Some(avail) = available_bytes(&self.dir)
            && avail < FREE_SPACE_MARGIN
        {
            eprintln!(
                "session cache: {:.1} GB free on the volume, not keeping the evicted session",
                avail as f64 / 1e9
            );
            return Ok(None);
        }
        let meta = Meta {
            format: self.format.clone(),
            tokens: tokens.to_vec(),
            checkpoints: positions.clone(),
            bytes,
            last_used: now_secs(),
            cache_key: cache_key.map(str::to_owned),
            durable,
            images: images.to_vec(),
        };
        fs::write(path.join(META), serde_json::to_vec(&meta)?)?;
        Ok(Some(DiskEntry {
            id: path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            tokens: meta.tokens,
            checkpoints: positions,
            bytes,
            last_used: meta.last_used,
            cache_key: meta.cache_key,
            durable,
            images: meta.images,
        }))
    }

    /// Opens the per-token cache file of `id` for reading.
    pub fn open_prefix(&self, id: &str) -> Result<Box<dyn Read>> {
        let path = self.dir.join(id).join(PREFIX);
        Ok(Box::new(BufReader::with_capacity(
            8 << 20,
            File::open(&path).with_context(|| format!("opening {}", path.display()))?,
        )))
    }

    /// Opens the checkpoint at `pos` of `id` for reading.
    pub fn open_checkpoint(&self, id: &str, pos: usize) -> Result<Box<dyn Read>> {
        let path = self.dir.join(id).join(format!("ckpt-{pos}.bin"));
        Ok(Box::new(BufReader::with_capacity(
            8 << 20,
            File::open(&path).with_context(|| format!("opening {}", path.display()))?,
        )))
    }

    /// Marks `id` as just used (in memory and in its meta file).
    pub fn touch(&mut self, id: &str) {
        let now = now_secs();
        let dir = self.dir.clone();
        let format = self.format.clone();
        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            entry.last_used = now;
            let meta = Meta {
                format,
                tokens: entry.tokens.clone(),
                checkpoints: entry.checkpoints.clone(),
                bytes: entry.bytes,
                last_used: now,
                cache_key: entry.cache_key.clone(),
                durable: entry.durable,
                images: entry.images.clone(),
            };
            if let Ok(json) = serde_json::to_vec(&meta) {
                let _ = fs::write(dir.join(id).join(META), json);
            }
        }
    }

    /// Deletes least-recently-used durable entries until at most
    /// [`DURABLE_MAX_ENTRIES`] remain. Evicted sessions are not candidates.
    fn trim_durable(&mut self) {
        while self.durable_len() > DURABLE_MAX_ENTRIES {
            let Some(victim) = self
                .entries
                .iter()
                .filter(|e| e.durable)
                .min_by_key(|e| e.last_used)
                .map(|e| e.id.clone())
            else {
                break;
            };
            self.remove(&victim);
        }
    }

    /// Deletes `id`.
    pub fn remove(&mut self, id: &str) {
        if let Some(i) = self.entries.iter().position(|e| e.id == id) {
            self.entries.swap_remove(i);
            let _ = fs::remove_dir_all(self.dir.join(id));
        }
    }

    /// Deletes least-recently-used entries until `extra` more bytes fit.
    fn trim(&mut self, extra: u64) {
        while !self.entries.is_empty() && self.used_bytes() + extra > self.budget {
            let Some(victim) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
            else {
                break;
            };
            let id = self.entries.swap_remove(victim).id;
            let _ = fs::remove_dir_all(self.dir.join(&id));
        }
    }
}

/// Bytes available to this user on the volume holding `path`.
fn available_bytes(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    // Darwin's `struct statvfs`: block counts are `unsigned int`.
    #[repr(C)]
    struct StatVfs {
        f_bsize: u64,
        f_frsize: u64,
        f_blocks: u32,
        f_bfree: u32,
        f_bavail: u32,
        f_files: u32,
        f_ffree: u32,
        f_favail: u32,
        f_fsid: u64,
        f_flag: u64,
        f_namemax: u64,
    }
    unsafe extern "C" {
        fn statvfs(path: *const std::ffi::c_char, buf: *mut StatVfs) -> i32;
    }
    let c = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut st = StatVfs {
        f_bsize: 0,
        f_frsize: 0,
        f_blocks: 0,
        f_bfree: 0,
        f_bavail: 0,
        f_files: 0,
        f_ffree: 0,
        f_favail: 0,
        f_fsid: 0,
        f_flag: 0,
        f_namemax: 0,
    };
    // SAFETY: valid NUL-terminated path and a properly sized out-struct.
    let rc = unsafe { statvfs(c.as_ptr(), &mut st) };
    (rc == 0).then(|| u64::from(st.f_bavail) * st.f_frsize)
}

#[cfg(test)]
#[path = "../../tests/unit/serve/disk.rs"]
mod tests;
