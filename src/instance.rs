//! One lily per machine. Every process that loads the model's weights (the
//! server, `lily-bench`, `lily-probe`, `lily-vision-probe`, `lily-experts`
//! and the ignored tests that load a real checkpoint) takes an exclusive
//! advisory lock first, and refuses to run while another process holds it:
//! two full models on a 128 GB machine panicked it on 2026-09-12 (102.8 GB
//! and 68.6 GB resident, 62 MB free).
//!
//! The lock is `flock(LOCK_EX | LOCK_NB)` on [`default_path`], taken once per
//! process ([`acquire`], from the load path, so a reload after an idle unload
//! finds it held) and never released: the kernel drops it when the process
//! exits, however it exits, so no stale lock can block the next start. The
//! holder writes its pid and binary into the file for the refusal message;
//! the contents are informational only, the lock is what counts.
//!
//! A refused process exits with [`EXIT_ALREADY_RUNNING`] (75, `EX_TEMPFAIL`
//! in `sysexits.h`: try again later). Under the launchd agent that is a
//! failed exit, so launchd starts it again after its 30 s throttle and the
//! service comes back by itself once the other instance has exited.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read as _, Seek as _, Write as _};
use std::os::fd::AsRawFd as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;

use anyhow::{Context as _, Result};

/// The exit status of a process refused because another instance runs.
pub const EXIT_ALREADY_RUNNING: u8 = 75;

/// The lock every weight-loading process takes:
/// `~/Library/Caches/lily/instance.lock`, next to the disk tier's default
/// directory. Deliberately not configurable, so no two processes can
/// disagree about it; tests lock their own file through [`InstanceLock::acquire_at`].
pub fn default_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join("Library").join("Caches").join("lily").join("instance.lock")
}

/// Another process holds the lock.
#[derive(Debug)]
pub struct AlreadyRunning {
    pub path: PathBuf,
    /// What the holder wrote into the file (`pid <n>` and its binary), when
    /// it had written anything yet.
    pub holder: Option<Holder>,
}

/// The holder's own record in the lock file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Holder {
    pub pid: u32,
    pub binary: String,
}

impl Holder {
    fn parse(text: &str) -> Option<Self> {
        let mut lines = text.lines();
        let pid = lines.next()?.strip_prefix("pid ")?.trim().parse().ok()?;
        let binary = lines.next().unwrap_or("").trim().to_owned();
        Some(Self { pid, binary })
    }
}

impl fmt::Display for AlreadyRunning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let who = match &self.holder {
            Some(h) if !h.binary.is_empty() => format!("pid {} ({})", h.pid, h.binary),
            Some(h) => format!("pid {}", h.pid),
            None => "a process that has not written its pid yet".to_owned(),
        };
        write!(
            f,
            "another lily instance is running: {who} holds {}. Only one process may load the model \
             at a time (two full models do not fit this machine). If it is the background service, \
             stop it with `tools/service/lily-service.sh stop` (it stays down until `start`)",
            self.path.display()
        )
    }
}

impl std::error::Error for AlreadyRunning {}

/// A held lock; dropping it (or exiting) releases it.
#[derive(Debug)]
pub struct InstanceLock {
    _file: File,
    path: PathBuf,
}

impl InstanceLock {
    /// Takes the lock at `path` without waiting, creating the file and its
    /// directory as needed, and records this process in it. Fails with
    /// [`AlreadyRunning`] (downcastable from the error) when another open
    /// file description holds it, which includes another handle in this
    /// same process.
    pub fn acquire_at(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("creating {}", dir.display()))?;
        }
        // Not truncated on open: the holder's record must survive a
        // refused open so the refusal can name it.
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("opening the instance lock {}", path.display()))?;
        // SAFETY: flock on a descriptor we own.
        let rc =
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::EWOULDBLOCK) {
                let mut text = String::new();
                let holder = file
                    .read_to_string(&mut text)
                    .ok()
                    .and_then(|_| Holder::parse(&text));
                return Err(AlreadyRunning { path: path.to_owned(), holder }.into());
            }
            return Err(error).with_context(|| {
                format!("locking the instance lock {}", path.display())
            });
        }
        let binary = std::env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "unknown binary".to_owned());
        file.set_len(0)?;
        file.rewind()?;
        write!(file, "pid {}\n{binary}\n", std::process::id())
            .and_then(|()| file.flush())
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(Self { _file: file, path: path.to_owned() })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// The process's lock once taken; never dropped, so it lives until exit.
static HELD: Mutex<Option<InstanceLock>> = Mutex::new(None);

/// Takes the process-wide lock at [`default_path`] unless this process
/// already holds it: the first model load (or the server's start) takes it,
/// every later load (a reload after an idle unload or a GPU fault) finds it
/// held. Errors with [`AlreadyRunning`] when another process holds it.
pub fn acquire() -> Result<()> {
    let mut held = HELD.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if held.is_none() {
        *held = Some(InstanceLock::acquire_at(&default_path())?);
    }
    Ok(())
}

/// The exit status for an error that ends a process: 75 when it is (or was
/// caused by) [`AlreadyRunning`], 1 otherwise.
pub fn exit_status(error: &anyhow::Error) -> u8 {
    if error.chain().any(|e| e.is::<AlreadyRunning>()) {
        EXIT_ALREADY_RUNNING
    } else {
        1
    }
}

/// What a binary's `main` returns for `result`: success, or the error
/// printed the way a `main` returning `Result` prints it and the exit
/// status [`exit_status`] picks, except that a refusal prints only its
/// message.
pub fn exit_code(result: Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            match error.chain().find_map(|e| e.downcast_ref::<AlreadyRunning>()) {
                Some(refused) => {
                    eprintln!(
                        "error: {refused}; exiting with status {EXIT_ALREADY_RUNNING}"
                    )
                }
                None => eprintln!("Error: {error:?}"),
            }
            ExitCode::from(exit_status(&error))
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/instance.rs"]
mod tests;
