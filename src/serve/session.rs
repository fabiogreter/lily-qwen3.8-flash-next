//! The session cache: token lineages with resumable positions under a byte
//! budget.
//!
//! A session owns one decode state (per-token caches valid up to `pos`) and a
//! few checkpoints of the recurrent part at earlier positions. A new prompt
//! resumes from the largest position that is both a checkpoint (or the live
//! end) and a common prefix of the session's tokens and the prompt, capped at
//! `prompt_len - 1` so at least one token is fed to produce logits.
//!
//! Extending the live end reuses the session in place. Anything else forks:
//! the per-token caches up to the resume position are copied into a fresh
//! state and the checkpoint restored, so the original lineage survives (a
//! parallel conversation sharing only the system prompt must not destroy a
//! long context). Sessions are evicted least-recently-used under the budget.
//!
//! Below the GPU tier sits an optional disk tier ([`DiskStore`]): sessions
//! evicted from GPU memory are written out (per-token caches plus their
//! checkpoints and a snapshot of the live end) and a later prompt that shares
//! a prefix reads them back instead of recomputing it. A disk hit at the live
//! end moves the session back to the GPU (the file copy is dropped); a hit at
//! an earlier checkpoint forks from the file and leaves it in place.
//!
//! The disk tier also holds **durable prefix entries**: a prefix that two
//! prompts shared (the `agreement`) but that no lineage could resume from,
//! because checkpoints sit at the end of served prompts and two agent runs
//! diverge before that. The engine materialises such a boundary once, as a
//! disk entry whose live end is the boundary, and every later prompt with the
//! same preamble resumes there. These entries live only on disk, never as
//! resident sessions or extra checkpoints: they are hit far more rarely than
//! the live conversation's cache and must not compete with it for the GPU
//! budget. A hit never consumes them.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result, ensure};

use super::disk::DiskStore;
use crate::engine::{DecodeStateApi, LanguageModel, SnapshotApi};
use crate::metal::MetalContext;

type SnapshotOf<M> = <<M as LanguageModel>::State as DecodeStateApi>::Snapshot;

/// Sessions shorter than this are not worth a round trip to disk.
const MIN_DISK_TOKENS: usize = 256;

/// Tokens of the best-agreeing lineage returned past the agreement, enough
/// for the divergence diagnostic to show a line of text.
const DIVERGENT_TAIL_TOKENS: usize = 16;

/// How far `prompt` agrees with the closest of `lineages`: the longest
/// common prefix with any of them, capped at `prompt.len() - 1` like every
/// resume position (the last prompt token is always fed). Also returns up to
/// [`DIVERGENT_TAIL_TOKENS`] of that lineage's tokens from the agreement on,
/// which is what the prompt would have had to continue with to keep
/// matching; empty when nothing agrees.
pub fn agreement<'a>(prompt: &[u32], lineages: impl IntoIterator<Item = &'a [u32]>) -> (usize, Vec<u32>) {
    let cap = prompt.len().saturating_sub(1);
    let mut best: Option<(usize, &[u32])> = None;
    for tokens in lineages {
        let lcp = common_prefix_len(tokens, prompt).min(cap);
        if lcp > 0 && best.is_none_or(|(b, _)| lcp > b) {
            best = Some((lcp, tokens));
        }
    }
    match best {
        Some((lcp, tokens)) => (lcp, tokens[lcp..tokens.len().min(lcp + DIVERGENT_TAIL_TOKENS)].to_vec()),
        None => (0, Vec::new()),
    }
}

/// Where a durable prefix entry should be materialised for a prompt that
/// agreed with a cached lineage for `agreement` tokens but resumed at
/// `reused`: the agreement itself when it is long enough to be worth a disk
/// entry (`min_tokens`; 0 turns the feature off), lies beyond where the run
/// resumed (so no resumable state exists there yet) and leaves at least one
/// token to feed. Two real prompts shared that prefix, so a third is likely;
/// within one growing conversation `reused == agreement` and nothing is
/// written turn after turn.
pub fn boundary_position(agreement: usize, reused: usize, prompt_len: usize, min_tokens: usize) -> Option<usize> {
    let worth_it = min_tokens > 0 && agreement >= min_tokens;
    (worth_it && agreement > reused && agreement <= prompt_len.checked_sub(1)?).then_some(agreement)
}

/// The best position to resume `prompt` from given a lineage's tokens and
/// its resumable positions: the live end when the lineage is a strict prefix
/// of the prompt (and `live_end` is resumable), else the latest resumable
/// position within the common prefix. Never `>= prompt.len()`.
fn resume_position(tokens: &[u32], checkpoints: &[usize], live_end: bool, prompt: &[u32]) -> Option<usize> {
    let lcp = common_prefix_len(tokens, prompt);
    let limit = lcp.min(prompt.len().checked_sub(1)?);
    if tokens.len() <= limit {
        if live_end {
            return (!tokens.is_empty()).then_some(tokens.len());
        }
        return checkpoints.iter().rev().copied().find(|&p| p > 0 && p <= limit);
    }
    checkpoints.iter().rev().copied().find(|&p| p > 0 && p <= limit)
}

pub struct Session<M: LanguageModel> {
    /// Tokens fed into `state`; `state.pos() == tokens.len()` when at rest.
    pub tokens: Vec<u32>,
    pub state: M::State,
    /// Recurrent-state checkpoints, ascending by position, all `<= tokens.len()`.
    checkpoints: Vec<Arc<SnapshotOf<M>>>,
    cache_key: Option<String>,
    last_used: u64,
}

impl<M: LanguageModel> Session<M> {
    fn new(state: M::State) -> Self {
        Self { tokens: Vec::new(), state, checkpoints: Vec::new(), cache_key: None, last_used: 0 }
    }

    /// Records a checkpoint of the state's current position. Duplicates by
    /// position are dropped.
    pub fn add_checkpoint(&mut self, snapshot: SnapshotOf<M>) {
        let pos = snapshot.pos();
        if self.checkpoints.iter().any(|c| c.pos() == pos) {
            return;
        }
        let at = self.checkpoints.partition_point(|c| c.pos() < pos);
        self.checkpoints.insert(at, Arc::new(snapshot));
    }

    /// GPU bytes the session holds.
    pub fn bytes(&self) -> usize {
        self.state.bytes() + self.checkpoints.iter().map(|c| c.bytes()).sum::<usize>()
    }

    /// The best position to resume `prompt` from: the live end when the
    /// session is a strict prefix of the prompt, else the latest checkpoint
    /// within the common prefix. Never `>= prompt.len()`.
    fn resume_position(&self, prompt: &[u32]) -> Option<usize> {
        let positions: Vec<usize> = self.checkpoints.iter().map(|c| c.pos()).collect();
        resume_position(&self.tokens, &positions, true, prompt)
    }

    fn checkpoint_at(&self, pos: usize) -> Option<&Arc<SnapshotOf<M>>> {
        self.checkpoints.iter().find(|c| c.pos() == pos)
    }
}

/// What [`SessionStore::acquire`] hands out.
pub struct Acquired<M: LanguageModel> {
    pub session: Session<M>,
    /// Tokens of the prompt already fed into `session.state`.
    pub reused: usize,
    /// Whether the session was forked from a cached lineage (diagnostics).
    pub forked: bool,
    /// Whether the reused prefix was read from the disk tier, and how long
    /// that took.
    pub from_disk: Option<std::time::Duration>,
    /// The longest common prefix between the prompt and any lineage the store
    /// knows (resident or on disk), capped at `prompt.len() - 1`. Always
    /// `>= reused`; the gap is a prefix that was shared but not resumable,
    /// which is what a durable prefix entry fixes (see [`boundary_position`]).
    pub agreement: usize,
    /// The best-agreeing lineage's tokens from `agreement` on (at most
    /// [`DIVERGENT_TAIL_TOKENS`]), for the divergence diagnostic. Empty when
    /// nothing agreed.
    pub divergent_tail: Vec<u32>,
}

impl<M: LanguageModel> Acquired<M> {
    /// A session nothing was reused for; the other fields are filled in by
    /// the path that built it.
    fn fresh(session: Session<M>) -> Self {
        Self { session, reused: 0, forked: false, from_disk: None, agreement: 0, divergent_tail: Vec::new() }
    }
}

pub struct SessionStore<M: LanguageModel> {
    entries: Vec<Session<M>>,
    budget_bytes: usize,
    max_sessions: usize,
    max_checkpoints: usize,
    clock: u64,
    disk: Option<DiskStore>,
    /// Shortest shared prefix worth a durable disk entry (0: never).
    durable_min_tokens: usize,
}

impl<M: LanguageModel> SessionStore<M> {
    pub fn new(budget_bytes: usize, max_sessions: usize, max_checkpoints: usize) -> Self {
        Self {
            entries: Vec::new(),
            budget_bytes,
            max_sessions,
            max_checkpoints: max_checkpoints.max(1),
            clock: 0,
            disk: None,
            durable_min_tokens: 0,
        }
    }

    /// Attaches the disk tier.
    pub fn with_disk(mut self, disk: DiskStore) -> Self {
        self.disk = Some(disk);
        self
    }

    /// Sets the shortest shared prefix worth a durable disk entry (0 turns
    /// durable entries off). Only meaningful with a disk tier attached.
    pub fn with_durable_min_tokens(mut self, min_tokens: usize) -> Self {
        self.durable_min_tokens = min_tokens;
        self
    }

    pub fn durable_min_tokens(&self) -> usize {
        self.durable_min_tokens
    }

    pub fn disk(&self) -> Option<&DiskStore> {
        self.disk.as_ref()
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    pub fn used_bytes(&self) -> usize {
        self.entries.iter().map(Session::bytes).sum()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Checks out the session that can resume `prompt` from the furthest
    /// position, forking when that position is not the live end. `cache_key`
    /// only breaks ties between equally good candidates. Also reports how far
    /// the prompt agreed with any lineage at all (`agreement`), which the
    /// engine compares with `reused` to decide on a durable prefix entry.
    pub fn acquire(
        &mut self,
        ctx: &MetalContext,
        model: &M,
        prompt: &[u32],
        cache_key: Option<&str>,
    ) -> Result<Acquired<M>> {
        ensure!(!prompt.is_empty(), "empty prompt");
        if let Some(disk) = self.disk.as_mut() {
            disk.expire();
        }
        let (agreement, divergent_tail) = agreement(
            prompt,
            self.entries
                .iter()
                .map(|s| s.tokens.as_slice())
                .chain(self.disk.iter().flat_map(|d| d.entries().iter().map(|e| e.tokens.as_slice()))),
        );
        let mut acquired = self.acquire_resumable(ctx, model, prompt, cache_key)?;
        debug_assert!(acquired.reused <= agreement, "resumed past the agreement");
        acquired.agreement = agreement;
        acquired.divergent_tail = divergent_tail;
        Ok(acquired)
    }

    /// [`Self::acquire`] without the agreement: picks the lineage and builds
    /// the session. `agreement` and `divergent_tail` are left at their
    /// defaults for the caller to fill in.
    fn acquire_resumable(
        &mut self,
        ctx: &MetalContext,
        model: &M,
        prompt: &[u32],
        cache_key: Option<&str>,
    ) -> Result<Acquired<M>> {
        let best = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.resume_position(prompt).map(|p| (i, p)))
            .max_by_key(|&(i, p)| {
                let entry = &self.entries[i];
                let key_match = cache_key.is_some_and(|k| entry.cache_key.as_deref() == Some(k));
                (p, key_match, entry.last_used)
            });

        // The disk tier competes on resume position; ties go to the GPU.
        let best_disk = self.disk.as_ref().and_then(|disk| {
            disk.entries()
                .iter()
                .filter_map(|e| resume_position(&e.tokens, &e.checkpoints, true, prompt).map(|p| (e.id.clone(), p, e.tokens.len())))
                .max_by_key(|(id, p, _)| {
                    let key_match = cache_key.is_some_and(|k| disk.entries().iter().any(|e| &e.id == id && e.cache_key.as_deref() == Some(k)));
                    (*p, key_match)
                })
        });
        if let Some((id, pos, len)) = best_disk
            && best.is_none_or(|(_, gpu_pos)| pos > gpu_pos)
        {
            return self.acquire_from_disk(ctx, model, prompt, &id, pos, len);
        }

        let Some((index, resume_at)) = best else {
            let state = model.new_state(ctx, prompt.len())?;
            let session = Session::new(state);
            self.trim(ctx, session.bytes());
            return Ok(Acquired::fresh(session));
        };

        if resume_at == self.entries[index].tokens.len() {
            // Pure extension of the live end: take the session as is.
            let session = self.entries.swap_remove(index);
            ensure!(session.state.pos() == resume_at, "cached decode state out of step with its tokens");
            return Ok(Acquired { reused: resume_at, ..Acquired::fresh(session) });
        }

        // Fork: copy the per-token prefix, restore the recurrent checkpoint.
        // The new state briefly exceeds the budget until `trim` runs after
        // the copy (the source must survive until then).
        let mut state = model.new_state(ctx, prompt.len())?;
        let source = &self.entries[index];
        let checkpoint = source
            .checkpoint_at(resume_at)
            .ok_or_else(|| anyhow::anyhow!("resume position without checkpoint"))?
            .clone();
        state.copy_prefix_from(ctx, &source.state, resume_at)?;
        state.restore(ctx, &checkpoint)?;
        let mut session = Session::new(state);
        session.tokens = source.tokens[..resume_at].to_vec();
        session.checkpoints.push(checkpoint);
        self.trim(ctx, session.bytes());
        Ok(Acquired { reused: resume_at, forked: true, ..Acquired::fresh(session) })
    }

    /// Builds a session from disk entry `id` resumed at `pos`: the per-token
    /// caches up to `pos` and the checkpoint there. A live-end hit consumes
    /// the entry; a checkpoint hit forks from it and leaves it on disk. A
    /// durable prefix entry is forked from even at its live end: it exists to
    /// be hit again, and every hit is a lineage of its own.
    fn acquire_from_disk(
        &mut self,
        ctx: &MetalContext,
        model: &M,
        prompt: &[u32],
        id: &str,
        pos: usize,
        len: usize,
    ) -> Result<Acquired<M>> {
        let started = Instant::now();
        let disk = self.disk.as_mut().expect("disk tier");
        let entry = disk.entries().iter().find(|e| e.id == id).context("disk entry vanished")?;
        let (tokens, durable) = (entry.tokens[..pos].to_vec(), entry.durable);
        let mut state = model.new_state(ctx, prompt.len().max(pos))?;
        let result = (|| -> Result<SnapshotOf<M>> {
            let mut prefix = disk.open_prefix(id)?;
            state.read_prefix(ctx, len, pos, &mut prefix)?;
            let mut ckpt = disk.open_checkpoint(id, pos)?;
            let snapshot = model.read_snapshot(ctx, &mut ckpt)?;
            ensure!(snapshot.pos() == pos, "checkpoint file at {pos} holds position {}", snapshot.pos());
            state.restore(ctx, &snapshot)?;
            Ok(snapshot)
        })();
        let snapshot = match result {
            Ok(snapshot) => snapshot,
            Err(error) => {
                // A damaged entry must not poison every later request.
                eprintln!("session cache: dropping unreadable disk entry {id}: {error:#}");
                disk.remove(id);
                let state = model.new_state(ctx, prompt.len())?;
                let session = Session::new(state);
                self.trim(ctx, session.bytes());
                return Ok(Acquired::fresh(session));
            }
        };
        let forked = pos != len || durable;
        if forked {
            disk.touch(id);
        } else {
            disk.remove(id);
        }
        let mut session = Session::new(state);
        session.tokens = tokens;
        session.checkpoints.push(Arc::new(snapshot));
        self.trim(ctx, session.bytes());
        Ok(Acquired { reused: pos, forked, from_disk: Some(started.elapsed()), ..Acquired::fresh(session) })
    }

    /// Writes a durable prefix entry for `tokens`, whose per-token caches
    /// `state` holds and whose recurrent state at `tokens.len()` is
    /// `snapshot`: the boundary [`boundary_position`] found, materialised so
    /// later prompts with the same prefix resume there. Only the disk tier
    /// keeps it (see the module docs for why). Returns the entry id, `None`
    /// when the tier did not take it, or the write error; failures only cost
    /// the entry, so the caller logs and carries on.
    pub fn store_durable(
        &mut self,
        tokens: &[u32],
        cache_key: Option<&str>,
        state: &M::State,
        snapshot: &SnapshotOf<M>,
    ) -> Result<Option<String>> {
        let disk = self.disk.as_mut().context("no disk tier")?;
        let n = tokens.len();
        ensure!(n > 0 && snapshot.pos() == n, "durable snapshot at {} for {n} tokens", snapshot.pos());
        ensure!(state.pos() >= n, "decode state at {} holds no caches for {n} tokens", state.pos());
        disk.store_durable(tokens, cache_key, &[n], &mut |w| state.write_prefix(n, w), &mut |_, w| snapshot.write_to(w))
    }

    /// Evicts least-recently-used sessions until `extra` more bytes fit the
    /// budget (or the store is empty), spilling them to the disk tier.
    fn trim(&mut self, ctx: &MetalContext, extra: usize) {
        while !self.entries.is_empty() && self.used_bytes() + extra > self.budget_bytes {
            let Some(victim) = self.lru_index() else { break };
            let session = self.entries.swap_remove(victim);
            self.spill(ctx, session);
        }
    }

    /// Moves every resident session to the disk tier and empties the store;
    /// sessions the tier does not take (too short, or no tier at all) are
    /// lost. Returns how many were written and how many were dropped.
    pub fn spill_all(&mut self, ctx: &MetalContext) -> (usize, usize) {
        let (mut spilled, mut dropped) = (0, 0);
        let entries = std::mem::take(&mut self.entries);
        if self.disk.is_none() && !entries.is_empty() {
            eprintln!("session cache: no disk tier; {} resident sessions are lost", entries.len());
        }
        for session in entries {
            if self.spill(ctx, session) {
                spilled += 1;
            } else {
                dropped += 1;
            }
        }
        (spilled, dropped)
    }

    /// Empties the store without writing anything to disk, for a context
    /// whose GPU state can no longer be trusted or read (a fault). Returns
    /// how many sessions were dropped; the disk tier's earlier copies stay.
    pub fn drop_all(&mut self) -> usize {
        let dropped = self.entries.len();
        self.entries.clear();
        dropped
    }

    /// Writes an evicted session to the disk tier (when there is one and the
    /// session is long enough to be worth it). Failures only cost the copy.
    /// Returns whether the session is now on disk.
    fn spill(&mut self, ctx: &MetalContext, session: Session<M>) -> bool {
        let Some(disk) = self.disk.as_mut() else { return false };
        if session.tokens.len() < MIN_DISK_TOKENS || session.state.pos() != session.tokens.len() {
            return false;
        }
        // A faulted context cannot run the snapshot blit and its caches are
        // not trustworthy; the engine is about to drop every session and
        // reports that once, so do not log a failure per session here.
        if ctx.fault().is_some() {
            return false;
        }
        let started = Instant::now();
        let live = match session.state.snapshot(ctx) {
            Ok(live) => live,
            Err(error) => {
                eprintln!("session cache: cannot snapshot an evicted session: {error:#}");
                return false;
            }
        };
        let n = session.tokens.len();
        let mut positions: Vec<usize> = session.checkpoints.iter().map(|c| c.pos()).filter(|&p| p > 0 && p < n).collect();
        positions.push(n);
        let result = disk.store(
            &session.tokens,
            session.cache_key.as_deref(),
            &positions,
            &mut |w| session.state.write_prefix(n, w),
            &mut |pos, w| {
                if pos == n {
                    live.write_to(w)
                } else {
                    session.checkpoint_at(pos).context("checkpoint vanished")?.write_to(w)
                }
            },
        );
        match result {
            Ok(Some(id)) => {
                eprintln!(
                    "session cache: spilled {n} tokens to disk as {id} in {:.2}s ({} entries, {:.1}/{:.1} GB on disk)",
                    started.elapsed().as_secs_f64(),
                    disk.len(),
                    disk.used_bytes() as f64 / 1e9,
                    disk.budget_bytes() as f64 / 1e9
                );
                true
            }
            Ok(None) => false,
            Err(error) => {
                eprintln!("session cache: spilling to disk failed: {error:#}");
                false
            }
        }
    }

    fn lru_index(&self) -> Option<usize> {
        self.entries.iter().enumerate().min_by_key(|(_, s)| s.last_used).map(|(i, _)| i)
    }

    /// Returns a session to the cache after a request. Its state must sit at
    /// `tokens.len()`. Old checkpoints beyond the per-session cap are dropped
    /// (the newest are the ones the next request most likely resumes from),
    /// and the store is trimmed to budget and count.
    pub fn release(&mut self, ctx: &MetalContext, mut session: Session<M>, cache_key: Option<&str>) {
        if session.state.pos() != session.tokens.len() || session.tokens.is_empty() {
            return;
        }
        session.checkpoints.retain(|c| c.pos() <= session.tokens.len());
        while session.checkpoints.len() > self.max_checkpoints {
            session.checkpoints.remove(0);
        }
        self.clock = self.clock.wrapping_add(1);
        session.last_used = self.clock;
        session.cache_key = cache_key.map(str::to_owned);
        self.entries.push(session);
        while self.entries.len() > self.max_sessions
            || (self.used_bytes() > self.budget_bytes && self.entries.len() > 1)
        {
            let Some(victim) = self.lru_index() else { break };
            let session = self.entries.swap_remove(victim);
            self.spill(ctx, session);
        }
    }
}

pub fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
#[path = "../../tests/unit/serve/session.rs"]
mod tests;
