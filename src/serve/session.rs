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

use std::sync::Arc;

use anyhow::{Result, ensure};

use crate::engine::{DecodeStateApi, LanguageModel, SnapshotApi};
use crate::metal::MetalContext;

type SnapshotOf<M> = <<M as LanguageModel>::State as DecodeStateApi>::Snapshot;

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
        let lcp = common_prefix_len(&self.tokens, prompt);
        let limit = lcp.min(prompt.len().checked_sub(1)?);
        if self.tokens.len() <= limit {
            return (!self.tokens.is_empty()).then_some(self.tokens.len());
        }
        self.checkpoints.iter().rev().map(|c| c.pos()).find(|&p| p > 0 && p <= limit)
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
}

pub struct SessionStore<M: LanguageModel> {
    entries: Vec<Session<M>>,
    budget_bytes: usize,
    max_sessions: usize,
    max_checkpoints: usize,
    clock: u64,
}

impl<M: LanguageModel> SessionStore<M> {
    pub fn new(budget_bytes: usize, max_sessions: usize, max_checkpoints: usize) -> Self {
        Self { entries: Vec::new(), budget_bytes, max_sessions, max_checkpoints: max_checkpoints.max(1), clock: 0 }
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
    /// only breaks ties between equally good candidates.
    pub fn acquire(
        &mut self,
        ctx: &MetalContext,
        model: &M,
        prompt: &[u32],
        cache_key: Option<&str>,
    ) -> Result<Acquired<M>> {
        ensure!(!prompt.is_empty(), "empty prompt");
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

        let Some((index, resume_at)) = best else {
            let state = model.new_state(ctx, prompt.len())?;
            let session = Session::new(state);
            self.trim(session.bytes());
            return Ok(Acquired { session, reused: 0, forked: false });
        };

        if resume_at == self.entries[index].tokens.len() {
            // Pure extension of the live end: take the session as is.
            let session = self.entries.swap_remove(index);
            ensure!(session.state.pos() == resume_at, "cached decode state out of step with its tokens");
            return Ok(Acquired { session, reused: resume_at, forked: false });
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
        self.trim(session.bytes());
        Ok(Acquired { session, reused: resume_at, forked: true })
    }

    /// Evicts least-recently-used sessions until `extra` more bytes fit the
    /// budget (or the store is empty).
    fn trim(&mut self, extra: usize) {
        while !self.entries.is_empty() && self.used_bytes() + extra > self.budget_bytes {
            let Some(victim) = self.lru_index() else { break };
            self.entries.swap_remove(victim);
        }
    }

    fn lru_index(&self) -> Option<usize> {
        self.entries.iter().enumerate().min_by_key(|(_, s)| s.last_used).map(|(i, _)| i)
    }

    /// Returns a session to the cache after a request. Its state must sit at
    /// `tokens.len()`. Old checkpoints beyond the per-session cap are dropped
    /// (the newest are the ones the next request most likely resumes from),
    /// and the store is trimmed to budget and count.
    pub fn release(&mut self, mut session: Session<M>, cache_key: Option<&str>) {
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
            self.entries.swap_remove(victim);
        }
    }
}

pub fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
#[path = "../../tests/unit/serve/session.rs"]
mod tests;
