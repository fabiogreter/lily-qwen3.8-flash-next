//! Speculative decoding with the checkpoint's multi-token-prediction head.
//!
//! A step verifies the pending token plus `k` drafts in one batched trunk
//! pass (`BatchMode::Verify`), which draws one token per row and records the
//! GDN state after every row. The draft pass follows it on the queue without
//! the host in between: a first dispatch compares the draws with the drafts
//! and writes the accepted count and everything derived from it (positions,
//! indexer blocks) into a control block, which the pass's later dispatches
//! read as inline arguments (`kernels::spec`, `kernels::Arg::Gpu`). The pass
//! rolls the recurrent state back to the accepted prefix (state copy plus
//! conv-window rewind, no recomputation), catches the draft head up on every
//! verified row with real trunk hiddens (rows past the accepted one are
//! position-indexed garbage the chain overwrites before anything reads them),
//! chains `k` new proposals from the head's own residual at GPU-supplied
//! positions, and writes the next verify pass's ids. Attention caches are
//! position indexed, so rejected rows are simply overwritten later.
//!
//! The host meanwhile reads the draws, emits tokens, and commits the next
//! verify pass (encoded ahead for every possible accepted count while the
//! verify pass ran) parked behind the draft pass. When a generation ends at a
//! row before the one the GPU accepted (a stop token among accepted drafts),
//! a small host-encoded pass rolls the state back further.
//!
//! Outputs are exact: every emitted token is a draw from the trunk's own
//! distribution for its prefix; the drafts only decide how many rows a pass
//! can confirm.

use std::time::Instant;

use anyhow::{Result, ensure};

use crate::engine::{DecodeStateApi, NextStep};
use crate::kernels::elementwise::copy_words;
use crate::kernels::gdn::{GDN_HEAD_DIM, conv_window_rollback};
use crate::kernels::sample::{SamplingParams, sample_f32};
use crate::kernels::spec::{SLOT_ACCEPTED, SLOT_KEEP, copy_row, ctrl_word, slot_block, slot_count, slot_pos, spec_accept};
use crate::kernels::{Arg, Pos};
use crate::metal::{EncodedPass, MetalContext, PendingPass};
use crate::moe_ffn::{prefix_rows, project_mat};
use crate::tensor::Tensor;

use super::model::{AttnPos, BatchMode, DecodeState, LayerState, MAX_DRAFTS, Prepared, Qwen4ExpModel, Scratch, SpecPending};
use super::ngram::NgramHasher;

const GREEDY: SamplingParams = SamplingParams::greedy();

fn profiling() -> bool {
    std::env::var_os("LILY_PROFILE").is_some()
}

impl Qwen4ExpModel {
    /// Draft tokens per step, or 0 without the head.
    pub fn max_drafts(&self) -> usize {
        if self.weights.mtp.is_some() { MAX_DRAFTS } else { 0 }
    }

    /// Makes room for a verify pass of `m` rows plus the draft pass after it.
    fn ensure_room(&self, ctx: &MetalContext, state: &mut DecodeState, s: &mut Scratch, m: usize) -> Result<()> {
        state.ensure_capacity(ctx, state.pos + m + MAX_DRAFTS)?;
        self.ensure_prefill_scratch(ctx, s, m)
    }

    /// See [`crate::engine::LanguageModel::verify`].
    #[allow(clippy::too_many_arguments)]
    pub fn verify<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        pending: u32,
        drafts: &[u32],
        params: &SamplingParams,
        step0: usize,
        parked: Option<PendingPass<'a>>,
        next_drafts: usize,
    ) -> Result<(Vec<u32>, PendingPass<'a>)> {
        ensure!(state.spec.is_none(), "verify while a speculative step is pending");
        ensure!(drafts.len() <= MAX_DRAFTS, "{} drafts exceed {MAX_DRAFTS}", drafts.len());
        ensure!(s.spec.is_some() && self.weights.mtp.is_some(), "verify without a draft head");
        let mut tokens = Vec::with_capacity(drafts.len() + 1);
        tokens.push(pending);
        tokens.extend_from_slice(drafts);
        let m = tokens.len();
        let chain = next_drafts.min(MAX_DRAFTS);
        // Room for this pass, the draft pass behind it (its chain rows), the
        // next verify pass (1 + chain rows at any of m positions) and the draft
        // pass after that, so a parked pass (which cannot grow the caches)
        // still finds room. Growing is only possible when nothing runs: the
        // unparked case.
        let ahead_rows = 2 * (1 + chain);
        if parked.is_none() {
            self.ensure_room(ctx, state, s, m + ahead_rows)?;
        }
        let can_prepare = chain > 0 && state.pos + m + ahead_rows + MAX_DRAFTS <= state.capacity();
        let s = &*s;
        // Whatever happens below, a parked pass must be let go.
        let _release = parked.as_ref().map(|_| s.sync.release_on_drop());
        let capacity = s.prefill.as_ref().expect("prefill scratch just ensured");
        // The token-dependent inputs. A parked pass is already running with
        // ids the draft pass wrote; only the n-gram rows are staged here, read
        // after its wait. Otherwise both are written now and the pass encoded.
        let ps = if parked.is_some() { capacity.rows(m)? } else { capacity.chunk(&tokens)? };
        let ple_w = self.weights.layers.iter().find_map(|l| l.ple.as_deref());
        if let (Some(w), Some(p), Some(pst)) = (ple_w, &ps.ple, &state.ple) {
            self.stage_ngram(w, p, &tokens, pst.hist)?;
        }
        let pos_before = state.pos;
        let hist_before = state.ple.as_ref().map(|p| p.hist);
        let committed = match parked {
            Some(pass) => {
                s.sync.release()?;
                pass
            }
            None => {
                let pass = self.encode_batch(ctx, state, s, &ps, BatchMode::Verify { params, step0, park: None })?.commit()?;
                let mut pace = s.sync.pace_verify.get();
                pace.begin(Instant::now());
                s.sync.pace_verify.set(pace);
                pass
            }
        };
        // Same bookkeeping as a prefill chunk; the draft pass rolls it back.
        // Done before the draft pass is encoded: it sees the state as the
        // verify pass leaves it.
        state.pos += m;
        if let Some(pst) = &mut state.ple {
            pst.hist = NgramHasher::advance(pst.hist, &tokens);
        }
        state.conv_slot = 1 - state.conv_slot;
        // The draft pass follows on the queue; the GPU reads the accepted
        // count itself.
        let draft = self.encode_draft_selected(ctx, state, s, pos_before, m, chain)?.commit()?;
        // While the GPU works, encode the next verify pass for every possible
        // outcome; finish_speculation then only commits the matching one.
        let prepared = if can_prepare { self.prepare_variants(ctx, state, s, pos_before, m, chain, params, step0)? } else { Vec::new() };
        let mut pace = s.sync.pace_verify.get();
        let done = committed.wait_retain_paced(&mut pace)?;
        s.sync.pace_verify.set(pace);
        // The draft pass starts as the verify pass ends.
        let mut pace = s.sync.pace_draft.get();
        pace.begin(Instant::now());
        s.sync.pace_draft.set(pace);
        let verify_gpu_end = if profiling() {
            let t = done.timing()?;
            eprintln!("profile verify m={m}: gpu {:.2} ms", (t.gpu_end_secs - t.gpu_start_secs) * 1e3);
            Some(t.gpu_end_secs)
        } else {
            None
        };
        let sampled = s.spec.as_ref().expect("spec scratch").verify_tokens.to_u32()?[..m].to_vec();
        state.spec = Some(SpecPending {
            pos_before,
            hist_before,
            tokens,
            sampled: sampled.clone(),
            uses_penalties: params.uses_penalties(),
            chain,
            prepared,
            verify_gpu_end,
        });
        Ok((sampled, draft))
    }

    /// Encodes, for each accepted count `a` in `0..m` of the running verify
    /// pass, the next verify pass (1 + `chain` rows at the position after
    /// `a`, parked on the step sync). `state` is as the verify pass leaves it
    /// (position advanced by `m`); the variants see the position after `a`.
    /// Nothing is committed; the ids are written by the draft pass.
    #[allow(clippy::too_many_arguments)]
    fn prepare_variants(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &Scratch,
        pos_before: usize,
        m: usize,
        chain: usize,
        params: &SamplingParams,
        step0: usize,
    ) -> Result<Vec<Prepared>> {
        let capacity = s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("prefill scratch missing"))?;
        let pos_after = state.pos;
        let park = s.sync.peek();
        let mut prepared = Vec::with_capacity(m);
        let result = (|| -> Result<()> {
            for a in 0..m {
                state.pos = pos_before + a + 1;
                let ps = capacity.rows(1 + chain)?;
                let mode = BatchMode::Verify { params, step0: step0 + a + 1, park: Some(park) };
                let verify = self.encode_batch(ctx, state, s, &ps, mode)?.detach();
                prepared.push(Prepared { verify, park, drafts: chain });
            }
            Ok(())
        })();
        state.pos = pos_after;
        result?;
        Ok(prepared)
    }

    /// See [`crate::engine::LanguageModel::finish_speculation`].
    pub fn finish_speculation<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        accepted: usize,
        next: Option<NextStep<'_>>,
        draft: PendingPass<'a>,
    ) -> Result<(Vec<u32>, Option<PendingPass<'a>>)> {
        let entered = Instant::now();
        let mut pending = state.spec.take().ok_or_else(|| anyhow::anyhow!("finish_speculation without a verify pass"))?;
        let m = pending.tokens.len();
        ensure!(accepted < m, "accepted {accepted} of {} drafts", m - 1);
        // What the GPU accepted: the leading rows whose draw equals the draft
        // after them. The engine may stop earlier (a stop token among them),
        // never later.
        let gpu_accepted = pending.sampled.iter().zip(&pending.tokens[1..]).take_while(|(draw, draft)| draw == draft).count();
        ensure!(accepted <= gpu_accepted, "accepted {accepted} rows but only {gpu_accepted} drafts were confirmed");
        let chain = pending.chain;

        // Host-side rollback: position, hash history, penalty counts of the
        // rejected rows' draws. (The running draft pass never touches the
        // counts: its draws are greedy.)
        state.pos = pending.pos_before + accepted + 1;
        if let (Some(pst), Some(before)) = (&mut state.ple, pending.hist_before) {
            pst.hist = NgramHasher::advance(before, &pending.tokens[..=accepted]);
        }
        if pending.uses_penalties {
            for &token in &pending.sampled[accepted + 1..] {
                s.sampler.uncount(token);
            }
        }

        let (proposals, parked) = match next {
            None => {
                // The generation ends here: the draft pass's proposals are not
                // needed, but its state changes must be complete before the
                // state is used again.
                self.collect_drafts(s, draft, 0, &pending)?;
                if accepted < gpu_accepted {
                    // The draft pass rolled back to the GPU's count; roll back
                    // further to the engine's.
                    let s = &*s;
                    let capacity = s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("prefill scratch missing"))?;
                    let wide = self.config.hc_width();
                    let hidden = capacity.hyper.view(0, &[1, wide])?;
                    let keep = capacity.hyper.view(accepted * wide, &[wide])?;
                    self.encode_draft(ctx, state, s, &hidden, 0, 0, pending.pos_before, Some(&keep), Some((accepted, m)), 0, None)?.commit()?.wait()?;
                }
                (Vec::new(), None)
            }
            Some(n) => {
                ensure!(accepted == gpu_accepted, "the step continues past a row the GPU did not confirm");
                ensure!(chain > 0, "the draft pass proposed nothing to verify next");
                let ahead = (pending.prepared.len() == m && pending.prepared[accepted].drafts == chain).then(|| pending.prepared.swap_remove(accepted));
                match ahead {
                    Some(prep) => {
                        // Encoded ahead, ids written by the draft pass: commit
                        // it parked, then read the proposals.
                        let s = &*s;
                        let value = s.sync.arm()?;
                        ensure!(value == prep.park, "step sync value {value} does not match the prepared pass ({})", prep.park);
                        let parked = match prep.verify.attach(ctx).commit() {
                            Ok(pass) => pass,
                            Err(e) => {
                                s.sync.release()?;
                                return Err(e);
                            }
                        };
                        let proposals = self.collect_drafts(s, draft, chain, &pending)?;
                        (proposals, Some(parked))
                    }
                    None => {
                        // Nothing (fitting) encoded ahead: wait for the draft
                        // pass, then make room (only possible with the GPU
                        // idle) and encode the next verify pass now.
                        let proposals = self.collect_drafts(s, draft, chain, &pending)?;
                        self.ensure_room(ctx, state, s, 1 + chain)?;
                        let s = &*s;
                        let capacity = s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("prefill scratch missing"))?;
                        let ps = capacity.rows(1 + chain)?;
                        let value = s.sync.arm()?;
                        let mode = BatchMode::Verify { params: n.params, step0: n.step0, park: Some(value) };
                        match self.encode_batch(ctx, state, s, &ps, mode).and_then(|pass| pass.commit()) {
                            Ok(pass) => (proposals, Some(pass)),
                            Err(e) => {
                                s.sync.release()?;
                                return Err(e);
                            }
                        }
                    }
                }
            }
        };
        drop(pending.prepared);
        let s = &*s;
        if profiling() {
            eprintln!("profile host finish: done {:.2} ms after entry", entered.elapsed().as_secs_f64() * 1e3);
        }
        if parked.is_some() {
            // The parked verify pass starts when the draft pass finishes.
            let mut pace = s.sync.pace_verify.get();
            pace.begin(Instant::now());
            s.sync.pace_verify.set(pace);
        }
        Ok((proposals, parked))
    }

    /// See [`crate::engine::LanguageModel::draft_initial`].
    pub fn draft_initial(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        first: u32,
        drafts: usize,
    ) -> Result<Vec<u32>> {
        ensure!(state.spec.is_none(), "draft_initial during a pending speculative step");
        ensure!(state.pos > 0, "draft_initial before any token was fed");
        let wide = self.config.hc_width();
        let drafts = drafts.min(MAX_DRAFTS);
        self.ensure_room(ctx, state, s, 1 + drafts)?;
        let hidden = state.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no draft head state"))?.hidden.view(0, &[1, wide])?;
        let encoded = self.encode_draft(ctx, state, s, &hidden, 1, 0, state.pos - 1, None, None, drafts, None)?;
        let sp = s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no spec scratch"))?;
        sp.mtp_ids.view(0, &[1])?.write_bytes(bytemuck::cast_slice(&[first]))?;
        let mut pace = s.sync.pace_draft.get();
        pace.begin(Instant::now());
        s.sync.pace_draft.set(pace);
        let committed = encoded.commit()?;
        let done = committed.wait_retain_paced(&mut pace)?;
        s.sync.pace_draft.set(pace);
        if profiling() {
            let t = done.timing()?;
            eprintln!("profile draft initial drafts={drafts}: gpu {:.2} ms", (t.gpu_end_secs - t.gpu_start_secs) * 1e3);
        }
        if drafts == 0 {
            return Ok(Vec::new());
        }
        Ok(sp.draft_tokens.to_u32()?[..drafts].to_vec())
    }

    /// Waits for the committed draft pass of `pending`'s step and reads its
    /// first `drafts` proposals.
    fn collect_drafts(&self, s: &Scratch, committed: PendingPass<'_>, drafts: usize, pending: &SpecPending) -> Result<Vec<u32>> {
        let mut pace = s.sync.pace_draft.get();
        let done = committed.wait_retain_paced(&mut pace)?;
        s.sync.pace_draft.set(pace);
        if profiling() {
            let t = done.timing()?;
            let gap = pending.verify_gpu_end.map_or(f64::NAN, |end| (t.gpu_start_secs - end) * 1e3);
            eprintln!(
                "profile draft m={} chain={}: gpu {:.2} ms, started {gap:.3} ms after the verify pass ended",
                pending.tokens.len(),
                pending.chain,
                (t.gpu_end_secs - t.gpu_start_secs) * 1e3
            );
        }
        if drafts == 0 {
            return Ok(Vec::new());
        }
        let sp = s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no spec scratch"))?;
        Ok(sp.draft_tokens.to_u32()?[..drafts].to_vec())
    }

    /// Encodes the draft pass that follows a verify pass of `m` rows at
    /// `pos0`, without committing. Nothing about the accepted count is known
    /// on the host: the pass decides it (`spec_accept`) and its dispatches
    /// read the control words. It rolls the recurrent state back to the
    /// accepted prefix, sets the head's hidden to the accepted row's trunk
    /// hidden, catches the head up on every verified row (ids are the trunk's
    /// draws: the accepted rows' successors, garbage past them), chains
    /// `chain` proposals into the spec scratch and writes the next verify
    /// pass's ids (accepted draw, then the proposals).
    fn encode_draft_selected<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        pos0: usize,
        m: usize,
        chain: usize,
    ) -> Result<EncodedPass<'a>> {
        let cfg = &self.config;
        let wide = cfg.hc_width();
        let ratio = cfg.indexer.compress_ratio;
        let mtp = self.weights.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no draft head"))?;
        let mst = state.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no draft head state"))?;
        let sp = s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no spec scratch"))?;
        let capacity = s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("prefill scratch missing"))?;
        ensure!((1..=MAX_DRAFTS + 1).contains(&m) && chain <= MAX_DRAFTS, "draft pass for {m} rows and {chain} proposals");
        ensure!(pos0 + m + chain <= state.capacity(), "no room for the draft head's chain");
        let ctrl = &sp.ctrl;
        let accepted = ctrl_word(ctrl, SLOT_ACCEPTED)?;
        let keep_rows = ctrl_word(ctrl, SLOT_KEEP)?;
        let chain_words: Vec<(Tensor, Tensor, Tensor)> = (0..chain)
            .map(|i| Ok((ctrl_word(ctrl, slot_pos(i))?, ctrl_word(ctrl, slot_block(i))?, ctrl_word(ctrl, slot_count(i))?)))
            .collect::<Result<_>>()?;

        let pass = ctx.begin_concurrent()?;
        let draws = sp.verify_tokens.view(0, &[m])?;
        spec_accept(ctx, &pass, &draws, &capacity.ids.view(0, &[m])?, ctrl, pos0, ratio, chain)?;
        pass.level_barrier(&[ctrl])?;

        // Rollback to the accepted prefix. With every draft accepted the
        // state is already right: the mid-state copy skips (that row does
        // not exist) and the conv rewind keeping all m rows reproduces the
        // window the verify pass wrote.
        if m > 1 {
            let heads = cfg.linear_num_value_heads;
            let per_state = heads * GDN_HEAD_DIM * GDN_HEAD_DIM;
            let mut gdn_index = 0usize;
            for lstate in &state.layers {
                if let LayerState::Gdn { state: st, conv_windows } = lstate {
                    let mid = sp.mid[gdn_index].view(0, &[m - 1, per_state])?;
                    copy_row(ctx, &pass, &mid, Arg::Gpu(&accepted), st)?;
                    let x = prefix_rows(&sp.conv_in[gdn_index], m)?;
                    conv_window_rollback(ctx, &pass, &conv_windows[1 - state.conv_slot], &x, &conv_windows[state.conv_slot], Pos::gpu(&keep_rows, 1, m))?;
                    gdn_index += 1;
                }
            }
            if let (Some(pst), Some(x)) = (&state.ple, &sp.ple_conv_in) {
                let x = prefix_rows(x, m)?;
                conv_window_rollback(ctx, &pass, &pst.conv_windows[1 - state.conv_slot], &x, &pst.conv_windows[state.conv_slot], Pos::gpu(&keep_rows, 1, m))?;
            }
            pass.level_barrier(&[])?;
        }

        // The head's next input: the trunk hidden of the accepted row.
        let hidden = capacity.hyper.view(0, &[m, wide])?;
        copy_row(ctx, &pass, &hidden, Arg::Gpu(&accepted), &mst.hidden)?;
        // Catch-up over every verified row; the head's caches are position
        // indexed, so the rows past the accepted one are overwritten by the
        // chain (and by the next step) before anything valid reads them.
        let ps = capacity.rows(m)?;
        self.mtp_block(ctx, &pass, mtp, mst, &hidden, &draws, AttnPos::host(pos0), s, &ps)?;
        if chain > 0 {
            let hyper = ps.mtp_hyper.as_ref().ok_or_else(|| anyhow::anyhow!("draft head without scratch"))?;
            copy_row(ctx, &pass, &hyper.view(0, &[m, wide])?, Arg::Gpu(&accepted), &sp.chain_in)?;
            pass.level_barrier(&[&sp.chain_in])?;
            let ps1 = capacity.rows(1)?;
            let mut last = sp.chain_in.view(0, &[1, wide])?;
            for (i, (pos, block, count)) in chain_words.iter().enumerate() {
                // Head: mixer over the last residual row, shared LM head, argmax.
                self.hc_read_batched(ctx, &pass, &mtp.mixer, &last, s, &ps1)?;
                let logits = sp.logits.view(0, &[1, cfg.vocab_size])?;
                project_mat(ctx, &pass, &ps1.hc.mixed, &self.weights.lm_head, &logits, &s.dequant)?;
                pass.level_barrier(&[&logits])?;
                let out = sp.draft_tokens.view(i, &[1])?;
                sample_f32(ctx, &pass, &logits.view(0, &[cfg.vocab_size])?, &s.sampler, &GREEDY, 0, &out)?;
                pass.level_barrier(&[&sp.draft_tokens])?;
                if i + 1 < chain {
                    // Chain: the head's own residual stands in for the trunk
                    // hidden of the token it just proposed, at position
                    // pos0 + accepted + 1 + i (the GPU knows which).
                    let at = AttnPos::gpu(pos, pos0 + 1 + i, pos0 + m + i, block, count);
                    self.mtp_block(ctx, &pass, mtp, mst, &last, &out, at, s, &ps1)?;
                    last = hyper.view(0, &[1, wide])?;
                }
            }
            // The next verify pass's ids: the accepted row's draw, then the
            // proposals. (The accept dispatch read the old ids long before.)
            copy_row(ctx, &pass, &sp.verify_tokens.view(0, &[m, 1])?, Arg::Gpu(&accepted), &capacity.ids.view(0, &[1])?)?;
            copy_words(ctx, &pass, &sp.draft_tokens.view(0, &[chain])?, &capacity.ids.view(1, &[chain])?)?;
        }
        s.sync.signal_done(&pass)?;
        pass.end()
    }

    /// Encodes a host-driven draft pass without committing: optional rollback
    /// of the recurrent state to `accepted` of `m` verified rows, the head's
    /// catch-up over `rows` rows (`hidden` rows paired with the ids in the
    /// spec scratch's `mtp_ids`, head positions from `pos0`), `keep` copied
    /// into the state's hidden, and `drafts` chained proposals from the
    /// residual of catch-up row `chain_row` (chain positions follow it) into
    /// the spec scratch, also copied into `next_ids` when given. Used for the
    /// first proposals of a request, for rolling back past the GPU's count,
    /// and as the reference the GPU-selected pass is tested against.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_draft<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        hidden: &Tensor,
        rows: usize,
        chain_row: usize,
        pos0: usize,
        keep: Option<&Tensor>,
        rollback: Option<(usize, usize)>,
        drafts: usize,
        next_ids: Option<&Tensor>,
    ) -> Result<EncodedPass<'a>> {
        let cfg = &self.config;
        let wide = cfg.hc_width();
        let mtp = self.weights.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no draft head"))?;
        let mst = state.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no draft head state"))?;
        let sp = s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no spec scratch"))?;
        let capacity = s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("prefill scratch missing"))?;
        ensure!(rows <= MAX_DRAFTS + 1, "too many catch-up rows");
        ensure!(drafts == 0 || chain_row < rows, "drafts need a catch-up row to follow");

        let pass = ctx.begin_concurrent()?;
        if let Some((accepted, m)) = rollback {
            let heads = cfg.linear_num_value_heads;
            let per_state = heads * GDN_HEAD_DIM * GDN_HEAD_DIM;
            let mut gdn_index = 0usize;
            for lstate in &state.layers {
                if let LayerState::Gdn { state: st, conv_windows } = lstate {
                    let mid = sp.mid[gdn_index].view(accepted * per_state, &[heads, GDN_HEAD_DIM, GDN_HEAD_DIM])?;
                    copy_words(ctx, &pass, &mid, st)?;
                    let x = prefix_rows(&sp.conv_in[gdn_index], m)?;
                    conv_window_rollback(ctx, &pass, &conv_windows[1 - state.conv_slot], &x, &conv_windows[state.conv_slot], accepted + 1)?;
                    gdn_index += 1;
                }
            }
            if let (Some(pst), Some(x)) = (&state.ple, &sp.ple_conv_in) {
                let x = prefix_rows(x, m)?;
                conv_window_rollback(ctx, &pass, &pst.conv_windows[1 - state.conv_slot], &x, &pst.conv_windows[state.conv_slot], accepted + 1)?;
            }
            pass.level_barrier(&[])?;
        }
        if let Some(keep) = keep {
            copy_words(ctx, &pass, keep, &mst.hidden)?;
        }
        if rows > 0 {
            let ps = capacity.rows(rows)?;
            let ids = sp.mtp_ids.view(0, &[rows])?;
            self.mtp_block(ctx, &pass, mtp, mst, hidden, &ids, AttnPos::host(pos0), s, &ps)?;
            if drafts > 0 {
                let hyper = ps.mtp_hyper.as_ref().expect("draft head scratch");
                let ps1 = capacity.rows(1)?;
                let mut last = hyper.view(chain_row * wide, &[1, wide])?;
                for i in 0..drafts {
                    // Head: mixer over the last residual row, shared LM head, argmax.
                    self.hc_read_batched(ctx, &pass, &mtp.mixer, &last, s, &ps1)?;
                    let logits = sp.logits.view(0, &[1, cfg.vocab_size])?;
                    project_mat(ctx, &pass, &ps1.hc.mixed, &self.weights.lm_head, &logits, &s.dequant)?;
                    pass.level_barrier(&[&logits])?;
                    let out = sp.draft_tokens.view(i, &[1])?;
                    sample_f32(ctx, &pass, &logits.view(0, &[cfg.vocab_size])?, &s.sampler, &GREEDY, 0, &out)?;
                    pass.level_barrier(&[&sp.draft_tokens])?;
                    if i + 1 < drafts {
                        // Chain: the head's own residual stands in for the trunk
                        // hidden of the token it just proposed.
                        self.mtp_block(ctx, &pass, mtp, mst, &last, &out, AttnPos::host(pos0 + chain_row + 1 + i), s, &ps1)?;
                        last = hyper.view(0, &[1, wide])?;
                    }
                }
                if let Some(ids) = next_ids {
                    copy_words(ctx, &pass, &sp.draft_tokens.view(0, &[drafts])?, ids)?;
                }
            }
        }
        s.sync.signal_done(&pass)?;
        pass.end()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/spec.rs"]
mod tests;
