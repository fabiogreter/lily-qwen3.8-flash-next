//! Speculative decoding with the checkpoint's multi-token-prediction head.
//!
//! A step verifies the pending token plus `k` drafts in one batched trunk
//! pass (`BatchMode::Verify`), which draws one token per row and records the
//! GDN state after every row. The host compares the draws with the drafts;
//! the draft pass then rolls the recurrent state back to the accepted prefix
//! (state copy plus conv-window rewind, no recomputation), catches the draft
//! head up on the accepted rows with real trunk hiddens, and chains `k` new
//! proposals from the head's own residual. Attention caches are position
//! indexed, so rejected rows are simply overwritten later.
//!
//! Outputs are exact: every emitted token is a draw from the trunk's own
//! distribution for its prefix; the drafts only decide how many rows a pass
//! can confirm.

use std::time::Instant;

use anyhow::{Result, ensure};

use crate::engine::{DecodeStateApi, NextStep};
use crate::kernels::gdn::{GDN_HEAD_DIM, conv_window_rollback};
use crate::kernels::elementwise::copy_words;
use crate::kernels::sample::{SamplingParams, sample_f32};
use crate::metal::{EncodedPass, MetalContext, PendingPass};
use crate::moe_ffn::{prefix_rows, project_mat};
use crate::tensor::Tensor;

use super::model::{BatchMode, DecodeState, LayerState, MAX_DRAFTS, Prepared, Qwen4ExpModel, Scratch, SpecPending};
use super::ngram::NgramHasher;

const GREEDY: SamplingParams = SamplingParams::greedy();

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
    ) -> Result<Vec<u32>> {
        ensure!(state.spec.is_none(), "verify while a speculative step is pending");
        ensure!(drafts.len() <= MAX_DRAFTS, "{} drafts exceed {MAX_DRAFTS}", drafts.len());
        ensure!(s.spec.is_some() && self.weights.mtp.is_some(), "verify without a draft head");
        let mut tokens = Vec::with_capacity(drafts.len() + 1);
        tokens.push(pending);
        tokens.extend_from_slice(drafts);
        let m = tokens.len();
        let k = drafts.len();
        // Room for this pass, the passes encoded ahead for the step after it
        // (a verify of 1 + k rows at any of m positions, plus the head's
        // chained rows), and the step after that, so a parked pass (which
        // cannot grow the caches) still finds room to encode ahead. Growing is
        // only possible when nothing runs: the unparked case.
        let ahead_rows = 2 * (1 + k);
        if parked.is_none() {
            self.ensure_room(ctx, state, s, m + ahead_rows)?;
        }
        let can_prepare = k > 0 && state.pos + m + ahead_rows + MAX_DRAFTS <= state.capacity();
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
        // While the GPU verifies, encode what follows for every possible
        // outcome; finish_speculation then only fills in tokens and commits.
        let prepared = if can_prepare { self.prepare_variants(ctx, state, s, m, k, params, step0)? } else { Vec::new() };
        let mut pace = s.sync.pace_verify.get();
        let done = committed.wait_retain_paced(&mut pace)?;
        s.sync.pace_verify.set(pace);
        if std::env::var_os("LILY_PROFILE").is_some() {
            let t = done.timing()?;
            eprintln!("profile verify m={m}: gpu {:.2} ms", (t.gpu_end_secs - t.gpu_start_secs) * 1e3);
        }
        // Same bookkeeping as a prefill chunk; the draft pass rolls it back.
        state.pos += m;
        if let Some(pst) = &mut state.ple {
            pst.hist = NgramHasher::advance(pst.hist, &tokens);
        }
        state.conv_slot = 1 - state.conv_slot;
        let sampled = s.spec.as_ref().expect("spec scratch").verify_tokens.to_u32()?[..m].to_vec();
        state.spec = Some(SpecPending {
            pos_before,
            hist_before,
            tokens,
            sampled: sampled.clone(),
            uses_penalties: params.uses_penalties(),
            prepared,
        });
        Ok(sampled)
    }

    /// Encodes, for each accepted count `a` in `0..m` of the running verify
    /// pass, the draft pass (rollback to `a`, head catch-up over rows `0..=a`,
    /// `k` chained drafts, next ids) and the next verify pass (1 + k rows at
    /// the position after `a`, parked on the step sync). The state is shown
    /// to the encoders as it will be after the verify pass; nothing is
    /// committed and the inputs written by the host (head ids, pending token)
    /// are filled in at commit time.
    #[allow(clippy::too_many_arguments)]
    fn prepare_variants(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &Scratch,
        m: usize,
        k: usize,
        params: &SamplingParams,
        step0: usize,
    ) -> Result<Vec<Prepared>> {
        let wide = self.config.hc_width();
        let capacity = s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("prefill scratch missing"))?;
        let (pos_before, conv_before) = (state.pos, state.conv_slot);
        let park = s.sync.peek();
        let mut prepared = Vec::with_capacity(m);
        let result = (|| -> Result<()> {
            state.conv_slot = 1 - conv_before;
            for a in 0..m {
                state.pos = pos_before + a + 1;
                let hidden = capacity.hyper.view(0, &[a + 1, wide])?;
                let keep = capacity.hyper.view(a * wide, &[wide])?;
                let rollback = (a + 1 < m).then_some((a, m));
                let next_ids = capacity.ids.view(1, &[k])?;
                let draft = self.encode_draft(ctx, state, s, &hidden, a + 1, pos_before, Some(&keep), rollback, k, Some(&next_ids))?.detach();
                let ps = capacity.rows(1 + k)?;
                let mode = BatchMode::Verify { params, step0: step0 + a + 1, park: Some(park) };
                let verify = self.encode_batch(ctx, state, s, &ps, mode)?.detach();
                prepared.push(Prepared { draft, verify, park, drafts: k });
            }
            Ok(())
        })();
        state.pos = pos_before;
        state.conv_slot = conv_before;
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
        drafts: usize,
    ) -> Result<(Vec<u32>, Option<PendingPass<'a>>)> {
        let entered = Instant::now();
        let mut pending = state.spec.take().ok_or_else(|| anyhow::anyhow!("finish_speculation without a verify pass"))?;
        let m = pending.tokens.len();
        ensure!(accepted < m, "accepted {accepted} of {} drafts", m - 1);
        let wide = self.config.hc_width();
        let drafts = drafts.min(MAX_DRAFTS);

        // Host-side rollback: position, hash history, penalty counts of the
        // rejected rows' draws.
        state.pos = pending.pos_before + accepted + 1;
        if let (Some(pst), Some(before)) = (&mut state.ple, pending.hist_before) {
            pst.hist = NgramHasher::advance(before, &pending.tokens[..=accepted]);
        }
        if pending.uses_penalties {
            for &token in &pending.sampled[accepted + 1..] {
                s.sampler.uncount(token);
            }
        }

        // The head's rows: trunk hiddens 0..=accepted of the verify pass (still
        // in the prefill scratch), paired with the tokens that followed them.
        let want = if next.is_some() { drafts } else { 0 };
        let rows = accepted + usize::from(next.is_some());
        let mut tokens: Vec<u32> = pending.tokens[1..=accepted].to_vec();
        tokens.extend(next.map(|n| n.token));

        let ahead = match next {
            Some(n) if want > 0 && pending.prepared.len() == m && pending.prepared[accepted].drafts == want => {
                Some((n, pending.prepared.swap_remove(accepted)))
            }
            _ => None,
        };
        let (committed, parked) = match ahead {
            Some((n, prep)) => {
                // Encoded ahead: fill in the pending token (the draft pass
                // writes the drafts after it) and the head's ids, commit both.
                let s = &*s;
                let capacity = s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("prefill scratch missing"))?;
                capacity.ids.view(0, &[1])?.write_bytes(bytemuck::cast_slice(&[n.token]))?;
                let committed = self.commit_draft(s, prep.draft.attach(ctx), &tokens)?;
                let value = s.sync.arm()?;
                ensure!(value == prep.park, "step sync value {value} does not match the prepared pass ({})", prep.park);
                match prep.verify.attach(ctx).commit() {
                    Ok(pass) => (committed, Some(pass)),
                    Err(e) => {
                        s.sync.release()?;
                        return Err(e);
                    }
                }
            }
            None => {
                // Nothing (fitting) encoded ahead, or the generation ends
                // here: encode now. The next verify pass (1 + drafts rows)
                // needs room before its graph is encoded; growing now keeps
                // the GPU-idle requirement (nothing has been committed yet).
                if want > 0 {
                    self.ensure_room(ctx, state, s, 1 + want)?;
                }
                let s = &*s;
                let capacity = s.prefill.as_ref().ok_or_else(|| anyhow::anyhow!("prefill scratch missing"))?;
                let hidden = capacity.hyper.view(0, &[rows.max(1), wide])?;
                let keep = capacity.hyper.view(accepted * wide, &[wide])?;
                let rollback = (accepted + 1 < m).then_some((accepted, m));
                let next_ids = match next {
                    Some(n) if want > 0 => {
                        capacity.ids.view(0, &[1])?.write_bytes(bytemuck::cast_slice(&[n.token]))?;
                        Some(capacity.ids.view(1, &[want])?)
                    }
                    _ => None,
                };
                let encoded = self.encode_draft(ctx, state, s, &hidden, rows, pending.pos_before, Some(&keep), rollback, want, next_ids.as_ref())?;
                let committed = self.commit_draft(s, encoded, &tokens)?;
                let parked = match next {
                    Some(n) if want > 0 => {
                        let ps = capacity.rows(1 + want)?;
                        let value = s.sync.arm()?;
                        let mode = BatchMode::Verify { params: n.params, step0: n.step0, park: Some(value) };
                        match self.encode_batch(ctx, state, s, &ps, mode).and_then(|pass| pass.commit()) {
                            Ok(pass) => Some(pass),
                            Err(e) => {
                                s.sync.release()?;
                                return Err(e);
                            }
                        }
                    }
                    _ => None,
                };
                (committed, parked)
            }
        };
        drop(pending.prepared);
        let s = &*s;
        if std::env::var_os("LILY_PROFILE").is_some() {
            eprintln!("profile host finish: both passes committed {:.2} ms after entry", entered.elapsed().as_secs_f64() * 1e3);
        }
        let proposals = self.collect_drafts(s, committed, rows, want, accepted + 1 < m)?;
        if parked.is_some() {
            // The parked verify pass started when the draft pass finished.
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
        let encoded = self.encode_draft(ctx, state, s, &hidden, 1, state.pos - 1, None, None, drafts, None)?;
        let committed = self.commit_draft(s, encoded, &[first])?;
        self.collect_drafts(s, committed, 1, drafts, false)
    }

    /// Waits for a committed draft pass and reads its proposals.
    fn collect_drafts(&self, s: &Scratch, committed: PendingPass<'_>, rows: usize, drafts: usize, rollback: bool) -> Result<Vec<u32>> {
        let mut pace = s.sync.pace_draft.get();
        let done = committed.wait_retain_paced(&mut pace)?;
        s.sync.pace_draft.set(pace);
        if std::env::var_os("LILY_PROFILE").is_some() {
            let t = done.timing()?;
            eprintln!("profile draft rows={rows} drafts={drafts} rollback={rollback}: gpu {:.2} ms", (t.gpu_end_secs - t.gpu_start_secs) * 1e3);
        }
        if drafts == 0 {
            return Ok(Vec::new());
        }
        let sp = s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no spec scratch"))?;
        Ok(sp.draft_tokens.to_u32()?[..drafts].to_vec())
    }

    /// Writes the head's catch-up ids (`tokens`, one per row the pass was
    /// encoded for) and commits a draft pass encoded by [`Self::encode_draft`].
    fn commit_draft<'a>(&self, s: &Scratch, encoded: EncodedPass<'a>, tokens: &[u32]) -> Result<PendingPass<'a>> {
        if !tokens.is_empty() {
            let sp = s.spec.as_ref().ok_or_else(|| anyhow::anyhow!("no spec scratch"))?;
            sp.mtp_ids.view(0, &[tokens.len()])?.write_bytes(bytemuck::cast_slice(tokens))?;
        }
        let mut pace = s.sync.pace_draft.get();
        pace.begin(Instant::now());
        s.sync.pace_draft.set(pace);
        encoded.commit()
    }

    /// Encodes the draft pass without committing: optional rollback of the
    /// recurrent state to `accepted` of `m` verified rows, the head's
    /// catch-up over `rows` rows (`hidden` rows paired with the ids
    /// [`Self::commit_draft`] writes, head positions from `pos0`), `keep`
    /// copied into the state's hidden, and `drafts` chained proposals into
    /// the spec scratch, also copied into `next_ids` (the next verify pass's
    /// ids after its pending token) when given.
    #[allow(clippy::too_many_arguments)]
    fn encode_draft<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        hidden: &Tensor,
        rows: usize,
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
        ensure!(drafts == 0 || rows > 0, "drafts need a row to follow");

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
            self.mtp_block(ctx, &pass, mtp, mst, hidden, &ids, pos0, s, &ps)?;
            if drafts > 0 {
                let hyper = ps.mtp_hyper.as_ref().expect("draft head scratch");
                let ps1 = capacity.rows(1)?;
                let mut last = hyper.view((rows - 1) * wide, &[1, wide])?;
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
                        self.mtp_block(ctx, &pass, mtp, mst, &last, &out, pos0 + rows + i, s, &ps1)?;
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
