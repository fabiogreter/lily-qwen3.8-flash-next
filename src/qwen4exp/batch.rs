//! Batched decode across sessions: one decode step over independent rows
//! (`LanguageModel::decode_rows`, the server's `--max-batch`).
//!
//! Row `r` is one session's token at that session's own position. The rows
//! sit at `r * width` of the batched scratch (`PrefillScratch::rows(m)`, as
//! in a verify pass), and every part of the graph that does not touch
//! per-session state runs once over all rows with the kernels the verify
//! pass runs at 2 to 4 rows: the projections (skinny GEMMs), the fused
//! small-batch hyper-connection reads, the injects, the norms, the small-m
//! MoE and the LM head. What reads or writes a session's state (the GDN
//! state and conv window, the PLE conv window, the attention and indexer
//! caches, the position and rope delta, the draw with the row's own sampler
//! settings and penalty counts) is the decode step's own kernel dispatched
//! once per row with that row's buffers: Metal 4 binds every dispatch
//! through its own argument table entries, so per-row cache pointers need no
//! new kernel. The rows of one dependency level are encoded between the same
//! level barriers and run concurrently; every per-row writer has its own
//! output (a row view, the row's own state, or per-row scratch in
//! [`BatchScratch`]), so no two dispatches of a level write one buffer.
//!
//! The recurrent state is updated in place exactly as by the decode step
//! (the conv windows shift in place, `conv_slot` stays), so a row's state
//! afterwards is a decode state like any other. With the draft head loaded,
//! the head is caught up on every row (the prefill catch-up for a one-token
//! chunk), so a session can return to speculative decoding afterwards with
//! complete head caches.
//!
//! Numerics: the batched kernels reduce in a different order than the
//! decode step's GEMVs, so a row agrees with the same session decoded alone
//! up to bf16 rounding, as a verify row does (`docs/architecture.md`,
//! "Speculative decoding"). Nothing here has run on a GPU yet
//! (`docs/continuous-batching-draft.md`).

use super::*;
use crate::engine::{BatchRow, CountsSlot};

/// Per-slot and per-row buffers of a batched decode step, allocated by the
/// first one.
pub(in crate::qwen4exp) struct BatchScratch {
    /// One per batch slot: the sampler (its penalty counts belong to the
    /// row that holds the slot) and, indexed by row instead, the attention
    /// split scratch.
    rows: Vec<RowScratch>,
    /// F32 `[slots, vocab]`: the rows' logits.
    logits: Tensor,
    /// U32 `[slots]`: the rows' draws.
    tokens: Tensor,
}

struct RowScratch {
    sampler: SamplerScratch,
    /// The sparse path's scores, selection and split partials for one query.
    qsa: QsaScratch,
    /// The dense split decode's partials, F32 `[NQ, splits, D]` and
    /// `[NQ, splits, 2]`.
    sdpa_partials: Tensor,
    sdpa_stats: Tensor,
}

/// One row's attention bindings: the caches of one session's layer (trunk
/// or draft head), the sequence index the row writes and the rope delta.
struct AttnRow<'t> {
    k_cache: &'t Tensor,
    v_cache: &'t Tensor,
    idx_keys: &'t Tensor,
    blk_keys: &'t Tensor,
    pos: usize,
    rope_delta: i64,
}

impl<'t> AttnRow<'t> {
    fn of(layer: &'t LayerState, pos: usize, rope_delta: i64) -> Result<Self> {
        let LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys } = layer else {
            anyhow::bail!("attention row over a recurrent layer's state")
        };
        Ok(Self { k_cache, v_cache, idx_keys, blk_keys, pos, rope_delta })
    }
}

impl Qwen4ExpModel {
    /// Rows a batched decode step takes: the fused small-batch
    /// hyper-connection read's limit (`HC_FUSED_MAX_ROWS`). None under the
    /// expert cache, whose MoE path and one-session budget are not built for
    /// it.
    pub(super) fn batch_rows(&self) -> usize {
        if self.expert_cache_stats().is_some() { 0 } else { HC_FUSED_MAX_ROWS }
    }

    fn ensure_batch_scratch(&self, ctx: &MetalContext, s: &mut Scratch) -> Result<()> {
        if s.batch.is_some() {
            return Ok(());
        }
        let cfg = &self.config;
        let slots = self.batch_rows().max(1);
        let (nq, hd) = (cfg.num_attention_heads, cfg.head_dim);
        // Dense decode only ever runs up to the indexer's dense limit.
        let splits = sdpa_split_scratch_splits(cfg.indexer.dense_limit());
        let rows = (0..slots)
            .map(|_| {
                Ok(RowScratch {
                    sampler: SamplerScratch::new(ctx, cfg.vocab_size)?,
                    qsa: QsaScratch::new(ctx, cfg, MAX_SEQ, 1)?,
                    sdpa_partials: Tensor::zeros(ctx, &[nq, splits, hd], DType::F32)?,
                    sdpa_stats: Tensor::zeros(ctx, &[nq, splits, 2], DType::F32)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        s.batch = Some(BatchScratch {
            rows,
            logits: Tensor::zeros(ctx, &[slots, cfg.vocab_size], DType::F32)?,
            tokens: Tensor::zeros(ctx, &[slots], DType::U32)?,
        });
        Ok(())
    }

    /// See [`crate::engine::LanguageModel::move_sampler_counts`].
    pub(super) fn move_counts(
        &self,
        ctx: &MetalContext,
        s: &mut Scratch,
        from: CountsSlot,
        to: CountsSlot,
    ) -> Result<()> {
        if from == to {
            return Ok(());
        }
        self.ensure_batch_scratch(ctx, s)?;
        let s = &*s;
        let batch = s.batch.as_ref().expect("allocated above");
        let sampler = |slot: CountsSlot| -> Result<&SamplerScratch> {
            match slot {
                CountsSlot::Engine => Ok(&s.sampler),
                CountsSlot::Batch(i) => {
                    batch.rows.get(i).map(|r| &r.sampler).ok_or_else(|| {
                        anyhow::anyhow!("batch slot {i} of {}", batch.rows.len())
                    })
                }
            }
        };
        sampler(to)?.copy_counts_from(sampler(from)?)
    }

    /// Stages the n-gram rows of a batched step: row `r` hashes `tokens[r]`
    /// after its own history `hists[r]`, in row order, so the gather sees one
    /// batch with sequential ids. The GPU must not be reading `p`.
    fn stage_ngram_rows(
        &self,
        w: &PleWeights,
        p: &PleScratch,
        tokens: &[u32],
        hists: &[[u32; 2]],
    ) -> Result<()> {
        let hasher = self.hasher.as_ref().expect("PLE weights without hasher");
        let mut ids = Vec::with_capacity(tokens.len() * hasher.heads());
        for (&token, &hist) in tokens.iter().zip(hists) {
            hasher.ids(&[token], hist, &mut ids);
        }
        match &w.table {
            NgramTable::Resident(_) => p.ids.write_bytes(bytemuck::cast_slice(&ids)),
            NgramTable::Paged(table) => {
                let stage = p.stage.as_ref().expect("paged table without staging");
                stage.fill(table, &ids)
            }
        }
    }

    /// See [`crate::engine::LanguageModel::decode_rows`].
    pub(super) fn decode_session_rows(
        &self,
        ctx: &MetalContext,
        s: &mut Scratch,
        rows: &mut [BatchRow<'_, DecodeState>],
    ) -> Result<Vec<u32>> {
        let m = rows.len();
        let slots = self.batch_rows();
        ensure!(
            (1..=slots).contains(&m),
            "a batched decode step takes 1 to {slots} rows, got {m}"
        );
        for (r, row) in rows.iter().enumerate() {
            let state = &*row.state;
            ensure!(
                state.spec.is_none(),
                "row {r}: batched decode during a pending speculative step"
            );
            ensure!(
                state.pos >= 1 && state.pos < state.capacity,
                "row {r}: position {} outside 1..{}",
                state.pos,
                state.capacity
            );
            ensure!(
                state.pos as i64 + state.rope_delta >= 0,
                "row {r}: rotary position {} + {} is negative",
                state.pos,
                state.rope_delta
            );
            ensure!(row.slot < slots, "row {r}: batch slot {} of {slots}", row.slot);
            ensure!(
                rows[..r].iter().all(|o| o.slot != row.slot),
                "row {r}: batch slot {} taken twice",
                row.slot
            );
            ensure!(
                state.mtp.is_some() == self.weights.mtp.is_some(),
                "row {r}: draft head state does not match the model"
            );
        }
        self.ensure_prefill_scratch(ctx, s, m)?;
        self.ensure_batch_scratch(ctx, s)?;
        let s = &*s;
        let capacity = s
            .prefill
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("prefill scratch missing after growth"))?;
        let bs = s.batch.as_ref().expect("allocated above");
        let ps = capacity.rows(m)?;
        let tokens: Vec<u32> = rows.iter().map(|r| r.token).collect();
        ps.ids.write_bytes(bytemuck::cast_slice(&tokens))?;
        let ple_w = self.weights.layers.iter().find_map(|l| l.ple.as_deref());
        if let (Some(w), Some(p)) = (ple_w, &ps.ple) {
            let hists = rows
                .iter()
                .map(|r| r.state.ple.as_ref().map(|pst| pst.hist))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| anyhow::anyhow!("PLE weights without PLE state"))?;
            self.stage_ngram_rows(w, p, &tokens, &hists)?;
        }
        // TODO(batch): park the next batched step like the decode loop does
        // (commit it before its n-gram rows are staged): its row set is only
        // known once this step's draws are, so for now every step pays the
        // host round trip (0.4 to 0.6 ms).
        let pass = self.begin_batched(ctx, m)?;
        pass.set_label("decode rows");
        self.encode_rows(ctx, &pass, rows, s, &ps, bs)?;
        pass.commit()?.wait()?;
        for row in rows.iter_mut() {
            let state = &mut *row.state;
            state.pos += 1;
            if let Some(pst) = &mut state.ple {
                pst.hist = NgramHasher::advance(pst.hist, &[row.token]);
            }
        }
        bs.tokens.view(0, &[m])?.to_u32()
    }

    /// The batched decode graph over `rows` (their tokens already in
    /// `ps.ids`, their n-gram rows staged): the trunk, the LM head, a draw
    /// per row into `bs.tokens[r]`, then the draft head's catch-up.
    fn encode_rows(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        rows: &[BatchRow<'_, DecodeState>],
        s: &Scratch,
        ps: &PrefillScratch,
        bs: &BatchScratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        let m = rows.len();
        let vocab = cfg.vocab_size;

        quant::gather_rows_q4(ctx, pass, &self.weights.embed_tokens, &ps.ids, &ps.x)?;
        pass.level_barrier(&[&ps.x])?;
        hc_broadcast_bf16(ctx, pass, &ps.x, &ps.hyper, h, g)?;
        pass.level_barrier(&[&ps.hyper])?;

        for (li, layer) in self.weights.layers.iter().enumerate() {
            if let (Some(w), Some(p)) = (&layer.ple, &ps.ple) {
                self.ple_rows(ctx, pass, w, p, rows, &ps.hyper, s)?;
            }
            self.hc_read_batched(ctx, pass, &layer.attn_hc, &ps.hyper, s, ps)?;
            match &layer.mixer {
                Mixer::Gdn(w) => self.gdn_rows(ctx, pass, w, li, rows, s, ps)?,
                Mixer::Attn(w) => {
                    let attn = rows
                        .iter()
                        .map(|r| {
                            AttnRow::of(
                                &r.state.layers[li],
                                r.state.pos,
                                r.state.rope_delta,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    self.attn_rows(
                        ctx,
                        pass,
                        w,
                        &attn,
                        cfg.rope_parameters.rope_theta,
                        s,
                        ps,
                        bs,
                    )?;
                }
            }
            pass.level_barrier(&[&ps.branch_out])?;
            hc_inject_bf16(ctx, pass, &ps.hyper, &ps.branch_out, &ps.hc.inj, h, g)?;
            pass.level_barrier(&[&ps.hyper])?;

            self.hc_read_batched(ctx, pass, &layer.mlp_hc, &ps.hyper, s, ps)?;
            prefill_moe(
                ctx,
                pass,
                &moe_dims(cfg),
                &layer.ffn,
                &PrefillMoeIo {
                    x: &ps.hc.mixed,
                    out: &ps.branch_out,
                    stack: &ps.stack,
                    mlp_gate: &ps.mlp_gate,
                    mlp_up: &ps.mlp_up,
                    mlp_act: &ps.mlp_act,
                    dequant: &s.dequant,
                },
                &ps.moe,
            )?;
            pass.level_barrier(&[&ps.branch_out])?;
            hc_inject_bf16(ctx, pass, &ps.hyper, &ps.branch_out, &ps.hc.inj, h, g)?;
            pass.level_barrier(&[&ps.hyper])?;
        }

        // The LM head over every row, then each row's draw with its own
        // settings, draw index and slot sampler (no two rows share one, so
        // the draws run in one level).
        self.hc_read_batched(ctx, pass, &self.weights.final_mixer, &ps.hyper, s, ps)?;
        let logits = bs.logits.view(0, &[m, vocab])?;
        project_mat(
            ctx,
            pass,
            &ps.hc.mixed,
            &self.weights.lm_head,
            &logits,
            &s.dequant,
        )?;
        pass.level_barrier(&[&logits])?;
        for (r, row) in rows.iter().enumerate() {
            sample_f32(
                ctx,
                pass,
                &bs.logits.view(r * vocab, &[vocab])?,
                &bs.rows[row.slot].sampler,
                row.draw.params,
                row.draw.step,
                &bs.tokens.view(r, &[1])?,
            )?;
        }
        pass.level_barrier(&[&bs.tokens])?;

        if let Some(mtp) = &self.weights.mtp {
            self.mtp_rows(ctx, pass, mtp, rows, s, ps, bs)?;
        }
        Ok(())
    }

    /// The n-gram embedding over the rows (`ple_batched` up to the conv),
    /// then each row's dilated conv step on its own window, in place as the
    /// decode step does it.
    #[allow(clippy::too_many_arguments)]
    fn ple_rows(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &PleWeights,
        p: &PleScratch,
        rows: &[BatchRow<'_, DecodeState>],
        hyper: &Tensor,
        s: &Scratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let ple = cfg.ple.as_ref().expect("PLE weights without PLE config");
        let (h, g, eps) = (cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps);
        let wide = cfg.hc_width();
        self.gather_ngram(ctx, pass, w, p, rows.len())?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            hyper,
            &w.norm_query,
            &p.query_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&p.emb, &p.query_n])?;
        project_mat(ctx, pass, &p.emb, &w.key_proj, &p.key, &s.dequant)?;
        project_mat(ctx, pass, &p.emb, &w.value_proj, &p.value, &s.dequant)?;
        pass.level_barrier(&[&p.key, &p.value])?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &p.key,
            &w.norm_key,
            &p.key_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&p.key_n])?;
        ple::ple_gate_value_bf16(
            ctx, pass, &p.key_n, &p.query_n, &p.value, &p.gated, h, g,
        )?;
        pass.level_barrier(&[&p.gated])?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &p.gated,
            &w.norm_conv,
            &p.gated_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&p.gated_n])?;
        for (r, row) in rows.iter().enumerate() {
            let pst = row.state.ple.as_ref().ok_or_else(|| {
                anyhow::anyhow!("row {r}: PLE weights without PLE state")
            })?;
            ple::ple_conv1d_step(
                ctx,
                pass,
                &pst.conv_windows[row.state.conv_slot],
                &p.gated_n.view(r * wide, &[wide])?,
                &w.conv_w,
                &p.gated.view(r * wide, &[wide])?,
                &hyper.view(r * wide, &[wide])?,
                ple.ngram_size,
            )?;
        }
        pass.level_barrier(&[hyper])
    }

    /// A GDN layer over the rows: the projections batched, each row's conv
    /// step and recurrent step on its own window and state (the decode
    /// step's kernels, in place).
    #[allow(clippy::too_many_arguments)]
    fn gdn_rows(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &GdnWeights,
        layer: usize,
        rows: &[BatchRow<'_, DecodeState>],
        s: &Scratch,
        ps: &PrefillScratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let c = cfg.gdn_conv_channels();
        let heads = cfg.linear_num_value_heads;
        let dim_v = heads * cfg.linear_value_head_dim;
        project_stack_or_slices(
            ctx,
            pass,
            &ps.hc.mixed,
            &w.in_proj,
            &ps.stack,
            [
                (&w.in_proj_qkv, &ps.qkv),
                (&w.in_proj_z, &ps.z),
                (&w.in_proj_a, &ps.a),
                (&w.in_proj_b, &ps.b),
            ],
            &s.dequant,
        )?;
        pass.level_barrier(&[&ps.qkv, &ps.z, &ps.a, &ps.b])?;
        let states = rows
            .iter()
            .enumerate()
            .map(|(r, row)| match &row.state.layers[layer] {
                LayerState::Gdn { state, conv_windows } => {
                    Ok((state, &conv_windows[row.state.conv_slot]))
                }
                LayerState::Attn { .. } => {
                    anyhow::bail!("row {r}: layer {layer} state is not recurrent")
                }
            })
            .collect::<Result<Vec<_>>>()?;
        for (r, (_, window)) in states.iter().enumerate() {
            conv1d_step(
                ctx,
                pass,
                window,
                &ps.qkv.view(r * c, &[c])?,
                &w.conv_w,
                &ps.qkv_conv.view(r * c, &[c])?,
            )?;
        }
        pass.level_barrier(&[&ps.qkv_conv])?;
        for (r, (state, _)) in states.iter().enumerate() {
            gdn_step_gated_fused(
                ctx,
                pass,
                &ps.qkv_conv.view(r * c, &[c])?,
                &ps.a.view(r * heads, &[heads])?,
                &ps.b.view(r * heads, &[heads])?,
                &w.a_log,
                &w.dt_bias,
                state,
                &ps.z.view(r * dim_v, &[dim_v])?,
                &w.norm_w,
                &ps.gdn_gated.view(r * dim_v, &[dim_v])?,
                self.gdn_scale,
                cfg.linear_num_key_heads,
                cfg.rms_norm_eps,
                self.gdn_gate,
            )?;
        }
        pass.level_barrier(&[&ps.gdn_gated])?;
        project_mat(ctx, pass, &ps.gdn_gated, &w.out_proj, &ps.branch_out, &s.dequant)
    }

    /// An attention layer over the rows (the trunk's, or the draft head's
    /// with its own `theta`): the projections and the output gate batched,
    /// everything between them per row against the row's caches at the row's
    /// position, with the decode step's kernels. Levels: Q and K prep with
    /// the cache scatters; the block keys of rows that complete a block;
    /// dense attention or sparse scores; the sparse selection; the sparse
    /// attention. Each row has its own split scratch (indexed by row).
    #[allow(clippy::too_many_arguments)]
    fn attn_rows(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &AttnWeights,
        rows: &[AttnRow<'_>],
        theta: f32,
        s: &Scratch,
        ps: &PrefillScratch,
        bs: &BatchScratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let eps = cfg.rms_norm_eps;
        let rot = cfg.rotary_dim();
        let idx = &cfg.indexer;
        let ratio = idx.compress_ratio;
        let (nq, nkv, hd) =
            (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let nh = idx.n_heads;
        let qk_width = (nh + 1) * INDEXER_D;
        ensure!(rows.len() <= bs.rows.len(), "more attention rows than batch scratch");

        project_stack_or_slices(
            ctx,
            pass,
            &ps.hc.mixed,
            &w.qkv_proj,
            &ps.stack,
            [(&w.q_proj, &ps.qg), (&w.k_proj, &ps.k_new), (&w.v_proj, &ps.v_new)],
            &s.dequant,
        )?;
        project_mat(
            ctx,
            pass,
            &ps.hc.mixed,
            &w.indexer.qk_proj,
            &ps.idx_qk,
            &s.dequant,
        )?;
        pass.level_barrier(&[&ps.qg, &ps.k_new, &ps.v_new, &ps.idx_qk])?;

        // Row views of the batched buffers, in the decode step's shapes.
        let q = |r: usize| ps.q.view(r * nq * hd, &[nq, hd]);
        let idx_q = |r: usize| ps.idx_q.view(r * nh * INDEXER_D, &[nh, INDEXER_D]);
        let idx_qk = |r: usize| ps.idx_qk.view(r * qk_width, &[qk_width]);
        let attn_o = |r: usize| ps.attn_o.view(r * nq * hd, &[nq, hd]);

        for (r, a) in rows.iter().enumerate() {
            ensure!(
                a.pos as i64 + a.rope_delta >= 0,
                "rotary position {} + {} is negative",
                a.pos,
                a.rope_delta
            );
            q_norm_rope_split_decode(
                ctx,
                pass,
                &ps.qg.view(r * nq * 2 * hd, &[nq, 2 * hd])?,
                &w.q_norm,
                &q(r)?,
                &ps.gate.view(r * nq * hd, &[nq, hd])?,
                rot,
                a.pos,
                theta,
                eps,
                a.rope_delta,
            )?;
            k_norm_rope_scatter_decode(
                ctx,
                pass,
                &ps.k_new.view(r * nkv * hd, &[nkv, hd])?,
                &w.k_norm,
                a.k_cache,
                rot,
                a.pos,
                theta,
                eps,
                a.rope_delta,
            )?;
            scatter_kv(
                ctx,
                pass,
                a.v_cache,
                &ps.v_new.view(r * nkv * hd, &[nkv, hd])?,
                a.pos,
            )?;
            qsa::qsa_prep_q(
                ctx,
                pass,
                &idx_qk(r)?,
                &w.indexer.q_norm,
                &idx_q(r)?,
                nh,
                rot,
                a.pos,
                theta,
                eps,
                Rope::Delta(a.rope_delta),
            )?;
            qsa::qsa_scatter_keys(ctx, pass, &idx_qk(r)?, a.idx_keys, nh, a.pos)?;
        }
        pass.level_barrier(&[&ps.q, &ps.gate, &ps.idx_q])?;

        // A row whose token completes an indexer block: its key becomes
        // selectable from this position on (beyond the dense limit).
        let mut completes = false;
        for a in rows {
            let len = a.pos + 1;
            if len.is_multiple_of(ratio) {
                qsa::qsa_block_keys(
                    ctx,
                    pass,
                    a.idx_keys,
                    &w.indexer.k_norm,
                    a.blk_keys,
                    ratio,
                    len / ratio - 1,
                    1,
                    rot,
                    theta,
                    eps,
                    Rope::Delta(a.rope_delta),
                )?;
                completes = true;
            }
        }
        if completes {
            pass.level_barrier(&[])?;
        }

        // Dense rows attend in one go (the split kernel and its combine,
        // with the barrier between them inside `sdpa_decode`); sparse rows
        // score their visible blocks.
        let sparse = |a: &AttnRow<'_>| a.pos + 1 > idx.dense_limit();
        for (r, a) in rows.iter().enumerate() {
            let rs = &bs.rows[r];
            if sparse(a) {
                qsa::qsa_scores(
                    ctx,
                    pass,
                    &idx_q(r)?,
                    a.blk_keys,
                    &rs.qsa.scores,
                    nh,
                    qsa::visible_blocks(a.pos, ratio),
                    a.pos,
                    ratio,
                )?;
            } else {
                sdpa_decode(
                    ctx,
                    pass,
                    &q(r)?,
                    a.k_cache,
                    a.v_cache,
                    &attn_o(r)?,
                    a.pos + 1,
                    self.attn_scale,
                    Some((&rs.sdpa_partials, &rs.sdpa_stats)),
                )?;
            }
        }
        if rows.iter().any(sparse) {
            pass.level_barrier(&[])?;
            for (r, a) in rows.iter().enumerate().filter(|(_, a)| sparse(a)) {
                let rs = &bs.rows[r];
                qsa::qsa_select_blocks(
                    ctx,
                    pass,
                    &rs.qsa.scores,
                    &rs.qsa.sel,
                    &rs.qsa.n_sel,
                    1,
                    qsa::visible_blocks(a.pos, ratio),
                    a.pos,
                    ratio,
                    idx.block_topk(),
                )?;
            }
            pass.level_barrier(&[])?;
            for (r, a) in rows.iter().enumerate().filter(|(_, a)| sparse(a)) {
                let rs = &bs.rows[r];
                qsa::qsa_attention(
                    ctx,
                    pass,
                    &q(r)?,
                    a.k_cache,
                    a.v_cache,
                    &rs.qsa.sel,
                    &rs.qsa.n_sel,
                    &attn_o(r)?,
                    &rs.qsa.split_scratch(),
                    1,
                    idx.block_topk(),
                    ratio,
                    a.pos,
                    self.attn_scale,
                )?;
            }
        }
        pass.level_barrier(&[&ps.attn_o])?;
        sigmoid_mul_bf16(ctx, pass, &ps.gate, &ps.attn_o, &ps.attn_gated)?;
        pass.level_barrier(&[&ps.attn_gated])?;
        project_mat(ctx, pass, &ps.attn_gated, &w.o_proj, &ps.branch_out, &s.dequant)
    }

    /// The draft head's catch-up over the rows: row `r` pairs the session's
    /// previous trunk hidden (the state's) with the token this step fed, at
    /// head position `pos - 1` against the session's head caches, and the
    /// state's hidden becomes this step's trunk hidden: what
    /// `mtp_catch_up` does for a one-token chunk. `mtp_block` with the
    /// attention per row.
    #[allow(clippy::too_many_arguments)]
    fn mtp_rows(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        mtp: &MtpWeights,
        rows: &[BatchRow<'_, DecodeState>],
        s: &Scratch,
        ps: &PrefillScratch,
        bs: &BatchScratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let (h, g, eps) = (cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps);
        let wide = cfg.hc_width();
        let m = rows.len();
        let hidden_in = ps
            .mtp_hidden_in
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("draft head without prefill scratch"))?;
        let hyper = ps
            .mtp_hyper
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("draft head without scratch"))?;
        ensure!(
            hidden_in.shape() == [m, wide] && hyper.shape() == [m, wide],
            "draft head scratch is not {m} rows wide"
        );
        let msts = rows
            .iter()
            .map(|r| r.state.mtp.as_ref())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow::anyhow!("draft head weights without head state"))?;
        for (r, mst) in msts.iter().enumerate() {
            copy_words(ctx, pass, &mst.hidden, &hidden_in.view(r * wide, &[wide])?)?;
        }
        pass.level_barrier(&[hidden_in])?;
        let theta = cfg.mtp.map_or(cfg.rope_parameters.rope_theta, |m| m.rope_theta);

        // `mtp_block`'s input: per stream fc_hidden(norm(stream)), shared
        // fc_embedding(norm(embed(token))).
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            hidden_in,
            &mtp.norm_hidden,
            &ps.hc.hn,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        quant::gather_rows_q4(ctx, pass, &self.weights.embed_tokens, &ps.ids, &ps.x)?;
        pass.level_barrier(&[&ps.hc.hn, &ps.x])?;
        let hn_streams = ps.hc.hn.view(0, &[m * g, h])?;
        let hyper_streams = hyper.view(0, &[m * g, h])?;
        project_mat(
            ctx,
            pass,
            &hn_streams,
            &mtp.fc_hidden,
            &hyper_streams,
            &s.dequant,
        )?;
        rmsnorm_bf16(
            ctx,
            pass,
            &ps.x,
            &mtp.norm_embedding,
            &ps.x,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[hyper, &ps.x])?;
        project_mat(ctx, pass, &ps.x, &mtp.fc_embedding, &ps.branch_out, &s.dequant)?;
        pass.level_barrier(&[&ps.branch_out])?;
        hc_broadcast_bf16(ctx, pass, &ps.branch_out, &ps.hc.up, h, g)?;
        pass.level_barrier(&[&ps.hc.up])?;
        add_bf16(ctx, pass, hyper, &ps.hc.up, hyper)?;
        pass.level_barrier(&[hyper])?;

        let Mixer::Attn(w) = &mtp.layer.mixer else {
            anyhow::bail!("draft head block is not attention")
        };
        let attn = rows
            .iter()
            .zip(&msts)
            .map(|(r, mst)| {
                AttnRow::of(&mst.layer, r.state.pos - 1, r.state.rope_delta)
            })
            .collect::<Result<Vec<_>>>()?;
        self.hc_read_batched(ctx, pass, &mtp.layer.attn_hc, hyper, s, ps)?;
        self.attn_rows(ctx, pass, w, &attn, theta, s, ps, bs)?;
        pass.level_barrier(&[&ps.branch_out])?;
        hc_inject_bf16(ctx, pass, hyper, &ps.branch_out, &ps.hc.inj, h, g)?;
        pass.level_barrier(&[hyper])?;
        self.hc_read_batched(ctx, pass, &mtp.layer.mlp_hc, hyper, s, ps)?;
        prefill_moe(
            ctx,
            pass,
            &moe_dims(cfg),
            &mtp.layer.ffn,
            &PrefillMoeIo {
                x: &ps.hc.mixed,
                out: &ps.branch_out,
                stack: &ps.stack,
                mlp_gate: &ps.mlp_gate,
                mlp_up: &ps.mlp_up,
                mlp_act: &ps.mlp_act,
                dequant: &s.dequant,
            },
            &ps.moe,
        )?;
        pass.level_barrier(&[&ps.branch_out])?;
        hc_inject_bf16(ctx, pass, hyper, &ps.branch_out, &ps.hc.inj, h, g)?;
        pass.level_barrier(&[hyper])?;

        // This step's trunk hidden pairs with the session's next token.
        for (r, mst) in msts.iter().enumerate() {
            copy_words(ctx, pass, &ps.hyper.view(r * wide, &[wide])?, &mst.hidden)?;
        }
        pass.level_barrier(&[])
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/batch.rs"]
mod tests;
