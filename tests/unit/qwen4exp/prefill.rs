//! The chunked prefill's n-gram staging pipeline against the 4-layer
//! checkpoint (`LILY_MODEL_DIR_FLASH`).

use std::path::Path;

use crate::engine::{
    DecodeStateApi, LanguageModel, LoadOptions, ScratchApi, SnapshotApi,
};
use crate::metal::MetalContext;

use super::*;

/// Everything a prefill leaves behind that a later step reads: the last
/// token's logits, every per-token cache and the recurrent state (the PLE
/// hash history and conv window included).
fn outcome(
    ctx: &MetalContext,
    model: &Qwen4ExpModel,
    prompt: &[u32],
) -> (Vec<u8>, Vec<u8>, Vec<u8>, crate::stats::Counters) {
    let capacity = prompt.len() + 64;
    let mut s = model.new_scratch_with_capacity(ctx, capacity).expect("scratch");
    let mut state = model.new_state(ctx, capacity).expect("state");
    let before = crate::stats::counters();
    // Two separate calls: the second continues the first's history and
    // position, as a request resuming a cached session does.
    let split = prompt.len() / 3;
    model.prefill(ctx, &mut state, &mut s, &prompt[..split], None).expect("prefill");
    model.prefill(ctx, &mut state, &mut s, &prompt[split..], None).expect("prefill");
    let counters = crate::stats::counters().since(before);
    let logits = s.logits().raw_bytes().to_vec();
    let mut caches = Vec::new();
    state.write_prefix(prompt.len(), &mut caches).expect("prefix");
    let mut recurrent = Vec::new();
    state.snapshot(ctx).expect("snapshot").write_to(&mut recurrent).expect("write");
    (logits, caches, recurrent, counters)
}

/// Staging chunk k+1's rows while the GPU runs chunk k (the default) leaves
/// exactly the state and logits of staging every chunk right before its own
/// commit: the rows are the same, the buffers alternate so the GPU never
/// reads rows being overwritten, and the hash history carries across the
/// chunks and across two prefill calls. Prompts with a partial last chunk,
/// with whole chunks only, and eos tokens on the chunk boundaries.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn staging_ahead_is_bit_identical_to_staging_per_chunk() {
    let Some(dir) = std::env::var("LILY_MODEL_DIR_FLASH").ok() else { return };
    let ctx = MetalContext::new().expect("metal context");
    let mut model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(&dir),
        &LoadOptions::default(),
    )
    .expect("load");
    assert!(
        model.ple_table().is_some_and(NgramTable::is_paged),
        "the test needs the paged n-gram table"
    );
    let chunk = 256;
    model.set_prefill_chunk(chunk).expect("chunk");
    let eos = model.hasher.as_ref().expect("hasher").eos();
    for len in [3 * chunk + 37, 2 * chunk + 1, 4 * chunk] {
        let mut prompt: Vec<u32> =
            (0..len).map(|i| 1000 + (i * 37 % 5000) as u32).collect();
        // eos as the last token of a chunk, the first of the next and next
        // to the boundary of the second prefill call.
        for i in [chunk - 1, chunk, 2 * chunk - 2, len / 3, len / 3 + 1] {
            prompt[i] = eos;
        }
        model.set_ngram_ahead(false);
        let (logits_a, caches_a, rec_a, per_chunk) = outcome(&ctx, &model, &prompt);
        model.set_ngram_ahead(true);
        let (logits_b, caches_b, rec_b, ahead) = outcome(&ctx, &model, &prompt);
        assert_eq!(per_chunk.gather.hidden_secs, 0.0, "{len}: nothing staged ahead");
        assert!(ahead.gather.hidden_secs > 0.0, "{len}: the pipeline did not engage");
        assert_eq!(per_chunk.gather.rows, ahead.gather.rows, "{len}: rows gathered");
        assert_eq!(per_chunk.prefill.chunks, ahead.prefill.chunks);
        assert!(logits_a == logits_b, "{len}: logits differ");
        assert!(caches_a == caches_b, "{len}: per-token caches differ");
        assert!(rec_a == rec_b, "{len}: recurrent state differs");
    }
}
