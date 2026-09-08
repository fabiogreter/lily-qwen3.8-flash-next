//! Optional Qwen3.8-Flash-Next checks that need a converted checkpoint with
//! the draft head (`LILY_MODEL_DIR_FLASH`; the 4-layer `-l4` conversion is
//! enough and fast): speculative decoding is invariant to the draft count,
//! and a session survives the disk-tier round trip.

use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result};
use lily::engine::{DecodeStateApi, LanguageModel, LoadOptions, SnapshotApi};
use lily::generate::{GenerateOptions, Generator};
use lily::kernels::sample::SamplingParams;
use lily::metal::MetalContext;
use lily::qwen4exp::Qwen4ExpModel;

fn model_dir() -> Result<String> {
    std::env::var("LILY_MODEL_DIR_FLASH").context("set LILY_MODEL_DIR_FLASH to a Qwen3.8-Flash-Next conversion with the MTP head")
}

fn prompt(generator: &Generator) -> Result<Vec<u32>> {
    generator.tokenizer().encode("<|im_start|>user\nName three colours and explain each in one sentence.<|im_end|>\n<|im_start|>assistant\n")
}

fn run(ctx: &MetalContext, model: &Qwen4ExpModel, generator: &Generator, prompt: &[u32], params: &SamplingParams, drafts: usize, max_tokens: usize) -> Result<(Vec<u32>, usize, usize)> {
    let capacity = prompt.len() + max_tokens + 8;
    let mut state = model.new_state(ctx, capacity)?;
    let mut scratch = model.new_scratch_with_capacity(ctx, capacity)?;
    let options = GenerateOptions { max_tokens, sampling: params, stop_tokens: &[], drafts };
    let g = generator.generate(ctx, model, &mut state, &mut scratch, prompt, &options, &mut |_| Ok(true))?;
    Ok((g.tokens, g.drafted, g.accepted))
}

/// Every draft count yields the same tokens (each emitted token is the
/// trunk's own draw for its prefix; drafts only decide how many rows a pass
/// confirms), under greedy and under seeded sampling.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn speculative_output_is_invariant_to_the_draft_count() -> Result<()> {
    let dir = model_dir()?;
    let ctx = MetalContext::new()?;
    let model = <Qwen4ExpModel as LanguageModel>::load(&ctx, Path::new(&dir), &LoadOptions { mtp_drafts: 3, ..LoadOptions::default() })?;
    assert!(model.max_drafts() > 0, "checkpoint has no draft head");
    let mut generator = Generator::from_model_dir(Path::new(&dir))?;
    generator.add_stop_tokens(&model.eos_token_ids());
    let prompt = prompt(&generator)?;
    let sampled = SamplingParams { temperature: 0.8, top_k: 40, top_p: 0.95, seed: 11, ..SamplingParams::greedy() };
    for params in [SamplingParams::greedy(), sampled] {
        let (one, drafted, _) = run(&ctx, &model, &generator, &prompt, &params, 1, 48)?;
        assert!(drafted > 0, "no drafts were proposed");
        for k in [2usize, 3] {
            let (many, _, _) = run(&ctx, &model, &generator, &prompt, &params, k, 48)?;
            assert_eq!(many, one, "drafts={k} vs drafts=1 (temperature {})", params.temperature);
        }
    }
    Ok(())
}

/// A session written to and read from the disk-tier layout continues exactly
/// like the original: prefix caches, recurrent snapshot and draft-head state.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn persisted_session_continues_like_the_original() -> Result<()> {
    let dir = model_dir()?;
    let ctx = MetalContext::new()?;
    let model = <Qwen4ExpModel as LanguageModel>::load(&ctx, Path::new(&dir), &LoadOptions { mtp_drafts: 2, ..LoadOptions::default() })?;
    let mut generator = Generator::from_model_dir(Path::new(&dir))?;
    generator.add_stop_tokens(&model.eos_token_ids());
    let prompt = prompt(&generator)?;
    let greedy = SamplingParams::greedy();
    let capacity = prompt.len() + 64;
    let mut scratch = model.new_scratch_with_capacity(&ctx, capacity)?;

    // Original: prefix the prompt (but its last token), snapshot, generate.
    let mut original = model.new_state(&ctx, capacity)?;
    let n = prompt.len();
    model.prefill(&ctx, &mut original, &mut scratch, &prompt[..n - 1], None)?;
    let snapshot = original.snapshot(&ctx)?;
    let mut prefix_bytes = Vec::new();
    original.write_prefix(n - 1, &mut prefix_bytes)?;
    let mut snapshot_bytes = Vec::new();
    snapshot.write_to(&mut snapshot_bytes)?;
    assert!(!prefix_bytes.is_empty() && !snapshot_bytes.is_empty());
    let options = GenerateOptions { max_tokens: 24, sampling: &greedy, stop_tokens: &[], drafts: 2 };
    let expected = generator.generate(&ctx, &model, &mut original, &mut scratch, &prompt[n - 1..], &options, &mut |_| Ok(true))?;

    // Restored from the bytes into a fresh state.
    let mut restored = model.new_state(&ctx, 8)?;
    restored.read_prefix(&ctx, n - 1, &mut Cursor::new(&prefix_bytes))?;
    let read_back = model.read_snapshot(&ctx, &mut Cursor::new(&snapshot_bytes))?;
    assert_eq!(read_back.pos(), n - 1);
    restored.restore(&ctx, &read_back)?;
    assert_eq!(restored.pos(), n - 1);
    let actual = generator.generate(&ctx, &model, &mut restored, &mut scratch, &prompt[n - 1..], &options, &mut |_| Ok(true))?;
    assert_eq!(actual.tokens, expected.tokens);
    assert!(model.persistence_format().is_some());
    Ok(())
}
