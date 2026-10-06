//! Optional Qwen3.8-Flash-Next checks that need a converted checkpoint with
//! the draft head (`LILY_MODEL_DIR_FLASH`; the 4-layer `-l4` conversion is
//! enough and fast): speculative decoding is invariant to the draft count,
//! a session survives the disk-tier round trip, and plain decoding grows the
//! caches across a capacity step while a step is parked.

use std::cell::RefCell;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use lily::engine::{DecodeStateApi, LanguageModel, LoadOptions, SnapshotApi};
use lily::generate::{FinishReason, GenerateOptions, Generator};
use lily::kernels::sample::SamplingParams;
use lily::metal::MetalContext;
use lily::qwen4exp::Qwen4ExpModel;
use lily::thinking::{
    ThinkingControl, ThinkingSettings, ThinkingTexts, ThinkingTokens,
};

fn model_dir() -> Result<String> {
    std::env::var("LILY_MODEL_DIR_FLASH").context(
        "set LILY_MODEL_DIR_FLASH to a Qwen3.8-Flash-Next conversion with the MTP head",
    )
}

fn prompt(generator: &Generator) -> Result<Vec<u32>> {
    generator.tokenizer().encode("<|im_start|>user\nName three colours and explain each in one sentence.<|im_end|>\n<|im_start|>assistant\n")
}

fn run(
    ctx: &MetalContext,
    model: &Qwen4ExpModel,
    generator: &Generator,
    prompt: &[u32],
    params: &SamplingParams,
    drafts: usize,
    max_tokens: usize,
) -> Result<(Vec<u32>, usize, usize)> {
    let capacity = prompt.len() + max_tokens + 8;
    let mut state = model.new_state(ctx, capacity)?;
    let mut scratch = model.new_scratch_with_capacity(ctx, capacity)?;
    let options = GenerateOptions {
        max_tokens,
        sampling: params,
        stop_tokens: &[],
        drafts,
        thinking: None,
    };
    let g = generator.generate(
        ctx,
        model,
        &mut state,
        &mut scratch,
        prompt,
        &options,
        &mut |_| Ok(true),
    )?;
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
    let model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(&dir),
        &LoadOptions { mtp_drafts: 3, ..LoadOptions::default() },
    )?;
    assert!(model.max_drafts() > 0, "checkpoint has no draft head");
    let mut generator = Generator::from_model_dir(Path::new(&dir))?;
    generator.add_stop_tokens(&model.eos_token_ids());
    let prompt = prompt(&generator)?;
    let sampled = SamplingParams {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.95,
        seed: 11,
        ..SamplingParams::greedy()
    };
    for params in [SamplingParams::greedy(), sampled] {
        let (one, drafted, _) = run(&ctx, &model, &generator, &prompt, &params, 1, 48)?;
        assert!(drafted > 0, "no drafts were proposed");
        for k in [2usize, 3] {
            let (many, _, _) = run(&ctx, &model, &generator, &prompt, &params, k, 48)?;
            assert_eq!(
                many, one,
                "drafts={k} vs drafts=1 (temperature {})",
                params.temperature
            );
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
    let model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(&dir),
        &LoadOptions { mtp_drafts: 2, ..LoadOptions::default() },
    )?;
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
    let options = GenerateOptions {
        max_tokens: 24,
        sampling: &greedy,
        stop_tokens: &[],
        drafts: 2,
        thinking: None,
    };
    let expected = generator.generate(
        &ctx,
        &model,
        &mut original,
        &mut scratch,
        &prompt[n - 1..],
        &options,
        &mut |_| Ok(true),
    )?;

    // Restored from the bytes into a fresh state.
    let mut restored = model.new_state(&ctx, 8)?;
    restored.read_prefix(&ctx, n - 1, n - 1, &mut Cursor::new(&prefix_bytes))?;
    let read_back = model.read_snapshot(&ctx, &mut Cursor::new(&snapshot_bytes))?;
    assert_eq!(read_back.pos(), n - 1);
    restored.restore(&ctx, &read_back)?;
    assert_eq!(restored.pos(), n - 1);
    let actual = generator.generate(
        &ctx,
        &model,
        &mut restored,
        &mut scratch,
        &prompt[n - 1..],
        &options,
        &mut |_| Ok(true),
    )?;
    assert_eq!(actual.tokens, expected.tokens);
    assert!(model.persistence_format().is_some());

    // A checkpoint hit inside a longer entry: the disk tier writes an evicted
    // session at its live end (the generated tokens included) and a later
    // prompt resumes at an earlier checkpoint. The regions of the longer
    // layout must be skipped, not read back to back.
    let live_end = original.pos();
    assert!(live_end > n - 1, "generation advanced the state");
    let mut longer_bytes = Vec::new();
    original.write_prefix(live_end, &mut longer_bytes)?;
    assert!(longer_bytes.len() > prefix_bytes.len());
    let mut from_longer = model.new_state(&ctx, 8)?;
    from_longer.read_prefix(&ctx, live_end, n - 1, &mut Cursor::new(&longer_bytes))?;
    from_longer.restore(&ctx, &read_back)?;
    let from_checkpoint = generator.generate(
        &ctx,
        &model,
        &mut from_longer,
        &mut scratch,
        &prompt[n - 1..],
        &options,
        &mut |_| Ok(true),
    )?;
    assert_eq!(
        from_checkpoint.tokens, expected.tokens,
        "a prefix read out of a longer layout must match"
    );
    Ok(())
}

/// A checkpoint hit inside a disk entry that spans several capacity steps,
/// restored into a state built for the prompt that hit it (as
/// `acquire_from_disk` builds it, smaller than the entry's): the caches read
/// back are the entry's own first tokens, byte for byte, every attention
/// layer and the draft head's, and the generation from there is the one
/// from a state forked in memory at the same checkpoint. The block-key skip
/// was once sized by the restoring state's capacity, which read every region
/// after the first layer's block keys from the wrong offset.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn a_checkpoint_inside_an_entry_longer_than_the_restoring_state_restores_exactly()
-> Result<()> {
    let dir = model_dir()?;
    let ctx = MetalContext::new()?;
    let model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(&dir),
        &LoadOptions { mtp_drafts: 2, ..LoadOptions::default() },
    )?;
    let mut generator = Generator::from_model_dir(Path::new(&dir))?;
    generator.add_stop_tokens(&model.eos_token_ids());
    let step = model.new_state(&ctx, 8)?.capacity();
    // Three steps and a bit of varied text, then the question.
    let filler = generator.tokenizer().encode(
        "The river bends twice before the mill, and the miller counts the sacks \
         by lamplight while the wheel turns slowly in the dark water. ",
    )?;
    let total = 3 * step + 1234;
    let entry: Vec<u32> = filler.iter().copied().cycle().take(total).collect();
    let at = 5000;
    let question = prompt(&generator)?;

    // The entry: fed in two parts with the checkpoint between them, written
    // at its live end as the disk tier writes an evicted session.
    let mut scratch = model.new_scratch_with_capacity(&ctx, total + 64)?;
    let mut original = model.new_state(&ctx, total)?;
    model.prefill(&ctx, &mut original, &mut scratch, &entry[..at], None)?;
    let checkpoint = original.snapshot(&ctx)?;
    model.prefill(&ctx, &mut original, &mut scratch, &entry[at..], None)?;
    assert_eq!(original.pos(), total);
    let mut entry_bytes = Vec::new();
    original.write_prefix(total, &mut entry_bytes)?;
    let mut expected = Vec::new();
    original.write_prefix(at, &mut expected)?;

    // The restore, into a state sized for the checkpoint.
    let mut restored = model.new_state(&ctx, at)?;
    assert!(
        restored.capacity() < original.capacity(),
        "the restoring state must be smaller than the entry's"
    );
    let mut cursor = Cursor::new(&entry_bytes);
    restored.read_prefix(&ctx, total, at, &mut cursor)?;
    assert_eq!(cursor.position() as usize, entry_bytes.len(), "the whole layout read");
    restored.restore(&ctx, &checkpoint)?;
    let mut actual = Vec::new();
    restored.write_prefix(at, &mut actual)?;
    assert!(actual == expected, "the restored caches differ from the entry's");

    // The same checkpoint forked in memory decodes the same tokens.
    let mut forked = model.new_state(&ctx, at)?;
    forked.copy_prefix_from(&ctx, &original, at)?;
    forked.restore(&ctx, &checkpoint)?;
    let greedy = SamplingParams::greedy();
    let options = GenerateOptions {
        max_tokens: 16,
        sampling: &greedy,
        stop_tokens: &[],
        drafts: 2,
        thinking: None,
    };
    let mut decode = |state: &mut _| {
        generator.generate(
            &ctx,
            &model,
            state,
            &mut scratch,
            &question,
            &options,
            &mut |_| Ok(true),
        )
    };
    let from_disk = decode(&mut restored)?;
    let from_memory = decode(&mut forked)?;
    assert_eq!(from_disk.tokens, from_memory.tokens);
    Ok(())
}

/// Plain decoding (no drafts) keeps one step parked on the GPU while the
/// current one runs, and the caches must still be able to grow when the
/// prompt plus the output crosses a capacity step. A prompt ending a few
/// tokens below the step once failed here with "parked step beyond the
/// state's capacity": the parked step had been committed against the old
/// buffers at the moment the loop needed to grow them.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn plain_decoding_grows_the_caches_across_a_capacity_step() -> Result<()> {
    let dir = model_dir()?;
    let ctx = MetalContext::new()?;
    let model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(&dir),
        &LoadOptions { mtp_drafts: 0, ..LoadOptions::default() },
    )?;
    let generator = Generator::from_model_dir(Path::new(&dir))?;
    // The first capacity step is what a tiny state rounds up to.
    let step = model.new_state(&ctx, 8)?.capacity();
    let head = prompt(&generator)?;
    let pad = generator.tokenizer().encode(" and")?;
    assert_eq!(pad.len(), 1, "the padding must be a single token");
    let mut prompt = vec![pad[0]; step - 20 - head.len()];
    prompt.extend(head);
    let mut state = model.new_state(&ctx, prompt.len())?;
    assert_eq!(state.capacity(), step, "the prompt must fit the first step exactly");
    let mut scratch = model.new_scratch_with_capacity(&ctx, step + 64)?;
    let params = SamplingParams::greedy();
    let options = GenerateOptions {
        max_tokens: 64,
        sampling: &params,
        stop_tokens: &[],
        drafts: 0,
        thinking: None,
    };
    let g = generator.generate(
        &ctx,
        &model,
        &mut state,
        &mut scratch,
        &prompt,
        &options,
        &mut |_| Ok(true),
    )?;
    assert_eq!(
        g.finish,
        FinishReason::Length,
        "a stop token ended the generation before the capacity step; rerun"
    );
    assert_eq!(g.tokens.len(), 64);
    assert!(state.capacity() > step, "the caches did not grow past {step}");
    assert!(state.pos() > step, "position {} never crossed {step}", state.pos());
    Ok(())
}

/// Decode checkpoints (`Generator::generate_checkpointed`) drain the
/// pipeline for one step and, with the draft head, restart the proposals;
/// neither may change a greedy token. Every checkpoint lands where the state
/// was at rest, at least the interval after the one before it.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn decode_checkpoints_change_no_token() -> Result<()> {
    use lily::generate::DecodeCheckpointer;
    use lily::qwen4exp::DecodeState;
    use lily::serve::session::DecodeCheckpoints;

    let dir = model_dir()?;
    let ctx = MetalContext::new()?;
    let model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(&dir),
        &LoadOptions { mtp_drafts: 2, ..LoadOptions::default() },
    )?;
    let generator = Generator::from_model_dir(Path::new(&dir))?;
    let prompt = prompt(&generator)?;
    let n = prompt.len();
    let greedy = SamplingParams::greedy();
    let max_tokens = 48;
    let capacity = n + max_tokens + 8;
    let mut scratch = model.new_scratch_with_capacity(&ctx, capacity)?;
    for drafts in [0usize, 2] {
        let options = GenerateOptions {
            max_tokens,
            sampling: &greedy,
            stop_tokens: &[],
            drafts,
            thinking: None,
        };
        let mut run = |every: usize| -> Result<_> {
            let mut state = model.new_state(&ctx, capacity)?;
            model.prefill(&ctx, &mut state, &mut scratch, &prompt[..n - 1], None)?;
            let mut checkpoints =
                DecodeCheckpoints::<DecodeState>::new(every, 64, n - 1);
            let checkpointer: &mut dyn DecodeCheckpointer<DecodeState> =
                &mut checkpoints;
            let g = generator.generate_checkpointed(
                &ctx,
                &model,
                &mut state,
                &mut scratch,
                &prompt[n - 1..],
                &options,
                Some(checkpointer),
                &mut |_| Ok(true),
            )?;
            Ok((g.tokens, state.pos(), checkpoints))
        };
        let (plain, _, none) = run(0)?;
        assert_eq!(none.taken(), 0);
        let (tokens, end, checkpoints) = run(8)?;
        assert_eq!(tokens, plain, "drafts={drafts}: checkpoints changed the output");
        let positions = checkpoints.positions();
        assert!(positions.len() >= 3, "drafts={drafts}: {positions:?}");
        assert!(positions.windows(2).all(|w| w[1] >= w[0] + 8));
        assert!(positions.iter().all(|&p| p >= n - 1 + 8 && p <= end));
        assert_eq!(checkpoints.taken(), positions.len());
    }
    Ok(())
}

// --- thinking controls -------------------------------------------------------

/// A chat prompt whose generation prompt opens a reasoning block, as the
/// template writes it with thinking on.
fn thinking_prompt(generator: &Generator) -> Result<Vec<u32>> {
    generator.tokenizer().encode(
        "<|im_start|>user\nName three colours and explain each in one sentence.<|im_end|>\n\
         <|im_start|>assistant\n<think>\n",
    )
}

/// The controls' token sequences with the built-in texts, as the server
/// makes them.
fn thinking_tokens(generator: &Generator) -> Result<Arc<ThinkingTokens>> {
    let t = generator.tokenizer();
    let id =
        |s: &str| t.token_id(s).with_context(|| format!("no {s} in the vocabulary"));
    Ok(Arc::new(ThinkingTokens::new(
        id("</think>")?,
        id("<tool_call>")?,
        id("</tool_call>")?,
        &ThinkingTexts::default(),
        |s| t.encode(s),
    )?))
}

/// A greedy generation of `max_tokens` from a fresh state with `thinking`
/// (and decode checkpoints every `every` tokens, 0 for none); returns the
/// tokens, the state and the control.
#[allow(clippy::too_many_arguments)]
fn controlled_run(
    ctx: &MetalContext,
    model: &Qwen4ExpModel,
    generator: &Generator,
    prompt: &[u32],
    drafts: usize,
    max_tokens: usize,
    thinking: Option<&RefCell<ThinkingControl>>,
    every: usize,
) -> Result<(Vec<u32>, lily::qwen4exp::DecodeState, FinishReason)> {
    use lily::generate::DecodeCheckpointer;
    use lily::qwen4exp::DecodeState;
    use lily::serve::session::DecodeCheckpoints;

    let greedy = SamplingParams::greedy();
    let n = prompt.len();
    let capacity = n + max_tokens + 64;
    let mut state = model.new_state(ctx, capacity)?;
    let mut scratch = model.new_scratch_with_capacity(ctx, capacity)?;
    model.prefill(ctx, &mut state, &mut scratch, &prompt[..n - 1], None)?;
    let mut checkpoints = DecodeCheckpoints::<DecodeState>::new(every, 64, n - 1);
    let checkpointer: &mut dyn DecodeCheckpointer<DecodeState> = &mut checkpoints;
    let options = GenerateOptions {
        max_tokens,
        sampling: &greedy,
        stop_tokens: &[],
        drafts,
        thinking,
    };
    let g = generator.generate_checkpointed(
        ctx,
        model,
        &mut state,
        &mut scratch,
        &prompt[n - 1..],
        &options,
        Some(checkpointer),
        &mut |_| Ok(true),
    )?;
    // (`generate_checkpointed` checks that the state fed every token but
    // the last, or all of them.)
    Ok((g.tokens, state, g.finish))
}

fn load_with_head(ctx: &MetalContext) -> Result<(Qwen4ExpModel, Generator)> {
    let dir = model_dir()?;
    let model = <Qwen4ExpModel as LanguageModel>::load(
        ctx,
        Path::new(&dir),
        &LoadOptions { mtp_drafts: 2, ..LoadOptions::default() },
    )?;
    assert!(model.max_drafts() > 0, "checkpoint has no draft head");
    let mut generator = Generator::from_model_dir(Path::new(&dir))?;
    generator.add_stop_tokens(&model.eos_token_ids());
    Ok((model, generator))
}

/// Controls that are present but have nothing to do (no setting on, or a
/// prompt that does not open a reasoning block) change no token, plain or
/// speculative: off means byte-identical output.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn idle_thinking_controls_change_no_token() -> Result<()> {
    let ctx = MetalContext::new()?;
    let (model, generator) = load_with_head(&ctx)?;
    let tokens = thinking_tokens(&generator)?;
    let opened = thinking_prompt(&generator)?;
    let plain = prompt(&generator)?;
    for drafts in [0usize, 2] {
        for p in [&opened, &plain] {
            let (expected, _, _) =
                controlled_run(&ctx, &model, &generator, p, drafts, 40, None, 0)?;
            let idle = RefCell::new(ThinkingControl::new(
                tokens.clone(),
                ThinkingSettings::default(),
                true,
            ));
            let (got, _, _) = controlled_run(
                &ctx,
                &model,
                &generator,
                p,
                drafts,
                40,
                Some(&idle),
                0,
            )?;
            assert_eq!(got, expected, "drafts={drafts}: settings all off");
            let unopened = RefCell::new(ThinkingControl::new(
                tokens.clone(),
                ThinkingSettings {
                    budget: Some(1),
                    nudges: true,
                    tool_call_ends_thinking: true,
                    ..ThinkingSettings::default()
                },
                false,
            ));
            let (got, _, _) = controlled_run(
                &ctx,
                &model,
                &generator,
                p,
                drafts,
                40,
                Some(&unopened),
                0,
            )?;
            assert_eq!(got, expected, "drafts={drafts}: no reasoning block opened");
        }
    }
    Ok(())
}

/// A budget closes the block: `</think>` is in the stream right after the
/// transition text, and the state the generation leaves is the state a
/// prefill of the same tokens makes: the next draw and the continuation
/// agree, draft head included.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn a_forced_close_leaves_the_state_a_prefill_of_the_same_tokens_makes() -> Result<()> {
    let ctx = MetalContext::new()?;
    let (model, generator) = load_with_head(&ctx)?;
    let tokens = thinking_tokens(&generator)?;
    let think_end = tokens.think_end;
    let prompt = thinking_prompt(&generator)?;
    let n = prompt.len();
    for drafts in [0usize, 2] {
        // Grace 0: the close comes right at the budget, wherever that is.
        let control = RefCell::new(ThinkingControl::new(
            tokens.clone(),
            ThinkingSettings {
                budget: Some(8),
                grace: 0,
                ..ThinkingSettings::default()
            },
            true,
        ));
        let (generated, mut state, finish) = controlled_run(
            &ctx,
            &model,
            &generator,
            &prompt,
            drafts,
            48,
            Some(&control),
            0,
        )?;
        let closed = control.borrow().closed().context("the budget did not close")?;
        assert_eq!(closed.thinking_tokens, 8);
        let at = generated
            .iter()
            .position(|&t| t == think_end)
            .context("no </think> in the stream")?;
        assert!(at > 8, "the transition text comes first");
        let text = generator.decode_text(&generated[..at + 1])?;
        assert!(text.contains("time to act"), "drafts={drafts}: {text:?}");
        if finish != FinishReason::Length {
            eprintln!(
                "drafts={drafts}: ended by {finish:?}; the continuation check is skipped"
            );
            continue;
        }

        // The reference: the prompt and every generated token but the last
        // fed as one prompt.
        let all: Vec<u32> =
            prompt.iter().chain(&generated[..generated.len() - 1]).copied().collect();
        let capacity = all.len() + 64;
        let mut reference = model.new_state(&ctx, capacity)?;
        let mut scratch = model.new_scratch_with_capacity(&ctx, capacity)?;
        model.prefill(&ctx, &mut reference, &mut scratch, &all, None)?;
        assert_eq!(state.pos(), reference.pos(), "drafts={drafts}: positions");
        assert_eq!(state.pos(), n + generated.len() - 1);

        // The same next token fed into both: the same logits' argmax, and
        // the same continuation.
        let greedy = SamplingParams::greedy();
        let last = *generated.last().expect("generated");
        let options = GenerateOptions {
            max_tokens: 16,
            sampling: &greedy,
            stop_tokens: &[],
            drafts,
            thinking: None,
        };
        let mut logits = Vec::new();
        let mut continuations = Vec::new();
        for s in [&mut state, &mut reference] {
            use lily::engine::ScratchApi;
            scratch.begin_request();
            model.prefill(
                &ctx,
                s,
                &mut scratch,
                &[last],
                Some(lily::engine::Draw { params: &greedy, step: 0 }),
            )?;
            logits.push(scratch.logits().to_f32()?);
            let first = scratch.next_token().view(0, &[1])?.to_u32()?[0];
            // On from that draw, both states the same way (the draft head
            // proposes from what the prefill left it).
            let mut tokens = vec![first];
            generator.resume(
                &ctx,
                &model,
                s,
                &mut scratch,
                &mut tokens,
                &options,
                None,
                &mut |_| Ok(true),
            )?;
            continuations.push(tokens);
        }
        let worst = logits[0]
            .iter()
            .zip(&logits[1])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let argmax = |l: &[f32]| {
            l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i)
        };
        eprintln!("drafts={drafts}: worst logit gap {worst}");
        assert_eq!(
            argmax(&logits[0]),
            argmax(&logits[1]),
            "drafts={drafts}: next draw"
        );
        assert_eq!(continuations[0], continuations[1], "drafts={drafts}: continuation");
    }
    Ok(())
}

/// The speculative loop (resting, feeding the inserted tokens, proposing
/// afresh) and the plain loop (unpipelined around an insertion) emit the
/// same tokens under a budget with nudges, with and without decode
/// checkpoints.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn speculative_and_plain_decoding_agree_under_a_budget() -> Result<()> {
    let ctx = MetalContext::new()?;
    let (model, generator) = load_with_head(&ctx)?;
    let tokens = thinking_tokens(&generator)?;
    let prompt = thinking_prompt(&generator)?;
    let settings = ThinkingSettings {
        budget: Some(24),
        grace: 2,
        nudges: true,
        tool_call_ends_thinking: true,
        seed: 3,
    };
    let run = |drafts: usize, every: usize| -> Result<(Vec<u32>, usize)> {
        let control =
            RefCell::new(ThinkingControl::new(tokens.clone(), settings, true));
        let (generated, _, _) = controlled_run(
            &ctx,
            &model,
            &generator,
            &prompt,
            drafts,
            72,
            Some(&control),
            every,
        )?;
        let nudged = control.borrow().nudged();
        Ok((generated, nudged))
    };
    let (plain, nudged) = run(0, 0)?;
    assert!(
        plain.contains(&tokens.think_end),
        "the budget closed the block: {:?}",
        generator.decode_text(&plain)?
    );
    eprintln!("nudges: {nudged}; {:?}", generator.decode_text(&plain)?);
    for (drafts, every) in [(2usize, 0usize), (1, 0), (2, 8), (0, 8)] {
        let (got, got_nudged) = run(drafts, every)?;
        assert_eq!(got, plain, "drafts={drafts} checkpoints every {every}");
        assert_eq!(got_nudged, nudged);
    }
    Ok(())
}
