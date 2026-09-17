//! Chat-template checks against the checkpoint's own tokenizer.
//!
//! The golden is the reference rendering, recorded from the checkpoint with
//! Hugging Face `transformers`; `tests/goldens/golden_flash_chat.json` records
//! the exact call. Only the tokenizer and the template are read, so these
//! tests need no GPU and no model weights.

use anyhow::{Context, Result, ensure};
use lily::chat::{Conversation, Message};
use lily::generate::{Generator, Thinking};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    prompt: String,
    rendered_prompt: String,
    prompt_token_ids: Vec<u32>,
}

fn generator() -> Result<Generator> {
    let dir = std::env::var("LILY_MODEL_DIR_FLASH")
        .context("set LILY_MODEL_DIR_FLASH to the Qwen3.8-Flash-Next checkpoint")?;
    Generator::from_model_dir(dir.as_ref())
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn chat_prompt_matches_the_reference_rendering() -> Result<()> {
    let generator = generator()?;
    let golden: Golden =
        serde_json::from_slice(include_bytes!("goldens/golden_flash_chat.json"))?;
    let conversation: Conversation = vec![Message::new_user(&golden.prompt)];
    let rendered =
        generator.tokenizer().render_chat(&conversation, Thinking::Enabled)?;
    ensure!(
        rendered == golden.rendered_prompt,
        "chat prompt rendering drifted from the reference:\n  got {rendered:?}\n  want {:?}",
        golden.rendered_prompt
    );
    let actual = generator.encode_chat(&conversation, Thinking::Enabled)?;
    ensure!(actual == golden.prompt_token_ids, "chat prompt tokens changed");
    Ok(())
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn chat_history_is_deterministic_and_cacheable() -> Result<()> {
    let generator = generator()?;
    let first: Conversation = vec![Message::new_user("Name a prime number.")];
    let followed: Conversation = vec![
        Message::new_user("Name a prime number."),
        Message::new_assistant("2"),
        Message::new_user("And the next one?"),
    ];
    let turn1 = generator.encode_chat(&first, Thinking::Disabled)?;
    let turn2 = generator.encode_chat(&followed, Thinking::Disabled)?;
    ensure!(
        turn2.starts_with(&turn1),
        "the follow-up prompt must extend the cached first-turn prompt"
    );
    ensure!(
        generator.encode_chat(&followed, Thinking::Disabled)? == turn2,
        "chat encoding must be deterministic"
    );
    Ok(())
}
