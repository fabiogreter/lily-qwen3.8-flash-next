//! A recorded conversation rendered whole: one token sequence through the
//! server's own template path ([`super::api::template_messages`],
//! [`Tokenizer::encode_chat`]) with every assistant turn marked as the
//! request that produced it, the input of offline cache and routing studies
//! (`lily-render-chat`, `lily-trace-routing`, `lily-cache-replay`).
//!
//! The conversation is rendered once, with every assistant turn's reasoning
//! kept (`preserve_thinking`, the template's default), so the sequence is
//! what the server's session cache holds after the last turn. Request `k`
//! prefilled everything since turn `k - 1` ended, its own
//! `<|im_start|>assistant\n` header included, and generated the turn from
//! there through its `<|im_end|>`. That is an approximation of the live
//! requests in two places: the generation prompt's `<think>\n` was
//! prefilled by the server, and the closing `<|im_end|>` was sampled but
//! never fed (the next request prefilled it), yet both are counted as
//! generated here, about three positions per turn. The disk tier's recorded
//! sessions are split the same way, so traces from both compare directly.
//! Tools are rendered as the server renders them (none when `tool_choice`
//! is `none`); `chat_template_kwargs` are not applied.

use anyhow::{Context as _, Result, bail, ensure};

use super::api::{ChatRequest, template_messages};
use super::tools::ToolSchema;
use crate::tokenizer::{ChatRender, Tokenizer};

/// A whole conversation's ids and its assistant turns as requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    /// The conversation through its last message, without the trailing
    /// generation prompt the template appends.
    pub tokens: Vec<u32>,
    /// One `[p0, p1, p2]` per assistant turn, in order: prefilled `p0..p1`,
    /// generated `p1..p2`. Consecutive (`p0` is the previous `p2`, the first
    /// is 0), so a trailing non-assistant tail lies past the last `p2`.
    pub requests: Vec<[usize; 3]>,
}

impl Transcript {
    /// Tokens inside the generated spans.
    pub fn generated(&self) -> usize {
        self.requests.iter().map(|r| r[2] - r[1]).sum()
    }
}

/// The ids that delimit turns in a render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnMarkers {
    pub im_start: u32,
    pub im_end: u32,
    /// The ids after `<|im_start|>` that open an assistant turn
    /// (`assistant\n`).
    pub assistant: Vec<u32>,
}

impl TurnMarkers {
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Result<Self> {
        let im_start = tokenizer
            .token_id("<|im_start|>")
            .context("the tokenizer has no <|im_start|> token")?;
        let im_end = tokenizer
            .token_id("<|im_end|>")
            .context("the tokenizer has no <|im_end|> token")?;
        let assistant = tokenizer.encode("assistant\n")?;
        ensure!(!assistant.is_empty(), "`assistant\\n` encodes to nothing");
        Ok(Self { im_start, im_end, assistant })
    }
}

/// Renders a recorded chat request (all its messages, the last one usually
/// an assistant turn) as one sequence with each assistant turn a request.
/// `tools` are validated as the server would; `reasoning_effort` applies
/// only with thinking on. The request's own sampling and thinking fields
/// are ignored.
pub fn render_transcript(
    request: ChatRequest,
    tokenizer: &Tokenizer,
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
) -> Result<Transcript> {
    ensure!(!request.messages.is_empty(), "messages must not be empty");
    // As `prepare_chat`: `tool_choice: "none"` renders no tools.
    let tool_choice_none =
        request.tool_choice.as_ref().is_some_and(|c| c.as_str() == Some("none"));
    let tools = request.tools.filter(|t| !t.is_empty() && !tool_choice_none);
    if let Some(tools) = &tools {
        ToolSchema::from_request(tools)?;
    }
    let turns = request.messages.iter().filter(|m| m.role == "assistant").count();
    let messages = template_messages(request.messages)?;
    let ids = tokenizer
        .encode_chat(&ChatRender {
            messages: &messages,
            tools: tools.as_deref(),
            enable_thinking,
            reasoning_effort: if enable_thinking { reasoning_effort } else { None },
            preserve_thinking: Some(true),
        })
        .context("rendering the chat template")?;
    let transcript = split_turns(ids, &TurnMarkers::from_tokenizer(tokenizer)?)?;
    ensure!(
        transcript.requests.len() == turns,
        "the render has {} assistant turns, the conversation {turns}",
        transcript.requests.len()
    );
    Ok(transcript)
}

/// Cuts the generation prompt off a whole-conversation render (`ids`, which
/// must end in one) and marks each assistant turn before it as a request.
///
/// Only the template writes `<|im_start|>` and `<|im_end|>`
/// ([`Tokenizer::encode_chat`] encodes message text as text), so an
/// assistant turn is an `<|im_start|>` followed by `markers.assistant`, and
/// it ends at the next `<|im_end|>`.
pub fn split_turns(mut ids: Vec<u32>, markers: &TurnMarkers) -> Result<Transcript> {
    let header = markers.assistant.as_slice();
    let starts: Vec<usize> = (0..ids.len())
        .filter(|&i| ids[i] == markers.im_start && ids[i + 1..].starts_with(header))
        .collect();
    let Some((&prompt, turns)) = starts.split_last() else {
        bail!("the render has no assistant header, so no generation prompt");
    };
    ensure!(
        !ids[prompt..].contains(&markers.im_end),
        "the render does not end in a generation prompt"
    );
    let mut requests = Vec::with_capacity(turns.len());
    let mut p0 = 0;
    for &start in turns {
        let p1 = start + 1 + header.len();
        let len = ids[p1..prompt]
            .iter()
            .position(|&id| id == markers.im_end)
            .with_context(|| format!("the assistant turn at {start} never ends"))?;
        let p2 = p1 + len + 1;
        ensure!(
            !ids[p1..p2].contains(&markers.im_start),
            "the assistant turn at {start} contains another turn's start"
        );
        requests.push([p0, p1, p2]);
        p0 = p2;
    }
    ids.truncate(prompt);
    Ok(Transcript { tokens: ids, requests })
}

#[cfg(test)]
#[path = "../../tests/unit/serve/transcript.rs"]
mod tests;
