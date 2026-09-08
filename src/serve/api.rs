//! OpenAI-compatible request schemas and their validation into a
//! [`Prepared`] generation.

use anyhow::{Context as _, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;

use super::tools::{ToolSchema, template_tool_calls};
use crate::kernels::sample::SamplingParams;
use crate::tokenizer::{ChatRender, Tokenizer};

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum StringOrVec {
    One(String),
    Many(Vec<String>),
}

impl StringOrVec {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<Part>),
}

#[derive(Deserialize, Debug, Clone)]
pub struct Part {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

impl Content {
    fn into_text(self) -> Result<String> {
        match self {
            Self::Text(t) => Ok(t),
            Self::Parts(parts) => {
                let mut out = String::new();
                for part in parts {
                    match part.kind.as_str() {
                        "text" | "input_text" => out.push_str(part.text.as_deref().unwrap_or("")),
                        other => bail!("content part type {other:?} is not supported (text only)"),
                    }
                }
                Ok(out)
            }
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct InMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Content>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

/// Sampling fields shared by both endpoints. Unset fields take the server
/// defaults (the checkpoint's `generation_config.json` unless overridden).
#[derive(Deserialize, Debug, Clone, Default)]
pub struct SamplingFields {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<InMessage>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(flatten)]
    pub sampling: SamplingFields,
    #[serde(default)]
    pub stop: Option<StringOrVec>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub chat_template_kwargs: Option<Value>,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: StringOrVec,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(flatten)]
    pub sampling: SamplingFields,
    #[serde(default)]
    pub stop: Option<StringOrVec>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub logprobs: Option<Value>,
    #[serde(default)]
    pub echo: Option<bool>,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
}

/// Server-side defaults a request may override.
#[derive(Debug, Clone)]
pub struct Defaults {
    pub sampling: SamplingParams,
    /// Whether chat prompts open a reasoning block unless the request says otherwise.
    pub thinking: bool,
    /// Template `reasoning_effort` (`low`, `medium`, `xhigh`) when thinking is on.
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Chat,
    Completion,
}

/// A validated request, ready for the engine.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub kind: Kind,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub stop_strings: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
    /// The chat prompt ends inside an open `<think>` block.
    pub thinking_open: bool,
    /// Tools the model may call (`None` when none were offered or
    /// `tool_choice` is `none`).
    pub tools: Option<Vec<ToolSchema>>,
    pub cache_key: Option<String>,
}

fn resolve_sampling(fields: &SamplingFields, defaults: &SamplingParams) -> Result<SamplingParams> {
    let top_k = match fields.top_k {
        None => defaults.top_k,
        Some(k) if k < 0 => 0,
        Some(k) => k as usize,
    };
    let seed = match fields.seed {
        Some(s) => s as u64,
        None => {
            // Fresh entropy per request so repeated prompts do not replay.
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            t ^ (std::process::id() as u64).rotate_left(32)
        }
    };
    let params = SamplingParams {
        temperature: fields.temperature.unwrap_or(defaults.temperature),
        top_k,
        top_p: fields.top_p.unwrap_or(defaults.top_p),
        min_p: fields.min_p.unwrap_or(defaults.min_p),
        presence_penalty: fields.presence_penalty.unwrap_or(defaults.presence_penalty),
        frequency_penalty: fields.frequency_penalty.unwrap_or(defaults.frequency_penalty),
        repetition_penalty: fields.repetition_penalty.unwrap_or(defaults.repetition_penalty),
        seed,
    };
    params.validate()?;
    Ok(params)
}

fn resolve_budget(prompt_tokens: usize, requested: Option<usize>, max_seq: usize) -> Result<usize> {
    ensure!(prompt_tokens > 0, "the prompt is empty");
    ensure!(
        prompt_tokens < max_seq,
        "the prompt has {prompt_tokens} tokens; this server's context is {max_seq} tokens and needs room for at least one completion token"
    );
    let room = max_seq - prompt_tokens;
    match requested {
        Some(0) => bail!("max_tokens must be greater than zero"),
        Some(n) => {
            ensure!(
                n <= room,
                "prompt ({prompt_tokens}) plus max_tokens ({n}) exceeds the server context of {max_seq} tokens"
            );
            Ok(n)
        }
        None => Ok(room),
    }
}

fn map_reasoning_effort(value: &str) -> Result<Option<Option<String>>> {
    // Outer None: no opinion. Some(None): thinking off. Some(Some(e)): effort.
    Ok(match value {
        "none" | "minimal" | "off" => Some(None),
        "low" => Some(Some("low".into())),
        "medium" => Some(Some("medium".into())),
        "high" | "xhigh" | "max" => Some(Some("xhigh".into())),
        other => bail!("unknown reasoning_effort {other:?}; use none, low, medium or high"),
    })
}

/// Validates a chat request and renders its prompt.
pub fn prepare_chat(
    request: ChatRequest,
    tokenizer: &Tokenizer,
    defaults: &Defaults,
    max_seq: usize,
) -> Result<Prepared> {
    ensure!(request.n.is_none_or(|n| n == 1), "n must be 1");
    ensure!(!request.logprobs.unwrap_or(false), "logprobs are not supported");
    if let Some(format) = &request.response_format {
        let kind = format.get("type").and_then(Value::as_str).unwrap_or("");
        ensure!(kind == "text", "response_format {kind:?} is not supported (only text)");
    }
    ensure!(!request.messages.is_empty(), "messages must not be empty");
    ensure!(
        request.stream || request.stream_options.is_none(),
        "stream_options requires stream: true"
    );

    // Thinking and effort: chat_template_kwargs win, then the top-level
    // fields, then the server defaults.
    let kwargs = request.chat_template_kwargs.as_ref().and_then(Value::as_object);
    let mut enable_thinking = defaults.thinking;
    let mut effort = defaults.reasoning_effort.clone();
    if let Some(value) = &request.reasoning_effort {
        match map_reasoning_effort(value)? {
            Some(None) => enable_thinking = false,
            Some(Some(e)) => {
                enable_thinking = true;
                effort = Some(e);
            }
            None => {}
        }
    }
    if let Some(flag) = request.enable_thinking {
        enable_thinking = flag;
    }
    let mut preserve_thinking = None;
    if let Some(kwargs) = kwargs {
        if let Some(flag) = kwargs.get("enable_thinking").and_then(Value::as_bool) {
            enable_thinking = flag;
        }
        if let Some(value) = kwargs.get("reasoning_effort").and_then(Value::as_str) {
            match map_reasoning_effort(value)? {
                Some(None) => enable_thinking = false,
                Some(Some(e)) => effort = Some(e),
                None => {}
            }
        }
        if let Some(flag) = kwargs.get("preserve_thinking").and_then(Value::as_bool) {
            preserve_thinking = Some(flag);
        }
    }

    let tool_choice_none = request
        .tool_choice
        .as_ref()
        .and_then(Value::as_str)
        .is_some_and(|c| c == "none");
    let tools_json: Option<Vec<Value>> = match &request.tools {
        Some(tools) if !tools.is_empty() && !tool_choice_none => Some(tools.clone()),
        _ => None,
    };
    let tools = tools_json.as_deref().map(ToolSchema::from_request).transpose()?;

    let mut messages = Vec::with_capacity(request.messages.len());
    for (index, message) in request.messages.into_iter().enumerate() {
        let text = message.content.map(Content::into_text).transpose()?.unwrap_or_default();
        let mut value = match message.role.as_str() {
            "system" | "developer" => serde_json::json!({"role": "system", "content": text}),
            "user" => serde_json::json!({"role": "user", "content": text}),
            "assistant" => {
                let mut v = serde_json::json!({"role": "assistant", "content": text});
                if let Some(r) = message.reasoning_content.or(message.reasoning) {
                    v["reasoning_content"] = Value::String(r);
                }
                if let Some(calls) = &message.tool_calls {
                    v["tool_calls"] = Value::Array(template_tool_calls(calls)?);
                }
                v
            }
            "tool" => serde_json::json!({"role": "tool", "content": text}),
            other => bail!("message {index}: unsupported role {other:?}"),
        };
        if let Some(name) = message.name {
            value["name"] = Value::String(name);
        }
        messages.push(value);
    }
    let last_role = messages.last().and_then(|m| m["role"].as_str()).unwrap_or("");
    ensure!(
        matches!(last_role, "user" | "tool"),
        "the final message must have role user or tool (got {last_role:?})"
    );

    let rendered = tokenizer
        .render(&ChatRender {
            messages: &messages,
            tools: tools_json.as_deref(),
            enable_thinking,
            reasoning_effort: if enable_thinking { effort.as_deref() } else { None },
            preserve_thinking,
        })
        .context("rendering the chat template")?;
    let prompt = tokenizer.encode(&rendered)?;
    let max_tokens = resolve_budget(
        prompt.len(),
        request.max_completion_tokens.or(request.max_tokens),
        max_seq,
    )?;
    Ok(Prepared {
        kind: Kind::Chat,
        prompt,
        max_tokens,
        sampling: resolve_sampling(&request.sampling, &defaults.sampling)?,
        stop_strings: request.stop.map(StringOrVec::into_vec).unwrap_or_default(),
        stream: request.stream,
        include_usage: request.stream_options.and_then(|o| o.include_usage).unwrap_or(false),
        thinking_open: enable_thinking,
        tools,
        cache_key: request.prompt_cache_key,
    })
}

/// Validates a raw text completion request.
pub fn prepare_completion(
    request: CompletionRequest,
    tokenizer: &Tokenizer,
    defaults: &Defaults,
    max_seq: usize,
) -> Result<Prepared> {
    ensure!(request.n.is_none_or(|n| n == 1), "n must be 1");
    ensure!(request.logprobs.as_ref().is_none_or(Value::is_null), "logprobs are not supported");
    ensure!(!request.echo.unwrap_or(false), "echo is not supported");
    ensure!(
        request.stream || request.stream_options.is_none(),
        "stream_options requires stream: true"
    );
    let prompts = request.prompt.into_vec();
    ensure!(prompts.len() == 1, "exactly one prompt string is supported");
    let prompt = tokenizer.encode(&prompts[0])?;
    // OpenAI's completions default is 16 tokens.
    let max_tokens = resolve_budget(prompt.len(), Some(request.max_tokens.unwrap_or(16)), max_seq)?;
    Ok(Prepared {
        kind: Kind::Completion,
        prompt,
        max_tokens,
        sampling: resolve_sampling(&request.sampling, &defaults.sampling)?,
        stop_strings: request.stop.map(StringOrVec::into_vec).unwrap_or_default(),
        stream: request.stream,
        include_usage: request.stream_options.and_then(|o| o.include_usage).unwrap_or(false),
        thinking_open: false,
        tools: None,
        cache_key: request.prompt_cache_key,
    })
}

#[cfg(test)]
pub(super) fn resolve_budget_for_test(prompt: usize, requested: Option<usize>, max_seq: usize) -> Result<usize> {
    resolve_budget(prompt, requested, max_seq)
}
