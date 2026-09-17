//! OpenAI-compatible request schemas and their validation into a
//! [`Prepared`] generation.
//!
//! Images (`docs/vision-support-plan.md` item 6) arrive as `image_url`
//! content parts whose URL is a base64 data URI; they are decoded,
//! preprocessed and digested here, on the connection thread, so the engine
//! thread only ever sees pixel rows it can hand straight to the tower. The
//! chat template writes `<|vision_start|><|image_pad|><|vision_end|>` where
//! an image sits; after tokenisation the single pad is expanded to one
//! placeholder per 2 x 2 patch block, which is what the reference processor
//! does before tokenising, and the placeholder ids are counted against the
//! images so no message text can smuggle one in.

use anyhow::{Context as _, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;

use super::data_uri::parse_image_data_uri;
use super::tools::{ToolSchema, template_tool_calls};
use crate::kernels::sample::SamplingParams;
use crate::qwen4exp::ImageSpan;
use crate::qwen4exp::image::{ImageLimits, preprocess};
use crate::sha256::Sha256;
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

/// The `image_url` field of an image part: OpenAI's `{url, detail}` object
/// (`detail` is accepted and ignored: the server-side pixel cap decides the
/// resolution) or, for symmetry with `input_text`, the URL as a string.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum ImageUrl {
    Object {
        url: String,
        #[serde(default)]
        detail: Option<String>,
    },
    Plain(String),
}

impl ImageUrl {
    fn url(&self) -> &str {
        match self {
            Self::Object { url, .. } | Self::Plain(url) => url,
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct Part {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<ImageUrl>,
}

/// One piece of a message's content, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentItem {
    Text(String),
    /// The image's URL as sent (a data URI, or something to refuse).
    Image(String),
}

impl Content {
    /// The content as ordered items: text parts (`text`, `input_text`) and
    /// image parts (`image_url` with `image_url: {url}` or a string,
    /// `input_image` with `image_url` as a string).
    pub(super) fn into_items(self) -> Result<Vec<ContentItem>> {
        match self {
            Self::Text(t) => Ok(vec![ContentItem::Text(t)]),
            Self::Parts(parts) => {
                let mut out = Vec::with_capacity(parts.len());
                for part in parts {
                    match part.kind.as_str() {
                        "text" | "input_text" => {
                            out.push(ContentItem::Text(part.text.unwrap_or_default()))
                        }
                        "image_url" | "input_image" => {
                            let url = part.image_url.with_context(|| {
                                format!(
                                    "content part type {:?} has no image_url",
                                    part.kind
                                )
                            })?;
                            out.push(ContentItem::Image(url.url().to_owned()));
                        }
                        other => bail!(
                            "content part type {other:?} is not supported (text, input_text, \
                             image_url and input_image)"
                        ),
                    }
                }
                Ok(out)
            }
        }
    }
}

/// The items as the chat template takes them: one flat string when there is
/// no image (byte-identical to the text-only rendering), otherwise the list
/// of `{type: "text", text}` and `{type: "image"}` objects in order, from
/// which the template writes the vision markers where the image sits.
pub(super) fn template_content(items: Vec<ContentItem>) -> Value {
    if items.iter().all(|i| matches!(i, ContentItem::Text(_))) {
        let mut out = String::new();
        for item in items {
            if let ContentItem::Text(t) = item {
                out.push_str(&t);
            }
        }
        return Value::String(out);
    }
    Value::Array(
        items
            .into_iter()
            .map(|item| match item {
                ContentItem::Text(text) => {
                    serde_json::json!({"type": "text", "text": text})
                }
                ContentItem::Image(_) => serde_json::json!({"type": "image"}),
            })
            .collect(),
    )
}

/// What the server accepts as images: the preprocessing limits, how many a
/// request may carry, and whether the tower is there to run them at all
/// (`Err` carries the reason a request with an image is refused).
#[derive(Debug, Clone)]
pub struct ImagePolicy {
    pub limits: ImageLimits,
    pub max_images: usize,
    pub available: Result<(), String>,
}

impl ImagePolicy {
    /// No images at all, with `why` as the refusal.
    pub fn unavailable(why: &str) -> Self {
        Self {
            limits: ImageLimits::default(),
            max_images: 0,
            available: Err(why.to_owned()),
        }
    }
}

/// One image of a request after preprocessing: the tower's input rows, the
/// placeholder span in the expanded prompt and the digest the caches
/// identify it by.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    /// `[grid_h * grid_w, PATCH_DIM]` f32 rows (see
    /// [`crate::qwen4exp::image::preprocess`]).
    pub pixels: Vec<f32>,
    pub span: ImageSpan,
    /// SHA-256 over the f32 pixel rows followed by the grid: two files that
    /// preprocess to the same rows are the same image to the model.
    pub digest: [u8; 32],
}

/// Digests the preprocessed rows and the grid ([`PreparedImage::digest`]).
pub fn image_digest(pixels: &[f32], grid_h: usize, grid_w: usize) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytemuck::cast_slice(pixels));
    h.update(&(grid_h as u64).to_le_bytes());
    h.update(&(grid_w as u64).to_le_bytes());
    h.finish()
}

/// The vocabulary ids the chat template writes for vision content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaceholderIds {
    pub image_pad: u32,
    pub vision_start: u32,
    pub vision_end: u32,
    /// `<|video_pad|>`, which no request may carry (video is not supported).
    pub video_pad: Option<u32>,
}

impl PlaceholderIds {
    /// The ids from the tokenizer, `None` when the vocabulary has no image
    /// placeholder (a text-only checkpoint).
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Option<Self> {
        Some(Self {
            image_pad: tokenizer.token_id("<|image_pad|>")?,
            vision_start: tokenizer.token_id("<|vision_start|>")?,
            vision_end: tokenizer.token_id("<|vision_end|>")?,
            video_pad: tokenizer.token_id("<|video_pad|>"),
        })
    }
}

/// Checks that the tokenised prompt carries exactly the vision markers the
/// template wrote for `images` images (one `<|vision_start|>`, one
/// `<|image_pad|>` and one `<|vision_end|>` each, and no `<|video_pad|>`),
/// so a placeholder typed into message text is refused rather than
/// tokenised into a fake image span (or, for text, fed to the model as one).
pub fn check_placeholders(
    prompt: &[u32],
    ids: &PlaceholderIds,
    images: usize,
) -> Result<()> {
    let count = |id: u32| prompt.iter().filter(|&&t| t == id).count();
    let n = images;
    let (pads, starts, ends) =
        (count(ids.image_pad), count(ids.vision_start), count(ids.vision_end));
    ensure!(
        pads == n && starts == n && ends == n,
        "the prompt carries {pads} <|image_pad|>, {starts} <|vision_start|> and {ends} <|vision_end|> \
         tokens for {n} images: these placeholder tokens are reserved for image content and \
         cannot appear in message text"
    );
    if let Some(video) = ids.video_pad {
        ensure!(
            count(video) == 0,
            "the prompt carries a <|video_pad|> token: video input is not supported and the \
             placeholder cannot appear in message text"
        );
    }
    Ok(())
}

/// After [`check_placeholders`] for `grids.len()` images: expands each single
/// `<|image_pad|>` to `grid_h * grid_w / 4` copies for its image in order and
/// returns the expanded prompt with the spans.
pub fn expand_image_pads(
    prompt: &[u32],
    ids: &PlaceholderIds,
    grids: &[(usize, usize)],
) -> Result<(Vec<u32>, Vec<ImageSpan>)> {
    check_placeholders(prompt, ids, grids.len())?;
    let n = grids.len();
    let extra: usize = grids.iter().map(|&(h, w)| h * w / 4).sum();
    let mut out = Vec::with_capacity(prompt.len() + extra);
    let mut spans = Vec::with_capacity(n);
    let mut next = grids.iter();
    for &token in prompt {
        if token == ids.image_pad {
            let &(grid_h, grid_w) = next.next().expect("one grid per counted pad");
            ensure!(
                grid_h >= 2 && grid_w >= 2 && grid_h % 2 == 0 && grid_w % 2 == 0,
                "image grid ({grid_h}, {grid_w}) is not made of 2 x 2 merge blocks"
            );
            let len = grid_h * grid_w / 4;
            spans.push(ImageSpan { start: out.len(), len, grid_h, grid_w });
            out.extend(std::iter::repeat_n(token, len));
        } else {
            out.push(token);
        }
    }
    Ok((out, spans))
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
    /// The `max_tokens` the request asked for when it had to be clamped to
    /// the room the prompt leaves in the context (for the log).
    pub clamped_from: Option<usize>,
    /// The request's images in prompt order, preprocessed; their spans are
    /// runs of `<|image_pad|>` in `prompt`. Empty for text.
    pub images: Vec<PreparedImage>,
}

/// The completion budget a request gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    pub max_tokens: usize,
    /// Set when the request asked for more than the prompt leaves room for.
    pub clamped_from: Option<usize>,
}

fn resolve_sampling(
    fields: &SamplingFields,
    defaults: &SamplingParams,
) -> Result<SamplingParams> {
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
        frequency_penalty: fields
            .frequency_penalty
            .unwrap_or(defaults.frequency_penalty),
        repetition_penalty: fields
            .repetition_penalty
            .unwrap_or(defaults.repetition_penalty),
        seed,
    };
    params.validate()?;
    Ok(params)
}

/// How many completion tokens a request may generate. A prompt that fills
/// the context is refused (nothing could be generated); a `max_tokens` the
/// prompt leaves no room for is clamped to that room, as OpenAI-compatible
/// servers do, and the response then ends with `finish_reason: "length"`.
fn resolve_budget(
    prompt_tokens: usize,
    requested: Option<usize>,
    max_seq: usize,
) -> Result<Budget> {
    ensure!(prompt_tokens > 0, "the prompt is empty");
    ensure!(
        prompt_tokens < max_seq,
        "prompt exceeds the server context: {prompt_tokens} prompt tokens, {max_seq} tokens of context \
         (at least one completion token must fit)"
    );
    let room = max_seq - prompt_tokens;
    match requested {
        Some(0) => bail!("max_tokens must be greater than zero"),
        Some(n) if n > room => Ok(Budget { max_tokens: room, clamped_from: Some(n) }),
        Some(n) => Ok(Budget { max_tokens: n, clamped_from: None }),
        None => Ok(Budget { max_tokens: room, clamped_from: None }),
    }
}

fn map_reasoning_effort(value: &str) -> Result<Option<Option<String>>> {
    // Outer None: no opinion. Some(None): thinking off. Some(Some(e)): effort.
    Ok(match value {
        "none" | "minimal" | "off" => Some(None),
        "low" => Some(Some("low".into())),
        "medium" => Some(Some("medium".into())),
        "high" | "xhigh" | "max" => Some(Some("xhigh".into())),
        other => {
            bail!("unknown reasoning_effort {other:?}; use none, low, medium or high")
        }
    })
}

/// Validates a chat request and renders its prompt; `images` says what the
/// server takes as image content.
pub fn prepare_chat(
    request: ChatRequest,
    tokenizer: &Tokenizer,
    defaults: &Defaults,
    max_seq: usize,
    images: &ImagePolicy,
) -> Result<Prepared> {
    ensure!(request.n.is_none_or(|n| n == 1), "n must be 1");
    ensure!(!request.logprobs.unwrap_or(false), "logprobs are not supported");
    if let Some(format) = &request.response_format {
        let kind = format.get("type").and_then(Value::as_str).unwrap_or("");
        ensure!(
            kind == "text",
            "response_format {kind:?} is not supported (only text)"
        );
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
    let mut prepared_images: Vec<PreparedImage> = Vec::new();
    let mut grids: Vec<(usize, usize)> = Vec::new();
    for (index, message) in request.messages.into_iter().enumerate() {
        let items = message
            .content
            .map(Content::into_items)
            .transpose()
            .with_context(|| format!("message {index}"))?
            .unwrap_or_default();
        let urls: Vec<&str> = items
            .iter()
            .filter_map(|i| match i {
                ContentItem::Image(url) => Some(url.as_str()),
                ContentItem::Text(_) => None,
            })
            .collect();
        if !urls.is_empty() {
            if let Err(why) = &images.available {
                bail!("message {index} carries an image: {why}");
            }
            ensure!(
                message.role == "user",
                "message {index}: images are accepted in user messages only (got role {:?})",
                message.role
            );
            let total = prepared_images.len() + urls.len();
            ensure!(
                total <= images.max_images,
                "the request carries {total} images; the server accepts at most {} per request \
                 (--max-images)",
                images.max_images
            );
            for url in urls {
                let k = prepared_images.len() + 1;
                let data =
                    parse_image_data_uri(url).with_context(|| format!("image {k}"))?;
                let pv = preprocess(&data.bytes, &images.limits)
                    .with_context(|| format!("image {k} ({})", data.media_type))?;
                let digest = image_digest(&pv.data, pv.grid_h, pv.grid_w);
                grids.push((pv.grid_h, pv.grid_w));
                // The span is filled in after tokenisation.
                prepared_images.push(PreparedImage {
                    pixels: pv.data,
                    span: ImageSpan {
                        start: 0,
                        len: 0,
                        grid_h: pv.grid_h,
                        grid_w: pv.grid_w,
                    },
                    digest,
                });
            }
        }
        let text = template_content(items);
        let mut value = match message.role.as_str() {
            "system" | "developer" => {
                serde_json::json!({"role": "system", "content": text})
            }
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
    let mut prompt = tokenizer.encode(&rendered)?;
    match PlaceholderIds::from_tokenizer(tokenizer) {
        Some(ids) if grids.is_empty() => check_placeholders(&prompt, &ids, 0)?,
        Some(ids) => {
            let (expanded, spans) = expand_image_pads(&prompt, &ids, &grids)?;
            prompt = expanded;
            for (image, span) in prepared_images.iter_mut().zip(spans) {
                image.span = span;
            }
        }
        None => ensure!(
            prepared_images.is_empty(),
            "this checkpoint's tokenizer has no image placeholder token"
        ),
    }
    let budget = resolve_budget(
        prompt.len(),
        request.max_completion_tokens.or(request.max_tokens),
        max_seq,
    )?;
    Ok(Prepared {
        kind: Kind::Chat,
        prompt,
        max_tokens: budget.max_tokens,
        sampling: resolve_sampling(&request.sampling, &defaults.sampling)?,
        stop_strings: request.stop.map(StringOrVec::into_vec).unwrap_or_default(),
        stream: request.stream,
        include_usage: request
            .stream_options
            .and_then(|o| o.include_usage)
            .unwrap_or(false),
        thinking_open: enable_thinking,
        tools,
        cache_key: request.prompt_cache_key,
        clamped_from: budget.clamped_from,
        images: prepared_images,
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
    ensure!(
        request.logprobs.as_ref().is_none_or(Value::is_null),
        "logprobs are not supported"
    );
    ensure!(!request.echo.unwrap_or(false), "echo is not supported");
    ensure!(
        request.stream || request.stream_options.is_none(),
        "stream_options requires stream: true"
    );
    let prompts = request.prompt.into_vec();
    ensure!(prompts.len() == 1, "exactly one prompt string is supported");
    let prompt = tokenizer.encode(&prompts[0])?;
    // OpenAI's completions default is 16 tokens.
    let budget =
        resolve_budget(prompt.len(), Some(request.max_tokens.unwrap_or(16)), max_seq)?;
    Ok(Prepared {
        kind: Kind::Completion,
        prompt,
        max_tokens: budget.max_tokens,
        sampling: resolve_sampling(&request.sampling, &defaults.sampling)?,
        stop_strings: request.stop.map(StringOrVec::into_vec).unwrap_or_default(),
        stream: request.stream,
        include_usage: request
            .stream_options
            .and_then(|o| o.include_usage)
            .unwrap_or(false),
        thinking_open: false,
        tools: None,
        cache_key: request.prompt_cache_key,
        clamped_from: budget.clamped_from,
        images: Vec::new(),
    })
}

#[cfg(test)]
pub(super) fn resolve_budget_for_test(
    prompt: usize,
    requested: Option<usize>,
    max_seq: usize,
) -> Result<Budget> {
    resolve_budget(prompt, requested, max_seq)
}
