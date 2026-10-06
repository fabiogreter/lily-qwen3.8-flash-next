//! OpenAI-compatible request schemas and their validation into a
//! [`Prepared`] generation.
//!
//! Images (`docs/architecture.md`, "The server") arrive as `image_url`
//! content parts whose URL is a base64 data URI; a new one is decoded,
//! preprocessed and digested here, on the connection thread, so the engine
//! thread gets pixel rows it can hand straight to the tower. One the
//! [`ImageMemo`] already knows (an agent resends its whole history every
//! turn) only has its URI hashed, and is preprocessed on the engine thread
//! in the rare case the tower has to run over it again. The
//! chat template writes `<|vision_start|><|image_pad|><|vision_end|>` where
//! an image sits; after tokenisation the single pad is expanded to one
//! placeholder per 2 x 2 patch block, which is what the reference processor
//! does before tokenising. Message text cannot produce a placeholder: the
//! prompt is encoded with [`Tokenizer::encode_chat`], which tokenizes a
//! special token's spelling in content as plain text. The placeholder ids
//! are still counted against the images, which guards the invariant itself
//! (one marker triple per image) whatever writes the prompt.

use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;

use super::data_uri::parse_image_data_uri;
use super::timings::Speculation;
use super::tools::{ToolSchema, template_tool_calls};
use crate::kernels::sample::SamplingParams;
use crate::qwen4exp::ImageSpan;
use crate::qwen4exp::image::{ImageLimits, preprocess};
use crate::sha256::Sha256;
use crate::thinking::ThinkingSettings;
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

/// What the server accepts as images: the preprocessing limits, whether the
/// tower is there to run them at all (`Err` carries the reason a request with
/// an image is refused), and the memo of images already seen. There is no
/// limit on how many images a request carries: agent clients resend the whole
/// history every turn, and the context is the only real bound.
#[derive(Debug, Clone)]
pub struct ImagePolicy {
    pub limits: ImageLimits,
    pub available: Result<(), String>,
    pub memo: Arc<ImageMemo>,
}

impl ImagePolicy {
    /// No images at all, with `why` as the refusal.
    pub fn unavailable(why: &str) -> Self {
        Self {
            limits: ImageLimits::default(),
            available: Err(why.to_owned()),
            memo: Arc::default(),
        }
    }
}

/// What preprocessing an image produced, short of the rows themselves: its
/// grid and [`image_digest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageIdentity {
    pub grid_h: usize,
    pub grid_w: usize,
    pub digest: [u8; 32],
}

/// Images seen recently, by the SHA-256 of their data URI, with the
/// identity preprocessing gave them. An agent session resends every earlier
/// image on every turn; with the memo those cost a hash of the URI instead of
/// a decode, a resize and a hash of the rows, and their rows are only made
/// again if the tower has to run over them (the session cache lost them).
/// The limits are fixed for the process, so an identity never goes stale.
/// Bounded at [`ImageMemo::CAPACITY`] entries, oldest insertion first out.
#[derive(Debug, Default)]
pub struct ImageMemo {
    inner: Mutex<MemoInner>,
}

#[derive(Debug, Default)]
struct MemoInner {
    map: HashMap<[u8; 32], ImageIdentity>,
    order: VecDeque<[u8; 32]>,
}

impl ImageMemo {
    /// About 80 bytes an entry.
    pub const CAPACITY: usize = 4096;

    /// The memo key of a data URI.
    pub fn key(url: &str) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(url.as_bytes());
        h.finish()
    }

    pub fn get(&self, key: &[u8; 32]) -> Option<ImageIdentity> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).map.get(key).copied()
    }

    pub fn insert(&self, key: [u8; 32], identity: ImageIdentity) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.map.insert(key, identity).is_none() {
            inner.order.push_back(key);
            while inner.order.len() > Self::CAPACITY {
                if let Some(old) = inner.order.pop_front() {
                    inner.map.remove(&old);
                }
            }
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).map.len()
    }
}

/// The tower's input rows of an image: made on the connection thread for an
/// image the memo did not know, or kept as the data URI for one it did, to
/// be preprocessed only if the tower has to run over it.
#[derive(Debug, Clone)]
pub enum ImagePixels {
    Ready(Vec<f32>),
    Deferred { url: String, limits: ImageLimits },
}

/// One image of a request: its rows (or how to make them), the placeholder
/// span in the expanded prompt and the digest the caches identify it by.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    /// `[grid_h * grid_w, PATCH_DIM]` f32 rows (see
    /// [`crate::qwen4exp::image::preprocess`]).
    pub pixels: ImagePixels,
    pub span: ImageSpan,
    /// SHA-256 over the f32 pixel rows followed by the grid: two files that
    /// preprocess to the same rows are the same image to the model.
    pub digest: [u8; 32],
}

impl PreparedImage {
    /// The rows, preprocessing a deferred image now. A deferred image must
    /// come out with the grid and digest the memo gave it; anything else is
    /// refused rather than encoded under the wrong cache identity.
    pub fn rows(&self) -> Result<Cow<'_, [f32]>> {
        match &self.pixels {
            ImagePixels::Ready(rows) => Ok(Cow::Borrowed(rows)),
            ImagePixels::Deferred { url, limits } => {
                let data = parse_image_data_uri(url)?;
                let pv = preprocess(&data.bytes, limits)
                    .with_context(|| format!("preprocessing a {}", data.media_type))?;
                let identity = ImageIdentity {
                    grid_h: pv.grid_h,
                    grid_w: pv.grid_w,
                    digest: image_digest(&pv.data, pv.grid_h, pv.grid_w),
                };
                ensure!(
                    identity.grid_h == self.span.grid_h
                        && identity.grid_w == self.span.grid_w
                        && identity.digest == self.digest,
                    "a remembered image preprocessed differently the second time"
                );
                Ok(Cow::Owned(pv.data))
            }
        }
    }
}

/// The identity and, when it had to be made, the rows of the image at `url`:
/// from the memo without decoding when it knows the URI, otherwise
/// preprocessed here and remembered. `k` numbers the image in errors.
fn prepare_image(
    url: String,
    k: usize,
    images: &ImagePolicy,
) -> Result<(ImagePixels, ImageIdentity)> {
    let key = ImageMemo::key(&url);
    if let Some(identity) = images.memo.get(&key) {
        return Ok((ImagePixels::Deferred { url, limits: images.limits }, identity));
    }
    let data = parse_image_data_uri(&url).with_context(|| format!("image {k}"))?;
    let pv = preprocess(&data.bytes, &images.limits)
        .with_context(|| format!("image {k} ({})", data.media_type))?;
    let identity = ImageIdentity {
        grid_h: pv.grid_h,
        grid_w: pv.grid_w,
        digest: image_digest(&pv.data, pv.grid_h, pv.grid_w),
    };
    images.memo.insert(key, identity);
    Ok((ImagePixels::Ready(pv.data), identity))
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
/// so no prompt reaches the model with a fake image span or a bare
/// placeholder. Message text cannot produce these ids
/// ([`Tokenizer::encode_chat`]); what is left to catch is a template that
/// writes markers other than one triple per image.
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
         tokens for {n} images: these placeholder tokens are reserved for image content, one \
         triple per image"
    );
    if let Some(video) = ids.video_pad {
        ensure!(
            count(video) == 0,
            "the prompt carries a <|video_pad|> token: video input is not supported"
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
    #[serde(flatten)]
    pub thinking_controls: ThinkingFields,
}

/// The thinking controls' request fields ([`crate::thinking`]), on both
/// endpoints (and in a chat request's `chat_template_kwargs`, which win).
#[derive(Deserialize, Debug, Clone, Default)]
pub struct ThinkingFields {
    /// The model's thinking tokens before the block is closed; negative
    /// turns a server default off.
    #[serde(default)]
    pub thinking_budget: Option<i64>,
    /// Nudges before the budget.
    #[serde(default)]
    pub thinking_nudges: Option<bool>,
    /// A tool call at a line start inside the block ends it.
    #[serde(default)]
    pub tool_call_ends_thinking: Option<bool>,
}

impl ThinkingFields {
    /// The fields of a `chat_template_kwargs` object (the others ignored).
    fn from_kwargs(kwargs: &serde_json::Map<String, Value>) -> Result<Self> {
        let bool_field = |name: &str| -> Result<Option<bool>> {
            match kwargs.get(name) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::Bool(b)) => Ok(Some(*b)),
                Some(other) => {
                    bail!("chat_template_kwargs.{name} must be a boolean, got {other}")
                }
            }
        };
        let thinking_budget = match kwargs.get("thinking_budget") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_i64().with_context(|| {
                format!(
                    "chat_template_kwargs.thinking_budget must be an integer, got {v}"
                )
            })?),
        };
        Ok(Self {
            thinking_budget,
            thinking_nudges: bool_field("thinking_nudges")?,
            tool_call_ends_thinking: bool_field("tool_call_ends_thinking")?,
        })
    }

    /// `self` where set, `other` elsewhere.
    fn or(self, other: Self) -> Self {
        Self {
            thinking_budget: self.thinking_budget.or(other.thinking_budget),
            thinking_nudges: self.thinking_nudges.or(other.thinking_nudges),
            tool_call_ends_thinking: self
                .tool_call_ends_thinking
                .or(other.tool_call_ends_thinking),
        }
    }

    /// The settings these fields ask for over `defaults`; `default_budget`
    /// is the server's budget for the request (by its effort and turn), and
    /// `seed` picks the inserted texts' variants.
    fn resolve(
        &self,
        defaults: &ThinkingDefaults,
        default_budget: Option<usize>,
        seed: u64,
    ) -> ThinkingSettings {
        let budget = match self.thinking_budget {
            Some(b) if b < 0 => None,
            Some(b) => Some(b as usize),
            None => default_budget,
        };
        ThinkingSettings {
            budget,
            grace: defaults.grace,
            nudges: budget.is_some() && self.thinking_nudges.unwrap_or(defaults.nudges),
            tool_call_ends_thinking: self
                .tool_call_ends_thinking
                .unwrap_or(defaults.tool_call_ends_thinking),
            seed,
        }
    }
}

/// The server's thinking budget per template reasoning effort
/// (`--thinking-budget`): `low=4000,medium=8000,xhigh=16000` (`high` is
/// `xhigh`, as in requests), or one number for every level. A level left
/// out has no budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ThinkingBudgets {
    pub low: Option<usize>,
    pub medium: Option<usize>,
    pub xhigh: Option<usize>,
}

impl ThinkingBudgets {
    pub fn parse(text: &str) -> Result<Self> {
        let text = text.trim();
        if let Ok(all) = text.parse::<usize>() {
            return Ok(Self { low: Some(all), medium: Some(all), xhigh: Some(all) });
        }
        let mut budgets = Self::default();
        for part in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (level, value) = part.split_once('=').with_context(|| {
                format!("thinking budget {part:?} is not level=tokens or a number")
            })?;
            let value: usize = value.trim().parse().with_context(|| {
                format!("thinking budget {part:?}: {value:?} is not a token count")
            })?;
            let slot = match level.trim() {
                "low" => &mut budgets.low,
                "medium" => &mut budgets.medium,
                "high" | "xhigh" => &mut budgets.xhigh,
                other => bail!(
                    "unknown reasoning effort {other:?} in the thinking budget (low, medium, xhigh)"
                ),
            };
            *slot = Some(value);
        }
        Ok(budgets)
    }

    /// The budget at a template effort (`None`: the template's default,
    /// xhigh).
    pub fn at(&self, effort: Option<&str>) -> Option<usize> {
        match effort {
            Some("low") => self.low,
            Some("medium") => self.medium,
            _ => self.xhigh,
        }
    }
}

/// The server's defaults for the thinking controls (all off unless the
/// flags turn them on), for chat requests; a raw completion gets only what
/// it asks for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThinkingDefaults {
    pub budgets: ThinkingBudgets,
    /// Scales the budget of a turn whose last message is a tool result
    /// (`--thinking-budget-tool-turn-factor`; 1 leaves it).
    pub tool_turn_factor: f64,
    pub nudges: bool,
    pub tool_call_ends_thinking: bool,
    pub grace: usize,
}

impl Default for ThinkingDefaults {
    fn default() -> Self {
        Self {
            budgets: ThinkingBudgets::default(),
            tool_turn_factor: 1.0,
            nudges: false,
            tool_call_ends_thinking: false,
            grace: crate::thinking::DEFAULT_GRACE,
        }
    }
}

impl ThinkingDefaults {
    /// The server's budget for a chat turn at `effort` (the template's,
    /// `None` for its default), after a tool result when `tool_turn`.
    pub fn budget(&self, effort: Option<&str>, tool_turn: bool) -> Option<usize> {
        let budget = self.budgets.at(effort)?;
        Some(if tool_turn {
            ((budget as f64 * self.tool_turn_factor).round() as usize).max(1)
        } else {
            budget
        })
    }
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
    #[serde(flatten)]
    pub thinking_controls: ThinkingFields,
}

/// Server-side defaults a request may override.
#[derive(Debug, Clone)]
pub struct Defaults {
    pub sampling: SamplingParams,
    /// Whether chat prompts open a reasoning block unless the request says otherwise.
    pub thinking: bool,
    /// Template `reasoning_effort` (`low`, `medium`, `xhigh`) when thinking is on.
    pub reasoning_effort: Option<String>,
    pub thinking_controls: ThinkingDefaults,
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
    /// The thinking controls asked for ([`ThinkingSettings::any`] false when
    /// none). Chat requests only close thinking at a tool call when tools
    /// were offered: without them the call would come back as text.
    pub thinking: ThinkingSettings,
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
    /// Images decoded on the connection thread, the ones the memo did not
    /// know (for the log).
    pub images_decoded: usize,
    /// Wall time of parsing and preparing the request on the connection
    /// thread, images included (set by the caller, for the log).
    pub prepare_secs: f64,
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
    let thinking_fields = match kwargs {
        Some(kwargs) => {
            ThinkingFields::from_kwargs(kwargs)?.or(request.thinking_controls.clone())
        }
        None => request.thinking_controls.clone(),
    };

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
    let mut images_decoded = 0;
    for (index, message) in request.messages.into_iter().enumerate() {
        let mut items = message
            .content
            .map(Content::into_items)
            .transpose()
            .with_context(|| format!("message {index}"))?
            .unwrap_or_default();
        if items.iter().any(|i| matches!(i, ContentItem::Image(_))) {
            if let Err(why) = &images.available {
                bail!("message {index} carries an image: {why}");
            }
            ensure!(
                message.role == "user",
                "message {index}: images are accepted in user messages only (got role {:?})",
                message.role
            );
        }
        for item in &mut items {
            let ContentItem::Image(url) = item else { continue };
            let k = prepared_images.len() + 1;
            let (pixels, identity) = prepare_image(std::mem::take(url), k, images)?;
            if matches!(pixels, ImagePixels::Ready(_)) {
                images_decoded += 1;
            }
            grids.push((identity.grid_h, identity.grid_w));
            // The span is filled in after tokenisation.
            prepared_images.push(PreparedImage {
                pixels,
                span: ImageSpan {
                    start: 0,
                    len: 0,
                    grid_h: identity.grid_h,
                    grid_w: identity.grid_w,
                },
                digest: identity.digest,
            });
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

    // Message text is encoded as text: a special token's spelling in it (a
    // model answer quoting `<|image_pad|>`, a tool result writing
    // `<|im_end|>`) never becomes the special token.
    let mut prompt = tokenizer
        .encode_chat(&ChatRender {
            messages: &messages,
            tools: tools_json.as_deref(),
            enable_thinking,
            reasoning_effort: if enable_thinking { effort.as_deref() } else { None },
            preserve_thinking,
        })
        .context("rendering the chat template")?;
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
    let sampling = resolve_sampling(&request.sampling, &defaults.sampling)?;
    let mut thinking = thinking_fields.resolve(
        &defaults.thinking_controls,
        defaults.thinking_controls.budget(
            if enable_thinking { effort.as_deref() } else { None },
            last_role == "tool",
        ),
        sampling.seed,
    );
    thinking.tool_call_ends_thinking &= tools.is_some();
    Ok(Prepared {
        kind: Kind::Chat,
        prompt,
        max_tokens: budget.max_tokens,
        sampling,
        stop_strings: request.stop.map(StringOrVec::into_vec).unwrap_or_default(),
        stream: request.stream,
        include_usage: request
            .stream_options
            .and_then(|o| o.include_usage)
            .unwrap_or(false),
        thinking_open: enable_thinking,
        thinking,
        tools,
        cache_key: request.prompt_cache_key,
        clamped_from: budget.clamped_from,
        images: prepared_images,
        images_decoded,
        prepare_secs: 0.0,
    })
}

/// Validates a raw text completion request. The prompt is encoded as given:
/// special tokens spelled in it stay special, since a client sending a raw
/// prompt renders the template itself (unlike chat, where only the
/// server's template writes them).
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
    let sampling = resolve_sampling(&request.sampling, &defaults.sampling)?;
    // A raw prompt gets exactly the controls it asks for: the server's
    // defaults are for chat, and the engine applies them only when the
    // prompt ends with `<think>\n`.
    let thinking = request.thinking_controls.resolve(
        &ThinkingDefaults {
            grace: defaults.thinking_controls.grace,
            ..ThinkingDefaults::default()
        },
        None,
        sampling.seed,
    );
    Ok(Prepared {
        kind: Kind::Completion,
        prompt,
        max_tokens: budget.max_tokens,
        sampling,
        stop_strings: request.stop.map(StringOrVec::into_vec).unwrap_or_default(),
        stream: request.stream,
        include_usage: request
            .stream_options
            .and_then(|o| o.include_usage)
            .unwrap_or(false),
        thinking_open: false,
        thinking,
        tools: None,
        cache_key: request.prompt_cache_key,
        clamped_from: budget.clamped_from,
        images: Vec::new(),
        images_decoded: 0,
        prepare_secs: 0.0,
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

/// The response's `usage` object. `reasoning_tokens` is `Some` wherever a
/// reasoning block is possible (chat completions), 0 when none was
/// generated, and lands in `completion_tokens_details` next to the draft
/// head's accepted and rejected counts, which appear only when drafts ran.
pub(super) fn usage(
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
    reasoning_tokens: Option<usize>,
    speculation: Option<Speculation>,
) -> Value {
    let mut usage = serde_json::json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
    });
    let mut details = serde_json::Map::new();
    if let Some(reasoning) = reasoning_tokens {
        details.insert("reasoning_tokens".into(), reasoning.into());
    }
    if let Some(s) = speculation {
        details.insert("accepted_prediction_tokens".into(), s.accepted.into());
        details.insert(
            "rejected_prediction_tokens".into(),
            (s.drafted - s.accepted).into(),
        );
    }
    if !details.is_empty() {
        usage["completion_tokens_details"] = Value::Object(details);
    }
    usage
}

#[cfg(test)]
#[path = "../../tests/unit/serve/api.rs"]
mod tests;
