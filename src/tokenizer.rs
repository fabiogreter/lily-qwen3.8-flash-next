//! Tokenization and chat-template rendering, read from the checkpoint.
//!
//! Both halves come out of the model directory rather than being compiled in:
//! `tokenizer.json` drives encode/decode through the `tokenizers` crate, and
//! `chat_template.jinja` is rendered with minijinja. That keeps lily's prompt
//! byte-identical to what any other server applying the same checkpoint's
//! template produces, which is the only way a golden recorded elsewhere can
//! be replayed here.
//!
//! `tests/test_tokenizer.rs` pins rendering against the checkpoint's own
//! tokenizer and verifies that direct-answer histories remain
//! prefix-cacheable.

use std::path::Path;

use anyhow::{Context as _, Result, bail};
use minijinja::value::{Kwargs, Value, ValueKind};
use minijinja::{Environment, Error as JinjaError, ErrorKind as JinjaErrorKind};

use crate::chat::{Conversation, Role};

/// Whether the generation prompt opens a reasoning block for the model to
/// continue inside, or closes an empty one so the model answers directly.
///
/// This is what OpenAI-style `chat_template_kwargs: {"enable_thinking": …}`
/// selects. It is a prompt-shape decision, not a sampling one: nothing
/// downstream of the tokenizer needs to know.
///
/// The generation prompt is byte-identical to the checkpoint's own template
/// under both settings — `<think>\n` on, `<think>\n\n</think>\n\n` off.
/// History rendering deliberately differs under [`Thinking::Disabled`]; see
/// [`Tokenizer::render_chat`].
///
/// The API server chooses [`Thinking::Disabled`] so it returns direct answers
/// and can reuse exact multi-turn token prefixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Thinking {
    #[default]
    Enabled,
    Disabled,
}

/// The closed empty reasoning block. Under [`Thinking::Disabled`] the
/// checkpoint's template emits exactly this after the assistant header, and
/// lily additionally writes it into older assistant turns — see
/// [`Tokenizer::render_chat`].
const EMPTY_REASONING_BLOCK: &str = "<think>\n\n</think>\n\n";

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    env: Environment<'static>,
    /// Ids that end a turn: the tokenizer's `eos_token`, plus whatever the
    /// caller adds from the checkpoint config.
    stop_tokens: Vec<u32>,
    /// The template drops the reasoning block from older assistant turns
    /// (Qwen3), so [`Self::render_chat`] writes an empty one in under
    /// nothink to keep histories prefix-cacheable. Newer templates
    /// (`preserve_thinking`) keep every block themselves.
    legacy_nothink_rewrite: bool,
}

/// Everything a chat prompt render needs, in the template's own vocabulary.
pub struct ChatRender<'a> {
    /// Messages as the template expects them (`role`, `content` text,
    /// optional `reasoning_content` and `tool_calls` with object arguments).
    pub messages: &'a [serde_json::Value],
    /// OpenAI tool definitions, verbatim.
    pub tools: Option<&'a [serde_json::Value]>,
    pub enable_thinking: bool,
    /// `low`, `medium` or `xhigh`; `None` leaves the template default.
    pub reasoning_effort: Option<&'a str>,
    pub preserve_thinking: Option<bool>,
}

impl Tokenizer {
    /// Loads `tokenizer.json` and the chat template from a checkpoint
    /// directory.
    ///
    /// The template is looked for in `chat_template.jinja` first and in
    /// `tokenizer_config.json`'s `chat_template` second — newer HF exports
    /// use the standalone file, older ones inline it.
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let tokenizer_path = dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| {
                format!("loading tokenizer from {}", tokenizer_path.display())
            })?;

        let config: serde_json::Value =
            match std::fs::read(dir.join("tokenizer_config.json")) {
                Ok(bytes) => serde_json::from_slice(&bytes)
                    .context("parsing tokenizer_config.json")?,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    serde_json::Value::Null
                }
                Err(e) => return Err(e).context("reading tokenizer_config.json"),
            };

        let template = Self::load_template(dir, &config)?;
        let legacy_nothink_rewrite = !template.contains("preserve_thinking");
        let mut env = Environment::new();
        // HF templates are written against Python's `str`; minijinja only
        // ships the Jinja builtins, so string methods arrive through this
        // shim. Without it the Qwen template fails on `.startswith(...)`.
        env.set_unknown_method_callback(
            minijinja_contrib::pycompat::unknown_method_callback,
        );
        // Templates signal unusable input (no user turn, a system message out
        // of position) by calling this; surface it as a render error so the
        // request is refused rather than served a malformed prompt.
        env.add_function(
            "raise_exception",
            |message: String| -> Result<Value, JinjaError> {
                Err(JinjaError::new(JinjaErrorKind::InvalidOperation, message))
            },
        );
        // The reference renders through transformers, whose `tojson` is
        // Python's `json.dumps` (`, ` and `: ` separators, no escaping).
        // minijinja's own filter emits compact JSON and escapes `<`, `>`, `&`
        // and `'`, so the tools block and every non-string tool argument
        // would render differently from the reference.
        env.add_filter("tojson", py_tojson);
        env.add_template_owned("chat", template)
            .context("compiling the checkpoint's chat template")?;

        let mut stop_tokens = Vec::new();
        if let Some(eos) = config.get("eos_token").and_then(token_text)
            && let Some(id) = inner.token_to_id(&eos)
        {
            stop_tokens.push(id);
        }

        Ok(Self { inner, env, stop_tokens, legacy_nothink_rewrite })
    }

    /// Renders one prompt through the checkpoint's template exactly as given.
    pub fn render(&self, chat: &ChatRender<'_>) -> Result<String> {
        if chat.messages.is_empty() {
            bail!("empty conversation");
        }
        let template = self.env.get_template("chat")?;
        let mut ctx = minijinja::value::Value::from_serialize(serde_json::json!({
            "messages": chat.messages,
            "tools": chat.tools,
            "add_generation_prompt": true,
            "enable_thinking": chat.enable_thinking,
        }));
        // Only pass the optional knobs when set, so the template's own
        // `is undefined` defaults apply otherwise.
        let mut extra = serde_json::Map::new();
        if let Some(effort) = chat.reasoning_effort {
            extra.insert(
                "reasoning_effort".into(),
                serde_json::Value::String(effort.into()),
            );
        }
        if let Some(flag) = chat.preserve_thinking {
            extra.insert("preserve_thinking".into(), serde_json::Value::Bool(flag));
        }
        if !extra.is_empty() {
            ctx = minijinja::context! { ..ctx, ..Value::from_serialize(serde_json::Value::Object(extra)) };
        }
        Ok(template.render(ctx)?)
    }

    /// The id of the token spelled `text` (an added token such as
    /// `<|image_pad|>`), if the vocabulary has one.
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.inner.token_to_id(text)
    }

    /// Whether `id` is a special (control) token such as `<|im_end|>`.
    pub fn is_special(&self, id: u32) -> bool {
        self.inner
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .get(&id)
            .is_some_and(|t| t.special)
    }

    fn load_template(dir: &Path, config: &serde_json::Value) -> Result<String> {
        let path = dir.join("chat_template.jinja");
        match std::fs::read_to_string(&path) {
            Ok(text) => return Ok(text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(e).with_context(|| format!("reading {}", path.display()));
            }
        }
        match config.get("chat_template").and_then(|v| v.as_str()) {
            Some(text) => Ok(text.to_string()),
            None => bail!(
                "no chat template in {}: expected chat_template.jinja or a \
                 `chat_template` field in tokenizer_config.json",
                dir.display()
            ),
        }
    }

    /// Turn-ending ids known to the tokenizer. The checkpoint config may name
    /// more; [`Self::add_stop_tokens`] folds those in.
    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop_tokens
    }

    pub fn add_stop_tokens(&mut self, ids: &[u32]) {
        for &id in ids {
            if !self.stop_tokens.contains(&id) {
                self.stop_tokens.push(id);
            }
        }
    }

    /// Raw encode. Special tokens are never added implicitly: the prompt must
    /// end exactly where the model should continue, and the chat template
    /// already writes every marker it wants.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("encoding text")?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("decoding tokens")
    }

    /// Renders a conversation through the checkpoint's chat template, ending
    /// at the assistant header so the model continues the next turn.
    ///
    /// Under [`Thinking::Disabled`] this departs from the template in one
    /// place, deliberately. The template wraps an assistant turn in
    /// `<think>…</think>` only when the turn comes *after* the last user
    /// message; older turns render as bare content. That makes a nothink
    /// reply re-render differently once another user turn arrives — and
    /// because the session cache needs the cached sequence to be an exact
    /// prefix of the new prompt, one diverging byte costs the whole prefill.
    /// So lily writes the closed empty block into every assistant turn. The
    /// rendering is otherwise the template's, byte for byte.
    pub fn render_chat(
        &self,
        messages: &Conversation,
        thinking: Thinking,
    ) -> Result<String> {
        if messages.is_empty() {
            bail!("empty conversation");
        }
        let messages = self.template_messages(messages, thinking)?;
        self.render(&ChatRender {
            messages: &messages,
            tools: None,
            enable_thinking: thinking == Thinking::Enabled,
            reasoning_effort: None,
            preserve_thinking: None,
        })
    }

    /// The message list as the template should see it: content flattened to
    /// text, and — under nothink — every assistant turn carrying the closed
    /// empty block that the template would otherwise drop for older turns.
    fn template_messages(
        &self,
        messages: &Conversation,
        thinking: Thinking,
    ) -> Result<Vec<serde_json::Value>> {
        let last_query = last_query_index(messages);
        let mut out = Vec::with_capacity(messages.len());
        for (index, message) in messages.iter().enumerate() {
            let text = message.content.clone();
            let mut value = serde_json::json!({
                "role": message.role,
                "content": text,
            });
            // The template's own wrapper covers assistant turns after the
            // last user message; only the earlier ones need the block written
            // in by hand. Setting `reasoning_content` to a string also stops
            // the template splitting `</think>` back out of the content.
            let needs_block = self.legacy_nothink_rewrite
                && thinking == Thinking::Disabled
                && message.role == Role::Assistant
                && last_query.is_some_and(|last| index <= last);
            if needs_block {
                let body = if text.starts_with(EMPTY_REASONING_BLOCK) {
                    text
                } else {
                    format!("{EMPTY_REASONING_BLOCK}{text}")
                };
                value["content"] = serde_json::Value::String(body);
                value["reasoning_content"] = serde_json::Value::String(String::new());
            }
            out.push(value);
        }
        Ok(out)
    }
}

/// `eos_token` is either a bare string or an `AddedToken` object; both
/// shapes appear in checkpoints from the same family.
fn token_text(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(map) => {
            map.get("content")?.as_str().map(str::to_string)
        }
        _ => None,
    }
}

/// transformers' template filter `tojson(x, ensure_ascii=False, indent=None,
/// separators=None, sort_keys=False)`, which is `json.dumps` with those
/// arguments, reproduced byte for byte: the separators default to `, ` and
/// `: ` (`,` between items when indenting), strings escape only what Python
/// escapes, and floats print as Python's `repr`.
fn py_tojson(value: &Value, kwargs: Kwargs) -> Result<String, JinjaError> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<usize> = kwargs.get("indent")?;
    let separators: Option<Vec<String>> = kwargs.get("separators")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    kwargs.assert_all_used()?;
    let (item, key) = match separators.as_deref() {
        Some([item, key]) => (item.clone(), key.clone()),
        Some(other) => {
            return Err(JinjaError::new(
                JinjaErrorKind::InvalidOperation,
                format!("tojson: separators must be two strings, got {}", other.len()),
            ));
        }
        None => {
            let item = if indent.is_some() { "," } else { ", " };
            (item.to_string(), ": ".to_string())
        }
    };
    let dumper = PyJson {
        ensure_ascii: ensure_ascii.unwrap_or(false),
        indent,
        item,
        key,
        sort_keys: sort_keys.unwrap_or(false),
    };
    let mut out = String::new();
    dumper.value(value, 0, &mut out)?;
    Ok(out)
}

struct PyJson {
    ensure_ascii: bool,
    indent: Option<usize>,
    item: String,
    key: String,
    sort_keys: bool,
}

impl PyJson {
    fn value(
        &self,
        value: &Value,
        level: usize,
        out: &mut String,
    ) -> Result<(), JinjaError> {
        match value.kind() {
            ValueKind::None => out.push_str("null"),
            ValueKind::Bool => {
                out.push_str(if value.is_true() { "true" } else { "false" })
            }
            ValueKind::Number if value.is_integer() => out.push_str(&value.to_string()),
            ValueKind::Number => {
                let x = f64::try_from(value.clone())?;
                out.push_str(&py_float_repr(x));
            }
            ValueKind::String => self.string(value.as_str().unwrap_or_default(), out),
            ValueKind::Seq | ValueKind::Iterable => {
                let items: Vec<Value> = value.try_iter()?.collect();
                self.container(
                    '[',
                    ']',
                    &items,
                    level,
                    out,
                    |this, item, level, out| this.value(item, level, out),
                )?;
            }
            ValueKind::Map => {
                let mut entries = value
                    .try_iter()?
                    .map(|k| Ok((py_key(&k)?, value.get_item(&k)?)))
                    .collect::<Result<Vec<_>, JinjaError>>()?;
                if self.sort_keys {
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                }
                self.container(
                    '{',
                    '}',
                    &entries,
                    level,
                    out,
                    |this, (k, v), level, out| {
                        this.string(k, out);
                        out.push_str(&this.key);
                        this.value(v, level, out)
                    },
                )?;
            }
            kind => {
                return Err(JinjaError::new(
                    JinjaErrorKind::InvalidOperation,
                    format!("tojson: a {kind} value is not JSON serializable"),
                ));
            }
        }
        Ok(())
    }

    /// `[`/`{` ... `]`/`}` around `items`, one per line at `level + 1` when
    /// indenting; empty containers stay `[]` and `{}` as in Python.
    fn container<T>(
        &self,
        open: char,
        close: char,
        items: &[T],
        level: usize,
        out: &mut String,
        mut each: impl FnMut(&Self, &T, usize, &mut String) -> Result<(), JinjaError>,
    ) -> Result<(), JinjaError> {
        out.push(open);
        if items.is_empty() {
            out.push(close);
            return Ok(());
        }
        let newline = |out: &mut String, level: usize| {
            if let Some(width) = self.indent {
                out.push('\n');
                out.extend(std::iter::repeat_n(' ', width * level));
            }
        };
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                out.push_str(&self.item);
            }
            newline(out, level + 1);
            each(self, item, level + 1, out)?;
        }
        newline(out, level);
        out.push(close);
        Ok(())
    }

    /// A JSON string as `json.dumps` writes it: `"`, `\` and the control
    /// characters escaped (short forms where JSON has them), and with
    /// `ensure_ascii` everything outside space..`~` as `\uXXXX`, astral
    /// characters as surrogate pairs.
    fn string(&self, s: &str, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20
                    || (self.ensure_ascii && !(' '..='~').contains(&c)) =>
                {
                    let mut units = [0u16; 2];
                    for unit in c.encode_utf16(&mut units) {
                        out.push_str(&format!("\\u{unit:04x}"));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
}

/// A map key as `json.dumps` coerces it: strings as they are, integers,
/// floats, booleans and None as their JSON spelling.
fn py_key(key: &Value) -> Result<String, JinjaError> {
    Ok(match key.kind() {
        ValueKind::String => key.as_str().unwrap_or_default().to_string(),
        ValueKind::None => "null".into(),
        ValueKind::Bool => (if key.is_true() { "true" } else { "false" }).into(),
        ValueKind::Number if key.is_integer() => key.to_string(),
        ValueKind::Number => py_float_repr(f64::try_from(key.clone())?),
        kind => {
            return Err(JinjaError::new(
                JinjaErrorKind::InvalidOperation,
                format!(
                    "tojson: keys must be str, int, float, bool or None, not {kind}"
                ),
            ));
        }
    })
}

/// Python's `repr(float)` (and so `json.dumps`): the shortest round-trip
/// digits, positional when the decimal exponent is in -4..16 (with at least
/// one fractional digit, `1.0`), otherwise `1e-05` / `1.5e+16` style with a
/// signed exponent of at least two digits. Non-finite values print as
/// `NaN`, `Infinity` and `-Infinity`, as `json.dumps` allows by default.
fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    // `{:e}` gives the shortest round-trip digits: `1.5e-5`, `1e2`, `0e0`.
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').expect("`{:e}` always has an exponent");
    let exp: i32 = exp.parse().expect("`{:e}` writes an integer exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if x == 0.0 || (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
        let body = if point <= 0 {
            format!("0.{}{digits}", "0".repeat((-point) as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            let (int, frac) = digits.split_at(point as usize);
            format!("{int}.{frac}")
        };
        format!("{sign}{body}")
    } else {
        let exp_sign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exp_sign}{:02}", exp.abs())
    }
}

/// The last user turn, matching the only query role the minimal API accepts.
fn last_query_index(messages: &Conversation) -> Option<usize> {
    messages.iter().rposition(|message| message.role == Role::User)
}

#[cfg(test)]
#[path = "../tests/unit/tokenizer.rs"]
mod tests;
