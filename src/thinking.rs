//! Decode-time controls for overlong thinking.
//!
//! The model sometimes reasons for many thousands of tokens without closing
//! its `<think>` block, and sometimes writes a tool call inside the block
//! and ends the turn there, so the client receives nothing but reasoning.
//! Three controls, all off unless a request (or the server's defaults) asks
//! for them, act on the token stream while it is drawn:
//!
//! - **A tool call ends thinking** (`tool_call_ends_thinking`): a
//!   `<tool_call>` drawn at the start of a line while the block is open is
//!   taken as the end of the reasoning, and `</think>\n\n` alone is inserted
//!   in front of it (no preface: the model has already decided to act), so
//!   the state and the client see the shape the template writes for a
//!   reasoned tool call (`...\n</think>\n\n<tool_call>`). vLLM's Qwen3
//!   reasoning parser makes the same call on the response side; the
//!   server's output parser does too, as a safety net
//!   (`serve::stream`).
//! - **A thinking budget** (`thinking_budget`): once the block holds that
//!   many of the model's tokens, the block is closed at the next good place
//!   (see [`ThinkingSettings::grace`]) with a short action-oriented preface
//!   and `\n</think>\n\n`, the way llama.cpp's reasoning budget ends a block
//!   early. The precedent is Qwen's own Qwen3 thinking-budget recipe, which
//!   stops the block with "Considering the limited time by the user, I have
//!   to give the solution based on the thinking directly now.\n</think>.\n\n"
//!   and lets the model answer from what it has.
//! - **Nudges** (`thinking_nudges`, with a budget): at fractions of the
//!   budget (by default 50 %, 75 % and 90 %), a sentence of increasingly
//!   firm wording is inserted into the reasoning on a paragraph of its own,
//!   and the model goes on thinking. Each level has a few variants, picked
//!   by the request's seed, so an agent's history does not repeat one phrase
//!   turn after turn; a template that keeps old reasoning
//!   (`preserve_thinking`) keeps these texts in the history.
//!
//! Insertions land only at a line end, never inside a code fence or a tool
//! call: within the grace window after the threshold the control waits for a
//! token that ends a line, within a second window it also takes a sentence
//! end, and after both it forces the close (a nudge that cannot land is
//! dropped instead). Every inserted text is untested as a prompt: the
//! wording is configurable (`--thinking-texts`).
//!
//! [`ThinkingControl`] is the state machine: it sees every token the
//! generation emits, in order, with its text, and answers with an
//! [`Action`]. It is pure host logic; the decode loops (`generate.rs`, the
//! batch scheduler) feed the tokens it asks for into the state exactly like
//! drawn ones, so the state, the session's token list and the client all
//! see one sequence. A loop that pipelines its steps cannot insert a token
//! in front of or right after one an already committed step feeds, so it
//! asks [`ThinkingControl::may_act_next`] before it commits a step ahead and
//! decodes unpipelined while that says yes (a line start with tool calls
//! ending thinking, the windows around a nudge or the budget): rare steps,
//! so the pipelining is kept nearly everywhere.

use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use serde::Deserialize;

/// Tokens of each of the two waiting windows after a threshold (see the
/// module docs): an insertion lands at most twice this many tokens late.
pub const DEFAULT_GRACE: usize = 128;

/// One nudge level: the fraction of the budget it comes at and its
/// variants.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NudgeLevel {
    pub at: f64,
    pub texts: Vec<String>,
}

/// The texts the controls insert: the budget's close prefaces and the
/// nudge levels, each with variants. [`Default`] is the built-in set; a
/// JSON file of the same shape replaces it (`--thinking-texts`):
/// `{"close": ["..."], "nudges": [{"at": 0.5, "texts": ["..."]}]}`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThinkingTexts {
    #[serde(default)]
    pub close: Vec<String>,
    #[serde(default)]
    pub nudges: Vec<NudgeLevel>,
}

impl Default for ThinkingTexts {
    fn default() -> Self {
        let texts = |t: &[&str]| t.iter().map(|s| (*s).to_owned()).collect();
        Self {
            close: texts(&[
                "I've thought enough; time to act: make the next tool call or give the answer.",
                "That is enough thinking. Now I act: the next tool call, or the answer.",
                "Time to stop thinking and act: make the next tool call or give the answer.",
            ]),
            nudges: vec![
                NudgeLevel {
                    at: 0.5,
                    texts: texts(&[
                        "I should start converging on a decision.",
                        "Time to start converging on a decision.",
                        "I should begin narrowing this down to a decision.",
                    ]),
                },
                NudgeLevel {
                    at: 0.75,
                    texts: texts(&[
                        "I need to wrap up my thinking and decide the next action.",
                        "I need to wrap this up and settle on the next action.",
                        "Time to wrap up and decide what to do next.",
                    ]),
                },
                NudgeLevel {
                    at: 0.9,
                    texts: texts(&[
                        "I must stop deliberating now.",
                        "I must stop deliberating and act now.",
                        "No more deliberation: I must act now.",
                    ]),
                },
            ],
        }
    }
}

impl ThinkingTexts {
    /// Reads a JSON file of [`ThinkingTexts`].
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing thinking texts in {}", path.display()))
    }
}

/// An inserted text encoded for the three places it can land: after a
/// token whose text ends a paragraph (`\n\n`), a line (`\n`), or in the
/// middle of a line. Each puts the text on a paragraph of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Placed {
    after_paragraph: Vec<u32>,
    after_line: Vec<u32>,
    mid_line: Vec<u32>,
}

impl Placed {
    fn at(&self, trailing_newlines: usize) -> Vec<u32> {
        match trailing_newlines {
            0 => self.mid_line.clone(),
            1 => self.after_line.clone(),
            _ => self.after_paragraph.clone(),
        }
    }
}

/// The vocabulary's markers and every token sequence the controls insert,
/// encoded once per server.
#[derive(Debug, Clone, PartialEq)]
pub struct ThinkingTokens {
    pub think_end: u32,
    pub tool_call: u32,
    pub tool_call_end: u32,
    /// `</think>\n\n`: what a tool call that ends thinking gets in front.
    close_tag: Vec<u32>,
    /// The budget's close variants: preface, line end, `</think>\n\n`.
    closes: Vec<Placed>,
    /// The nudge levels (fraction of the budget, variants), ascending.
    nudges: Vec<(f64, Vec<Placed>)>,
}

impl ThinkingTokens {
    /// The sequences for `texts`, with `encode` the tokenizer's plain
    /// encoding (it matches `</think>` as its token). The pieces are encoded
    /// separately from the text before them, so the ids at the seam can
    /// differ from what encoding the whole text would give (a later turn
    /// that re-renders the reasoning may then fork the session cache a few
    /// tokens early); the state is fed exactly these ids either way.
    pub fn new(
        think_end: u32,
        tool_call: u32,
        tool_call_end: u32,
        texts: &ThinkingTexts,
        encode: impl Fn(&str) -> Result<Vec<u32>>,
    ) -> Result<Self> {
        let paragraph = encode("\n\n")?;
        ensure!(!paragraph.is_empty(), "\"\\n\\n\" encodes to nothing");
        let mut close_tag = vec![think_end];
        close_tag.extend_from_slice(&paragraph);
        let markers = [think_end, tool_call, tool_call_end];
        // `text` on a paragraph of its own after each kind of place, then
        // `tail`; an empty text is just a line end before the tail.
        let place = |text: &str, end: &str, tail: &[u32]| -> Result<Placed> {
            let text = text.trim();
            let with = |sep: &str| -> Result<Vec<u32>> {
                let s = if text.is_empty() {
                    if sep.is_empty() { String::new() } else { "\n".to_owned() }
                } else {
                    format!("{sep}{text}{end}")
                };
                let mut ids = if s.is_empty() { Vec::new() } else { encode(&s)? };
                ensure!(
                    !ids.iter().any(|id| markers.contains(id)),
                    "the thinking text {text:?} spells </think> or a tool call tag"
                );
                ids.extend_from_slice(tail);
                ensure!(!ids.is_empty(), "an inserted text encodes to nothing");
                Ok(ids)
            };
            Ok(Placed {
                after_paragraph: with("")?,
                after_line: with("\n")?,
                mid_line: with("\n\n")?,
            })
        };
        let mut closes = texts
            .close
            .iter()
            .map(|t| place(t, "\n", &close_tag))
            .collect::<Result<Vec<_>>>()?;
        if closes.is_empty() {
            closes.push(place("", "\n", &close_tag)?);
        }
        let mut nudges = Vec::with_capacity(texts.nudges.len());
        let mut previous = 0.0;
        for level in &texts.nudges {
            ensure!(
                level.at > previous && level.at < 1.0,
                "nudge levels must be ascending fractions of the budget in (0, 1), got {}",
                level.at
            );
            previous = level.at;
            let variants = level
                .texts
                .iter()
                .filter(|t| !t.trim().is_empty())
                .map(|t| place(t, "\n\n", &[]))
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                !variants.is_empty(),
                "the nudge level at {} has no text",
                level.at
            );
            nudges.push((level.at, variants));
        }
        Ok(Self { think_end, tool_call, tool_call_end, close_tag, closes, nudges })
    }
}

/// What a request asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ThinkingSettings {
    /// The model's tokens in the block before it is closed.
    pub budget: Option<usize>,
    /// The two waiting windows' length (see the module docs; 0 inserts
    /// right at the threshold).
    pub grace: usize,
    /// Nudges before the budget (only with a budget).
    pub nudges: bool,
    pub tool_call_ends_thinking: bool,
    /// Picks the variant of each inserted text (the request's sampling seed).
    pub seed: u64,
}

impl ThinkingSettings {
    /// Whether any control is on (otherwise no [`ThinkingControl`] is made
    /// and the decode is exactly what it was without them).
    pub fn any(&self) -> bool {
        self.budget.is_some() || self.tool_call_ends_thinking
    }
}

/// What to do with a token the generation is about to emit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Emit it as drawn.
    Keep,
    /// Emit these tokens first, then it (a tool call that ends thinking).
    Before(Vec<u32>),
    /// Emit it, then these tokens (a nudge, or the budget's close).
    After(Vec<u32>),
}

/// Which control closed the block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosedBy {
    ToolCall,
    Budget,
}

/// A close the control inserted: by what, after how many of the model's
/// thinking tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Closed {
    pub by: ClosedBy,
    pub thinking_tokens: usize,
}

/// Where a waiting insertion may land now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    /// Not here; keep waiting.
    Wait,
    Here,
    /// Both windows are over and this is no place for it (a fence): a
    /// nudge is dropped.
    Never,
}

/// One generation's thinking state (see the module docs).
#[derive(Debug, Clone)]
pub struct ThinkingControl {
    tokens: Arc<ThinkingTokens>,
    settings: ThinkingSettings,
    /// The reasoning block is open: the prompt opened it and nothing has
    /// closed it yet.
    open: bool,
    /// The model's own tokens emitted inside the block (inserted nudges
    /// are not counted).
    counted: usize,
    /// The next nudge level to insert.
    level: usize,
    nudged: usize,
    /// `\n` characters the emitted text ends with (the prompt's `<think>\n`
    /// counts: the first token starts a line).
    trailing_newlines: usize,
    /// The last character that is not whitespace ends a sentence.
    sentence_end: bool,
    /// Backticks at the start of the current line (indentation skipped),
    /// until something else arrives there.
    lead_ticks: u8,
    lead_done: bool,
    in_fence: bool,
    /// Between a `<tool_call>` the block kept and its `</tool_call>`.
    in_tool_call: bool,
    closed: Option<Closed>,
}

impl ThinkingControl {
    /// The control for a generation whose prompt `opens_thinking` (ends
    /// with `<think>\n`); one that does not is never in a block and never
    /// acts.
    pub fn new(
        tokens: Arc<ThinkingTokens>,
        settings: ThinkingSettings,
        opens_thinking: bool,
    ) -> Self {
        Self {
            tokens,
            settings,
            open: opens_thinking,
            counted: 0,
            level: 0,
            nudged: 0,
            trailing_newlines: 1,
            sentence_end: false,
            lead_ticks: 0,
            lead_done: false,
            in_fence: false,
            in_tool_call: false,
            closed: None,
        }
    }

    /// Whether the reasoning block is still open.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// The close this control inserted, if any.
    pub fn closed(&self) -> Option<Closed> {
        self.closed
    }

    /// Nudges inserted so far.
    pub fn nudged(&self) -> usize {
        self.nudged
    }

    /// The model's tokens counted in the block so far.
    pub fn thinking_tokens(&self) -> usize {
        self.counted
    }

    /// The count at which nudge `level` is due.
    fn nudge_at(&self, level: usize) -> Option<usize> {
        let budget = self.settings.budget?;
        if !self.settings.nudges {
            return None;
        }
        let (at, _) = self.tokens.nudges.get(level)?;
        Some(((at * budget as f64).ceil() as usize).max(1))
    }

    /// Whether [`Self::decide`] may answer anything but [`Action::Keep`] for
    /// the next token: a step that draws it must not have a following step
    /// committed behind it, since that step would feed the token before
    /// anything inserted in front of or after it. True at a line start
    /// inside the block when tool calls end thinking, and from the token
    /// that reaches a nudge's or the budget's threshold until it landed.
    pub fn may_act_next(&self) -> bool {
        let next = self.counted + 1;
        self.open
            && ((self.settings.tool_call_ends_thinking && self.trailing_newlines > 0)
                || self.settings.budget.is_some_and(|b| next >= b)
                || self.nudge_at(self.level).is_some_and(|at| next >= at))
    }

    /// Takes the next token the generation emits (never a stop token) and
    /// its text (the token decoded alone, special tokens spelled out), and
    /// says what to do. Tokens it asks to insert are emitted without being
    /// passed back through here.
    pub fn decide(&mut self, token: u32, text: &str) -> Action {
        if !self.open {
            return Action::Keep;
        }
        let tokens = Arc::clone(&self.tokens);
        if token == tokens.think_end {
            // Closed by the model itself.
            self.open = false;
            return Action::Keep;
        }
        if token == tokens.tool_call
            && self.settings.tool_call_ends_thinking
            && self.trailing_newlines > 0
        {
            self.open = false;
            self.closed =
                Some(Closed { by: ClosedBy::ToolCall, thinking_tokens: self.counted });
            return Action::Before(tokens.close_tag.clone());
        }
        self.counted += 1;
        if token == tokens.tool_call {
            self.in_tool_call = true;
        } else if token == tokens.tool_call_end {
            self.in_tool_call = false;
        }
        self.observe_text(text);
        let Some(budget) = self.settings.budget else {
            return Action::Keep;
        };
        let seed = self.settings.seed as usize;
        if self.counted >= budget {
            // A close is never dropped: past both windows it lands in a
            // fence too, though still not inside a tool call.
            if matches!(self.place(self.counted - budget), Place::Here | Place::Never)
                && !self.in_tool_call
            {
                self.open = false;
                self.closed = Some(Closed {
                    by: ClosedBy::Budget,
                    thinking_tokens: self.counted,
                });
                let variant = &tokens.closes[seed % tokens.closes.len()];
                return Action::After(variant.at(self.trailing_newlines));
            }
            return Action::Keep;
        }
        let Some(at) = self.nudge_at(self.level).filter(|&at| self.counted >= at)
        else {
            return Action::Keep;
        };
        match self.place(self.counted - at) {
            Place::Wait => Action::Keep,
            Place::Never => {
                self.level += 1;
                Action::Keep
            }
            Place::Here => {
                let (_, variants) = &tokens.nudges[self.level];
                let variant = &variants[(seed + self.level) % variants.len()];
                let ids = variant.at(self.trailing_newlines);
                self.level += 1;
                self.nudged += 1;
                // The text ends a paragraph: the model goes on at a line
                // start, outside any fence (a nudge never lands in one).
                self.trailing_newlines = 2;
                self.sentence_end = false;
                self.lead_ticks = 0;
                self.lead_done = false;
                Action::After(ids)
            }
        }
    }

    /// Whether an insertion `over` tokens past its threshold may land after
    /// the token just observed: at a line end in the first window, also at
    /// a sentence end in the second, anywhere after both; never inside a
    /// tool call or a code fence ([`Place::Never`] once a fence outlasts
    /// both windows).
    fn place(&self, over: usize) -> Place {
        let grace = self.settings.grace;
        if self.in_tool_call {
            return Place::Wait;
        }
        if self.in_fence {
            return if over >= 2 * grace { Place::Never } else { Place::Wait };
        }
        let line_end = self.trailing_newlines > 0;
        if over >= 2 * grace || line_end || (over >= grace && self.sentence_end) {
            Place::Here
        } else {
            Place::Wait
        }
    }

    /// Line ends, sentence ends and code fences: a line whose first
    /// non-blank characters are three backticks opens or closes a fence.
    fn observe_text(&mut self, text: &str) {
        for c in text.chars() {
            if c == '\n' {
                self.trailing_newlines += 1;
                self.lead_ticks = 0;
                self.lead_done = false;
                continue;
            }
            self.trailing_newlines = 0;
            if !c.is_whitespace() {
                self.sentence_end = matches!(c, '.' | '!' | '?');
            }
            if self.lead_done {
                continue;
            }
            match c {
                '`' => {
                    self.lead_ticks += 1;
                    if self.lead_ticks == 3 {
                        self.in_fence = !self.in_fence;
                        self.lead_done = true;
                    }
                }
                ' ' | '\t' if self.lead_ticks == 0 => {}
                _ => self.lead_done = true,
            }
        }
    }
}

/// Whether a prompt's text ends inside a freshly opened reasoning block,
/// the generation prompt the template writes with thinking on.
pub fn opens_thinking(prompt_tail: &str) -> bool {
    prompt_tail.ends_with("<think>\n")
}

#[cfg(test)]
#[path = "../tests/unit/thinking.rs"]
mod tests;
