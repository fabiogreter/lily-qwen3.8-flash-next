//! Decode-time controls for overlong thinking.
//!
//! The model sometimes reasons for many thousands of tokens without closing
//! its `<think>` block, and sometimes writes a tool call inside the block
//! and ends the turn there, so the client receives nothing but reasoning.
//! Four controls, all off unless a request (or the server's defaults) asks
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
//!   firm wording is inserted into the reasoning on a line of its own, and
//!   the model goes on thinking. Each level has a few variants, picked by
//!   the request's seed, so an agent's history does not repeat one phrase
//!   turn after turn; a template that keeps old reasoning
//!   (`preserve_thinking`) keeps these texts in the history.
//! - **An end of turn inside the block is replaced** (with a budget): a stop
//!   token (`<|im_end|>`, EOS) drawn while the block is open would end the
//!   turn with nothing but reasoning, which an agent sees as a stalled turn.
//!   It is not emitted or fed; `\n</think>\n\n` alone is inserted instead
//!   (no preface: the model has decided it is done thinking) and the decode
//!   goes on, so the model writes its tool call or answer. Once per request:
//!   the block is closed then, and a stop token after it ends the turn as
//!   usual. A client's stop *string* is the output parser's business and is
//!   left alone.
//!
//! Insertions never land inside a code fence or a tool call. A nudge lands
//! only at a line end, within the grace window after its threshold; a level
//! that finds none is dropped. The budget's close waits for a line end in
//! the grace window, also takes a sentence end in a second window, and lands
//! anywhere after both (but not right after a space or tab mid-line, see
//! below), still not inside a fence or a tool call: it waits for those to
//! end, bounded only by `max_tokens` (a fence the model never closes means
//! no close). A `<tool_call>` inside a fence (an example the reasoning
//! quotes), or anywhere but at the start of a line, is reasoning text: it
//! neither ends the block nor counts as a tool call the block kept. Every
//! inserted text is untested as a prompt: the wording is configurable
//! (`--thinking-texts`).
//!
//! Fences follow CommonMark ([`ReasoningScan`]): a line of at least three
//! backticks or tildes, indented by at most three spaces, opens one (a
//! backtick fence's info string has no backtick), and only a line of the
//! same character, at least as long, with nothing after it but whitespace,
//! closes it. A block indented by four spaces is code to CommonMark too,
//! but not to these rules: an insertion or a `<tool_call>` line there is
//! taken as reasoning outside a fence. The control and the output parser
//! read the reasoning with the same [`ReasoningScan`], so they agree on
//! every `<tool_call>`.
//!
//! **Seams.** An agent's next turn re-renders the reasoning as text, and
//! the session cache reuses the state only as far as that re-encoding
//! matches the tokens the state was fed. So every insertion is made of the
//! ids the whole text encodes to, given the model's token in front of it:
//! at a line start the text follows directly (no blank line: a second `\n`
//! after the model's `\n` would encode as one `\n\n` token); mid-line after
//! a letter, mark or digit it follows a paragraph break; mid-line after
//! anything else (punctuation, a symbol) a space comes before the break,
//! since the tokenizer's pre-split takes a punctuation run together with
//! the line ends after it, while a space starts a piece of its own after
//! any character. Mid-line right after a space or tab no seam is clean
//! (whitespace and the line ends after it are one piece): the budget's close
//! waits one more token there; a replaced end of turn cannot wait and takes
//! the break. Each text ends in a line end (a nudge in `\n\n`, the close in
//! `\n</think>\n\n`), which the model's next token, starting with anything
//! but another line end, does not merge with. The pieces are encoded once
//! per server; the tokenizer test checks the seams against the whole-text
//! encoding.
//!
//! [`ThinkingControl`] is the state machine: it sees every token the
//! generation emits, in order, with its text, and every stop token drawn,
//! and answers with an [`Action`]. It is pure host logic; the decode loops
//! (`generate.rs`, the batch scheduler) feed the tokens it asks for into
//! the state exactly like drawn ones, so the state, the session's token
//! list and the client all see one sequence. A loop that pipelines its
//! steps cannot insert a token in front of or right after one an already
//! committed step feeds, nor take back a stop token it fed, so it asks
//! [`ThinkingControl::may_act_next`] before it commits a step ahead and
//! decodes unpipelined while that says yes. That is not rare: with a budget
//! it says yes at every token inside the block (a stop token can come at
//! any of them), and with only `tool_call_ends_thinking` at every line
//! start inside the block. The plain loop then decodes unpipelined there,
//! and the batch scheduler parks no step while such a row is in its block;
//! the speculative loop, which rests between verify passes anyway, is not
//! affected. Outside the block, and without the controls, nothing changes.

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

/// Tokens the controls insert at one place, with their text (which the
/// control reads back like the model's: the output parser reads it too).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Insert {
    ids: Vec<u32>,
    text: String,
}

/// What the emitted text ends with where an insertion lands, which decides
/// how the inserted text starts (see the module docs, "Seams").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Seam {
    /// A line end (one or more).
    LineStart,
    /// Mid-line after a letter, mark or digit.
    Word,
    /// Mid-line after any other character but whitespace.
    Symbol,
}

/// An inserted text encoded for each [`Seam`]; each puts the text on a line
/// of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Placed {
    line_start: Insert,
    word: Insert,
    symbol: Insert,
}

impl Placed {
    fn at(&self, seam: Seam) -> &Insert {
        match seam {
            Seam::LineStart => &self.line_start,
            Seam::Word => &self.word,
            Seam::Symbol => &self.symbol,
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
    /// The close without a preface, in place of an end of turn.
    bare_close: Placed,
    /// The nudge levels (fraction of the budget, variants), ascending.
    nudges: Vec<(f64, Vec<Placed>)>,
}

impl ThinkingTokens {
    /// The sequences for `texts`, with `encode` the tokenizer's plain
    /// encoding (it matches `</think>` as its token) and `is_added` whether
    /// an id is one of the vocabulary's added tokens (special or not). Each
    /// piece is encoded on its own, and the pieces start and end where the
    /// whole text's encoding has a boundary too (see the module docs,
    /// "Seams"). A text whose encoding holds an added token (`</think>`, a
    /// tool call tag, `<|im_end|>`, ...) is refused: inserted, it would
    /// close the block, open a call or end the turn behind the loops' back.
    pub fn new(
        think_end: u32,
        tool_call: u32,
        tool_call_end: u32,
        texts: &ThinkingTexts,
        encode: impl Fn(&str) -> Result<Vec<u32>>,
        is_added: impl Fn(u32) -> bool,
    ) -> Result<Self> {
        let paragraph = encode("\n\n")?;
        ensure!(!paragraph.is_empty(), "\"\\n\\n\" encodes to nothing");
        let mut close_tag = vec![think_end];
        close_tag.extend_from_slice(&paragraph);
        let close_text = "</think>\n\n";
        let markers = [think_end, tool_call, tool_call_end];
        // `text` on a line of its own after each kind of seam, then `end`
        // and `tail` (ids and text); an empty text is just a line end where
        // one is needed.
        let place = |text: &str, end: &str, tail: (&[u32], &str)| -> Result<Placed> {
            let text = text.trim();
            let with = |lead: &str, empty: &str| -> Result<Insert> {
                let s = if text.is_empty() {
                    empty.to_owned()
                } else {
                    format!("{lead}{text}{end}")
                };
                let mut ids = if s.is_empty() { Vec::new() } else { encode(&s)? };
                ensure!(
                    !ids.iter().any(|&id| is_added(id) || markers.contains(&id)),
                    "the thinking text {text:?} encodes a special or added token \
                     (such as </think>, a tool call tag or <|im_end|>)"
                );
                ids.extend_from_slice(tail.0);
                ensure!(!ids.is_empty(), "an inserted text encodes to nothing");
                Ok(Insert { ids, text: s + tail.1 })
            };
            Ok(Placed {
                line_start: with("", "")?,
                word: with("\n\n", "\n")?,
                symbol: with(" \n\n", " \n")?,
            })
        };
        let closes = texts
            .close
            .iter()
            .map(|t| place(t, "\n", (&close_tag, close_text)))
            .collect::<Result<Vec<_>>>()?;
        let bare_close = place("", "\n", (&close_tag, close_text))?;
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
                .map(|t| place(t, "\n\n", (&[], "")))
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                !variants.is_empty(),
                "the nudge level at {} has no text",
                level.at
            );
            nudges.push((level.at, variants));
        }
        Ok(Self {
            think_end,
            tool_call,
            tool_call_end,
            close_tag,
            closes,
            bare_close,
            nudges,
        })
    }
}

/// What a request asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ThinkingSettings {
    /// The model's tokens in the block before it is closed. Also turns on
    /// the replacement of an end of turn inside the block.
    pub budget: Option<usize>,
    /// The waiting windows' length (see the module docs): a nudge's one
    /// window, the close's two. 0 closes right at the budget (outside a
    /// fence or a tool call, and not right after a space mid-line) and lets
    /// a nudge land only on the threshold's own token.
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
    /// Emit it as drawn (a stop token: end the generation as usual).
    Keep,
    /// Emit these tokens first, then it (a tool call that ends thinking).
    Before(Vec<u32>),
    /// Emit it, then these tokens (a nudge, or the budget's close).
    After(Vec<u32>),
    /// Drop it (a stop token, never emitted or fed) and emit these tokens
    /// instead (the close in place of an end of turn); the generation goes
    /// on.
    Replace(Vec<u32>),
}

/// Which control closed the block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosedBy {
    ToolCall,
    Budget,
    /// A stop token drawn inside the block, replaced by the close.
    EndOfTurn,
}

/// A close the control asked for: by what, after how many of the model's
/// thinking tokens. The loops may still cut it short at `max_tokens`
/// ([`ThinkingControl::close_emitted`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Closed {
    pub by: ClosedBy,
    pub thinking_tokens: usize,
}

/// Where a waiting nudge may land now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    /// Not here; keep waiting.
    Wait,
    Here,
    /// Its window is over without a line end it could take: dropped.
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
    /// Lines, fences and tool calls of the block's text so far, inserted
    /// nudges included.
    scan: ReasoningScan,
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
            scan: ReasoningScan::new(),
            closed: None,
        }
    }

    /// Whether the reasoning block is still open.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// The close this control asked for, if any.
    pub fn closed(&self) -> Option<Closed> {
        self.closed
    }

    /// Whether the close this control asked for made it into `generated`
    /// (the generation's emitted tokens): `max_tokens` can cut an inserted
    /// group short, or leave no room for it at all. False without a close.
    pub fn close_emitted(&self, generated: &[u32]) -> bool {
        // The block was open until the close, so the model drew no
        // `</think>` before it; a generation cut short ends inside the
        // group, so none after it either.
        self.closed.is_some() && generated.contains(&self.tokens.think_end)
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

    /// Whether [`Self::decide`] or [`Self::decide_stop`] may answer
    /// anything but [`Action::Keep`] for the next draw: a step that draws
    /// it must not have a following step committed behind it, since that
    /// step would feed the draw before anything inserted in front of or
    /// after it, or feed a stop token that is to be replaced. With a budget
    /// that is every draw inside the block (a stop token can come at any);
    /// with only tool calls ending thinking, every line start inside it.
    pub fn may_act_next(&self) -> bool {
        self.open && (self.settings.budget.is_some() || self.tool_call_would_close())
    }

    /// Whether a `<tool_call>` emitted next would end the block: the rule
    /// is on and a call may start here ([`ReasoningScan::call_may_start`]).
    fn tool_call_would_close(&self) -> bool {
        self.settings.tool_call_ends_thinking && self.scan.call_may_start()
    }

    /// Where an insertion lands if it lands now, or `None` mid-line right
    /// after whitespace, where no seam is clean (see the module docs).
    fn seam(&self) -> Option<Seam> {
        if self.scan.trailing_newlines() > 0 {
            return Some(Seam::LineStart);
        }
        match self.scan.last_char() {
            None => Some(Seam::LineStart),
            Some(c) if c.is_whitespace() => None,
            // `is_alphanumeric` is within the letters, marks and digits the
            // pre-split keeps apart from line ends; anything it misses
            // takes the space, which is clean after any character.
            Some(c) if c.is_alphanumeric() => Some(Seam::Word),
            Some(_) => Some(Seam::Symbol),
        }
    }

    /// Takes the next token the generation emits (never a stop token: see
    /// [`Self::decide_stop`]) and its text (the token decoded alone, special
    /// tokens spelled out), and says what to do. Tokens it asks to insert
    /// are emitted without being passed back through here.
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
        if token == tokens.tool_call && self.tool_call_would_close() {
            self.open = false;
            self.closed =
                Some(Closed { by: ClosedBy::ToolCall, thinking_tokens: self.counted });
            return Action::Before(tokens.close_tag.clone());
        }
        self.counted += 1;
        self.scan.feed(text);
        let Some(budget) = self.settings.budget else {
            return Action::Keep;
        };
        // The seed picks a variant; any u64 (a request's `seed: -1` too).
        let pick = |offset: usize, len: usize| {
            (self.settings.seed.wrapping_add(offset as u64) % len as u64) as usize
        };
        if self.counted >= budget {
            // A close is never dropped, and never lands inside a code fence
            // or a tool call: it waits for them to end, bounded only by
            // `max_tokens` (a fence the model never closes gets no close).
            if let Some(seam) = self.close_seam(self.counted - budget) {
                self.open = false;
                self.closed = Some(Closed {
                    by: ClosedBy::Budget,
                    thinking_tokens: self.counted,
                });
                let ids = match tokens.closes.len() {
                    0 => &tokens.bare_close,
                    n => &tokens.closes[pick(0, n)],
                }
                .at(seam)
                .ids
                .clone();
                return Action::After(ids);
            }
            return Action::Keep;
        }
        let Some(at) = self.nudge_at(self.level).filter(|&at| self.counted >= at)
        else {
            return Action::Keep;
        };
        match self.nudge_place(self.counted - at) {
            Place::Wait => Action::Keep,
            Place::Never => {
                self.level += 1;
                Action::Keep
            }
            Place::Here => {
                let (_, variants) = &tokens.nudges[self.level];
                let insert =
                    variants[pick(self.level, variants.len())].at(Seam::LineStart);
                self.level += 1;
                self.nudged += 1;
                // The text is read like the model's: it ends a paragraph,
                // and the model goes on at a line start.
                self.scan.feed(&insert.text);
                Action::After(insert.ids.clone())
            }
        }
    }

    /// Takes a stop token the generation drew, before it ends the
    /// generation: with a budget, inside the block, [`Action::Replace`]
    /// with the close (`\n</think>\n\n` placed by the seam rules, no
    /// preface), and the block is closed; otherwise [`Action::Keep`]. So it
    /// replaces at most one per generation.
    // TODO(thinking): the replaced draw stays in the sampler's penalty
    // counts (with penalties on), which makes a later end of turn slightly
    // less likely. Undoing it needs an engine call that reaches the row's
    // counts slot (engine scratch or batch slot) in all three loops; left
    // until it can be tested on the GPU.
    pub fn decide_stop(&mut self) -> Action {
        if !self.open || self.settings.budget.is_none() {
            return Action::Keep;
        }
        self.open = false;
        self.closed =
            Some(Closed { by: ClosedBy::EndOfTurn, thinking_tokens: self.counted });
        // The model is done thinking: the close comes now, inside a fence
        // or a tool call too, and right after whitespace with the break
        // (no clean seam there).
        let seam = self.seam().unwrap_or(Seam::Word);
        Action::Replace(self.tokens.bare_close.at(seam).ids.clone())
    }

    /// Where the budget's close, `over` tokens past the budget, lands after
    /// the token just observed: at a line end in the first grace window,
    /// also at a sentence end in the second, anywhere after both; never
    /// inside a tool call or a code fence, however long, nor mid-line right
    /// after whitespace.
    fn close_seam(&self, over: usize) -> Option<Seam> {
        if self.scan.in_call() || self.scan.in_fence() {
            return None;
        }
        let seam = self.seam()?;
        let grace = self.settings.grace;
        let sentence_end = matches!(self.scan.last_char(), Some('.' | '!' | '?'));
        (over >= 2 * grace
            || seam == Seam::LineStart
            || (over >= grace && sentence_end))
            .then_some(seam)
    }

    /// Where a nudge `over` tokens past its threshold goes: only at a line
    /// end outside a code fence and a tool call, within its window (the
    /// threshold's token and `grace` more); a level that finds none is
    /// dropped ([`Place::Never`]). A nudge is optional, unlike the close.
    fn nudge_place(&self, over: usize) -> Place {
        let grace = self.settings.grace;
        if over <= grace
            && self.scan.trailing_newlines() > 0
            && !self.scan.in_fence()
            && !self.scan.in_call()
        {
            Place::Here
        } else if over >= self.settings.grace {
            Place::Never
        } else {
            Place::Wait
        }
    }
}

const TOOL_START: &str = "<tool_call>";
const TOOL_END: &str = "</tool_call>";

/// An open code fence, or a line's run that may be one: its character and
/// length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fence {
    ch: char,
    len: usize,
}

/// What the current line starts with, for the fence rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lead {
    /// Spaces only so far (at most three).
    Indent(u8),
    /// A run of backticks or tildes, nothing after it yet.
    Run(Fence),
    /// Something after a run of three or more: only whitespace so far
    /// (`blank`), a backtick among it (`tick`).
    After { run: Fence, blank: bool, tick: bool },
    /// Not a fence line.
    Text,
}

impl Lead {
    fn next(self, c: char) -> Self {
        match self {
            Lead::Indent(n) => match c {
                ' ' if n < 3 => Lead::Indent(n + 1),
                '`' | '~' => Lead::Run(Fence { ch: c, len: 1 }),
                // A fourth space or a tab: an indented code line to
                // CommonMark, not a fence (and not tracked as code here).
                _ => Lead::Text,
            },
            Lead::Run(run) if c == run.ch => {
                Lead::Run(Fence { len: run.len + 1, ..run })
            }
            Lead::Run(run) if run.len >= 3 => {
                Lead::After { run, blank: c.is_whitespace(), tick: c == '`' }
            }
            Lead::Run(_) | Lead::Text => Lead::Text,
            Lead::After { run, blank, tick } => Lead::After {
                run,
                blank: blank && c.is_whitespace(),
                tick: tick || c == '`',
            },
        }
    }

    /// The fence this line opens, if it ends here.
    fn opens(self) -> Option<Fence> {
        match self {
            Lead::Run(run) if run.len >= 3 => Some(run),
            Lead::After { run, tick, .. } if !(run.ch == '`' && tick) => Some(run),
            _ => None,
        }
    }

    /// Whether this line closes `open`, if it ends here.
    fn closes(self, open: Fence) -> bool {
        match self {
            Lead::Run(run) | Lead::After { run, blank: true, .. } => {
                run.ch == open.ch && run.len >= open.len
            }
            _ => false,
        }
    }
}

/// Where a reasoning text stands, character by character: line ends, the
/// last character, CommonMark code fences (see the module docs), and the
/// tool calls the block kept. A `<tool_call>` that starts a line outside a
/// fence opens a call (until `</tool_call>`, inside which fences are not
/// tracked: an argument may hold any text); one anywhere else is text. The
/// thinking control and the output parser's `<tool_call>` rule
/// (`serve::stream`) both read the reasoning with it, so they agree.
#[derive(Debug, Clone)]
pub struct ReasoningScan {
    /// `\n` characters the text ends with.
    trailing_newlines: usize,
    last: Option<char>,
    /// Characters of the current line so far.
    line_len: usize,
    lead: Lead,
    fence: Option<Fence>,
    in_call: bool,
    /// The last characters, to find the tags across pieces.
    tail: String,
}

impl Default for ReasoningScan {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningScan {
    /// At the start of a block: the prompt's `<think>\n` ended a line.
    pub fn new() -> Self {
        Self {
            trailing_newlines: 1,
            last: None,
            line_len: 0,
            lead: Lead::Indent(0),
            fence: None,
            in_call: false,
            tail: String::new(),
        }
    }

    pub fn feed(&mut self, text: &str) {
        for c in text.chars() {
            self.push(c);
        }
    }

    fn push(&mut self, c: char) {
        self.last = Some(c);
        self.tail.push(c);
        if self.tail.len() > TOOL_END.len() {
            let cut = self.tail.len() - TOOL_END.len();
            let cut = (cut..self.tail.len())
                .find(|&i| self.tail.is_char_boundary(i))
                .unwrap_or(self.tail.len());
            self.tail.drain(..cut);
        }
        if c == '\n' {
            self.trailing_newlines += 1;
            if !self.in_call {
                match self.fence {
                    Some(open) if self.lead.closes(open) => self.fence = None,
                    Some(_) => {}
                    None => self.fence = self.lead.opens(),
                }
            }
            self.lead = Lead::Indent(0);
            self.line_len = 0;
            return;
        }
        self.trailing_newlines = 0;
        self.line_len += 1;
        if self.in_call {
            if self.tail.ends_with(TOOL_END) {
                self.in_call = false;
            }
            return;
        }
        if self.fence.is_none()
            && self.line_len == TOOL_START.len()
            && self.tail.ends_with(TOOL_START)
        {
            self.in_call = true;
            self.lead = Lead::Text;
            return;
        }
        self.lead = self.lead.next(c);
    }

    /// `\n` characters the text ends with.
    pub fn trailing_newlines(&self) -> usize {
        self.trailing_newlines
    }

    /// The last character, if any.
    pub fn last_char(&self) -> Option<char> {
        self.last
    }

    /// Inside a tool call the block kept.
    pub fn in_call(&self) -> bool {
        self.in_call
    }

    /// Whether text inserted here, starting with a line end, would be
    /// inside a code fence: an open fence unless the current line so far
    /// would close it, or no fence but the current line so far would open
    /// one.
    pub fn in_fence(&self) -> bool {
        match self.fence {
            Some(open) => !self.lead.closes(open),
            None => !self.in_call && self.lead.opens().is_some(),
        }
    }

    /// Whether a `<tool_call>` arriving here is a call: at the start of a
    /// line, outside a code fence and outside a call the block kept.
    pub fn call_may_start(&self) -> bool {
        self.trailing_newlines > 0 && self.fence.is_none() && !self.in_call
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
