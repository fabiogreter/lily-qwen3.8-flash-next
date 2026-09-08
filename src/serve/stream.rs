//! Turns drawn token ids into API events: incremental detokenization that
//! respects UTF-8 boundaries, the `<think>` reasoning split, `<tool_call>`
//! blocks, and client stop strings.

use anyhow::Result;

use super::tools::{ParsedToolCall, ToolSchema, parse_tool_call};

const THINK_END: &str = "</think>";
const TOOL_START: &str = "<tool_call>";
const TOOL_END: &str = "</tool_call>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Reasoning(String),
    Content(String),
    ToolCall(ParsedToolCall),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Reasoning,
    Content,
    /// Inside `<tool_call>`, collecting the block.
    ToolBlock,
}

/// How the parser starts, decided by the prompt's shape.
pub struct ParserConfig {
    /// The prompt ended with an open `<think>` tag.
    pub thinking_open: bool,
    /// Tools offered in the request; `None` disables tool-call parsing.
    pub tools: Option<Vec<ToolSchema>>,
    pub stop_strings: Vec<String>,
    /// Raw text completions: no reasoning or tool parsing at all.
    pub raw: bool,
}

pub struct OutputParser<D: FnMut(&[u32]) -> Result<String>> {
    detokenize: D,
    /// Tokens whose text has not been released because it ends mid-codepoint.
    pending: Vec<u32>,
    phase: Phase,
    /// Text of the current phase not yet emitted.
    buf: String,
    tools: Option<Vec<ToolSchema>>,
    stop_strings: Vec<String>,
    raw: bool,
    /// Set once a stop string matched; further tokens are ignored.
    pub stopped: bool,
    /// Whether any content or reasoning has been emitted (to trim the
    /// leading whitespace of a phase exactly once).
    phase_started: bool,
    tool_calls_emitted: usize,
}

impl<D: FnMut(&[u32]) -> Result<String>> OutputParser<D> {
    pub fn new(detokenize: D, config: ParserConfig) -> Self {
        let phase = if config.thinking_open && !config.raw { Phase::Reasoning } else { Phase::Content };
        Self {
            detokenize,
            pending: Vec::new(),
            phase,
            buf: String::new(),
            tools: config.tools,
            stop_strings: config.stop_strings,
            raw: config.raw,
            stopped: false,
            phase_started: false,
            tool_calls_emitted: 0,
        }
    }

    pub fn tool_calls_emitted(&self) -> usize {
        self.tool_calls_emitted
    }

    /// Feeds one drawn token (never a stop token) and returns the events it
    /// completes.
    pub fn push(&mut self, token: u32) -> Result<Vec<Event>> {
        if self.stopped {
            return Ok(Vec::new());
        }
        self.pending.push(token);
        let text = (self.detokenize)(&self.pending)?;
        // A trailing replacement character means the byte sequence is cut
        // mid-codepoint; wait for the next token (bounded so garbage cannot
        // stall the stream forever).
        if text.ends_with('\u{FFFD}') && self.pending.len() < 4 {
            return Ok(Vec::new());
        }
        self.pending.clear();
        Ok(self.feed_text(&text, false))
    }

    /// Releases everything still held back.
    pub fn finish(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        if !self.pending.is_empty() {
            if let Ok(text) = (self.detokenize)(&self.pending) {
                let text = text.trim_end_matches('\u{FFFD}').to_string();
                events.extend(self.feed_text(&text, true));
            }
            self.pending.clear();
        }
        events.extend(self.feed_text("", true));
        events
    }

    fn feed_text(&mut self, text: &str, final_flush: bool) -> Vec<Event> {
        let mut events = Vec::new();
        self.buf.push_str(text);
        loop {
            match self.phase {
                Phase::Reasoning => {
                    if let Some(idx) = self.buf.find(THINK_END) {
                        let reasoning = self.buf[..idx].to_string();
                        let rest = self.buf[idx + THINK_END.len()..].to_string();
                        self.emit_text(&mut events, reasoning.trim_end_matches('\n'), Phase::Reasoning);
                        self.phase = Phase::Content;
                        self.phase_started = false;
                        self.buf = rest;
                        continue;
                    }
                    // Hold back a partial `</think>` and the newlines that
                    // precede it in the model's `\n</think>` habit.
                    let hold = if final_flush {
                        0
                    } else {
                        partial_suffix_len(&self.buf, THINK_END).max(trailing_newlines(&self.buf))
                    };
                    let release = self.buf.len() - hold;
                    let out: String = self.buf.drain(..release).collect();
                    self.emit_text(&mut events, &out, Phase::Reasoning);
                    if final_flush && !self.buf.is_empty() {
                        let rest = std::mem::take(&mut self.buf);
                        self.emit_text(&mut events, &rest, Phase::Reasoning);
                    }
                    return events;
                }
                Phase::Content => {
                    let tools_on = self.tools.is_some() && !self.raw;
                    let tool_idx = if tools_on { self.buf.find(TOOL_START) } else { None };
                    let stop_idx = self.stop_strings.iter().filter_map(|s| self.buf.find(s.as_str())).min();
                    if let Some(stop) = stop_idx
                        && tool_idx.is_none_or(|t| stop <= t)
                    {
                        let before = self.buf[..stop].to_string();
                        self.emit_text(&mut events, &before, Phase::Content);
                        self.stopped = true;
                        self.buf.clear();
                        return events;
                    }
                    if let Some(idx) = tool_idx {
                        let before = self.buf[..idx].trim_end().to_string();
                        let rest = self.buf[idx + TOOL_START.len()..].to_string();
                        self.emit_text(&mut events, &before, Phase::Content);
                        self.phase = Phase::ToolBlock;
                        self.buf = rest;
                        continue;
                    }
                    let mut hold = 0;
                    if !final_flush {
                        if tools_on {
                            // A `<tool_call>` may still arrive, split across
                            // tokens or after the whitespace the template puts
                            // between content and call.
                            hold = partial_suffix_len(&self.buf, TOOL_START).max(trailing_whitespace(&self.buf));
                        }
                        let stop_hold = self
                            .stop_strings
                            .iter()
                            .map(|s| s.len().saturating_sub(1))
                            .max()
                            .unwrap_or(0);
                        hold = hold.max(stop_hold.min(self.buf.len()));
                    }
                    let release = floor_char_boundary(&self.buf, self.buf.len() - hold);
                    let out: String = self.buf.drain(..release).collect();
                    self.emit_text(&mut events, &out, Phase::Content);
                    return events;
                }
                Phase::ToolBlock => {
                    let Some(idx) = self.buf.find(TOOL_END) else {
                        if final_flush {
                            // Unterminated block: give the client the raw text.
                            let raw = format!("{TOOL_START}{}", std::mem::take(&mut self.buf));
                            self.phase = Phase::Content;
                            self.emit_text(&mut events, &raw, Phase::Content);
                        }
                        return events;
                    };
                    let block = self.buf[..idx].to_string();
                    let rest = self.buf[idx + TOOL_END.len()..].to_string();
                    self.buf = rest;
                    self.phase = Phase::Content;
                    let tools = self.tools.as_deref().unwrap_or(&[]);
                    match parse_tool_call(&block, tools) {
                        Ok(call) => {
                            self.tool_calls_emitted += 1;
                            events.push(Event::ToolCall(call));
                        }
                        Err(_) => {
                            let raw = format!("{TOOL_START}{block}{TOOL_END}");
                            self.emit_text(&mut events, &raw, Phase::Content);
                        }
                    }
                    continue;
                }
            }
        }
    }

    fn emit_text(&mut self, events: &mut Vec<Event>, text: &str, phase: Phase) {
        let text = if self.phase_started { text } else { text.trim_start_matches('\n') };
        if text.is_empty() {
            return;
        }
        self.phase_started = true;
        match phase {
            Phase::Reasoning => events.push(Event::Reasoning(text.to_string())),
            _ => events.push(Event::Content(text.to_string())),
        }
    }
}

/// Length of the longest suffix of `buf` that is a proper prefix of `marker`.
fn partial_suffix_len(buf: &str, marker: &str) -> usize {
    let max = marker.len().saturating_sub(1).min(buf.len());
    (1..=max)
        .rev()
        .find(|&n| buf.is_char_boundary(buf.len() - n) && marker.starts_with(&buf[buf.len() - n..]))
        .unwrap_or(0)
}

fn trailing_newlines(buf: &str) -> usize {
    buf.len() - buf.trim_end_matches('\n').len()
}

fn trailing_whitespace(buf: &str) -> usize {
    buf.len() - buf.trim_end().len()
}

fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

#[cfg(test)]
#[path = "../../tests/unit/serve/stream.rs"]
mod tests;
