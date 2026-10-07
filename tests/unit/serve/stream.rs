use super::*;

/// A toy detokenizer: ids index a table of strings; ids >= 1000 are raw
/// bytes (id - 1000) so multi-byte code points can be split across tokens.
fn detok(table: &'static [&'static str]) -> impl FnMut(&[u32]) -> Result<String> {
    move |ids: &[u32]| {
        let mut bytes = Vec::new();
        for &id in ids {
            if id >= 1000 {
                bytes.push((id - 1000) as u8);
            } else {
                bytes.extend_from_slice(table[id as usize].as_bytes());
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

fn run(
    parser: &mut OutputParser<impl FnMut(&[u32]) -> Result<String>>,
    ids: &[u32],
) -> Vec<Event> {
    let mut events = Vec::new();
    for &id in ids {
        events.extend(parser.push(id).expect("push"));
        if parser.stopped {
            break;
        }
    }
    events.extend(parser.finish());
    events
}

fn text_of(events: &[Event]) -> (String, String, Vec<ParsedToolCall>) {
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut calls = Vec::new();
    for e in events {
        match e {
            Event::Reasoning(t) => reasoning.push_str(t),
            Event::Content(t) => content.push_str(t),
            Event::ToolCall(c) => calls.push(c.clone()),
        }
    }
    (reasoning, content, calls)
}

const TABLE: &[&str] = &[
    "Hello",
    " world",
    "\n",
    "</think>",
    "\n\n",
    "<tool_call>",
    "</tool_call>",
    "<function=f>",
    "<parameter=x>",
    "</parameter>",
    "</function>",
    "<",
    "/think",
    ">",
    "STOP",
    "ab",
    "The answer",
    "<tool",
    "_call>",
    "```",
    "````",
    "~~~",
    "Act now.",
    " in",
];

#[test]
fn reasoning_then_content_split_and_trimmed() {
    let mut p = OutputParser::new(
        detok(TABLE),
        ParserConfig {
            thinking_open: true,
            tools: None,
            stop_strings: vec![],
            raw: false,
            tool_call_ends_thinking: false,
        },
    );
    // "Hello world\n</think>\n\nThe answer"
    let events = run(&mut p, &[0, 1, 2, 3, 4, 16]);
    let (reasoning, content, calls) = text_of(&events);
    assert_eq!(reasoning, "Hello world");
    assert_eq!(content, "The answer");
    assert!(calls.is_empty());
}

#[test]
fn think_end_split_across_tokens_is_still_found() {
    let mut p = OutputParser::new(
        detok(TABLE),
        ParserConfig {
            thinking_open: true,
            tools: None,
            stop_strings: vec![],
            raw: false,
            tool_call_ends_thinking: false,
        },
    );
    // "Hello" "<" "/think" ">" " world"
    let events = run(&mut p, &[0, 11, 12, 13, 1]);
    let (reasoning, content, _) = text_of(&events);
    assert_eq!(reasoning, "Hello");
    assert_eq!(content, " world");
    // The partial "<" must not have been emitted as reasoning early.
    assert_eq!(events[0], Event::Reasoning("Hello".into()));
}

#[test]
fn tool_call_block_becomes_a_call_and_surrounding_text_is_content() {
    let tools = ToolSchema::from_request(&[
        serde_json::json!({"type": "function", "function": {
        "name": "f", "parameters": {"properties": {"x": {"type": "integer"}}}}}),
    ])
    .unwrap();
    let mut p = OutputParser::new(
        detok(TABLE),
        ParserConfig {
            thinking_open: false,
            tools: Some(tools),
            stop_strings: vec![],
            raw: false,
            tool_call_ends_thinking: false,
        },
    );
    // "The answer\n\n<tool_call>\n<function=f>\n<parameter=x>\nab\n</parameter>\n</function>\n</tool_call>"
    let ids = [16, 4, 5, 2, 7, 2, 8, 2, 15, 2, 9, 2, 10, 2, 6];
    let events = run(&mut p, &ids);
    let (_, content, calls) = text_of(&events);
    assert_eq!(content, "The answer");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "f");
    assert_eq!(calls[0].arguments, r#"{"x":"ab"}"#);
    assert_eq!(p.tool_calls_emitted(), 1);
}

#[test]
fn split_tool_start_marker_is_held_back_then_recognised() {
    let tools = ToolSchema::from_request(&[
        serde_json::json!({"type": "function", "function": {"name": "f"}}),
    ])
    .unwrap();
    let mut p = OutputParser::new(
        detok(TABLE),
        ParserConfig {
            thinking_open: false,
            tools: Some(tools),
            stop_strings: vec![],
            raw: false,
            tool_call_ends_thinking: false,
        },
    );
    // "Hello" "<tool" "_call>" "<function=f>" "</function>" "</tool_call>"
    let events = run(&mut p, &[0, 17, 18, 7, 10, 6]);
    let (_, content, calls) = text_of(&events);
    assert_eq!(content, "Hello");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments, "{}");
}

#[test]
fn unterminated_tool_block_is_returned_as_content() {
    let tools = ToolSchema::from_request(&[
        serde_json::json!({"type": "function", "function": {"name": "f"}}),
    ])
    .unwrap();
    let mut p = OutputParser::new(
        detok(TABLE),
        ParserConfig {
            thinking_open: false,
            tools: Some(tools),
            stop_strings: vec![],
            raw: false,
            tool_call_ends_thinking: false,
        },
    );
    let events = run(&mut p, &[5, 2, 7]);
    let (_, content, calls) = text_of(&events);
    assert_eq!(content, "<tool_call>\n<function=f>");
    assert!(calls.is_empty());
}

#[test]
fn stop_strings_truncate_and_stop() {
    let mut p = OutputParser::new(
        detok(TABLE),
        ParserConfig {
            thinking_open: false,
            tools: None,
            stop_strings: vec!["STOP".into()],
            raw: false,
            tool_call_ends_thinking: false,
        },
    );
    let events = run(&mut p, &[0, 1, 14, 0, 0]);
    let (_, content, _) = text_of(&events);
    assert_eq!(content, "Hello world");
    assert!(p.stopped);
    // Streaming holds back stop-length text: nothing after "Hello world" leaks.
    assert!(
        events.iter().all(|e| !matches!(e, Event::Content(t) if t.contains("STOP")))
    );
}

#[test]
fn multibyte_codepoints_split_over_tokens_are_not_emitted_partially() {
    // "é" is 0xC3 0xA9.
    let mut p = OutputParser::new(
        detok(TABLE),
        ParserConfig {
            thinking_open: false,
            tools: None,
            stop_strings: vec![],
            raw: true,
            tool_call_ends_thinking: false,
        },
    );
    let events = run(&mut p, &[0, 1000 + 0xC3, 1000 + 0xA9, 1]);
    let (_, content, _) = text_of(&events);
    assert_eq!(content, "Helloé world");
    assert!(
        events
            .iter()
            .all(|e| !matches!(e, Event::Content(t) if t.contains('\u{FFFD}')))
    );
}

#[test]
fn raw_mode_ignores_markers() {
    let mut p = OutputParser::new(
        detok(TABLE),
        ParserConfig {
            thinking_open: true,
            tools: None,
            stop_strings: vec![],
            raw: true,
            tool_call_ends_thinking: false,
        },
    );
    let events = run(&mut p, &[0, 3, 5]);
    let (reasoning, content, _) = text_of(&events);
    assert_eq!(reasoning, "");
    assert_eq!(content, "Hello</think><tool_call>");
}

#[test]
fn reasoning_tokens_count_the_block_and_its_closing_tag() {
    let config = |thinking_open| ParserConfig {
        thinking_open,
        tools: None,
        stop_strings: vec![],
        raw: false,
        tool_call_ends_thinking: false,
    };
    // "Hello world\n</think>\n\nThe answer": four tokens up to and
    // including `</think>`, two after it.
    let mut p = OutputParser::new(detok(TABLE), config(true));
    run(&mut p, &[0, 1, 2, 3, 4, 16]);
    assert_eq!(p.reasoning_tokens(), 4);
    // A `</think>` split over three tokens counts all three.
    let mut p = OutputParser::new(detok(TABLE), config(true));
    run(&mut p, &[0, 11, 12, 13, 1]);
    assert_eq!(p.reasoning_tokens(), 4);
    // Cut off before the block closed: everything was reasoning.
    let mut p = OutputParser::new(detok(TABLE), config(true));
    run(&mut p, &[0, 1, 0]);
    assert_eq!(p.reasoning_tokens(), 3);
    // No open block (thinking off): none.
    let mut p = OutputParser::new(detok(TABLE), config(false));
    run(&mut p, &[0, 3, 16]);
    assert_eq!(p.reasoning_tokens(), 0);
}

fn reasoning_with_tools(tool_call_ends_thinking: bool) -> ParserConfig {
    ParserConfig {
        thinking_open: true,
        tools: Some(
            ToolSchema::from_request(&[
                serde_json::json!({"type": "function", "function": {"name": "f"}}),
            ])
            .unwrap(),
        ),
        stop_strings: vec![],
        raw: false,
        tool_call_ends_thinking,
    }
}

/// The response-side safety net: with the rule on, a `<tool_call>` at a
/// line start inside the reasoning block ends it and is parsed as a call,
/// the way vLLM's Qwen3 reasoning parser treats it.
#[test]
fn a_tool_call_at_a_line_start_ends_the_reasoning_with_the_rule_on() {
    // "Hello world\n<tool_call>\n<function=f>\n</function>\n</tool_call>"
    let ids = [0, 1, 2, 5, 2, 7, 2, 10, 2, 6];
    let mut p = OutputParser::new(detok(TABLE), reasoning_with_tools(true));
    let (reasoning, content, calls) = text_of(&run(&mut p, &ids));
    assert_eq!(reasoning, "Hello world");
    assert_eq!(content, "");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "f");
    // The block's tokens, the `<tool_call>` that closed it included.
    assert_eq!(p.reasoning_tokens(), 4);

    // Split across tokens ("<tool" "_call>") and right at the block's start.
    let mut p = OutputParser::new(detok(TABLE), reasoning_with_tools(true));
    let (reasoning, _, calls) = text_of(&run(&mut p, &[17, 18, 7, 10, 6]));
    assert_eq!(reasoning, "");
    assert_eq!(calls.len(), 1);

    // After the decode-time close the stream is the trained shape and the
    // ordinary `</think>` path takes it.
    let mut p = OutputParser::new(detok(TABLE), reasoning_with_tools(true));
    let (reasoning, content, calls) = text_of(&run(&mut p, &[0, 2, 3, 4, 5, 7, 10, 6]));
    assert_eq!((reasoning.as_str(), content.as_str(), calls.len()), ("Hello", "", 1));
}

#[test]
fn a_tool_call_stays_reasoning_mid_line_or_with_the_rule_off() {
    // Off (the default): reasoning text, byte for byte as before.
    let ids = [0, 1, 2, 5, 2, 7, 2, 10, 2, 6];
    let mut p = OutputParser::new(detok(TABLE), reasoning_with_tools(false));
    let (reasoning, content, calls) = text_of(&run(&mut p, &ids));
    assert_eq!(
        reasoning,
        "Hello world\n<tool_call>\n<function=f>\n</function>\n</tool_call>"
    );
    assert_eq!((content.as_str(), calls.len()), ("", 0));

    // On, but mid-line: "Hello<tool_call>..." stays reasoning.
    let mut p = OutputParser::new(detok(TABLE), reasoning_with_tools(true));
    let (reasoning, _, calls) = text_of(&run(&mut p, &[0, 5, 7, 10, 6]));
    assert!(reasoning.starts_with("Hello<tool_call>"), "{reasoning:?}");
    assert_eq!(calls.len(), 0);
}

#[test]
fn a_tool_call_quoted_in_a_fence_or_nested_in_a_kept_call_stays_reasoning() {
    // "```\n<tool_call>\n<function=f>\n</function>\n</tool_call>\n```\n"
    // then a real one at a line start outside the fence.
    let quoted = [19, 2, 5, 2, 7, 2, 10, 2, 6, 2, 19, 2];
    let real = [5, 7, 10, 6];
    let ids: Vec<u32> = quoted.iter().chain(&real).copied().collect();
    let mut p = OutputParser::new(detok(TABLE), reasoning_with_tools(true));
    let (reasoning, _, calls) = text_of(&run(&mut p, &ids));
    assert_eq!(
        reasoning, "```\n<tool_call>\n<function=f>\n</function>\n</tool_call>\n```",
        "the fenced example is reasoning"
    );
    assert_eq!(calls.len(), 1, "the call after the fence ends the block");

    // A mention mid-line opens no call: the call on the next line ends
    // the block. "Hello<tool_call>\n<tool_call><function=f></function></tool_call>"
    let ids = [0, 5, 2, 5, 7, 10, 6];
    let mut p = OutputParser::new(detok(TABLE), reasoning_with_tools(true));
    let (reasoning, _, calls) = text_of(&run(&mut p, &ids));
    assert_eq!(reasoning, "Hello<tool_call>");
    assert_eq!(calls.len(), 1, "the call on its own line closes");
}

/// The fences of `fenced`, an unmatched `<tool_call>` line inside them,
/// then `closing` and a real call: the decode-time control's own fence
/// rules ([`crate::thinking::ReasoningScan`]).
fn fenced_then_real(fenced: &[u32], closing: &[u32]) {
    let ids: Vec<u32> = fenced
        .iter()
        .chain(&[5, 2])
        .chain(closing)
        .chain(&[5, 7, 10, 6])
        .copied()
        .collect();
    let mut p = OutputParser::new(detok(TABLE), reasoning_with_tools(true));
    let (reasoning, _, calls) = text_of(&run(&mut p, &ids));
    assert_eq!(calls.len(), 1, "{fenced:?}: the call after the fence: {reasoning:?}");
    assert!(!reasoning.contains("<function=f>"), "{fenced:?}: {reasoning:?}");
}

#[test]
fn fences_follow_commonmark_for_the_tool_call_rule() {
    // "```\n" ... "```\n"
    fenced_then_real(&[19, 2], &[19, 2]);
    // A four-backtick fence holding a three-backtick line: "````\n```\n"
    // ... "````\n".
    fenced_then_real(&[20, 2, 19, 2], &[20, 2]);
    // Tildes, with backticks inside: "~~~\n```\n" ... "~~~\n".
    fenced_then_real(&[21, 2, 19, 2], &[21, 2]);
    // A three-backtick line does not close the four-backtick fence: the
    // call after it is still quoted.
    let ids = [20, 2, 19, 2, 5, 7, 10, 6];
    let mut p = OutputParser::new(detok(TABLE), reasoning_with_tools(true));
    let (reasoning, _, calls) = text_of(&run(&mut p, &ids));
    assert_eq!(calls.len(), 0, "{reasoning:?}");
}

/// After a budget's close the content is parsed as usual: a stop string
/// there ends it. "Hello\nAct now.\n</think>\n\nThe answer inSTOP world"
#[test]
fn a_stop_string_after_a_budget_close_ends_the_content() {
    let config = ParserConfig {
        thinking_open: true,
        tools: None,
        stop_strings: vec!["STOP".into()],
        raw: false,
        tool_call_ends_thinking: false,
    };
    let mut p = OutputParser::new(detok(TABLE), config);
    let events = run(&mut p, &[0, 2, 22, 2, 3, 4, 16, 23, 14, 1]);
    let (reasoning, content, _) = text_of(&events);
    assert_eq!(reasoning, "Hello\nAct now.");
    assert_eq!(content, "The answer in");
    assert!(p.stopped);
}
