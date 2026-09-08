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

fn run(parser: &mut OutputParser<impl FnMut(&[u32]) -> Result<String>>, ids: &[u32]) -> Vec<Event> {
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
    "Hello", " world", "\n", "</think>", "\n\n", "<tool_call>", "</tool_call>", "<function=f>",
    "<parameter=x>", "</parameter>", "</function>", "<", "/think", ">", "STOP", "ab", "The answer",
    "<tool", "_call>",
];

#[test]
fn reasoning_then_content_split_and_trimmed() {
    let mut p = OutputParser::new(
        detok(TABLE),
        ParserConfig { thinking_open: true, tools: None, stop_strings: vec![], raw: false },
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
        ParserConfig { thinking_open: true, tools: None, stop_strings: vec![], raw: false },
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
    let tools = ToolSchema::from_request(&[serde_json::json!({"type": "function", "function": {
        "name": "f", "parameters": {"properties": {"x": {"type": "integer"}}}}})])
    .unwrap();
    let mut p = OutputParser::new(
        detok(TABLE),
        ParserConfig { thinking_open: false, tools: Some(tools), stop_strings: vec![], raw: false },
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
    let tools = ToolSchema::from_request(&[serde_json::json!({"type": "function", "function": {"name": "f"}})]).unwrap();
    let mut p = OutputParser::new(
        detok(TABLE),
        ParserConfig { thinking_open: false, tools: Some(tools), stop_strings: vec![], raw: false },
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
    let tools = ToolSchema::from_request(&[serde_json::json!({"type": "function", "function": {"name": "f"}})]).unwrap();
    let mut p = OutputParser::new(
        detok(TABLE),
        ParserConfig { thinking_open: false, tools: Some(tools), stop_strings: vec![], raw: false },
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
        ParserConfig { thinking_open: false, tools: None, stop_strings: vec!["STOP".into()], raw: false },
    );
    let events = run(&mut p, &[0, 1, 14, 0, 0]);
    let (_, content, _) = text_of(&events);
    assert_eq!(content, "Hello world");
    assert!(p.stopped);
    // Streaming holds back stop-length text: nothing after "Hello world" leaks.
    assert!(events.iter().all(|e| !matches!(e, Event::Content(t) if t.contains("STOP"))));
}

#[test]
fn multibyte_codepoints_split_over_tokens_are_not_emitted_partially() {
    // "é" is 0xC3 0xA9.
    let mut p = OutputParser::new(
        detok(TABLE),
        ParserConfig { thinking_open: false, tools: None, stop_strings: vec![], raw: true },
    );
    let events = run(&mut p, &[0, 1000 + 0xC3, 1000 + 0xA9, 1]);
    let (_, content, _) = text_of(&events);
    assert_eq!(content, "Helloé world");
    assert!(events.iter().all(|e| !matches!(e, Event::Content(t) if t.contains('\u{FFFD}'))));
}

#[test]
fn raw_mode_ignores_markers() {
    let mut p = OutputParser::new(
        detok(TABLE),
        ParserConfig { thinking_open: true, tools: None, stop_strings: vec![], raw: true },
    );
    let events = run(&mut p, &[0, 3, 5]);
    let (reasoning, content, _) = text_of(&events);
    assert_eq!(reasoning, "");
    assert_eq!(content, "Hello</think><tool_call>");
}
