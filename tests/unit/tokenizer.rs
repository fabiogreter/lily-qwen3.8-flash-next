use super::*;
use crate::chat::Message;

fn conversation() -> Conversation {
    vec![
        Message::new_user("first"),
        Message::new_assistant("older reply"),
        Message::new_user("second"),
        Message::new_assistant("recent reply"),
    ]
}

/// The template wraps assistant turns after the last user message; this
/// index is what tells the nothink rewrite which turns it must cover.
#[test]
fn last_query_index_finds_the_last_real_user_turn() {
    assert_eq!(last_query_index(&conversation()), Some(2));
}

#[test]
fn a_conversation_with_no_user_turn_has_no_query_index() {
    assert_eq!(last_query_index(&vec![Message::new_system("s")]), None);
}

/// Renders `template` with `x` bound to `value` and lily's `tojson`.
fn tojson(template: &str, value: serde_json::Value) -> String {
    let mut env = Environment::new();
    env.add_filter("tojson", py_tojson);
    env.render_str(template, minijinja::context! { x => Value::from_serialize(value) })
        .unwrap()
}

// The expected strings below are Python 3's `json.dumps` output for the same
// values, which is what transformers' `tojson` filter returns.

#[test]
fn tojson_matches_json_dumps_on_a_tool_schema() {
    let tool = serde_json::json!({"type": "function", "function": {
        "name": "grep", "description": "Search <files> & 'text'",
        "parameters": {"type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"]}}});
    assert_eq!(
        tojson("{{ x | tojson }}", tool),
        "{\"type\": \"function\", \"function\": {\"name\": \"grep\", \"description\": \
         \"Search <files> & 'text'\", \"parameters\": {\"type\": \"object\", \"properties\": \
         {\"pattern\": {\"type\": \"string\"}, \"path\": {\"type\": \"string\"}}, \
         \"required\": [\"pattern\"]}}}"
    );
}

#[test]
fn tojson_escapes_strings_like_json_dumps() {
    let s = "caf\u{e9} \u{1F600} \u{2028} tab\t nl\n cr\r bs\u{8} ff\u{c} ctl\u{1} del\u{7f} quote\" back\\";
    assert_eq!(
        tojson("{{ x | tojson }}", serde_json::json!({"s": s})),
        "{\"s\": \"caf\u{e9} \u{1F600} \u{2028} tab\\t nl\\n cr\\r bs\\b ff\\f ctl\\u0001 del\u{7f} quote\\\" back\\\\\"}"
    );
    assert_eq!(
        tojson(
            "{{ x | tojson(ensure_ascii=true) }}",
            serde_json::json!({"s": "caf\u{e9} \u{1F600} del\u{7f}"})
        ),
        "{\"s\": \"caf\\u00e9 \\ud83d\\ude00 del\\u007f\"}"
    );
}

#[test]
fn tojson_prints_numbers_like_python() {
    let values = serde_json::json!([
        0.1,
        1e-05,
        1e16,
        1.5e16,
        100.0,
        -0.0,
        0.0,
        123.456,
        1e15,
        0.00025,
        12345678901234567890u64,
        -7,
        true,
        false,
        null,
        [],
        {}
    ]);
    assert_eq!(
        tojson("{{ x | tojson }}", values),
        "[0.1, 1e-05, 1e+16, 1.5e+16, 100.0, -0.0, 0.0, 123.456, 1000000000000000.0, 0.00025, \
         12345678901234567890, -7, true, false, null, [], {}]"
    );
}

#[test]
fn tojson_takes_the_json_dumps_options() {
    assert_eq!(
        tojson(
            "{{ x | tojson(indent=2) }}",
            serde_json::json!({"b": [1, {"c": []}], "a": {}})
        ),
        "{\n  \"b\": [\n    1,\n    {\n      \"c\": []\n    }\n  ],\n  \"a\": {}\n}"
    );
    assert_eq!(
        tojson(
            "{{ x | tojson(sort_keys=true, separators=[',', ':']) }}",
            serde_json::json!({"b": 1, "a": 2})
        ),
        "{\"a\":2,\"b\":1}"
    );
}
