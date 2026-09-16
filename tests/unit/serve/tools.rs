use super::*;

fn schemas() -> Vec<ToolSchema> {
    ToolSchema::from_request(&[
        serde_json::json!({"type": "function", "function": {
            "name": "read_file",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "follow": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "mode": {"enum": ["a", "b"]},
                "maybe": {"type": ["string", "null"]}
            }}
        }}),
        serde_json::json!({"type": "function", "function": {"name": "noargs"}}),
    ])
    .expect("schemas")
}

#[test]
fn parses_typed_parameters_from_the_xml_block() {
    let block = "\n<function=read_file>\n<parameter=path>\nsrc/main.rs\n</parameter>\n\
                 <parameter=offset>\n42\n</parameter>\n<parameter=follow>\ntrue\n</parameter>\n\
                 <parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n<parameter=mode>\nb\n</parameter>\n\
                 <parameter=maybe>\n123\n</parameter>\n<parameter=extra>\n7\n</parameter>\n</function>\n";
    let call = parse_tool_call(block, &schemas()).expect("parse");
    assert_eq!(call.name, "read_file");
    let args: serde_json::Value = serde_json::from_str(&call.arguments).unwrap();
    assert_eq!(args["path"], "src/main.rs");
    assert_eq!(args["offset"], 42);
    assert_eq!(args["follow"], true);
    assert_eq!(args["tags"], serde_json::json!(["a", "b"]));
    assert_eq!(args["mode"], "b");
    assert_eq!(args["maybe"], "123", "nullable strings stay strings");
    assert_eq!(args["extra"], "7", "unknown parameters stay strings");
}

#[test]
fn multiline_string_values_keep_inner_newlines() {
    let block = "<function=read_file>\n<parameter=path>\nline one\n\nline three\n</parameter>\n</function>";
    let call = parse_tool_call(block, &schemas()).expect("parse");
    let args: serde_json::Value = serde_json::from_str(&call.arguments).unwrap();
    assert_eq!(args["path"], "line one\n\nline three");
}

#[test]
fn unknown_functions_and_bad_json_fall_back_to_strings() {
    let block =
        "<function=mystery>\n<parameter=x>\n{not json\n</parameter>\n</function>";
    let call = parse_tool_call(block, &schemas()).expect("parse");
    assert_eq!(call.name, "mystery");
    assert_eq!(call.arguments, r#"{"x":"{not json"}"#);
    let call =
        parse_tool_call("<function=noargs>\n</function>", &schemas()).expect("parse");
    assert_eq!(call.arguments, "{}");
    let block = "<function=read_file>\n<parameter=offset>\nnot a number\n</parameter>\n</function>";
    let call = parse_tool_call(block, &schemas()).expect("parse");
    assert_eq!(call.arguments, r#"{"offset":"not a number"}"#);
}

#[test]
fn malformed_blocks_are_errors() {
    assert!(parse_tool_call("<functio=read_file></function>", &schemas()).is_err());
    assert!(
        parse_tool_call("<function=read_file>\n<parameter=path>\nx\n", &schemas())
            .is_err()
    );
}

#[test]
fn incoming_tool_calls_are_shaped_for_the_template() {
    let calls = template_tool_calls(&[
        serde_json::json!({"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}),
        serde_json::json!({"function": {"name": "g", "arguments": ""}}),
        serde_json::json!({"function": {"name": "h", "arguments": {"z": true}}}),
    ])
    .expect("shape");
    assert_eq!(calls[0]["function"]["arguments"]["a"], 1);
    assert_eq!(calls[1]["function"]["arguments"], serde_json::json!({}));
    assert_eq!(calls[2]["function"]["arguments"]["z"], true);
    assert!(
        template_tool_calls(&[
            serde_json::json!({"function": {"name": "f", "arguments": "nope"}})
        ])
        .is_err()
    );
    assert!(
        template_tool_calls(&[
            serde_json::json!({"function": {"name": "f", "arguments": "[1]"}})
        ])
        .is_err()
    );
}
