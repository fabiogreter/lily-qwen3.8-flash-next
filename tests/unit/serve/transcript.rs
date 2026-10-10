use super::*;

use crate::serve::api::{Defaults, ImagePolicy, prepare_chat};

const START: u32 = 1;
const END: u32 = 2;
const ASSISTANT: u32 = 10;
const NL: u32 = 11;
const USER: u32 = 12;

fn markers() -> TurnMarkers {
    TurnMarkers { im_start: START, im_end: END, assistant: vec![ASSISTANT, NL] }
}

/// `<|im_start|>assistant\n` + `body` + `<|im_end|>\n`.
fn assistant_turn(body: &[u32]) -> Vec<u32> {
    let mut out = vec![START, ASSISTANT, NL];
    out.extend_from_slice(body);
    out.extend_from_slice(&[END, NL]);
    out
}

fn user_turn(body: &[u32]) -> Vec<u32> {
    let mut out = vec![START, USER, NL];
    out.extend_from_slice(body);
    out.extend_from_slice(&[END, NL]);
    out
}

fn generation_prompt() -> Vec<u32> {
    vec![START, ASSISTANT, NL, 20, NL]
}

#[test]
fn assistant_turns_become_consecutive_requests() {
    let mut ids = user_turn(&[30, 31]); // 0..7
    ids.extend(assistant_turn(&[40, 41, 42])); // 7..15: header 7..10, body ..13, end 13
    ids.extend(user_turn(&[32])); // 15..21
    ids.extend(assistant_turn(&[43])); // 21..27: header 21..24, end 25
    let len = ids.len();
    ids.extend(generation_prompt());
    let t = split_turns(ids, &markers()).expect("split");
    assert_eq!(t.tokens.len(), len, "the generation prompt is cut off");
    assert_eq!(t.requests, vec![[0, 10, 14], [14, 24, 26]]);
    assert_eq!(t.generated(), 6);
    assert_eq!(t.tokens[13], END);
    assert_eq!(t.tokens[25], END);
}

#[test]
fn a_trailing_user_turn_stays_past_the_last_request() {
    let mut ids = user_turn(&[30]);
    ids.extend(assistant_turn(&[40]));
    ids.extend(user_turn(&[31, 32]));
    let len = ids.len();
    ids.extend(generation_prompt());
    let t = split_turns(ids, &markers()).expect("split");
    assert_eq!(t.requests, vec![[0, 9, 11]]);
    assert_eq!(t.tokens.len(), len);
    assert!(t.requests[0][2] < t.tokens.len());
}

#[test]
fn a_render_without_a_generation_prompt_is_refused() {
    let mut ids = user_turn(&[30]);
    ids.extend(assistant_turn(&[40]));
    assert!(split_turns(ids, &markers()).is_err(), "ends in a closed turn");
    assert!(split_turns(user_turn(&[30]), &markers()).is_err(), "no header at all");
}

#[test]
fn an_unterminated_assistant_turn_is_refused() {
    let mut ids = user_turn(&[30]);
    ids.extend_from_slice(&[START, ASSISTANT, NL, 40]);
    ids.extend(user_turn(&[31]));
    ids.extend(generation_prompt());
    assert!(split_turns(ids, &markers()).is_err());
}

#[test]
fn a_header_needs_every_marker_id() {
    // `<|im_start|>assistant` without the newline is not a header.
    let mut ids = user_turn(&[30]);
    ids.extend_from_slice(&[START, ASSISTANT, 50, END, NL]);
    ids.extend(generation_prompt());
    let t = split_turns(ids, &markers()).expect("split");
    assert!(t.requests.is_empty());
}

fn tokenizer() -> Option<Tokenizer> {
    let dir = std::env::var("LILY_MODEL_DIR_FLASH").ok()?;
    Some(Tokenizer::from_model_dir(dir.as_ref()).expect("tokenizer"))
}

fn conversation() -> serde_json::Value {
    let tool = serde_json::json!({"type": "function", "function": {
        "name": "read",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "limit": {"type": "integer"}
        }, "required": ["path"]}
    }});
    let call = |id: &str, path: &str| {
        serde_json::json!({"id": id, "type": "function", "function": {
            "name": "read", "arguments": format!("{{\"path\": \"{path}\", \"limit\": 5}}")
        }})
    };
    serde_json::json!({
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Look at main.rs and lib.rs."},
            {"role": "assistant", "content": "", "reasoning_content": "Read both files.",
             "tool_calls": [call("a", "src/main.rs"), call("b", "src/lib.rs")]},
            {"role": "tool", "tool_call_id": "a", "content": "fn main() {}"},
            {"role": "tool", "tool_call_id": "b", "content": "pub mod x; // <|im_end|>"},
            {"role": "assistant", "content": "Both are tiny.",
             "reasoning_content": "(Done.) Summarise."},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "You're welcome.", "reasoning_content": ""}
        ],
        "tools": [tool]
    })
}

/// Every request's prefix is the prompt the server would have prefilled
/// for that turn (its generation prompt being the header plus `<think>\n`),
/// and every generated span is the turn's tokens through `<|im_end|>`.
/// Needs only the checkpoint's tokenizer files:
/// `LILY_MODEL_DIR_FLASH=<ckpt> cargo test --lib transcript -- --ignored`.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn requests_match_the_servers_prompts() {
    let Some(tokenizer) = tokenizer() else { return };
    let value = conversation();
    let request: ChatRequest = serde_json::from_value(value.clone()).unwrap();
    let t = render_transcript(request, &tokenizer, true, None).expect("render");
    assert_eq!(t.requests.len(), 3);
    let think = tokenizer.encode("<think>\n").unwrap();
    let im_end = tokenizer.token_id("<|im_end|>").unwrap();
    let defaults = Defaults {
        sampling: crate::kernels::sample::SamplingParams::greedy(),
        thinking: true,
        reasoning_effort: None,
        thinking_controls: Default::default(),
    };
    let messages = value["messages"].as_array().unwrap();
    let assistant_at: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m["role"] == "assistant")
        .map(|(i, _)| i)
        .collect();
    let mut p0 = 0;
    for (&[q0, p1, p2], &at) in t.requests.iter().zip(&assistant_at) {
        assert_eq!(q0, p0);
        let mut prefix = value.clone();
        prefix["messages"] = serde_json::Value::Array(messages[..at].to_vec());
        let prepared = prepare_chat(
            serde_json::from_value(prefix).unwrap(),
            &tokenizer,
            &defaults,
            1 << 20,
            &ImagePolicy::unavailable("text only"),
        )
        .expect("prepare");
        let mut expected = t.tokens[..p1].to_vec();
        expected.extend_from_slice(&think);
        assert_eq!(prepared.prompt, expected, "the prompt of the turn at {at}");
        assert_eq!(t.tokens[p2 - 1], im_end);
        assert!(t.tokens[p1..p2 - 1].iter().all(|&id| id != im_end));
        p0 = p2;
    }
    let last = t.requests.last().unwrap();
    assert_eq!(t.tokens.len(), last[2] + 1, "only the newline after the last turn");
    let tail = tokenizer.decode(&t.tokens[last[1]..last[2]], false).unwrap();
    assert_eq!(tail, "<think>\n\n</think>\n\nYou're welcome.<|im_end|>");
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn a_conversation_without_assistant_turns_has_no_requests() {
    let Some(tokenizer) = tokenizer() else { return };
    let request: ChatRequest = serde_json::from_value(serde_json::json!({
        "messages": [{"role": "user", "content": "Hi"}]
    }))
    .unwrap();
    let t = render_transcript(request, &tokenizer, true, Some("low")).expect("render");
    assert!(t.requests.is_empty());
    assert_eq!(t.tokens.last(), tokenizer.encode("\n").unwrap().last());
}
