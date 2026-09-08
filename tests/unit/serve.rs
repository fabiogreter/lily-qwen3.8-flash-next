use super::*;

#[test]
fn call_ids_are_stable_per_request_and_index() {
    assert_eq!(call_id("chatcmpl-1-1", 0), call_id("chatcmpl-1-1", 0));
    assert_ne!(call_id("chatcmpl-1-1", 0), call_id("chatcmpl-1-1", 1));
    assert_ne!(call_id("chatcmpl-1-1", 0), call_id("chatcmpl-1-2", 0));
    assert!(call_id("x", 3).starts_with("call_"));
}

#[test]
fn chunks_carry_the_openai_shape() {
    let c = chunk("id", 7, "m", json!({"content": "hi"}), Some("stop"));
    assert_eq!(c["object"], "chat.completion.chunk");
    assert_eq!(c["choices"][0]["delta"]["content"], "hi");
    assert_eq!(c["choices"][0]["finish_reason"], "stop");
    let t = text_chunk("id", 7, "m", "hi", None);
    assert_eq!(t["object"], "text_completion");
    assert_eq!(t["choices"][0]["text"], "hi");
    assert!(t["choices"][0]["finish_reason"].is_null());
}

#[test]
fn request_budgets_fill_the_context_by_default() {
    assert_eq!(api::resolve_budget_for_test(10, None, 100).unwrap(), 90);
    assert_eq!(api::resolve_budget_for_test(10, Some(20), 100).unwrap(), 20);
    assert!(api::resolve_budget_for_test(10, Some(91), 100).is_err());
    assert!(api::resolve_budget_for_test(100, None, 100).is_err());
    assert!(api::resolve_budget_for_test(10, Some(0), 100).is_err());
}

#[test]
fn chat_requests_parse_the_agent_surface() {
    let body = br#"{
        "model": "anything",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "f", "arguments": "{\"a\": 1}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"}
        ],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}}}}],
        "tool_choice": "auto",
        "temperature": 0.7, "top_p": 0.9, "top_k": 40, "seed": 5,
        "stream": true, "stream_options": {"include_usage": true},
        "max_completion_tokens": 100, "stop": ["END"],
        "reasoning_effort": "low",
        "chat_template_kwargs": {"enable_thinking": true},
        "prompt_cache_key": "k",
        "user": "ignored", "parallel_tool_calls": true, "logit_bias": {}
    }"#;
    let request: api::ChatRequest = serde_json::from_slice(body).expect("parse");
    assert_eq!(request.messages.len(), 4);
    assert!(request.stream);
    assert_eq!(request.sampling.top_k, Some(40));
    assert_eq!(request.max_completion_tokens, Some(100));
    assert_eq!(request.reasoning_effort.as_deref(), Some("low"));
}

#[test]
fn sampling_defaults_apply_overrides() {
    let dir = std::env::temp_dir().join(format!("lily-serve-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        br#"{"do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95}"#,
    )
    .unwrap();
    let base = sampling_defaults(&dir, &SamplingOverrides::default()).unwrap();
    assert_eq!((base.temperature, base.top_k, base.top_p), (1.0, 20, 0.95));
    let over = sampling_defaults(&dir, &SamplingOverrides { temperature: Some(0.0), ..Default::default() }).unwrap();
    assert!(over.is_greedy());
    std::fs::write(dir.join("config.json"), br#"{"model_type": "x", "text_config": {"eos_token_id": [1, 2]}}"#).unwrap();
    assert_eq!(checkpoint_eos_ids(&dir).unwrap(), vec![1, 2]);
    std::fs::remove_dir_all(&dir).ok();
}
