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
fn request_budgets_fill_the_context_by_default_and_clamp_to_it() {
    let budget = |prompt, requested, max_seq| api::resolve_budget_for_test(prompt, requested, max_seq);
    let fits = |max_tokens| api::Budget { max_tokens, clamped_from: None };
    assert_eq!(budget(10, None, 100).unwrap(), fits(90));
    assert_eq!(budget(10, Some(20), 100).unwrap(), fits(20));
    assert_eq!(budget(10, Some(90), 100).unwrap(), fits(90));
    // Asking for more than the prompt leaves room for clamps instead of
    // refusing (the response then ends with finish_reason "length").
    assert_eq!(budget(10, Some(91), 100).unwrap(), api::Budget { max_tokens: 90, clamped_from: Some(91) });
    assert_eq!(
        budget(99_204, Some(32_000), 131_072).unwrap(),
        api::Budget { max_tokens: 131_072 - 99_204, clamped_from: Some(32_000) }
    );
    // Only a prompt that fills the context is refused, with the numbers.
    let error = budget(100, None, 100).unwrap_err().to_string();
    assert!(error.starts_with("prompt exceeds the server context: 100 prompt tokens, 100 tokens of context"), "{error}");
    let error = budget(131_072, Some(1), 131_072).unwrap_err().to_string();
    assert!(error.contains("131072 prompt tokens"), "{error}");
    assert!(budget(101, Some(5), 100).is_err());
    assert!(budget(0, None, 100).is_err());
    assert!(budget(10, Some(0), 100).is_err());
}

#[test]
fn default_cache_budget_leaves_room_for_the_paged_table_and_other_apps() {
    const GB: usize = 1 << 30;
    // The 128 GB machine: 115.4 GB working set, 73.0 GB weights, 32.0 GB
    // paged table. The arithmetic gives 2.4 GB; the floor lifts it to 8 GiB.
    let (budget, floored) = derive_cache_budget(115_400_000_000, 73_000_000_000, 32_000_000_000);
    assert_eq!((budget, floored), (8 * GB, true));
    // With the table resident on the GPU (`--ngram-table resident`) it is in
    // the allocated bytes instead and counts once.
    let (budget, floored) = derive_cache_budget(115_400_000_000, 105_000_000_000, 0);
    assert_eq!((budget, floored), (8 * GB, true));
    // A small model on the same machine: working set - allocated - table - headroom.
    let (budget, floored) = derive_cache_budget(115_400_000_000, 8_200_000_000, 32_000_000_000);
    assert_eq!((budget, floored), (115_400_000_000 - 8_200_000_000 - 32_000_000_000 - 8 * GB, false));
    // No paged weights at all (the Qwen3.6 path).
    let (budget, floored) = derive_cache_budget(115_400_000_000, 20_000_000_000, 0);
    assert_eq!((budget, floored), (115_400_000_000 - 20_000_000_000 - 8 * GB, false));
    // Nothing underflows when the weights alone exceed the working set.
    assert_eq!(derive_cache_budget(10 * GB, 20 * GB, 0), (8 * GB, true));
    assert_eq!((BUDGET_HEADROOM_BYTES, BUDGET_FLOOR_BYTES), (8 * GB, 8 * GB));
}

#[test]
fn effective_context_takes_the_smallest_of_flag_kernel_and_checkpoint() {
    assert_eq!(effective_max_seq(131_072, 262_144), 131_072);
    assert_eq!(effective_max_seq(300_000, 262_144), MAX_SEQ.min(262_144));
    assert_eq!(effective_max_seq(131_072, 65_536), 65_536);
    // An undeclared window (0) only leaves the kernel limit.
    assert_eq!(effective_max_seq(131_072, 0), 131_072);
    assert_eq!(effective_max_seq(usize::MAX, 0), MAX_SEQ);
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

#[test]
fn durations_parse_with_units() {
    assert_eq!(parse_duration_secs("0").unwrap(), 0);
    assert_eq!(parse_duration_secs("45").unwrap(), 45);
    assert_eq!(parse_duration_secs("45s").unwrap(), 45);
    assert_eq!(parse_duration_secs("90m").unwrap(), 5400);
    assert_eq!(parse_duration_secs(" 2h ").unwrap(), 7200);
    assert_eq!(parse_duration_secs("1.5h").unwrap(), 5400);
    assert_eq!(parse_duration_secs("3d").unwrap(), 259_200);
    assert!(parse_duration_secs("30x").is_err());
    assert!(parse_duration_secs("").is_err());
    assert!(parse_duration_secs("m").is_err());
}

#[test]
fn health_keeps_ok_and_loading_and_adds_the_state() {
    // Status codes and `status` are what pre-idle-unload clients saw.
    assert_eq!(State::Loading.health(), (503, "loading"));
    assert_eq!(State::Ready.health(), (200, "ok"));
    // Unloaded or reloading: a request would still be served (after a wait).
    assert_eq!(State::Idle.health(), (200, "ok"));
    assert_eq!(State::Reloading.health(), (200, "ok"));
    assert_eq!(State::Stopping.health(), (503, "stopping"));
    // Recovering from a GPU fault: requests are queued for the reload, but
    // the code is 503 so a monitor sees that something went wrong.
    assert_eq!(State::Recovering.health(), (503, "recovering"));
    assert!(State::Recovering.accepting());
    for state in [State::Loading, State::Ready, State::Idle, State::Reloading, State::Stopping] {
        assert_eq!(state.accepting(), state.health().0 == 200, "{state:?}");
    }
    assert_eq!(State::Idle.as_str(), "idle");
    assert_eq!(State::Reloading.as_str(), "reloading");
    assert_eq!(State::Recovering.as_str(), "recovering");
}

#[test]
fn lifecycle_round_trips_every_state() {
    let lifecycle = Lifecycle::new(State::Loading);
    assert_eq!(lifecycle.get(), State::Loading);
    for state in [State::Ready, State::Idle, State::Reloading, State::Stopping, State::Recovering, State::Loading] {
        lifecycle.set(state);
        assert_eq!(lifecycle.get(), state);
    }
}

#[test]
fn recovery_budget_allows_three_faults_per_window_then_gives_up() {
    let t0 = Instant::now();
    let m = |minutes: u64| Duration::from_secs(minutes * 60);
    let mut budget = RecoveryBudget::new(3, m(10));
    assert_eq!(budget.record(t0), Some(1));
    assert_eq!(budget.record(t0 + m(1)), Some(2));
    assert_eq!(budget.record(t0 + m(2)), Some(3));
    // The fourth fault within ten minutes is one too many.
    assert_eq!(budget.record(t0 + m(3)), None);
    assert_eq!(budget.faults_in_window(), 4);
    // The window slides: once the early faults age out, recovering resumes.
    let mut budget = RecoveryBudget::new(3, m(10));
    for i in 0..3 {
        assert!(budget.record(t0 + m(i)).is_some());
    }
    assert_eq!(budget.record(t0 + m(10)), Some(3), "the fault at t0 has aged out");
    assert_eq!(budget.record(t0 + m(10) + Duration::from_secs(1)), None);
    // The server's own constants.
    assert_eq!((MAX_RECOVERIES, RECOVERY_WINDOW), (3, Duration::from_secs(600)));
}

#[test]
fn idle_timer_counts_from_the_last_activity() {
    let t0 = Instant::now();
    let s = Duration::from_secs;
    // 0 never expires: nothing to wait for.
    let never = IdleTimer::new(0, t0);
    assert_eq!(never.remaining(t0 + s(1_000_000)), None);
    assert!(!never.expired(t0 + s(1_000_000)));

    let mut timer = IdleTimer::new(120, t0);
    assert_eq!(timer.remaining(t0), Some(s(120)));
    assert_eq!(timer.remaining(t0 + s(50)), Some(s(70)));
    assert!(!timer.expired(t0 + s(119)));
    assert!(timer.expired(t0 + s(120)));
    assert!(timer.expired(t0 + s(500)));
    assert_eq!(timer.remaining(t0 + s(500)), Some(Duration::ZERO));
    // A request at t0+100 pushes the deadline out.
    timer.touch(t0 + s(100));
    assert!(!timer.expired(t0 + s(219)));
    assert_eq!(timer.remaining(t0 + s(219)), Some(s(1)));
    assert!(timer.expired(t0 + s(220)));
    // A clock reading before the last activity never underflows.
    assert_eq!(timer.remaining(t0), Some(s(120)));
}

#[test]
fn seconds_are_described_in_the_largest_exact_unit() {
    assert_eq!(describe_secs(1800), "30m");
    assert_eq!(describe_secs(7200), "2h");
    assert_eq!(describe_secs(90), "90s");
    assert_eq!(describe_secs(3660), "61m");
}

#[test]
fn unspecified_bind_addresses_are_reached_on_loopback() {
    let any: SocketAddr = "0.0.0.0:8000".parse().unwrap();
    assert_eq!(loopback_of(any), "127.0.0.1:8000".parse().unwrap());
    let any6: SocketAddr = "[::]:8000".parse().unwrap();
    assert_eq!(loopback_of(any6), "[::1]:8000".parse().unwrap());
    let local: SocketAddr = "192.168.1.5:8000".parse().unwrap();
    assert_eq!(loopback_of(local), local);
}
