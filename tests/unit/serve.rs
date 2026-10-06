use super::*;
use crate::qwen4exp::ImageSpan;

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
    let budget = |prompt, requested, max_seq| {
        api::resolve_budget_for_test(prompt, requested, max_seq)
    };
    let fits = |max_tokens| api::Budget { max_tokens, clamped_from: None };
    assert_eq!(budget(10, None, 100).unwrap(), fits(90));
    assert_eq!(budget(10, Some(20), 100).unwrap(), fits(20));
    assert_eq!(budget(10, Some(90), 100).unwrap(), fits(90));
    // Asking for more than the prompt leaves room for clamps instead of
    // refusing (the response then ends with finish_reason "length").
    assert_eq!(
        budget(10, Some(91), 100).unwrap(),
        api::Budget { max_tokens: 90, clamped_from: Some(91) }
    );
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
    let (budget, floored) =
        derive_cache_budget(115_400_000_000, 73_000_000_000, 32_000_000_000);
    assert_eq!((budget, floored), (8 * GB, true));
    // With the table resident on the GPU (`--ngram-table resident`) it is in
    // the allocated bytes instead and counts once.
    let (budget, floored) = derive_cache_budget(115_400_000_000, 105_000_000_000, 0);
    assert_eq!((budget, floored), (8 * GB, true));
    // A small model on the same machine: working set - allocated - table - headroom.
    let (budget, floored) =
        derive_cache_budget(115_400_000_000, 8_200_000_000, 32_000_000_000);
    assert_eq!(
        (budget, floored),
        (115_400_000_000 - 8_200_000_000 - 32_000_000_000 - 8 * GB, false)
    );
    // No paged weights at all (an in-GPU n-gram table, or none).
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
    let dir =
        std::env::temp_dir().join(format!("lily-serve-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        br#"{"do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95}"#,
    )
    .unwrap();
    let base = sampling_defaults(&dir, &SamplingOverrides::default()).unwrap();
    assert_eq!((base.temperature, base.top_k, base.top_p), (1.0, 20, 0.95));
    let over = sampling_defaults(
        &dir,
        &SamplingOverrides { temperature: Some(0.0), ..Default::default() },
    )
    .unwrap();
    assert!(over.is_greedy());
    std::fs::write(
        dir.join("config.json"),
        br#"{"model_type": "x", "text_config": {"eos_token_id": [1, 2]}}"#,
    )
    .unwrap();
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
    for state in
        [State::Loading, State::Ready, State::Idle, State::Reloading, State::Stopping]
    {
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
    for state in [
        State::Ready,
        State::Idle,
        State::Reloading,
        State::Stopping,
        State::Recovering,
        State::Loading,
    ] {
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

/// The placeholder ids of the Qwen3.8 vocabulary (config.json).
const IDS: api::PlaceholderIds = api::PlaceholderIds {
    image_pad: 248056,
    vision_start: 248053,
    vision_end: 248054,
    video_pad: Some(248057),
};

#[test]
fn image_parts_are_read_in_both_shapes_and_text_only_content_flattens_as_before() {
    use api::{Content, ContentItem, template_content};
    let parts: Content = serde_json::from_value(json!([
        {"type": "text", "text": "Look at "},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "high"}},
        {"type": "input_text", "text": " and "},
        {"type": "input_image", "image_url": "data:image/jpeg;base64,AA=="},
        {"type": "image_url", "image_url": "data:image/png;base64,AQ=="},
    ]))
    .unwrap();
    let items = parts.into_items().unwrap();
    assert_eq!(
        items,
        vec![
            ContentItem::Text("Look at ".into()),
            ContentItem::Image("data:image/png;base64,AA==".into()),
            ContentItem::Text(" and ".into()),
            ContentItem::Image("data:image/jpeg;base64,AA==".into()),
            ContentItem::Image("data:image/png;base64,AQ==".into()),
        ]
    );
    // With images the template gets structured items, in order.
    assert_eq!(
        template_content(items),
        json!([
            {"type": "text", "text": "Look at "},
            {"type": "image"},
            {"type": "text", "text": " and "},
            {"type": "image"},
            {"type": "image"},
        ])
    );
    // Text-only parts flatten to one string, exactly as before images.
    let text: Content = serde_json::from_value(json!([
        {"type": "text", "text": "a"},
        {"type": "input_text", "text": "b"},
        {"type": "text"},
    ]))
    .unwrap();
    assert_eq!(template_content(text.into_items().unwrap()), json!("ab"));
    let plain: Content = serde_json::from_value(json!("hello")).unwrap();
    assert_eq!(template_content(plain.into_items().unwrap()), json!("hello"));
    // An image part without its URL, and an unknown part type, are refused.
    let missing: Content =
        serde_json::from_value(json!([{"type": "image_url"}])).unwrap();
    assert!(missing.into_items().unwrap_err().to_string().contains("has no image_url"));
    let unknown: Content =
        serde_json::from_value(json!([{"type": "input_audio", "text": "x"}])).unwrap();
    let err = unknown.into_items().unwrap_err().to_string();
    assert!(err.contains("\"input_audio\" is not supported"), "{err}");
}

#[test]
fn placeholders_are_expanded_per_image_and_refused_out_of_place() {
    // A rendered prompt with two images: text, marker triple, text, triple, text.
    let prompt = [1, 2, 248053, 248056, 248054, 3, 248053, 248056, 248054, 4];
    // Grids (4, 6) -> 6 placeholders and (2, 4) -> 2.
    let (expanded, spans) =
        api::expand_image_pads(&prompt, &IDS, &[(4, 6), (2, 4)]).unwrap();
    let mut want = vec![1, 2, 248053];
    want.extend(std::iter::repeat_n(248056, 6));
    want.extend([248054, 3, 248053, 248056, 248056, 248054, 4]);
    assert_eq!(expanded, want);
    assert_eq!(
        spans,
        vec![
            ImageSpan { start: 3, len: 6, grid_h: 4, grid_w: 6 },
            ImageSpan { start: 12, len: 2, grid_h: 2, grid_w: 4 },
        ]
    );
    // Fewer or more markers than images (a template that wrote one too many
    // or dropped one): both are refused, never mapped.
    let err = api::expand_image_pads(&prompt, &IDS, &[(4, 6)]).unwrap_err().to_string();
    assert!(err.contains("2 <|image_pad|>, 2 <|vision_start|> and 2 <|vision_end|> tokens for 1 images"), "{err}");
    assert!(err.contains("reserved for image content"), "{err}");
    assert!(api::expand_image_pads(&prompt, &IDS, &[(4, 6), (2, 4), (2, 2)]).is_err());
    // A lone pad (no start/end around it) is caught by the marker counts
    // even when the pad count happens to match.
    let typed = [1, 248056, 2, 248053, 248056, 248054];
    assert!(api::expand_image_pads(&typed, &IDS, &[(2, 2), (2, 2)]).is_err());
    // A text request may carry none of them.
    assert!(api::check_placeholders(&[1, 2, 3], &IDS, 0).is_ok());
    for reserved in [248053, 248054, 248056, 248057] {
        let err = api::check_placeholders(&[1, reserved, 3], &IDS, 0)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("reserved for image content") || err.contains("video input"),
            "{reserved}: {err}"
        );
    }
    // A grid that is not made of 2 x 2 blocks is refused.
    assert!(
        api::expand_image_pads(&[248053, 248056, 248054], &IDS, &[(3, 4)]).is_err()
    );
}

#[test]
fn the_image_digest_covers_the_rows_and_the_grid() {
    let rows = vec![0.5f32; 1536 * 4];
    let a = api::image_digest(&rows, 2, 2);
    assert_eq!(a, api::image_digest(&rows.clone(), 2, 2));
    // The same rows read as another grid are another image.
    assert_ne!(a, api::image_digest(&rows, 4, 1));
    let mut other = rows.clone();
    other[100] = 0.25;
    assert_ne!(a, api::image_digest(&other, 2, 2));
    // Pinned so a change in what is digested is noticed: SHA-256 of the
    // 24 576 little-endian f32 0.5 bytes followed by two u64 2s.
    let mut bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
    bytes.extend(2u64.to_le_bytes());
    bytes.extend(2u64.to_le_bytes());
    assert_eq!(a, crate::sha256::sha256(&bytes));
}

/// Warm-up's cost on a load and on a reload in the same process (what an idle
/// unload leads to), printed as the server's own warm-up lines. Needs a
/// checkpoint; the four-layer one is enough and keeps it to a few GB:
/// `LILY_MODEL_DIR_FLASH=<ckpt> cargo test --lib warm_up_on_load_and_reload -- --ignored --nocapture`.
#[test]
#[ignore = "timing only; needs LILY_MODEL_DIR_FLASH"]
fn warm_up_on_load_and_reload() {
    let Ok(dir) = std::env::var("LILY_MODEL_DIR_FLASH") else { return };
    let options = LoadOptions {
        ngram_storage: NgramStorage::Paged,
        mtp_drafts: 2,
        vision: VisionMode::Off,
        expert_slots: None,
        expert_usage: None,
        memory_budget: None,
        expert_usage_out: None,
        session_context: None,
    };
    for round in ["load", "reload"] {
        let ctx = MetalContext::new().expect("metal");
        let model =
            <Qwen4ExpModel as LanguageModel>::load(&ctx, Path::new(&dir), &options)
                .expect("model");
        let mut scratch = model.new_scratch_with_capacity(&ctx, 8192).expect("scratch");
        eprintln!("{round}, {:.1} GB allocated:", ctx.current_allocated() as f64 / 1e9);
        warm_up(&ctx, &model, &mut scratch).expect("warm-up");
        eprintln!("{round}, once more:");
        warm_up(&ctx, &model, &mut scratch).expect("second warm-up");
    }
}

#[test]
fn engine_queue_limits_jobs_but_not_control_messages() {
    let queue = EngineQueue::new(2);
    assert_eq!(queue.capacity(), 4);
    assert!(queue.admit());
    assert!(queue.admit());
    assert!(!queue.admit(), "a third job exceeds --queue 2");
    queue.left();
    assert!(queue.admit(), "a received job frees its place");
    // One arrival at a time, whatever the jobs do.
    assert!(queue.claim_arrival());
    assert!(!queue.claim_arrival());
    queue.arrival_taken();
    assert!(queue.claim_arrival());
    assert_eq!(EngineQueue::new(0).limit, 1, "the limit is at least one");
}

#[test]
fn a_full_job_queue_still_takes_the_arrival_and_the_stop_wake() {
    let queue = EngineQueue::new(1);
    let (tx, rx) = mpsc::sync_channel::<Cmd>(queue.capacity());
    assert!(queue.claim_arrival());
    tx.try_send(Cmd::Arrival).expect("arrival fits");
    assert!(queue.admit());
    assert!(tx.try_send(Cmd::Wake).is_ok(), "the stop wake fits next to one arrival");
    assert!(!queue.admit());
    // Draining skips the control messages and refuses no job here (none sent).
    assert_eq!(refuse_queued(&rx, &queue, 503, "stopping"), 0);
    assert!(queue.claim_arrival(), "the drained arrival was released");
}

#[test]
fn keep_alive_ticks_once_an_interval_and_only_inside_the_window() {
    let t0 = Instant::now();
    let s = Duration::from_secs;
    let mut keep_alive = KeepAlive::new(s(1), t0);
    // A closed window (no pin, or the hold is over) never ticks.
    assert_eq!(keep_alive.next_tick(t0, None), None);
    assert_eq!(keep_alive.next_tick(t0, Some(Duration::ZERO)), None);
    // Inside the window: an interval after the last submission.
    assert_eq!(keep_alive.next_tick(t0, Some(s(60))), Some(t0 + s(1)));
    // A submission moves it.
    keep_alive.touch(t0 + s(5));
    assert_eq!(keep_alive.next_tick(t0 + s(5), Some(s(55))), Some(t0 + s(6)));
    // A tick at or after the window's end is not due at all.
    assert_eq!(keep_alive.next_tick(t0 + s(5), Some(s(1))), None);
    assert_eq!(
        keep_alive.next_tick(t0 + s(5), Some(Duration::from_millis(1001))),
        Some(t0 + s(6))
    );
    // A failed tick stops them until the engine submits again.
    keep_alive.stop();
    assert_eq!(keep_alive.next_tick(t0 + s(5), Some(s(55))), None);
    keep_alive.touch(t0 + s(7));
    assert_eq!(keep_alive.next_tick(t0 + s(7), Some(s(53))), Some(t0 + s(8)));
}

const TICK: Duration = Duration::from_millis(50);

/// Runs `recv_keeping_warm` with a 50 ms interval and records the ticks.
fn wait_recording<T>(
    rx: &Receiver<T>,
    deadline: Option<Instant>,
    window: impl Fn(Instant) -> Option<Duration>,
) -> (std::result::Result<T, RecvTimeoutError>, Vec<Instant>) {
    let mut keep_alive = KeepAlive::new(TICK, Instant::now());
    let mut ticks = Vec::new();
    let received = recv_keeping_warm(rx, deadline, &mut keep_alive, window, || {
        ticks.push(Instant::now());
        Ok(())
    });
    (received, ticks)
}

#[test]
fn the_idle_wait_ticks_while_the_window_is_open_and_returns_at_its_deadline() {
    let (_tx, rx) = mpsc::channel::<()>();
    let start = Instant::now();
    let window_end = start + Duration::from_millis(225);
    let deadline = start + Duration::from_millis(400);
    let (received, ticks) = wait_recording(&rx, Some(deadline), |now| {
        Some(window_end.saturating_duration_since(now))
    });
    let returned = Instant::now();
    assert!(matches!(received, Err(RecvTimeoutError::Timeout)));
    assert!(returned >= deadline && returned < deadline + TICK, "on time");
    assert!((3..=4).contains(&ticks.len()), "{} ticks", ticks.len());
    assert!(ticks.iter().all(|&at| at < window_end), "none after the window");
    for pair in ticks.windows(2) {
        assert!(pair[1] - pair[0] >= TICK, "one per interval");
    }
}

#[test]
fn a_request_ends_the_idle_wait_at_once_between_ticks() {
    let (tx, rx) = mpsc::channel::<Instant>();
    let sender = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(130));
        tx.send(Instant::now()).expect("send");
        // Keep the channel open: only the message may end the wait.
        std::thread::sleep(Duration::from_millis(300));
    });
    let (received, ticks) =
        wait_recording(&rx, None, |_| Some(Duration::from_secs(60)));
    let returned = Instant::now();
    let sent = received.expect("the request");
    assert!(returned - sent < Duration::from_millis(20), "{:?}", returned - sent);
    assert!(!ticks.is_empty(), "it ticked before the request");
    assert!(ticks.iter().all(|&at| at <= sent), "and not after it");
    sender.join().expect("sender");
}

#[test]
fn a_stop_wake_ends_the_idle_wait_at_once() {
    let (tx, rx) = mpsc::sync_channel::<Cmd>(2);
    let sent = Instant::now() + Duration::from_millis(80);
    let sender = std::thread::spawn(move || {
        std::thread::sleep(sent.saturating_duration_since(Instant::now()));
        tx.send(Cmd::Wake).expect("send");
        std::thread::sleep(Duration::from_millis(300));
    });
    let (received, _) = wait_recording(&rx, None, |_| Some(Duration::from_secs(60)));
    assert!(matches!(received, Ok(Cmd::Wake)));
    assert!(Instant::now() - sent < Duration::from_millis(20));
    sender.join().expect("sender");
}

#[test]
fn a_released_pin_stops_the_ticks_at_once() {
    let (tx, rx) = mpsc::channel::<()>();
    let pinned = Arc::new(AtomicBool::new(true));
    let released_at = Arc::new(std::sync::Mutex::new(None::<Instant>));
    let sender = {
        let (pinned, released_at) = (pinned.clone(), released_at.clone());
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(120));
            // The monitor thread's pressure release, or the hold's end.
            pinned.store(false, Ordering::SeqCst);
            *released_at.lock().expect("lock") = Some(Instant::now());
            std::thread::sleep(Duration::from_millis(300));
            tx.send(()).expect("send");
        })
    };
    let (received, ticks) = wait_recording(&rx, None, |_| {
        pinned.load(Ordering::SeqCst).then_some(Duration::from_secs(60))
    });
    received.expect("the request");
    let released = released_at.lock().expect("lock").expect("released");
    assert!(!ticks.is_empty(), "it ticked while pinned");
    assert!(
        ticks.iter().all(|&at| at < released + Duration::from_millis(5)),
        "no tick after the release"
    );
    sender.join().expect("sender");
}

#[test]
fn without_a_held_pin_the_idle_wait_submits_nothing() {
    // An unpinned engine (`--pin-weights off`, the expert cache's `auto`,
    // a released pin) or an unloaded one: no window, no tick.
    let (_tx, rx) = mpsc::channel::<()>();
    let deadline = Instant::now() + Duration::from_millis(200);
    let (received, ticks) = wait_recording(&rx, Some(deadline), |_| None);
    assert!(matches!(received, Err(RecvTimeoutError::Timeout)));
    assert!(ticks.is_empty());
    assert!(Instant::now() < deadline + TICK);
}

#[test]
fn the_idle_wait_deadline_is_never_delayed_by_ticks() {
    // The idle unload, the pin's release and the write-ahead poll are the
    // deadline; a tick due at or after it yields.
    let (_tx, rx) = mpsc::channel::<()>();
    for after in [TICK, TICK + Duration::from_millis(5), Duration::from_millis(120)] {
        let deadline = Instant::now() + after;
        let (received, ticks) =
            wait_recording(&rx, Some(deadline), |_| Some(Duration::from_secs(60)));
        let returned = Instant::now();
        assert!(matches!(received, Err(RecvTimeoutError::Timeout)));
        assert!(
            returned >= deadline && returned < deadline + Duration::from_millis(20)
        );
        assert!(ticks.iter().all(|&at| at < deadline));
    }
}

#[test]
fn a_failed_tick_is_not_retried_until_the_engine_submits_again() {
    let (_tx, rx) = mpsc::channel::<()>();
    let mut keep_alive = KeepAlive::new(TICK, Instant::now());
    let mut calls = 0;
    let deadline = Instant::now() + Duration::from_millis(300);
    let received = recv_keeping_warm(
        &rx,
        Some(deadline),
        &mut keep_alive,
        |_| Some(Duration::from_secs(60)),
        || {
            calls += 1;
            Err(anyhow::anyhow!("faulted queue"))
        },
    );
    assert!(matches!(received, Err(RecvTimeoutError::Timeout)));
    assert_eq!(calls, 1);
    assert_eq!(
        keep_alive.next_tick(Instant::now(), Some(Duration::from_secs(60))),
        None
    );
}

#[test]
fn the_log_line_says_whether_a_session_was_cut_back_or_forked() {
    assert_eq!(describe_reuse(Some(9_000), false), ", cut back by 9000");
    assert_eq!(describe_reuse(None, true), ", forked");
    assert_eq!(describe_reuse(None, false), "");
}
