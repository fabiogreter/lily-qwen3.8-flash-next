use super::*;

#[test]
fn usage_reports_reasoning_tokens_for_chat() {
    let u = usage(100, 40, 30, Some(12), None);
    assert_eq!(u["prompt_tokens"], 100);
    assert_eq!(u["completion_tokens"], 30);
    assert_eq!(u["total_tokens"], 130);
    assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 40);
    assert_eq!(
        u["completion_tokens_details"],
        serde_json::json!({"reasoning_tokens": 12})
    );
    // No reasoning generated still says so rather than leaving the field out.
    let u = usage(100, 0, 5, Some(0), None);
    assert_eq!(u["completion_tokens_details"]["reasoning_tokens"], 0);
}

#[test]
fn usage_keeps_the_prediction_counts_next_to_reasoning() {
    let u = usage(10, 0, 20, Some(7), Some(Speculation { drafted: 9, accepted: 6 }));
    assert_eq!(
        u["completion_tokens_details"],
        serde_json::json!({
            "reasoning_tokens": 7,
            "accepted_prediction_tokens": 6,
            "rejected_prediction_tokens": 3,
        })
    );
    // Text completions have no reasoning block: only the draft counts.
    let u = usage(10, 0, 20, None, Some(Speculation { drafted: 4, accepted: 4 }));
    assert_eq!(
        u["completion_tokens_details"],
        serde_json::json!({"accepted_prediction_tokens": 4, "rejected_prediction_tokens": 0})
    );
    let u = usage(10, 0, 20, None, None);
    assert!(u.get("completion_tokens_details").is_none());
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for (j, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            out.push(if j <= chunk.len() {
                ALPHABET[(n >> shift & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    out
}

/// A 300 x 200 RGB PNG with a gradient seeded by `seed`, as a data URI.
fn png_data_uri(seed: u8) -> String {
    let (w, h) = (300u32, 200u32);
    let data: Vec<u8> = (0..w * h * 3).map(|i| (i as u8).wrapping_mul(seed)).collect();
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&data).unwrap();
    }
    format!("data:image/png;base64,{}", base64(&out))
}

fn identity(n: u8) -> ImageIdentity {
    ImageIdentity { grid_h: n as usize, grid_w: 2, digest: [n; 32] }
}

#[test]
fn the_image_memo_remembers_by_uri_and_drops_the_oldest_past_capacity() {
    let memo = ImageMemo::default();
    let a = ImageMemo::key("data:image/png;base64,AA==");
    assert_ne!(a, ImageMemo::key("data:image/png;base64,AQ=="));
    assert_eq!(memo.get(&a), None);
    memo.insert(a, identity(1));
    assert_eq!(memo.get(&a), Some(identity(1)));
    // Inserting a known key again neither grows the memo nor its order.
    memo.insert(a, identity(1));
    assert_eq!(memo.len(), 1);
    for n in 0..ImageMemo::CAPACITY {
        memo.insert(ImageMemo::key(&n.to_string()), identity(2));
    }
    assert_eq!(memo.len(), ImageMemo::CAPACITY);
    assert_eq!(memo.get(&a), None, "the oldest entry went first");
    assert_eq!(memo.get(&ImageMemo::key("0")), Some(identity(2)));
}

#[test]
fn a_remembered_image_is_not_decoded_and_its_deferred_rows_match() {
    let policy = ImagePolicy {
        limits: ImageLimits::default(),
        available: Ok(()),
        memo: Arc::default(),
    };
    let url = png_data_uri(7);
    let (first, id1) = prepare_image(url.clone(), 1, &policy).unwrap();
    let ImagePixels::Ready(rows) = &first else { panic!("a new image is decoded") };
    let (second, id2) = prepare_image(url.clone(), 1, &policy).unwrap();
    assert!(
        matches!(second, ImagePixels::Deferred { .. }),
        "a known image is not decoded"
    );
    assert_eq!(id1, id2);
    let image = |pixels, id: ImageIdentity| PreparedImage {
        pixels,
        span: ImageSpan { start: 0, len: 0, grid_h: id.grid_h, grid_w: id.grid_w },
        digest: id.digest,
    };
    let deferred = image(second, id2);
    assert_eq!(deferred.rows().unwrap().as_ref(), rows.as_slice());
    // A deferred image that preprocesses to another identity is refused.
    let wrong = image(
        ImagePixels::Deferred { url, limits: policy.limits },
        ImageIdentity { digest: [0; 32], ..id1 },
    );
    let err = wrong.rows().unwrap_err().to_string();
    assert!(err.contains("preprocessed differently"), "{err}");
    // Another image is decoded again.
    let (third, _) = prepare_image(png_data_uri(9), 2, &policy).unwrap();
    assert!(matches!(third, ImagePixels::Ready(_)));
    assert_eq!(policy.memo.len(), 2);
}

#[test]
fn thinking_budgets_parse_per_effort_or_for_all() {
    let all = ThinkingBudgets::parse("8000").unwrap();
    assert_eq!(
        all,
        ThinkingBudgets { low: Some(8000), medium: Some(8000), xhigh: Some(8000) }
    );
    let map = ThinkingBudgets::parse("low=4000, medium=8000,high=16000").unwrap();
    assert_eq!(
        map,
        ThinkingBudgets { low: Some(4000), medium: Some(8000), xhigh: Some(16000) }
    );
    let partial = ThinkingBudgets::parse("xhigh=12000").unwrap();
    assert_eq!(partial.at(Some("low")), None);
    assert_eq!(partial.at(None), Some(12000), "the template's default effort is xhigh");
    assert_eq!(map.at(Some("medium")), Some(8000));
    assert!(ThinkingBudgets::parse("max=1").is_err());
    assert!(ThinkingBudgets::parse("low=many").is_err());
    assert!(ThinkingBudgets::parse("low").is_err());
}

#[test]
fn a_tool_turn_scales_the_default_budget() {
    let d = ThinkingDefaults {
        budgets: ThinkingBudgets::parse("medium=8000").unwrap(),
        tool_turn_factor: 0.5,
        ..ThinkingDefaults::default()
    };
    assert_eq!(d.budget(Some("medium"), false), Some(8000));
    assert_eq!(d.budget(Some("medium"), true), Some(4000));
    assert_eq!(d.budget(Some("low"), true), None);
    assert_eq!(ThinkingDefaults::default().budget(None, false), None, "off by default");
}

#[test]
fn thinking_fields_override_the_defaults_and_kwargs_win() {
    let defaults = ThinkingDefaults {
        budgets: ThinkingBudgets::parse("8000").unwrap(),
        nudges: true,
        tool_call_ends_thinking: true,
        grace: 64,
        ..ThinkingDefaults::default()
    };
    // Nothing asked: the server's defaults.
    let s = ThinkingFields::default().resolve(&defaults, Some(8000), 5);
    assert_eq!(
        s,
        ThinkingSettings {
            budget: Some(8000),
            grace: 64,
            nudges: true,
            tool_call_ends_thinking: true,
            seed: 5
        }
    );
    // A negative budget turns the default off, and the nudges with it.
    let off = ThinkingFields { thinking_budget: Some(-1), ..Default::default() };
    let s = off.resolve(&defaults, Some(8000), 5);
    assert_eq!((s.budget, s.nudges), (None, false));
    // The request's own budget wins over the default.
    let own = ThinkingFields {
        thinking_budget: Some(3000),
        thinking_nudges: Some(false),
        tool_call_ends_thinking: Some(false),
    };
    let s = own.resolve(&defaults, Some(8000), 5);
    assert_eq!(
        (s.budget, s.nudges, s.tool_call_ends_thinking),
        (Some(3000), false, false)
    );
    // chat_template_kwargs win over the top-level fields.
    let kwargs =
        serde_json::json!({"thinking_budget": 100, "tool_call_ends_thinking": true});
    let merged = ThinkingFields::from_kwargs(kwargs.as_object().unwrap())
        .unwrap()
        .or(own.clone());
    assert_eq!(merged.thinking_budget, Some(100));
    assert_eq!(merged.tool_call_ends_thinking, Some(true));
    assert_eq!(merged.thinking_nudges, Some(false));
    let bad = serde_json::json!({"thinking_nudges": "yes"});
    assert!(ThinkingFields::from_kwargs(bad.as_object().unwrap()).is_err());
    // Off by default: no control at all.
    let none = ThinkingFields::default().resolve(&ThinkingDefaults::default(), None, 5);
    assert!(!none.any());
}

#[test]
fn thinking_fields_deserialize_from_a_chat_request() {
    let request: ChatRequest = serde_json::from_value(serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "thinking_budget": 8000,
        "thinking_nudges": true,
        "tool_call_ends_thinking": true,
        "temperature": 0.6,
    }))
    .unwrap();
    assert_eq!(request.thinking_controls.thinking_budget, Some(8000));
    assert_eq!(request.thinking_controls.thinking_nudges, Some(true));
    assert_eq!(request.thinking_controls.tool_call_ends_thinking, Some(true));
    assert_eq!(request.sampling.temperature, Some(0.6), "both flattened groups");
}

/// A budget of 0 would close the block at the first line end: refused, in
/// the request, in `chat_template_kwargs` and in the server's flag.
#[test]
fn a_thinking_budget_of_zero_is_refused() {
    let zero = ThinkingFields { thinking_budget: Some(0), ..Default::default() };
    assert!(zero.check("").unwrap_err().to_string().contains("positive"));
    let negative = ThinkingFields { thinking_budget: Some(-1), ..Default::default() };
    assert!(negative.check("").is_ok(), "negative turns a default off");
    let kwargs = serde_json::json!({"thinking_budget": 0});
    let err = ThinkingFields::from_kwargs(kwargs.as_object().unwrap()).unwrap_err();
    assert!(err.to_string().contains("chat_template_kwargs.thinking_budget"), "{err}");
    assert!(ThinkingBudgets::parse("0").is_err());
    assert!(ThinkingBudgets::parse("low=0,medium=8000").is_err());
    assert!(ThinkingBudgets::parse("low=1").is_ok());
}
