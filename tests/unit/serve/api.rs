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
