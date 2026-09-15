use super::*;

use serde_json::{Value, json};

fn entry(id: &str) -> TimingsEntry {
    TimingsEntry { id: id.to_owned(), model: "m", created: 7, timings: Timings::measure(10, 0, 1.0, 5, 1.0, None) }
}

#[test]
fn rates_are_over_the_tokens_that_were_actually_computed() {
    let t = Timings::measure(24_538, 8_538, 16.0, 19, 0.19, None);
    // 16 000 tokens prefilled, not 24 538: the cached ones cost nothing.
    assert_eq!(t.prompt_tokens, 24_538);
    assert_eq!(t.cached_tokens, 8_538);
    assert_eq!(t.prefill_tokens, 16_000);
    assert_eq!(t.prefill_ms, 16_000.0);
    assert_eq!(t.prefill_per_second, Some(1_000.0));
    assert_eq!(t.generated_tokens, 19);
    assert_eq!(t.decode_ms, 190.0);
    assert_eq!(t.decode_per_second, Some(100.0));
}

#[test]
fn a_full_cache_hit_reports_the_time_and_no_prefill_rate() {
    // The session cache supplied everything but the last prompt token, which
    // the decode step feeds, so nothing is prefilled.
    let t = Timings::measure(4_096, 4_096, 0.004, 12, 0.12, None);
    assert_eq!(t.prefill_tokens, 0);
    assert_eq!(t.prefill_ms, 4.0);
    assert_eq!(t.prefill_per_second, None, "a rate over zero tokens divides by zero");
    assert_eq!(t.decode_per_second, Some(100.0));
    // Never negative, whatever the caller passes.
    assert_eq!(Timings::measure(10, 99, 0.0, 0, 0.0, None).prefill_tokens, 0);
}

#[test]
fn a_zero_duration_leaves_the_rate_null_rather_than_infinite() {
    let t = Timings::measure(10, 0, 0.0, 4, 0.0, None);
    assert_eq!(t.prefill_per_second, None);
    assert_eq!(t.decode_per_second, None);
    // serde_json turns a non-finite f64 into null; nothing here relies on that.
    assert_eq!(serde_json::to_value(t).unwrap()["prefill_per_second"], Value::Null);
}

#[test]
fn speculation_is_null_when_it_is_off_and_a_ratio_when_it_ran() {
    let off = Timings::measure(100, 0, 1.0, 10, 1.0, None);
    assert_eq!((off.drafted_tokens, off.accepted_tokens, off.acceptance_ratio), (None, None, None));
    let json = serde_json::to_value(off).unwrap();
    for field in ["drafted_tokens", "accepted_tokens", "acceptance_ratio"] {
        assert_eq!(json[field], Value::Null, "{field} must not read as a zero acceptance");
    }

    let on = Timings::measure(100, 0, 1.0, 10, 1.0, Some(Speculation { drafted: 14, accepted: 11 }));
    assert_eq!((on.drafted_tokens, on.accepted_tokens), (Some(14), Some(11)));
    assert_eq!(on.acceptance_ratio, Some(0.7857));

    // Speculation on but nothing proposed yet (a one-token answer): the
    // counts are real, the ratio would divide by zero.
    let idle = Timings::measure(100, 0, 1.0, 1, 0.01, Some(Speculation { drafted: 0, accepted: 0 }));
    assert_eq!((idle.drafted_tokens, idle.accepted_tokens, idle.acceptance_ratio), (Some(0), Some(0), None));
}

#[test]
fn the_json_shape_is_the_documented_one() {
    let t = Timings::measure(24_538, 0, 21.18, 19, 0.19, Some(Speculation { drafted: 14, accepted: 11 }));
    assert_eq!(
        serde_json::to_value(t).unwrap(),
        json!({
            "prompt_tokens": 24_538,
            "cached_tokens": 0,
            "prefill_tokens": 24_538,
            "prefill_ms": 21_180.0,
            "prefill_per_second": 1_158.55,
            "generated_tokens": 19,
            "decode_ms": 190.0,
            "decode_per_second": 100.0,
            "drafted_tokens": 14,
            "accepted_tokens": 11,
            "acceptance_ratio": 0.7857,
        })
    );
}

#[test]
fn the_log_is_bounded_and_newest_first() {
    let log = TimingsLog::new(3);
    assert!(log.recent().is_empty());
    for i in 0..5 {
        log.record(entry(&format!("id-{i}")));
    }
    let recent = log.recent();
    assert_eq!(recent.len(), 3, "the ring buffer keeps at most its capacity");
    assert_eq!(recent.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["id-4", "id-3", "id-2"]);
    // Reading does not consume.
    assert_eq!(log.recent().len(), 3);
    // A zero capacity would drop every entry; one is the floor.
    let tiny = TimingsLog::new(0);
    assert_eq!(tiny.capacity(), 1);
    tiny.record(entry("only"));
    tiny.record(entry("last"));
    assert_eq!(tiny.recent().iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["last"]);
}

#[test]
fn the_log_survives_a_poisoned_lock() {
    let log = std::sync::Arc::new(TimingsLog::new(4));
    log.record(entry("before"));
    let poisoner = log.clone();
    let _ = std::thread::spawn(move || {
        let _guard = poisoner.entries.lock().expect("fresh lock");
        panic!("poison the mutex");
    })
    .join();
    log.record(entry("after"));
    assert_eq!(log.recent().iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["after", "before"]);
}
