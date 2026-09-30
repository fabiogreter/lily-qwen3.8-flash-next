use super::*;

use serde_json::{Value, json};

fn entry(id: &str) -> TimingsEntry {
    TimingsEntry {
        id: id.to_owned(),
        model: "m",
        created: 7,
        timings: Timings::measure(10, 0, 1.0, 5, 1.0, None),
    }
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
    assert_eq!(
        (off.drafted_tokens, off.accepted_tokens, off.acceptance_ratio),
        (None, None, None)
    );
    let json = serde_json::to_value(off).unwrap();
    for field in ["drafted_tokens", "accepted_tokens", "acceptance_ratio"] {
        assert_eq!(
            json[field],
            Value::Null,
            "{field} must not read as a zero acceptance"
        );
    }

    let on = Timings::measure(
        100,
        0,
        1.0,
        10,
        1.0,
        Some(Speculation { drafted: 14, accepted: 11 }),
    );
    assert_eq!((on.drafted_tokens, on.accepted_tokens), (Some(14), Some(11)));
    assert_eq!(on.acceptance_ratio, Some(0.7857));

    // Speculation on but nothing proposed yet (a one-token answer): the
    // counts are real, the ratio would divide by zero.
    let idle = Timings::measure(
        100,
        0,
        1.0,
        1,
        0.01,
        Some(Speculation { drafted: 0, accepted: 0 }),
    );
    assert_eq!(
        (idle.drafted_tokens, idle.accepted_tokens, idle.acceptance_ratio),
        (Some(0), Some(0), None)
    );
}

#[test]
fn the_json_shape_is_the_documented_one() {
    let t = Timings::measure(
        24_538,
        0,
        21.18,
        19,
        0.19,
        Some(Speculation { drafted: 14, accepted: 11 }),
    );
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
            "agreement_tokens": 0,
            "queue_ms": 0.0,
            "prefill_phases": {
                "session_ms": 0.0,
                "alloc_ms": 0.0,
                "ngram_ms": 0.0,
                "encode_ms": 0.0,
                "gpu_ms": 0.0,
                "wait_ms": 0.0,
                "durable_ms": 0.0,
                "checkpoint_ms": 0.0,
                "other_ms": 0.0,
                "chunks": 0,
            },
            "ngram": {
                "prefill": {
                    "batches": 0,
                    "rows": 0,
                    "checked_rows": 0,
                    "pages": 0,
                    "cold_pages": 0,
                    "gather_ms": 0.0,
                },
                "decode": {
                    "batches": 0,
                    "rows": 0,
                    "checked_rows": 0,
                    "pages": 0,
                    "cold_pages": 0,
                    "gather_ms": 0.0,
                },
            },
            "memory": {
                "pressure_level": null,
                "pressure": null,
                "phys_footprint_bytes": null,
                "compressed_bytes": null,
                "prefill": null,
                "decode": null,
            },
        })
    );
}

#[test]
fn image_fields_are_absent_for_text_and_carry_the_tower_time_for_images() {
    let text = Timings::measure(1000, 300, 1.0, 5, 0.1, None).with_vision(0, 0.0);
    assert_eq!((text.image_tokens, text.vision_ms), (None, None));
    let json = serde_json::to_value(text).unwrap();
    assert!(json.get("image_tokens").is_none() && json.get("vision_ms").is_none());
    // The tower ran 0.7124 s over an image of 1 980 placeholders.
    let image = Timings::measure(2100, 0, 1.5, 5, 0.1, None).with_vision(1980, 0.7124);
    assert_eq!((image.image_tokens, image.vision_ms), (Some(1980), Some(712.4)));
    let json = serde_json::to_value(image).unwrap();
    assert_eq!(json["image_tokens"], json!(1980));
    assert_eq!(json["vision_ms"], json!(712.4));
    // A cached image still counts its tokens; the tower time is then zero.
    let cached = Timings::measure(2100, 2050, 0.2, 5, 0.1, None).with_vision(1980, 0.0);
    assert_eq!((cached.image_tokens, cached.vision_ms), (Some(1980), Some(0.0)));
}

#[test]
fn agreement_never_reads_below_the_cached_prefix_and_the_durable_field_is_absent_unless_written()
 {
    // Without the session cache's word, the agreement is what was reused.
    let plain = Timings::measure(1000, 300, 1.0, 5, 0.1, None);
    assert_eq!((plain.agreement_tokens, plain.durable_prefix_tokens), (300, None));
    let json = serde_json::to_value(plain).unwrap();
    assert_eq!(json["agreement_tokens"], json!(300));
    assert!(
        json.get("durable_prefix_tokens").is_none(),
        "absent, never a zero-length write"
    );

    // A run that shared 900 tokens, resumed none and wrote the entry there.
    let written =
        Timings::measure(1000, 0, 1.0, 5, 0.1, None).with_agreement(900, Some(900));
    assert_eq!(
        (written.agreement_tokens, written.durable_prefix_tokens),
        (900, Some(900))
    );
    assert_eq!(
        serde_json::to_value(written).unwrap()["durable_prefix_tokens"],
        json!(900)
    );

    // Clamped into [cached, prompt] whatever the caller passes.
    let t = Timings::measure(1000, 300, 1.0, 5, 0.1, None).with_agreement(10, None);
    assert_eq!(t.agreement_tokens, 300);
    assert_eq!(
        Timings::measure(1000, 300, 1.0, 5, 0.1, None)
            .with_agreement(5000, None)
            .agreement_tokens,
        1000
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
    assert_eq!(
        recent.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
        ["id-4", "id-3", "id-2"]
    );
    // Reading does not consume.
    assert_eq!(log.recent().len(), 3);
    // A zero capacity would drop every entry; one is the floor.
    let tiny = TimingsLog::new(0);
    assert_eq!(tiny.capacity(), 1);
    tiny.record(entry("only"));
    tiny.record(entry("last"));
    assert_eq!(
        tiny.recent().iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
        ["last"]
    );
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
    assert_eq!(
        log.recent().iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
        ["after", "before"]
    );
}

fn parts(session: f64, gpu: f64, alloc: f64) -> PrefillParts {
    PrefillParts {
        session_secs: session,
        durable_secs: 0.0,
        checkpoint_secs: 0.01,
        counters: stats::Prefill {
            chunks: 1,
            alloc_secs: alloc,
            ngram_secs: 0.002,
            encode_secs: 0.02,
            gpu_secs: gpu,
            wait_secs: 0.003,
        },
    }
}

#[test]
fn prefill_phases_add_up_to_the_prefill_with_the_rest_as_other() {
    let phases = PrefillPhases::split(0.5, 0.1, parts(0.05, 0.3, 0.001));
    assert_eq!(phases.session_ms, 50.0);
    assert_eq!(phases.gpu_ms, 300.0);
    assert_eq!(phases.chunks, 1);
    let sum = phases.session_ms
        + phases.alloc_ms
        + phases.ngram_ms
        + phases.encode_ms
        + phases.gpu_ms
        + phases.wait_ms
        + phases.durable_ms
        + phases.checkpoint_ms
        + phases.other_ms
        + 100.0; // the vision tower
    assert!((sum - 500.0).abs() < 1e-6, "phases sum to {sum}");
    assert!((phases.other_ms - 14.0).abs() < 1e-6, "{}", phases.other_ms);
    // Clocks that overlap never make the remainder negative.
    assert_eq!(PrefillPhases::split(0.1, 0.0, parts(0.05, 0.3, 0.0)).other_ms, 0.0);
}

#[test]
fn diagnostics_are_always_in_the_json() {
    let t = Timings::measure(100, 0, 1.0, 10, 1.0, None).with_diagnostics(
        0.25,
        PrefillPhases::split(1.0, 0.0, parts(0.01, 0.9, 0.0)),
        NgramStats {
            prefill: stats::Gather {
                batches: 1,
                rows: 1600,
                checked_rows: 100,
                pages: 320,
                cold_pages: 0,
                secs: 0.001,
            }
            .into(),
            decode: GatherStats::default(),
        },
        MemoryStats::from_samples(
            [Some(VmCounters::default()), Some(VmCounters::default()), None],
            Some(1),
            Some(stats::TaskMemory { phys_footprint: 80 << 30, compressed: 0 }),
        ),
    );
    let json = serde_json::to_value(t).unwrap();
    assert_eq!(json["queue_ms"], 250.0);
    assert_eq!(json["prefill_phases"]["gpu_ms"], 900.0);
    assert_eq!(json["ngram"]["prefill"]["rows"], 1600);
    assert_eq!(json["ngram"]["prefill"]["checked_rows"], 100);
    assert_eq!(json["ngram"]["decode"]["cold_pages"], 0);
    assert_eq!(json["memory"]["pressure"], "normal");
    assert_eq!(json["memory"]["prefill"]["swapins"], 0);
    assert_eq!(json["memory"]["decode"], Value::Null, "a failed sample is null");
    // Nothing stands out, so the log line gets nothing.
    assert_eq!(t.log_details(), "");
}

#[test]
fn the_log_line_shows_only_what_stands_out() {
    let base = Timings::measure(86_201, 85_939, 28.7, 12, 0.2, None);
    // A prefill that spent 28 s allocating: the phases appear, with the GPU
    // time next to them to show the stall was on the host.
    let slow = base.with_diagnostics(
        2.5,
        PrefillPhases::split(28.7, 0.0, parts(0.02, 0.21, 28.3)),
        NgramStats::default(),
        MemoryStats::default(),
    );
    let details = slow.log_details();
    assert!(
        details
            .starts_with("; queued 2.50s; prefill phases: session 0.02s, alloc 28.30s"),
        "{details}"
    );
    assert!(details.contains("gpu 0.21s"), "{details}");
    assert!(!details.contains("durable"), "no durable entry was written: {details}");
    assert!(
        !details.contains("ngram cold") && !details.contains("memory"),
        "{details}"
    );

    // Cold pages and swap traffic each bring their own group.
    let swapping = VmCounters { swapins: 5_000, pageins: 120, ..VmCounters::default() };
    let cold = base.with_diagnostics(
        0.0,
        PrefillPhases::split(0.3, 0.0, parts(0.01, 0.25, 0.0)),
        NgramStats {
            prefill: GatherStats {
                cold_pages: 120,
                pages: 840,
                ..GatherStats::default()
            },
            decode: GatherStats { pages: 576, ..GatherStats::default() },
        },
        MemoryStats::from_samples(
            [Some(VmCounters::default()), Some(swapping), Some(swapping)],
            Some(2),
            Some(stats::TaskMemory { phys_footprint: 80_100_000_000, compressed: 0 }),
        ),
    );
    let details = cold.log_details();
    assert!(!details.contains("prefill phases"), "{details}");
    assert!(details.contains("ngram cold pages: prefill 120/840 checked"), "{details}");
    assert!(
        details.contains("memory: pressure warning, footprint 80.1 GB"),
        "{details}"
    );
    assert!(
        details.contains("prefill pageins 120 pageouts 0 swapins 5000"),
        "{details}"
    );
    assert!(
        details.contains("decode pageins 0"),
        "the decode delta is its own: {details}"
    );
}
