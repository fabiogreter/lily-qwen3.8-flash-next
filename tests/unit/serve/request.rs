use std::time::Duration;

use serde_json::{Value, json};

use super::*;
use crate::serve::Collected;
use crate::serve::timings::{
    EvictionPhase, EvictionTimings, MemoryStats, NgramStats, PrefillParts,
    PrefillPhases, Timings,
};
use crate::serve::tools::ParsedToolCall;
use crate::stats::{Gather, Prefill as PrefillCounters};

// The functions below were written out in `Engine::run` and in the batch
// scheduler before they were shared. Each `before_*` is that code as it
// stood (6bb3057), only with its inputs as parameters, so these tests pin
// the timings, the log lines and the response bodies to what the server
// produced then.

fn counters(chunks: u64, secs: f64, batches: u64, pages: u64) -> Counters {
    Counters {
        gather: Gather {
            batches,
            rows: batches * 16,
            checked_rows: batches,
            pages,
            cold_pages: pages / 3,
            hinted_rows: 0,
            secs: secs / 4.0,
            hidden_secs: secs / 8.0,
        },
        prefill: PrefillCounters {
            chunks,
            alloc_secs: secs / 10.0,
            ngram_secs: secs / 5.0,
            encode_secs: secs / 20.0,
            gpu_secs: secs / 2.0,
            wait_secs: secs / 40.0,
        },
    }
}

fn vm(base: u64) -> Option<VmCounters> {
    Some(VmCounters {
        pageins: base,
        pageouts: base / 2,
        swapins: base * 3,
        swapouts: base * 4,
        compressions: base * 100,
        decompressions: base * 50,
    })
}

/// A request with every optional part present: cache reuse from disk, an
/// agreement past it, a durable prefix, images, evictions, pinned.
fn busy() -> Facts {
    Facts {
        n: 31_200,
        reused: 9_000,
        agreement: 15_649,
        cut_back: None,
        forked: true,
        from_disk: Some(Duration::from_millis(1_234)),
        durable: Some((15_600, 0.25)),
        durable_secs: 0.31,
        images: 3,
        images_decoded: 1,
        prepare_secs: 0.0123,
        image_tokens: 2_048,
        vision_secs: 0.42,
        encoded_images: 2,
        queued_secs: 1.5,
        pinned: true,
        session_secs: 0.6,
        checkpoint_secs: 0.021,
        prefix_secs: 12.5,
        acquire_evictions: Evictions {
            evicted: 2,
            written_ahead: 1,
            spilled: 1,
            spill_secs: 0.4,
            waited_secs: 0.05,
            cancelled: 0,
        },
        counters_start: counters(3, 1.0, 10, 30),
        counters_prefill: counters(9, 11.0, 16, 99),
        vm_start: vm(1_000),
        vm_prefill: vm(5_000),
    }
}

/// A short text request: nothing cached, nothing optional.
fn plain() -> Facts {
    Facts {
        n: 40,
        prefix_secs: 0.05,
        session_secs: 0.001,
        checkpoint_secs: 0.002,
        queued_secs: 0.0001,
        counters_prefill: counters(1, 0.04, 1, 3),
        vm_start: None,
        vm_prefill: None,
        ..Facts::default()
    }
}

fn batch_stats() -> BatchStats {
    let mut stats = BatchStats {
        solo_tokens: 90,
        preemptions: 1,
        interleaved_steps: 16,
        interleaved_secs: 0.37,
        ..BatchStats::default()
    };
    stats.record_step(2, &[2, 3]);
    stats.record_step(2, &[2, 3, 4]);
    stats
}

/// `Engine::run`'s and `try_finish`'s timings (the latter adds `batch`).
#[allow(clippy::too_many_arguments)]
fn before_answer_timings(
    f: &Facts,
    completion_tokens: usize,
    decode_secs: f64,
    speculation: Option<Speculation>,
    decode_checkpoints_taken: usize,
    decode_checkpoint_secs: f64,
    cancelled_by: Option<&'static str>,
    batch: Option<&BatchStats>,
    released: &Evictions,
    counters_end: Counters,
    vm_end: Option<VmCounters>,
    pressure: Option<u32>,
    task: Option<TaskMemory>,
) -> Timings {
    let measured = Timings::measure(
        f.n,
        f.reused,
        f.prefix_secs,
        completion_tokens,
        decode_secs,
        speculation,
    )
    .with_agreement(f.agreement, f.durable.map(|(b, _)| b))
    .with_vision(f.image_tokens, f.vision_secs)
    .with_decode_checkpoints(decode_checkpoints_taken, decode_checkpoint_secs)
    .with_pinned(f.pinned);
    let measured = match batch {
        Some(stats) => measured.with_batch(stats.timings()),
        None => measured,
    };
    measured
        .with_cancel(cancelled_by, None)
        .with_evictions(EvictionTimings {
            acquire: EvictionPhase::new(&f.acquire_evictions),
            release: EvictionPhase::new(released),
        })
        .with_diagnostics(
            f.queued_secs,
            PrefillPhases::split(
                f.prefix_secs,
                if f.images == 0 { 0.0 } else { f.vision_secs },
                PrefillParts {
                    session_secs: f.session_secs,
                    durable_secs: f.durable_secs,
                    checkpoint_secs: f.checkpoint_secs,
                    counters: f.counters_prefill.since(f.counters_start).prefill,
                },
            ),
            NgramStats {
                prefill: f.counters_prefill.since(f.counters_start).gather.into(),
                decode: counters_end.since(f.counters_prefill).gather.into(),
            },
            MemoryStats::from_samples(
                [f.vm_start, f.vm_prefill, vm_end],
                pressure,
                task,
            ),
        )
}

/// The cancelled prefill's timings in both places (the batch scheduler's
/// adds `batch`); `f.prefix_secs` is the time to the stop, and no
/// checkpoint or prefill-end sample exists.
#[allow(clippy::too_many_arguments)]
fn before_stopped_timings(
    f: &Facts,
    by: &'static str,
    at: usize,
    batch: Option<&BatchStats>,
    released: &Evictions,
    counters_end: Counters,
    vm_end: Option<VmCounters>,
    pressure: Option<u32>,
    task: Option<TaskMemory>,
) -> Timings {
    let (n, reused, prefix_secs) = (f.n, f.reused, f.prefix_secs);
    let measured = Timings::measure(n, reused, prefix_secs, 0, 0.0, None)
        .with_agreement(f.agreement, f.durable.map(|(b, _)| b))
        .with_vision(f.image_tokens, f.vision_secs)
        .with_pinned(f.pinned);
    let measured = match batch {
        Some(stats) => measured.with_batch(stats.timings()),
        None => measured,
    };
    measured
        .with_cancel(Some(by), Some(at))
        .with_evictions(EvictionTimings {
            acquire: EvictionPhase::new(&f.acquire_evictions),
            release: EvictionPhase::new(released),
        })
        .with_diagnostics(
            f.queued_secs,
            PrefillPhases::split(
                prefix_secs,
                if f.images == 0 { 0.0 } else { f.vision_secs },
                PrefillParts {
                    session_secs: f.session_secs,
                    durable_secs: f.durable_secs,
                    checkpoint_secs: 0.0,
                    counters: counters_end.since(f.counters_start).prefill,
                },
            ),
            NgramStats {
                prefill: counters_end.since(f.counters_start).gather.into(),
                decode: counters_end.since(counters_end).gather.into(),
            },
            MemoryStats::from_samples([f.vm_start, vm_end, vm_end], pressure, task),
        )
}

/// Both answer log lines as they were written out; `batch` is the batch
/// scheduler's extra group (`None`: `Engine::run`'s line, which has none).
#[allow(clippy::too_many_arguments)]
fn before_answer_line(
    id: &str,
    f: &Facts,
    completion_tokens: usize,
    decode_secs: f64,
    drafted: usize,
    accepted: usize,
    decode_checkpoints_taken: usize,
    decode_checkpoint_secs: f64,
    finish_reason: &str,
    cancelled_by: Option<&str>,
    batch: Option<&BatchStats>,
    sessions: (usize, usize, usize, Option<(usize, u64)>),
    details: &str,
) -> String {
    let (n, reused, agreement) = (f.n, f.reused, f.agreement);
    let (image_tokens, vision_secs, encoded_images) =
        (f.image_tokens, f.vision_secs, f.encoded_images);
    let images = f.images;
    let disk = sessions
        .3
        .map(|(len, used)| format!(", disk {} ({:.1} GB)", len, used as f64 / 1e9));
    let head = format!(
        "{}: {} prompt tokens ({} cached{}{}{}{}){}, {} generated, prefix {:.2}s, decode {:.2}s ({:.1} tok/s){}, finish={finish_reason}{}",
        id,
        n,
        reused,
        describe_reuse(f.cut_back, f.forked),
        f.from_disk
            .map(|d| format!(", from disk in {:.2}s", d.as_secs_f64()))
            .unwrap_or_default(),
        if agreement > reused {
            format!(", agreement {agreement}")
        } else {
            String::new()
        },
        f.durable
            .map(|(b, secs)| format!(", durable prefix {b} written in {secs:.2}s"))
            .unwrap_or_default(),
        if images == 0 {
            String::new()
        } else {
            format!(
                ", images {} ({image_tokens} tokens{}{}, prepared in {:.3}s), tower {vision_secs:.2}s",
                images,
                if encoded_images < images {
                    format!(", {encoded_images} encoded")
                } else {
                    String::new()
                },
                if f.images_decoded < images {
                    format!(", {} decoded", f.images_decoded)
                } else {
                    String::new()
                },
                f.prepare_secs
            )
        },
        completion_tokens,
        f.prefix_secs,
        decode_secs,
        completion_tokens as f64 / decode_secs.max(1e-9),
        format_args!(
            "{}{}",
            if drafted > 0 {
                format!(", drafts {}/{} accepted", accepted, drafted)
            } else {
                String::new()
            },
            if decode_checkpoints_taken > 0 {
                format!(
                    ", {decode_checkpoints_taken} decode checkpoints in {decode_checkpoint_secs:.2}s"
                )
            } else {
                String::new()
            }
        ),
        cancelled_by
            .map(|by| format!(" (cancelled by the {by} during the decode)"))
            .unwrap_or_default(),
    );
    let tail = format!(
        ", sessions={} ({:.1}/{:.1} GB){}{}",
        sessions.0,
        sessions.1 as f64 / 1e9,
        sessions.2 as f64 / 1e9,
        disk.unwrap_or_default(),
        details,
    );
    match batch {
        None => format!("{head}{tail}"),
        Some(stats) => format!("{head}{}{tail}", stats.describe(completion_tokens)),
    }
}

/// The cancelled prefill's log line, the same in both places.
fn before_stopped_line(
    id: &str,
    f: &Facts,
    by: &str,
    at: usize,
    sessions: (usize, usize, usize, Option<(usize, u64)>),
    details: &str,
) -> String {
    let (n, reused, prefix_secs) = (f.n, f.reused, f.prefix_secs);
    format!(
        "{id}: {n} prompt tokens ({reused} cached{}{}), cancelled by the {by} at {at} after {} prefilled in {prefix_secs:.2}s, kept {at} tokens as a session, sessions={} ({:.1}/{:.1} GB){}{}",
        describe_reuse(f.cut_back, f.forked),
        f.from_disk
            .map(|d| format!(", from disk in {:.2}s", d.as_secs_f64()))
            .unwrap_or_default(),
        at - reused,
        sessions.0,
        sessions.1 as f64 / 1e9,
        sessions.2 as f64 / 1e9,
        sessions
            .3
            .map(|(len, used)| format!(", disk {} ({:.1} GB)", len, used as f64 / 1e9))
            .unwrap_or_default(),
        details,
    )
}

fn store(sessions: (usize, usize, usize, Option<(usize, u64)>)) -> String {
    store_line(sessions.0, sessions.1, sessions.2, sessions.3)
}

const RELEASED: Evictions = Evictions {
    evicted: 1,
    written_ahead: 0,
    spilled: 1,
    spill_secs: 0.9,
    waited_secs: 0.0,
    cancelled: 1,
};

const TASK: Option<TaskMemory> =
    Some(TaskMemory { phys_footprint: 97_000_000_000, compressed: 1_200_000_000 });

#[test]
fn an_answers_timings_are_what_both_paths_measured_before() {
    let spec = Some(Speculation { drafted: 300, accepted: 190 });
    for (f, speculation, batch, by) in [
        (busy(), spec, None, None),
        (busy(), spec, Some(batch_stats()), Some("client")),
        (plain(), None, None, Some("shutdown")),
        (
            plain(),
            Some(Speculation { drafted: 0, accepted: 0 }),
            Some(batch_stats()),
            None,
        ),
    ] {
        let end = counters(f.counters_prefill.prefill.chunks, 30.0, 400, 800);
        let vm_end = vm(9_000);
        let outcome = Outcome {
            generated: 400,
            decode_secs: 4.2,
            speculation,
            decode_checkpoints: 2,
            decode_checkpoint_secs: 0.23,
            cancelled_by: by,
            cancelled_at: None,
            batch: batch.as_ref(),
        };
        let now = timings(&f, &outcome, &RELEASED, end, vm_end, Some(2), TASK);
        let before = before_answer_timings(
            &f,
            400,
            4.2,
            speculation,
            2,
            0.23,
            by,
            batch.as_ref(),
            &RELEASED,
            end,
            vm_end,
            Some(2),
            TASK,
        );
        assert_eq!(now, before);
        assert_eq!(
            serde_json::to_string(&now).unwrap(),
            serde_json::to_string(&before).unwrap()
        );
        assert_eq!(now.batch.is_some(), batch.is_some());
    }
}

#[test]
fn a_stopped_prefills_timings_are_what_both_paths_measured_before() {
    for (mut f, batch, by, at) in [
        (busy(), None, "client", 13_096),
        (busy(), Some(batch_stats()), "shutdown", 9_000),
        (plain(), None, "shutdown", 0),
    ] {
        // What admission records for a stopped prefill: no checkpoint, and
        // the stop's samples as the end of the prefill.
        let end = counters(f.counters_prefill.prefill.chunks + 2, 14.0, 20, 140);
        let vm_end = vm(6_000);
        f.checkpoint_secs = 0.0;
        f.counters_prefill = end;
        f.vm_prefill = vm_end;
        let outcome = Outcome {
            cancelled_by: Some(by),
            cancelled_at: Some(at),
            batch: batch.as_ref(),
            ..Outcome::default()
        };
        let now = timings(&f, &outcome, &RELEASED, end, vm_end, None, None);
        let before = before_stopped_timings(
            &f,
            by,
            at,
            batch.as_ref(),
            &RELEASED,
            end,
            vm_end,
            None,
            None,
        );
        assert_eq!(now, before);
        assert_eq!(
            serde_json::to_string(&now).unwrap(),
            serde_json::to_string(&before).unwrap()
        );
    }
}

#[test]
fn the_answer_line_is_what_both_paths_printed_before() {
    let resident = (3, 12_345_678_901, 40_000_000_000, None);
    let with_disk = (5, 30_000_000_000, 40_000_000_000, Some((17, 7_512_000_000)));
    let stats = batch_stats();
    for (f, drafted, ckpts, by, batch, sessions, details) in [
        (busy(), 300, 2, Some("client"), None, with_disk, "; queued 1.50s"),
        (busy(), 300, 0, None, Some(&stats), resident, ""),
        (plain(), 0, 0, None, None, resident, ""),
        (plain(), 0, 1, Some("shutdown"), Some(&stats), with_disk, ""),
    ] {
        let speculation =
            (drafted > 0).then_some(Speculation { drafted, accepted: 190 });
        let outcome = Outcome {
            generated: 412,
            decode_secs: 4.2,
            speculation,
            decode_checkpoints: ckpts,
            decode_checkpoint_secs: 0.23,
            cancelled_by: by,
            cancelled_at: None,
            batch,
        };
        let now = answer_line(
            "chatcmpl-1-7",
            &f,
            &outcome,
            "stop",
            &store(sessions),
            details,
        );
        let before = before_answer_line(
            "chatcmpl-1-7",
            &f,
            412,
            4.2,
            drafted,
            190,
            ckpts,
            0.23,
            "stop",
            by,
            batch,
            sessions,
            details,
        );
        assert_eq!(now, before);
    }
}

#[test]
fn the_answer_line_reads_as_it_always_did() {
    let outcome = Outcome {
        generated: 412,
        decode_secs: 4.0,
        speculation: Some(Speculation { drafted: 300, accepted: 190 }),
        decode_checkpoints: 2,
        decode_checkpoint_secs: 0.23,
        cancelled_by: Some("client"),
        cancelled_at: None,
        batch: None,
    };
    let store = store((5, 30_000_000_000, 40_000_000_000, Some((17, 7_512_000_000))));
    assert_eq!(
        answer_line(
            "chatcmpl-1-7",
            &busy(),
            &outcome,
            "tool_calls",
            &store,
            "; queued 1.50s"
        ),
        "chatcmpl-1-7: 31200 prompt tokens (9000 cached, forked, from disk in 1.23s, agreement 15649, \
         durable prefix 15600 written in 0.25s), images 3 (2048 tokens, 2 encoded, 1 decoded, \
         prepared in 0.012s), tower 0.42s, 412 generated, prefix 12.50s, decode 4.00s (103.0 tok/s), \
         drafts 190/300 accepted, 2 decode checkpoints in 0.23s, finish=tool_calls (cancelled by the \
         client during the decode), sessions=5 (30.0/40.0 GB), disk 17 (7.5 GB); queued 1.50s"
    );
    let stats = batch_stats();
    let lone = Outcome {
        speculation: None,
        decode_checkpoints: 0,
        cancelled_by: None,
        ..outcome
    };
    let store = store_line(1, 2_000_000_000, 40_000_000_000, None);
    assert_eq!(
        answer_line("cmpl-9-1", &plain(), &lone, "length", &store, ""),
        "cmpl-9-1: 40 prompt tokens (0 cached), 412 generated, prefix 0.05s, decode 4.00s (103.0 tok/s), \
         finish=length, sessions=1 (2.0/40.0 GB)"
    );
    let shared = Outcome { batch: Some(&stats), ..lone };
    assert_eq!(
        answer_line("cmpl-9-1", &plain(), &shared, "stop", &store, ""),
        "cmpl-9-1: 40 prompt tokens (0 cached), 412 generated, prefix 0.05s, decode 4.00s (103.0 tok/s), \
         finish=stop, batched: 2/412 tokens in steps of up to 3 rows (mean 2.50) shared with 2 other \
         requests, lone decode stopped 1 time for a new request, prefill shared with 16 decode steps \
         (0.37s), sessions=1 (2.0/40.0 GB)"
    );
}

#[test]
fn the_stopped_line_is_what_both_paths_printed_before() {
    let with_disk = (5, 30_000_000_000, 40_000_000_000, Some((17, 7_512_000_000)));
    let mut f = busy();
    f.prefix_secs = 4.1;
    let now = stopped_line("chatcmpl-1-8", &f, "client", 13_096, &store(with_disk), "");
    assert_eq!(
        now,
        before_stopped_line("chatcmpl-1-8", &f, "client", 13_096, with_disk, "")
    );
    assert_eq!(
        now,
        "chatcmpl-1-8: 31200 prompt tokens (9000 cached, forked, from disk in 1.23s), cancelled by the \
         client at 13096 after 4096 prefilled in 4.10s, kept 13096 tokens as a session, sessions=5 \
         (30.0/40.0 GB), disk 17 (7.5 GB)"
    );
    let resident = (0, 0, 8_000_000_000, None);
    let f = Facts { cut_back: Some(120), ..plain() };
    let details = "; queued 2.00s";
    assert_eq!(
        stopped_line("cmpl-1-1", &f, "shutdown", 0, &store(resident), details),
        before_stopped_line("cmpl-1-1", &f, "shutdown", 0, resident, details),
    );
}

fn closing(kind: Kind, finish_reason: &str) -> Closing<'_> {
    Closing {
        kind,
        id: "chatcmpl-5-3",
        created: 1_700_000_000,
        model: "qwen3.8-flash-next",
        finish_reason,
        usage: json!({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}),
        timings: json!({"prefill_ms": 1.5}),
    }
}

fn text(events: &[Value]) -> Vec<String> {
    events.iter().map(|e| serde_json::to_string(e).unwrap()).collect()
}

#[test]
fn a_streamed_answer_ends_with_the_chunks_it_always_sent() {
    // `timings` rides the finish chunk without usage ...
    assert_eq!(
        text(&stream_end(closing(Kind::Chat, "stop"), false)),
        [
            r#"{"id":"chatcmpl-5-3","object":"chat.completion.chunk","created":1700000000,"model":"qwen3.8-flash-next","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"timings":{"prefill_ms":1.5}}"#
        ]
    );
    // ... and the usage chunk after it with usage.
    assert_eq!(
        text(&stream_end(closing(Kind::Chat, "tool_calls"), true)),
        [
            r#"{"id":"chatcmpl-5-3","object":"chat.completion.chunk","created":1700000000,"model":"qwen3.8-flash-next","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"{"id":"chatcmpl-5-3","object":"chat.completion.chunk","created":1700000000,"model":"qwen3.8-flash-next","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12},"timings":{"prefill_ms":1.5}}"#,
        ]
    );
    assert_eq!(
        text(&stream_end(closing(Kind::Completion, "length"), true)),
        [
            r#"{"id":"chatcmpl-5-3","object":"text_completion","created":1700000000,"model":"qwen3.8-flash-next","choices":[{"index":0,"text":"","finish_reason":"length","logprobs":null}]}"#,
            r#"{"id":"chatcmpl-5-3","object":"text_completion","created":1700000000,"model":"qwen3.8-flash-next","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12},"timings":{"prefill_ms":1.5}}"#,
        ]
    );
}

#[test]
fn a_collected_answer_has_the_body_it_always_had() {
    let call = |name: &str| ParsedToolCall {
        name: name.to_owned(),
        arguments: r#"{"path":"a.rs"}"#.to_owned(),
    };
    // Tool calls without text: `content` is null, ids follow the request's.
    let collected = Collected {
        reasoning: "look first".to_owned(),
        content: String::new(),
        tool_calls: vec![call("read"), call("grep")],
    };
    let body = response_body(closing(Kind::Chat, "tool_calls"), collected);
    assert_eq!(
        serde_json::to_string(&body).unwrap(),
        format!(
            r#"{{"id":"chatcmpl-5-3","object":"chat.completion","created":1700000000,"model":"qwen3.8-flash-next","choices":[{{"index":0,"message":{{"role":"assistant","content":null,"reasoning_content":"look first","tool_calls":[{{"id":"{}","type":"function","function":{{"name":"read","arguments":"{{\"path\":\"a.rs\"}}"}}}},{{"id":"{}","type":"function","function":{{"name":"grep","arguments":"{{\"path\":\"a.rs\"}}"}}}}]}},"finish_reason":"tool_calls"}}],"usage":{{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}},"timings":{{"prefill_ms":1.5}}}}"#,
            call_id("chatcmpl-5-3", 0),
            call_id("chatcmpl-5-3", 1),
        )
    );
    // Plain text: no reasoning or tool call fields at all.
    let collected = Collected { content: "hello".to_owned(), ..Collected::default() };
    let body = response_body(closing(Kind::Chat, "stop"), collected);
    assert_eq!(
        serde_json::to_string(&body).unwrap(),
        r#"{"id":"chatcmpl-5-3","object":"chat.completion","created":1700000000,"model":"qwen3.8-flash-next","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12},"timings":{"prefill_ms":1.5}}"#
    );
    let collected = Collected { content: "fn main".to_owned(), ..Collected::default() };
    let body = response_body(closing(Kind::Completion, "length"), collected);
    assert_eq!(
        serde_json::to_string(&body).unwrap(),
        r#"{"id":"chatcmpl-5-3","object":"text_completion","created":1700000000,"model":"qwen3.8-flash-next","choices":[{"index":0,"text":"fn main","finish_reason":"length","logprobs":null}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12},"timings":{"prefill_ms":1.5}}"#
    );
}

#[test]
fn the_store_summary_names_the_disk_tier_only_when_there_is_one() {
    assert_eq!(store_line(0, 0, 8_000_000_000, None), "sessions=0 (0.0/8.0 GB)");
    assert_eq!(
        store_line(4, 21_460_000_000, 64_000_000_000, Some((211, 499_960_000_000))),
        "sessions=4 (21.5/64.0 GB), disk 211 (500.0 GB)"
    );
}
