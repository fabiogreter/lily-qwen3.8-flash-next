use super::*;
use crate::serve::dashboard::contract::RequestRow;
use crate::serve::recorder::{Writer, in_memory};

/// 2025-10-10 14:00:00 UTC, on every bucket edge (a whole hour).
const EDGE: u64 = 1_760_104_800_000;
const MIN: u64 = MINUTE_MS;

fn minute(decode_tokens: f64, decode_busy_ms: f64) -> Minute {
    Minute {
        requests: 1,
        decode_tokens,
        prefill_tokens: 0.0,
        decode_busy_ms,
        prefill_busy_ms: 0.0,
        busy_ms: decode_busy_ms,
        prompt_tokens: 1000,
        cached_tokens: 900,
        drafted_tokens: 20,
        accepted_tokens: 15,
        sessions_max: 1,
    }
}

#[test]
fn every_range_is_its_complete_grid_up_to_the_last_complete_minute() {
    for (range, buckets, step) in [
        (Range::QuarterHour, 15, 60_000),
        (Range::Hour, 60, 60_000),
        (Range::Day, 96, 900_000),
        (Range::Week, 168, 3_600_000),
    ] {
        let name = serde_json::to_value(range).unwrap();
        assert_eq!(
            Range::parse(name.as_str().unwrap()),
            Some(range),
            "the contract's name"
        );
        // On a bucket edge, just past it, in the middle of the bucket and
        // just before the next edge.
        for now in [EDGE, EDGE + 1, EDGE + step / 2 + 30_000, EDGE + step - 1] {
            let h = build(range, now, &[], None, &[]);
            assert_eq!((h.points.len(), h.step_s), (buckets, step / 1000), "{range:?}");
            let ends: Vec<u64> = h.points.iter().map(|p| p.t).collect();
            assert!(
                ends.windows(2).all(|w| w[1] - w[0] == step),
                "contiguous, oldest first"
            );
            // Bucket edges in UTC: a multiple of the step since the epoch.
            assert!(ends.iter().all(|t| t % step == 0));
            let data_end = now / MIN * MIN;
            let last = *ends.last().unwrap();
            // The last bucket holds the last complete minute.
            assert!(last >= data_end && last - step < data_end, "{range:?} at {now}");
            let grid = Grid::at(range, now);
            assert_eq!(grid.data_end, data_end);
            assert_eq!(grid.start, last - range.span_ms());
            assert_eq!(grid.covered(buckets - 1), data_end - (last - step));
        }
        // On the edge itself the last bucket is the one that just ended.
        assert_eq!(Grid::at(range, EDGE).end(), EDGE);
        assert_eq!(Grid::at(range, EDGE).covered(buckets - 1), step);
    }
    // 15m and 1h end at the last complete minute; 24h and 7d at the end of
    // the bucket under way, after now.
    let mid = EDGE + 7 * MIN + 30_000;
    assert_eq!(Grid::at(Range::QuarterHour, mid).end(), EDGE + 7 * MIN);
    assert_eq!(Grid::at(Range::Day, mid).end(), EDGE + 15 * MIN);
    assert_eq!(Grid::at(Range::Day, mid).covered(95), 7 * MIN);
    assert_eq!(Range::parse("2h"), None);
    assert_eq!(Range::parse(""), None);
}

#[test]
fn a_partial_last_bucket_shares_its_busy_time_over_what_it_covers() {
    // 7 minutes into a 24h bucket: 60 s busy in its third minute.
    let now = EDGE + 7 * MIN + 30_000;
    let minutes = [(EDGE / MIN + 2, minute(600.0, 60_000.0))];
    // A request in the minute under way is not data yet.
    let queue = [(EDGE + 2 * MIN, 5.0), (EDGE + 7 * MIN + 1_000, 999.0)];
    let h = build(Range::Day, now, &minutes, None, &queue);
    let p = h.points.last().unwrap();
    assert_eq!(p.t, EDGE + 15 * MIN, "its nominal end");
    assert_eq!(p.busy_share, round(1.0 / 7.0, 1e4));
    assert_eq!((p.queue_ms_p50, p.queue_ms_max), (Some(5.0), Some(5.0)));
    // The summary's wall time ends at the last complete minute too.
    let wall = (EDGE + 7 * MIN) - (EDGE + 15 * MIN - Range::Day.span_ms());
    assert_eq!(h.summary.busy_share, round(60_000.0 / wall as f64, 1e4));
}

#[test]
fn a_grid_without_data_is_all_null_with_zero_counts() {
    let h = build(Range::Hour, EDGE, &[], Some(&[]), &[]);
    for p in &h.points {
        let json = serde_json::to_value(p).unwrap();
        for (key, value) in json.as_object().unwrap() {
            match key.as_str() {
                "t" => {}
                "busy_share" | "requests" | "decode_tokens" | "prefill_tokens"
                | "sessions_max" => {
                    assert_eq!(value.as_f64(), Some(0.0), "{key}")
                }
                _ => assert!(value.is_null(), "{key} is {value}"),
            }
        }
    }
    let s = h.summary;
    assert_eq!((s.decode.avg, s.decode.max, s.prefill.avg), (None, None, None));
    assert_eq!((s.requests, s.tokens_generated, s.busy_share), (0, 0, 0.0));
    assert_eq!(
        (s.cache_hit, s.acceptance, s.power_avg_w, s.energy_wh),
        (None, None, None, None)
    );
}

#[test]
fn a_bucket_averages_over_busy_time_and_its_max_is_the_best_minute() {
    // Three minutes of one 15-minute bucket of the 24h range: 600 tokens in
    // 10 s, 200 in 10 s, 100 in 1 s.
    let first = (EDGE - 900_000) / MIN;
    let minutes = [
        (first, minute(600.0, 10_000.0)),
        (first + 1, minute(200.0, 10_000.0)),
        (first + 7, minute(100.0, 1000.0)),
    ];
    let h = build(Range::Day, EDGE, &minutes, None, &[]);
    let p = h.points.last().unwrap();
    // 900 tokens over 21 s of decoding, not over the 15 minutes.
    assert_eq!(p.decode_tps_avg, Some(42.86));
    assert_eq!(p.decode_tps_max, Some(100.0));
    assert_eq!((p.decode_tokens, p.requests, p.sessions_max), (900, 3, 1));
    assert_eq!(p.busy_share, round(21_000.0 / 900_000.0, 1e4));
    assert_eq!((p.cache_hit, p.acceptance), (Some(0.9), Some(0.75)));
    // Nothing prefilled: no prefill rate rather than zero.
    assert_eq!(
        (p.prefill_tps_avg, p.prefill_tps_max, p.prefill_tokens),
        (None, None, 0)
    );
    // The other buckets are empty.
    assert!(
        h.points[..95].iter().all(|p| p.decode_tps_avg.is_none() && p.requests == 0)
    );
    let s = &h.summary;
    assert_eq!((s.decode.avg, s.decode.max), (Some(42.86), Some(100.0)));
    assert_eq!((s.tokens_generated, s.requests), (900, 3));
    assert_eq!(s.busy_share, round(21_000.0 / 86_400_000.0, 1e4));
}

#[test]
fn the_summary_divides_by_the_whole_intervals_busy_time() {
    // Two 1-minute buckets of the 1h range: 50 tok/s and 150 tok/s, the
    // first busy ten times longer. The average weighs by busy time.
    let minutes = [
        (EDGE / MIN - 2, minute(500.0, 10_000.0)),
        (EDGE / MIN - 1, minute(150.0, 1000.0)),
    ];
    let h = build(Range::Hour, EDGE, &minutes, None, &[]);
    assert_eq!(h.summary.decode.avg, Some(59.09));
    assert_eq!(h.summary.decode.max, Some(150.0));
    assert_eq!(h.summary.tokens_generated, 650);
    // A minute outside the grid (under way, or older) counts nowhere.
    let outside = [(EDGE / MIN, minute(1e6, 1.0)), (EDGE / MIN - 61, minute(1e6, 1.0))];
    let h = build(Range::Hour, EDGE, &outside, None, &[]);
    assert_eq!((h.summary.decode.max, h.summary.requests), (None, 0));
}

#[test]
fn machine_series_are_minute_means_the_worst_pressure_and_the_energy() {
    let first = (EDGE - 900_000) / MIN;
    let machine = [
        MachineRow {
            minute: first,
            cpu_load: Some(0.2),
            pressure_level: Some(1),
            swap_bytes: Some(10),
            power_w: Some(30.0),
            ..MachineRow::default()
        },
        MachineRow {
            minute: first + 1,
            cpu_load: Some(0.4),
            pressure_level: Some(2),
            swap_bytes: Some(30),
            power_w: Some(90.0),
            ..MachineRow::default()
        },
    ];
    let h = build(Range::Day, EDGE, &[], Some(&machine), &[]);
    let p = h.points.last().unwrap();
    assert_eq!(p.cpu_load, Some(0.3));
    assert_eq!(p.memory_pressure, Some(MemoryPressure::Warning));
    assert_eq!((p.swap_bytes, p.power_w), (Some(30), Some(60.0)));
    assert_eq!((p.gpu_active, p.gpu_power_w, p.gpu_freq_mhz), (None, None, None));
    // Idle, the server up: machine values, no rates.
    assert_eq!((p.decode_tps_avg, p.requests, p.busy_share), (None, 0, 0.0));
    // Two minutes at 30 W and 90 W: 2 Wh over the interval, 60 W mean.
    assert_eq!((h.summary.power_avg_w, h.summary.energy_wh), (Some(60.0), Some(2.0)));
    // Buckets with no machine minute: null machine values.
    assert_eq!(h.points[0].cpu_load, None);
}

#[test]
fn queue_times_are_the_median_by_rank_and_the_longest_of_the_bucket() {
    let t = EDGE - 30_000;
    let queue = [(t, 4.0), (t, 1.0), (t, 100.0), (t, 2.0), (EDGE, 999.0)];
    let p = build(Range::Hour, EDGE, &[], None, &queue).points.pop().unwrap();
    assert_eq!((p.queue_ms_p50, p.queue_ms_max), (Some(2.0), Some(100.0)));
    assert_eq!(queue_figures(&mut []), (None, None));
    assert_eq!(queue_figures(&mut [3.0]), (Some(3.0), Some(3.0)));
}

fn row(t: u64, lineage: Option<u64>, generated: usize, decode_ms: f64) -> RequestRow {
    RequestRow {
        t,
        lineage,
        prompt_tokens: 1000,
        cached_tokens: 900,
        prefill_tokens: 100,
        prefill_ms: 50.0,
        prefill_tps: Some(2000.0),
        generated_tokens: generated,
        decode_ms,
        decode_tps: Some(generated as f64 / (decode_ms / 1e3)),
        drafted_tokens: Some(20),
        accepted_tokens: Some(15),
        queue_ms: 3.0,
        batch: 1,
        cancelled_by: None,
        thinking_forced: false,
        pinned: false,
        failed: false,
    }
}

#[test]
fn the_database_reads_as_the_global_and_the_per_session_history() {
    let mut w = Writer::new(in_memory().unwrap(), None, Default::default(), 0);
    let failed = RequestRow {
        failed: true,
        queue_ms: 500.0,
        prompt_tokens: 0,
        cached_tokens: 0,
        prefill_tokens: 0,
        drafted_tokens: None,
        accepted_tokens: None,
        ..row(EDGE - 10_000, Some(8), 0, 0.0)
    };
    assert!(w.record(vec![
        row(EDGE - 30_000, Some(7), 100, 2000.0),
        row(EDGE - 20_000, Some(8), 300, 2000.0),
        failed,
    ]));
    let mut m = crate::serve::recorder::MachineMinute::new(EDGE / MIN - 1);
    m.add(&crate::serve::recorder::MachineSample {
        cpu_load: Some(0.5),
        ..Default::default()
    });
    assert!(w.machine(&m));
    let conn = w.connection();
    let all =
        read(conn, Range::QuarterHour, EDGE, None, false).unwrap().expect("global");
    let p = all.points.last().unwrap();
    assert_eq!((p.requests, p.decode_tokens, p.sessions_max), (3, 400, 1));
    // 400 tokens over 4 s of decoding (the two did not overlap).
    assert_eq!(p.decode_tps_avg, Some(100.0));
    // The failed request has no queue time: 500 ms is not the maximum.
    assert_eq!(
        (p.cpu_load, p.queue_ms_p50, p.queue_ms_max),
        (Some(0.5), Some(3.0), Some(3.0))
    );

    let one = read(conn, Range::QuarterHour, EDGE, Some(8), false)
        .unwrap()
        .expect("lineage 8");
    let p = one.points.last().unwrap();
    assert_eq!((p.requests, p.decode_tokens, p.decode_tps_avg), (2, 300, Some(150.0)));
    assert_eq!(
        p.queue_ms_max,
        Some(3.0),
        "the session's queue skips the failed row too"
    );
    // A session's history has no machine series.
    assert!(
        one.points.iter().all(|p| p.cpu_load.is_none() && p.memory_pressure.is_none())
    );
    assert_eq!(one.summary.power_avg_w, None);

    // A lineage the database never saw: unknown, unless it is a cache
    // entry now, whose history is the empty grid.
    assert_eq!(read(conn, Range::Hour, EDGE, Some(99), false).unwrap(), None);
    let listed =
        read(conn, Range::Hour, EDGE, Some(99), true).unwrap().expect("listed");
    assert_eq!(listed.points.len(), 60);
    assert!(
        listed.points.iter().all(|p| p.requests == 0 && p.decode_tps_avg.is_none())
    );

    let t = session_totals(conn, 8).unwrap();
    assert_eq!(
        (t.requests, t.decode_tps_avg, t.decode_tps_max),
        (2, Some(150.0), Some(150.0))
    );
    assert_eq!((t.cache_hit, t.acceptance), (Some(0.9), Some(0.75)));
    assert_eq!(session_totals(conn, 99).unwrap(), SessionTotals::default());
}

#[test]
fn pruning_keeps_the_oldest_bucket_of_the_week_whole() {
    let mut w = Writer::new(in_memory().unwrap(), None, Default::default(), 0);
    // Mid-hour, so the 7d grid's last bucket is partial and its first one
    // starts the furthest back.
    let now = EDGE + 40 * MIN + 10_000;
    let grid = Grid::at(Range::Week, now);
    // Rows in the first minute of the oldest bucket and just before it.
    assert!(w.record(vec![
        row(grid.start + 30_000, Some(1), 10, 1000.0),
        row(grid.start - 30_000, Some(2), 10, 1000.0),
    ]));
    assert!(w.prune(now));
    let h = read(w.connection(), Range::Week, now, None, false).unwrap().unwrap();
    assert_eq!(h.points[0].requests, 1, "the oldest bucket kept its first minute");
    assert_eq!(h.summary.requests, 1);
    let kept: u64 = w
        .connection()
        .query_row("SELECT count(*) FROM requests", [], |r| r.get(0))
        .unwrap();
    assert_eq!(kept, 2, "pruning cuts an hour further back than the grid");
}
