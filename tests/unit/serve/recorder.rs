use std::fs;
use std::os::unix::fs::PermissionsExt as _;

use super::*;
use crate::serve::dashboard::contract::{DiskTier, GpuTier};
use crate::serve::session::Lineage;

/// A fresh directory under `$TMPDIR` for one test.
fn temp_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir()
        .join(format!("lily-recorder-test-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("temp dir");
    root
}

/// A row that finished at `t` after `prefill_ms` of prefill (100 tokens
/// computed of 1 000) and `decode_ms` of decode drawing `generated`.
fn row(
    t: u64,
    lineage: Option<u64>,
    prefill_ms: f64,
    generated: usize,
    decode_ms: f64,
) -> RequestRow {
    RequestRow {
        t,
        lineage,
        prompt_tokens: 1000,
        cached_tokens: 900,
        prefill_tokens: 100,
        prefill_ms,
        prefill_tps: Some(100.0 / (prefill_ms / 1e3)),
        generated_tokens: generated,
        decode_ms,
        decode_tps: (generated > 0).then(|| generated as f64 / (decode_ms / 1e3)),
        drafted_tokens: Some(20),
        accepted_tokens: Some(15),
        queue_ms: 2.0,
        batch: 1,
        cancelled_by: None,
        thinking_forced: false,
        pinned: false,
        failed: false,
    }
}

fn writer() -> Writer {
    Writer::new(in_memory().expect("in-memory database"), None, Arc::default(), 0)
}

fn on_disk(path: &Path) -> Writer {
    let (conn, _) = open(path, 1).expect("open");
    Writer::new(conn, Some(path.to_path_buf()), Arc::default(), 0)
}

fn count(conn: &Connection, table: &str) -> u64 {
    conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
        .expect("count")
}

fn pragma(conn: &Connection, name: &str) -> i64 {
    conn.query_row(&format!("PRAGMA {name}"), [], |r| r.get(0)).expect(name)
}

fn minute_row(
    conn: &Connection,
    minute: u64,
) -> Option<(u64, f64, f64, f64, f64, u64)> {
    conn.query_row(
        "SELECT requests, decode_tokens, decode_busy_ms, prefill_busy_ms, busy_ms, sessions_max
         FROM minute WHERE minute = ?1",
        [minute],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)),
    )
    .ok()
}

/// The `CREATE TABLE` statement of `table` in the schema.
fn create_table(table: &str) -> &'static str {
    let start = SCHEMA_V1.find(&format!("CREATE TABLE {table} (")).expect(table);
    let end = start + SCHEMA_V1[start..].find(");").expect("end") + 2;
    &SCHEMA_V1[start..end]
}

fn mode(path: &Path) -> u32 {
    fs::metadata(path).expect("metadata").permissions().mode() & 0o777
}

/// Waits up to 10 s for `done`.
fn eventually(mut done: impl FnMut() -> bool) -> bool {
    let until = Instant::now() + Duration::from_secs(10);
    while Instant::now() < until {
        if done() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

/// Unix minute 29 335 100 (2025-10-10 14:20 UTC): the tests' clock.
const M: u64 = 29_335_100;
const T0: u64 = M * MINUTE_MS;

// --- the schema ----------------------------------------------------------------

#[test]
fn an_empty_file_gets_the_schema_and_a_second_open_keeps_its_rows() {
    let root = temp_root("empty");
    let path = root.join("lily.sqlite");
    fs::write(&path, b"").expect("empty file");
    let (conn, aside) = open(&path, 1).expect("open");
    assert_eq!(aside, None, "an empty file is an empty database, not a corrupt one");
    assert_eq!(pragma(&conn, "user_version"), SCHEMA_VERSION);
    let mode: String = conn.query_row("PRAGMA journal_mode", [], |r| r.get(0)).unwrap();
    assert_eq!(mode, "wal");
    assert_eq!(pragma(&conn, "synchronous"), 1, "NORMAL");
    assert_eq!(pragma(&conn, "auto_vacuum"), 2, "INCREMENTAL");
    assert_eq!(pragma(&conn, "journal_size_limit"), JOURNAL_SIZE_LIMIT);
    for table in [
        "requests",
        "minute",
        "minute_session",
        "machine",
        "tiers",
        "settings",
        "settings_log",
    ] {
        assert_eq!(count(&conn, table), 0, "{table}");
    }
    let mut w = Writer::new(conn, Some(path.clone()), Arc::default(), 0);
    assert!(w.record(vec![row(T0 + 30_000, Some(7), 100.0, 10, 1000.0)]));
    drop(w);
    // A second start migrates nothing and keeps what is there.
    let (conn, aside) = open(&path, 2).expect("reopen");
    assert_eq!(aside, None);
    assert_eq!(count(&conn, "requests"), 1);
    assert_eq!(max_lineage(&conn).unwrap(), Some(7));
    drop(conn);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_fresh_database_vacuums_incrementally() {
    let root = temp_root("vacuum");
    let w = on_disk(&root.join("lily.sqlite"));
    assert_eq!(pragma(w.connection(), "auto_vacuum"), 2);
    assert_eq!(pragma(&in_memory().unwrap(), "auto_vacuum"), 2);
    drop(w);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn the_directory_lily_creates_is_private_and_so_are_the_files() {
    let root = temp_root("modes");
    let path = root.join("a").join("b").join("lily.sqlite");
    let root_mode = mode(&root);
    let mut w = on_disk(&path);
    assert!(w.record(vec![row(T0, Some(1), 10.0, 1, 10.0)]));
    assert_eq!(mode(&root.join("a")), 0o700);
    assert_eq!(mode(&root.join("a").join("b")), 0o700);
    assert_eq!(mode(&root), root_mode, "an existing directory is left alone");
    assert_eq!(mode(&path), 0o600);
    for side in ["-wal", "-shm"] {
        let file = with_suffix(&path, side);
        if file.exists() {
            assert_eq!(mode(&file), 0o600, "{side}");
        }
    }
    drop(w);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_corrupt_file_is_set_aside_with_its_wal_and_a_fresh_one_started() {
    let root = temp_root("corrupt");
    let path = root.join("lily.sqlite");
    fs::write(
        &path,
        b"this is not a database, it only sits where one should".repeat(100),
    )
    .expect("garbage");
    fs::write(with_suffix(&path, "-wal"), b"stale wal").expect("wal");
    let (conn, aside) = open(&path, 1_760_106_000).expect("open");
    let aside = aside.expect("set aside");
    assert_eq!(aside.to, root.join("lily.sqlite.corrupt-1760106000"));
    assert!(fs::read(&aside.to).unwrap().starts_with(b"this is not a database"));
    // The stale WAL went aside with it (or SQLite removed it on closing the
    // file it could not read): the fresh database never replays it.
    let fresh_wal = fs::read(with_suffix(&path, "-wal")).unwrap_or_default();
    assert_ne!(fresh_wal, b"stale wal");
    assert_eq!(pragma(&conn, "user_version"), SCHEMA_VERSION);
    assert_eq!(count(&conn, "requests"), 0);
    assert_eq!(mode(&path), 0o600);
    drop(conn);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_database_from_a_newer_lily_is_left_untouched_and_the_history_off() {
    let root = temp_root("newer");
    let path = root.join("lily.sqlite");
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE future (x); PRAGMA user_version = 99;")
            .unwrap();
    }
    let before = fs::read(&path).unwrap();
    match open(&path, 5) {
        Err(OpenFailure::Newer { found: 99 }) => {}
        other => panic!("{other:?}"),
    }
    let board = Arc::new(Board::default());
    let recorder = Recorder::start(Some(&path), &board);
    assert!(eventually(|| recorder.status().state == RecorderState::Failed));
    let status = recorder.status();
    assert!(!status.ok);
    let error = status.error.expect("why");
    assert!(error.contains("has schema 99, this lily knows 1: history off"), "{error}");
    assert_eq!(recorder.read(|_| Ok(())), Err(Unavailable::Failed));
    assert_eq!(fs::read(&path).unwrap(), before, "the file is untouched");
    let names: Vec<_> =
        fs::read_dir(&root).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert!(
        names.iter().all(|n| !n.to_string_lossy().contains("corrupt")),
        "nothing set aside: {names:?}"
    );
    // Rows offered to a recorder that will never write are counted.
    recorder.finish(Duration::from_secs(2));
    board.record(row(T0, None, 1.0, 1, 1.0));
    assert_eq!(recorder.status().dropped_rows, 1);
    let _ = fs::remove_dir_all(&root);
}

// --- opening on the recorder's thread ---------------------------------------------

#[test]
fn a_database_that_cannot_be_opened_yet_is_retried_while_rows_wait() {
    let root = temp_root("retry");
    // A file where the database's directory should be: nothing can be
    // created under it until it goes.
    let blocker = root.join("blocker");
    fs::write(&blocker, b"in the way").unwrap();
    let path = blocker.join("lily.sqlite");
    let board = Arc::new(Board::default());
    let started = Instant::now();
    let recorder = Recorder::start(Some(&path), &board);
    assert!(
        started.elapsed() < Duration::from_millis(500),
        "start never waits on SQLite"
    );
    assert!(eventually(|| recorder.status().error.is_some()));
    let status = recorder.status();
    assert_eq!((status.state, status.ok), (RecorderState::Opening, false));
    assert_eq!(recorder.read(|_| Ok(())), Err(Unavailable::Opening));
    // A row before the open waits for it (recent: the first prune keeps it).
    board.record(row(crate::serve::now_ms(), Some(3), 10.0, 1, 10.0));
    fs::remove_file(&blocker).unwrap();
    assert!(eventually(|| recorder.status().state == RecorderState::Open));
    assert!(eventually(|| {
        recorder.read(|conn| Ok(count(conn, "requests"))) == Ok(1)
    }));
    assert_eq!(recorder.status().dropped_rows, 0);
    recorder.finish(Duration::from_secs(2));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_stop_while_opening_ends_at_once() {
    let root = temp_root("stop-opening");
    let blocker = root.join("blocker");
    fs::write(&blocker, b"in the way").unwrap();
    let board = Arc::new(Board::default());
    let recorder = Recorder::start(Some(&blocker.join("lily.sqlite")), &board);
    assert!(eventually(|| recorder.status().error.is_some()));
    board.record(row(T0, Some(3), 10.0, 1, 10.0));
    let started = Instant::now();
    recorder.finish(Duration::from_secs(2));
    assert!(started.elapsed() < Duration::from_secs(1), "{:?}", started.elapsed());
    assert_eq!(recorder.status().dropped_rows, 1, "the waiting row is counted");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn the_recorder_thread_stores_the_boards_rows_and_seeds_the_lineage_counter() {
    let root = temp_root("thread");
    let path = root.join("lily.sqlite");
    let far = 9_000_000_000_000_000;
    // Recent enough that the thread's first prune keeps it.
    let then = crate::serve::now_ms() - 3_600_000;
    {
        let mut w = on_disk(&path);
        assert!(w.record(vec![row(then, Some(far), 10.0, 1, 10.0)]));
    }
    let board = Arc::new(Board::default());
    assert!(board.lineages().next().id < far, "the clock seed is below the stored id");
    let recorder = Recorder::start(Some(&path), &board);
    assert!(eventually(|| recorder.status().state == RecorderState::Open));
    assert!(board.lineages().next().id > far, "raised above the database's ids");
    let status = recorder.status();
    assert!(status.ok && status.error.is_none(), "{status:?}");
    // The oldest sample: the minute the stored row's prefill began in.
    assert_eq!(status.oldest, Some((then - 20) / MINUTE_MS * MINUTE_MS));
    let now = crate::serve::now_ms();
    board.record(row(now, Some(5), 10.0, 20, 400.0));
    let stored = || {
        recorder
            .read(|conn| {
                conn.query_row(
                    "SELECT count(*) FROM requests WHERE lineage = 5",
                    [],
                    |r| r.get::<_, u64>(0),
                )
            })
            .is_ok_and(|n| n == 1)
    };
    assert!(eventually(stored), "the row reached the database");
    let status = recorder.status();
    assert!(status.ok && status.db_bytes > 0, "{status:?}");
    assert_eq!(status.dropped_rows, 0);
    recorder.finish(Duration::from_secs(2));
    assert!(!recorder.status().ok, "ended");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn finish_writes_what_is_queued_checkpoints_and_never_waits_past_its_cap() {
    let root = temp_root("finish");
    let path = root.join("lily.sqlite");
    let board = Arc::new(Board::default());
    let recorder = Recorder::start(Some(&path), &board);
    assert!(eventually(|| recorder.status().state == RecorderState::Open));
    let now = crate::serve::now_ms();
    for i in 0..50 {
        board.record(row(now + i, Some(i), 10.0, 5, 50.0));
    }
    let started = Instant::now();
    recorder.finish(Duration::from_secs(2));
    assert!(started.elapsed() < Duration::from_millis(2500));
    let conn = open_reader(&path).unwrap();
    assert_eq!(count(&conn, "requests"), 50, "the queued rows were written first");
    drop(conn);
    // The WAL was checkpointed and truncated.
    let wal = with_suffix(&path, "-wal");
    assert!(!wal.exists() || fs::metadata(&wal).unwrap().len() == 0);
    // A second finish, or one on a recorder that is off, returns at once.
    recorder.finish(Duration::from_secs(2));
    Recorder::off().finish(Duration::from_secs(2));
    let _ = fs::remove_dir_all(&root);
}

// --- the rollups ---------------------------------------------------------------

#[test]
fn busy_time_is_the_union_of_the_running_intervals() {
    // Two requests decoding together in minute M: 10 s of decode each, the
    // second starting 5 s after the first. 15 s busy, not 20.
    let a = Span::of(&row(T0 + 20_000, Some(1), 1000.0, 500, 10_000.0), 0);
    let b = Span::of(&row(T0 + 25_000, Some(2), 1000.0, 300, 10_000.0), 0);
    let m = Minute::of(M, [a, b].iter()).expect("minute");
    assert_eq!(m.decode_busy_ms, 15_000.0);
    // Prefills at [9, 10) s and [14, 15) s, inside the first decode.
    assert_eq!(m.prefill_busy_ms, 2000.0);
    assert_eq!(m.busy_ms, 16_000.0, "the first prefill start to the second finish");
    assert_eq!(m.decode_tokens, 800.0);
    assert_eq!(m.prefill_tokens, 200.0);
    assert_eq!(m.sessions_max, 2);
    assert_eq!((m.requests, m.prompt_tokens, m.cached_tokens), (2, 2000, 1800));
    assert_eq!((m.drafted_tokens, m.accepted_tokens), (40, 30));
}

#[test]
fn a_request_spanning_minutes_shares_its_tokens_by_overlap() {
    // 30 s of decode across the minute edge: 10 s in M, 20 s in M + 1.
    let span = Span::of(&row(T0 + 80_000, Some(1), 0.0, 300, 30_000.0), 0);
    assert_eq!(span.minutes(), M..=M + 1);
    let first = Minute::of(M, [span].iter()).expect("ran through M");
    let second = Minute::of(M + 1, [span].iter()).expect("finished in M + 1");
    assert_eq!((first.decode_busy_ms, first.decode_tokens), (10_000.0, 100.0));
    assert_eq!((second.decode_busy_ms, second.decode_tokens), (20_000.0, 200.0));
    // Counts belong to the minute the request finished in.
    assert_eq!((first.requests, second.requests), (0, 1));
    assert_eq!(Minute::of(M + 2, [span].iter()), None);
    assert_eq!(Minute::of(M - 1, [span].iter()), None);
}

#[test]
fn a_full_cache_hit_adds_no_prefill_busy_time() {
    let mut hit = row(T0 + 10_000, Some(1), 400.0, 10, 1000.0);
    hit.prefill_tokens = 0;
    hit.cached_tokens = 1000;
    let m = Minute::of(M, [Span::of(&hit, 0)].iter()).unwrap();
    assert_eq!((m.prefill_busy_ms, m.prefill_tokens), (0.0, 0.0));
    // It was running all the same.
    assert_eq!(m.busy_ms, 1400.0);
}

#[test]
fn a_failed_request_counts_but_runs_for_no_time_and_has_no_queue_time() {
    let failed = RequestRow {
        failed: true,
        drafted_tokens: None,
        accepted_tokens: None,
        queue_ms: 5.0,
        ..row(T0 + 5_000, Some(3), 0.0, 0, 0.0)
    };
    let m = Minute::of(M, [Span::of(&failed, 0)].iter()).expect("counted");
    assert_eq!((m.requests, m.busy_ms, m.sessions_max), (1, 0.0, 0));
    assert_eq!((m.drafted_tokens, m.accepted_tokens), (0, 0));
    let mut w = writer();
    assert!(w.record(vec![failed, row(T0 + 6_000, Some(3), 1.0, 1, 1.0)]));
    let queues: Vec<Option<f64>> = w
        .connection()
        .prepare("SELECT queue_ms FROM requests ORDER BY t")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(queues, vec![None, Some(2.0)]);
}

#[test]
fn accepted_draft_tokens_are_stored_clamped_to_the_drafted() {
    let mut w = writer();
    let odd = RequestRow {
        drafted_tokens: Some(10),
        accepted_tokens: Some(15),
        ..row(T0, Some(1), 1.0, 1, 1.0)
    };
    assert!(w.record(vec![odd]));
    let (drafted, accepted): (u64, u64) = w
        .connection()
        .query_row("SELECT drafted_tokens, accepted_tokens FROM requests", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!((drafted, accepted), (10, 10));
}

#[test]
fn a_bogus_duration_is_clamped_to_a_day_and_to_the_process_start() {
    let mut w = writer();
    let bogus = row(T0, Some(1), 1e18, 10, 1e15);
    let started = Instant::now();
    assert!(w.record(vec![bogus.clone()]));
    assert!(started.elapsed() < Duration::from_secs(5), "{:?}", started.elapsed());
    let (decode, prefill, start): (f64, f64, u64) = w
        .connection()
        .query_row("SELECT decode_ms, prefill_ms, start_t FROM requests", [], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap();
    assert_eq!((decode, prefill), (MAX_SPAN_MS as f64, 0.0), "the decode first");
    assert_eq!(start, T0 - MAX_SPAN_MS);
    assert_eq!(count(w.connection(), "minute"), MAX_SPAN_MS / MINUTE_MS + 1);
    // Nothing starts before the process did.
    let span = Span::of(&row(T0 + 10_000, Some(1), 5_000.0, 10, 60_000.0), T0);
    assert_eq!((span.decode_ms, span.prefill_ms, span.start_t()), (10_000.0, 0.0, T0));
    let span = Span::of(&row(T0 + 10_000, Some(1), 5_000.0, 10, 3_000.0), T0);
    assert_eq!((span.decode_ms, span.prefill_ms), (3_000.0, 5_000.0), "inside: as is");
}

#[test]
fn rows_recompute_every_minute_they_ran_through_and_per_session_rows_too() {
    let mut w = writer();
    // A long request finishing in M + 2 that started in M.
    let long = row(T0 + 2 * MINUTE_MS + 30_000, Some(1), 10_000.0, 1500, 120_000.0);
    // A short one of another lineage in M + 1, which the long one overlaps.
    let short = row(T0 + MINUTE_MS + 30_000, Some(2), 1000.0, 100, 2000.0);
    assert!(w.record(vec![short]));
    let before = minute_row(w.connection(), M + 1).expect("M + 1");
    assert_eq!(before.2, 2000.0, "the short one alone");
    assert!(w.record(vec![long]));
    let conn = w.connection();
    // The long one ran from M + 20 s to M + 2 + 30 s.
    let first = minute_row(conn, M).expect("M");
    assert_eq!((first.0, first.4), (0, 40_000.0));
    let middle = minute_row(conn, M + 1).expect("M + 1 recomputed");
    assert_eq!(middle.2, 60_000.0, "the long decode covers the whole minute");
    assert_eq!(middle.5, 2, "both running at once");
    assert_eq!(middle.0, 1, "only the short one finished in it");
    let last = minute_row(conn, M + 2).expect("M + 2");
    assert_eq!(last.0, 1);
    // Recording is idempotent: the rollup is recomputed from the rows.
    let total: f64 = conn
        .query_row("SELECT sum(decode_tokens) FROM minute", [], |r| r.get(0))
        .unwrap();
    assert!((total - 1600.0).abs() < 1e-6, "{total}");
    let session: Vec<(u64, u64, f64)> = conn
        .prepare("SELECT lineage, minute, decode_busy_ms FROM minute_session ORDER BY lineage, minute")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(
        session,
        vec![
            (1, M, 30_000.0),
            (1, M + 1, 60_000.0),
            (1, M + 2, 30_000.0),
            (2, M + 1, 2000.0),
        ]
    );
}

#[test]
fn request_rows_hold_numbers_only() {
    // Every column of every table is a number, but the settings' names and
    // values (§ 2.1).
    let conn = in_memory().unwrap();
    let mut select = conn
        .prepare(
            "SELECT m.name, p.name, p.type FROM sqlite_master m
             JOIN pragma_table_info(m.name) p WHERE m.type = 'table'",
        )
        .unwrap();
    let columns: Vec<(String, String, String)> = select
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert!(columns.len() > 40);
    for (table, column, kind) in columns {
        if table.starts_with("settings") {
            continue;
        }
        assert!(
            matches!(kind.as_str(), "INTEGER" | "REAL"),
            "{table}.{column} is {kind}"
        );
    }
}

// --- failed batches and rows -------------------------------------------------------

#[test]
fn a_failed_batch_is_retried_once_with_the_next() {
    let mut w = writer();
    w.connection().execute_batch("DROP TABLE minute").unwrap();
    let a = vec![row(T0, Some(1), 1.0, 1, 1.0), row(T0 + 1, Some(1), 1.0, 1, 1.0)];
    assert!(!w.record(a), "kept for a retry");
    assert_eq!(w.health().dropped.load(Ordering::Relaxed), 0);
    w.connection().execute_batch(create_table("minute")).unwrap();
    assert!(w.record(vec![row(T0 + 2, Some(1), 1.0, 1, 1.0)]));
    assert_eq!(count(w.connection(), "requests"), 3, "the retry and the new batch");
    assert_eq!(w.health().dropped.load(Ordering::Relaxed), 0);
    assert!(w.health().ok.load(Ordering::Relaxed));
}

#[test]
fn a_batch_failing_twice_is_dropped_and_its_rows_counted() {
    let mut w = writer();
    w.connection().execute_batch("DROP TABLE minute").unwrap();
    assert!(
        !w.record(vec![row(T0, Some(1), 1.0, 1, 1.0), row(T0 + 1, None, 1.0, 1, 1.0)])
    );
    assert!(!w.record(vec![row(T0 + 2, Some(1), 1.0, 1, 1.0)]));
    assert_eq!(w.health().dropped.load(Ordering::Relaxed), 2, "the first batch");
    w.give_up_retry();
    assert_eq!(w.health().dropped.load(Ordering::Relaxed), 3, "and the second");
    assert!(!w.health().ok.load(Ordering::Relaxed));
    assert_eq!(count(w.connection(), "requests"), 0);
}

#[test]
fn a_row_sqlite_refuses_is_skipped_and_counted_and_the_rest_stored() {
    let mut w = writer();
    // An id SQLite's integers cannot hold.
    let bad = row(T0, Some(u64::MAX), 1.0, 1, 1.0);
    assert!(w.record(vec![bad, row(T0 + 1, Some(2), 1.0, 1, 1.0)]));
    assert_eq!(count(w.connection(), "requests"), 1);
    assert_eq!(w.health().dropped.load(Ordering::Relaxed), 1);
    assert!(w.health().ok.load(Ordering::Relaxed), "the batch itself succeeded");
}

#[test]
fn a_sqlite_error_marks_the_recorder_not_ok_and_never_panics() {
    let mut w = writer();
    assert!(w.health().ok.load(Ordering::Relaxed));
    w.connection().execute_batch("DROP TABLE requests").unwrap();
    assert!(!w.record(vec![row(T0, Some(1), 10.0, 1, 10.0)]));
    assert!(!w.health().ok.load(Ordering::Relaxed));
    assert!(!w.record(vec![row(T0, Some(1), 10.0, 1, 10.0)]), "again, logged once");
    // Another kind of write succeeding does not hide the failing one.
    assert!(w.machine(&MachineMinute::new(M)));
    assert!(!w.health().ok.load(Ordering::Relaxed));
    // Once the failing kind succeeds again the recorder is ok.
    w.connection().execute_batch(create_table("requests")).unwrap();
    assert!(w.record(vec![row(T0, Some(1), 10.0, 1, 10.0)]));
    assert!(w.health().ok.load(Ordering::Relaxed));
}

#[test]
fn a_failed_read_at_the_start_is_logged_and_leaves_the_recorder_ok() {
    let mut w = writer();
    w.connection().execute_batch("DROP TABLE requests").unwrap();
    assert_eq!(w.max_lineage(), None);
    w.publish_extent();
    assert!(w.health().ok.load(Ordering::Relaxed));
    assert!(w.failing.is_empty());
}

#[test]
fn a_full_channel_or_a_gone_recorder_drops_and_counts_the_row() {
    let (sink, rx, health) = sink(1);
    sink.send(row(T0, None, 1.0, 1, 1.0));
    sink.send(row(T0, None, 1.0, 1, 1.0));
    assert_eq!(health.dropped.load(Ordering::Relaxed), 1, "the second found it full");
    assert!(rx.try_recv().is_ok());
    drop(rx);
    sink.send(row(T0, None, 1.0, 1, 1.0));
    assert_eq!(health.dropped.load(Ordering::Relaxed), 2, "nobody is listening");
    // The board offers every row and never waits.
    let board = Board::default();
    let (sink, _rx, health) = super::sink(0);
    assert!(board.attach_recorder(sink));
    board.record(row(T0, None, 1.0, 1, 1.0));
    assert_eq!(health.dropped.load(Ordering::Relaxed), 1);
    let (again, _, _) = super::sink(1);
    assert!(!board.attach_recorder(again), "one recorder per board");
}

// --- pruning ---------------------------------------------------------------------

#[test]
fn pruning_drops_everything_older_than_seven_days_and_an_hour_but_the_settings() {
    let mut w = writer();
    let now = T0 + 10 * 86_400_000;
    let cutoff = Writer::prune_cutoff(now);
    assert_eq!(cutoff % 3_600_000, 0, "on the hour");
    assert!(cutoff <= now - RETENTION_MS - 3_600_000);
    assert!(cutoff > now - RETENTION_MS - 2 * 3_600_000);
    assert!(w.record(vec![
        row(cutoff - 61_000, Some(1), 0.0, 10, 500.0),
        row(cutoff + 61_000, Some(2), 0.0, 10, 500.0),
    ]));
    for minute in [cutoff / MINUTE_MS - 1, cutoff / MINUTE_MS + 1] {
        let mut m = MachineMinute::new(minute);
        m.add(&MachineSample { cpu_load: Some(0.5), ..MachineSample::default() });
        assert!(w.machine(&m));
        let tiers = TierSnapshot {
            store: StoreView::default(),
            active_sessions: 0,
            active_bytes: 0,
            weights_pinned: false,
        };
        assert!(w.tiers(minute, &tiers));
    }
    let conn = w.connection();
    conn.execute(
        "INSERT INTO settings_log (t, name, old, new, source) VALUES (?1, 'top_k', NULL, '20', 'dashboard'), (?2, 'top_k', '20', '40', 'dashboard')",
        params![cutoff - 1, cutoff + 1],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO settings (name, value, changed_t) VALUES ('top_k', '40', ?1)",
        [cutoff - 1],
    )
    .unwrap();
    assert!(w.prune(now));
    let conn = w.connection();
    for table in
        ["requests", "minute", "minute_session", "machine", "tiers", "settings_log"]
    {
        assert_eq!(count(conn, table), 1, "{table}");
    }
    assert_eq!(count(conn, "settings"), 1, "an override is kept until it is reset");
    assert_eq!(max_lineage(conn).unwrap(), Some(2));
    assert_eq!(
        w.health().oldest.load(Ordering::Relaxed),
        (cutoff + 61_000 - 500) / MINUTE_MS * MINUTE_MS,
        "the oldest sample: the surviving request's minute"
    );
}

#[test]
fn pruning_on_disk_vacuums_and_truncates_the_wal() {
    let root = temp_root("prune-disk");
    let path = root.join("lily.sqlite");
    let mut w = on_disk(&path);
    let rows: Vec<_> =
        (0..2000).map(|i| row(T0 + i * 1000, Some(i), 10.0, 5, 100.0)).collect();
    assert!(w.record(rows));
    assert!(w.prune(T0 + 30 * 86_400_000));
    assert_eq!(count(w.connection(), "requests"), 0);
    assert_eq!(pragma(w.connection(), "freelist_count"), 0, "pages handed back");
    let wal = with_suffix(&path, "-wal");
    assert_eq!(fs::metadata(&wal).map_or(0, |m| m.len()), 0, "WAL truncated");
    drop(w);
    let _ = fs::remove_dir_all(&root);
}

// --- the machine and the tiers ---------------------------------------------------

#[test]
fn a_machine_minute_keeps_least_mean_most_and_the_worst_pressure() {
    let mut m = MachineMinute::new(M);
    for (load, level, swap) in [(0.2, 1, 100), (0.6, 4, 300), (0.4, 2, 200)] {
        m.add(&MachineSample {
            cpu_load: Some(load),
            pressure_level: Some(level),
            swap_bytes: Some(swap),
            ..MachineSample::default()
        });
    }
    // A probe that had nothing leaves the series alone.
    m.add(&MachineSample::default());
    assert_eq!(m.samples, 4);
    assert_eq!((m.cpu_load.min(), m.cpu_load.max()), (Some(0.2), Some(0.6)));
    assert!((m.cpu_load.mean().unwrap() - 0.4).abs() < 1e-12);
    assert_eq!(m.pressure_level, Some(4));
    assert_eq!(m.swap_bytes.max(), Some(300.0));
    assert_eq!((m.power_w.mean(), m.gpu_freq_mhz.min()), (None, None));
    let mut w = writer();
    assert!(w.machine(&m));
    let (load, pressure, swap, power): (f64, u32, u64, Option<f64>) = w
        .connection()
        .query_row(
            "SELECT cpu_load_mean, memory_pressure, swap_bytes_max, power_w_mean FROM machine",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .unwrap();
    assert!((load - 0.4).abs() < 1e-12);
    assert_eq!((pressure, swap, power), (4, 300, None));
}

#[test]
fn after_the_clock_went_back_a_minute_keeps_its_latest_samples() {
    let mut w = writer();
    let written = |w: &Writer| -> (u32, f64) {
        w.connection()
            .query_row(
                "SELECT samples, cpu_load_mean FROM machine WHERE minute = ?1",
                [M],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap()
    };
    let mut first = MachineMinute::new(M);
    for _ in 0..6 {
        first.add(&MachineSample { cpu_load: Some(0.2), ..MachineSample::default() });
    }
    assert!(w.machine(&first));
    // The clock went back into M: the recorder samples M again and stores
    // it when it ends, replacing the first.
    let mut again = MachineMinute::new(M);
    again.add(&MachineSample { cpu_load: Some(0.8), ..MachineSample::default() });
    assert!(w.machine(&again));
    assert_eq!(written(&w), (1, 0.8));
    assert_eq!(count(w.connection(), "machine"), 1);
}

#[test]
fn the_tier_snapshot_counts_the_sessions_requests_hold() {
    let board = Board::default();
    board.publish(StoreView {
        gpu: GpuTier {
            used_bytes: 1000,
            budget_bytes: Some(9000),
            entries: 1,
            max_entries: 16,
        },
        disk: Some(DiskTier {
            used_bytes: 5000,
            budget_bytes: 50_000,
            entries: 2,
            ..DiskTier::default()
        }),
        ..StoreView::default()
    });
    board.admitted(Lineage { id: 4, created_ms: 1 }, 100, 700);
    let snapshot = board.tier_snapshot();
    assert_eq!((snapshot.active_sessions, snapshot.active_bytes), (1, 700));
    let mut w = writer();
    assert!(w.tiers(M, &snapshot));
    let (gpu, entries, disk): (u64, u64, u64) = w
        .connection()
        .query_row("SELECT gpu_bytes, gpu_entries, disk_bytes FROM tiers", [], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap();
    assert_eq!((gpu, entries, disk), (1700, 2, 5000));
}

// --- the flag and the read side ------------------------------------------------------

#[test]
fn stats_db_takes_an_absolute_path_or_off() {
    assert_eq!(stats_db(Some("off")), Ok(None));
    assert_eq!(stats_db(Some("/x/y.sqlite")), Ok(Some(PathBuf::from("/x/y.sqlite"))));
    assert!(stats_db(None).is_ok_and(|p| {
        p.is_some_and(|p| p.ends_with("Application Support/lily/lily.sqlite"))
    }));
    for bad in ["", "lily.sqlite", "./lily.sqlite", "~/lily.sqlite"] {
        let error = stats_db(Some(bad)).expect_err(bad);
        assert!(error.contains("--stats-db"), "{error}");
        assert!(error.contains("absolute path"), "{error}");
    }
}

#[test]
fn stats_db_off_disables_the_recorder() {
    let board = Arc::new(Board::default());
    let recorder = Recorder::start(None, &board);
    assert_eq!(
        recorder.status(),
        RecorderStatus {
            state: RecorderState::Off,
            ok: false,
            dropped_rows: 0,
            db_bytes: 0,
            oldest: None,
            error: None,
        }
    );
    assert_eq!(recorder.read(|_| Ok(())), Err(Unavailable::Off));
    // No sink: rows go to the ring only.
    board.record(row(T0, None, 1.0, 1, 1.0));
    assert!(board.attach_recorder(super::sink(1).0), "nothing was attached");
}

#[test]
fn readers_see_the_writers_rows_while_it_holds_the_database() {
    let root = temp_root("reader");
    let path = root.join("lily.sqlite");
    let mut w = on_disk(&path);
    assert!(w.record(vec![row(T0, Some(1), 10.0, 1, 10.0)]));
    let reader = open_reader(&path).expect("read-only");
    assert_eq!(count(&reader, "requests"), 1);
    assert!(reader.execute("DELETE FROM requests", []).is_err(), "read-only");
    assert!(w.record(vec![row(T0 + 1, Some(1), 10.0, 1, 10.0)]));
    assert_eq!(count(&reader, "requests"), 2);
    let recorder = Recorder::reading(&path, w.health().clone());
    // Both rows ran through the minute before T0 and finished in T0's.
    assert_eq!(recorder.read(|conn| Ok(count(conn, "minute"))), Ok(2));
    drop((reader, w));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_reader_facing_a_held_lock_answers_within_its_short_timeout() {
    let root = temp_root("locked");
    let path = root.join("lily.sqlite");
    let w = on_disk(&path);
    let recorder = Recorder::reading(&path, w.health().clone());
    assert_eq!(recorder.read(|conn| Ok(count(conn, "requests"))), Ok(0));
    drop(w);
    // Another process holding the file exclusively (a `sqlite3` shell, a
    // backup tool): readers cannot get in.
    let holder = Connection::open(&path).unwrap();
    holder.execute_batch("PRAGMA locking_mode = EXCLUSIVE; BEGIN EXCLUSIVE;").unwrap();
    holder
        .execute(
            "INSERT INTO settings (name, value, changed_t) VALUES ('x', '1', 0)",
            [],
        )
        .unwrap();
    let started = Instant::now();
    let read = recorder.read(|conn| {
        conn.query_row("SELECT count(*) FROM requests", [], |r| r.get::<_, u64>(0))
    });
    assert_eq!(read, Err(Unavailable::Read));
    let took = started.elapsed();
    assert!(took < Duration::from_secs(1), "{took:?}");
    holder.execute_batch("ROLLBACK").unwrap();
    drop(holder);
    let _ = fs::remove_dir_all(&root);
}
