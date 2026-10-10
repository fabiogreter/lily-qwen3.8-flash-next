use std::fs;
use std::os::unix::fs::PermissionsExt as _;

use super::*;
use crate::serve::dashboard::contract::{DiskTier, GpuTier, Range};
use crate::serve::dashboard::history;
use crate::serve::machine::Machine;
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
    let recorder = Recorder::start(Some(&path), &board, None);
    assert!(eventually(|| recorder.status().state == RecorderState::Failed));
    let status = recorder.status();
    assert!(!status.ok);
    let error = status.error.expect("why");
    assert!(
        error.contains(&format!(
            "has schema 99, this lily knows {SCHEMA_VERSION}: history off"
        )),
        "{error}"
    );
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
    let recorder = Recorder::start(Some(&path), &board, None);
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
    let recorder = Recorder::start(Some(&blocker.join("lily.sqlite")), &board, None);
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
    let recorder = Recorder::start(Some(&path), &board, None);
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
    let recorder = Recorder::start(Some(&path), &board, None);
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

/// The sampler's window `[start_ms, end_ms]` reading `sys_w`, with the other
/// figures following from it.
fn window(start_ms: u64, end_ms: u64, sys_w: f64) -> Sample {
    Sample {
        start_ms,
        end_ms,
        reading: Reading {
            cpu_w: sys_w / 2.0,
            gpu_w: sys_w / 4.0,
            ane_w: 0.0,
            sys_w: Some(sys_w),
            gpu_active: sys_w / 100.0,
            gpu_freq_mhz: sys_w * 10.0,
            cpu_temp_c: Some(40.0 + sys_w / 10.0),
            gpu_temp_c: None,
        },
    }
}

#[derive(Debug, PartialEq)]
struct PowerColumns {
    covered_ms: Option<u64>,
    sys_covered_ms: Option<u64>,
    sys_w: (Option<f64>, Option<f64>, Option<f64>),
    gpu_w: Option<f64>,
    gpu_active: Option<f64>,
    freq: (Option<f64>, Option<f64>, Option<f64>),
    temps: (Option<f64>, Option<f64>),
    samples: u32,
}

fn power_columns(w: &Writer, minute: u64) -> Option<PowerColumns> {
    w.connection()
        .query_row(
            "SELECT power_covered_ms, power_w_min, power_w_mean, power_w_max,
                    gpu_power_w_mean, gpu_active_mean, gpu_freq_mhz_min,
                    gpu_freq_mhz_mean, gpu_freq_mhz_max, cpu_temp_c_max,
                    gpu_temp_c_max, samples, sys_covered_ms
             FROM machine WHERE minute = ?1",
            [minute],
            |r| {
                Ok(PowerColumns {
                    covered_ms: r.get(0)?,
                    sys_w: (r.get(1)?, r.get(2)?, r.get(3)?),
                    gpu_w: r.get(4)?,
                    gpu_active: r.get(5)?,
                    freq: (r.get(6)?, r.get(7)?, r.get(8)?),
                    temps: (r.get(9)?, r.get(10)?),
                    samples: r.get(11)?,
                    sys_covered_ms: r.get(12)?,
                })
            },
        )
        .ok()
}

fn close(a: Option<f64>, b: f64) -> bool {
    a.is_some_and(|a| (a - b).abs() < 1e-9)
}

#[test]
fn a_window_is_spread_over_the_minutes_it_overlaps_by_the_time_in_each() {
    // 10 s in the first minute, 60 s in the second, 20 s in the third.
    let parts = spread(&[window(T0 + 50_000, T0 + 140_000, 10.0)]);
    let covered: Vec<_> = parts.iter().map(|(m, p)| (*m, p.covered_ms)).collect();
    assert_eq!(covered, [(M, 10_000), (M + 1, 60_000), (M + 2, 20_000)]);
    for p in parts.values() {
        assert_eq!(p.sys_w_mean(), Some(10.0), "the window's reading in each");
    }
    // A window with no span covers nothing, and one that ends on a minute's
    // edge stays in it.
    assert!(spread(&[window(T0, T0, 5.0), window(T0 + 9, T0 + 3, 5.0)]).is_empty());
    let edge = spread(&[window(T0 - 5_000, T0, 5.0)]);
    assert_eq!(edge.keys().copied().collect::<Vec<_>>(), [M - 1]);
    assert_eq!(edge[&(M - 1)].covered_ms, 5_000);
}

#[test]
fn a_minutes_power_is_time_weighted_with_the_least_and_most_of_its_windows() {
    // 2 s at 10 W, 6 s at 30 W, 2 s at 20 W in one minute.
    let parts = spread(&[
        window(T0, T0 + 2_000, 10.0),
        window(T0 + 2_000, T0 + 8_000, 30.0),
        window(T0 + 8_000, T0 + 10_000, 20.0),
    ]);
    let p = parts[&M];
    assert_eq!(p.covered_ms, 10_000);
    assert!(close(p.sys_w_mean(), (20.0 + 180.0 + 40.0) / 10.0), "{p:?}");
    assert_eq!((p.sys_w_min(), p.sys_w_max()), (Some(10.0), Some(30.0)));
    assert_eq!(
        (p.gpu_freq_mhz_min(), p.gpu_freq_mhz_max()),
        (Some(100.0), Some(300.0))
    );
    assert!(close(p.gpu_w_mean(), 24.0 / 4.0), "{p:?}");
    assert!(close(p.gpu_active_mean(), 0.24), "{p:?}");
    assert_eq!((p.cpu_temp_max, p.gpu_temp_max), (Some(43.0), None), "the hottest");
    // 24 W for 10 s: 240 J, 0.0667 Wh.
    assert!(close(p.energy_wh(), 240.0 / 3600.0), "{:?}", p.energy_wh());
    assert_eq!(PowerMinute::default().energy_wh(), None);
}

#[test]
fn power_is_stored_per_minute_and_parts_of_a_minute_merge_into_the_row() {
    let mut w = writer();
    // The first 20 s of the minute at 10 W, written by one tick; the next
    // 10 s at 40 W by a later one (or a later process).
    assert!(w.power(M, &spread(&[window(T0, T0 + 20_000, 10.0)])[&M]));
    let first = power_columns(&w, M).expect("row");
    assert_eq!(first.covered_ms, Some(20_000));
    assert_eq!(first.samples, 0, "a row with the sampler's figures only");
    assert!(close(first.sys_w.1, 10.0));
    assert!(w.power(M, &spread(&[window(T0 + 20_000, T0 + 30_000, 40.0)])[&M]));
    let both = power_columns(&w, M).expect("row");
    assert_eq!(both.covered_ms, Some(30_000));
    assert!(close(both.sys_w.1, 20.0), "(10 x 20 + 40 x 10) / 30: {both:?}");
    assert_eq!((both.sys_w.0, both.sys_w.2), (Some(10.0), Some(40.0)));
    assert_eq!((both.freq.0, both.freq.2), (Some(100.0), Some(400.0)));
    assert!(close(both.gpu_w, 5.0) && close(both.gpu_active, 0.2), "{both:?}");
    assert_eq!(both.temps, (Some(44.0), None));
    assert_eq!(count(w.connection(), "machine"), 1);

    // The machine's own samples arrive for the same minute: their columns
    // are written, the sampler's are left as they are.
    let mut m = MachineMinute::new(M);
    m.add(&MachineSample { cpu_load: Some(0.5), ..MachineSample::default() });
    assert!(w.machine(&m));
    let after = power_columns(&w, M).expect("row");
    assert_eq!(
        (after.covered_ms, after.sys_w, after.samples),
        (both.covered_ms, both.sys_w, 1)
    );
    // And the other way round: power merged into a row with samples keeps them.
    assert!(w.power(M, &spread(&[window(T0 + 30_000, T0 + 40_000, 20.0)])[&M]));
    let last = power_columns(&w, M).expect("row");
    assert_eq!((last.covered_ms, last.samples), (Some(40_000), 1));
    assert!(close(last.sys_w.1, (600.0 + 200.0) / 40.0), "{last:?}");
    let load: f64 = w
        .connection()
        .query_row("SELECT cpu_load_mean FROM machine WHERE minute = ?1", [M], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(load, 0.5);
    // Nothing covered writes nothing.
    assert!(w.power(M + 5, &PowerMinute::default()));
    assert_eq!(power_columns(&w, M + 5), None);
}

#[test]
fn a_v1_database_gains_the_sampler_columns_and_keeps_its_rows() {
    let root = temp_root("v1");
    let path = root.join("lily.sqlite");
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(SCHEMA_V1).unwrap();
        conn.execute_batch(&format!(
            "INSERT INTO machine (minute, samples, cpu_load_mean) VALUES ({M}, 6, 0.25);
             PRAGMA user_version = 1;"
        ))
        .unwrap();
    }
    let (conn, aside) = open(&path, 5).expect("migrates");
    assert_eq!(aside, None);
    assert_eq!(pragma(&conn, "user_version"), 2);
    let mut w = Writer::new(conn, Some(path.clone()), Arc::default(), 0);
    let old = power_columns(&w, M).expect("the old row is kept");
    assert_eq!((old.samples, old.covered_ms, old.sys_w.1), (6, None, None));
    // Power merges into the old row as into any other.
    assert!(w.power(M, &spread(&[window(T0, T0 + 60_000, 30.0)])[&M]));
    let now = power_columns(&w, M).expect("row");
    assert_eq!((now.samples, now.covered_ms), (6, Some(60_000)));
    assert!(close(now.sys_w.1, 30.0));
    drop(w);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn the_recorder_takes_the_samplers_windows_and_the_history_shows_their_energy() {
    let root = temp_root("power");
    let path = root.join("lily.sqlite");
    let board = Arc::new(Board::default());
    let machine = Machine::running();
    board.attach_machine(machine.clone());
    // The last complete minutes before now: a whole minute at 20 W, a gap,
    // then the last 30 s of one minute at 40 W.
    let edge = crate::serve::now_ms() / MINUTE_MS * MINUTE_MS;
    machine
        .publish(Duration::from_secs(30), window(edge - 300_000, edge - 270_000, 20.0));
    machine
        .publish(Duration::from_secs(30), window(edge - 270_000, edge - 240_000, 20.0));
    machine
        .publish(Duration::from_secs(30), window(edge - 150_000, edge - 120_000, 40.0));
    let recorder = Recorder::start(Some(&path), &board, None);
    assert!(
        eventually(|| {
            recorder
                .read(|c| history::read(c, Range::QuarterHour, edge + 1, None, false))
                .is_ok_and(|h| h.is_some_and(|h| h.summary.energy_wh.is_some()))
        }),
        "the first tick takes the queue and writes it"
    );
    assert!(machine.drain().is_empty(), "the recorder took them");
    let h = recorder
        .read(|c| history::read(c, Range::QuarterHour, edge + 1, None, false))
        .unwrap()
        .unwrap();
    // 20 W for 60 s and 40 W for 30 s: 2 400 W s, over the 90 s covered.
    assert_eq!(h.summary.energy_wh, Some(0.667));
    assert_eq!(h.summary.power_avg_w, Some(26.67));
    let powered: Vec<_> = h.points.iter().filter_map(|p| p.power_w).collect();
    assert_eq!(powered, [20.0, 40.0], "the gap minutes stay null");
    let nulls = h.points.iter().filter(|p| p.power_w.is_none()).count();
    assert_eq!(nulls, h.points.len() - 2);
    // What a stop still holds is written before the thread ends.
    machine
        .publish(Duration::from_secs(30), window(edge - 60_000, edge - 30_000, 60.0));
    recorder.finish(Duration::from_secs(5));
    let after = Recorder::reading(&path, Arc::default())
        .read(|c| history::read(c, Range::QuarterHour, edge + 1, None, false))
        .unwrap()
        .unwrap();
    assert_eq!(after.points.iter().filter_map(|p| p.power_w).count(), 3);
    let _ = fs::remove_dir_all(&root);
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
fn the_default_database_needs_an_absolute_home() {
    assert_eq!(
        default_path_in(Some("/Users/x".into())),
        Some(PathBuf::from("/Users/x/Library/Application Support/lily/lily.sqlite"))
    );
    assert_eq!(default_path_in(None), None);
    assert_eq!(default_path_in(Some("".into())), None);
    assert_eq!(default_path_in(Some("relative".into())), None);
}

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
    let recorder = Recorder::start(None, &board, None);
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

// --- runtime settings ---------------------------------------------------------------

#[test]
fn stored_overrides_are_applied_over_the_command_line_once_the_database_opens() {
    use crate::serve::dashboard::contract::{SettingValue, SettingsLogSource};
    use crate::serve::runtime::testing::runtime;
    let root = temp_root("settings-startup");
    let path = root.join("lily.sqlite");
    {
        let (conn, _) = open(&path, 5).expect("database");
        for (name, value) in [
            ("temperature", "0.7"),
            ("top_p", "5"),
            ("max_seq", "1"),
            ("pin_hold", "\"10s\""),
            // Equal to the default in effect: an override, but no log entry.
            ("disk_cache_ttl", "\"3d\""),
        ] {
            conn.execute(
                "INSERT INTO settings (name, value, changed_t) VALUES (?1, ?2, 1)",
                params![name, value],
            )
            .unwrap();
        }
    }
    let settings = runtime();
    let board = Arc::new(Board::default());
    assert!(!settings.settled());
    let recorder = Recorder::start(Some(&path), &board, Some(settings.clone()));
    assert!(eventually(|| recorder.status().state == RecorderState::Open));
    assert!(settings.settled(), "the disk tier may expire now");
    assert!(settings.overrides().contains_key("disk_cache_ttl"));
    // In effect before the recorder said `open`: database over the flag.
    assert_eq!(settings.snapshot().defaults.sampling.temperature, 0.7);
    assert_eq!(settings.pin_hold_secs(), 10);
    // The rows it could not apply stay where they are.
    let stored = recorder.read(stored_settings).unwrap();
    assert_eq!(stored.len(), 5);
    assert_eq!(settings.snapshot().defaults.sampling.top_p, 0.95, "top_p 5 left out");
    // One `startup` entry per override applied.
    let log = recorder.read(|c| settings_log(c, 10)).unwrap();
    let mut entries: Vec<_> = log
        .iter()
        .map(|e| (e.name.as_str(), e.old.clone(), e.new.clone(), e.source))
        .collect();
    entries.sort_by_key(|e| e.0);
    assert_eq!(
        entries,
        vec![
            (
                "pin_hold",
                Some(SettingValue::Text("1m".into())),
                Some(SettingValue::Text("10s".into())),
                SettingsLogSource::Startup
            ),
            (
                "temperature",
                Some(SettingValue::Number(0.6)),
                Some(SettingValue::Number(0.7)),
                SettingsLogSource::Startup
            ),
        ]
    );
    recorder.finish(Duration::from_secs(2));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_change_is_written_with_its_log_entry_before_it_takes_effect() {
    use crate::serve::dashboard::contract::{SettingValue, SettingsLogSource};
    use crate::serve::runtime::{Change, testing::runtime};
    let root = temp_root("settings-change");
    let path = root.join("lily.sqlite");
    let settings = runtime();
    let board = Arc::new(Board::default());
    let recorder = Recorder::start(Some(&path), &board, Some(settings.clone()));
    assert!(eventually(|| recorder.status().state == RecorderState::Open));
    let change = Change {
        set: vec![("disk_cache_ttl".into(), SettingValue::Text("12h".into()))],
        reset: Vec::new(),
    };
    let entries = recorder.change_settings(change, SETTINGS_WAIT).expect("written");
    assert_eq!(entries.len(), 1);
    assert_eq!(settings.disk_cache_ttl_secs(), 12 * 3600);
    assert_eq!(
        recorder.read(stored_settings).unwrap(),
        vec![("disk_cache_ttl".into(), "\"12h\"".into())]
    );
    let change = Change { set: Vec::new(), reset: vec!["disk_cache_ttl".into()] };
    recorder.change_settings(change, SETTINGS_WAIT).expect("reset");
    assert_eq!(settings.disk_cache_ttl_secs(), 3 * 86_400, "back to the default");
    assert!(recorder.read(stored_settings).unwrap().is_empty(), "the override deleted");
    let log = recorder.read(|c| settings_log(c, 10)).unwrap();
    assert_eq!(log.len(), 2);
    assert_eq!(
        (log[0].new.clone(), log[0].source),
        (Some(SettingValue::Text("3d".into())), SettingsLogSource::Dashboard),
        "newest first"
    );
    // A reset of a setting without an override changes and logs nothing.
    let change = Change { set: Vec::new(), reset: vec!["pin_hold".into()] };
    assert!(recorder.change_settings(change, SETTINGS_WAIT).unwrap().is_empty());
    assert_eq!(recorder.read(|c| settings_log(c, 10)).unwrap().len(), 2);
    recorder.finish(Duration::from_secs(2));
    // A stopped recorder takes no change.
    let change = Change {
        set: vec![("pin_hold".into(), SettingValue::Text("5s".into()))],
        reset: Vec::new(),
    };
    assert!(recorder.change_settings(change, SETTINGS_WAIT).is_err());
    assert_eq!(settings.pin_hold_secs(), 60, "nothing changed");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn without_a_database_or_a_thread_nothing_changes() {
    use crate::serve::dashboard::contract::SettingValue;
    use crate::serve::runtime::Change;
    let change = || Change {
        set: vec![("pin_hold".into(), SettingValue::Text("5s".into()))],
        reset: Vec::new(),
    };
    let off = Recorder::off();
    assert_eq!(
        off.change_settings(change(), SETTINGS_WAIT),
        Err(WriteError::not_applied(
            "settings changes are kept in the recorder's database, which is off (--stats-db off); nothing changed"
        ))
    );
    assert_eq!(off.read(|c| settings_log(c, 1)), Err(Unavailable::Off));
    // Open but without its thread (the read side's tests): refused too.
    let root = temp_root("settings-threadless");
    let path = root.join("lily.sqlite");
    drop(open(&path, 5).expect("database"));
    let reading = Recorder::reading(&path, Arc::default());
    assert_eq!(
        reading.change_settings(change(), SETTINGS_WAIT),
        Err(WriteError::not_applied("the recorder is not running; nothing changed"))
    );
    assert_eq!(reading.read(|c| settings_log(c, 1)), Ok(Vec::new()));
    let _ = fs::remove_dir_all(&root);
}

/// A recorder on a fresh database at `tag`, open, persisting `settings`.
fn live(tag: &str, settings: &Arc<RuntimeSettings>) -> (PathBuf, Arc<Board>, Recorder) {
    let root = temp_root(tag);
    let board = Arc::new(Board::default());
    let recorder = Recorder::start(
        Some(&root.join("lily.sqlite")),
        &board,
        Some(settings.clone()),
    );
    assert!(eventually(|| recorder.status().state == RecorderState::Open));
    (root, board, recorder)
}

fn hold(text: &str) -> Change {
    Change {
        set: vec![("pin_hold".into(), SettingValue::Text(text.into()))],
        reset: Vec::new(),
    }
}

use crate::serve::dashboard::contract::SettingValue;
use crate::serve::runtime::{Change, testing::runtime};

#[test]
fn without_a_database_the_settings_settle_at_once() {
    let settings = runtime();
    let _off =
        Recorder::start(None, &Arc::new(Board::default()), Some(settings.clone()));
    assert!(settings.settled());
    // A first open that fails settles them too: the flags stay in effect.
    let root = temp_root("settle-failing");
    fs::write(root.join("blocker"), b"").unwrap();
    let settings = runtime();
    let board = Arc::new(Board::default());
    let recorder = Recorder::start(
        Some(&root.join("blocker").join("lily.sqlite")),
        &board,
        Some(settings.clone()),
    );
    assert!(eventually(|| settings.settled()));
    assert_eq!(recorder.status().state, RecorderState::Opening);
    recorder.finish(Duration::from_secs(2));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_timeout_tells_a_change_never_made_from_one_being_written() {
    // Nobody takes it up: abandoned, so it is never made (503).
    let (recorder, rx) = Recorder::on_channel(4);
    let got = recorder.change_settings(hold("5s"), Duration::from_millis(20));
    assert_eq!(got.as_ref().map_err(|e| e.status), Err(503));
    assert!(got.unwrap_err().message.contains("nothing changed"));
    let Ok(Msg::Settings(write)) = rx.try_recv() else { panic!("the change was sent") };
    assert!(!write.claim.take(), "the recorder may not make it any more");
    // Taken up, not answered in time: the outcome is unknown (504).
    let (recorder, rx) = Recorder::on_channel(4);
    let writer = std::thread::spawn(move || {
        let Ok(Msg::Settings(write)) = rx.recv() else { panic!("the change") };
        assert!(write.claim.take());
        std::thread::sleep(Duration::from_millis(200));
        let _ = write.reply.try_send(Ok(Vec::new()));
    });
    let got = recorder.change_settings(hold("5s"), Duration::from_millis(50));
    assert_eq!(got, Err(WriteError { status: 504, message: OUTCOME_UNKNOWN.into() }));
    writer.join().unwrap();
    // The claim decides once.
    let claim = Claim::default();
    assert!(claim.abandon());
    assert!(!claim.take() && !claim.abandon());
}

#[test]
fn a_failed_write_publishes_nothing_nudges_nothing_and_does_not_mark_the_recorder_for_good()
 {
    let settings = runtime();
    let nudges = Arc::new(AtomicU64::new(0));
    {
        let nudges = nudges.clone();
        settings.on_timers_changed(move || {
            nudges.fetch_add(1, Ordering::Relaxed);
        });
    }
    let (root, board, recorder) = live("settings-failing-write", &settings);
    // The log table gone: the transaction fails after the settings row.
    Connection::open(root.join("lily.sqlite"))
        .unwrap()
        .execute_batch("DROP TABLE settings_log")
        .unwrap();
    let got = recorder.change_settings(hold("5s"), SETTINGS_WAIT);
    let error = got.unwrap_err();
    assert_eq!(error.status, 503);
    assert!(
        error.message.starts_with("the database refused the change"),
        "{}",
        error.message
    );
    assert_eq!(settings.pin_hold_secs(), 60, "not published");
    assert_eq!(nudges.load(Ordering::Relaxed), 0, "no nudge");
    assert!(recorder.read(stored_settings).unwrap().is_empty(), "rolled back");
    assert!(!recorder.status().ok);
    // The next write that succeeds, of any kind, clears it.
    board.record(row(super::super::now_ms(), None, 1.0, 1, 1.0));
    assert!(eventually(|| recorder.status().ok), "a request row's write clears it");
    recorder.finish(Duration::from_secs(2));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn two_changes_of_one_setting_queue_up_with_their_own_old_and_new() {
    let settings = runtime();
    let (root, _board, recorder) = live("settings-two-changes", &settings);
    std::thread::scope(|scope| {
        for text in ["5s", "7s"] {
            let recorder = &recorder;
            scope.spawn(move || {
                recorder.change_settings(hold(text), SETTINGS_WAIT).unwrap()
            });
        }
    });
    let mut log = recorder.read(|c| settings_log(c, 10)).unwrap();
    log.reverse();
    assert_eq!(log.len(), 2);
    // Whichever came first, each starts where the one before ended.
    assert_eq!(log[0].old, Some(SettingValue::Text("1m".into())));
    assert_eq!(log[1].old, log[0].new);
    assert_eq!(Some(settings.overrides()["pin_hold"].clone()), log[1].new);
    recorder.finish(Duration::from_secs(2));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn settings_that_cannot_be_read_leave_the_flags_and_refuse_changes() {
    let root = temp_root("settings-unreadable");
    let path = root.join("lily.sqlite");
    {
        let (conn, _) = open(&path, 5).expect("database");
        conn.execute_batch("DROP TABLE settings").unwrap();
    }
    let settings = runtime();
    let board = Arc::new(Board::default());
    let recorder = Recorder::start(Some(&path), &board, Some(settings.clone()));
    assert!(eventually(|| recorder.status().state == RecorderState::Open));
    assert!(settings.settled() && settings.overrides().is_empty());
    assert_eq!(
        recorder.change_settings(hold("5s"), SETTINGS_WAIT),
        Err(WriteError::not_applied(
            "settings could not be read from the database (see the server log); nothing changed"
        ))
    );
    // The history still works.
    assert!(recorder.read(oldest).is_ok());
    recorder.finish(Duration::from_secs(2));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_reset_deletes_a_stale_row_the_start_ignored() {
    let root = temp_root("settings-stale-reset");
    let path = root.join("lily.sqlite");
    {
        let (conn, _) = open(&path, 5).expect("database");
        conn.execute(
            "INSERT INTO settings (name, value, changed_t) VALUES ('top_p', '5', 1)",
            [],
        )
        .unwrap();
    }
    let settings = runtime();
    let board = Arc::new(Board::default());
    let recorder = Recorder::start(Some(&path), &board, Some(settings.clone()));
    assert!(eventually(|| recorder.status().state == RecorderState::Open));
    assert!(settings.overrides().is_empty(), "top_p 5 was ignored");
    let reset = Change { set: Vec::new(), reset: vec!["top_p".into()] };
    assert_eq!(recorder.change_settings(reset, SETTINGS_WAIT), Ok(Vec::new()));
    assert!(recorder.read(stored_settings).unwrap().is_empty(), "the row is gone");
    assert!(
        recorder.read(|c| settings_log(c, 10)).unwrap().is_empty(),
        "nothing changed"
    );
    recorder.finish(Duration::from_secs(2));
    let _ = fs::remove_dir_all(&root);
}

/// [`window`] with no system power: the SMC's key failed.
fn window_without_system_power(start_ms: u64, end_ms: u64, gpu_w: f64) -> Sample {
    let mut s = window(start_ms, end_ms, 0.0);
    s.reading.sys_w = None;
    s.reading.gpu_w = gpu_w;
    s
}

#[test]
fn a_window_without_system_power_covers_the_gpu_series_but_not_the_energy() {
    // 40 s at 20 W, then 20 s with no system power (the GPU at 8 W).
    let parts = spread(&[
        window(T0, T0 + 40_000, 20.0),
        window_without_system_power(T0 + 40_000, T0 + 60_000, 8.0),
    ]);
    let p = parts[&M];
    assert_eq!((p.covered_ms, p.sys_ms), (60_000, 40_000));
    assert_eq!(p.sys_w_mean(), Some(20.0), "over the time that had it");
    assert!(close(p.energy_wh(), 20.0 * 40.0 / 3600.0), "{:?}", p.energy_wh());
    // The GPU's figures cover the whole minute: (5 x 40 + 8 x 20) / 60.
    assert!(close(p.gpu_w_mean(), 6.0), "{p:?}");
    // A minute with no system power at all has none, and energy is None.
    let none = spread(&[window_without_system_power(T0, T0 + 30_000, 8.0)])[&M];
    assert_eq!((none.covered_ms, none.sys_ms), (30_000, 0));
    assert_eq!(
        (none.sys_w_mean(), none.sys_w_min(), none.energy_wh()),
        (None, None, None)
    );
    assert!(close(none.gpu_w_mean(), 8.0));

    let mut w = writer();
    assert!(w.power(M, &parts[&M]));
    assert!(w.power(M + 1, &none));
    let row = power_columns(&w, M).expect("row");
    assert_eq!((row.covered_ms, row.sys_covered_ms), (Some(60_000), Some(40_000)));
    assert!(close(row.sys_w.1, 20.0));
    let row = power_columns(&w, M + 1).expect("row");
    assert_eq!((row.covered_ms, row.sys_covered_ms), (Some(30_000), Some(0)));
    assert_eq!(row.sys_w, (None, None, None), "unknown, not 0 W");
    assert!(close(row.gpu_w, 8.0));
    // Another part with system power joins a minute that had none.
    assert!(w.power(
        M + 1,
        &spread(&[window(T0 + 60_000 + 30_000, T0 + 60_000 + 40_000, 50.0)])[&(M + 1)]
    ));
    let row = power_columns(&w, M + 1).expect("row");
    assert_eq!((row.covered_ms, row.sys_covered_ms), (Some(40_000), Some(10_000)));
    assert!(close(row.sys_w.1, 50.0), "{row:?}");
    assert!(close(row.gpu_w, (8.0 * 30.0 + 12.5 * 10.0) / 40.0), "{row:?}");
}

#[test]
fn a_minute_never_holds_more_than_sixty_seconds_of_coverage() {
    // Windows that overlap (a clock that jumped back): 50 s, then 30 s of
    // the same minute again.
    let p = spread(&[
        window(T0, T0 + 50_000, 10.0),
        window(T0 + 20_000, T0 + 50_000, 30.0),
    ])[&M];
    // They claim 80 s of a minute; it holds 60.
    assert_eq!((p.covered_ms, p.sys_ms), (60_000, 60_000));
    assert!(close(p.sys_w_mean(), (10.0 * 50.0 + 30.0 * 10.0) / 60.0), "{p:?}");
    let full =
        spread(&[window(T0, T0 + 60_000, 10.0), window(T0, T0 + 60_000, 30.0)])[&M];
    assert_eq!((full.covered_ms, full.sys_ms), (60_000, 60_000));
    assert!(close(full.sys_w_mean(), 10.0), "the first claimed it: {full:?}");
    assert!(close(full.energy_wh(), 10.0 * 60.0 / 3600.0));
    // The same across ticks: the stored row is full, a late overlap adds nothing.
    let mut w = writer();
    assert!(w.power(M, &spread(&[window(T0, T0 + 60_000, 10.0)])[&M]));
    assert!(w.power(M, &spread(&[window(T0 + 10_000, T0 + 40_000, 90.0)])[&M]));
    let row = power_columns(&w, M).expect("row");
    assert_eq!((row.covered_ms, row.sys_covered_ms), (Some(60_000), Some(60_000)));
    assert!(close(row.sys_w.1, 10.0), "{row:?}");
    // A part that half fits is taken in proportion: 40 s stored, 40 s offered,
    // 20 s of room.
    assert!(
        w.power(M + 1, &spread(&[window(T0 + 60_000, T0 + 100_000, 10.0)])[&(M + 1)])
    );
    assert!(
        w.power(M + 1, &spread(&[window(T0 + 60_000, T0 + 100_000, 40.0)])[&(M + 1)])
    );
    let row = power_columns(&w, M + 1).expect("row");
    assert_eq!(row.covered_ms, Some(60_000));
    assert!(close(row.sys_w.1, (10.0 * 40.0 + 40.0 * 20.0) / 60.0), "{row:?}");
    assert_eq!((row.sys_w.0, row.sys_w.2), (Some(10.0), Some(40.0)));
}

// --- the cadence ----------------------------------------------------------------

#[test]
fn the_machine_is_sampled_every_minute_only_while_lily_is_quiet() {
    let s = Duration::from_secs;
    assert_eq!((SAMPLE_EVERY, QUIET_SAMPLE_EVERY), (s(10), s(60)));
    // Unloaded, no page for longer than it stays open, the sampler idle.
    assert_eq!(sample_every(true, None, false), QUIET_SAMPLE_EVERY);
    assert_eq!(sample_every(true, Some(s(3600)), false), QUIET_SAMPLE_EVERY);
    assert_eq!(
        sample_every(true, Some(PAGE_OPEN + Duration::from_millis(1)), false),
        QUIET_SAMPLE_EVERY
    );
    // Any of the three otherwise brings the 10 s cadence back.
    assert_eq!(sample_every(false, None, false), SAMPLE_EVERY);
    assert_eq!(sample_every(true, Some(s(3)), false), SAMPLE_EVERY);
    assert_eq!(sample_every(true, Some(PAGE_OPEN), false), SAMPLE_EVERY);
    assert_eq!(sample_every(true, None, true), SAMPLE_EVERY);
}

#[test]
fn the_recorders_cadence_follows_the_engine_the_page_and_the_sampler() {
    use crate::serve::dashboard::Loaded;
    use crate::serve::machine::Intervals;
    let board = Arc::new(Board::default());
    let machine = Machine::running();
    board.attach_machine(machine.clone());
    let (_tx, rx) = mpsc::sync_channel(1);
    let recording = Recording {
        path: PathBuf::new(),
        health: Arc::default(),
        rows: rx,
        board: Arc::downgrade(&board),
        started_ms: 0,
        machine: None,
        ticks: None,
        settings: None,
        changes: Vec::new(),
    };
    let at = Instant::now();
    let every = |r: &Recording| r.next_sample(Some(at)) - at;
    assert!(recording.next_sample(None) <= Instant::now(), "the first at once");
    // The first load has not finished: busy.
    assert_eq!(every(&recording), SAMPLE_EVERY);
    let loaded = Loaded { max_batch: 1, weights_bytes: 0, ngram: None };
    board.loaded(loaded.clone());
    assert_eq!(every(&recording), SAMPLE_EVERY, "loaded");
    board.unloaded(StoreView::default());
    assert_eq!(every(&recording), QUIET_SAMPLE_EVERY, "unloaded, nobody looking");
    // A request open (a reload's for it, say) is not quiet.
    board.opened();
    assert_eq!(every(&recording), SAMPLE_EVERY);
    board.closed(None, Instant::now());
    assert_eq!(every(&recording), QUIET_SAMPLE_EVERY);
    // The page asked.
    machine.touch();
    assert_eq!(every(&recording), SAMPLE_EVERY, "the page is open");
    // The sampler at its busy cadence (a reload has no request yet).
    let quiet_board = Arc::new(Board::default());
    quiet_board.loaded(loaded);
    quiet_board.unloaded(StoreView::default());
    let busy = Machine::spawn(
        || Ok(Blank),
        || true,
        Intervals { first: Duration::from_millis(1), ..Intervals::default() },
    );
    quiet_board.attach_machine(busy.clone());
    let recording = Recording { board: Arc::downgrade(&quiet_board), ..recording };
    assert!(eventually(|| busy.short_cadence()));
    assert_eq!(every(&recording), SAMPLE_EVERY, "the sampler is busy");
    busy.finish(Duration::from_secs(5));
    assert!(!busy.short_cadence());
    assert_eq!(every(&recording), QUIET_SAMPLE_EVERY);
}

/// A machine source that waits out what it is asked for and reads nothing.
struct Blank;

impl crate::serve::machine::Source for Blank {
    fn read(&mut self, window: Duration) -> Result<Reading, String> {
        std::thread::sleep(window);
        Ok(Reading::default())
    }
}
