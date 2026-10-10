use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::PathBuf;

use super::contract::*;
use super::settings::{self, FlagValue};
use super::*;
use crate::serve::State;
use crate::serve::timings::{BatchTimings, Speculation};

// --- the app's files ---------------------------------------------------------

/// The embedded file whose path starts with `prefix` and ends with `suffix`
/// (the hashed names change with every build).
fn embedded_name(prefix: &str, suffix: &str) -> &'static str {
    assets::FILES
        .iter()
        .map(|(path, _, _)| *path)
        .find(|path| path.starts_with(prefix) && path.ends_with(suffix))
        .unwrap_or_else(|| panic!("no {prefix}*{suffix} in dashboard/dist"))
}

#[test]
fn the_app_is_embedded_with_its_content_types_and_cache_headers() {
    let index = page("/").expect("index");
    assert_eq!(index.content_type, "text/html; charset=utf-8");
    assert_eq!(index.cache_control, "no-cache");
    assert!(std::str::from_utf8(index.bytes).expect("utf-8").contains("<html"));
    assert_eq!(page("/index.html"), Some(index));

    let js = embedded_name("assets/index-", ".js");
    let asset = page(&format!("/{js}")).expect("js");
    assert_eq!(asset.content_type, "text/javascript; charset=utf-8");
    assert_eq!(asset.cache_control, "public, max-age=31536000, immutable");
    let css =
        page(&format!("/{}", embedded_name("assets/index-", ".css"))).expect("css");
    assert_eq!(css.content_type, "text/css; charset=utf-8");
    let font = page(&format!("/{}", embedded_name("assets/", ".woff2"))).expect("font");
    assert_eq!(font.content_type, "font/woff2");
    // Everything under /assets/ is cached for good, but the one file whose
    // name carries no content hash.
    for (path, _, _) in
        assets::FILES.iter().filter(|(p, _, _)| p.starts_with("assets/"))
    {
        let asset = page(&format!("/{path}")).expect(path);
        let expected =
            if *path == "assets/licenses.txt" { "no-cache" } else { IMMUTABLE };
        assert_eq!(asset.cache_control, expected, "{path}");
    }
    let licenses = page("/assets/licenses.txt").expect("licenses");
    assert_eq!(licenses.content_type, "text/plain; charset=utf-8");
}

#[test]
fn app_routes_fall_back_to_the_index_and_files_match_exactly() {
    let index = page("/").expect("index").bytes;
    for path in ["/sessions", "/sessions/42", "/settings/", "/engine/v2"] {
        assert_eq!(page(path).expect(path).bytes, index, "{path}");
    }
    // A path naming a file is looked up as is: no traversal, no decoding, no
    // normalisation, and a miss is a 404 rather than the index.
    for path in [
        "/favicon.ico",
        "/x.js",
        "/robots.txt",
        "/sessions/report.json",
        "/assets/missing-00000000.js",
        "/assets/../index.html",
        "/assets/..%2findex.html",
        "/assets/%2e%2e/index.html",
        "/assets//licenses.txt",
        "/assets/./licenses.txt",
        "/assets/",
        "/assets/LICENSES.TXT",
        "/../index.html",
    ] {
        assert_eq!(page(path), None, "{path}");
    }
}

#[test]
fn the_dashboard_routes_leave_the_api_and_health_alone() {
    for method in ["GET", "HEAD"] {
        assert_eq!(route(method, "/"), Some(Route::Page), "{method}");
        assert_eq!(route(method, "/assets/index-vgMbpbb6.js"), Some(Route::Page));
        assert_eq!(route(method, "/sessions/7"), Some(Route::Page));
    }
    assert_eq!(route("GET", "/healthz"), Some(Route::Page));
    for method in ["GET", "HEAD", "POST", "PATCH", "DELETE", "PUT"] {
        assert_eq!(route(method, "/v1/lily/status"), Some(Route::Api), "{method}");
        assert_eq!(route(method, "/v1/lily/nope"), Some(Route::Api), "{method}");
    }
    assert_eq!(route("GET", "/v1/lily"), Some(Route::Api));
    for (method, path) in [
        ("GET", "/health"),
        ("HEAD", "/health"),
        ("GET", "/v1/models"),
        ("GET", "/v1/timings"),
        ("GET", "/v1"),
        ("GET", "/v1/lilyx"),
        ("GET", "/v1/unknown"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/"),
        ("GET", "*"),
    ] {
        assert_eq!(route(method, path), None, "{method} {path}");
    }
}

// --- the guards --------------------------------------------------------------

fn at(ip: IpAddr) -> Option<SocketAddr> {
    Some(SocketAddr::new(ip, 50_000))
}

#[test]
fn only_loopback_peers_count_as_this_machine() {
    assert!(is_loopback(at(IpAddr::V4(Ipv4Addr::LOCALHOST))));
    assert!(is_loopback(at(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2)))));
    assert!(is_loopback(at(IpAddr::V6(Ipv6Addr::LOCALHOST))));
    // A dual-stack listener reports an IPv4 peer as IPv4-mapped.
    assert!(is_loopback(at(IpAddr::V6(Ipv4Addr::LOCALHOST.to_ipv6_mapped()))));
    assert!(!is_loopback(at(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 20)))));
    assert!(!is_loopback(at(IpAddr::V6(Ipv4Addr::new(10, 0, 0, 1).to_ipv6_mapped()))));
    assert!(!is_loopback(at(IpAddr::V4(Ipv4Addr::UNSPECIFIED))));
    assert!(!is_loopback(None), "an unknown peer is refused");
}

#[test]
fn only_lilys_own_loopback_names_pass_as_host() {
    for host in ["127.0.0.1:8000", "localhost:8000", "LocalHost:8000", "[::1]:8000"] {
        assert!(host_allowed(Some(host), 8000), "{host}");
    }
    // A rebinding page sends its own name; a missing port or another one,
    // a missing header, or anything else is refused.
    for host in [
        None,
        Some("evil.example:8000"),
        Some("localhost"),
        Some("127.0.0.1"),
        Some("localhost:9000"),
        Some("127.0.0.2:8000"),
        Some("localhost.evil.example:8000"),
        Some("[::1]"),
        Some(""),
    ] {
        assert!(!host_allowed(host, 8000), "{host:?}");
    }
}

#[test]
fn the_write_guard_wants_json_and_lilys_own_origin() {
    let json = Some("application/json");
    assert_eq!(write_guard(json, None, 8000), Ok(()));
    assert_eq!(
        write_guard(Some("application/json; charset=utf-8"), None, 8000),
        Ok(())
    );
    assert_eq!(
        write_guard(Some(" Application/JSON ;charset=UTF-8"), None, 8000),
        Ok(())
    );
    for origin in
        ["http://127.0.0.1:8000", "http://localhost:8000", "http://[::1]:8000"]
    {
        assert_eq!(write_guard(json, Some(origin), 8000), Ok(()), "{origin}");
    }
    // What a cross-site form or fetch without a preflight can send.
    for content_type in
        [None, Some("text/plain"), Some("application/x-www-form-urlencoded")]
    {
        assert!(write_guard(content_type, None, 8000).is_err(), "{content_type:?}");
    }
    assert!(write_guard(Some("application/jsonp"), None, 8000).is_err());
    for origin in [
        "http://evil.example",
        "http://localhost:9000",
        "https://localhost:8000",
        "http://127.0.0.1:8000.evil.example",
        "http://[::1]:9000",
        "null",
    ] {
        assert!(write_guard(json, Some(origin), 8000).is_err(), "{origin}");
    }
}

// --- answers -----------------------------------------------------------------

/// A request as a browser on this machine sends it: `Host` set.
fn request(method: &str, path: &str, headers: &[(&str, &str)]) -> Request {
    let mut all = vec![("Host".to_owned(), "127.0.0.1:8000".to_owned())];
    all.extend(headers.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())));
    Request {
        method: method.to_owned(),
        http11: true,
        path: path.to_owned(),
        headers: all,
        body: Vec::new(),
    }
}

fn config() -> Config {
    Config {
        model: "qwen3.8-flash-next",
        idle_unload_secs: 1800,
        pin_hold_secs: 60,
        max_batch: 4,
        queue: 32,
        max_sessions: 16,
        cache_bytes: None,
        settings: settings::build(&[], &[]),
    }
}

fn dashboard() -> (Arc<Board>, Dashboard) {
    let board = Arc::new(Board::default());
    (board.clone(), Dashboard::new(board, config()))
}

const READY: EngineNow = EngineNow { state: State::Ready, waiting: 0 };
const LOCAL: Connection = Connection {
    peer: Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 50_000)),
    port: 8000,
};

/// `method target` from this machine, with `headers`.
fn answer(
    d: &Dashboard,
    method: &str,
    target: &str,
    headers: &[(&str, &str)],
) -> Response {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let route = route(method, path).expect("a dashboard route");
    d.respond(route, &request(method, path, headers), (path, query), LOCAL, READY)
}

fn body(response: &Response) -> serde_json::Value {
    serde_json::from_slice(&response.body).expect("json")
}

const ROUTES: [(&str, &str); 6] = [
    ("GET", "/"),
    ("GET", "/assets/licenses.txt"),
    ("GET", "/sessions"),
    ("GET", "/v1/lily/status"),
    ("GET", "/v1/lily/nope"),
    ("PATCH", "/v1/lily/settings"),
];

#[test]
fn every_dashboard_route_refuses_a_peer_that_is_not_loopback() {
    let (_, d) = dashboard();
    let remote =
        Connection { peer: at(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 20))), port: 8000 };
    for (method, target) in ROUTES {
        let route = route(method, target).expect("route");
        let r = d.respond(
            route,
            &request(method, target, &[]),
            (target, ""),
            remote,
            READY,
        );
        assert_eq!(r.status, 403, "{method} {target}");
        assert!(body(&r)["error"]["message"].is_string());
    }
    // Logged once for the peer, not once per request.
    assert_eq!(lock(&d.refused).len(), 1);
}

#[test]
fn every_dashboard_route_refuses_a_host_that_is_not_lilys_own() {
    let (_, d) = dashboard();
    for host in [None, Some("rebound.evil.example:8000"), Some("localhost:9000")] {
        for (method, target) in ROUTES {
            let mut request = request(method, target, &[]);
            request.headers.retain(|(k, _)| k != "Host");
            if let Some(host) = host {
                request.headers.push(("Host".into(), host.into()));
            }
            let route = route(method, target).expect("route");
            let r = d.respond(route, &request, (target, ""), LOCAL, READY);
            assert_eq!(r.status, 403, "{method} {target} {host:?}");
        }
    }
    assert_eq!(answer(&d, "GET", "/", &[]).status, 200, "the page's own host passes");
}

#[test]
fn every_answer_forbids_framing_and_sniffing_and_carries_the_policy() {
    let (_, d) = dashboard();
    for target in ["/", "/assets/licenses.txt", "/v1/lily/status", "/v1/lily/nope"] {
        let headers = answer(&d, "GET", target, &[]).headers();
        let get =
            |name: &str| headers.iter().find(|(k, _)| *k == name).map(|(_, v)| *v);
        assert_eq!(get("X-Frame-Options"), Some("DENY"), "{target}");
        assert_eq!(get("X-Content-Type-Options"), Some("nosniff"));
        let csp = get("Content-Security-Policy").expect("csp");
        for directive in [
            "default-src 'self'",
            "script-src 'self'",
            "frame-ancestors 'none'",
            "img-src 'self' data:",
            "object-src 'none'",
        ] {
            assert!(csp.contains(directive), "{directive} in {csp}");
        }
        assert!(get("Cache-Control").is_some());
    }
    // The policy the built app runs under: no inline script, its favicon a
    // data: URI, its scripts and styles from /assets/.
    let index = std::str::from_utf8(page("/").expect("index").bytes).expect("utf-8");
    assert!(!index.contains("<script>") && !index.contains("<script type=\"module\">"));
    assert!(index.contains("href=\"data:image/svg+xml"));
    assert!(index.contains("src=\"/assets/"));
}

#[test]
fn a_head_is_answered_as_its_get() {
    let (_, d) = dashboard();
    for target in ["/", "/assets/licenses.txt", "/v1/lily/status", "/v1/lily/settings"]
    {
        let get = answer(&d, "GET", target, &[]);
        let head = answer(&d, "HEAD", target, &[]);
        assert_eq!(
            (head.status, head.content_type, head.cache_control),
            (get.status, get.content_type, get.cache_control),
            "{target}"
        );
    }
}

#[test]
fn pages_carry_their_cache_headers_and_the_api_is_never_cached() {
    let (_, d) = dashboard();
    let index = answer(&d, "GET", "/sessions/3", &[]);
    assert_eq!((index.status, index.cache_control), (200, "no-cache"));
    assert_eq!(index.content_type, "text/html; charset=utf-8");
    let missing = answer(&d, "GET", "/assets/gone-12345678.js", &[]);
    assert_eq!((missing.status, missing.content_type), (404, "application/json"));
    assert_eq!(answer(&d, "GET", "/favicon.ico", &[]).status, 404);
    let status = answer(&d, "GET", "/v1/lily/status", &[]);
    assert_eq!((status.status, status.cache_control), (200, "no-store"));
}

#[test]
fn an_answer_that_cannot_be_serialized_is_a_500_with_an_error_body() {
    struct Broken;
    impl Serialize for Broken {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("broken"))
        }
    }
    let r = Response::json(200, &Broken);
    assert_eq!((r.status, r.content_type), (500, "application/json"));
    assert!(body(&r)["error"]["message"].is_string());
}

#[test]
fn the_api_answers_its_routes_and_says_what_it_cannot_do_yet() {
    let (_, d) = dashboard();
    let get = |target: &str| answer(&d, "GET", target, &[]);
    for target in [
        "/v1/lily/status",
        "/v1/lily/sessions",
        "/v1/lily/requests",
        "/v1/lily/requests?limit=5",
        "/v1/lily/settings",
        "/v1/lily/settings/log",
    ] {
        assert_eq!(get(target).status, 200, "{target}");
    }
    assert_eq!(body(&get("/v1/lily/settings/log")), serde_json::json!({"entries": []}));
    let history = get("/v1/lily/history?range=1h");
    assert_eq!(history.status, 503);
    assert_eq!(body(&history)["error"]["message"], "history needs the recorder");
    assert_eq!(get("/v1/lily/sessions/12/history?range=15m").status, 503);
    assert_eq!(get("/v1/lily/sessions/abc/history").status, 404);
    assert_eq!(get("/v1/lily/requests?limit=-1").status, 400);
    let unknown = get("/v1/lily/nope");
    assert_eq!(unknown.status, 404);
    assert_eq!(unknown.content_type, "application/json");
    assert_eq!(body(&unknown)["error"]["message"], "not found");
    assert_eq!(answer(&d, "POST", "/v1/lily/status", &[]).status, 405);
    assert_eq!(
        answer(
            &d,
            "DELETE",
            "/v1/lily/history",
            &[("Content-Type", "application/json")]
        )
        .status,
        405
    );
}

#[test]
fn settings_writes_pass_the_guard_first_and_are_refused_until_phase_four() {
    let (_, d) = dashboard();
    let json = ("Content-Type", "application/json; charset=utf-8");
    let own = ("Origin", "http://localhost:8000");
    let patch = answer(&d, "PATCH", "/v1/lily/settings", &[json, own]);
    assert_eq!(patch.status, 409);
    assert_eq!(
        body(&patch)["error"]["message"],
        "runtime settings are not editable yet"
    );
    let ipv6 = answer(
        &d,
        "PATCH",
        "/v1/lily/settings",
        &[json, ("Origin", "http://[::1]:8000")],
    );
    assert_eq!(ipv6.status, 409);
    let reset = answer(&d, "DELETE", "/v1/lily/settings/temperature", &[json]);
    assert_eq!(reset.status, 409);
    assert_eq!(body(&reset)["error"]["setting"], "temperature");
    let unknown = answer(&d, "DELETE", "/v1/lily/settings/nope", &[json]);
    assert_eq!(
        (unknown.status, body(&unknown)["error"]["setting"].as_str()),
        (404, Some("nope"))
    );
    // The guard comes before anything else, the route's own answer included.
    for headers in [
        vec![],
        vec![("Content-Type", "text/plain")],
        vec![json, ("Origin", "http://evil.example")],
        vec![json, ("Origin", "http://localhost:9999")],
    ] {
        assert_eq!(answer(&d, "PATCH", "/v1/lily/settings", &headers).status, 403);
        assert_eq!(
            answer(&d, "DELETE", "/v1/lily/settings/nope", &headers).status,
            403
        );
    }
}

#[test]
fn the_request_limit_defaults_caps_and_refuses_garbage() {
    assert_eq!(request_limit(""), Ok(40));
    assert_eq!(request_limit("x=1"), Ok(40));
    assert_eq!(request_limit("limit=7"), Ok(7));
    assert_eq!(request_limit("a=b&limit=0"), Ok(0));
    assert_eq!(request_limit("limit=100000"), Ok(200));
    assert!(request_limit("limit=ten").is_err());
    assert!(request_limit("limit=").is_err());
}

// --- what the engine publishes -----------------------------------------------

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
        decode_tps: (generated > 0).then(|| generated as f64 / (decode_ms / 1e3)),
        drafted_tokens: Some(20),
        accepted_tokens: Some(15),
        queue_ms: 1.5,
        batch: 1,
        cancelled_by: None,
        thinking_forced: false,
        pinned: true,
        failed: false,
    }
}

#[test]
fn request_rows_come_from_the_timings_and_hold_numbers_only() {
    let timings = Timings::measure(
        24_538,
        8_538,
        16.0,
        19,
        0.19,
        Some(Speculation { drafted: 10, accepted: 7 }),
    )
    .with_cancel(Some("client"), None)
    .with_pinned(true);
    let r = request_row(1_760_000_000_000, Some(3), &timings, Some(ClosedBy::Budget));
    assert_eq!(
        (r.t, r.lineage, r.prompt_tokens, r.cached_tokens),
        (1_760_000_000_000, Some(3), 24_538, 8_538)
    );
    assert_eq!(
        (r.prefill_tokens, r.prefill_tps, r.decode_tps),
        (16_000, Some(1000.0), Some(100.0))
    );
    assert_eq!((r.drafted_tokens, r.accepted_tokens), (Some(10), Some(7)));
    assert_eq!(
        (r.batch, r.cancelled_by, r.thinking_forced, r.pinned),
        (1, Some(CancelledBy::Client), true, true)
    );
    // A request that shared steps reports the most rows it shared one with.
    let batched =
        timings.with_batch(BatchTimings { max_rows: 3, ..BatchTimings::default() });
    assert_eq!(request_row(0, None, &batched, None).batch, 3);
    // Only the budget's close counts as forced; a tool call is the model's.
    assert!(request_row(0, None, &timings, Some(ClosedBy::EndOfTurn)).thinking_forced);
    assert!(!request_row(0, None, &timings, Some(ClosedBy::ToolCall)).thinking_forced);
    assert!(!request_row(0, None, &timings, None).thinking_forced);
    // A cancellation the contract does not know is left out, not guessed.
    let odd = timings.with_cancel(Some("watchdog"), None);
    assert_eq!(request_row(0, None, &odd, None).cancelled_by, None);
    assert!(!r.failed);
}

#[test]
fn a_failed_request_still_counts() {
    let (board, d) = dashboard();
    board.record(failed_row(super::super::now_ms(), Some(9)));
    let rows = d.requests(10).requests;
    assert_eq!(rows.len(), 1);
    let r = &rows[0];
    assert!(r.failed);
    assert_eq!(
        (r.lineage, r.prompt_tokens, r.decode_tps, r.cancelled_by),
        (Some(9), 0, None, None)
    );
    assert_eq!(d.status(READY).live.requests_per_min, 1);
    let f = figures(rows.iter());
    assert_eq!((f.requests, f.decode_tps_avg, f.cache_hit), (1, None, None));
}

#[test]
fn the_request_ring_keeps_the_newest_rows_and_serves_them_newest_first() {
    let (board, d) = dashboard();
    for i in 0..(REQUEST_ROWS as u64 + 5) {
        board.record(row(i, None, 1, 10.0));
    }
    let all = d.requests(MAX_REQUEST_LIMIT).requests;
    assert_eq!(all.len(), MAX_REQUEST_LIMIT);
    assert_eq!(all[0].t, REQUEST_ROWS as u64 + 4, "newest first");
    assert_eq!(lock(&board.rows).len(), REQUEST_ROWS);
    assert_eq!(lock(&board.rows).front().map(|r| r.t), Some(5), "oldest dropped");
    assert!(d.requests(0).requests.is_empty());
}

#[test]
fn per_session_figures_are_busy_time_averages_and_the_best_minute() {
    let minute = 60_000;
    let rows = [
        // Minute 0: 100 tokens in 2 s and 100 in 2 s: 50 tok/s.
        row(10, Some(1), 100, 2000.0),
        row(20, Some(1), 100, 2000.0),
        // Minute 1: one short burst at 200 tok/s.
        row(minute + 5, Some(1), 20, 100.0),
        // A prefill that was all cache hit: no prefill time to count.
        RequestRow {
            prefill_tokens: 0,
            prefill_ms: 0.0,
            ..row(minute + 9, Some(1), 0, 0.0)
        },
    ];
    let f = figures(rows.iter());
    assert_eq!(f.requests, 4);
    // 220 tokens over 4.1 s of decoding, not over the wall time.
    assert_eq!(f.decode_tps_avg, Some(53.66));
    assert_eq!(f.decode_tps_max, Some(200.0));
    // 300 tokens over 150 ms.
    assert_eq!(f.prefill_tps_avg, Some(2000.0));
    assert_eq!((f.cache_hit, f.acceptance), (Some(0.9), Some(0.75)));
    assert_eq!(figures(std::iter::empty()), Figures::default());
}

fn entry(id: Option<u64>, on_disk: bool, last_used_ms: u64) -> EntryView {
    EntryView {
        lineage: id.map(|id| Lineage { id, created_ms: 1000 + id }),
        on_disk,
        durable: false,
        tokens: 300,
        bytes: 3_000,
        last_used_ms,
    }
}

#[test]
fn sessions_list_the_running_ones_first_then_the_cache_newest_first() {
    let (board, d) = dashboard();
    board.publish(StoreView {
        gpu: GpuTier {
            used_bytes: 6_000,
            budget_bytes: Some(10_000),
            entries: 2,
            max_entries: 16,
        },
        disk: Some(DiskTier {
            used_bytes: 9_000,
            budget_bytes: 50_000,
            entries: 3,
            ..DiskTier::default()
        }),
        entries: vec![
            entry(Some(1), false, 500),
            entry(Some(2), false, 900),
            entry(None, true, 100),
            // The stale disk copy of the session request 7 holds.
            entry(Some(7), true, 50),
            entry(Some(8), true, 700),
        ],
        experts: None,
    });
    board.opened();
    board.admitted(Lineage { id: 7, created_ms: 1007 }, 1200, 4_000);
    board.record(row(super::super::now_ms(), Some(7), 50, 1000.0));

    let sessions = d.sessions().sessions;
    let order: Vec<(Option<u64>, SessionTier, bool)> =
        sessions.iter().map(|s| (s.lineage, s.tier, s.active)).collect();
    assert_eq!(
        order,
        vec![
            (Some(7), SessionTier::Gpu, true),
            (Some(2), SessionTier::Gpu, false),
            (Some(8), SessionTier::Disk, false),
            (Some(1), SessionTier::Gpu, false),
            (None, SessionTier::Disk, false),
        ]
    );
    let running = &sessions[0];
    assert_eq!((running.tokens, running.bytes, running.requests), (1200, 4_000, 1));
    assert_eq!(running.decode_tps_avg, Some(50.0));
    assert_eq!(running.created_t, Some(1007));
    // An entry from before lineage ids: no figures, no creation time.
    let legacy = &sessions[4];
    assert_eq!(
        (legacy.requests, legacy.decode_tps_avg, legacy.created_t),
        (0, None, None)
    );

    // The GPU tier counts the running session with the resident ones.
    let status = d.status(READY);
    assert_eq!((status.tiers.gpu.used_bytes, status.tiers.gpu.entries), (10_000, 3));
    assert_eq!(
        (status.state, status.queue.running, status.live.active_sessions),
        (LifecycleState::Busy, 1, 1)
    );
    assert_eq!((status.idle_for_s, status.live.requests_per_min), (0, 1));
    assert_eq!(status.live.acceptance, Some(0.75));

    board.closed(Some(7), Instant::now());
    let status = d.status(READY);
    assert_eq!(
        (status.state, status.queue.running, status.live.active_sessions),
        (LifecycleState::Ready, 0, 0)
    );
    assert_eq!(
        (status.live.decode_tps, status.live.acceptance),
        (0.0, None),
        "idle: zero and null"
    );
    assert!(d.sessions().sessions.iter().all(|s| !s.active));
}

#[test]
fn a_session_both_published_and_running_is_counted_once() {
    // Between an admission and its publish (or a release's publish and the
    // close), the running session is also in the published cache.
    let (board, d) = dashboard();
    board.publish(StoreView {
        gpu: GpuTier {
            used_bytes: 7_000,
            budget_bytes: Some(20_000),
            entries: 2,
            max_entries: 16,
        },
        entries: vec![entry(Some(7), false, 900), entry(Some(1), false, 500)],
        ..StoreView::default()
    });
    board.opened();
    board.admitted(Lineage { id: 7, created_ms: 1007 }, 1200, 4_000);
    let gpu = d.status(READY).tiers.gpu;
    assert_eq!((gpu.used_bytes, gpu.entries), (7_000, 2));
    let sessions = d.sessions().sessions;
    assert_eq!(sessions.iter().filter(|s| s.lineage == Some(7)).count(), 1);
    assert!(sessions[0].active);
}

#[test]
fn before_the_first_load_the_cache_shows_the_command_lines_limits() {
    let board = Arc::new(Board::default());
    let d = Dashboard::new(board.clone(), config());
    let gpu = d.status(READY).tiers.gpu;
    assert_eq!((gpu.max_entries, gpu.budget_bytes), (16, None), "derived at the load");
    let given = Config { cache_bytes: Some(24 << 30), ..config() };
    let d = Dashboard::new(Arc::new(Board::default()), given);
    assert_eq!(d.status(READY).tiers.gpu.budget_bytes, Some(24 << 30));
}

#[test]
fn the_ngram_tier_follows_the_loaded_table() {
    let (board, d) = dashboard();
    assert_eq!(d.status(READY).tiers.ngram, None, "nothing loaded");
    board.loaded(Loaded { max_batch: 4, weights_bytes: 1, ngram: None });
    assert_eq!(d.status(READY).tiers.ngram, None, "a model without a table");
    let resident = NgramFacts { bytes: 30 << 30, paged: None };
    board.loaded(Loaded { max_batch: 4, weights_bytes: 1, ngram: Some(resident) });
    assert_eq!(
        d.status(READY).tiers.ngram,
        Some(NgramTier { bytes: 30 << 30, resident_share: 1.0 }),
        "a resident table is all in memory"
    );
    // A measurement belongs to the load it was taken in.
    let gone = || Some(NgramFacts { bytes: 100, paged: Some(Weak::new()) });
    lock(&d.meter).ngram = Some((board_loads(&board), Instant::now(), 0.5));
    board.loaded(Loaded { max_batch: 4, weights_bytes: 1, ngram: gone() });
    assert_eq!(
        d.status(READY).tiers.ngram.map(|n| n.resident_share),
        Some(0.0),
        "the reload's table, measured again (its mappings are gone here)"
    );
    board.unloaded(StoreView::default());
    assert_eq!(
        d.status(EngineNow { state: State::Idle, waiting: 0 }).tiers.ngram,
        None
    );
}

fn board_loads(board: &Board) -> u64 {
    lock(&board.activity).loads
}

#[test]
fn the_status_maps_the_lifecycle_and_holds_the_pin_only_while_idle() {
    let (board, d) = dashboard();
    let with = |state| d.status(EngineNow { state, waiting: 0 }).state;
    assert_eq!(with(State::Loading), LifecycleState::Loading);
    assert_eq!(with(State::Reloading), LifecycleState::Loading);
    assert_eq!(with(State::Recovering), LifecycleState::Loading);
    assert_eq!(with(State::Idle), LifecycleState::Unloaded);
    assert_eq!(with(State::Stopping), LifecycleState::Stopping);
    assert_eq!(
        d.status(EngineNow { state: State::Ready, waiting: 2 }).state,
        LifecycleState::Busy
    );

    let status = d.status(READY);
    assert_eq!(
        (status.pin_hold_s, status.idle_unload_s),
        (None, Some(1800)),
        "not pinned"
    );
    assert_eq!((status.queue.max_batch, status.queue.capacity), (4, 32));
    assert_eq!((status.tiers.disk, status.tiers.experts), (None, None));
    assert_eq!((status.recorder.ok, status.recorder.oldest), (false, None));
    assert_eq!(status.machine.sampler, SamplerState::Off);
    assert_eq!((status.machine.power, status.machine.gpu_active), (None, None));
    assert!(status.machine.ram_total_bytes.is_some_and(|t| t > 0));

    board.loaded(Loaded { max_batch: 1, weights_bytes: 42, ngram: None });
    board.pin_mirror().store(true, Ordering::Release);
    board.closed(None, Instant::now());
    let status = d.status(READY);
    assert_eq!(status.queue.max_batch, 1, "the load's cap");
    assert_eq!(status.tiers.weights, WeightsTier { bytes: 42, pinned: true });
    assert!(status.pin_hold_s.is_some_and(|s| (59..=60).contains(&s)));
    board.opened();
    assert_eq!(d.status(READY).pin_hold_s, None, "busy: the hold has not started");

    // The unload keeps the disk tier and the weights' size, drops the rest.
    board.unloaded(StoreView {
        disk: Some(DiskTier::default()),
        experts: Some(ExpertTier { slots: 1, used: 1, hit_rate: Some(1.0) }),
        ..StoreView::default()
    });
    let status = d.status(EngineNow { state: State::Idle, waiting: 0 });
    assert_eq!((status.tiers.disk.is_some(), status.tiers.experts), (true, None));
    assert_eq!((status.tiers.weights.bytes, status.queue.max_batch), (42, 1));
}

#[test]
fn live_rates_are_measured_against_a_sample_one_to_six_seconds_old() {
    let t0 = Instant::now();
    let sample = |secs: f64, decoded: u64| Sample {
        at: t0 + Duration::from_secs_f64(secs),
        decoded,
        prefilled: 0,
        cpu: None,
    };
    let mut meter = Meter::default();
    assert!(meter.sample(sample(0.0, 0)).is_none(), "nothing to measure against");
    assert!(meter.sample(sample(0.5, 10)).is_none(), "too soon");
    let base = meter.sample(sample(2.0, 100)).expect("two seconds back");
    assert_eq!(base.decoded, 0);
    assert_eq!(rate(base.decoded, 100, 2.0), 50.0);
    // Older samples fall out of the window.
    assert_eq!(meter.sample(sample(7.0, 400)).map(|b| b.decoded), Some(100));
    assert_eq!(meter.samples.len(), 2);
}

// --- settings ----------------------------------------------------------------

fn flag(name: &str, given: Option<&str>, default: Option<&str>) -> FlagValue {
    FlagValue {
        name: name.to_owned(),
        given: given.map(str::to_owned),
        default: default.map(str::to_owned),
    }
}

#[test]
fn settings_show_the_flag_text_its_default_and_where_the_value_came_from() {
    let flags = [
        flag("idle_unload", Some("30m"), Some("0")),
        flag("pin_hold", None, Some("1m")),
        flag("max_seq", None, Some("131072")),
        flag("thinking_nudges", Some("false"), Some("true")),
        flag("temperature", Some("0.7"), None),
        flag("top_k", None, None),
        flag("model", Some("/models/q4"), None),
    ];
    let computed =
        [("top_k", SettingValue::Int(20)), ("temperature", SettingValue::Number(1.0))];
    let all = settings::build(&flags, &computed);
    let get = |name: &str| all.iter().find(|s| s.name == name).expect(name).clone();

    let idle = get("idle_unload");
    assert_eq!(
        (idle.kind, idle.flag.as_str()),
        (SettingType::Duration, "--idle-unload")
    );
    assert_eq!(idle.value, Some(SettingValue::Text("30m".into())));
    assert_eq!((idle.value_s, idle.source), (Some(1800), SettingSource::Flag));
    assert_eq!(idle.default, Some(SettingValue::Text("0".into())));
    let hold = get("pin_hold");
    assert_eq!(
        (hold.value_s, hold.source, hold.flag_value),
        (Some(60), SettingSource::Default, None)
    );
    assert_eq!(get("max_seq").value, Some(SettingValue::Int(131_072)));
    assert_eq!(get("thinking_nudges").value, Some(SettingValue::Bool(false)));
    let temperature = get("temperature");
    assert_eq!(
        temperature.default,
        Some(SettingValue::Number(1.0)),
        "the checkpoint's"
    );
    assert_eq!(temperature.value, Some(SettingValue::Number(0.7)));
    assert_eq!((temperature.min, temperature.max), (Some(0.0), None));
    assert_eq!(get("top_k").value, Some(SettingValue::Int(20)));
    // A flag without a value or default is unset: null, not a placeholder.
    let effort = get("reasoning_effort");
    assert_eq!((effort.value, effort.default), (None, None));
    assert_eq!(
        effort.options,
        Some(vec!["low".into(), "medium".into(), "xhigh".into()])
    );
    assert_eq!(get("model").value, Some(SettingValue::Text("/models/q4".into())));
    // Nothing can change at runtime yet.
    assert!(all.iter().all(|s| !s.editable && s.applies.is_none()));
    // Bounds only on numbers, the duration's seconds only on durations.
    assert!(all.iter().all(|s| s.kind == SettingType::Duration || s.value_s.is_none()));
    assert!(
        all.iter().all(|s| matches!(s.kind, SettingType::Number | SettingType::Int)
            || (s.min, s.max) == (None, None))
    );
    // The mock's sections, in its order.
    let mut groups: Vec<&str> = all.iter().map(|s| s.group.as_str()).collect();
    groups.dedup();
    assert_eq!(groups, ["Sampling", "Thinking", "Lifecycle", "Caches", "Engine"]);
}

// --- the contract's fixtures -------------------------------------------------

/// Where the fixtures live: TypeScript the dashboard's typecheck holds
/// against `types.ts`, each value typed with its response type, so string
/// literals, nulls and excess fields are all checked.
const FIXTURES: &str = "dashboard/src/api/fixtures.ts";

/// Every variant of each enum the contract names, as the TypeScript union
/// must accept them. A variant added in Rust fails to compile the match
/// below until it is listed here too.
fn variants() -> Vec<(&'static str, &'static str, serde_json::Value)> {
    use serde_json::to_value as v;
    let states = [
        LifecycleState::Unloaded,
        LifecycleState::Loading,
        LifecycleState::Ready,
        LifecycleState::Busy,
        LifecycleState::Stopping,
    ];
    let pressures =
        [MemoryPressure::Normal, MemoryPressure::Warning, MemoryPressure::Critical];
    let samplers = [SamplerState::Ok, SamplerState::Off, SamplerState::Failed];
    let tiers = [SessionTier::Gpu, SessionTier::Disk];
    let cancels = [CancelledBy::Client, CancelledBy::Shutdown];
    let types = [
        SettingType::Number,
        SettingType::Int,
        SettingType::Enum,
        SettingType::String,
        SettingType::Bool,
        SettingType::Duration,
    ];
    let sources =
        [SettingSource::Default, SettingSource::Flag, SettingSource::Database];
    let log_sources = [SettingsLogSource::Dashboard, SettingsLogSource::Startup];
    // Exhaustive matches: the lists above name every variant.
    let _ = |a: LifecycleState, b: MemoryPressure, c: SamplerState, d: SessionTier| {
        use LifecycleState as L;
        match a {
            L::Unloaded | L::Loading | L::Ready | L::Busy | L::Stopping => {}
        }
        match b {
            MemoryPressure::Normal
            | MemoryPressure::Warning
            | MemoryPressure::Critical => {}
        }
        match c {
            SamplerState::Ok | SamplerState::Off | SamplerState::Failed => {}
        }
        match d {
            SessionTier::Gpu | SessionTier::Disk => {}
        }
    };
    let _ = |a: CancelledBy, b: SettingType, c: SettingSource, d: SettingsLogSource| {
        use SettingType as T;
        match a {
            CancelledBy::Client | CancelledBy::Shutdown => {}
        }
        match b {
            T::Number | T::Int | T::Enum | T::String | T::Bool | T::Duration => {}
        }
        match c {
            SettingSource::Default | SettingSource::Flag | SettingSource::Database => {}
        }
        match d {
            SettingsLogSource::Dashboard | SettingsLogSource::Startup => {}
        }
    };
    let ok = "serialize";
    vec![
        ("lifecycleStates", "LifecycleState[]", v(states).expect(ok)),
        ("memoryPressures", "MemoryPressure[]", v(pressures).expect(ok)),
        ("samplerStates", "SamplerState[]", v(samplers).expect(ok)),
        ("sessionTiers", "SessionTier[]", v(tiers).expect(ok)),
        ("cancellations", "CancelledBy[]", v(cancels).expect(ok)),
        ("settingTypes", "SettingType[]", v(types).expect(ok)),
        ("settingSources", "SettingSource[]", v(sources).expect(ok)),
        (
            "settingsLogSources",
            "SettingsLogEntry['source'][]",
            v(log_sources).expect(ok),
        ),
    ]
}

fn busy_status() -> Status {
    Status {
        now: 1_760_106_000_000,
        state: LifecycleState::Busy,
        model: "qwen3.8-flash-next".into(),
        uptime_s: 86_400,
        pin_hold_s: Some(42),
        idle_unload_s: Some(1800),
        idle_for_s: 0,
        queue: QueueStatus { waiting: 1, running: 2, max_batch: 4, capacity: 32 },
        live: LiveStatus {
            decode_tps: 92.5,
            prefill_tps: 1840.0,
            acceptance: Some(0.78),
            requests_per_min: 7,
            active_sessions: 2,
        },
        machine: MachineStatus {
            sampler: SamplerState::Ok,
            cpu_load: Some(0.21),
            gpu_active: Some(0.88),
            gpu_freq_mhz: Some(1578.0),
            memory_pressure: Some(MemoryPressure::Normal),
            ram_total_bytes: Some(137_438_953_472),
            ram_used_bytes: Some(104_857_600_000),
            lily_resident_bytes: Some(76_450_000_000),
            swap_used_bytes: Some(1_181_116_006),
            power: Some(Power { cpu_w: 7.5, gpu_w: 48.25, ane_w: 0.0, sys_w: 71.5 }),
            temp_c: Some(Temperatures { cpu: 66.0, gpu: 72.5 }),
        },
        tiers: Tiers {
            gpu: GpuTier {
                used_bytes: 12_884_901_888,
                budget_bytes: Some(25_769_803_776),
                entries: 5,
                max_entries: 16,
            },
            disk: Some(DiskTier {
                used_bytes: 41_000_000_000,
                budget_bytes: 107_374_182_400,
                entries: 14,
                durable_entries: 3,
                durable_bytes: 6_000_000_000,
            }),
            experts: Some(ExpertTier {
                slots: 3072,
                used: 3072,
                hit_rate: Some(0.961),
            }),
            ngram: Some(NgramTier { bytes: 32_856_000_000, resident_share: 0.874 }),
            weights: WeightsTier { bytes: 45_960_000_000, pinned: true },
        },
        recorder: RecorderStatus {
            ok: true,
            dropped_rows: 0,
            db_bytes: 6_606_028,
            oldest: Some(1_759_501_200_000),
        },
    }
}

/// An unloaded engine on a machine without the probes: every value lily
/// does not have is null.
fn unloaded_status() -> Status {
    Status {
        now: 1_760_106_000_000,
        state: LifecycleState::Unloaded,
        model: "qwen3.8-flash-next".into(),
        uptime_s: 90_000,
        pin_hold_s: None,
        idle_unload_s: None,
        idle_for_s: 3_600,
        queue: QueueStatus { waiting: 0, running: 0, max_batch: 4, capacity: 32 },
        live: LiveStatus {
            decode_tps: 0.0,
            prefill_tps: 0.0,
            acceptance: None,
            requests_per_min: 0,
            active_sessions: 0,
        },
        machine: MachineStatus {
            sampler: SamplerState::Off,
            cpu_load: None,
            gpu_active: None,
            gpu_freq_mhz: None,
            memory_pressure: None,
            ram_total_bytes: None,
            ram_used_bytes: None,
            lily_resident_bytes: None,
            swap_used_bytes: None,
            power: None,
            temp_c: None,
        },
        tiers: Tiers {
            gpu: GpuTier {
                used_bytes: 0,
                budget_bytes: None,
                entries: 0,
                max_entries: 16,
            },
            disk: None,
            experts: None,
            ngram: None,
            weights: WeightsTier { bytes: 45_960_000_000, pinned: false },
        },
        recorder: RecorderStatus {
            ok: false,
            dropped_rows: 0,
            db_bytes: 0,
            oldest: None,
        },
    }
}

/// Small-machine mode before its first lookup: the expert tier's null.
fn small_machine_status() -> Status {
    let mut status = busy_status();
    status.tiers.experts = Some(ExpertTier { slots: 1536, used: 1536, hit_rate: None });
    status
}

fn sessions_fixture() -> SessionsResponse {
    SessionsResponse {
        sessions: vec![
            Session {
                lineage: Some(1_760_101_000_000_118),
                tier: SessionTier::Gpu,
                durable: false,
                active: true,
                tokens: 48_211,
                bytes: 1_482_000_000,
                last_used_t: 1_760_106_000_000,
                requests: 23,
                decode_tps_avg: Some(64.2),
                decode_tps_max: Some(78.9),
                prefill_tps_avg: Some(2210.5),
                cache_hit: Some(0.9712),
                acceptance: Some(0.81),
                created_t: Some(1_760_101_000_000),
            },
            Session {
                lineage: None,
                tier: SessionTier::Disk,
                durable: true,
                active: false,
                tokens: 12_004,
                bytes: 369_000_000,
                last_used_t: 1_760_020_000_000,
                requests: 0,
                decode_tps_avg: None,
                decode_tps_max: None,
                prefill_tps_avg: None,
                cache_hit: None,
                acceptance: None,
                created_t: None,
            },
        ],
    }
}

fn requests_fixture() -> RequestsResponse {
    let mut cancelled = row(1_760_105_990_000, None, 0, 0.0);
    cancelled.prefill_tps = None;
    cancelled.decode_tps = None;
    cancelled.drafted_tokens = None;
    cancelled.accepted_tokens = None;
    cancelled.cancelled_by = Some(CancelledBy::Shutdown);
    cancelled.batch = 3;
    RequestsResponse {
        requests: vec![
            RequestRow {
                thinking_forced: true,
                ..row(1_760_106_000_000, Some(1_760_101_000_000_118), 412, 6_120.5)
            },
            cancelled,
            failed_row(1_760_105_980_000, Some(1_760_101_000_000_118)),
        ],
    }
}

fn settings_fixture() -> SettingsResponse {
    SettingsResponse {
        settings: settings::build(
            &[
                flag("temperature", Some("0.6"), None),
                flag("thinking_nudges", None, Some("true")),
                flag("idle_unload", Some("30m"), Some("0")),
                flag("max_seq", None, Some("131072")),
                flag("model", Some("/models/q4"), None),
            ],
            &[
                ("top_k", SettingValue::Int(20)),
                ("kv_cache", SettingValue::Text("q8".into())),
            ],
        ),
    }
}

/// Phase 2 always serves an empty log; these entries are the shape phase 4
/// fills in.
fn settings_log_fixture() -> SettingsLogResponse {
    SettingsLogResponse {
        entries: vec![
            SettingsLogEntry {
                t: 1_760_098_800_000,
                name: "repetition_penalty".into(),
                old: Some(SettingValue::Number(1.0)),
                new: Some(SettingValue::Number(1.05)),
                source: SettingsLogSource::Dashboard,
            },
            SettingsLogEntry {
                t: 1_760_000_000_000,
                name: "reasoning_effort".into(),
                old: None,
                new: Some(SettingValue::Text("low".into())),
                source: SettingsLogSource::Startup,
            },
        ],
    }
}

/// The fixtures file as the Rust types serialize it.
fn fixtures_ts() -> String {
    let json = |value: &dyn erased::Json| value.json();
    let mut out = String::from(
        "// Generated by tests/unit/serve/dashboard.rs from lily's Rust types; do not edit.\n\
         // Regenerate: LILY_UPDATE_FIXTURES=1 cargo test --locked --lib serve::dashboard\n\
         // The typecheck holds every value against types.ts (literals, nulls and no\n\
         // excess fields), so a drift on either side fails `npm run typecheck`.\n\
         import type {\n  CancelledBy,\n  ErrorBody,\n  LifecycleState,\n  MemoryPressure,\n  \
         RequestsResponse,\n  SamplerState,\n  SessionTier,\n  SessionsResponse,\n  \
         SettingSource,\n  SettingsLogEntry,\n  SettingsLogResponse,\n  SettingsResponse,\n  \
         SettingType,\n  Status,\n} from './types';\n",
    );
    let values: [(&str, &str, &dyn erased::Json); 9] = [
        ("status", "Status", &busy_status()),
        ("statusUnloaded", "Status", &unloaded_status()),
        ("statusSmallMachine", "Status", &small_machine_status()),
        ("sessions", "SessionsResponse", &sessions_fixture()),
        ("requests", "RequestsResponse", &requests_fixture()),
        ("settings", "SettingsResponse", &settings_fixture()),
        ("settingsLog", "SettingsLogResponse", &settings_log_fixture()),
        (
            "error",
            "ErrorBody",
            &ErrorBody {
                error: ErrorDetail {
                    message: "history needs the recorder".into(),
                    setting: None,
                },
            },
        ),
        (
            "errorSetting",
            "ErrorBody",
            &ErrorBody {
                error: ErrorDetail {
                    message: "runtime settings are not editable yet".into(),
                    setting: Some("temperature".into()),
                },
            },
        ),
    ];
    for (name, ty, value) in values {
        out.push_str(&format!("\nexport const {name}: {ty} = {};\n", json(value)));
    }
    for (name, ty, value) in variants() {
        let text = serde_json::to_string_pretty(&value).expect("serialize");
        out.push_str(&format!("\nexport const {name}: {ty} = {text};\n"));
    }
    out
}

/// `serde_json` over values of different types in one list.
mod erased {
    pub trait Json {
        fn json(&self) -> String;
    }

    impl<T: serde::Serialize> Json for T {
        fn json(&self) -> String {
            serde_json::to_string_pretty(self).expect("serialize")
        }
    }
}

#[test]
fn the_responses_serialize_as_the_committed_fixtures() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURES);
    let generated = fixtures_ts();
    if std::env::var_os("LILY_UPDATE_FIXTURES").is_some_and(|v| v == "1") {
        std::fs::write(&path, &generated).expect("write fixtures");
        return;
    }
    let committed = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!("{} is missing; run with LILY_UPDATE_FIXTURES=1", path.display())
    });
    assert!(
        committed == generated,
        "{FIXTURES} differs from what the server serializes; if the change is meant, \
         regenerate with LILY_UPDATE_FIXTURES=1 and check types.ts"
    );
}

#[test]
fn every_nullable_field_is_null_in_some_fixture() {
    // Paths (array indices dropped) where a fixture has null, and where one
    // has a value: every field that is ever null must be null at least once.
    fn walk(value: &serde_json::Value, path: &str, nulls: &mut HashSet<String>) {
        match value {
            serde_json::Value::Null => {
                nulls.insert(path.to_owned());
            }
            serde_json::Value::Object(map) => {
                for (k, v) in map {
                    walk(v, &format!("{path}.{k}"), nulls);
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    walk(item, &format!("{path}[]"), nulls);
                }
            }
            _ => {}
        }
    }
    let mut nulls = HashSet::new();
    let status = [busy_status(), unloaded_status(), small_machine_status()];
    for s in &status {
        walk(&serde_json::to_value(s).unwrap(), "status", &mut nulls);
    }
    walk(&serde_json::to_value(sessions_fixture()).unwrap(), "sessions", &mut nulls);
    walk(&serde_json::to_value(requests_fixture()).unwrap(), "requests", &mut nulls);
    walk(&serde_json::to_value(settings_fixture()).unwrap(), "settings", &mut nulls);
    walk(&serde_json::to_value(settings_log_fixture()).unwrap(), "log", &mut nulls);
    for nullable in [
        "status.pin_hold_s",
        "status.idle_unload_s",
        "status.live.acceptance",
        "status.machine.cpu_load",
        "status.machine.gpu_active",
        "status.machine.gpu_freq_mhz",
        "status.machine.memory_pressure",
        "status.machine.ram_total_bytes",
        "status.machine.ram_used_bytes",
        "status.machine.lily_resident_bytes",
        "status.machine.swap_used_bytes",
        "status.machine.power",
        "status.machine.temp_c",
        "status.tiers.gpu.budget_bytes",
        "status.tiers.disk",
        "status.tiers.experts",
        "status.tiers.experts.hit_rate",
        "status.tiers.ngram",
        "status.recorder.oldest",
        "sessions.sessions[].lineage",
        "sessions.sessions[].decode_tps_avg",
        "sessions.sessions[].decode_tps_max",
        "sessions.sessions[].prefill_tps_avg",
        "sessions.sessions[].cache_hit",
        "sessions.sessions[].acceptance",
        "sessions.sessions[].created_t",
        "requests.requests[].lineage",
        "requests.requests[].prefill_tps",
        "requests.requests[].decode_tps",
        "requests.requests[].drafted_tokens",
        "requests.requests[].accepted_tokens",
        "requests.requests[].cancelled_by",
        "settings.settings[].default",
        "settings.settings[].flag_value",
        "settings.settings[].value",
        "settings.settings[].value_s",
        "settings.settings[].min",
        "settings.settings[].max",
        "settings.settings[].step",
        "settings.settings[].options",
        "settings.settings[].help",
        "settings.settings[].applies",
        "log.entries[].old",
    ] {
        assert!(nulls.contains(nullable), "no fixture has {nullable} null");
    }
}

#[test]
fn what_the_server_builds_has_the_fixtures_shape() {
    // The served values, not only the hand-made ones, have every field the
    // contract lists: the same keys as a fixture, recursively.
    fn keys(value: &serde_json::Value, prefix: &str, out: &mut Vec<String>) {
        if let Some(object) = value.as_object() {
            for (key, inner) in object {
                let path = format!("{prefix}.{key}");
                out.push(path.clone());
                if inner.is_object() {
                    keys(inner, &path, out);
                }
            }
        }
    }
    let (board, d) = dashboard();
    board.loaded(Loaded { max_batch: 4, weights_bytes: 1, ngram: None });
    let served = serde_json::to_value(d.status(READY)).expect("json");
    let made = serde_json::to_value(unloaded_status()).expect("json");
    let (mut a, mut b) = (Vec::new(), Vec::new());
    keys(&served, "", &mut a);
    keys(&made, "", &mut b);
    assert_eq!(a, b);
}
