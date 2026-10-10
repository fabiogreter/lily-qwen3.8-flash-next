use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::PathBuf;

use super::contract::*;
use super::settings::{self, FlagValue};
use super::*;
use crate::serve::State;
use crate::serve::machine;
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

/// What a browser's navigation sends as `Accept`.
const BROWSER: &str = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8";

#[test]
fn the_dashboard_routes_leave_the_api_and_health_alone() {
    for method in ["GET", "HEAD"] {
        for accept in [None, Some(BROWSER), Some("*/*")] {
            assert_eq!(route(method, "/", accept), Some(Route::Page), "{method}");
            assert_eq!(
                route(method, "/assets/licenses.txt", accept),
                Some(Route::Page)
            );
        }
        assert_eq!(route(method, "/sessions/7", Some(BROWSER)), Some(Route::Page));
    }
    assert_eq!(route("GET", "/healthz", Some(BROWSER)), Some(Route::Page));
    for method in ["GET", "HEAD", "POST", "PATCH", "DELETE", "PUT"] {
        for accept in [None, Some(BROWSER)] {
            assert_eq!(
                route(method, "/v1/lily/status", accept),
                Some(Route::Api),
                "{method}"
            );
            assert_eq!(
                route(method, "/v1/lily/nope", accept),
                Some(Route::Api),
                "{method}"
            );
        }
    }
    assert_eq!(route("GET", "/v1/lily", None), Some(Route::Api));
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
        ("POST", "/sessions"),
        ("GET", "*"),
    ] {
        for accept in [None, Some(BROWSER)] {
            assert_eq!(route(method, path, accept), None, "{method} {path}");
        }
    }
}

#[test]
fn only_a_browsers_navigation_falls_back_to_the_app() {
    // What clients probe for (an Ollama client, a version check, a metrics
    // scraper) is not the dashboard's: the server's own 404, as before the
    // dashboard existed (the dispatch's last arm, `serve::handle`).
    for path in ["/models", "/api/tags", "/version", "/metrics", "/sessions"] {
        for accept in [None, Some("*/*"), Some("application/json")] {
            for method in ["GET", "HEAD"] {
                assert_eq!(
                    route(method, path, accept),
                    None,
                    "{method} {path} {accept:?}"
                );
            }
        }
    }
    // A browser's navigation to one of the app's routes gets the app.
    for path in ["/sessions", "/sessions/42", "/settings/", "/models"] {
        assert_eq!(route("GET", path, Some(BROWSER)), Some(Route::Page), "{path}");
        assert_eq!(page(path), page("/index.html"), "{path}");
    }
    assert_eq!(route("GET", "/sessions", Some("TEXT/HTML")), Some(Route::Page));
    // Not under `/api/`, and not for a path naming a file, whatever it accepts.
    for path in [
        "/api",
        "/api/tags",
        "/api/version",
        "/favicon.ico",
        "/robots.txt",
        "/x.js",
        "/sessions/report.json",
        "/assets/gone-12345678.js",
        "/assets/",
    ] {
        assert_eq!(route("GET", path, Some(BROWSER)), None, "{path}");
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

/// `--temperature 0.6 --idle-unload 30m` (`runtime::testing`): the idle
/// unload 30 minutes, the pin hold a minute.
fn config() -> Config {
    Config {
        model: "qwen3.8-flash-next",
        runtime: crate::serve::runtime::testing::runtime(),
        max_batch: 4,
        queue: 32,
        max_sessions: 16,
        cache_bytes: None,
        settings: crate::serve::runtime::testing::listing(),
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
    answer_with(d, method, target, headers, b"")
}

/// [`answer`] with a body.
fn answer_with(
    d: &Dashboard,
    method: &str,
    target: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> Response {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let route = route(method, path, Some(BROWSER)).expect("a dashboard route");
    let mut request = request(method, path, headers);
    request.body = body.to_vec();
    d.respond(route, &request, (path, query), LOCAL, READY)
}

const JSON: (&str, &str) = ("Content-Type", "application/json");

/// `PATCH /v1/lily/settings` with `body`, as lily's own page sends it.
fn patch(d: &Dashboard, body: serde_json::Value) -> Response {
    let body = serde_json::to_vec(&body).expect("json");
    answer_with(d, "PATCH", "/v1/lily/settings", &[JSON], &body)
}

/// `DELETE /v1/lily/settings/{name}`, as lily's own page sends it.
fn reset(d: &Dashboard, name: &str) -> Response {
    answer(d, "DELETE", &format!("/v1/lily/settings/{name}"), &[JSON])
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
        let route = route(method, target, Some(BROWSER)).expect("route");
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
            let route = route(method, target, Some(BROWSER)).expect("route");
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
    // A file that does not exist is not the dashboard's (the server's 404).
    assert_eq!(route("GET", "/assets/gone-12345678.js", Some(BROWSER)), None);
    assert_eq!(route("GET", "/favicon.ico", Some(BROWSER)), None);
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
    ] {
        assert_eq!(get(target).status, 200, "{target}");
    }
    // Without the recorder (`--stats-db off`): no history and no change
    // log, said so.
    let log = get("/v1/lily/settings/log");
    assert_eq!(log.status, 503);
    assert_eq!(
        body(&log)["error"]["message"],
        "the change log is kept in the recorder's database, which is off (--stats-db off)"
    );
    let history = get("/v1/lily/history?range=1h");
    assert_eq!(history.status, 503);
    assert_eq!(body(&history)["error"]["message"], "history is off (--stats-db off)");
    assert_eq!(get("/v1/lily/sessions/12/history?range=15m").status, 503);
    assert_eq!(get("/v1/lily/sessions/abc/history?range=15m").status, 404);
    // A range the contract does not name is refused first.
    for target in [
        "/v1/lily/history",
        "/v1/lily/history?range=2h",
        "/v1/lily/history?range=",
        "/v1/lily/sessions/12/history?range=1d",
    ] {
        assert_eq!(get(target).status, 400, "{target}");
    }
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
fn settings_writes_pass_the_guard_first() {
    let (_, d) = dashboard();
    let json = ("Content-Type", "application/json; charset=utf-8");
    // The guard comes before anything else, the route's own answer included.
    for headers in [
        vec![],
        vec![("Content-Type", "text/plain")],
        vec![json, ("Origin", "http://evil.example")],
        vec![json, ("Origin", "http://localhost:9999")],
    ] {
        let body = br#"{"temperature": 0.5}"#;
        let r = answer_with(&d, "PATCH", "/v1/lily/settings", &headers, body);
        assert_eq!(r.status, 403);
        assert_eq!(
            answer(&d, "DELETE", "/v1/lily/settings/nope", &headers).status,
            403
        );
        assert_eq!(
            answer(&d, "DELETE", "/v1/lily/settings/temperature", &headers).status,
            403
        );
    }
    // Lily's own origins pass (here to the 503 of a server without its
    // database).
    for origin in
        ["http://localhost:8000", "http://[::1]:8000", "http://127.0.0.1:8000"]
    {
        let body = br#"{"temperature": 0.5}"#;
        let r = answer_with(
            &d,
            "PATCH",
            "/v1/lily/settings",
            &[json, ("Origin", origin)],
            body,
        );
        assert_eq!(r.status, 503, "{origin}");
    }
}

#[test]
fn a_write_names_unknown_and_restart_only_settings_and_refuses_invalid_values() {
    let (_, d) = dashboard();
    let error = |r: &Response| {
        let b = body(r);
        (
            r.status,
            b["error"]["setting"].as_str().map(str::to_owned),
            b["error"]["message"].clone(),
        )
    };
    let r = patch(&d, serde_json::json!({"temperature": 0.5, "nope": 1}));
    assert_eq!(error(&r), (404, Some("nope".into()), "unknown setting".into()));
    let r = patch(&d, serde_json::json!({"temperature": "warm", "max_seq": 1}));
    assert_eq!(
        error(&r),
        (
            409,
            Some("max_seq".into()),
            "max_seq needs a restart and cannot be changed here".into()
        )
    );
    let r = patch(&d, serde_json::json!({"temperature": 0.5, "idle_unload": "10x"}));
    assert_eq!(
        error(&r),
        (
            400,
            Some("idle_unload".into()),
            "unknown duration unit \"x\" (use s, m, h or d)".into()
        )
    );
    for (status, name) in [(404, "nope"), (409, "max_seq"), (409, "queue")] {
        assert_eq!(error(&reset(&d, name)).0, status, "{name}");
        assert_eq!(error(&reset(&d, name)).1.as_deref(), Some(name));
    }
    // Not an object of names and values.
    for bad in [&b"[1]"[..], b"", b"{", b"\"x\""] {
        let r = answer_with(&d, "PATCH", "/v1/lily/settings", &[JSON], bad);
        assert_eq!(r.status, 400);
        assert_eq!(
            body(&r)["error"]["message"],
            "the body must be a JSON object of setting names and values"
        );
    }
    // A name twice is refused, not silently the last one.
    let r = answer_with(
        &d,
        "PATCH",
        "/v1/lily/settings",
        &[JSON],
        br#"{"pin_hold": "5s", "pin_hold": "1h"}"#,
    );
    assert_eq!(
        (r.status, body(&r)["error"]["message"].clone()),
        (400, "the body names pin_hold more than once".into())
    );
    // A body past 64 KiB is refused before it is read as JSON, after the
    // guard; the chat routes keep their own limit.
    let big = vec![b' '; MAX_WRITE_BODY + 1];
    assert_eq!(
        answer_with(&d, "PATCH", "/v1/lily/settings", &[JSON], &big).status,
        413
    );
    assert_eq!(
        answer_with(&d, "DELETE", "/v1/lily/settings/top_p", &[JSON], &big).status,
        413
    );
    assert_eq!(answer_with(&d, "PATCH", "/v1/lily/settings", &[], &big).status, 403);
    let at_cap = format!("{{{}}}", " ".repeat(MAX_WRITE_BODY - 2));
    assert_eq!(
        answer_with(&d, "PATCH", "/v1/lily/settings", &[JSON], at_cap.as_bytes())
            .status,
        200
    );
    // Nothing to change needs no database.
    let r = patch(&d, serde_json::json!({}));
    assert_eq!((r.status, body(&r)), (200, serde_json::json!({"updated": []})));
}

#[test]
fn without_an_open_database_a_write_changes_nothing_and_says_why() {
    let (_, d) = dashboard();
    let r = patch(&d, serde_json::json!({"temperature": 0.5}));
    assert_eq!(r.status, 503);
    assert_eq!(
        body(&r)["error"]["message"],
        "settings changes are kept in the recorder's database, which is off (--stats-db off); nothing changed"
    );
    assert_eq!(d.config.runtime.snapshot().defaults.sampling.temperature, 0.6);
    let temperature = body(&answer(&d, "GET", "/v1/lily/settings", &[]))["settings"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["name"] == "temperature")
        .unwrap()
        .clone();
    assert_eq!(
        (temperature["value"].clone(), temperature["source"].clone()),
        (0.6.into(), "flag".into())
    );
    // A reset always goes to the database (it deletes the row by name).
    let r = reset(&d, "temperature");
    assert_eq!(
        (r.status, body(&r)["error"]["setting"].clone()),
        (503, "temperature".into())
    );
    // With `--stats-db off` the runtime settings are listed read-only, and
    // say why; the restart-only ones are as they were.
    let all = body(&answer(&d, "GET", "/v1/lily/settings", &[]))["settings"].clone();
    let get = |name: &str| {
        all.as_array().unwrap().iter().find(|s| s["name"] == name).unwrap().clone()
    };
    for name in ["temperature", "idle_unload", "durable_min_tokens"] {
        let s = get(name);
        assert_eq!(
            (s["editable"].clone(), s["applies"].clone()),
            (false.into(), serde_json::Value::Null),
            "{name}"
        );
        assert!(s["help"].as_str().unwrap().ends_with(
            "Read-only: changes are kept in the recorder's database, which is off (--stats-db off)."
        ));
    }
    assert_eq!(get("max_seq")["help"], serde_json::Value::Null);
    // A recorder still opening its database refuses the same way: a file
    // where its directory should be keeps it retrying.
    let root = std::env::temp_dir()
        .join(format!("lily-dashboard-test-settings-opening-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("blocker"), b"").unwrap();
    let board = Arc::new(Board::default());
    let config = config();
    let recorder = crate::serve::recorder::Recorder::start(
        Some(&root.join("blocker").join("lily.sqlite")),
        &board,
        Some(config.runtime.clone()),
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    while recorder.status().error.is_none() {
        assert!(Instant::now() < deadline, "the open did not fail");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(recorder.status().state, RecorderState::Opening);
    let opening = Dashboard::new(board, config).with_recorder(recorder.clone());
    let r = patch(&opening, serde_json::json!({"pin_hold": "5s"}));
    assert_eq!(r.status, 503);
    assert_eq!(
        body(&r)["error"]["message"],
        "the recorder's database is not open yet (see the server log); nothing changed"
    );
    assert_eq!(opening.config.runtime.pin_hold_secs(), 60);
    // While opening, the runtime settings stay editable.
    assert!(
        opening.settings().settings.iter().any(|s| s.name == "pin_hold" && s.editable)
    );
    recorder.finish(Duration::from_secs(2));
    let _ = std::fs::remove_dir_all(&root);
}

/// A dashboard over a running recorder on a fresh database, open.
fn live(
    tag: &str,
) -> (PathBuf, Arc<Board>, Dashboard, crate::serve::recorder::Recorder) {
    use crate::serve::recorder::Recorder;
    let root = std::env::temp_dir()
        .join(format!("lily-dashboard-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let board = Arc::new(Board::default());
    let config = config();
    let recorder = Recorder::start(
        Some(&root.join("lily.sqlite")),
        &board,
        Some(config.runtime.clone()),
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    while recorder.status().state != RecorderState::Open {
        assert!(Instant::now() < deadline, "the recorder did not open");
        std::thread::sleep(Duration::from_millis(5));
    }
    let d = Dashboard::new(board.clone(), config).with_recorder(recorder.clone());
    (root, board, d, recorder)
}

#[test]
fn a_write_is_durable_before_it_is_answered_and_a_reset_goes_back_to_the_flag() {
    let (root, _, d, recorder) = live("settings-write");
    let r = patch(&d, serde_json::json!({"temperature": "0.5", "idle_unload": " 5m"}));
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let updated = body(&r)["updated"].as_array().unwrap().clone();
    let names: Vec<_> =
        updated.iter().map(|s| s["name"].as_str().unwrap().to_owned()).collect();
    assert_eq!(names, ["temperature", "idle_unload"], "in the listing's order");
    assert!(updated.iter().all(|s| s["source"] == "database" && s["editable"] == true));
    assert_eq!(updated[0]["value"], 0.5);
    assert_eq!(
        (updated[1]["value"].clone(), updated[1]["value_s"].clone()),
        ("5m".into(), 300.into())
    );
    // In effect: the next request's snapshot and the status.
    assert_eq!(d.config.runtime.snapshot().defaults.sampling.temperature, 0.5);
    assert_eq!(d.status(READY).idle_unload_s, Some(300));
    // Durable: in the database already.
    let stored = recorder.read(crate::serve::recorder::stored_settings).unwrap();
    assert_eq!(
        stored,
        vec![
            ("idle_unload".into(), "\"5m\"".into()),
            ("temperature".into(), "0.5".into())
        ]
    );
    // The same again changes nothing and logs nothing.
    let r = patch(&d, serde_json::json!({"temperature": 0.5}));
    assert_eq!(body(&r), serde_json::json!({"updated": []}));
    // Reset: back to the flag's 30m, the override gone.
    let r = reset(&d, "idle_unload");
    assert_eq!(r.status, 200);
    let setting = body(&r)["setting"].clone();
    assert_eq!(
        (setting["value"].clone(), setting["source"].clone()),
        ("30m".into(), "flag".into())
    );
    assert_eq!(d.status(READY).idle_unload_s, Some(1800));
    // The change log, newest first (a write's entries in its body's order,
    // the last one newest).
    let log = body(&answer(&d, "GET", "/v1/lily/settings/log", &[]));
    let entries: Vec<_> = log["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                e["name"].as_str().unwrap().to_owned(),
                e["old"].clone(),
                e["new"].clone(),
                e["source"].clone(),
            )
        })
        .collect();
    assert_eq!(
        entries,
        vec![
            ("idle_unload".into(), "5m".into(), "30m".into(), "dashboard".into()),
            ("idle_unload".into(), "30m".into(), "5m".into(), "dashboard".into()),
            ("temperature".into(), 0.6.into(), 0.5.into(), "dashboard".into()),
        ]
    );
    recorder.finish(Duration::from_secs(2));
    let _ = std::fs::remove_dir_all(&root);
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

// --- the history, from the recorder ------------------------------------------

/// A dashboard reading the database at `path`, which `rows` were written to.
fn recorded(tag: &str, rows: &[RequestRow]) -> (PathBuf, Arc<Board>, Dashboard) {
    use crate::serve::recorder::{Recorder, Writer, open};
    let root = std::env::temp_dir()
        .join(format!("lily-dashboard-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let path = root.join("lily.sqlite");
    let (conn, _) = open(&path, 1).expect("database");
    let mut writer = Writer::new(conn, Some(path.clone()), Arc::default(), 0);
    assert!(writer.record(rows.to_vec()));
    writer.publish_extent();
    let health = writer.health().clone();
    drop(writer);
    let (board, d) = dashboard();
    let d = d.with_recorder(Recorder::reading(&path, health));
    (root, board, d)
}

#[test]
fn the_history_routes_serve_the_recorders_grid() {
    // Two requests just before the last hour edge, read as of just after
    // it: inside the last complete bucket of every range.
    let edge = super::super::now_ms() / 3_600_000 * 3_600_000;
    let (root, board, d) = recorded(
        "history",
        &[
            row(edge - 40_000, Some(7), 100, 2000.0),
            row(edge - 30_000, Some(8), 50, 500.0),
        ],
    );
    let at = |range, lineage| body(&d.history_at(range, lineage, edge + 1));
    for (range, name, n, step) in [
        (Range::QuarterHour, "15m", 15, 60),
        (Range::Hour, "1h", 60, 60),
        (Range::Day, "24h", 96, 900),
        (Range::Week, "7d", 168, 3600),
    ] {
        let h = at(range, None);
        assert_eq!(h["range"], name);
        assert_eq!(
            (h["points"].as_array().unwrap().len(), h["step_s"].as_u64()),
            (n, Some(step))
        );
        assert_eq!(h["summary"]["requests"], 2, "{name}");
        assert_eq!(h["points"][n - 1]["requests"], 2, "{name}: the last bucket");
    }
    let one = at(Range::QuarterHour, Some(7));
    assert_eq!(one["summary"]["requests"], 1);
    assert_eq!(one["summary"]["tokens_generated"], 100);
    assert!(one["points"].as_array().unwrap().iter().all(|p| p["cpu_load"].is_null()));
    // The routes themselves, on the real clock.
    let get = |target: &str| answer(&d, "GET", target, &[]);
    let r = get("/v1/lily/history?range=15m");
    assert_eq!(
        (r.status, body(&r)["points"].as_array().map(Vec::len)),
        (200, Some(15))
    );
    assert_eq!(get("/v1/lily/sessions/8/history?range=7d").status, 200);
    // A lineage neither recorded nor in the cache is unknown; one in the
    // cache that the recorder has nothing of has the empty grid.
    assert_eq!(get("/v1/lily/sessions/99/history?range=15m").status, 404);
    board.publish(StoreView {
        entries: vec![entry(Some(99), true, edge)],
        ..StoreView::default()
    });
    let empty = get("/v1/lily/sessions/99/history?range=15m");
    assert_eq!(empty.status, 200);
    assert_eq!(body(&empty)["summary"]["requests"], 0);
    // The status reports the recorder's health and extent.
    let status = d.status(READY).recorder;
    assert!(status.ok && status.db_bytes > 0, "{status:?}");
    assert!(status.oldest.is_some_and(|t| t <= edge - 40_000));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn with_the_recorder_per_session_figures_come_from_the_database_not_the_ring() {
    let now = super::super::now_ms();
    // Older than anything the ring would still hold: the ring is empty.
    let (root, board, d) =
        recorded("sessions", &[row(now - 3_600_000, Some(7), 100, 2000.0)]);
    board.publish(StoreView {
        entries: vec![entry(Some(7), true, now)],
        ..StoreView::default()
    });
    assert!(lock(&board.rows).is_empty());
    let sessions = d.sessions().sessions;
    let s = &sessions[0];
    assert_eq!((s.lineage, s.requests, s.decode_tps_avg), (Some(7), 1, Some(50.0)));
    assert_eq!(
        (s.decode_tps_max, s.cache_hit, s.acceptance),
        (Some(50.0), Some(0.9), Some(0.75))
    );
    let _ = std::fs::remove_dir_all(&root);
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

// --- the machine sampler -------------------------------------------------------

fn sampled(sys_w: f64) -> machine::Sample {
    machine::Sample {
        start_ms: 1_000,
        end_ms: 3_000,
        reading: machine::Reading {
            cpu_w: 8.123,
            gpu_w: 41.987,
            ane_w: 0.0,
            sys_w: Some(sys_w),
            gpu_active: 0.9137,
            gpu_freq_mhz: 1577.6,
            cpu_temp_c: Some(61.26),
            gpu_temp_c: Some(55.0),
        },
    }
}

/// A dashboard whose board has a sampler that `play` publishes to.
fn with_sampler() -> (Arc<Machine>, Dashboard) {
    let (board, d) = dashboard();
    let machine = Machine::running();
    assert!(board.attach_machine(machine.clone()));
    assert!(!board.attach_machine(Machine::off()), "a second sampler is refused");
    (machine, d)
}

#[test]
fn the_status_shows_the_samplers_fresh_reading_and_nothing_stale() {
    let (machine, d) = with_sampler();
    // Running, nothing measured yet: the sampler is ok and the figures null.
    let m = d.status(READY).machine;
    assert_eq!(m.sampler, SamplerState::Ok);
    assert_eq!(
        (m.power, m.gpu_active, m.gpu_freq_mhz, m.temp_c),
        (None, None, None, None)
    );

    machine.publish(machine::BUSY_INTERVAL, sampled(61.456));
    let m = d.status(READY).machine;
    assert_eq!(m.sampler, SamplerState::Ok);
    assert_eq!(
        m.power,
        Some(Power { cpu_w: 8.12, gpu_w: 41.99, ane_w: 0.0, sys_w: Some(61.46) })
    );
    assert_eq!((m.gpu_active, m.gpu_freq_mhz), (Some(0.914), Some(1578.0)));
    assert_eq!(m.temp_c, Some(Temperatures { cpu: 61.3, gpu: 55.0 }));
    // CPU load and memory stay lily's own probes.
    assert!(m.ram_total_bytes.is_some_and(|t| t > 0));

    // The contract has no half a pair of temperatures.
    let mut half = sampled(10.0);
    half.reading.gpu_temp_c = None;
    machine.publish(machine::BUSY_INTERVAL, half);
    let m = d.status(READY).machine;
    assert_eq!(m.temp_c, None);
    assert!(m.power.is_some(), "the rest of the reading stays");
    // A ratio past 1 (a rounding in the source) is shown as 1.
    let mut over = sampled(10.0);
    over.reading.gpu_active = 1.0004;
    machine.publish(machine::BUSY_INTERVAL, over);
    assert_eq!(d.status(READY).machine.gpu_active, Some(1.0));
}

#[test]
fn a_failed_or_off_sampler_leaves_the_machine_figures_null_and_serving_on() {
    // Off: no sampler on the board at all, and one that is `Off`.
    let (_, d) = dashboard();
    assert_eq!(d.status(READY).machine.sampler, SamplerState::Off);
    let board = Arc::new(Board::default());
    board.attach_machine(Machine::off());
    let off = Dashboard::new(board, config());
    assert_eq!(off.status(READY).machine.sampler, SamplerState::Off);
    assert_eq!(answer(&off, "GET", "/v1/lily/status", &[]).status, 200);

    // Failed at the start: the real thread, a source that cannot be made.
    let board = Arc::new(Board::default());
    let failing = Machine::spawn(
        || -> Result<NoSource, String> { Err("Failed to create subscription".into()) },
        || false,
        machine::Intervals::default(),
    );
    board.attach_machine(failing.clone());
    let d = Dashboard::new(board, config());
    let until = Instant::now() + Duration::from_secs(10);
    while !failing.finished() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(5));
    }
    let r = answer(&d, "GET", "/v1/lily/status", &[]);
    assert_eq!(r.status, 200, "serving is unaffected");
    let m = &body(&r)["machine"];
    assert_eq!(m["sampler"], "failed");
    for field in ["power", "gpu_active", "gpu_freq_mhz", "temp_c"] {
        assert!(m[field].is_null(), "{field}");
    }
    assert!(m["ram_total_bytes"].as_u64().is_some_and(|t| t > 0));

    // Failed mid-run: what was shown is gone.
    let (machine, d) = with_sampler();
    machine.publish(machine::BUSY_INTERVAL, sampled(50.0));
    assert!(d.status(READY).machine.power.is_some());
    machine.fail("sampling failed: boom".into());
    let m = d.status(READY).machine;
    assert_eq!((m.sampler, m.power, m.temp_c), (SamplerState::Failed, None, None));
}

struct NoSource;

impl machine::Source for NoSource {
    fn read(&mut self, _: Duration) -> Result<machine::Reading, String> {
        Err("never created".into())
    }
}

#[test]
fn a_dashboard_request_opens_the_page_and_a_refused_one_does_not() {
    let (board, d) = dashboard();
    let machine = Machine::off();
    board.attach_machine(machine.clone());
    assert_eq!(machine.since_touch(), None);
    let remote =
        Connection { peer: at(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 20))), port: 8000 };
    let target = "/v1/lily/status";
    let refused = d.respond(
        Route::Api,
        &request("GET", target, &[]),
        (target, ""),
        remote,
        READY,
    );
    assert_eq!(refused.status, 403);
    assert_eq!(machine.since_touch(), None, "a refused peer does not keep it open");
    assert_eq!(answer(&d, "GET", target, &[]).status, 200);
    assert!(machine.since_touch().is_some_and(|age| age < Duration::from_secs(1)));
    // The page's own files count too.
    let machine = Machine::off();
    let board = Arc::new(Board::default());
    board.attach_machine(machine.clone());
    let d = Dashboard::new(board, config());
    assert_eq!(answer(&d, "GET", "/", &[]).status, 200);
    assert!(machine.since_touch().is_some());
}

#[test]
fn the_board_is_busy_with_a_request_open_or_tokens_moving() {
    let board = Board::default();
    assert!(!board.busy(), "nothing has happened");
    assert!(!board.busy());
    board.opened();
    assert!(board.busy(), "a request is running");
    board.closed(None, Instant::now());
    assert!(!board.busy());
    // Tokens drawn since the last look count even with no request open, once.
    board.drew();
    assert!(board.busy());
    assert!(!board.busy(), "the same tokens are not news twice");
    board.prefilled(512);
    assert!(board.busy());
    assert!(!board.busy());
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
    // The runtime settings are editable and say when a change applies;
    // the rest need a restart.
    for s in &all {
        assert_eq!(
            s.editable,
            crate::serve::runtime::applies(&s.name).is_some(),
            "{}",
            s.name
        );
        assert_eq!(
            s.applies.as_deref(),
            crate::serve::runtime::applies(&s.name),
            "{}",
            s.name
        );
    }
    assert!(get("idle_unload").editable && !get("max_seq").editable);
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
    let ranges = Range::ALL;
    let recorder_states = [
        RecorderState::Off,
        RecorderState::Opening,
        RecorderState::Open,
        RecorderState::Failed,
    ];
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
    let _ = |a: Range| match a {
        Range::QuarterHour | Range::Hour | Range::Day | Range::Week => {}
    };
    let _ = |a: RecorderState| match a {
        RecorderState::Off
        | RecorderState::Opening
        | RecorderState::Open
        | RecorderState::Failed => {}
    };
    let ok = "serialize";
    vec![
        ("lifecycleStates", "LifecycleState[]", v(states).expect(ok)),
        ("memoryPressures", "MemoryPressure[]", v(pressures).expect(ok)),
        ("samplerStates", "SamplerState[]", v(samplers).expect(ok)),
        ("sessionTiers", "SessionTier[]", v(tiers).expect(ok)),
        ("cancellations", "CancelledBy[]", v(cancels).expect(ok)),
        ("ranges", "Range[]", v(ranges).expect(ok)),
        ("recorderStates", "RecorderState[]", v(recorder_states).expect(ok)),
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
            power: Some(Power {
                cpu_w: 7.5,
                gpu_w: 48.25,
                ane_w: 0.0,
                sys_w: Some(71.5),
            }),
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
            state: RecorderState::Open,
            ok: true,
            dropped_rows: 0,
            db_bytes: 6_606_028,
            oldest: Some(1_759_501_200_000),
            error: None,
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
            state: RecorderState::Off,
            ok: false,
            dropped_rows: 0,
            db_bytes: 0,
            oldest: None,
            error: None,
        },
    }
}

/// Small-machine mode before its first lookup: the expert tier's null; the
/// recorder still opening its database, rows waiting.
fn small_machine_status() -> Status {
    let mut status = busy_status();
    status.tiers.experts = Some(ExpertTier { slots: 1536, used: 1536, hit_rate: None });
    // The SMC's system power failed: the parts are still measured.
    if let Some(power) = &mut status.machine.power {
        power.sys_w = None;
    }
    status.recorder = RecorderStatus {
        state: RecorderState::Opening,
        ok: false,
        dropped_rows: 3,
        db_bytes: 0,
        oldest: None,
        error: Some(
            "opening /Users/me/Library/Application Support/lily/lily.sqlite: database is locked"
                .into(),
        ),
    };
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

/// The command line's settings: flags and defaults.
fn settings_listing() -> Vec<Setting> {
    settings::build(
        &[
            flag("temperature", Some("0.6"), None),
            flag("thinking_nudges", None, Some("true")),
            flag("idle_unload", Some("30m"), Some("0")),
            flag("pin_hold", None, Some("1m")),
            flag("max_seq", None, Some("131072")),
            flag("model", Some("/models/q4"), None),
        ],
        &[
            ("top_k", SettingValue::Int(20)),
            ("repetition_penalty", SettingValue::Int(1)),
            ("kv_cache", SettingValue::Text("q8".into())),
        ],
    )
}

/// [`settings_listing`] with two of the database's overrides (`pin_hold`
/// and `repetition_penalty`).
fn settings_fixture() -> SettingsResponse {
    let overrides = [
        ("pin_hold", SettingValue::Text("5m".into())),
        ("repetition_penalty", SettingValue::Number(1.05)),
    ];
    SettingsResponse {
        settings: settings_listing()
            .iter()
            .map(|s| match overrides.iter().find(|(n, _)| *n == s.name) {
                Some((_, value)) => settings::with_override(s, value),
                None => s.clone(),
            })
            .collect(),
    }
}

fn fixture_setting(name: &str) -> Setting {
    settings_fixture().settings.into_iter().find(|s| s.name == name).expect(name)
}

/// A PATCH that changed one setting.
fn settings_update_fixture() -> SettingsUpdateResponse {
    SettingsUpdateResponse { updated: vec![fixture_setting("repetition_penalty")] }
}

/// A DELETE of `pin_hold`'s override: back at its default.
fn setting_reset_fixture() -> SettingResetResponse {
    let listing = settings_listing();
    SettingResetResponse {
        setting: listing.into_iter().find(|s| s.name == "pin_hold").expect("pin_hold"),
    }
}

/// The write errors as the server words them: 400, 403, 404, 409, 503 and
/// 504.
fn write_errors() -> Vec<(&'static str, ErrorBody)> {
    let error = |message: &str, setting: Option<&str>| ErrorBody {
        error: ErrorDetail {
            message: message.into(),
            setting: setting.map(str::to_owned),
        },
    };
    vec![
        ("errorInvalid", error("top_p must be in (0, 1]", Some("top_p"))),
        ("errorGuard", error("writes need Content-Type: application/json", None)),
        ("errorUnknown", error("unknown setting", Some("nope"))),
        (
            "errorRestartOnly",
            error(
                "max_seq needs a restart and cannot be changed here",
                Some("max_seq"),
            ),
        ),
        ("errorOutcomeUnknown", error(crate::serve::recorder::OUTCOME_UNKNOWN, None)),
        (
            "errorNoDatabase",
            error(
                "settings changes are kept in the recorder's database, which is off (--stats-db off); nothing changed",
                None,
            ),
        ),
    ]
}

/// The change log as the database serves it: a change from the dashboard
/// and an override applied at startup.
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

/// The fixtures' clock: the status fixture's `now`, 14:20 UTC.
const FIXTURE_NOW: u64 = 1_760_106_000_000;

/// A busy minute of the fixtures: `k` scales it.
fn fixture_minute(k: f64) -> crate::serve::recorder::Minute {
    crate::serve::recorder::Minute {
        requests: 2,
        decode_tokens: 2400.0 * k,
        prefill_tokens: 6000.0 * k,
        decode_busy_ms: 30_000.0,
        prefill_busy_ms: 3_000.0 * k,
        busy_ms: 33_000.0,
        prompt_tokens: 90_000,
        cached_tokens: 84_000,
        drafted_tokens: 1200,
        accepted_tokens: 930,
        sessions_max: 2,
    }
}

/// The 15-minute history at [`FIXTURE_NOW`]: five buckets with no data at
/// all (the server was off), five idle ones (machine figures, no rates) and
/// five busy ones, the last two with power.
fn history_fixture() -> History {
    use history::MachineRow;
    let last = FIXTURE_NOW / 60_000 - 1;
    let minutes: Vec<_> =
        (0..5).map(|i| (last - i, fixture_minute(1.0 + i as f64 / 4.0))).collect();
    let machine: Vec<MachineRow> = (0..10)
        .map(|i| MachineRow {
            minute: last - i,
            cpu_load: Some(if i < 5 { 0.25 } else { 0.04 }),
            pressure_level: Some(if i == 0 { 2 } else { 1 }),
            swap_bytes: Some(1_181_116_006),
            gpu_active: None,
            power_w: (i < 2).then_some(71.5),
            gpu_power_w: (i < 2).then_some(48.25),
            gpu_freq_mhz: (i < 2).then_some(1578.0),
            power_covered_ms: (i < 2).then_some(60_000),
            sys_covered_ms: (i < 2).then_some(60_000),
        })
        .collect();
    let queue = [(FIXTURE_NOW - 30_000, 12.0), (FIXTURE_NOW - 31_000, 40.0)];
    history::build(Range::QuarterHour, FIXTURE_NOW, &minutes, Some(&machine), &queue)
}

/// One session's 15 minutes: busy in two of them, no machine series.
fn session_history_fixture() -> History {
    let last = FIXTURE_NOW / 60_000 - 1;
    let mut m = fixture_minute(0.5);
    m.sessions_max = 1;
    let queue = [(FIXTURE_NOW - 30_000, 12.0)];
    history::build(
        Range::QuarterHour,
        FIXTURE_NOW,
        &[(last, m), (last - 3, m)],
        None,
        &queue,
    )
}

/// A cache entry the recorder has nothing of: every bucket and the summary
/// empty.
fn empty_history_fixture() -> History {
    history::build(Range::QuarterHour, FIXTURE_NOW, &[], None, &[])
}

/// The fixtures file as the Rust types serialize it.
fn fixtures_ts() -> String {
    let json = |value: &dyn erased::Json| value.json();
    let mut out = String::from(
        "// Generated by tests/unit/serve/dashboard.rs from lily's Rust types; do not edit.\n\
         // Regenerate: LILY_UPDATE_FIXTURES=1 cargo test --locked --lib serve::dashboard\n\
         // The typecheck holds every value against types.ts (literals, nulls and no\n\
         // excess fields), so a drift on either side fails `npm run typecheck`.\n\
         import type {\n  CancelledBy,\n  ErrorBody,\n  History,\n  LifecycleState,\n  \
         MemoryPressure,\n  Range,\n  RecorderState,\n  RequestsResponse,\n  SamplerState,\n  SessionTier,\n  \
         SessionsResponse,\n  SettingSource,\n  SettingsLogEntry,\n  SettingsLogResponse,\n  \
         SettingsResponse,\n  SettingsUpdateResponse,\n  SettingResetResponse,\n  SettingType,\n  Status,\n} from './types';\n",
    );
    let values: [(&str, &str, &dyn erased::Json); 13] = [
        ("status", "Status", &busy_status()),
        ("statusUnloaded", "Status", &unloaded_status()),
        ("statusSmallMachine", "Status", &small_machine_status()),
        ("sessions", "SessionsResponse", &sessions_fixture()),
        ("requests", "RequestsResponse", &requests_fixture()),
        ("settings", "SettingsResponse", &settings_fixture()),
        ("settingsLog", "SettingsLogResponse", &settings_log_fixture()),
        ("settingsUpdate", "SettingsUpdateResponse", &settings_update_fixture()),
        ("settingReset", "SettingResetResponse", &setting_reset_fixture()),
        ("history", "History", &history_fixture()),
        ("sessionHistory", "History", &session_history_fixture()),
        ("historyEmpty", "History", &empty_history_fixture()),
        (
            "error",
            "ErrorBody",
            &ErrorBody {
                error: ErrorDetail {
                    message: "history is off (--stats-db off)".into(),
                    setting: None,
                },
            },
        ),
    ];
    for (name, ty, value) in values {
        out.push_str(&format!("\nexport const {name}: {ty} = {};\n", json(value)));
    }
    for (name, value) in write_errors() {
        out.push_str(&format!(
            "\nexport const {name}: ErrorBody = {};\n",
            json(&value)
        ));
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
    for h in [history_fixture(), session_history_fixture(), empty_history_fixture()] {
        walk(&serde_json::to_value(h).unwrap(), "history", &mut nulls);
    }
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
        "status.machine.power.sys_w",
        "status.machine.temp_c",
        "status.tiers.gpu.budget_bytes",
        "status.tiers.disk",
        "status.tiers.experts",
        "status.tiers.experts.hit_rate",
        "status.tiers.ngram",
        "status.recorder.oldest",
        "status.recorder.error",
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
        "history.points[].decode_tps_avg",
        "history.points[].decode_tps_max",
        "history.points[].prefill_tps_avg",
        "history.points[].prefill_tps_max",
        "history.points[].acceptance",
        "history.points[].cache_hit",
        "history.points[].queue_ms_p50",
        "history.points[].queue_ms_max",
        "history.points[].cpu_load",
        "history.points[].gpu_active",
        "history.points[].memory_pressure",
        "history.points[].swap_bytes",
        "history.points[].power_w",
        "history.points[].gpu_power_w",
        "history.points[].gpu_freq_mhz",
        "history.summary.decode.avg",
        "history.summary.decode.max",
        "history.summary.prefill.avg",
        "history.summary.prefill.max",
        "history.summary.cache_hit",
        "history.summary.acceptance",
        "history.summary.power_avg_w",
        "history.summary.energy_wh",
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
