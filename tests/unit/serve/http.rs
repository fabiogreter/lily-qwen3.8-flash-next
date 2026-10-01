//! The connection watcher over real loopback sockets: a client that closes or
//! resets is flagged, one that only half-closed its side is still served.

use std::io::{Read as _, Write as _};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::os::fd::AsRawFd as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::{ChunkedResponse, PROBE_WINDOW, Watched};

/// The server's and the client's end of one loopback connection, the
/// request already read off the server's end.
fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let client =
        TcpStream::connect(listener.local_addr().expect("addr")).expect("connect");
    let (server, _) = listener.accept().expect("accept");
    (server, client)
}

fn watch(server: TcpStream) -> (Watched, Arc<AtomicBool>) {
    let cancelled = Arc::new(AtomicBool::new(false));
    (Watched::new(server, true, cancelled.clone()), cancelled)
}

/// Whether `flag` is raised within `within`.
fn raised(flag: &AtomicBool, within: Duration) -> bool {
    let deadline = Instant::now() + within;
    while Instant::now() < deadline {
        if flag.load(Ordering::Relaxed) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    flag.load(Ordering::Relaxed)
}

/// Closes `stream` with a reset instead of a FIN (`SO_LINGER` of zero).
fn reset(stream: TcpStream) {
    let linger = libc::linger { l_onoff: 1, l_linger: 0 };
    // SAFETY: a valid socket and a correctly sized option value.
    let rc = unsafe {
        libc::setsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_LINGER,
            (&raw const linger).cast(),
            size_of::<libc::linger>() as libc::socklen_t,
        )
    };
    assert_eq!(rc, 0, "SO_LINGER");
    drop(stream);
}

fn read_all(mut client: TcpStream) -> String {
    // macOS refuses socket options once the connection is closed (EINVAL).
    client.set_read_timeout(Some(Duration::from_secs(5))).ok();
    let mut out = Vec::new();
    if let Err(e) = client.read_to_end(&mut out) {
        panic!("read: {e} after {:?}", String::from_utf8_lossy(&out));
    }
    String::from_utf8(out).expect("utf-8")
}

#[test]
fn a_client_that_closes_while_its_request_waits_is_flagged() {
    let (server, client) = pair();
    let (_watched, cancelled) = watch(server);
    assert!(!raised(&cancelled, Duration::from_millis(50)), "nothing happened yet");
    drop(client);
    assert!(raised(&cancelled, PROBE_WINDOW * 4), "a closed client must be flagged");
}

#[test]
fn a_client_that_resets_is_flagged() {
    let (server, client) = pair();
    let (_watched, cancelled) = watch(server);
    reset(client);
    assert!(raised(&cancelled, PROBE_WINDOW * 4), "a reset must be flagged");
}

#[test]
fn a_half_closed_client_is_not_flagged_and_gets_its_response() {
    let (server, client) = pair();
    let (watched, cancelled) = watch(server);
    // The client is done sending but still reads.
    client.shutdown(Shutdown::Write).expect("half-close");
    assert!(
        !raised(&cancelled, PROBE_WINDOW * 3),
        "a half-closed client that still reads is not gone"
    );
    watched.respond(200, "application/json", b"{\"ok\":true}").expect("respond");
    let text = read_all(client);
    // The probe was an interim response, which an HTTP/1.1 client skips.
    assert!(
        text.starts_with("HTTP/1.1 102 Processing\r\n\r\nHTTP/1.1 200 OK\r\n"),
        "{text}"
    );
    assert!(text.ends_with("{\"ok\":true}"), "{text}");
    assert!(!cancelled.load(Ordering::Relaxed));
}

#[test]
fn an_http10_client_is_never_probed_before_the_head() {
    let (server, client) = pair();
    let cancelled = Arc::new(AtomicBool::new(false));
    let watched = Watched::new(server, false, cancelled.clone());
    client.shutdown(Shutdown::Write).expect("half-close");
    assert!(!raised(&cancelled, PROBE_WINDOW * 2));
    watched.respond(200, "text/plain", b"hi").expect("respond");
    let text = read_all(client);
    assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
}

#[test]
fn an_event_stream_is_probed_with_a_comment_and_a_closed_client_is_flagged() {
    // Half-closed during the stream: a comment chunk, then the stream goes on.
    let (server, client) = pair();
    let (watched, cancelled) = watch(server);
    let mut response =
        ChunkedResponse::start(watched, 200, "text/event-stream").expect("head");
    response.write_chunk(b"data: 1\n\n").expect("chunk");
    client.shutdown(Shutdown::Write).expect("half-close");
    assert!(!raised(&cancelled, PROBE_WINDOW * 3));
    response.write_chunk(b"data: 2\n\n").expect("chunk");
    response.finish().expect("finish");
    let text = read_all(client);
    let body = &text[text.find("\r\n\r\n").expect("head end") + 4..];
    assert_eq!(body, "9\r\ndata: 1\n\n\r\n3\r\n:\n\n\r\n9\r\ndata: 2\n\n\r\n0\r\n\r\n");
    assert!(!cancelled.load(Ordering::Relaxed));

    // Closed during the stream: flagged without another write of ours.
    let (server, client) = pair();
    let (watched, cancelled) = watch(server);
    let mut response =
        ChunkedResponse::start(watched, 200, "text/event-stream").expect("head");
    response.write_chunk(b"data: 1\n\n").expect("chunk");
    drop(client);
    assert!(raised(&cancelled, PROBE_WINDOW * 4));
    // Its next writes fail rather than block or succeed.
    std::thread::sleep(Duration::from_millis(20));
    assert!(response.write_chunk(b"data: 2\n\n").is_err());
}

#[test]
fn the_server_closing_the_connection_is_not_a_departure() {
    let (server, client) = pair();
    let (watched, cancelled) = watch(server);
    let response =
        ChunkedResponse::start(watched, 200, "application/json").expect("head");
    response.finish().expect("finish");
    let text = read_all(client);
    assert!(text.ends_with("0\r\n\r\n"), "{text}");
    assert!(!raised(&cancelled, Duration::from_millis(100)));
}

#[test]
fn dropping_a_watched_connection_closes_it() {
    let (server, mut client) = pair();
    let (watched, _cancelled) = watch(server);
    drop(watched);
    client.set_read_timeout(Some(Duration::from_secs(2))).expect("timeout");
    let mut buf = [0u8; 8];
    assert_eq!(client.read(&mut buf).expect("eof"), 0);
    let _ = client.write_all(b"x");
}
