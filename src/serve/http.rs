//! A minimal HTTP/1.1 server on `std::net`: request line, headers,
//! `Content-Length` bodies, fixed or chunked responses, one connection per
//! thread, `Connection: close` after every response.
//!
//! Owning the socket is the point: a client that goes away is noticed the
//! moment a write fails or, from the moment its request is queued, when its
//! connection reads a reset or an EOF that a probe confirms ([`Watched`]),
//! which is what lets the engine stop a prefill or a decode for a departed
//! client. General-purpose crates hide that behind buffering and background
//! writers.

use std::io::{BufRead as _, BufReader, ErrorKind, Read as _, Write as _};
use std::net::{Shutdown, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail, ensure};

const MAX_HEADER_BYTES: usize = 64 << 10;
const HEADER_TIMEOUT: Duration = Duration::from_secs(30);

pub struct Request {
    pub method: String,
    /// Whether the request line said HTTP/1.1 (rather than 1.0).
    pub http11: bool,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Reads one request from `stream`. `max_body` bounds `Content-Length`.
pub fn read_request(stream: &mut TcpStream, max_body: usize) -> Result<Request> {
    stream.set_read_timeout(Some(HEADER_TIMEOUT)).ok();
    let mut reader = BufReader::new(stream.try_clone().context("cloning socket")?);
    let mut line = String::new();
    let mut total = 0usize;
    reader.read_line(&mut line).context("reading request line")?;
    total += line.len();
    ensure!(!line.is_empty(), "connection closed before a request");
    let mut parts = line.split_whitespace();
    let method = parts.next().context("missing method")?.to_string();
    let target = parts.next().context("missing request target")?.to_string();
    let version = parts.next().context("missing HTTP version")?;
    ensure!(version.starts_with("HTTP/1."), "unsupported protocol {version}");
    let http11 = version != "HTTP/1.0";
    let mut headers = Vec::new();
    loop {
        line.clear();
        reader.read_line(&mut line).context("reading headers")?;
        total += line.len();
        ensure!(total <= MAX_HEADER_BYTES, "request headers too large");
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        let (name, value) = trimmed.split_once(':').context("malformed header line")?;
        headers.push((name.trim().to_string(), value.trim().to_string()));
    }
    let request = Request { method, http11, path: target, headers, body: Vec::new() };
    if request
        .header("Transfer-Encoding")
        .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
    {
        bail!("chunked request bodies are not supported; send Content-Length");
    }
    let length: usize = match request.header("Content-Length") {
        Some(v) => v.trim().parse().context("invalid Content-Length")?,
        None => 0,
    };
    ensure!(length <= max_body, "request body exceeds {max_body} bytes");
    if request.header("Expect").is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
    {
        stream
            .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
            .context("writing 100 Continue")?;
    }
    let mut body = vec![0u8; length];
    // Bytes already buffered by the header reader come first.
    let buffered = reader.buffer().to_vec();
    let take = buffered.len().min(length);
    body[..take].copy_from_slice(&buffered[..take]);
    reader.consume(take);
    if take < length {
        stream.set_read_timeout(Some(Duration::from_secs(120))).ok();
        stream.read_exact(&mut body[take..]).context("reading request body")?;
    }
    Ok(Request { body, ..request })
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "",
    }
}

fn head(status: u16, content_type: &str, extra: &[(&str, &str)]) -> String {
    let mut head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nConnection: close\r\n",
        reason(status)
    );
    for (k, v) in extra {
        head.push_str(k);
        head.push_str(": ");
        head.push_str(v);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    head
}

/// Writes a complete response and closes the connection.
pub fn respond(
    mut stream: TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    let length = body.len().to_string();
    stream
        .write_all(
            head(status, content_type, &[("Content-Length", &length)]).as_bytes(),
        )
        .context("writing response head")?;
    stream.write_all(body).context("writing response body")?;
    stream.flush().ok();
    stream.shutdown(Shutdown::Both).ok();
    Ok(())
}

/// How long a probe waits for the peer's reset before it takes the
/// connection for a half-closed client that still reads (loopback answers
/// within microseconds, a LAN within milliseconds).
const PROBE_WINDOW: Duration = Duration::from_millis(250);
const PROBE_STEP: Duration = Duration::from_millis(5);

/// What has been written to a watched connection so far, which decides how
/// the watcher may probe it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Nothing yet: the request is queued or the engine is working on it.
    Waiting,
    /// The head of an event stream: further writes are chunks of events.
    Events,
    /// The head of any other chunked body, whose bytes follow at once.
    Body,
    /// The response is complete or abandoned; the server closed the socket.
    Done,
}

struct Conn {
    stream: TcpStream,
    /// Held around every write, so a probe never lands inside a response
    /// line or a chunk.
    phase: Mutex<Phase>,
    cancelled: Arc<AtomicBool>,
    /// Whether an interim `1xx` response may be sent (HTTP/1.1 only).
    interim: bool,
}

impl Conn {
    fn phase(&self) -> MutexGuard<'_, Phase> {
        self.phase.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The peer's read side reached EOF: either it closed the connection, or
    /// it only shut down its sending side and still waits for the response
    /// (an HTTP/1.1 client may half-close after its request). Only a write
    /// tells the two apart: a closed peer answers it with a reset. The probe
    /// is something the client must accept anyway: an interim `102
    /// Processing` before the head (RFC 9110 §15.2: a client parses 1xx
    /// responses it did not expect), an SSE comment chunk in an event
    /// stream. Inside any other body nothing can be written, and the
    /// response's own writes decide.
    fn departed(&self) -> bool {
        {
            let phase = self.phase();
            let probe: &[u8] = match *phase {
                Phase::Waiting if self.interim => b"HTTP/1.1 102 Processing\r\n\r\n",
                Phase::Events => b"3\r\n:\n\n\r\n",
                Phase::Waiting | Phase::Body | Phase::Done => return false,
            };
            if (&self.stream).write_all(probe).is_err() {
                return true;
            }
        }
        let deadline = Instant::now() + PROBE_WINDOW;
        loop {
            match self.stream.take_error() {
                Ok(None) => {}
                Ok(Some(_)) | Err(_) => return true,
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(PROBE_STEP);
        }
    }

    /// Writes `bytes` under the lock unless the response is done; a failure
    /// flags the client as gone.
    fn write(&self, phase: &mut Phase, bytes: &[u8]) -> std::io::Result<()> {
        if *phase == Phase::Done {
            return Err(std::io::ErrorKind::NotConnected.into());
        }
        let result = (&self.stream).write_all(bytes);
        if result.is_err() {
            self.cancelled.store(true, Ordering::Relaxed);
        }
        result
    }

    fn close(&self, phase: &mut Phase) {
        *phase = Phase::Done;
        (&self.stream).flush().ok();
        self.stream.shutdown(Shutdown::Both).ok();
    }
}

/// A connection whose request the engine is working on, watched for the
/// client going away from the moment the request is queued, before any
/// byte of the response exists: a helper thread blocks in `read` on it, and
/// with `Connection: close` the client sends nothing more, so a read that
/// completes means it left. A reset sets `cancelled` at once; an EOF only
/// once a probe confirmed the peer is gone ([`Conn::departed`]), so a
/// client that half-closed its side after the request is still served (and
/// no longer watched: it can only be noticed by a failing write).
///
/// Dropping it, or the response made from it, closes the connection.
pub struct Watched {
    conn: Arc<Conn>,
}

impl Watched {
    /// Starts watching `stream`. `http11` says whether the request was
    /// HTTP/1.1, which allows the interim probe.
    pub fn new(stream: TcpStream, http11: bool, cancelled: Arc<AtomicBool>) -> Self {
        let reader = stream.try_clone();
        let conn = Arc::new(Conn {
            stream,
            phase: Mutex::new(Phase::Waiting),
            cancelled,
            interim: http11,
        });
        if let Ok(mut reader) = reader {
            let conn = conn.clone();
            std::thread::Builder::new()
                .name("lily-watch".into())
                .spawn(move || {
                    reader.set_read_timeout(None).ok();
                    let mut scratch = [0u8; 256];
                    let reset = loop {
                        match reader.read(&mut scratch) {
                            Ok(0) => break false,
                            Ok(_) => continue,
                            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                            Err(_) => break true,
                        }
                    };
                    let gone = if reset {
                        *conn.phase() != Phase::Done
                    } else {
                        conn.departed()
                    };
                    if gone {
                        conn.cancelled.store(true, Ordering::Relaxed);
                    }
                })
                .ok();
        }
        Self { conn }
    }

    /// Writes a complete response and closes the connection.
    pub fn respond(self, status: u16, content_type: &str, body: &[u8]) -> Result<()> {
        let mut phase = self.conn.phase();
        let length = body.len().to_string();
        let head = head(status, content_type, &[("Content-Length", &length)]);
        let result = self
            .conn
            .write(&mut phase, head.as_bytes())
            .and_then(|_| self.conn.write(&mut phase, body))
            .context("writing response");
        self.conn.close(&mut phase);
        result
    }
}

impl Drop for Watched {
    fn drop(&mut self) {
        let mut phase = self.conn.phase();
        if *phase != Phase::Done {
            self.conn.close(&mut phase);
        }
    }
}

/// A chunked response in progress on a [`Watched`] connection. Dropping it
/// without [`Self::finish`] closes the connection mid-body, which clients
/// report as an error.
pub struct ChunkedResponse {
    watched: Watched,
}

impl ChunkedResponse {
    /// Writes the head. The connection's `cancelled` flag is set as soon as
    /// a write fails or the watcher finds the client gone.
    pub fn start(watched: Watched, status: u16, content_type: &str) -> Result<Self> {
        let conn = &watched.conn;
        conn.stream.set_nodelay(true).ok();
        let events = content_type.starts_with("text/event-stream");
        let extra: &[(&str, &str)] = if events {
            &[
                ("Transfer-Encoding", "chunked"),
                ("Cache-Control", "no-cache"),
                ("X-Accel-Buffering", "no"),
            ]
        } else {
            &[("Transfer-Encoding", "chunked")]
        };
        {
            let mut phase = conn.phase();
            conn.write(&mut phase, head(status, content_type, extra).as_bytes())
                .context("writing response head")?;
            *phase = if events { Phase::Events } else { Phase::Body };
        }
        Ok(Self { watched })
    }

    pub fn write_chunk(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let conn = &self.watched.conn;
        let mut phase = conn.phase();
        let mut chunk = format!("{:x}\r\n", bytes.len()).into_bytes();
        chunk.extend_from_slice(bytes);
        chunk.extend_from_slice(b"\r\n");
        conn.write(&mut phase, &chunk).context("writing chunk")
    }

    pub fn finish(self) -> Result<()> {
        let conn = &self.watched.conn;
        let mut phase = conn.phase();
        let result =
            conn.write(&mut phase, b"0\r\n\r\n").context("writing final chunk");
        conn.close(&mut phase);
        result
    }
}

#[cfg(test)]
#[path = "../../tests/unit/serve/http.rs"]
mod tests;
