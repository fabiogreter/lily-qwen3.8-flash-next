//! A minimal HTTP/1.1 server on `std::net`: request line, headers,
//! `Content-Length` bodies, fixed or chunked responses, one connection per
//! thread, `Connection: close` after every response.
//!
//! Owning the socket is the point: a client that goes away is noticed the
//! moment a write fails or its read side hits EOF, which is what lets the
//! engine stop generating for a departed client. General-purpose crates hide
//! that behind buffering and background writers.

use std::io::{BufRead as _, BufReader, Read as _, Write as _};
use std::net::{Shutdown, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context as _, Result, bail, ensure};

const MAX_HEADER_BYTES: usize = 64 << 10;
const HEADER_TIMEOUT: Duration = Duration::from_secs(30);

pub struct Request {
    pub method: String,
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
    let request = Request { method, path: target, headers, body: Vec::new() };
    if request.header("Transfer-Encoding").is_some_and(|v| !v.eq_ignore_ascii_case("identity")) {
        bail!("chunked request bodies are not supported; send Content-Length");
    }
    let length: usize = match request.header("Content-Length") {
        Some(v) => v.trim().parse().context("invalid Content-Length")?,
        None => 0,
    };
    ensure!(length <= max_body, "request body exceeds {max_body} bytes");
    if request.header("Expect").is_some_and(|v| v.eq_ignore_ascii_case("100-continue")) {
        stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").context("writing 100 Continue")?;
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

fn write_head(stream: &mut TcpStream, status: u16, content_type: &str, extra: &[(&str, &str)]) -> Result<()> {
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
    stream.write_all(head.as_bytes()).context("writing response head")
}

/// Writes a complete response and closes the connection.
pub fn respond(mut stream: TcpStream, status: u16, content_type: &str, body: &[u8]) -> Result<()> {
    let length = body.len().to_string();
    write_head(&mut stream, status, content_type, &[("Content-Length", &length)])?;
    stream.write_all(body).context("writing response body")?;
    stream.flush().ok();
    stream.shutdown(Shutdown::Both).ok();
    Ok(())
}

/// A chunked response in progress. Dropping it without [`Self::finish`]
/// closes the connection mid-body, which clients report as an error.
pub struct ChunkedResponse {
    stream: TcpStream,
    cancelled: Arc<AtomicBool>,
}

impl ChunkedResponse {
    /// Writes the head. `cancelled` is set as soon as a write fails or the
    /// client's read side reaches EOF (watched by a helper thread).
    pub fn start(stream: TcpStream, status: u16, content_type: &str, cancelled: Arc<AtomicBool>) -> Result<Self> {
        let mut stream = stream;
        stream.set_nodelay(true).ok();
        let extra: &[(&str, &str)] = if content_type.starts_with("text/event-stream") {
            &[("Transfer-Encoding", "chunked"), ("Cache-Control", "no-cache"), ("X-Accel-Buffering", "no")]
        } else {
            &[("Transfer-Encoding", "chunked")]
        };
        write_head(&mut stream, status, content_type, extra)?;
        // With `Connection: close` the client sends nothing more on this
        // socket, so any read completion means it went away.
        if let Ok(mut watcher) = stream.try_clone() {
            let flag = cancelled.clone();
            std::thread::Builder::new()
                .name("lily-watch".into())
                .spawn(move || {
                    watcher.set_read_timeout(None).ok();
                    let mut scratch = [0u8; 256];
                    loop {
                        match watcher.read(&mut scratch) {
                            Ok(0) | Err(_) => break,
                            Ok(_) => continue,
                        }
                    }
                    flag.store(true, Ordering::Relaxed);
                })
                .ok();
        }
        Ok(Self { stream, cancelled })
    }

    pub fn write_chunk(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let result = self
            .stream
            .write_all(format!("{:x}\r\n", bytes.len()).as_bytes())
            .and_then(|_| self.stream.write_all(bytes))
            .and_then(|_| self.stream.write_all(b"\r\n"));
        if result.is_err() {
            self.cancelled.store(true, Ordering::Relaxed);
        }
        result.context("writing chunk")
    }

    pub fn finish(mut self) -> Result<()> {
        self.stream.write_all(b"0\r\n\r\n").context("writing final chunk")?;
        self.stream.flush().ok();
        self.stream.shutdown(Shutdown::Both).ok();
        Ok(())
    }
}
