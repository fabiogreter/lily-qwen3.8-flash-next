//! The server against clients that leave during a long prefill, over HTTP
//! on loopback with the four-layer checkpoint (`LILY_MODEL_DIR_FLASH`):
//! the engine stops at the next chunk, serves the queue, keeps the prefix
//! for a retry that then answers as an uncancelled run does, and a client
//! that only half-closed its side is still answered.
//!
//! `LILY_MODEL_DIR_FLASH=<4-layer ckpt> cargo test --release --locked --test test_serve_cancel -- --ignored --nocapture`
//! (it takes the machine's instance lock, so no other lily may run).

use std::io::{Read as _, Write as _};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

struct Server {
    child: Child,
    port: u16,
    dir: PathBuf,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").expect("bind").local_addr().expect("addr").port()
}

/// A server on a fresh disk tier, ready to serve.
fn start(model: &str, tag: &str) -> Server {
    let dir = std::env::temp_dir()
        .join(format!("lily-serve-cancel-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let port = free_port();
    let child = Command::new(env!("CARGO_BIN_EXE_lily"))
        .args(["--model", model, "--bind", &format!("127.0.0.1:{port}")])
        .args(["--max-seq", "65536", "--pin-weights", "off"])
        .arg("--disk-cache-dir")
        .arg(&dir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn lily");
    let server = Server { child, port, dir };
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        if let Ok((200, body)) = request(port, "GET", "/health", None)
            && body["state"] == "ready"
        {
            return server;
        }
        assert!(Instant::now() < deadline, "the server did not become ready");
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn send(port: u16, method: &str, path: &str, body: Option<&Value>) -> TcpStream {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let body = body.map(|b| b.to_string()).unwrap_or_default();
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .expect("write request");
    stream
}

/// Reads a whole response; the JSON body of a chunked or a sized one.
fn read_response(mut stream: TcpStream) -> std::io::Result<(u16, Value)> {
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    // Interim responses (a half-closed client's probe) come first.
    let mut rest = text.as_str();
    while rest.starts_with("HTTP/1.1 1") {
        rest = &rest[rest.find("\r\n\r\n").expect("interim end") + 4..];
    }
    let status: u16 = rest[9..12].parse().expect("status");
    let split = rest.find("\r\n\r\n").expect("head end");
    let (head, body) = (&rest[..split], &rest[split + 4..]);
    let body = if head.contains("Transfer-Encoding: chunked") {
        let mut out = String::new();
        let mut b = body;
        loop {
            let line = b.find("\r\n").expect("chunk size");
            let size = usize::from_str_radix(&b[..line], 16).expect("hex");
            if size == 0 {
                break;
            }
            out.push_str(&b[line + 2..line + 2 + size]);
            b = &b[line + 2 + size + 2..];
        }
        out
    } else {
        body.to_owned()
    };
    Ok((status, serde_json::from_str(&body).unwrap_or(Value::String(body))))
}

fn request(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> std::io::Result<(u16, Value)> {
    TcpStream::connect(("127.0.0.1", port))?;
    read_response(send(port, method, path, body))
}

/// A text prompt of `words` words (about six tokens each), distinct per
/// `seed`.
fn long_prompt(seed: usize, words: usize) -> String {
    (0..words).map(|i| format!("w{} ", (i * 7919 + seed * 104_729) % 50_000)).collect()
}

fn completion(prompt: &str, stream: bool) -> Value {
    json!({"prompt": prompt, "max_tokens": 8, "temperature": 0, "stream": stream})
}

/// Sends `body`, leaves after `after`, then sends a short request at once:
/// returns the cancelled request's timings and the short one's.
fn leave_then_short(
    port: u16,
    body: &Value,
    after: Duration,
    short: &Value,
) -> (Value, Value) {
    let client = send(port, "POST", "/v1/completions", Some(body));
    std::thread::sleep(after);
    drop(client);
    let (status, answer) =
        read_response(send(port, "POST", "/v1/completions", Some(short)))
            .expect("short");
    assert_eq!(status, 200, "{answer}");
    let short_timings = answer["timings"].clone();
    // The cancelled request's prompt length: the short one's prompt differs.
    let (_, list) = request(port, "GET", "/v1/timings", None).expect("timings");
    let cancelled = list["data"]
        .as_array()
        .expect("list")
        .iter()
        .map(|e| e["timings"].clone())
        .find(|t| t["cancelled_by"] == "client")
        .expect("the cancelled request is in /v1/timings");
    (cancelled, short_timings)
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH and no other lily process"]
fn a_client_that_leaves_during_the_prefill_frees_the_engine_and_its_retry_resumes() {
    let Ok(model) = std::env::var("LILY_MODEL_DIR_FLASH") else { return };
    let prompt = long_prompt(1, 7_000);
    let short = completion("Hello", false);

    // The reference: the prompt served to the end on a fresh server.
    let reference = {
        let server = start(&model, "reference");
        let (status, answer) = read_response(send(
            server.port,
            "POST",
            "/v1/completions",
            Some(&completion(&prompt, false)),
        ))
        .expect("reference");
        assert_eq!(status, 200, "{answer}");
        eprintln!("reference timings: {}", answer["timings"]);
        answer
    };
    let n = reference["timings"]["prompt_tokens"].as_u64().expect("prompt tokens");
    let full_prefill_ms =
        reference["timings"]["prefill_ms"].as_f64().expect("prefill ms");
    let reference_text = reference["choices"][0]["text"].clone();

    let server = start(&model, "cancel");
    let port = server.port;
    // Streaming: the client leaves after a third of the reference prefill.
    let after = Duration::from_secs_f64(full_prefill_ms / 3e3);
    let (cancelled, short_timings) =
        leave_then_short(port, &completion(&prompt, true), after, &short);
    eprintln!("cancelled: {cancelled}\nshort: {short_timings}");
    let at = cancelled["cancelled_at"].as_u64().expect("cancelled_at");
    assert!(at > 0 && at < n - 1, "stopped inside the prefill, at {at} of {n}");
    assert_eq!(at % 4096, 0, "on a chunk boundary");
    assert_eq!(cancelled["prefill_tokens"], at);
    let queue_ms = short_timings["queue_ms"].as_f64().expect("queue_ms");
    assert!(
        queue_ms < full_prefill_ms / 2.0,
        "the short request queued {queue_ms} ms behind a {full_prefill_ms} ms prefill"
    );

    // The retry resumes at the kept boundary and answers as the reference.
    let (status, retry) = read_response(send(
        port,
        "POST",
        "/v1/completions",
        Some(&completion(&prompt, false)),
    ))
    .expect("retry");
    assert_eq!(status, 200, "{retry}");
    eprintln!("retry timings: {}", retry["timings"]);
    assert_eq!(retry["timings"]["cached_tokens"], at, "resumed at the kept prefix");
    assert_eq!(retry["choices"][0]["text"], reference_text, "same greedy output");

    // Non-streaming: a second long prompt, left the same way.
    let other = long_prompt(2, 7_000);
    let client =
        send(port, "POST", "/v1/completions", Some(&completion(&other, false)));
    std::thread::sleep(after);
    drop(client);
    let (status, _) =
        read_response(send(port, "POST", "/v1/completions", Some(&short)))
            .expect("short");
    assert_eq!(status, 200);
    let other_n = {
        let (_, list) = request(port, "GET", "/v1/timings", None).expect("timings");
        list["data"][1]["timings"].clone()
    };
    eprintln!("non-streaming cancelled: {other_n}");
    assert_eq!(other_n["cancelled_by"], "client");
    assert!(other_n["cancelled_at"].as_u64().expect("cancelled_at") < n - 1);

    // A client that half-closes after its request still gets its answer.
    let half = send(port, "POST", "/v1/completions", Some(&completion(&prompt, false)));
    half.shutdown(Shutdown::Write).expect("half-close");
    let (status, answer) = read_response(half).expect("half-closed answer");
    assert_eq!(status, 200, "{answer}");
    assert_eq!(answer["choices"][0]["text"], reference_text);
    assert!(answer["timings"].get("cancelled_by").is_none());
}
