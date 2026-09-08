//! Lily's OpenAI-compatible API server.
//!
//! One engine thread owns the GPU and runs requests strictly one at a time
//! from a bounded queue. The HTTP thread parses and validates requests (400s
//! never touch the engine), and a small responder thread per request relays
//! the engine's output to the socket, so a slow or vanished client never
//! stalls the decode loop; its write failure cancels the generation at the
//! next token.

pub mod api;
pub mod disk;
pub mod http;
mod session;
pub mod stream;
pub mod tools;

use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};

use crate::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi};
use crate::generate::{FinishReason, GenerateOptions, Generator};
use crate::kernels::attention::MAX_SEQ;
use crate::kernels::sample::SamplingParams;
use crate::metal::MetalContext;
use crate::model::Qwen3_5Model;
use crate::qwen4exp::{NgramStorage, Qwen4ExpModel};
use api::{Defaults, Kind, Prepared};
use session::SessionStore;
use stream::{Event, OutputParser, ParserConfig};
use tools::ParsedToolCall;

const MAX_REQUEST_BYTES: usize = 32 << 20;
/// Recurrent-state checkpoints kept per session (the newest ones).
const CHECKPOINTS_PER_SESSION: usize = 3;
/// Left free below the device's recommended working set when the cache
/// budget is derived automatically.
const BUDGET_MARGIN_BYTES: usize = 2 << 30;

/// Sampling defaults from the command line; unset fields fall back to the
/// checkpoint's `generation_config.json`, then to OpenAI's defaults.
#[derive(Debug, Clone, Default)]
pub struct SamplingOverrides {
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub bind: String,
    pub max_seq: usize,
    pub cache_bytes: Option<usize>,
    pub max_sessions: usize,
    pub ngram_storage: NgramStorage,
    pub ngram_preload: bool,
    /// Pin the preloaded table in memory (`mlock`).
    pub ngram_lock: bool,
    /// Draft tokens per speculative step (0 disables the draft head).
    pub mtp_drafts: usize,
    /// Where evicted sessions are kept on disk (`None` disables the tier).
    pub disk_cache_dir: Option<std::path::PathBuf>,
    /// Most bytes the disk tier may hold.
    pub disk_cache_bytes: u64,
    /// Seconds an entry may go unused on disk before it is deleted (0: never).
    pub disk_cache_ttl_secs: u64,
    pub thinking: bool,
    pub reasoning_effort: Option<String>,
    pub queue: usize,
    pub sampling: SamplingOverrides,
}

// --- HTTP plumbing -----------------------------------------------------------

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

fn error_json(kind: &'static str, message: impl Into<String>) -> Vec<u8> {
    serde_json::to_vec(&ErrorEnvelope { error: ErrorBody { message: message.into(), kind } })
        .unwrap_or_else(|_| b"{\"error\":{\"message\":\"error\"}}".to_vec())
}

fn send_json<T: Serialize>(stream: TcpStream, status: u16, value: &T) {
    let body = serde_json::to_vec(value).unwrap_or_default();
    if let Err(error) = http::respond(stream, status, "application/json", &body) {
        eprintln!("response error: {error:#}");
    }
}

fn send_error(stream: TcpStream, status: u16, kind: &'static str, message: impl Into<String>) {
    if let Err(error) = http::respond(stream, status, "application/json", &error_json(kind, message)) {
        eprintln!("response error: {error:#}");
    }
}

/// What the engine sends the responder thread for one request.
enum Out {
    Start { status: u16, content_type: &'static str },
    Body(Vec<u8>),
    End,
}

/// The engine's handle on a request's response.
struct Sink {
    tx: Sender<Out>,
    cancelled: Arc<AtomicBool>,
    started: bool,
}

impl Sink {
    fn start(&mut self, status: u16, content_type: &'static str) {
        if !self.started {
            self.started = true;
            let _ = self.tx.send(Out::Start { status, content_type });
        }
    }

    fn send(&self, body: Vec<u8>) {
        let _ = self.tx.send(Out::Body(body));
    }

    fn sse(&self, value: &Value) {
        let mut line = b"data: ".to_vec();
        line.extend_from_slice(serde_json::to_string(value).unwrap_or_default().as_bytes());
        line.extend_from_slice(b"\n\n");
        self.send(line);
    }

    fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    fn end(&self) {
        let _ = self.tx.send(Out::End);
    }
}

struct Job {
    prepared: Prepared,
    sink: Sink,
    queued_at: Instant,
}

/// Relays one request's output from the engine to the client on the
/// connection's own thread; a failed write or the client's EOF flags
/// cancellation for the engine.
fn relay(stream: TcpStream, rx: Receiver<Out>, cancelled: Arc<AtomicBool>) {
    let (status, content_type) = match rx.recv() {
        Ok(Out::Start { status, content_type }) => (status, content_type),
        _ => {
            send_error(stream, 500, "server_error", "internal server error");
            return;
        }
    };
    let mut response = match http::ChunkedResponse::start(stream, status, content_type, cancelled.clone()) {
        Ok(response) => response,
        Err(_) => {
            cancelled.store(true, Ordering::Relaxed);
            return;
        }
    };
    while let Ok(message) = rx.recv() {
        match message {
            Out::Body(bytes) => {
                if response.write_chunk(&bytes).is_err() {
                    // Keep draining so the engine's sends never block.
                    for _ in rx.iter() {}
                    return;
                }
            }
            Out::End => break,
            Out::Start { .. } => {}
        }
    }
    if cancelled.load(Ordering::Relaxed) {
        return;
    }
    if let Err(error) = response.finish() {
        eprintln!("response error: {error:#}");
    }
}

// --- engine ------------------------------------------------------------------

struct Engine<M: LanguageModel> {
    ctx: MetalContext,
    model: M,
    generator: Arc<Generator>,
    sessions: SessionStore<M>,
    scratch: M::Scratch,
    max_seq: usize,
    drafts: usize,
    next_id: u64,
}

/// Where a request's text ends up when not streaming.
#[derive(Default)]
struct Collected {
    reasoning: String,
    content: String,
    tool_calls: Vec<ParsedToolCall>,
}

impl<M: LanguageModel> Engine<M> {
    fn load(model_dir: &Path, options: &ServeOptions, generator: Arc<Generator>) -> Result<Self> {
        let ctx = MetalContext::new()?;
        let started = Instant::now();
        let model = M::load(&ctx, model_dir, &LoadOptions { ngram_storage: options.ngram_storage, mtp_drafts: options.mtp_drafts })?;
        let drafts = options.mtp_drafts.min(model.max_drafts());
        eprintln!(
            "loaded {} in {:.1}s ({:.1} GB resident){}",
            M::MODEL_ID,
            started.elapsed().as_secs_f64(),
            ctx.current_allocated() as f64 / 1e9,
            if drafts > 0 { format!(", speculative decoding with {drafts} drafts per step") } else { String::new() }
        );
        if options.ngram_preload || options.ngram_lock {
            let started = Instant::now();
            let bytes = model.warm_storage(options.ngram_lock)?;
            if bytes > 0 {
                eprintln!(
                    "paged weights: {:.1} GB resident after preload in {:.1}s{}",
                    bytes as f64 / 1e9,
                    started.elapsed().as_secs_f64(),
                    if options.ngram_lock { " (locked)" } else { "" }
                );
            }
        }
        let declared = model.max_position_embeddings();
        let mut max_seq = options.max_seq.min(MAX_SEQ);
        if declared > 0 {
            max_seq = max_seq.min(declared);
        }
        ensure!(max_seq > 1, "max_seq must be at least 2");
        let mut scratch = model.new_scratch_with_capacity(&ctx, max_seq)?;
        warm_up(&ctx, &model, &mut scratch)?;

        let allocated = ctx.current_allocated();
        let working_set = ctx.recommended_working_set();
        let budget = match options.cache_bytes {
            Some(b) => b,
            None => working_set.saturating_sub(allocated).saturating_sub(BUDGET_MARGIN_BYTES).max(512 << 20),
        };
        let per_request = model.bytes_per_token() * max_seq;
        eprintln!(
            "memory: {:.1} GB allocated, {:.1} GB recommended working set, {:.1} GB session cache budget \
             ({} B/token of context; a full {}-token request needs {:.1} GB)",
            allocated as f64 / 1e9,
            working_set as f64 / 1e9,
            budget as f64 / 1e9,
            model.bytes_per_token(),
            max_seq,
            per_request as f64 / 1e9,
        );
        if per_request > budget {
            eprintln!(
                "warning: a request using the whole {max_seq}-token context exceeds the cache budget; \
                 lower --max-seq or raise --cache-bytes"
            );
        }
        let mut sessions = SessionStore::new(budget, options.max_sessions, CHECKPOINTS_PER_SESSION);
        if let (Some(dir), true) = (&options.disk_cache_dir, options.disk_cache_bytes > 0) {
            match model.persistence_format() {
                Some(format) => {
                    let disk = disk::DiskStore::open(dir, &format, options.disk_cache_bytes, options.disk_cache_ttl_secs)?;
                    eprintln!(
                        "session cache: disk tier at {} ({} entries, {:.1}/{:.1} GB, entries expire after {})",
                        disk.dir().display(),
                        disk.len(),
                        disk.used_bytes() as f64 / 1e9,
                        disk.budget_bytes() as f64 / 1e9,
                        if disk.max_age_secs() == 0 { "never".to_owned() } else { format!("{:.1} days unused", disk.max_age_secs() as f64 / 86_400.0) }
                    );
                    sessions = sessions.with_disk(disk);
                }
                None => eprintln!("session cache: {} cannot persist sessions; disk tier off", M::MODEL_ID),
            }
        }
        Ok(Self {
            ctx,
            model,
            generator,
            sessions,
            scratch,
            max_seq,
            drafts,
            next_id: 1,
        })
    }

    fn serve(&mut self, job: Job) {
        let mut sink = job.sink;
        let stream = job.prepared.stream;
        let kind = job.prepared.kind;
        let queued = job.queued_at.elapsed();
        let result = self.run(job.prepared, &mut sink);
        if let Err(error) = result {
            eprintln!("request failed: {error:#}");
            if !sink.started {
                sink.start(500, "application/json");
                sink.send(error_json("server_error", "internal server error"));
            } else if stream {
                sink.sse(&json!({"error": {"message": "internal server error", "type": "server_error"}}));
                sink.send(b"data: [DONE]\n\n".to_vec());
            }
        }
        sink.end();
        let _ = (kind, queued);
    }

    fn run(&mut self, p: Prepared, sink: &mut Sink) -> Result<()> {
        if sink.cancelled() {
            return Ok(());
        }
        let Engine { ctx, model, generator, sessions, scratch, max_seq, drafts, next_id } = self;
        ensure!(p.prompt.len() < *max_seq, "prompt too long for the server context");
        let n = p.prompt.len();
        let started = Instant::now();
        let acquired = sessions.acquire(ctx, model, &p.prompt, p.cache_key.as_deref())?;
        let mut session = acquired.session;
        let reused = acquired.reused;
        ensure!(reused < n, "session cache returned the whole prompt");

        // Prefix up to the last prompt token, then checkpoint there so an
        // identical or extended prompt can resume without re-feeding it.
        if reused < n - 1 {
            model.prefill(ctx, &mut session.state, scratch, &p.prompt[reused..n - 1], None)?;
        }
        let snapshot = session.state.snapshot(ctx)?;
        session.add_checkpoint(snapshot);
        let prefix_secs = started.elapsed().as_secs_f64();

        let created = now();
        let id = format!("{}-{}-{}", if p.kind == Kind::Chat { "chatcmpl" } else { "cmpl" }, created, *next_id);
        *next_id += 1;
        let tokenizer = generator.tokenizer();
        let mut parser = OutputParser::new(
            |ids: &[u32]| tokenizer.decode(ids, false),
            ParserConfig {
                thinking_open: p.thinking_open,
                tools: p.tools.clone(),
                stop_strings: p.stop_strings.clone(),
                raw: p.kind == Kind::Completion,
            },
        );
        let mut collected = Collected::default();
        let mut tool_index = 0usize;
        if p.stream {
            sink.start(200, "text/event-stream");
            if p.kind == Kind::Chat {
                sink.sse(&chunk(&id, created, M::MODEL_ID, json!({"role": "assistant", "content": ""}), None));
            }
        }
        let mut deliver = |events: Vec<Event>, sink: &Sink| {
            for event in events {
                match event {
                    Event::Reasoning(text) => {
                        if p.stream {
                            sink.sse(&chunk(&id, created, M::MODEL_ID, json!({"reasoning_content": text}), None));
                        } else {
                            collected.reasoning.push_str(&text);
                        }
                    }
                    Event::Content(text) => {
                        if p.stream {
                            if p.kind == Kind::Chat {
                                sink.sse(&chunk(&id, created, M::MODEL_ID, json!({"content": text}), None));
                            } else {
                                sink.sse(&text_chunk(&id, created, M::MODEL_ID, &text, None));
                            }
                        } else {
                            collected.content.push_str(&text);
                        }
                    }
                    Event::ToolCall(call) => {
                        if p.stream {
                            let delta = json!({"tool_calls": [{
                                "index": tool_index,
                                "id": call_id(&id, tool_index),
                                "type": "function",
                                "function": {"name": call.name, "arguments": call.arguments},
                            }]});
                            sink.sse(&chunk(&id, created, M::MODEL_ID, delta, None));
                        } else {
                            collected.tool_calls.push(call);
                        }
                        tool_index += 1;
                    }
                }
            }
        };

        let options = GenerateOptions { max_tokens: p.max_tokens, sampling: &p.sampling, stop_tokens: &[], drafts: *drafts };
        let decode_started = Instant::now();
        let generation = generator.generate(
            ctx,
            model,
            &mut session.state,
            scratch,
            &p.prompt[n - 1..],
            &options,
            &mut |token| {
                let events = parser.push(token)?;
                deliver(events, sink);
                Ok(!sink.cancelled() && !parser.stopped)
            },
        )?;
        let final_events = parser.finish();
        deliver(final_events, sink);
        let decode_secs = decode_started.elapsed().as_secs_f64();

        // Bookkeeping: the state holds the prompt plus every generated token
        // but the last (drawn, never fed).
        let fed_generated = generation.tokens.len() - 1;
        ensure!(
            generation.fed == 1 + fed_generated,
            "decode state advanced {} tokens for {} drawn",
            generation.fed,
            generation.tokens.len()
        );
        session.tokens.truncate(reused);
        session.tokens.extend_from_slice(&p.prompt[reused..]);
        session.tokens.extend_from_slice(&generation.tokens[..fed_generated]);
        ensure!(session.state.pos() == session.tokens.len(), "session token/state position mismatch");
        sessions.release(ctx, session, p.cache_key.as_deref());

        let completion_tokens = generation.tokens.len();
        let finish_reason = match generation.finish {
            FinishReason::Length => "length",
            _ if parser.tool_calls_emitted() > 0 => "tool_calls",
            _ => "stop",
        };
        eprintln!(
            "{}: {} prompt tokens ({} cached{}{}), {} generated, prefix {:.2}s, decode {:.2}s ({:.1} tok/s){}, finish={finish_reason}, sessions={} ({:.1}/{:.1} GB){}",
            id,
            n,
            reused,
            if acquired.forked { ", forked" } else { "" },
            acquired.from_disk.map(|d| format!(", from disk in {:.2}s", d.as_secs_f64())).unwrap_or_default(),
            completion_tokens,
            prefix_secs,
            decode_secs,
            completion_tokens as f64 / decode_secs.max(1e-9),
            if generation.drafted > 0 {
                format!(", drafts {}/{} accepted", generation.accepted, generation.drafted)
            } else {
                String::new()
            },
            sessions.len(),
            sessions.used_bytes() as f64 / 1e9,
            sessions.budget_bytes() as f64 / 1e9,
            sessions.disk().map(|d| format!(", disk {} ({:.1} GB)", d.len(), d.used_bytes() as f64 / 1e9)).unwrap_or_default(),
        );
        if sink.cancelled() {
            return Ok(());
        }
        let mut usage = json!({
            "prompt_tokens": n,
            "completion_tokens": completion_tokens,
            "total_tokens": n + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": reused},
        });
        if generation.drafted > 0 {
            usage["completion_tokens_details"] = json!({
                "accepted_prediction_tokens": generation.accepted,
                "rejected_prediction_tokens": generation.drafted - generation.accepted,
            });
        }
        if p.stream {
            if p.kind == Kind::Chat {
                sink.sse(&chunk(&id, created, M::MODEL_ID, json!({}), Some(finish_reason)));
            } else {
                sink.sse(&text_chunk(&id, created, M::MODEL_ID, "", Some(finish_reason)));
            }
            if p.include_usage {
                sink.sse(&json!({
                    "id": id,
                    "object": if p.kind == Kind::Chat { "chat.completion.chunk" } else { "text_completion" },
                    "created": created,
                    "model": M::MODEL_ID,
                    "choices": [],
                    "usage": usage,
                }));
            }
            sink.send(b"data: [DONE]\n\n".to_vec());
        } else {
            let body = if p.kind == Kind::Chat {
                let mut message = json!({"role": "assistant", "content": collected.content});
                if !collected.reasoning.is_empty() {
                    message["reasoning_content"] = Value::String(collected.reasoning);
                }
                if !collected.tool_calls.is_empty() {
                    if collected.content.is_empty() {
                        message["content"] = Value::Null;
                    }
                    message["tool_calls"] = Value::Array(
                        collected
                            .tool_calls
                            .iter()
                            .enumerate()
                            .map(|(i, call)| {
                                json!({
                                    "id": call_id(&id, i),
                                    "type": "function",
                                    "function": {"name": call.name, "arguments": call.arguments},
                                })
                            })
                            .collect(),
                    );
                }
                json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": created,
                    "model": M::MODEL_ID,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                    "usage": usage,
                })
            } else {
                json!({
                    "id": id,
                    "object": "text_completion",
                    "created": created,
                    "model": M::MODEL_ID,
                    "choices": [{"index": 0, "text": collected.content, "finish_reason": finish_reason, "logprobs": null}],
                    "usage": usage,
                })
            };
            sink.start(200, "application/json");
            sink.send(serde_json::to_vec(&body)?);
        }
        Ok(())
    }
}

fn call_id(request_id: &str, index: usize) -> String {
    let digest = request_id.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
    format!("call_{:016x}{index:02}", digest)
}

fn chunk(id: &str, created: u64, model: &str, delta: Value, finish_reason: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    })
}

fn text_chunk(id: &str, created: u64, model: &str, text: &str, finish_reason: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason, "logprobs": null}],
    })
}

/// Compiles the Metal pipelines a short request needs by running a throwaway
/// two-token prompt and two decode steps, so the first real request does not
/// pay the shader compile (tens of seconds on the 48-layer model). Kernels
/// only long contexts reach (sparse attention) still compile on first use.
fn warm_up<M: LanguageModel>(ctx: &MetalContext, model: &M, scratch: &mut M::Scratch) -> Result<()> {
    let started = Instant::now();
    let greedy = SamplingParams::greedy();
    let mut state = model.new_state(ctx, 4)?;
    scratch.begin_request();
    // Two arbitrary in-vocabulary ids: the values do not matter, only that the
    // prefill and decode graphs get encoded once.
    model.prefill(ctx, &mut state, scratch, &[1, 2], Some(Draw { params: &greedy, step: 0 }))?;
    for (step, (slot_in, slot_out)) in [(0, 1), (1, 0)].into_iter().enumerate() {
        let token = scratch.next_token().view(slot_in, &[1])?.to_u32()?[0];
        let encoded = model.encode_decode_step(ctx, &state, scratch, slot_in, slot_out, Draw { params: &greedy, step: step + 1 })?;
        model.prepare_step_inputs(&mut state, scratch, token)?;
        let pending = encoded.commit()?;
        state.advance(1);
        pending.wait()?;
    }
    // Snapshot/restore compile no shaders but exercise the blit path once.
    let snapshot = state.snapshot(ctx)?;
    state.restore(ctx, &snapshot)?;
    eprintln!("warm-up done in {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}

// --- startup -----------------------------------------------------------------

/// The `model_type` a checkpoint's `config.json` declares.
pub fn checkpoint_model_type(model_dir: &Path) -> Result<String> {
    let config = read_config(model_dir)?;
    config
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .with_context(|| format!("{} has no model_type", model_dir.join("config.json").display()))
}

fn read_config(model_dir: &Path) -> Result<Value> {
    let path = model_dir.join("config.json");
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).context("parsing config.json")
}

/// `eos_token_id` from `config.json` (top level or `text_config`), int or list.
fn checkpoint_eos_ids(model_dir: &Path) -> Result<Vec<u32>> {
    let config = read_config(model_dir)?;
    let value = config
        .get("eos_token_id")
        .or_else(|| config.get("text_config").and_then(|t| t.get("eos_token_id")));
    Ok(match value {
        Some(Value::Number(n)) => n.as_u64().map(|v| v as u32).into_iter().collect(),
        Some(Value::Array(items)) => items.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect(),
        _ => Vec::new(),
    })
}

/// Sampling defaults: `generation_config.json` over OpenAI's defaults, then
/// the command-line overrides.
fn sampling_defaults(model_dir: &Path, overrides: &SamplingOverrides) -> Result<SamplingParams> {
    let mut params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repetition_penalty: 1.0,
        seed: 0,
    };
    let path = model_dir.join("generation_config.json");
    if let Ok(bytes) = std::fs::read(&path) {
        let cfg: Value = serde_json::from_slice(&bytes).context("parsing generation_config.json")?;
        let f = |key: &str| cfg.get(key).and_then(Value::as_f64).map(|v| v as f32);
        if cfg.get("do_sample").and_then(Value::as_bool) == Some(false) {
            params.temperature = 0.0;
        }
        if let Some(t) = f("temperature") {
            params.temperature = t;
        }
        if let Some(p) = f("top_p") {
            params.top_p = p;
        }
        if let Some(k) = cfg.get("top_k").and_then(Value::as_u64) {
            params.top_k = k as usize;
        }
        if let Some(m) = f("min_p") {
            params.min_p = m;
        }
        if let Some(r) = f("repetition_penalty") {
            params.repetition_penalty = r;
        }
        if let Some(p) = f("presence_penalty") {
            params.presence_penalty = p;
        }
    }
    if let Some(v) = overrides.temperature {
        params.temperature = v;
    }
    if let Some(v) = overrides.top_k {
        params.top_k = v;
    }
    if let Some(v) = overrides.top_p {
        params.top_p = v;
    }
    if let Some(v) = overrides.min_p {
        params.min_p = v;
    }
    if let Some(v) = overrides.presence_penalty {
        params.presence_penalty = v;
    }
    if let Some(v) = overrides.frequency_penalty {
        params.frequency_penalty = v;
    }
    if let Some(v) = overrides.repetition_penalty {
        params.repetition_penalty = v;
    }
    params.validate()?;
    Ok(params)
}

/// Serves the checkpoint at `model_dir` with the engine its `model_type` names.
pub fn run(model_dir: &Path, options: ServeOptions) -> Result<()> {
    match checkpoint_model_type(model_dir)?.as_str() {
        "qwen3_5_moe" => run_with::<Qwen3_5Model>(model_dir, options),
        "qwen4_exp" => run_with::<Qwen4ExpModel>(model_dir, options),
        other => anyhow::bail!(
            "unsupported model_type {other:?}; lily serves qwen3_5_moe \
             (Qwen3.6-35B-A3B) and qwen4_exp (Qwen3.8-Flash-Next)"
        ),
    }
}

/// Everything the HTTP thread needs without the engine.
struct Front {
    generator: Arc<Generator>,
    defaults: Defaults,
    max_seq: usize,
    jobs: SyncSender<Job>,
    ready: Arc<AtomicBool>,
    fatal: Arc<Mutex<Option<String>>>,
    model_id: &'static str,
}

fn run_with<M: LanguageModel + 'static>(model_dir: &Path, options: ServeOptions) -> Result<()> {
    let address = options
        .bind
        .to_socket_addrs()
        .with_context(|| format!("resolving bind address {}", options.bind))?
        .next()
        .with_context(|| format!("bind address {} resolved to nothing", options.bind))?;
    let mut generator = Generator::from_model_dir(model_dir)?;
    generator.add_stop_tokens(&checkpoint_eos_ids(model_dir)?);
    let generator = Arc::new(generator);
    let defaults = Defaults {
        sampling: sampling_defaults(model_dir, &options.sampling)?,
        thinking: options.thinking,
        reasoning_effort: options.reasoning_effort.clone(),
    };
    eprintln!(
        "defaults: temperature {} top_k {} top_p {} min_p {} repetition_penalty {} presence {} frequency {}; thinking {}{}",
        defaults.sampling.temperature,
        defaults.sampling.top_k,
        defaults.sampling.top_p,
        defaults.sampling.min_p,
        defaults.sampling.repetition_penalty,
        defaults.sampling.presence_penalty,
        defaults.sampling.frequency_penalty,
        if defaults.thinking { "on" } else { "off" },
        defaults.reasoning_effort.as_deref().map(|e| format!(" (effort {e})")).unwrap_or_default(),
    );
    let max_seq = options.max_seq.min(MAX_SEQ);

    let listener = TcpListener::bind(address).with_context(|| format!("binding http://{}", options.bind))?;
    let (jobs, job_rx) = mpsc::sync_channel::<Job>(options.queue.max(1));
    let ready = Arc::new(AtomicBool::new(false));
    let fatal = Arc::new(Mutex::new(None));
    {
        let model_dir = model_dir.to_path_buf();
        let options = options.clone();
        let generator = generator.clone();
        let ready = ready.clone();
        let fatal = fatal.clone();
        std::thread::Builder::new()
            .name("lily-engine".into())
            .spawn(move || {
                let mut engine = match Engine::<M>::load(&model_dir, &options, generator) {
                    Ok(engine) => engine,
                    Err(error) => {
                        eprintln!("engine failed to start: {error:#}");
                        *fatal.lock().unwrap_or_else(|p| p.into_inner()) = Some(format!("{error:#}"));
                        return;
                    }
                };
                ready.store(true, Ordering::Release);
                eprintln!("ready: serving {} on http://{address}", M::MODEL_ID);
                while let Ok(job) = job_rx.recv() {
                    engine.serve(job);
                }
            })
            .context("spawning the engine thread")?;
    }
    eprintln!("listening on http://{address} (loading model)");

    let front = Arc::new(Front { generator, defaults, max_seq, jobs, ready, fatal, model_id: M::MODEL_ID });
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("accept error: {error}");
                continue;
            }
        };
        if let Some(fatal) = front.fatal.lock().unwrap_or_else(|p| p.into_inner()).clone() {
            send_json(stream, 500, &json!({"error": {"message": fatal, "type": "server_error"}}));
            std::process::exit(1);
        }
        let front = front.clone();
        if let Err(error) = std::thread::Builder::new()
            .name("lily-http".into())
            .spawn(move || handle(&front, stream))
        {
            eprintln!("failed to spawn connection thread: {error}");
        }
    }
    Ok(())
}

fn handle(front: &Front, mut stream: TcpStream) {
    let request = match http::read_request(&mut stream, MAX_REQUEST_BYTES) {
        Ok(request) => request,
        Err(error) => {
            send_error(stream, 400, "invalid_request_error", format!("{error:#}"));
            return;
        }
    };
    let path = request.path.split('?').next().unwrap_or("").to_string();
    let ready = front.ready.load(Ordering::Acquire);
    match (request.method.as_str(), path.as_str()) {
        ("GET", "/health") => {
            if ready {
                send_json(stream, 200, &json!({"status": "ok", "model": front.model_id}));
            } else {
                send_json(stream, 503, &json!({"status": "loading", "model": front.model_id}));
            }
        }
        ("GET", "/v1/models") => send_json(
            stream,
            200,
            &json!({
                "object": "list",
                "data": [{"id": front.model_id, "object": "model", "created": 0, "owned_by": "lily"}]
            }),
        ),
        ("POST", "/v1/chat/completions" | "/v1/completions") => {
            let prepared = if path == "/v1/chat/completions" {
                serde_json::from_slice::<api::ChatRequest>(&request.body)
                    .context("parsing the chat request")
                    .and_then(|r| api::prepare_chat(r, front.generator.tokenizer(), &front.defaults, front.max_seq))
            } else {
                serde_json::from_slice::<api::CompletionRequest>(&request.body)
                    .context("parsing the completion request")
                    .and_then(|r| api::prepare_completion(r, front.generator.tokenizer(), &front.defaults, front.max_seq))
            };
            let prepared = match prepared {
                Ok(p) => p,
                Err(error) => {
                    send_error(stream, 400, "invalid_request_error", format!("{error:#}"));
                    return;
                }
            };
            if !ready {
                send_error(stream, 503, "server_error", "model is still loading");
                return;
            }
            let (tx, rx) = mpsc::channel();
            let cancelled = Arc::new(AtomicBool::new(false));
            let job = Job { prepared, sink: Sink { tx, cancelled: cancelled.clone(), started: false }, queued_at: Instant::now() };
            match front.jobs.try_send(job) {
                Ok(()) => relay(stream, rx, cancelled),
                Err(TrySendError::Full(_)) => {
                    send_error(stream, 503, "server_error", "the request queue is full; retry later");
                }
                Err(TrySendError::Disconnected(_)) => {
                    send_error(stream, 500, "server_error", "engine stopped");
                }
            }
        }
        (_, "/health" | "/v1/models" | "/v1/chat/completions" | "/v1/completions") => {
            send_error(stream, 405, "invalid_request_error", "method not allowed");
        }
        _ => send_error(stream, 404, "invalid_request_error", "not found"),
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

#[cfg(test)]
#[path = "../tests/unit/serve.rs"]
mod tests;
