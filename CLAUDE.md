# CLAUDE.md

lily is a Metal LLM inference server for Apple Silicon. It serves
Qwen3.8-Flash-Next (a 48-layer MoE with MTP drafting, vision and an n-gram
table) over an OpenAI-compatible API on one machine, an M5 Max with 128 GB.
`README.md` is the user-facing overview, `docs/architecture.md` the design
and `CONTRIBUTING.md` the gates. Read the relevant one before changing a
subsystem; they are kept current.

The workspace rules in `../CLAUDE.md` and the global JLS principles apply on
top of this file.

## Layout

- `src/main.rs`: the `lily` server CLI (clap). `src/lib.rs` lists the modules
  and gates every GPU module on `target_os = "macos"`.
- `src/serve.rs` and `src/serve/`: the server. One engine thread owns the GPU
  and the model; HTTP threads parse requests and hand them over. The session
  cache (`session`, GPU tier) spills to the disk tier (`disk`); `batch` is
  continuous batching, `pin` the weight mlock, `http` a small HTTP/1.1 server
  on `std::net`.
- `src/qwen4exp/`: the model. `model` is the graph, `spec` MTP speculative
  decoding, `ngram` the paged n-gram table, `vision*` the vision tower,
  `expert_*` the expert cache for machines that cannot hold all experts.
- `src/kernels.rs`, `src/kernels/`: Rust dispatch. The shaders are
  `src/kernels/metal/*.metal`, compiled from MSL at runtime.
- `src/bin/`: `lily-bench` (engine benchmark without the server), `lily-probe`
  (logits for parity checks), and smaller tools (`lily-experts`,
  `lily-cache-replay`, `lily-trace-routing`, ...).
- Unit tests live in `tests/unit/<module>.rs` and are included into their
  module with `#[path = "../tests/unit/<module>.rs"] mod tests;`. Put a new
  module's tests there, not inline. `tests/support/cpu_ref.rs` holds the f32
  CPU references every kernel is checked against.
- `tools/`: Python and shell around the port (`convert/`, `reference/`,
  `bench/`, `e2e/`, `service/`); `tools/README.md` documents each.
- `docs/`: only a few files are tracked (`architecture.md`, `server.md`,
  `performance.md`, `low-ram-experts.md`, `dashboard.md`, the checkpoint
  format). Dated
  reports, `docs/bench/` and drafts are gitignored on purpose. Check with
  `git ls-files docs` before linking a doc from tracked text.

## Commands

```sh
cargo build --release --locked
cargo fmt
cargo clippy --release --locked --all-targets --all-features -- -D warnings
cargo test --locked                  # needs a Metal device, no checkpoint
```

- Checkpoint tests are `#[ignore = "requires LILY_MODEL_DIR_FLASH"]`; run them
  with the 4-layer checkpoint, `--ignored` and `--test-threads=1`.
- Timing probes are `#[ignore = "timing only"]`: `cargo test --release --lib
  -- --ignored <name> --nocapture` (`docs/performance.md`, "Reproducing this").
- Python runs from the repo venv: `.venv/bin/python tools/...`.
- The Claude sandbox has no Metal device and cannot bind ports: GPU tests,
  `lily` runs and the HTTP tests need to run outside it.
- Versions come from `.tool-versions` (asdf) and `rust-toolchain.toml`. Do
  not change either without asking.

The gates in `CONTRIBUTING.md` define "done": fmt, clippy, tests, a CPU
reference for every kernel change, numerics accounted for (bit-identical with
a test, or the 4-layer HF comparison showing near-ties), and performance
claims from an interleaved A/B with repeats.

## The machine

- **One instance.** Everything that loads the weights (the server,
  `lily-bench`, `lily-probe`, checkpoint tests) takes the machine-wide
  `/Users/Shared/lily-instance.lock` and exits 75 while another process, of
  any user, holds it.
  Two full models at once panicked this machine (2026-09-12); never work
  around the lock.
- **The background service** is the launchd agent `com.lily.server`, managed
  only with `tools/service/lily-service.sh`
  (`status|stop|start|restart|install|logs`). Log:
  `~/Library/Logs/lily/server.log`. Do not stop it, restart it or reinstall it
  without the owner's go: it may be serving a session. A dev run or benchmark
  needs it stopped first and started again afterwards.
- **Checkpoints** are in `/Users/fabio/projects/personal/local-llms/models/`:
  `Qwen3.8-Flash-Next-lily-q4-xl` (the service's), `-lily-q4`, and
  `-lily-q4-l4` (4 layers, for tests). The README's `~/models/...` paths are
  examples, not this machine.
- **Benchmarks are noisy** (3 to 4 % between identical binaries, GPU clock
  states, swap after repeated 73 GB loads). Read `docs/performance.md`, "The
  noise band", before measuring; compare interleaved, on mains power, never
  against a number from an earlier session.
- **Test and benchmark servers** run with `--stats-db off`, so they neither
  take the dashboard's stored settings nor write into the service's history
  (`docs/server.md`, "Test and benchmark servers").

## Data

Recorded sessions are real client conversations. Traces, routing, hidden
states and anything derived from them stay outside the repository (see the
`.gitignore` patterns and `tools/README.md`); never commit them, quote them
in docs or commit messages, or copy them into tests. The dashboard and its
database store numbers only, never prompt text or token ids.

## Conventions

- Comments carry the why: every module starts with a `//!` header, doc
  comments state the reasoning, measurements with their date and
  cross-references into `docs/`. Full sentences. Dependency choices are
  explained in `Cargo.toml`.
- Errors with `anyhow` (`Context`, `ensure!`, `bail!`); no `unwrap()` in
  `src/` outside tests and binaries.
- Change the docs in the same change as the behaviour they describe
  (`docs/server.md` for flags and routes, `--help` text, `tools/README.md`).
  Small stale texts can be fixed and committed directly.
- Commit subjects name the area and say what it does now, e.g. `Disk tier:
  log each entry the maximum age expires`, `docs: ...` for documentation.
  Bodies are wrapped prose or bullets explaining why. The git identity is
  set in the repo; do not override it.
- History on `main` stays linear: rebase a branch onto `main` and
  fast-forward (`git merge --ff-only`); no merge commits.
- Before anything goes to `main`, a neutral independent review pass (a fresh
  subagent), its fixes as their own commit. Merges to `main` and pushes only
  with the owner's go. Dependencies (crates like `rayon`) are fine when they
  earn their place.

## The dashboard

`dashboard/` is a Vue 3 + Vite + TypeScript app served by lily at `/` and
embedded at build time from the committed `dashboard/dist`. After changing
anything under `dashboard/src`, run `npm test`, `npm run typecheck` and
`npm run build` in `dashboard/` and commit `dist` with it (`npm run
check:dist` fails otherwise). The contract is `dashboard/src/api/types.ts`;
the Rust side regenerates `dashboard/src/api/fixtures.ts` with
`LILY_UPDATE_FIXTURES=1`. Spec and open items: `docs/dashboard.md`.
