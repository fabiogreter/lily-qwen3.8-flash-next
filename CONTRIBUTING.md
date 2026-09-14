# Contributing

## Building

```sh
cargo build --release --locked
cargo fmt
cargo clippy --release --locked --all-targets --all-features -- -D warnings
```

`rust-toolchain.toml` pins the compiler. Formatting is `rustfmt.toml`:
88 columns, edition-2024 style.

## Tests

```sh
cargo test --locked --lib                      # Rust/Metal kernel checks
cargo test --locked --test test_shader_compile # every kernel builds a pipeline
```

Those two need a Metal device and nothing else. The checkpoint-backed
tokenizer, 35B golden and Qwen3.8-Flash-Next tests read
`LILY_MODEL_DIR_35B` and `LILY_MODEL_DIR_FLASH` and are ignored when those
are unset; the README lists the invocations.

Two things to know before trusting a green run:

- **A skipped test reports `ok`.** libtest captures the stdout of passing
  tests, so a target that found no checkpoint and returned early looks exactly
  like one that ran. The duration is the tell: `0.00s` means it did nothing.
- **The tests share one GPU.** Pass `--test-threads=1` for anything that loads
  a model.

## Adding a kernel

Every kernel is checked against a plain-Rust f32 reference in
`tests/support/cpu_ref.rs`, not against a previous version of itself. A new
kernel needs its reference in the same change.

If a kernel has a shape-dependent route, the route needs a boundary test: the
predicate at each edge, and numeric agreement on both sides of it. A route that
silently falls back is worse than one that errors, because a benchmark of the
fallback reads as a benchmark of the kernel.

A pass holds its buffers by GPU address only, with no reference back to them,
so every buffer a pass reads must outlive the pass's completion. Production
code binds scratch and state tensors, which live as long as the session; a
test has to keep its inputs in locals, or the temporary is freed before the
GPU runs.

## The gates

A change is ready when all of these hold.

1. `cargo fmt` leaves no diff.
2. `cargo clippy --release --locked --all-targets --all-features -- -D warnings`
   is clean.
3. `cargo test --locked` passes, including the shader compilation test.
4. Every new or changed kernel has a CPU reference test, and every
   shape-dependent route has a boundary test.
5. The numerics are accounted for. Either the change is bit-identical to what
   it replaces, and a test asserts that, or it is not, and then it carries its
   evidence: the 4-layer Hugging Face comparison in `tools/reference/`
   (argmax agreement and the worst shared-id logit gap), and, where outputs
   move, the demonstration that the divergences are near-ties rather than
   errors. "The digest changed" is not a finding on its own; which token
   changed and by what logit margin is.
6. Performance claims come from an interleaved A/B with repeats, not from a
   single run against an earlier number. See [docs/performance.md](docs/performance.md)
   for the noise band and what distorts a measurement.
