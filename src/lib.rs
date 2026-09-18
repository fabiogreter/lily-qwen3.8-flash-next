//! Lily: Lightweight Local Inference.
//!
//! A Metal (Apple Silicon) LLM inference engine built on the `objc2` bindings.
//! Kernels are compiled from MSL source at runtime, so building or running
//! this crate needs no offline `.metal` -> `.metallib` step (which would
//! require a non-hermetic Xcode Metal toolchain that Bazel does not model).

pub mod activity;
pub mod chat;
pub mod config;
#[cfg(target_os = "macos")]
pub mod engine;
#[cfg(target_os = "macos")]
pub mod generate;
#[cfg(target_os = "macos")]
pub mod kernels;
#[cfg(target_os = "macos")]
pub mod metal;

/// Diagnostic ablation of decode kernel groups: `LILY_ABLATE=a,b,c` skips the
/// named groups' dispatches (and the levels only they occupy) so a production
/// step's time without them measures their marginal cost in the level chain,
/// which the per-kernel profile mode cannot see (docs/performance.md, "Where
/// the decode step's time goes"). Results are garbage. Groups: `hc`,
/// `inject`, `gdn_proj`, `gdn_step`, `attn`, `head`, `ple`, `moe`, `router`,
/// `shared`, `topk`, `gather`, `down`, or `all`.
pub mod ablate {
    use std::sync::OnceLock;

    /// A kernel group of the decode step.
    #[derive(Clone, Copy)]
    pub enum Group {
        Hc,
        Inject,
        GdnProj,
        GdnStep,
        Attn,
        Head,
        Ple,
        Moe,
        Router,
        Shared,
        Topk,
        Gather,
        Down,
    }

    const NAMES: [(&str, Group); 13] = [
        ("hc", Group::Hc),
        ("inject", Group::Inject),
        ("gdn_proj", Group::GdnProj),
        ("gdn_step", Group::GdnStep),
        ("attn", Group::Attn),
        ("head", Group::Head),
        ("ple", Group::Ple),
        ("moe", Group::Moe),
        ("router", Group::Router),
        ("shared", Group::Shared),
        ("topk", Group::Topk),
        ("gather", Group::Gather),
        ("down", Group::Down),
    ];

    static MASK: OnceLock<u32> = OnceLock::new();

    /// Whether `group` is ablated (one bit test after the first call).
    pub fn on(group: Group) -> bool {
        let mask = MASK.get_or_init(|| {
            let Ok(list) = std::env::var("LILY_ABLATE") else { return 0 };
            let mut mask = 0u32;
            for name in list.split(',').map(str::trim) {
                if name == "all" {
                    return u32::MAX;
                }
                match NAMES.iter().find(|(n, _)| *n == name) {
                    Some((_, g)) => mask |= 1 << *g as u32,
                    None => eprintln!("LILY_ABLATE: unknown group {name:?}, ignored"),
                }
            }
            mask
        });
        mask & (1 << group as u32) != 0
    }
}
#[cfg(target_os = "macos")]
pub mod moe_ffn;
pub mod npy;
#[cfg(target_os = "macos")]
pub mod qwen4exp;
pub mod safetensors;
#[cfg(target_os = "macos")]
pub mod serve;
pub mod sha256;
#[cfg(target_os = "macos")]
pub mod tensor;
#[cfg(target_os = "macos")]
pub mod tokenizer;
#[cfg(target_os = "macos")]
pub mod weights;

#[cfg(all(target_os = "macos", test))]
#[path = "../tests/support/cpu_ref.rs"]
pub mod cpu_ref;

#[cfg(not(target_os = "macos"))]
use anyhow::bail;
