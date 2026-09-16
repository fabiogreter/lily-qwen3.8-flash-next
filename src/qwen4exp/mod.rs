//! Qwen3.8-Flash-Next (`qwen4_exp`): configuration, checkpoint loading, and
//! the model graph. Shares the Metal kernels with the Qwen3.5 path where the
//! architectures agree (Gated DeltaNet, sparse MoE, dense attention) and adds
//! hyper-connections, the per-layer n-gram embedding, and Qwen Sparse
//! Attention on top.

pub mod config;
pub mod model;
pub mod ngram;
mod spec;
pub mod vision_weights;
pub mod weights;

pub use model::{DecodeState, Encoder, MAX_DRAFTS, Qwen4ExpModel, Scratch};
pub use ngram::NgramStorage;
