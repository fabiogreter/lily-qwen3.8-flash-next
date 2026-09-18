//! Qwen3.8-Flash-Next (`qwen4_exp`): configuration, checkpoint loading, and
//! the model graph. Shares the Metal kernels with the Qwen3.5 path where the
//! architectures agree (Gated DeltaNet, sparse MoE, dense attention) and adds
//! hyper-connections, the per-layer n-gram embedding, and Qwen Sparse
//! Attention on top.

pub mod config;
pub mod expert_cache;
pub mod expert_store;
pub mod image;
pub mod model;
pub mod ngram;
pub mod positions;
pub mod probe;
mod spec;
pub mod vision;
pub mod vision_weights;
pub mod weights;

pub use expert_cache::ExpertCacheStats;
pub use model::{
    DecodeState, Encoder, ImageEmbeds, MAX_DRAFTS, Qwen4ExpModel, Scratch, VisionInput,
};
pub use ngram::NgramStorage;
pub use positions::{ImageSpan, Positions, positions_for_prompt};
