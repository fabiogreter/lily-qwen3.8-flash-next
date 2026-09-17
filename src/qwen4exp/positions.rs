//! Token positions of a prompt that may contain images (`docs/vision-support-plan.md`
//! item 5; the rule is `tools/reference/VISION.md`, "Prompt, tokens,
//! positions", and the four `hf_vision_positions_*` goldens pin it).
//!
//! For text, every token's position is its sequence index on all three
//! rotary axes, which is the engine's scalar RoPE. An image breaks that: its
//! placeholder rows share one temporal position and spread over the merged
//! grid on the height and width axes, and the text after it continues from
//! a position smaller than its sequence index. Positions are a pure function
//! of the tokens and the image spans, so nothing about them is stored with a
//! session: a resumed session's cached keys are valid whenever its tokens and
//! image spans match the prompt's, and the engine recomputes the positions
//! from the prompt every time.

use anyhow::{Result, ensure};
use serde::Serialize;

/// One image's run of `<|image_pad|>` tokens in a prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ImageSpan {
    /// Sequence index of the first placeholder.
    pub start: usize,
    /// Placeholders: `grid_h * grid_w / 4`, one per 2 x 2 merge block.
    pub len: usize,
    /// The image's patch grid (`image_grid_thw[1]`, an even count).
    pub grid_h: usize,
    /// The image's patch grid (`image_grid_thw[2]`, an even count).
    pub grid_w: usize,
}

impl ImageSpan {
    /// The merged grid the placeholders cover, `(rows, cols)`.
    pub fn merged_grid(&self) -> (usize, usize) {
        (self.grid_h / 2, self.grid_w / 2)
    }

    /// Sequence index one past the last placeholder.
    pub fn end(&self) -> usize {
        self.start + self.len
    }
}

/// The 3-axis rotary positions of every token of a prompt and the offset
/// every later token takes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Positions {
    /// Per token: `[temporal, height, width]`.
    pub rows: Vec<[u32; 3]>,
    /// `max(position) + 1 - seq_len` (the reference's `rope_deltas`): a
    /// token generated at sequence index `s` sits at `s + rope_delta` on all
    /// three axes. Zero for a text-only prompt.
    pub rope_delta: i64,
}

impl Positions {
    /// Tokens covered.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Whether every token's three axes equal its sequence index, in which
    /// case the scalar RoPE with delta 0 is exact.
    pub fn is_identity(&self) -> bool {
        self.rope_delta == 0
            && self.rows.iter().enumerate().all(|(i, row)| {
                let p = i as u32;
                *row == [p, p, p]
            })
    }

    /// The rows of sequence indices `range`, flattened for upload.
    pub fn flat(&self, range: std::ops::Range<usize>) -> Vec<u32> {
        self.rows[range].iter().flatten().copied().collect()
    }
}

/// Positions of `tokens` with `images` at their spans, by the reference's
/// rule (`Qwen4ExpModel.get_rope_index`): a counter `p` starts at 0; a text
/// token takes `p` on all axes and advances it by one; an image's
/// placeholders take `t = p`, `h = p + row`, `w = p + col` over its merged
/// grid in raster order and then advance `p` by `max(grid_h, grid_w) / 2`;
/// `rope_delta = max(position) + 1 - seq_len`. Spans must be in order,
/// disjoint, inside the prompt and sized to their grids.
pub fn positions_for_prompt(tokens: &[u32], images: &[ImageSpan]) -> Result<Positions> {
    let n = tokens.len();
    let mut rows: Vec<[u32; 3]> = Vec::with_capacity(n);
    let mut p: usize = 0;
    let mut next = 0usize;
    for (k, span) in images.iter().enumerate() {
        ensure!(
            span.start >= next,
            "image span {k} at {} overlaps or precedes the span before it (ends at {next})",
            span.start
        );
        ensure!(
            span.end() <= n,
            "image span {k} ({}..{}) exceeds the prompt of {n} tokens",
            span.start,
            span.end()
        );
        ensure!(
            span.grid_h >= 2
                && span.grid_w >= 2
                && span.grid_h.is_multiple_of(2)
                && span.grid_w.is_multiple_of(2),
            "image span {k}: grid ({}, {}) is not made of 2 x 2 merge blocks",
            span.grid_h,
            span.grid_w
        );
        let (mh, mw) = span.merged_grid();
        ensure!(
            span.len == mh * mw,
            "image span {k}: {} placeholders for a ({}, {}) grid, expected {}",
            span.len,
            span.grid_h,
            span.grid_w,
            mh * mw
        );
        for _ in next..span.start {
            rows.push([p as u32; 3]);
            p += 1;
        }
        for i in 0..span.len {
            let (row, col) = (i / mw, i % mw);
            rows.push([p as u32, (p + row) as u32, (p + col) as u32]);
        }
        p += mh.max(mw);
        next = span.end();
    }
    for _ in next..n {
        rows.push([p as u32; 3]);
        p += 1;
    }
    let max_position = rows.iter().flatten().copied().max().map_or(0, |m| m as i64 + 1);
    let rope_delta = if rows.is_empty() { 0 } else { max_position - n as i64 };
    Ok(Positions { rows, rope_delta })
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/positions.rs"]
mod tests;
