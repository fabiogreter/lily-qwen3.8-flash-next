# Vision reference: what lily must reproduce

Facts read off the Hugging Face implementation that the goldens in `goldens/` were made
with (transformers 5.17.0.dev0, torch 2.14.0, torchvision 0.29.0, Pillow 12.3.0), for the
Rust work in docs/vision-support-plan.md items 3 to 6. File paths are relative to
`.venv/lib/python3.13/site-packages/transformers/`; line numbers are of that install.
Every measured number below is reproducible with `hf_vision_reference.py measure-*` and
is stored in `goldens/vision_tolerance_floors.json`.

## The pixel cap

- Server default: `--image-max-pixels 2097152` (2 048 x 32 x 32), `--image-min-pixels 65536`
  (the checkpoint's `size.shortest_edge`). Passed to the reference as
  `size={"shortest_edge": min, "longest_edge": max}`; `Qwen2VLImageProcessor.resize`
  (`models/qwen2_vl/image_processing_qwen2_vl.py` 138-160) hands exactly these two keys to
  `smart_resize(min_pixels=size.shortest_edge, max_pixels=size.longest_edge)`, and passing
  `min_pixels=`/`max_pixels=` instead gives identical output (checked).
- Why 2 048 tokens: patch 16 and merge 2 make one language token per 32 x 32 pixels, so the cap
  is 2 048 prompt tokens (about 1.5 s of prefill) and 8 192 tower patches with full attention.
- The checkpoint itself has `longest_edge` 16 777 216, effectively no cap.

| image | default cap: resized w x h, grid (h, w), tokens | uncapped |
|---|---|---|
| 333 x 777 | 320 x 768, (48, 20), 240 | same |
| 640 x 480 | 640 x 480, (30, 40), 300 | same |
| 1920 x 1080 | 1920 x 1088, (68, 120), 2 040 | same |
| 3840 x 2160 | 1920 x 1056, (66, 120), 1 980 | 3840 x 2176, (136, 240), 8 160 |

The plan said the Retina capture is "halved to its logical size"; the cap branch of
`smart_resize` uses `floor`, so it comes out as 1920 x 1056, not 1920 x 1088.

## Preprocessing (`Qwen2VLImageProcessor`, torchvision backend)

The checkpoint's `preprocessor_config.json` names `Qwen2VLImageProcessorFast`; in this
transformers that is `Qwen2VLImageProcessor(TorchvisionBackend)` in
`models/qwen2_vl/image_processing_qwen2_vl.py`. The chain, in order:

1. **Colour**: `image_transforms.convert_to_rgb` (757-773): `PIL.Image.convert("RGB")` when the
   mode is not RGB. No alpha compositing: an RGBA image simply drops its alpha.
2. **To tensor**: `image_processing_backends.py` 116-149: `tvF.pil_to_tensor` gives uint8 CHW.
3. **smart_resize** (`image_processing_qwen2_vl.py` 60-84), `factor = patch_size * merge_size = 32`,
   on (height, width) of the source, Python `round` (banker's rounding), `floor`, `ceil`:
   ```
   if max(h, w) / min(h, w) > 200: error
   h_bar = round(h / 32) * 32;  w_bar = round(w / 32) * 32
   if h_bar * w_bar > max_pixels:
       beta = sqrt(h * w / max_pixels)
       h_bar = max(32, floor(h / beta / 32) * 32);  w_bar = max(32, floor(w / beta / 32) * 32)
   elif h_bar * w_bar < min_pixels:
       beta = sqrt(min_pixels / (h * w))
       h_bar = ceil(h * beta / 32) * 32;  w_bar = ceil(w * beta / 32) * 32
   ```
   Transcribed and asserted equal to the installed function in
   `hf_vision_reference.smart_resize_reference`.
4. **Resample** (`image_processing_backends.py` 205-258): `PILImageResampling.BICUBIC` maps to
   `torchvision InterpolationMode.BICUBIC` (`image_utils.py` 58-65) and the call is
   `tvF.resize(uint8 CHW, [h_bar, w_bar], BICUBIC, antialias=True)`. Measured facts about that call:
   - It equals `PIL.Image.resize(..., Image.BICUBIC)` within **one uint8 level** on every test image
     (fraction of differing values 2e-6 to 1e-4). It is PIL's algorithm: Keys cubic with **a = -0.5**
     (the antialiased kernel fits a = -0.5 to 1e-7 and a = -0.75 to 0.035; torchvision's
     non-antialiased bicubic is a = -0.75), filter support scaled by the downscale factor,
     **separable with clamp-and-round to uint8 after each pass**.
   - A float bicubic (a = -0.5, antialiased) rounded once at the end is **not** the same: up to
     12 levels apart on 333 x 777 and 11 on 3840 x 2160 (where both axes shrink), 1 level on
     1920 x 1080 (one axis grows, one is identity). Unrounded, up to 30 levels. Non-antialiased
     bicubic: up to 89 levels. Identity resize (640 x 480) is exact on every path.
   - So lily's resampler must round to uint8 between the horizontal and vertical pass, in PIL's
     order, with PIL's kernel, to land inside the tolerance below.
5. **Rescale and normalise** (`image_processing_backends.py` 295-337), fused, in float32:
   `(x - mean / rescale) / (std / rescale)` with rescale 1/255, mean = std = 0.5, so a uint8 level
   `k` becomes `(k - 127.5) / 127.5`, range [-1, 1], one level = 2/255 = 0.007843.
6. **Patchify** (`image_processing_qwen2_vl.py` 163-200). With `gh = h_bar / 16`, `gw = w_bar / 16`:
   `reshape(B, 3, gh/2, 2, 16, gw/2, 2, 16)`, `permute(0, 2, 5, 3, 6, 1, 4, 7)` to
   `[B, gh/2, gw/2, 2, 2, 3, 16, 16]`, insert the temporal axis after channel and expand to 2, then
   `reshape(B, gh * gw, 3 * 2 * 16 * 16)`. Hence:
   - **Row order** is block-major: row `r = ((bh * (gw/2) + bw) * 2 + ih) * 2 + iw` holds the patch
     at pixel rows `(2 bh + ih) * 16 ..` and columns `(2 bw + iw) * 16 ..`. Four consecutive rows
     are one 2 x 2 merge block, row-major inside the block.
   - **Row layout** is `index = ((c * 2 + t) * 16 + py) * 16 + px`, channel-major, 1 536 floats; the
     two temporal frames are identical copies of the still image.
   - `pixel_values`: float32 `[gh * gw, 1536]`. `image_grid_thw = [1, gh, gw]` (temporal **1**
     although the row carries two frames). Tokens = `gh * gw / 4`.

## The tower (`Qwen4ExpVisionModel`, `models/qwen4_exp/modeling_qwen4_exp.py`)

Config (`vision_config`): depth 27, hidden 1152, 16 heads of 72, MLP 4304, `gelu_pytorch_tanh`,
patch 16, temporal patch 2, merge 2, 3 channels, 2304 position embeddings (a 48 x 48 grid),
`out_hidden_size` 2560, `deepstack_visual_indexes` empty. Weights: 333 bf16 tensors under
`model.visual.` (patch_embed.proj.{weight [1152,3,2,16,16], bias}, pos_embed.weight [2304,1152],
blocks.N.{norm1,norm2,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}.{weight,bias},
merger.{norm,linear_fc1,linear_fc2}.{weight,bias}).

Forward (1930-1987), for one image with grid (1, gh, gw), N = gh * gw patches:

1. **Patch embed** (1735-1752): the row is viewed as `[3, 2, 16, 16]` and passed through
   `Conv3d(3, 1152, kernel (2, 16, 16), stride same, bias)`; with one kernel step per patch that is a
   linear map 1536 -> 1152 whose weight is the conv weight flattened in (C, T, H, W) order, the same
   order as the pixel row. Input is cast to the tower's dtype first (`hidden_states.to(target_dtype)`,
   1751; `get_image_features` 2171 casts pixel_values to `self.visual.dtype` as well).
2. **Position embedding** (1917-1922, 1948-1956; `vision_utils.py` 191-297): the 48 x 48 table is
   resampled to (gh, gw) **bilinearly with align_corners=True**: per axis, for target index `i` on an
   axis of size `n`, `src = i * 47 / max(n - 1, 1)`, taps `floor(src)` and `floor(src) + 1` clamped
   to [0, 47], weights `1 - |src - tap|`; the four 2-D taps are the outer product; result
   `sum(w_k * table[tap_k])` with float32 weights (so the sum is float32 even for a bf16 table, 1962), cast to the activation dtype and added to the patch embedding (1963). Rows are
   emitted in the same block-major order as the patches.
3. **Rotary** (1710-1719, 1957-1963; `vision_utils.get_vision_position_ids` 81-127): `dim = 72 / 2 = 36`,
   `inv_freq[i] = 1 / 10000 ** (2 i / 36)` for i in 0..17 (float32). Each patch has absolute grid
   coordinates (row, col) in 0..gh-1, 0..gw-1 (not divided by the merge size, not offset), in
   block-major order; `rotary = [row * inv_freq (18), col * inv_freq (18)]`, `emb = cat(rotary, rotary)`
   (72), cos/sin of it. `apply_rotary_pos_emb_vision` (1771-1782) upcasts q and k to float32 and applies
   `q * cos + rotate_half(q) * sin` over the whole 72-wide head, `rotate_half` pairing (i, i + 36).
4. **Block** (1868-1896): `x = x + attn(norm1(x)); x = x + mlp(norm2(x))`, both `LayerNorm(1152, eps 1e-6)`
   with bias.
5. **Attention** (1785-1865): `qkv = Linear(1152, 3456, bias)`, reshaped `(N, 3, 16, 72)`; rotary on q
   and k; scale `72 ** -0.5`; **full bidirectional attention over all N patches of the image** (one
   segment per image, `cu_seqlens = [0, N]`, `vision_utils.py` 42-65; no mask, `is_causal=False`);
   eager softmax in float32 (`eager_attention_forward`, 821); `proj = Linear(1152, 1152, bias)`.
6. **MLP** (1722-1732): `linear_fc2(gelu_tanh(linear_fc1(x)))`, 1152 -> 4304 -> 1152, both with bias.
7. **Merger** (1755-1768): `LayerNorm(1152, eps 1e-6)` on every patch (the pre-shuffle norm;
   `use_postshuffle_norm=False`), then `view(-1, 4608)` so the four rows of one merge block are
   concatenated in their row order (ih, iw) = (0,0), (0,1), (1,0), (1,1); `linear_fc1 = Linear(4608, 4608)`,
   **`nn.GELU()` which is the exact erf GELU, not tanh**, `linear_fc2 = Linear(4608, 2560)`. Output
   `[N / 4, 2560]`, one row per merge block in raster order of the (gh/2, gw/2) grid.
8. `last_hidden_state` is the pre-merger `[N, 1152]`; `pooler_output` is the merged `[N / 4, 2560]`
   (goldens: `pre_merger` and `merged`).

The merged rows replace the `<|image_pad|>` rows of the token embedding in order
(`Qwen4ExpModel.forward` 2302-2311, `inputs_embeds.masked_scatter(input_ids == 248056, image_embeds)`,
after casting to the embedding dtype). Nothing else enters the language model.

Scale of the outputs on the test images: merged mean 0.0013, std 0.028, |max| 0.39; pre-merger
std about 160 with single elements up to 10 400 (per-token norms span four orders of magnitude).

## Prompt, tokens, positions

- Token ids (config.json): `<|vision_start|>` 248053, `<|vision_end|>` 248054, `<|image_pad|>` 248056,
  `<|video_pad|>` 248057. The chat template emits `<|vision_start|><|image_pad|><|vision_end|>` per
  image item, inline with the surrounding text of the same content array.
- The processor (`models/qwen3_vl/processing_qwen3_vl.py` 80-83) replaces the one `<|image_pad|>` with
  `grid_t * grid_h * grid_w / merge_size ** 2` copies, then tokenises. `mm_token_type_ids`
  (`processing_utils.py` 925-955) is 1 at `<|image_pad|>` and 0 elsewhere; `<|vision_start|>` and
  `<|vision_end|>` are text (0).
- **Position ids** (`Qwen4ExpModel.get_rope_index` 2057-2148 with `get_vision_position_ids` 2005-2055;
  entered from `compute_3d_position_ids` 2223-2270). Runs of equal token type are visited in order with a
  counter `p` starting at 0:
  - a text run of length n: all three axes get `p, p+1, ..., p+n-1`; `p += n`;
  - an image with grid (1, gh, gw): over the merged grid `(gh/2, gw/2)` in raster order,
    `t = p`, `h = p + row`, `w = p + col`; then `p += max(gh, gw) / 2`.
  - `rope_deltas = max(position over all axes) + 1 - seq_len` (one integer per sequence).
  - Every token generated afterwards, at sequence index `s`, gets `s + rope_deltas` on all three axes
    (2254-2266; `_prepare_position_ids_for_generation` 2513-2549 does the same for `generate`).
  Golden 333 x 777: text 0..19, `<|vision_start|>` at 19, image rows 20..259 with t = 20, h = 20..43,
  w = 20..29, `<|vision_end|>` at 260 with position 44, max position 65, `rope_deltas` -216
  (`hf_vision_positions_333x777_cap2097152.json`). The forward golden asserts that the language model
  saw exactly these ids and kept exactly this delta.
- **Interleaved M-RoPE** in the text model (85-160): `dim = 256 * 0.25 = 64`, 32 frequency pairs,
  `inv_freq[i] = 1 / 1e7 ** (2 i / 64)`; `freqs[axis] = pos[axis] * inv_freq`; `mrope_section` [11, 11, 10]
  interleaves as: pair `i` takes the **T** axis when `i % 3 == 0` (11 pairs: 0, 3, ..., 30), **H** when
  `i % 3 == 1` and `i < 33` (11 pairs: 1, 4, ..., 31), **W** when `i % 3 == 2` and `i < 30` (10 pairs:
  2, 5, ..., 29). `emb = cat(freqs, freqs)`, cos/sin in float32 then cast to the activation dtype, applied
  to the first 64 of the 256 head dims with `rotate_half` pairing (i, i + 32) (`apply_rotary_pos_emb` 638-660).
  For text-only prompts all axes coincide and this is the engine's current 1-D RoPE.
- The **QSA indexer** (683-780) uses the same cos/sin: queries at their own positions (724),
  block keys at the cos/sin of the **first position of each block** (`group_starts`, 758-759).

## Goldens and tolerances

Goldens (all at the default cap unless the name says otherwise):
`hf_vision_preprocess_<img>_cap<max>.json`, `hf_vision_tower_<img>_cap<max>.json`,
`hf_vision_positions_<img>_cap<max>.json` for the four images,
`hf_vision_preprocess_3840x2160_cap16777216.json` (uncapped token count only, no `.npy`),
`hf_l4_vision_333x777_dequant.json` (4 text layers from the dequantized lily checkpoint, tower from the
raw checkpoint, bf16, greedy 4) and `hf_l4_vision_textonly_dequant.json` (the same prompt without the
image; bit-identical argmax, top-8 logits and greedy tokens to `hf_reference.py` on
`l4_vision_textonly_tokens.json`). Full tensors live in `goldens/large/` (untracked, 232 MB).

**Comparison 1, preprocessing** (`PREPROCESS_TOLERANCE`): the floor is PIL against the reference's
torchvision path, at most one uint8 level; a bf16 copy of the reference moves by at most 0.00193.
Default: `atol` 0.00984 (one level + 0.002), at least 99.9 % of the elements within it, no element
beyond `max_atol` 0.0255 (three levels + 0.002); `image_grid_thw` and the resized size exact. An
implementation that rounds once at the end fails this on downscaled images (up to 12 levels); one
without antialiasing fails by a wide margin.

**Comparison 2, tower** (`TOWER_TOLERANCE`): the floor is the reference tower in bf16 on MPS against
itself in f32 on the CPU on the same `pixel_values` (float32 inv_freq in both; bf16 eager attention on
the CPU was abandoned after 50 minutes on the first large image). Merged output, 333 x 777 / 640 x 480 /
1920 x 1080 / 3840 x 2160: relative L2 error 0.054 / 0.050 / 0.065 / 0.078, cosine 0.99856 / 0.99877 /
0.99789 / 0.99694, max abs error 0.055 / 0.033 / 0.127 / 0.076, fraction within `0.02 + 0.05 |golden|`
0.9995 / 1.0000 / 0.9996 / 0.9994. Pre-merger relative L2 0.042 to 0.061. The CPU bf16 run on the small
images gave the same picture (relative L2 0.053 / 0.064, cosine 0.99859 / 0.99866; per-token relative
L2 up to 0.51 on flat-colour tokens with tiny norms). Defaults: `atol` 0.02, `rtol` 0.05,
`min_cosine` 0.995, `max_rel_l2` 0.10 on the merged output; the pre-merger states are reported, not
judged. The within-fraction gate is relative to the measured floor: the threshold is the bf16
reference's own fraction on that image and cap (`vision_tolerance_floors.json`,
`tower_bf16_vs_f32.images.<img>.merged_bf16_vs_f32.fraction_within`: 0.99954 / 0.99999 / 0.99961 /
0.99937) minus a margin of 0.002; an image with no recorded floor falls back to an absolute 0.999 and
`compare_vision.py` says so. The absolute gate was not separable: the elements outside the tolerance
sit in a handful of tokens (5 of 240 on 333 x 777) whose massive-activation channel, near 1e4 in the
residual, flips by about 1 000 under bf16 rounding in the reference as much as in lily's port, and the
reference itself with an f32 residual stream lands at 0.99878 on 333 x 777. lily's residual tracks the
f32 tower block by block as closely as the bf16 reference does (relative L2 0.036 against 0.035 after 27
blocks) and its merger reproduces the reference merger on the same input to 0.0005, so the spread is
rounding, not a defect. The margin covers the measured run-to-run spread of the bf16 reference with
room: CPU against MPS about 0.0005 on the small images, the f32-residual variant 0.0008. A candidate
with Gaussian noise at the bf16 level (std 0.0016) passes, three times that fails on cosine and
relative L2.

**Comparison 3, positions**: exact on `input_ids`, `mm_token_type_ids`, `image_grid_thw`, the three
position axes and `rope_deltas`.

**Comparison 4, forward**: `compare.py` semantics (argmax agreement asserted, top-8 overlap and logit gap
reported) on the 4-layer goldens; the positions block is checked exactly when the candidate carries one.

## Reference cost on this machine (CPU, float32 tower, eager attention)

| image | patches | tokens | tower | peak RSS |
|---|---|---|---|---|
| 333 x 777 | 960 | 240 | 1.2 s | 3.8 GB |
| 640 x 480 | 1 200 | 300 | 1.3 s | 3.8 GB |
| 1920 x 1080 | 8 160 | 2 040 | 22.2 s | 11.3 GB |
| 3840 x 2160 (capped) | 7 920 | 1 980 | 20.7 s | 10.9 GB |

The 4-layer forward with the 333 x 777 image (282 tokens) took 26 s of prefill after a 12 s build and
peaked at 54 GB RSS (the dequantized text layers dominate); the text-only control 2 s of prefill.

## Notes for the Rust items

- The bf16 forward goldens (this one and the existing text ones) carry the text rotary `inv_freq` in
  **bfloat16**: `hf_reference.build_model` assigns the buffer and then calls `model.to(bf16)`, which
  casts it. `from_pretrained` would keep it float32. The vision goldens mirror the existing harness so
  the text-only control reproduces it exactly; the tower's `inv_freq` is float32 in every golden.
  Worth deciding deliberately before item 5 adds a second rotary path.
- The draft head's handling of image rows (plan, item 5) is not covered by these goldens; the reference
  MTP path was not exercised here.
- Video is out of scope: `video_grid_thw` splits per frame and adds timestamp text, none of which the
  goldens cover.
