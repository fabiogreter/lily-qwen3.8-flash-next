# Qwen3.8-Flash-Next checkpoint format for lily

This is the on-disk layout lily's `qwen4_exp` loader reads. It is produced by
`tools/convert/convert_qwen38_flash_next.py` from the raw Hugging Face BF16
checkpoint `Qwen/Qwen3.8-Flash-Next` (revision `f5d08274`). The format is
lily's own: upstream mlx-lm has no `qwen4_exp` implementation, so there is no
MLX-community convention to match.

## Directory

```
config.json                       HF config.json + a top-level "lily" block (see below)
model-XXXXX-of-YYYYY.safetensors  quantized text-model tensors, ~2 GB shards
mtp-XXXXX-of-YYYYY.safetensors    the multi-token-prediction draft head (optional, see below)
vision-XXXXX-of-YYYYY.safetensors the vision tower, unquantized bf16 (optional, see below)
model.safetensors.index.json      standard HF weight_map (covers all shard families)
tokenizer.json, tokenizer_config.json, chat_template.jinja,
generation_config.json            copied verbatim from the source checkpoint
```

The `mtp.*` draft head (one full-attention decoder block,
`pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `fc_embedding`, `fc_hidden`, and
its own `hyper_connection_mixer`) is converted with the same rules as the trunk
(`fc_embedding`/`fc_hidden` at 8 bits, the two `pre_fc_norm_*` weights
verbatim) and written to `mtp-*` shards: by default in the same run, or later
with `--mtp-only` into an existing conversion, which merges the index and adds
a `lily.mtp` block to `config.json`. `--no-mtp` drops it.

The vision tower (`model.visual.*`: 333 bf16 tensors, 0.90 GB; 27 blocks of
hidden size 1152, the patch embedding, the 2304-row position table and the
merger, see `tools/reference/VISION.md`) is **copied byte for byte, unquantized**,
to `vision-*` shards. It is 1.3 % of the checkpoint and runs once per image,
and the tower's tolerance floor against the reference is already only relative
L2 0.05, so quantization noise has no room there. By default it is written in
the same run; `--vision-only` appends it to an existing conversion (merging the
index, refusing if `vision-*` shards exist) and `--no-vision` drops it. Names
and shapes are the Hugging Face ones verbatim (`patch_embed.proj.weight` stays
`[1152, 3, 2, 16, 16]`; lily views it as `[1152, 1536]` at load, which is the
same bytes).

`config.json` matches the weights in both directions: with the tower present
it keeps the wrapper's `vision_config` and the four token ids (`image_token_id`,
`video_token_id`, `vision_start_token_id`, `vision_end_token_id`) and carries a
`lily.vision` block; without it those keys are removed and `model.visual.` is
listed in `lily.dropped`. The loader reads `lily.vision` to decide whether to
expect the tensors, and refuses a checkpoint whose files hold `model.visual.*`
tensors without the block (or the reverse).

## Tensor naming

Names are the Hugging Face names verbatim (`model.language_model.layers.N.*`,
`model.language_model.embed_tokens`, `model.language_model.hyper_connection_mixer.*`,
`lm_head`). A quantized linear or embedding `foo.weight` becomes three tensors:

| suffix    | dtype | shape                              |
|-----------|-------|------------------------------------|
| `.weight` | U32   | `[out, in * bits / 32]`            |
| `.scales` | BF16  | `[out, in / group_size]`           |
| `.biases` | BF16  | `[out, in / group_size]`           |

Packing is MLX affine: `bits`-wide codes packed low-element-first into each
u32 along the input dimension, and `w = scale * q + bias` per `group_size`
input elements. Produced with `mlx.core.quantize(w, group_size, bits)` so the
layout is bit-identical to what lily's existing Q4/Q8 kernels already consume.

Stacked expert tensors keep a leading expert dimension: `[E, out, in*bits/32]`
and `[E, out, in/group_size]`. lily flattens the leading dims at load.

### Quantization policy

| tensors                                                                 | bits | group |
|-------------------------------------------------------------------------|------|-------|
| experts `gate_proj`, `up_proj`, `down_proj`                             | 4    | 64    |
| attention `q_proj`, `k_proj`, `v_proj`, `o_proj`                        | 4    | 64    |
| GDN `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`    | 4    | 64    |
| shared expert `gate_proj`, `up_proj`, `down_proj`                       | 4    | 64    |
| `embed_tokens`, `lm_head`                                               | 4    | 64    |
| PLE n-gram table `ple.ple_embedding.ngram_embedding.shard_i`            | 4    | 32    |
| MoE router `mlp.gate`, `mlp.shared_expert_gate`                         | 8    | 64    |
| hyper-connection `input_mix_weight_down`, `input_mix_weight_up`, `block_inject_weight` (per layer and the model-level mixer) | 8 | 64 |
| PLE `key_proj`, `value_proj`                                            | 8    | 64    |
| QSA indexer `index_qk_proj`                                             | 8    | 64    |
| MTP `fc_embedding`, `fc_hidden`                                         | 8    | 64    |

The n-gram table rows are 160 wide, which is not a multiple of 64, hence
group 32 there. Routers, gates and the small mixing projections stay at 8 bits
because they steer the computation and cost almost nothing.

### Expert split

HF stores `mlp.experts.gate_up_proj` as `[E, 2*I, H]` with the gate rows first
and `mlp.experts.down_proj` as `[E, H, I]`. The converter splits gate/up into

```
model.language_model.layers.N.mlp.experts.gate_proj.{weight,scales,biases}   [E, I, ...]
model.language_model.layers.N.mlp.experts.up_proj.{weight,scales,biases}     [E, I, ...]
model.language_model.layers.N.mlp.experts.down_proj.{weight,scales,biases}   [E, H, ...]
```

so lily's separate gate/up expert stacks apply unchanged.

### Unquantized tensors (BF16, HF values verbatim)

- Every RMSNorm weight (`input`-less here: `hc_norm`, `q_norm`, `k_norm`,
  indexer `q_layernorm`/`k_layernorm`, PLE `norm_key`/`norm_query`/`norm_conv`).
  These are **zero-centered**: the gain is `1 + w`. lily applies the `+1` in
  the kernel (`w_bias = 1.0`) and does not subtract anything at load.
- GDN `norm.weight` (plain gain, init ones), `A_log`, `dt_bias`, `conv1d.weight`
  (`[C, 1, KD]`, transposed to tap-major at load).
- PLE `conv1d.weight` (`[4H, 1, 4]`, dilation 3).

### PLE hashing constants

`ple.ple_embedding.layer_multipliers` (3 x i64), `ngram_heads_vocab_sizes`
(16 x i64) and `ngram_heads_offsets` (16 x i64) are copied out of the source
checkpoint into `config.json` under `lily.ple` as plain JSON integers, and the
converter asserts they match the formulas in
`transformers/models/qwen4_exp/modeling_qwen4_exp.py` (`_build_layer_multipliers`,
`_find_nth_prime_after`). safetensors I64 is not a dtype lily loads.

## `config.json` additions

```json
"lily": {
  "format": "qwen4_exp-affine-v1",
  "source_repository": "Qwen/Qwen3.8-Flash-Next",
  "source_revision": "f5d08274bafd880402bd16f5e3e6c514136ec06c",
  "layers": 48,                       // number of decoder layers kept (truncation for tests)
  "quantization": {"default": {"bits": 4, "group_size": 64}, ...per-tensor overrides...},
  "ple": {"layer_multipliers": [...], "ngram_heads_vocab_sizes": [...], "ngram_heads_offsets": [...]},
  "mtp": {"layers": 1, "layer_types": ["full_attention"], "rope_theta": 10000000, ...}, // or null
  "vision": {"dtype": "bf16", "tensors": 333, "bytes": 897862112},                  // or null
  "dropped": []                       // tensor name prefixes left out: "model.visual." and/or "mtp."
}
```

The engine (`src/qwen4exp/config.rs`) ignores keys it does not know, so a
binary from before a block was added still reads a newer `config.json`. With
`lily.vision` set it also parses `vision_config` and the token ids and checks
what lily's vision path is written for (`gelu_pytorch_tanh`, no deepstack
injection, `out_hidden_size` equal to the text hidden size, 2 x 2 merge over
2-frame patches, `rope_parameters.mrope_section` `[11, 11, 10]` interleaved);
a checkpoint outside that fails at load naming the field.

`text_config.num_hidden_layers` and `text_config.layer_types` are rewritten to
the kept layer count when the converter truncates (`--layers N`), so a
truncated checkpoint is a valid smaller model that the HF reference can run too.
