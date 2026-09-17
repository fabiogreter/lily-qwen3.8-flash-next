#!/usr/bin/env python3
"""Reference goldens for lily's vision path, from the Hugging Face implementation
of Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration` and friends).

One subcommand per comparison in docs/architecture.md ("How the tower was verified"), plus the two
measurements that set the tolerances:

  preprocess   pixel_values and image_grid_thw of one image at one pixel cap
  tower        Qwen4ExpVisionModel in float32 on the CPU for that input
  positions    expanded input_ids, 3-axis position_ids and rope_deltas of a chat
               prompt with one image between text (no weights needed)
  forward      logits of the truncated model (--layers) for that prompt, with the
               image or, as the text-only control, without it; same JSON format
               as hf_reference.py so compare.py works on it
  measure-preprocess   how far the reference's own resampling moves between
               PIL.Image.BICUBIC, torchvision's uint8 path and a float path
  measure-tower        the tower in bfloat16 against float32 on the same input

The tower and the processor always come from the raw checkpoint (`--src`); the
text layers come from it too, or from lily's quantized checkpoint dequantized
back to bf16 (`--lily`, as hf_reference.py does). The model is built on the meta
device and only the truncated text layers, the embedding, the head and the
0.9 GB tower are materialised.

    .venv/bin/python tools/reference/hf_vision_reference.py preprocess \
        --src ~/models/Qwen3.8-Flash-Next --image tools/reference/images/333x777.png
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hf_reference import LAYER_PREFIX, LazyNGramEmbedding, LilyWeights, Store, build_state_dict  # noqa: E402
from vision_golden import (  # noqa: E402
    CHECKPOINT_MAX_PIXELS,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    GOLDENS_DIR,
    LARGE_DIR,
    PREPROCESS_TOLERANCE,
    TOWER_TOLERANCE,
    cap_dict,
    compare_values,
    fmt,
    run_meta,
    sha256_file,
    tensor_record,
    write_json,
)

# Token ids from config.json of the checkpoint.
VISION_START = 248053
VISION_END = 248054
IMAGE_PAD = 248056

# The one prompt every positions and forward golden uses: system, then user with
# text, image, text. The leading space in TEXT_AFTER keeps the text-only control
# readable when the image part is dropped.
SYSTEM = "You are a helpful assistant."
TEXT_BEFORE = "Look at this picture."
TEXT_AFTER = " What colour is the top rectangle? Answer in one word."


# --- inputs --------------------------------------------------------------------------


def image_info(path: Path) -> tuple["PIL.Image.Image", dict]:
    from PIL import Image

    image = Image.open(path)
    info = {"file": path.name, "sha256": sha256_file(path), "width": image.width, "height": image.height, "mode": image.mode}
    return image, info


def load_processor(src: Path):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(src)


def messages(with_image: bool) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": TEXT_BEFORE}]
    if with_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": TEXT_AFTER})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def render(processor, with_image: bool) -> str:
    return processor.apply_chat_template(messages(with_image), tokenize=False, add_generation_prompt=True, enable_thinking=False)


def smart_resize_reference(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Transcription of transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize
    (lines 60-84 of the installed file), the formula lily must reproduce."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess(processor, image, min_pixels: int, max_pixels: int):
    """The reference processor on one image with the cap passed as `size`."""
    return processor.image_processor(images=[image], return_tensors="pt", size=cap_dict(min_pixels, max_pixels))


def default_out(kind: str, image_name: str, max_pixels: int, suffix: str = "") -> Path:
    return GOLDENS_DIR / f"hf_vision_{kind}_{image_name}_cap{max_pixels}{suffix}.json"


# --- preprocess ------------------------------------------------------------------------


def cmd_preprocess(args) -> None:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    started = time.time()
    processor = load_processor(args.src)
    ip = processor.image_processor
    image, info = image_info(args.image)
    batch = preprocess(processor, image, args.image_min_pixels, args.image_max_pixels)
    pv = batch["pixel_values"].numpy()
    grid = batch["image_grid_thw"].tolist()
    factor = ip.patch_size * ip.merge_size
    resized = smart_resize(info["height"], info["width"], factor, args.image_min_pixels, args.image_max_pixels)
    ours = smart_resize_reference(info["height"], info["width"], factor, args.image_min_pixels, args.image_max_pixels)
    if ours != resized or (grid[0][1] * ip.patch_size, grid[0][2] * ip.patch_size) != tuple(resized):
        raise SystemExit(f"smart_resize disagreement: transcription {ours}, reference {resized}, grid {grid}")
    tokens = grid[0][0] * grid[0][1] * grid[0][2] // ip.merge_size**2
    name = args.image.stem
    npy = None if args.no_npy else LARGE_DIR / f"preprocess_{name}_cap{args.image_max_pixels}.pixel_values.npy"
    record = {
        "kind": "preprocess",
        "image": info,
        "cap": {"min_pixels": args.image_min_pixels, "max_pixels": args.image_max_pixels, "as_size": cap_dict(args.image_min_pixels, args.image_max_pixels)},
        "processor": {
            "class": f"{type(processor).__name__} / {type(ip).__name__} ({type(ip).__mro__[1].__name__})",
            "declared_in_checkpoint": "Qwen2VLImageProcessorFast",
            "color": "PIL Image.convert('RGB') when mode != RGB (no alpha compositing); tvF.pil_to_tensor -> uint8 CHW",
            "smart_resize": "factor = patch_size * merge_size = 32; see smart_resize_reference() in hf_vision_reference.py",
            "resample": "PILImageResampling.BICUBIC -> torchvision.transforms.v2.functional.resize(uint8 CHW, [h, w], InterpolationMode.BICUBIC, antialias=True): Keys cubic a=-0.5, separable, uint8 clamp+round after each pass (matches PIL Image.BICUBIC within 1 level)",
            "rescale_factor": ip.rescale_factor,
            "image_mean": list(ip.image_mean),
            "image_std": list(ip.image_std),
            "normalize": "float32: (x * rescale - mean) / std, fused as (x - mean/rescale) / (std/rescale)",
            "patch_size": ip.patch_size,
            "merge_size": ip.merge_size,
            "temporal_patch_size": ip.temporal_patch_size,
            "patch_order": "block-major: (grid_h/2, grid_w/2, 2, 2) -> row = ((bh*2 + ih) * grid_w) ... i.e. merge blocks in raster order, 2x2 patches row-major inside each block",
            "row_layout": "channel(3) x temporal(2) x patch_row(16) x patch_col(16) = 1536, channel-major; the two temporal frames are identical copies of the still image",
        },
        "resized_hw": list(resized),
        "image_grid_thw": grid,
        "tokens": tokens,
        "pixel_values": tensor_record(pv, npy, "pixel_values"),
        "tolerance": PREPROCESS_TOLERANCE,
        "meta": run_meta(started, {"transformers": _versions()}),
    }
    out = args.out or default_out("preprocess", name, args.image_max_pixels)
    write_json(out, record)
    print(f"wrote {out}: {info['width']}x{info['height']} -> {resized[1]}x{resized[0]} (w x h), grid {grid[0]}, {tokens} tokens, pixel_values {pv.shape} {pv.dtype}", file=sys.stderr)


def _versions() -> dict:
    import PIL
    import torchvision
    import transformers

    return {"transformers": transformers.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__, "pillow": PIL.__version__}


# --- tower -----------------------------------------------------------------------------


def vision_inv_freq(rotary) -> torch.Tensor:
    """The tower rotary's non-persistent buffer, recomputed on the CPU in float32
    (Qwen4ExpVisionRotaryEmbedding.__init__: 1 / theta ** (arange(0, dim, 2) / dim))."""
    return type(rotary)(rotary.dim, rotary.theta).inv_freq


def build_tower(src: Path, dtype: torch.dtype):
    from transformers import Qwen4ExpConfig, Qwen4ExpVisionModel

    raw_cfg = json.loads((src / "config.json").read_text())
    config = Qwen4ExpConfig(**raw_cfg).vision_config
    config._attn_implementation = "eager"
    store = Store(src)
    with torch.device("meta"):
        tower = Qwen4ExpVisionModel(config)
    state = build_state_dict(tower, store, None, raw_cfg["text_config"], None, rename=lambda n: "model.visual." + n)
    missing, unexpected = tower.load_state_dict(state, strict=False, assign=True)
    bad = [m for m in missing if not m.endswith("inv_freq")]
    if bad or unexpected:
        raise SystemExit(f"tower state dict mismatch: missing={bad[:8]} unexpected={list(unexpected)[:8]}")
    tower = tower.to(dtype)
    # from_pretrained keeps this float32 whatever the weight dtype; so does lily's plan.
    tower.rotary_pos_emb.inv_freq = vision_inv_freq(tower.rotary_pos_emb)
    tower.eval()
    return tower, config


@torch.no_grad()
def run_tower(tower, pixel_values: torch.Tensor, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    started = time.time()
    out = tower(pixel_values, grid_thw=grid)
    return out.pooler_output, out.last_hidden_state, time.time() - started


def cmd_tower(args) -> None:
    started = time.time()
    processor = load_processor(args.src)
    image, info = image_info(args.image)
    batch = preprocess(processor, image, args.image_min_pixels, args.image_max_pixels)
    pv, grid = batch["pixel_values"], batch["image_grid_thw"]
    tower, config = build_tower(args.src, torch.float32)
    built = time.time() - started
    merged, pre, seconds = run_tower(tower, pv, grid)
    name = args.image.stem
    npy_base = LARGE_DIR / f"tower_{name}_cap{args.image_max_pixels}"
    record = {
        "kind": "tower",
        "image": info,
        "cap": {"min_pixels": args.image_min_pixels, "max_pixels": args.image_max_pixels},
        "input": {"pixel_values_sha256": tensor_record(pv.numpy(), None, "pixel_values")["sha256"], "pixel_values_shape": list(pv.shape), "image_grid_thw": grid.tolist()},
        "run": {"dtype": "float32", "device": "cpu", "attn_implementation": "eager", "inv_freq_dtype": str(tower.rotary_pos_emb.inv_freq.dtype), "tower_seconds": round(seconds, 2), "build_seconds": round(built, 2)},
        "tower_config": {k: getattr(config, k) for k in ("depth", "hidden_size", "num_heads", "intermediate_size", "hidden_act", "patch_size", "spatial_merge_size", "temporal_patch_size", "num_position_embeddings", "out_hidden_size", "in_channels")},
        "tokens": int(merged.shape[0]),
        "merged": tensor_record(merged.numpy(), None if args.no_npy else npy_base.with_suffix(".merged.npy"), "merged"),
        "pre_merger": tensor_record(pre.numpy(), None if args.no_npy else npy_base.with_suffix(".pre_merger.npy"), "pre_merger"),
        "tolerance": TOWER_TOLERANCE,
        "meta": run_meta(started, {"transformers": _versions()}),
    }
    out = args.out or default_out("tower", name, args.image_max_pixels)
    write_json(out, record)
    print(f"wrote {out}: {merged.shape[0]} tokens x {merged.shape[1]}, pre-merger {tuple(pre.shape)}, tower {seconds:.1f}s, build {built:.1f}s, peak RSS {record['meta']['peak_rss_gb']} GB", file=sys.stderr)


def cmd_measure_tower(args) -> None:
    """The bf16 tower against the f32 CPU tower on the same input: the floor for a bf16 Metal
    port. The bf16 copy runs on --device (default mps: bf16 eager attention on the CPU takes
    the better part of an hour for an 8 000-patch image; on Metal it takes seconds)."""
    started = time.time()
    processor = load_processor(args.src)
    tower32, _ = build_tower(args.src, torch.float32)
    device = torch.device(args.device)
    tower16 = copy.deepcopy(tower32).to(torch.bfloat16).to(device)
    tower16.rotary_pos_emb.inv_freq = vision_inv_freq(tower16.rotary_pos_emb).to(device)
    results = {}
    for path in args.images:
        image, info = image_info(path)
        batch = preprocess(processor, image, args.image_min_pixels, args.image_max_pixels)
        pv, grid = batch["pixel_values"], batch["image_grid_thw"]
        m32, p32, s32 = run_tower(tower32, pv, grid)
        m16, p16, s16 = run_tower(tower16, pv.to(device), grid.to(device))
        m16, p16 = m16.cpu(), p16.cpu()
        tol = TOWER_TOLERANCE
        merged = compare_values(m16.float().numpy(), m32.numpy(), tol["atol"], tol["rtol"])
        pre = compare_values(p16.float().numpy(), p32.numpy(), tol["atol"], tol["rtol"])
        results[path.stem] = {"tokens": int(m32.shape[0]), "f32_seconds": round(s32, 2), "bf16_seconds": round(s16, 2), "bf16_device": args.device, "merged_bf16_vs_f32": merged, "pre_merger_bf16_vs_f32": pre}
        print(f"{path.stem}: {m32.shape[0]} tokens, f32 {s32:.1f}s, bf16 {s16:.1f}s | merged: max_abs {fmt(merged['max_abs_err'])} max_rel {fmt(merged['max_rel_err'])} rel_l2 {fmt(merged['rel_l2_err'])} cos {merged['cosine']:.6f} within(atol {tol['atol']}, rtol {tol['rtol']}) {merged['fraction_within']:.4f} | pre-merger: max_abs {fmt(pre['max_abs_err'])} rel_l2 {fmt(pre['rel_l2_err'])} cos {pre['cosine']:.6f}", file=sys.stderr)
    _merge_floors(args.out, {"tower_bf16_vs_f32": {"cap": {"min_pixels": args.image_min_pixels, "max_pixels": args.image_max_pixels}, "tolerance_tested": TOWER_TOLERANCE, "images": results, "meta": run_meta(started, {"versions": _versions()})}})


def cmd_measure_preprocess(args) -> None:
    """How much the reference's own pixel_values move between resampling paths."""
    from PIL import Image
    from torchvision.transforms.v2 import functional as tvF
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    started = time.time()
    processor = load_processor(args.src)
    ip = processor.image_processor
    results = {}
    level = 2 / 255  # one uint8 level after normalisation to [-1, 1]

    def m(a, b):
        r = compare_values(a.numpy(), b.numpy(), PREPROCESS_TOLERANCE["atol"], 0.0)
        d = (a - b).abs()
        r["fraction_beyond_1_level"] = float((d > level + 1e-6).float().mean())
        r["fraction_beyond_2_levels"] = float((d > 2 * level + 1e-6).float().mean())
        r["max_levels"] = float(d.max() / level)
        return r

    for path in args.images:
        image, info = image_info(path)
        image = image.convert("RGB")
        ref = preprocess(processor, image, args.image_min_pixels, args.image_max_pixels)["pixel_values"]
        rh, rw = smart_resize(info["height"], info["width"], ip.patch_size * ip.merge_size, args.image_min_pixels, args.image_max_pixels)
        pil = image.resize((rw, rh), Image.BICUBIC)
        pv_pil = ip(images=[pil], return_tensors="pt", do_resize=False)["pixel_values"]
        t = tvF.pil_to_tensor(image)
        f_aa = tvF.resize(t.float(), [rh, rw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
        pv_f_aa = ip(images=[f_aa], return_tensors="pt", do_resize=False)["pixel_values"]
        pv_f_aa_cr = ip(images=[f_aa.clamp(0, 255).round()], return_tensors="pt", do_resize=False)["pixel_values"]
        f_na = tvF.resize(t.float(), [rh, rw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=False)
        pv_f_na = ip(images=[f_na.clamp(0, 255).round()], return_tensors="pt", do_resize=False)["pixel_values"]
        results[path.stem] = {
            "resized_hw": [rh, rw],
            "pil_bicubic_vs_reference": m(pv_pil, ref),
            "float_bicubic_aa_unrounded_vs_reference": m(pv_f_aa, ref),
            "float_bicubic_aa_clamp_round_once_vs_reference": m(pv_f_aa_cr, ref),
            "float_bicubic_no_aa_clamp_round_vs_reference": m(pv_f_na, ref),
            "bf16_roundtrip_of_reference": m(ref.to(torch.bfloat16).float(), ref),
        }
        print(f"{path.stem} -> {rh}x{rw}: PIL max {results[path.stem]['pil_bicubic_vs_reference']['max_levels']:.1f} levels (beyond 1: {results[path.stem]['pil_bicubic_vs_reference']['fraction_beyond_1_level']:.2e}); float AA unrounded max {results[path.stem]['float_bicubic_aa_unrounded_vs_reference']['max_levels']:.1f}; float AA clamp+round once max {results[path.stem]['float_bicubic_aa_clamp_round_once_vs_reference']['max_levels']:.1f}; no-AA max {results[path.stem]['float_bicubic_no_aa_clamp_round_vs_reference']['max_levels']:.1f}; bf16 roundtrip max {results[path.stem]['bf16_roundtrip_of_reference']['max_abs_err']:.5f}", file=sys.stderr)
    _merge_floors(args.out, {"preprocess_resampling": {"cap": {"min_pixels": args.image_min_pixels, "max_pixels": args.image_max_pixels}, "one_level": level, "tolerance_proposed": PREPROCESS_TOLERANCE, "images": results, "meta": run_meta(started, {"versions": _versions()})}})


def _merge_floors(out: Path, update: dict) -> None:
    record = json.loads(out.read_text()) if out.exists() else {}
    record.update(update)
    write_json(out, record)
    print(f"wrote {out}", file=sys.stderr)


# --- positions -------------------------------------------------------------------------


class RopeIndex:
    """`Qwen4ExpModel.get_rope_index` and `get_vision_position_ids` (modeling_qwen4_exp.py
    lines 2005-2148) bound to a config only, so no weights are needed."""

    from transformers import Qwen4ExpModel as _M

    get_vision_position_ids = _M.get_vision_position_ids
    get_rope_index = _M.get_rope_index
    del _M

    def __init__(self, config):
        self.config = config


def encode(processor, image, min_pixels: int, max_pixels: int):
    """The processor on the chat prompt: expanded input_ids, mm_token_type_ids,
    pixel_values, image_grid_thw. `image=None` gives the text-only control."""
    text = render(processor, image is not None)
    if image is None:
        return text, processor(text=[text], return_tensors="pt")
    return text, processor(text=[text], images=[image], return_tensors="pt", size=cap_dict(min_pixels, max_pixels))


def positions_block(config, batch) -> dict:
    ids = batch["input_ids"]
    position_ids, deltas = RopeIndex(config).get_rope_index(
        ids, batch["mm_token_type_ids"], image_grid_thw=batch.get("image_grid_thw"), attention_mask=batch["attention_mask"]
    )
    flat = ids[0].tolist()
    pads = [i for i, t in enumerate(flat) if t == IMAGE_PAD]
    span = {"start": pads[0], "length": len(pads)} if pads else None
    if pads and pads[-1] - pads[0] + 1 != len(pads):
        raise SystemExit("image pads are not contiguous")
    block = {
        "input_ids": flat,
        "mm_token_type_ids": batch["mm_token_type_ids"][0].tolist(),
        "image_grid_thw": batch["image_grid_thw"].tolist() if "image_grid_thw" in batch else None,
        "image_span": span,
        "position_ids": position_ids[:, 0, :].tolist(),
        "rope_deltas": int(deltas[0, 0]),
        "seq_len": len(flat),
        "max_position": int(position_ids.max()),
        "next_position": len(flat) + int(deltas[0, 0]),
        "rule": "text: all three axes = running position; image: t = start, h = start + row, w = start + col over the (grid_h/2, grid_w/2) merged grid in raster order; text after resumes at start + max(grid_h, grid_w)/2; rope_deltas = max_position + 1 - seq_len; every later token gets seq_index + rope_deltas on all axes",
    }
    return block


def cmd_positions(args) -> None:
    from transformers import Qwen4ExpConfig

    started = time.time()
    processor = load_processor(args.src)
    config = Qwen4ExpConfig(**json.loads((args.src / "config.json").read_text()))
    image, info = image_info(args.image)
    text, batch = encode(processor, image, args.image_min_pixels, args.image_max_pixels)
    block = positions_block(config, batch)
    record = {
        "kind": "positions",
        "image": info,
        "cap": {"min_pixels": args.image_min_pixels, "max_pixels": args.image_max_pixels},
        "messages": messages(True),
        "chat_template": {"enable_thinking": False, "add_generation_prompt": True},
        "text": text,
        "token_ids": {"vision_start": VISION_START, "vision_end": VISION_END, "image_pad": IMAGE_PAD},
        "mrope_section": config.text_config.rope_parameters["mrope_section"],
        **block,
        "meta": run_meta(started),
    }
    out = args.out or default_out("positions", args.image.stem, args.image_max_pixels)
    write_json(out, record)
    print(f"wrote {out}: {block['seq_len']} tokens, image span {block['image_span']}, rope_deltas {block['rope_deltas']}, max position {block['max_position']}", file=sys.stderr)


# --- forward ---------------------------------------------------------------------------


def build_cg_model(src: Path, lily_dir: Path | None, layers: int, dtype: torch.dtype):
    """Qwen4ExpForConditionalGeneration truncated to `layers` text layers, on the CPU.
    Tower and processor from `src`; text layers, embedding and head from `lily_dir`
    (dequantized) when given, else from `src`."""
    from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

    raw_cfg = json.loads((src / "config.json").read_text())
    text_cfg = dict(raw_cfg["text_config"])
    total = text_cfg["num_hidden_layers"]
    if not 1 <= layers <= total:
        raise SystemExit(f"--layers must be in 1..{total}")
    text_cfg["num_hidden_layers"] = layers
    text_cfg["layer_types"] = text_cfg["layer_types"][:layers]
    cfg_dict = dict(raw_cfg)
    cfg_dict["text_config"] = text_cfg
    config = Qwen4ExpConfig(**cfg_dict)
    for c in (config, config.text_config, config.vision_config):
        c._attn_implementation = "eager"

    raw_store = Store(src)
    if lily_dir is not None:
        lily_cfg = json.loads((lily_dir / "config.json").read_text())
        if lily_cfg.get("lily", {}).get("format") != "qwen4_exp-affine-v1":
            raise SystemExit("--lily expects a lily qwen4_exp-affine-v1 checkpoint")
        text_store = Store(lily_dir)
        lily = LilyWeights(text_store, lily_cfg)
        ple_consts = lily_cfg["lily"].get("ple")
    else:
        text_store, lily, ple_consts = raw_store, None, None

    with torch.device("meta"):
        model = Qwen4ExpForConditionalGeneration(config)
    lm = model.model.language_model
    ple_layers = [i for i in range(layers) if i + 1 in (text_cfg.get("ple_layer_ids") or [])]
    for i in ple_layers:
        ple = lm.layers[i].ple.ple_embedding
        ple.ngram_embedding = LazyNGramEmbedding(f"{LAYER_PREFIX}{i}.", text_store, lily, text_cfg.get("split_ngram_parts", 512), ple.ngram_embedding.embedding_dim)

    def load(module, state, label):
        missing, unexpected = module.load_state_dict(state, strict=False, assign=True)
        bad = [m for m in missing if not m.endswith("inv_freq") and "ngram_embedding.weight" not in m]
        if bad or unexpected:
            raise SystemExit(f"{label} state dict mismatch: missing={bad[:8]} unexpected={list(unexpected)[:8]}")

    load(lm, build_state_dict(lm, text_store, lily, text_cfg, ple_consts, rename=lambda n: "model.language_model." + n), "language model")
    load(model.lm_head, build_state_dict(model.lm_head, text_store, lily, text_cfg, None, rename=lambda n: "lm_head." + n), "lm_head")
    load(model.model.visual, build_state_dict(model.model.visual, raw_store, None, text_cfg, None, rename=lambda n: "model.visual." + n), "tower")

    # Same order as hf_reference.build_model for the text rotary (so the text-only control
    # reproduces its goldens); the tower rotary is set after the cast, in float32.
    rotary = lm.rotary_emb
    inv_freq, _ = rotary.compute_default_rope_parameters(config.text_config)
    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = inv_freq.clone()
    model = model.to(dtype)
    for i in ple_layers:
        emb = lm.layers[i].ple.ple_embedding.ngram_embedding
        emb.weight.data = emb.weight.data.to(torch.bfloat16)
    model.model.visual.rotary_pos_emb.inv_freq = vision_inv_freq(model.model.visual.rotary_pos_emb)
    model.eval()
    return model, config


@torch.no_grad()
def run_forward(model, batch, last: int, greedy_steps: int) -> dict:
    tokens = batch["input_ids"][0].tolist()
    kwargs = {"input_ids": batch["input_ids"], "use_cache": greedy_steps > 0}
    if "pixel_values" in batch:
        kwargs.update(pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"], mm_token_type_ids=batch["mm_token_type_ids"])
    seen: dict = {}

    def grab(module, args, kw):
        pos = kw.get("position_ids")
        seen["position_ids"] = None if pos is None else pos.detach().clone()

    hook = model.model.language_model.register_forward_pre_hook(grab, with_kwargs=True)
    started = time.time()
    out = model(**kwargs)
    elapsed = time.time() - started
    hook.remove()
    logits = out.logits[0].float()
    argmax = logits.argmax(dim=-1).tolist()
    top = []
    for pos in range(max(0, len(tokens) - last), len(tokens)):
        values, idx = logits[pos].topk(8)
        top.append({"position": pos, "ids": idx.tolist(), "logits": [round(v, 4) for v in values.tolist()]})
    result = {"prompt_token_ids": tokens, "argmax": argmax, "top8_last": top, "prefill_seconds": round(elapsed, 2)}
    result["language_model_position_ids"] = None if seen.get("position_ids") is None else seen["position_ids"][:, 0, :].tolist()
    result["model_rope_deltas"] = None if model.model.rope_deltas is None else int(model.model.rope_deltas[0, 0])
    if greedy_steps > 0:
        past = out.past_key_values
        chosen = [argmax[-1]]
        for _ in range(greedy_steps - 1):
            step = model(input_ids=torch.tensor([[chosen[-1]]]), past_key_values=past, use_cache=True)
            past = step.past_key_values
            chosen.append(int(step.logits[0, -1].argmax()))
        result["greedy"] = chosen
    return result


def cmd_forward(args) -> None:
    started = time.time()
    processor = load_processor(args.src)
    with_image = not args.no_image
    image, info = image_info(args.image) if with_image else (None, None)
    text, batch = encode(processor, image, args.image_min_pixels, args.image_max_pixels)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, config = build_cg_model(args.src, args.lily, args.layers, dtype)
    built = time.time() - started
    print(f"model built in {built:.1f}s: {args.layers} layers, {batch['input_ids'].shape[1]} prompt tokens, image={with_image}", file=sys.stderr)
    result = run_forward(model, batch, args.last, args.greedy)
    positions = positions_block(config, batch)
    if with_image:
        if result["language_model_position_ids"] != positions["position_ids"]:
            raise SystemExit("position ids seen by the language model differ from get_rope_index")
        if result["model_rope_deltas"] != positions["rope_deltas"]:
            raise SystemExit("rope_deltas kept by the model differ from get_rope_index")
        result["positions"] = positions
        result["positions_match_language_model"] = True
    else:
        # Text only: the model builds plain arange positions itself (no rope index).
        result["positions"] = None
    result["meta"] = {
        "source": "lily-dequant" if args.lily else "hf-bf16",
        "directory": str(args.lily or args.src),
        "tower_directory": str(args.src),
        "layers": args.layers,
        "dtype": args.dtype,
        "budget": config.text_config.indexer_budget,
        "dense_limit": config.text_config.indexer_budget + config.text_config.indexer_compress_ratio - 1,
        "image": info,
        "cap": {"min_pixels": args.image_min_pixels, "max_pixels": args.image_max_pixels} if with_image else None,
        "messages": messages(with_image),
        "text": text,
        "text_rotary_inv_freq_dtype": str(model.model.language_model.rotary_emb.inv_freq.dtype),
        "tower_rotary_inv_freq_dtype": str(model.model.visual.rotary_pos_emb.inv_freq.dtype),
        "build_seconds": round(built, 2),
        **run_meta(started, {"transformers": _versions()}),
    }
    result["kind"] = "forward"
    out = args.out
    if out is None:
        flavour = "dequant" if args.lily else "bf16"
        out = GOLDENS_DIR / (f"hf_l{args.layers}_vision_{args.image.stem}_{flavour}.json" if with_image else f"hf_l{args.layers}_vision_textonly_{flavour}.json")
    write_json(out, result)
    if args.tokens_out:
        Path(args.tokens_out).write_text(json.dumps(result["prompt_token_ids"]) + "\n")
        print(f"wrote {args.tokens_out}", file=sys.stderr)
    print(f"wrote {out}: argmax tail {result['argmax'][-8:]}, greedy {result.get('greedy')}, prefill {result['prefill_seconds']}s, peak RSS {result['meta']['peak_rss_gb']} GB", file=sys.stderr)


# --- cli -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, image=True, multi=False):
        p.add_argument("--src", type=lambda s: Path(s).expanduser(), required=True, help="raw HF checkpoint directory (tower, processor, tokenizer)")
        p.add_argument("--image-max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help=f"pixel cap passed as size.longest_edge (default {DEFAULT_MAX_PIXELS}; the checkpoint's own is {CHECKPOINT_MAX_PIXELS})")
        p.add_argument("--image-min-pixels", type=int, default=DEFAULT_MIN_PIXELS, help=f"size.shortest_edge (default {DEFAULT_MIN_PIXELS})")
        if multi:
            p.add_argument("images", type=Path, nargs="+")
            p.add_argument("--out", type=Path, default=GOLDENS_DIR / "vision_tolerance_floors.json")
        else:
            if image:
                p.add_argument("--image", type=Path, required=True)
            p.add_argument("--out", type=Path, help="golden path (default derived from image and cap)")

    p = sub.add_parser("preprocess", help="pixel_values and image_grid_thw")
    common(p)
    p.add_argument("--no-npy", action="store_true", help="do not write the full tensor to goldens/large/")
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("tower", help="Qwen4ExpVisionModel output in float32 on the CPU")
    common(p)
    p.add_argument("--no-npy", action="store_true")
    p.set_defaults(func=cmd_tower)

    p = sub.add_parser("positions", help="input_ids, 3-axis position_ids and rope_deltas for the chat prompt")
    common(p)
    p.set_defaults(func=cmd_positions)

    p = sub.add_parser("forward", help="truncated-model logits for the chat prompt, with or without the image")
    common(p)
    p.add_argument("--lily", type=lambda s: Path(s).expanduser(), help="lily quantized checkpoint for the text layers (dequantized)")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--no-image", action="store_true", help="text-only control: same prompt without the image part")
    p.add_argument("--last", type=int, default=16)
    p.add_argument("--greedy", type=int, default=0)
    p.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    p.add_argument("--tokens-out", help="also write the prompt token ids as a JSON list (for hf_reference.py --tokens)")
    p.set_defaults(func=cmd_forward)

    p = sub.add_parser("measure-preprocess", help="resampling paths against the reference processor")
    common(p, multi=True)
    p.set_defaults(func=cmd_measure_preprocess)

    p = sub.add_parser("measure-tower", help="bf16 tower against f32 tower")
    common(p, multi=True)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu", help="device for the bf16 copy (the f32 reference stays on the CPU)")
    p.set_defaults(func=cmd_measure_tower)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
