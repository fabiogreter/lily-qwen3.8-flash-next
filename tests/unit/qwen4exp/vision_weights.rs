//! Vision tower loading: a tiny synthetic checkpoint checks the shape
//! contract and the conv flattening; the four-layer conversion with the
//! tower (`LILY_MODEL_DIR_FLASH`) checks the real tensors.

use std::io::Write as _;

use half::bf16;

use super::*;
use crate::config::QuantizationConfig;
use crate::metal::MetalContext;
use crate::qwen4exp::config::Qwen4ExpConfig;
use crate::safetensors::Checkpoint;

fn tiny_config() -> VisionConfig {
    VisionConfig {
        depth: 2,
        hidden_size: 8,
        num_heads: 2,
        intermediate_size: 16,
        patch_size: 2,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        in_channels: 3,
        num_position_embeddings: 4,
        out_hidden_size: 16,
        hidden_act: "gelu_pytorch_tanh".to_string(),
        deepstack_visual_indexes: vec![],
        image_token_id: 1,
        video_token_id: 2,
        vision_start_token_id: 3,
        vision_end_token_id: 4,
        tensors: 3 + 12 * 2 + 6,
    }
}

/// Every tensor the tower has, with its shape, in checkpoint naming.
fn tower_shapes(v: &VisionConfig) -> Vec<(String, Vec<usize>)> {
    let h = v.hidden_size;
    let m = v.merge_dim();
    let mut out = vec![
        (
            "patch_embed.proj.weight".to_string(),
            vec![h, v.in_channels, v.temporal_patch_size, v.patch_size, v.patch_size],
        ),
        ("patch_embed.proj.bias".to_string(), vec![h]),
        ("pos_embed.weight".to_string(), vec![v.num_position_embeddings, h]),
    ];
    for i in 0..v.depth {
        for (suffix, shape) in [
            ("norm1.weight", vec![h]),
            ("norm1.bias", vec![h]),
            ("attn.qkv.weight", vec![3 * h, h]),
            ("attn.qkv.bias", vec![3 * h]),
            ("attn.proj.weight", vec![h, h]),
            ("attn.proj.bias", vec![h]),
            ("norm2.weight", vec![h]),
            ("norm2.bias", vec![h]),
            ("mlp.linear_fc1.weight", vec![v.intermediate_size, h]),
            ("mlp.linear_fc1.bias", vec![v.intermediate_size]),
            ("mlp.linear_fc2.weight", vec![h, v.intermediate_size]),
            ("mlp.linear_fc2.bias", vec![h]),
        ] {
            out.push((format!("blocks.{i}.{suffix}"), shape));
        }
    }
    out.extend([
        ("merger.norm.weight".to_string(), vec![h]),
        ("merger.norm.bias".to_string(), vec![h]),
        ("merger.linear_fc1.weight".to_string(), vec![m, m]),
        ("merger.linear_fc1.bias".to_string(), vec![m]),
        ("merger.linear_fc2.weight".to_string(), vec![v.out_hidden_size, m]),
        ("merger.linear_fc2.bias".to_string(), vec![v.out_hidden_size]),
    ]);
    out
}

/// Writes a one-shard checkpoint holding `tensors` (bf16 values `i * 0.5`
/// in element order) and returns its directory.
fn write_checkpoint(tag: &str, tensors: &[(String, Vec<usize>)]) -> std::path::PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("lily-vision-weights-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut weight_map = serde_json::Map::new();
    for (name, shape) in tensors {
        let full = format!("{VISION_PREFIX}{name}");
        let numel: usize = shape.iter().product();
        let bytes: Vec<u8> = (0..numel)
            .flat_map(|i| bf16::from_f32(i as f32 * 0.5).to_le_bytes())
            .collect();
        header.insert(
            full.clone(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [data.len(), data.len() + bytes.len()],
            }),
        );
        data.extend_from_slice(&bytes);
        weight_map.insert(full, "vision-00001-of-00001.safetensors".into());
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut f = std::fs::File::create(dir.join("vision-00001-of-00001.safetensors"))
        .expect("shard");
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&header).unwrap();
    f.write_all(&data).unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .expect("index");
    dir
}

fn loader<'a>(ctx: &'a MetalContext, dir: &std::path::Path) -> Loader<'a> {
    Loader::new(
        ctx,
        Checkpoint::open(dir).expect("checkpoint"),
        QuantizationConfig { group_size: 64, bits: 4 },
        &[],
        |_| 4,
    )
}

#[test]
fn a_synthetic_tower_loads_with_every_shape_checked() {
    let ctx = MetalContext::new().expect("metal");
    let v = tiny_config();
    let shapes = tower_shapes(&v);
    assert_eq!(shapes.len(), v.expected_tensors());
    let dir = write_checkpoint("ok", &shapes);
    let l = loader(&ctx, &dir);
    let w = load(&l, &v).expect("tower loads");
    // Nothing under the prefix is left unread.
    l.finish().expect("every tensor consumed");

    assert_eq!(w.blocks.len(), v.depth);
    assert_eq!(w.patch_proj_w.shape(), &[v.hidden_size, v.patch_dim()]);
    assert_eq!(w.patch_proj_b.shape(), &[v.hidden_size]);
    assert_eq!(w.pos_embed.shape(), &[v.num_position_embeddings, v.hidden_size]);
    assert_eq!(w.blocks[1].qkv_w.shape(), &[3 * v.hidden_size, v.hidden_size]);
    assert_eq!(w.blocks[1].fc1_w.shape(), &[v.intermediate_size, v.hidden_size]);
    assert_eq!(w.blocks[1].fc2_w.shape(), &[v.hidden_size, v.intermediate_size]);
    assert_eq!(w.merger_fc1_w.shape(), &[v.merge_dim(), v.merge_dim()]);
    assert_eq!(w.merger_fc2_w.shape(), &[v.out_hidden_size, v.merge_dim()]);
    assert_eq!(w.merger_fc2_b.shape(), &[v.out_hidden_size]);
    let total: usize =
        shapes.iter().map(|(_, s)| 2 * s.iter().product::<usize>()).sum();
    assert_eq!(w.bytes(), total);

    // The conv weight's flattening is the checkpoint's own element order:
    // element (o, c, t, y, x) lands at row o, column ((c*T + t)*P + y)*P + x.
    let flat = w.patch_proj_w.to_f32().expect("read back");
    let (c_n, t_n, p) = (v.in_channels, v.temporal_patch_size, v.patch_size);
    for (o, c, t, y, x) in [(0, 0, 0, 0, 0), (1, 2, 1, 1, 0), (7, 1, 0, 1, 1)] {
        let src = (((o * c_n + c) * t_n + t) * p + y) * p + x;
        let col = ((c * t_n + t) * p + y) * p + x;
        assert_eq!(flat[o * v.patch_dim() + col], src as f32 * 0.5);
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_tensor_whose_shape_disagrees_with_the_config_is_named() {
    let ctx = MetalContext::new().expect("metal");
    let v = tiny_config();
    let mut shapes = tower_shapes(&v);
    let pos = shapes.iter().position(|(n, _)| n == "blocks.1.mlp.linear_fc1.weight");
    shapes[pos.unwrap()].1 = vec![v.intermediate_size + 1, v.hidden_size];
    let dir = write_checkpoint("badshape", &shapes);
    let Err(err) = load(&loader(&ctx, &dir), &v) else { panic!("rejected") };
    let err = format!("{err:#}");
    assert!(err.contains("model.visual.blocks.1.mlp.linear_fc1.weight"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_missing_tensor_is_named() {
    let ctx = MetalContext::new().expect("metal");
    let v = tiny_config();
    let mut shapes = tower_shapes(&v);
    shapes.retain(|(n, _)| n != "merger.linear_fc2.bias");
    let dir = write_checkpoint("missing", &shapes);
    let Err(err) = load(&loader(&ctx, &dir), &v) else { panic!("rejected") };
    let err = format!("{err:#}");
    assert!(err.contains("model.visual.merger.linear_fc2.bias"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}

/// The real tower from the four-layer conversion. The reference values are
/// bf16 bit patterns printed once with Python from the source checkpoint's
/// `model-00001-of-00131.safetensors` (`np.frombuffer(raw, np.uint16)`), so
/// they are the source's bytes, not lily's.
fn bf16s(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| bf16::from_bits(b).to_f32()).collect()
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH with the vision tower appended"]
fn the_four_layer_checkpoint_tower_matches_the_source() {
    let Ok(dir) = std::env::var("LILY_MODEL_DIR_FLASH") else { return };
    let config = Qwen4ExpConfig::from_model_dir(&dir).expect("config");
    let v = config.vision.as_ref().expect("the checkpoint declares the tower");
    let ctx = MetalContext::new().expect("metal");
    let l = loader(&ctx, std::path::Path::new(&dir));
    let w = load(&l, v).expect("tower loads");

    assert_eq!(w.blocks.len(), 27);
    assert_eq!(w.bytes(), 897_862_112);
    for (name, shape) in tower_shapes(v) {
        let want =
            if name == "patch_embed.proj.weight" { vec![1152, 1536] } else { shape };
        let full = format!("{VISION_PREFIX}{name}");
        let meta = l.checkpoint().meta(&full).expect("in checkpoint");
        if name != "patch_embed.proj.weight" {
            assert_eq!(meta.shape, want, "{full}");
        }
        assert_eq!(meta.dtype, crate::safetensors::SafetensorsDType::BF16, "{full}");
    }
    assert_eq!(w.patch_proj_w.shape(), &[1152, 1536]);
    assert_eq!(w.merger_fc2_w.shape(), &[2560, 4608]);

    // The flattened conv weight is the checkpoint's bytes unchanged.
    let raw = l
        .checkpoint()
        .read("model.visual.patch_embed.proj.weight")
        .expect("raw conv weight");
    assert_eq!(w.patch_proj_w.raw_bytes(), &raw[..]);

    let conv = w.patch_proj_w.to_f32().expect("conv");
    assert_eq!(conv.len(), 1152 * 1536);
    assert_eq!(conv[..4], bf16s(&[0x3c05, 0xbbe5, 0xbb9a, 0xbc43]));
    assert_eq!(conv[conv.len() - 4..], bf16s(&[0x3c8d, 0xbb7f, 0xbc1a, 0x3b7d]));
    // w[0, c=1, t=0, y=0, x=0] and w[5, c=2, t=1, y=15, x=15] in (C, T, H, W).
    assert_eq!(conv[512], bf16::from_bits(0x3c15).to_f32());
    assert_eq!(conv[5 * 1536 + 1535], bf16::from_bits(0x3bc7).to_f32());

    let bias = w.patch_proj_b.to_f32().expect("bias");
    assert_eq!(bias[..4], bf16s(&[0x3dde, 0xbeaf, 0x39f7, 0xbe27]));
    let fc2 = w.merger_fc2_w.to_f32().expect("merger fc2");
    assert_eq!(fc2[..4], bf16s(&[0xbaad, 0xbb67, 0x3bde, 0x3b95]));
}
