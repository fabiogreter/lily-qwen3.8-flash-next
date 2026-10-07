//! `config.json` parsing, on the four-layer conversion's config with the
//! vision tower (`tests/goldens/qwen4exp_config_l4_vision.json`, the file the
//! converter's `--vision-only` wrote) and edits of it.

use serde_json::{Value, json};

use super::*;

const FIXTURE: &str = include_str!("../../goldens/qwen4exp_config_l4_vision.json");

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("fixture json")
}

fn parse(v: &Value) -> Result<Qwen4ExpConfig> {
    Qwen4ExpConfig::from_json(&serde_json::to_vec(v).unwrap())
}

fn error_of(v: &Value) -> String {
    let Err(err) = parse(v) else { panic!("expected the config to be rejected") };
    format!("{err:#}")
}

#[test]
fn the_tower_carrying_config_parses_to_a_vision_config() {
    let cfg = parse(&fixture()).expect("fixture parses");
    let v = cfg.vision.as_ref().expect("lily.vision is set");
    assert_eq!(
        *v,
        VisionConfig {
            depth: 27,
            hidden_size: 1152,
            num_heads: 16,
            intermediate_size: 4304,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            num_position_embeddings: 2304,
            out_hidden_size: 2560,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            deepstack_visual_indexes: vec![],
            image_token_id: 248056,
            video_token_id: 248057,
            vision_start_token_id: 248053,
            vision_end_token_id: 248054,
            tensors: 333,
        }
    );
    assert_eq!(v.head_dim(), 72);
    assert_eq!(v.patch_dim(), 3 * 2 * 16 * 16);
    assert_eq!(v.merge_dim(), 4 * 1152);
    assert_eq!(v.expected_tensors(), 333);
    assert_eq!(cfg.rope_parameters.mrope_section.as_deref(), Some(MROPE_SECTION));
    assert_eq!(cfg.rope_parameters.mrope_interleaved, Some(true));
    // The text side is untouched by the tower.
    assert_eq!(cfg.num_hidden_layers, 4);
    assert!(cfg.mtp.is_some());
}

#[test]
fn a_config_without_the_lily_vision_block_has_no_tower() {
    // The pre-item-2 shape: vision_config still present, no lily.vision.
    let mut v = fixture();
    v["lily"]["vision"] = Value::Null;
    let cfg = parse(&v).expect("parses");
    assert!(cfg.vision.is_none());

    // The converter's --no-vision output: tower keys stripped, prefix dropped.
    let mut v = fixture();
    v["lily"]["vision"] = Value::Null;
    v["lily"]["dropped"] = json!(["model.visual."]);
    let obj = v.as_object_mut().unwrap();
    for key in [
        "vision_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ] {
        obj.remove(key);
    }
    let cfg = parse(&v).expect("parses");
    assert!(cfg.vision.is_none());
    // The mrope keys are still read for text-only configs.
    assert_eq!(cfg.rope_parameters.mrope_section.as_deref(), Some(MROPE_SECTION));
}

#[test]
fn the_top_level_quantization_block_is_ignored() {
    // The converter writes an MLX-style block for the Hugging Face Hub's
    // parameter count; only lily.quantization drives the loader.
    let mut v = fixture();
    v["quantization"] = json!({"group_size": 32, "bits": 8, "mode": "affine"});
    let cfg = parse(&v).expect("parses");
    assert_eq!((cfg.quantization.bits, cfg.quantization.group_size), (4, 64));
}

#[test]
fn a_config_without_the_q8_flags_keeps_the_q4_storage_policy() {
    // Every checkpoint converted before `--q8-dense` existed.
    let cfg = parse(&fixture()).expect("fixture parses");
    assert_eq!(cfg.storage, StoragePolicy::Q4);
}

#[test]
fn the_q8_flags_select_8_bit_dense_and_embedding_storage() {
    use Q8Group::*;
    let mut v = fixture();
    v["lily"]["quantization"]["q8_dense"] = json!(true);
    let cfg = parse(&v).expect("parses");
    assert_eq!(cfg.storage, StoragePolicy::q8(&[Attn, Gdn, Shared, Head]));
    assert_eq!(cfg.storage.bits(Embed), 4);
    v["lily"]["quantization"]["q8_embed"] = json!(true);
    let cfg = parse(&v).expect("parses");
    assert_eq!(cfg.storage, StoragePolicy::q8(&Q8Group::ALL));
    // The default width of the rest (the experts) is unchanged.
    assert_eq!((cfg.quantization.bits, cfg.quantization.group_size), (4, 64));

    let mut v = fixture();
    v["lily"]["quantization"]["q8_embed"] = json!(true);
    assert_eq!(parse(&v).expect("parses").storage, StoragePolicy::q8(&[Embed]));
}

#[test]
fn q8_groups_selects_any_subset() {
    use Q8Group::*;
    let mut v = fixture();
    v["lily"]["quantization"]["q8_groups"] = json!(["attn", "head", "embed"]);
    let s = parse(&v).expect("parses").storage;
    assert_eq!(s, StoragePolicy::q8(&[Attn, Head, Embed]));
    assert_eq!(s.q8_groups(), [Attn, Head, Embed]);
    assert_eq!(
        Q8Group::ALL.map(|g| s.bits(g)),
        [8, 4, 4, 8, 8],
        "attn gdn shared head embed"
    );
    // Order in the list does not matter; the set does.
    v["lily"]["quantization"]["q8_groups"] = json!(["embed", "attn", "head"]);
    assert_eq!(parse(&v).expect("parses").storage, s);
    v["lily"]["quantization"]["q8_groups"] = json!(["gdn"]);
    assert_eq!(parse(&v).expect("parses").storage, StoragePolicy::q8(&[Gdn]));
}

#[test]
fn q8_groups_rejects_unknown_repeated_or_doubly_specified_groups() {
    let mut v = fixture();
    v["lily"]["quantization"]["q8_groups"] = json!(["attn", "experts"]);
    assert!(error_of(&v).contains("unknown group \"experts\""), "{}", error_of(&v));
    v["lily"]["quantization"]["q8_groups"] = json!(["head", "head"]);
    assert!(error_of(&v).contains("twice"), "{}", error_of(&v));
    v["lily"]["quantization"]["q8_groups"] = json!(["head"]);
    v["lily"]["quantization"]["q8_embed"] = json!(true);
    assert!(
        error_of(&v).contains("both q8_groups and q8_dense/q8_embed"),
        "{}",
        error_of(&v)
    );
}

#[test]
fn a_config_that_declares_and_drops_the_tower_is_rejected() {
    let mut v = fixture();
    v["lily"]["dropped"] = json!(["model.visual."]);
    assert!(error_of(&v).contains("lily.dropped"));
}

#[test]
fn a_declared_tower_needs_vision_config_and_the_token_ids() {
    let mut v = fixture();
    v.as_object_mut().unwrap().remove("vision_config");
    assert!(error_of(&v).contains("vision_config"));

    let mut v = fixture();
    v.as_object_mut().unwrap().remove("vision_start_token_id");
    assert!(error_of(&v).contains("vision_start_token_id"));
}

#[test]
fn unsupported_tower_shapes_are_rejected_by_field_name() {
    let mut v = fixture();
    v["vision_config"]["deepstack_visual_indexes"] = json!([8, 16, 24]);
    assert!(error_of(&v).contains("deepstack_visual_indexes"));

    let mut v = fixture();
    v["vision_config"]["hidden_act"] = json!("gelu");
    assert!(error_of(&v).contains("hidden_act"));

    let mut v = fixture();
    v["vision_config"]["out_hidden_size"] = json!(4096);
    assert!(error_of(&v).contains("out_hidden_size"));

    let mut v = fixture();
    v["vision_config"]["spatial_merge_size"] = json!(1);
    assert!(error_of(&v).contains("spatial_merge_size"));

    let mut v = fixture();
    v["vision_config"]["temporal_patch_size"] = json!(1);
    assert!(error_of(&v).contains("temporal_patch_size"));

    let mut v = fixture();
    v["lily"]["vision"]["tensors"] = json!(300);
    assert!(error_of(&v).contains("lily.vision.tensors"));

    let mut v = fixture();
    v["lily"]["vision"]["dtype"] = json!("q4");
    assert!(error_of(&v).contains("dtype"));
}

#[test]
fn the_mrope_layout_is_checked_when_the_tower_is_present() {
    let mut v = fixture();
    v["text_config"]["rope_parameters"]["mrope_section"] = json!([10, 11, 11]);
    assert!(error_of(&v).contains("mrope_section"));

    let mut v = fixture();
    v["text_config"]["rope_parameters"]["mrope_interleaved"] = json!(false);
    assert!(error_of(&v).contains("mrope_interleaved"));

    let mut v = fixture();
    let rope = v["text_config"]["rope_parameters"].as_object_mut().unwrap();
    rope.remove("mrope_section");
    rope.remove("mrope_interleaved");
    assert!(error_of(&v).contains("mrope_section"));

    // Without a tower the keys may be absent (older text-only conversions).
    v["lily"]["vision"] = Value::Null;
    let cfg = parse(&v).expect("text-only config without mrope keys parses");
    assert!(cfg.vision.is_none());
    assert_eq!(cfg.rope_parameters.mrope_section, None);
}
