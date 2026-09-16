use std::io::Write as _;

use super::*;

/// The Hugging Face `Qwen4ExpTextNGramEmbedding` hashing, written out plainly.
fn reference_ids(
    tokens: &[u32],
    hist: [u32; 2],
    mult: &[u64],
    sizes: &[u64],
    offsets: &[u64],
    hpn: usize,
    eos: u32,
) -> Vec<u32> {
    let heads = sizes.len();
    let mut out = Vec::with_capacity(tokens.len() * heads);
    for r in 0..tokens.len() {
        let t0 = tokens[r];
        let p1 = if r >= 1 { tokens[r - 1] } else { hist[1] };
        let p2 = if r >= 2 {
            tokens[r - 2]
        } else if r == 1 {
            hist[1]
        } else {
            hist[0]
        };
        let s1 = if p1 == eos { eos } else { p1 };
        let s2 = if p1 == eos || p2 == eos { eos } else { p2 };
        for j in 0..heads {
            let mut mixed = (t0 as u64 * mult[0]) ^ (s1 as u64 * mult[1]);
            if j / hpn >= 1 {
                mixed ^= s2 as u64 * mult[2];
            }
            out.push((mixed % sizes[j]) as u32 + offsets[j] as u32);
        }
    }
    out
}

#[test]
fn hasher_matches_reference_and_respects_eos_segments() {
    let mult = [23_703_573_157_769u64, 20_109_073_645_365, 8_052_911_324_071];
    let sizes: Vec<u64> = (0..16).map(|j| 20_000_003 + 10 * j as u64).collect();
    let offsets: Vec<u64> = (0..16).map(|j| j as u64 * 20_000_100).collect();
    let eos = 248_044u32;
    let hasher = NgramHasher::new(&mult, &sizes, &offsets, 8, eos).expect("hasher");
    let tokens = vec![17u32, 248_319, eos, 5, 6, eos, eos, 9];
    let hist = [eos, 42];
    let mut got = Vec::new();
    hasher.ids(&tokens, hist, &mut got);
    assert_eq!(got, reference_ids(&tokens, hist, &mult, &sizes, &offsets, 8, eos));

    // Feeding token by token with the advancing history gives the same ids.
    let mut h = hist;
    let mut stepwise = Vec::new();
    for &t in &tokens {
        hasher.ids(&[t], h, &mut stepwise);
        h = NgramHasher::advance(h, &[t]);
    }
    assert_eq!(stepwise, got);
    assert_eq!(NgramHasher::advance(hist, &tokens), [eos, 9]);
    assert_eq!(NgramHasher::advance([eos, 9], &[77]), [9, 77]);
    assert_eq!(NgramHasher::advance(hist, &[]), hist);
}

/// Writes a two-shard checkpoint holding a tiny table split across two files
/// (the second shard shares its file with an unrelated tensor) and checks
/// that paged gathers return the same bytes as the source rows.
#[test]
fn paged_table_gathers_rows_from_shard_files() {
    let dir =
        std::env::temp_dir().join(format!("lily-ngram-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let (rows0, rows1, words, groups) = (7usize, 5usize, 20usize, 5usize);
    let base = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
    let codes0: Vec<u32> = (0..rows0 * words).map(|i| i as u32 * 7 + 1).collect();
    let codes1: Vec<u32> = (0..rows1 * words).map(|i| 1_000_000 + i as u32).collect();
    let scales0: Vec<u16> = (0..rows0 * groups).map(|i| 0x3f00 + i as u16).collect();
    let scales1: Vec<u16> = (0..rows1 * groups).map(|i| 0x3e00 + i as u16).collect();
    let biases0: Vec<u16> = (0..rows0 * groups).map(|i| 0xbf00 + i as u16).collect();
    let biases1: Vec<u16> = (0..rows1 * groups).map(|i| 0xbe00 + i as u16).collect();

    let write_shard = |name: &str, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]| {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (tname, dtype, shape, bytes) in tensors {
            header.insert(
                tname.to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes.len()],
                }),
            );
            offset += bytes.len();
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = std::fs::File::create(dir.join(name)).expect("shard file");
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        for (_, _, _, bytes) in tensors {
            f.write_all(bytes).unwrap();
        }
    };
    let u32s = |v: &[u32]| bytemuck::cast_slice::<u32, u8>(v).to_vec();
    let u16s = |v: &[u16]| bytemuck::cast_slice::<u16, u8>(v).to_vec();
    write_shard(
        "model-00001-of-00002.safetensors",
        &[
            (
                &format!("{base}.shard_0.weight"),
                "U32",
                vec![rows0, words],
                u32s(&codes0),
            ),
            (
                &format!("{base}.shard_0.scales"),
                "BF16",
                vec![rows0, groups],
                u16s(&scales0),
            ),
            (
                &format!("{base}.shard_0.biases"),
                "BF16",
                vec![rows0, groups],
                u16s(&biases0),
            ),
        ],
    );
    write_shard(
        "model-00002-of-00002.safetensors",
        &[
            ("other.weight", "U32", vec![3], u32s(&[9, 9, 9])),
            (
                &format!("{base}.shard_1.weight"),
                "U32",
                vec![rows1, words],
                u32s(&codes1),
            ),
            (
                &format!("{base}.shard_1.scales"),
                "BF16",
                vec![rows1, groups],
                u16s(&scales1),
            ),
            (
                &format!("{base}.shard_1.biases"),
                "BF16",
                vec![rows1, groups],
                u16s(&biases1),
            ),
        ],
    );
    let index = serde_json::json!({"weight_map": {
        format!("{base}.shard_0.weight"): "model-00001-of-00002.safetensors",
        format!("{base}.shard_0.scales"): "model-00001-of-00002.safetensors",
        format!("{base}.shard_0.biases"): "model-00001-of-00002.safetensors",
        "other.weight": "model-00002-of-00002.safetensors",
        format!("{base}.shard_1.weight"): "model-00002-of-00002.safetensors",
        format!("{base}.shard_1.scales"): "model-00002-of-00002.safetensors",
        format!("{base}.shard_1.biases"): "model-00002-of-00002.safetensors",
    }});
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .expect("index");

    let ckpt = Checkpoint::open(&dir).expect("checkpoint");
    let bases = shard_bases("model.language_model.layers.1.ple.", 2);
    let table = PagedTable::open(&ckpt, &bases, 32).expect("paged table");
    assert_eq!(table.rows(), rows0 + rows1);
    assert_eq!(table.width(), 160);

    // Rows straddling both shards, out of order, with a repeat.
    let ids = [0u32, 11, 6, 7, 3, 7];
    let n = ids.len();
    let mut codes = vec![0u8; n * words * 4];
    let mut scales = vec![0u8; n * groups * 2];
    let mut biases = vec![0u8; n * groups * 2];
    table.gather(&ids, &mut codes, &mut scales, &mut biases).expect("gather");
    for (i, &id) in ids.iter().enumerate() {
        let (want_codes, want_scales, want_biases): (&[u32], &[u16], &[u16]) =
            if (id as usize) < rows0 {
                let r = id as usize;
                (
                    &codes0[r * words..(r + 1) * words],
                    &scales0[r * groups..(r + 1) * groups],
                    &biases0[r * groups..(r + 1) * groups],
                )
            } else {
                let r = id as usize - rows0;
                (
                    &codes1[r * words..(r + 1) * words],
                    &scales1[r * groups..(r + 1) * groups],
                    &biases1[r * groups..(r + 1) * groups],
                )
            };
        assert_eq!(
            &codes[i * words * 4..(i + 1) * words * 4],
            bytemuck::cast_slice::<u32, u8>(want_codes),
            "codes of row {id}"
        );
        assert_eq!(
            &scales[i * groups * 2..(i + 1) * groups * 2],
            bytemuck::cast_slice::<u16, u8>(want_scales)
        );
        assert_eq!(
            &biases[i * groups * 2..(i + 1) * groups * 2],
            bytemuck::cast_slice::<u16, u8>(want_biases)
        );
    }
    assert!(
        table
            .gather(
                &[12],
                &mut codes[..words * 4],
                &mut scales[..groups * 2],
                &mut biases[..groups * 2]
            )
            .is_err()
    );
    assert!(table.preload(false).expect("preload") >= table.bytes());
    assert_eq!(table.bytes() as usize, (rows0 + rows1) * (words * 4 + groups * 4));

    // The same rows through the staging buffers on the GPU side.
    let ctx = MetalContext::new().expect("metal");
    let stage = StagedRows::new(&ctx, &table, 8).expect("staging");
    stage.fill(&table, &ids).expect("fill");
    let staged = stage.as_table(n).expect("view");
    assert_eq!(staged.out_features(), n);
    assert_eq!(staged.in_features(), 160);
    let staged_codes = staged.codes.to_u32().expect("codes");
    assert_eq!(&staged_codes[..words], &codes0[..words]);
    assert_eq!(
        &staged_codes[words..2 * words],
        &codes1[(11 - rows0) * words..(12 - rows0) * words]
    );
    assert_eq!(
        stage.seq_ids(n).expect("ids").to_u32().expect("ids"),
        (0..n as u32).collect::<Vec<_>>()
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Host cost of staging one token's rows from a real checkpoint; run with
/// `LILY_MODEL_DIR_FLASH=<ckpt> cargo test --release --lib paged_gather_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing only; needs LILY_MODEL_DIR_FLASH"]
fn paged_gather_timing() {
    let Ok(dir) = std::env::var("LILY_MODEL_DIR_FLASH") else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint");
    let bases = shard_bases("model.language_model.layers.1.ple.", 128);
    let table = PagedTable::open(&ckpt, &bases, 32).expect("paged table");
    eprintln!(
        "resident before: {:.2} GB of {:.2} GB",
        table.resident_bytes().unwrap() as f64 / 1e9,
        table.bytes() as f64 / 1e9
    );
    let n = 16;
    let (cb, gb) = (table.codes_bytes(), table.group_bytes());
    let mut codes = vec![0u8; n * cb];
    let mut scales = vec![0u8; n * gb];
    let mut biases = vec![0u8; n * gb];
    let mut seed = 12345u64;
    let mut fresh_ids = move || -> Vec<u32> {
        (0..n)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) % 320_001_536) as u32
            })
            .collect()
    };
    let time = |label: &str, f: &mut dyn FnMut()| {
        let mut samples = Vec::new();
        for _ in 0..100 {
            let t = std::time::Instant::now();
            f();
            samples.push(t.elapsed().as_secs_f64());
        }
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "{label}: median {:.1} us, p90 {:.1} us, max {:.1} us",
            samples[50] * 1e6,
            samples[90] * 1e6,
            samples[99] * 1e6
        );
    };
    time("fresh random rows (likely cold)", &mut || {
        let ids = fresh_ids();
        table.gather(&ids, &mut codes, &mut scales, &mut biases).unwrap();
    });
    let ids = fresh_ids();
    table.gather(&ids, &mut codes, &mut scales, &mut biases).unwrap();
    time("same rows again (warm)", &mut || {
        table.gather(&ids, &mut codes, &mut scales, &mut biases).unwrap()
    });
    if std::env::var_os("LILY_PRELOAD").is_some() {
        let t = std::time::Instant::now();
        let resident = table.preload(false).unwrap();
        eprintln!(
            "preload: {:.2} GB resident after {:.1}s",
            resident as f64 / 1e9,
            t.elapsed().as_secs_f64()
        );
        time("fresh random rows after preload", &mut || {
            let ids = fresh_ids();
            table.gather(&ids, &mut codes, &mut scales, &mut biases).unwrap();
        });
    }
}
