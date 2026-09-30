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

/// A chunked prefill hashes chunk k+1 (staged ahead, while the GPU runs
/// chunk k) with the history advanced over chunk k. For every split point,
/// eos tokens on and next to the boundary included, that gives the ids of
/// hashing the whole prompt at once.
#[test]
fn chunked_ids_with_advanced_history_equal_whole_prompt_ids() {
    let mult = [23_703_573_157_769u64, 20_109_073_645_365, 8_052_911_324_071];
    let sizes: Vec<u64> = (0..16).map(|j| 20_000_003 + 10 * j as u64).collect();
    let offsets: Vec<u64> = (0..16).map(|j| j as u64 * 20_000_100).collect();
    let eos = 248_044u32;
    let hasher = NgramHasher::new(&mult, &sizes, &offsets, 8, eos).expect("hasher");
    let tokens = vec![17u32, 248_319, eos, 5, 6, eos, eos, 9, 11, eos, 3, 3, 3, 7];
    let hist = [eos, 42];
    let mut whole = Vec::new();
    hasher.ids(&tokens, hist, &mut whole);
    for chunk in 1..=tokens.len() {
        let mut h = hist;
        let mut chunked = Vec::new();
        for part in tokens.chunks(chunk) {
            hasher.ids(part, h, &mut chunked);
            h = NgramHasher::advance(h, part);
        }
        assert_eq!(chunked, whole, "chunks of {chunk}");
        assert_eq!(h, NgramHasher::advance(hist, &tokens));
    }
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
    let never = std::sync::atomic::AtomicBool::new(false);
    let done = table.preload(false, &never).expect("preload");
    assert!(done.resident >= table.bytes() && !done.stopped);
    // Just written, so already resident: the preload only checks.
    assert_eq!(done.read, 0);
    // A stop raised before the first piece ends it there.
    let stop = std::sync::atomic::AtomicBool::new(true);
    let stopped = table.preload(true, &stop).expect("stopped preload");
    assert!(stopped.stopped && stopped.read == 0);
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
    // The same preload on its own thread reports through its callback; the
    // handle joins the thread when it goes, finished or not.
    let table = Arc::new(table);
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = table
        .preload_in_background(true, move |outcome, secs| {
            tx.send((outcome.map_err(|e| e.to_string()), secs)).unwrap();
        })
        .expect("background preload");
    let (outcome, secs) =
        rx.recv_timeout(std::time::Duration::from_secs(30)).expect("preload reported");
    let outcome = outcome.expect("background preload succeeded");
    assert!(outcome.resident >= table.bytes() && !outcome.stopped && secs >= 0.0);
    drop(handle);
    // Once the handle is gone, nothing but this test holds the table.
    assert_eq!(Arc::strong_count(&table), 1);
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
        let resident = table
            .preload(false, &std::sync::atomic::AtomicBool::new(false))
            .unwrap()
            .resident;
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

/// Host cost of `gather` on a warm synthetic table (a 2M-row shard written to
/// the temp directory, so every page is in the page cache): what the
/// residency checks cost on a decode step, a verify step and a prefill chunk.
/// Run with `cargo test --lib paged_gather_overhead -- --ignored --nocapture`.
#[test]
#[ignore = "timing only"]
fn paged_gather_overhead() {
    let dir = std::env::temp_dir()
        .join(format!("lily-ngram-overhead-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let (rows, words, groups) = (2_000_000usize, 20usize, 5usize);
    let base =
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0";
    let lens = [rows * words * 4, rows * groups * 2, rows * groups * 2];
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for ((suffix, dtype, width), len) in [
        ("weight", "U32", words),
        ("scales", "BF16", groups),
        ("biases", "BF16", groups),
    ]
    .into_iter()
    .zip(lens)
    {
        header.insert(
            format!("{base}.{suffix}"),
            serde_json::json!({"dtype": dtype, "shape": [rows, width], "data_offsets": [offset, offset + len]}),
        );
        offset += len;
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut f = std::fs::File::create(dir.join("model.safetensors")).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&header).unwrap();
    let block = vec![0x5au8; 1 << 20];
    let mut left = offset;
    while left > 0 {
        let n = left.min(block.len());
        f.write_all(&block[..n]).unwrap();
        left -= n;
    }
    drop(f);
    let ckpt = Checkpoint::open(&dir).expect("checkpoint");
    let table = PagedTable::open(
        &ckpt,
        &shard_bases("model.language_model.layers.1.ple.", 1),
        32,
    )
    .expect("paged table");
    table.preload(false, &std::sync::atomic::AtomicBool::new(false)).expect("preload");
    let (cb, gb) = (table.codes_bytes(), table.group_bytes());
    let mut seed = 99u64;
    let mut ids = |n: usize| -> Vec<u32> {
        (0..n)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) % rows as u64) as u32
            })
            .collect()
    };
    for (label, n, reps) in [
        ("decode step, 16 rows", 16, 20_000),
        ("verify step, 48 rows", 48, 20_000),
        ("prefill chunk, 65 536 rows", 65_536, 40),
    ] {
        let batches: Vec<Vec<u32>> = (0..64).map(|_| ids(n)).collect();
        let mut codes = vec![0u8; n * cb];
        let mut scales = vec![0u8; n * gb];
        let mut biases = vec![0u8; n * gb];
        let mut samples = Vec::with_capacity(reps);
        for r in 0..reps {
            let t = std::time::Instant::now();
            table
                .gather(
                    &batches[r % batches.len()],
                    &mut codes,
                    &mut scales,
                    &mut biases,
                )
                .unwrap();
            samples.push(t.elapsed().as_secs_f64());
        }
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "{label}: median {:.2} us, p90 {:.2} us",
            samples[reps / 2] * 1e6,
            samples[reps * 9 / 10] * 1e6
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Writes a one-shard table of `rows` rows (20 code words, 5 groups) with
/// `F_NOCACHE`, so its pages bypass the page cache and start cold. Returns
/// the three tensors' byte lengths and the data they hold, in file order.
fn write_cold_shard(dir: &Path, rows: usize) -> ([usize; 3], Vec<u8>) {
    use std::os::fd::AsRawFd as _;
    std::fs::create_dir_all(dir).expect("temp dir");
    let (words, groups) = (20usize, 5usize);
    let base =
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0";
    let lens = [rows * words * 4, rows * groups * 2, rows * groups * 2];
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for ((suffix, dtype, width), len) in [
        ("weight", "U32", words),
        ("scales", "BF16", groups),
        ("biases", "BF16", groups),
    ]
    .into_iter()
    .zip(lens)
    {
        header.insert(
            format!("{base}.{suffix}"),
            serde_json::json!({"dtype": dtype, "shape": [rows, width], "data_offsets": [offset, offset + len]}),
        );
        offset += len;
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let data: Vec<u8> = (0..offset).map(|i| (i % 251) as u8).collect();
    let mut f = std::fs::File::create(dir.join("model.safetensors")).unwrap();
    // SAFETY: an fcntl on a file this test owns.
    unsafe { libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1) };
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&header).unwrap();
    f.write_all(&data).unwrap();
    f.sync_all().unwrap();
    (lens, data)
}

/// Checks that gathered row `i` holds table row `id` of [`write_cold_shard`].
fn assert_row(
    lens: &[usize; 3],
    data: &[u8],
    staged: (&[u8], &[u8], &[u8]),
    i: usize,
    id: u32,
) {
    let (words, groups) = (20usize, 5usize);
    let (codes, scales, biases) = staged;
    let r = id as usize;
    let c = r * words * 4;
    assert_eq!(&codes[i * words * 4..(i + 1) * words * 4], &data[c..c + words * 4]);
    let s = lens[0] + r * groups * 2;
    assert_eq!(&scales[i * groups * 2..(i + 1) * groups * 2], &data[s..s + groups * 2]);
    let b = lens[0] + lens[1] + r * groups * 2;
    assert_eq!(&biases[i * groups * 2..(i + 1) * groups * 2], &data[b..b + groups * 2]);
}

/// A prefill-sized batch over a cold table: the 1-in-16 sample finds it cold,
/// so the other rows are hinted too (and their pages are not counted as
/// sampled); the rows are the table's. Gathered again, warm, nothing beyond
/// the sample is checked.
#[test]
fn a_cold_prefill_batch_hints_every_row_and_a_warm_one_only_the_sample() {
    let dir =
        std::env::temp_dir().join(format!("lily-ngram-dense-{}", std::process::id()));
    let rows = 400_000usize;
    let (lens, data) = write_cold_shard(&dir, rows);
    let ckpt = Checkpoint::open(&dir).expect("checkpoint");
    let table = PagedTable::open(
        &ckpt,
        &shard_bases("model.language_model.layers.1.ple.", 1),
        32,
    )
    .expect("paged table");
    let cold_before = table.resident_bytes().expect("mincore") < table.bytes() / 2;
    // 4 096 rows spread over the table, out of order: the parallel path.
    let ids: Vec<u32> =
        (0..4096u64).map(|i| (i * 97_331 % rows as u64) as u32).collect();
    let n = ids.len();
    let mut codes = vec![0u8; n * 80];
    let mut scales = vec![0u8; n * 10];
    let mut biases = vec![0u8; n * 10];
    let mut gather = || {
        let before = crate::stats::counters();
        table.gather(&ids, &mut codes, &mut scales, &mut biases).expect("gather");
        for (i, &id) in ids.iter().enumerate() {
            assert_row(&lens, &data, (&codes, &scales, &biases), i, id);
        }
        crate::stats::counters().since(before).gather
    };
    let cold = gather();
    assert_eq!(cold.checked_rows, n.div_ceil(PREFILL_CHECK_EVERY) as u64);
    if cold_before {
        assert!(cold.cold_pages * DENSE_HINT_SHARE >= cold.pages, "{cold:?}");
        assert_eq!(cold.hinted_rows + cold.checked_rows, n as u64, "{cold:?}");
    }
    let warm = gather();
    assert_eq!((warm.cold_pages, warm.hinted_rows), (0, 0), "{warm:?}");
    assert_eq!(warm.checked_rows, cold.checked_rows);
    std::fs::remove_dir_all(&dir).ok();
}

/// The preload reads a table that is not in the page cache: a shard written
/// with `F_NOCACHE` (so its pages bypass the cache) is read back through the
/// files, ends up resident, and gathers the bytes that were written.
#[test]
fn preload_reads_cold_shards_into_the_page_cache() {
    let dir =
        std::env::temp_dir().join(format!("lily-ngram-cold-{}", std::process::id()));
    let (rows, words, groups) = (400_000usize, 20usize, 5usize);
    let (lens, data) = write_cold_shard(&dir, rows);
    let ckpt = Checkpoint::open(&dir).expect("checkpoint");
    let table = PagedTable::open(
        &ckpt,
        &shard_bases("model.language_model.layers.1.ple.", 1),
        32,
    )
    .expect("paged table");
    let before = table.resident_bytes().expect("mincore");
    let done = table
        .preload(false, &std::sync::atomic::AtomicBool::new(false))
        .expect("preload");
    assert!(done.resident >= table.bytes() && !done.stopped);
    // The write left (nearly) nothing in the cache, so pieces were read (not
    // all of them: the kernel's read-ahead behind one read can make the next
    // piece resident before the preload checks it).
    if before < table.bytes() / 2 {
        assert!(done.read > 0, "{before} bytes were resident before");
    }
    let ids = [0u32, 123_456, 399_999];
    let mut codes = vec![0u8; ids.len() * words * 4];
    let mut scales = vec![0u8; ids.len() * groups * 2];
    let mut biases = vec![0u8; ids.len() * groups * 2];
    table.gather(&ids, &mut codes, &mut scales, &mut biases).expect("gather");
    for (i, &id) in ids.iter().enumerate() {
        assert_row(&lens, &data, (&codes, &scales, &biases), i, id);
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Host cost of gathering one prefill chunk's rows (65 536 random rows, or
/// `ROWS`) from a real checkpoint: cold (put the table into a partly cold
/// state first, e.g. `docs/bench/2026-10-01-ngram-prefetch/evict_table.py`),
/// then warm twice, with the pages the system read in meanwhile. Run with
/// `LILY_MODEL_DIR_FLASH=<ckpt> SEED=<n> cargo test --release --lib
/// paged_cold_chunk_timing -- --ignored --nocapture`; a new `SEED` picks new
/// rows.
#[test]
#[ignore = "timing only; needs LILY_MODEL_DIR_FLASH"]
fn paged_cold_chunk_timing() {
    let Ok(dir) = std::env::var("LILY_MODEL_DIR_FLASH") else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint");
    let bases = shard_bases("model.language_model.layers.1.ple.", 128);
    let table = PagedTable::open(&ckpt, &bases, 32).expect("paged table");
    let env = |name: &str, default: u64| -> u64 {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    let n = env("ROWS", 65_536) as usize;
    let mut seed = env("SEED", 1);
    let ids: Vec<u32> = (0..n)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) % table.rows() as u64) as u32
        })
        .collect();
    let (cb, gb) = (table.codes_bytes(), table.group_bytes());
    let mut codes = vec![0u8; n * cb];
    let mut scales = vec![0u8; n * gb];
    let mut biases = vec![0u8; n * gb];
    for pass in ["first", "again", "again"] {
        let before = crate::stats::counters();
        let vm = crate::stats::vm_counters();
        let t = std::time::Instant::now();
        table.gather(&ids, &mut codes, &mut scales, &mut biases).unwrap();
        let secs = t.elapsed().as_secs_f64();
        let pageins = crate::stats::vm_counters()
            .zip(vm)
            .map_or(0, |(after, before)| after.since(before).pageins);
        let g = crate::stats::counters().since(before).gather;
        eprintln!(
            "{pass}: {n} rows in {:.1} ms, sampled pages cold {}/{}, {} rows hinted beyond the sample, {pageins} pages read in",
            secs * 1e3,
            g.cold_pages,
            g.pages,
            g.hinted_rows
        );
    }
}
