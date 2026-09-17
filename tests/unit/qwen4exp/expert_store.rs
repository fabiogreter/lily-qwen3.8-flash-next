//! Usage ranking order and slot placement, pure; plus a byte-for-byte check of
//! the store's positioned reads against `Checkpoint::read` on the four-layer
//! checkpoint (`LILY_MODEL_DIR_FLASH`).

use super::*;

fn ranking(counts: &[&[u64]]) -> UsageRanking {
    UsageRanking {
        layers: counts.len(),
        experts: counts[0].len(),
        counts: counts.iter().map(|row| row.to_vec()).collect(),
    }
}

#[test]
fn ranked_sorts_by_count_then_position() {
    let r = ranking(&[&[5, 9, 5], &[9, 0, 7]]);
    assert_eq!(r.ranked(), vec![(0, 1), (1, 0), (1, 2), (0, 0), (0, 2), (1, 1)]);
    let u = UsageRanking::uniform(2, 2);
    assert_eq!(u.ranked(), vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
}

#[test]
fn ranking_json_round_trips() {
    let json = serde_json::json!({
        "layers": 2, "experts": 3, "top_k": 10,
        "counts": [[1, 2, 3], [4, 5, 6]],
        "prompts": 40,
    });
    let dir = std::env::temp_dir().join(format!("lily-expert-store-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("experts.json");
    std::fs::write(&path, serde_json::to_vec(&json).expect("json")).expect("write");
    let r = UsageRanking::load(&path).expect("load");
    assert_eq!((r.layers, r.experts), (2, 3));
    assert_eq!(r.ranked()[0], (1, 2));
    std::fs::remove_dir_all(&dir).ok();

    let bad = UsageRanking { layers: 2, experts: 3, counts: vec![vec![0; 3]] };
    assert!(SlotPolicy::new(4, 2, 3, &bad, 0.5).is_err());
}

/// Ranking order: (0,1)=9, (1,0)=9, (1,2)=7, (0,0)=5, (0,2)=5, (1,1)=0.
fn policy(n_slots: usize, lru_share: f64) -> SlotPolicy {
    let r = ranking(&[&[5, 9, 5], &[9, 0, 7]]);
    SlotPolicy::new(n_slots, 2, 3, &r, lru_share).expect("policy")
}

#[test]
fn pins_top_ranked_and_prefills_lru() {
    let p = policy(4, 0.5);
    assert_eq!(p.pinned(), 2);
    assert_eq!(p.n_slots(), 4);
    assert_eq!(p.slot_of(0, 1), Some(0));
    assert_eq!(p.slot_of(1, 0), Some(1));
    assert_eq!(p.slot_of(1, 2), Some(2));
    assert_eq!(p.slot_of(0, 0), Some(3));
    assert_eq!(p.slot_of(0, 2), None);
    assert_eq!(p.slot_of(1, 1), None);
    assert_eq!(
        p.initial_fill().collect::<Vec<_>>(),
        vec![(0, 0, 1), (1, 1, 0), (2, 1, 2), (3, 0, 0)]
    );
    assert_eq!(p.slot_table(0), &[3, 0, SlotPolicy::NONE]);
    assert_eq!(p.slot_table(1), &[1, SlotPolicy::NONE, 2]);
    assert_eq!(p.slice_in(2), Some((1, 2)));
}

#[test]
fn hits_never_move_and_misses_evict_worst_ranked_first() {
    let mut p = policy(4, 0.5);
    assert_eq!(p.lookup(0, 1, 1), Lookup::Hit(0));
    assert_eq!(p.lookup(1, 2, 1), Lookup::Hit(2));
    // Slot 3 (rank 4) is untouched since the fill and worse ranked than
    // slot 2, so the first cold expert evicts it.
    assert_eq!(
        p.lookup(0, 2, 1),
        Lookup::Miss { slot: 3, evicted: Some((0, 0)) }
    );
    assert_eq!(p.slot_of(0, 0), None);
    assert_eq!(p.slot_of(0, 2), Some(3));
    assert_eq!(p.slice_in(3), Some((0, 2)));
    // Both LRU slots were used at tick 1: nothing may be evicted this step.
    assert_eq!(p.lookup(1, 1, 1), Lookup::Full);
    assert_eq!(p.slot_of(1, 1), None);
    assert_eq!(p.slot_of(0, 2), Some(3));
    // Pinned slots never serve as victims, even when the LRU region is full.
    assert_eq!(p.slot_of(0, 1), Some(0));
    assert_eq!(p.slot_of(1, 0), Some(1));
}

#[test]
fn lru_evicts_oldest_tick() {
    let mut p = policy(4, 0.5);
    assert_eq!(p.lookup(1, 2, 1), Lookup::Hit(2));
    assert_eq!(p.lookup(0, 0, 2), Lookup::Hit(3));
    // Slot 2 was last used at tick 1, slot 3 at tick 2.
    assert_eq!(
        p.lookup(0, 2, 3),
        Lookup::Miss { slot: 2, evicted: Some((1, 2)) }
    );
    assert_eq!(p.lookup(0, 2, 4), Lookup::Hit(2));
    // Now slot 3 (tick 2) is older than slot 2 (tick 4).
    assert_eq!(
        p.lookup(1, 1, 5),
        Lookup::Miss { slot: 3, evicted: Some((0, 0)) }
    );
    // The evicted expert comes back through the LRU slot used longest ago.
    assert_eq!(
        p.lookup(1, 2, 6),
        Lookup::Miss { slot: 2, evicted: Some((0, 2)) }
    );
    assert_eq!(p.slot_table(0), &[SlotPolicy::NONE, 0, SlotPolicy::NONE]);
    assert_eq!(p.slot_table(1), &[1, 3, 2]);
}

#[test]
fn no_lru_region_means_cold_is_full() {
    let mut p = policy(4, 0.0);
    assert_eq!(p.pinned(), 4);
    assert_eq!(p.lookup(0, 2, 1), Lookup::Full);
    assert_eq!(p.slot_of(0, 2), None);
}

#[test]
fn empty_slots_fill_before_anything_is_evicted() {
    // More slots than slices: everything pinned that the ranking covers,
    // one slot stays empty.
    let p = policy(7, 0.0);
    assert_eq!(p.pinned(), 6);
    assert_eq!(p.initial_fill().count(), 6);
    assert_eq!(p.slice_in(6), None);

    let r = ranking(&[&[3, 2, 1]]);
    let mut p = SlotPolicy::new(4, 1, 3, &r, 0.5).expect("policy");
    assert_eq!(p.pinned(), 2);
    assert_eq!(p.slot_of(0, 2), Some(2));
    assert_eq!(p.slice_in(3), None);
    // Nothing is cold, so a lookup on the LRU-resident expert hits.
    assert_eq!(p.lookup(0, 2, 1), Lookup::Hit(2));
}

/// Reads two experts through the store and compares every region with the
/// corresponding slice of the whole tensor.
#[test]
fn store_reads_match_checkpoint() {
    let Ok(dir) = std::env::var("LILY_MODEL_DIR_FLASH") else {
        eprintln!("LILY_MODEL_DIR_FLASH unset; skipping store_reads_match_checkpoint");
        return;
    };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint");
    let config = Qwen4ExpConfig::from_model_dir(&dir).expect("config");
    let store = ExpertStore::open(&ckpt, &config).expect("store");
    assert_eq!(store.layers(), config.num_hidden_layers);
    assert_eq!(store.experts(), config.num_experts);

    let (i, h, g) = (
        config.moe_intermediate_size,
        config.hidden_size,
        config.quantization.group_size,
    );
    let lens = store.region_lens();
    assert_eq!(
        lens,
        [
            i * (h / 8) * 4,
            i * (h / g) * 2,
            i * (h / g) * 2,
            i * (h / 8) * 4,
            i * (h / g) * 2,
            i * (h / g) * 2,
            h * (i / 8) * 4,
            h * (i / g) * 2,
            h * (i / g) * 2,
        ]
    );
    assert_eq!(store.slice_bytes(), lens.iter().sum::<usize>());

    for (layer, expert) in [(1usize, 7usize), (0, config.num_experts - 1)] {
        let mut bufs: Vec<Vec<u8>> = lens.iter().map(|&n| vec![0u8; n]).collect();
        let dst: [&mut [u8]; REGIONS] = bufs
            .iter_mut()
            .map(Vec::as_mut_slice)
            .collect::<Vec<_>>()
            .try_into()
            .expect("nine buffers");
        store.read_into(layer, expert, dst).expect("read_into");
        let slice = store.slice(layer, expert);
        for (r, suffix) in REGION_NAMES.iter().enumerate() {
            let name = format!("model.language_model.layers.{layer}.mlp.experts.{suffix}");
            let meta = ckpt.meta(&name).expect("meta");
            let per_expert = meta.shape[1] * meta.shape[2] * meta.dtype.size().expect("dtype");
            assert_eq!(per_expert, lens[r], "{name}");
            assert_eq!(slice.regions[r].len, lens[r], "{name}");
            assert_eq!(
                slice.regions[r].offset,
                meta.start() + (expert * per_expert) as u64,
                "{name}"
            );
            assert_eq!(store.shard_path(slice.regions[r].shard), meta.shard(), "{name}");
            let whole = ckpt.read(&name).expect("read tensor");
            let want = &whole[expert * per_expert..(expert + 1) * per_expert];
            assert!(bufs[r] == want, "layer {layer} expert {expert} {suffix} differs");
        }
    }

    // A wrongly sized destination is refused before any read.
    let mut short: Vec<Vec<u8>> = lens.iter().map(|&n| vec![0u8; n]).collect();
    short[4].pop();
    let dst: [&mut [u8]; REGIONS] = short
        .iter_mut()
        .map(Vec::as_mut_slice)
        .collect::<Vec<_>>()
        .try_into()
        .expect("nine buffers");
    assert!(store.read_into(0, 0, dst).is_err());
}
