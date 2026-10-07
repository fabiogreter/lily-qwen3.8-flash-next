//! The persisted per-token cache prefix (`prefix_layout` / `read_prefix`):
//! the layout's arithmetic on the CPU, and the round trip through a
//! synthetic state (a Metal device, no checkpoint).

use std::io::Cursor;

use crate::engine::DecodeStateApi;
use crate::metal::MetalContext;

use super::*;

/// The ratio the config admits ([`Qwen4ExpConfig`] checks it).
const RATIO: usize = 4;
const HEADS: usize = 2;
/// Smaller than the checkpoint's 256: the layout's arithmetic does not
/// depend on it, and the round trip stays light.
const HEAD_DIM: usize = 32;
const BF16: usize = 2;
const FORMATS: [KvFormat; 2] = [KvFormat::Bf16, KvFormat::Q8];

/// A region's row width in bytes.
fn row_bytes(format: KvFormat, store: AttnStore) -> usize {
    match store {
        AttnStore::K | AttnStore::V => match format {
            KvFormat::Bf16 => HEAD_DIM * BF16,
            KvFormat::Q8 => HEAD_DIM,
        },
        AttnStore::KScales | AttnStore::VScales => HEAD_DIM / 32 * 2,
        AttnStore::IdxKeys | AttnStore::BlkKeys => INDEXER_D * BF16,
    }
}

/// The plan for a state of `capacity` tokens, its block-key rows sized as
/// the state allocates them.
fn plan(
    format: KvFormat,
    capacity: usize,
    tokens: usize,
    written: usize,
) -> Result<Vec<PrefixRegion>> {
    attn_prefix_plan(
        HEADS,
        capacity,
        format == KvFormat::Q8,
        block_key_rows(capacity, RATIO),
        RATIO,
        tokens,
        written,
    )
}

/// Per region, the byte range of the layout it reads (or writes) and the
/// end of what it skips past them.
fn spans(format: KvFormat, plan: &[PrefixRegion]) -> Vec<(usize, usize, usize)> {
    let mut at = 0;
    plan.iter()
        .map(|r| {
            let start = at;
            let read = start + r.rows * row_bytes(format, r.store);
            at = read + r.skip * row_bytes(format, r.store);
            (start, read, at)
        })
        .collect()
}

/// The layout's size for `tokens` per attention layer.
fn layer_bytes(format: KvFormat, tokens: usize) -> usize {
    2 * HEADS * tokens * format.row_bytes(HEAD_DIM)
        + tokens * INDEXER_D * BF16
        + tokens.div_ceil(RATIO) * INDEXER_D * BF16
}

/// A restore reads every region from where the writer put it and ends
/// where the writer's layout ends, whatever capacity the restoring state
/// was built with: the entry spans several capacity steps, the state is
/// sized for an earlier checkpoint (the prompt that hit it), and resumes
/// land on block boundaries, inside blocks and at either end. The bug this
/// pins: the block-key skip was sized by the reader's block-key rows, so a
/// checkpoint at 5 000 inside a 32 768-token entry restored into an 8 192
/// state skipped 798 rows instead of 6 942.
#[test]
fn a_restore_reads_each_region_where_the_writer_put_it_whatever_its_capacity() {
    for format in FORMATS {
        restore_reads_each_region_where_the_writer_put_it(format);
    }
}

fn restore_reads_each_region_where_the_writer_put_it(format: KvFormat) {
    let plan = |c, t, w| plan(format, c, t, w);
    let spans = |p: &[PrefixRegion]| spans(format, p);
    let layer_bytes = |t| layer_bytes(format, t);
    let writtens = [1, 4, 5, 8191, 8192, 8193, 3 * 8192 + 1234, 32_768, MAX_SEQ];
    for written in writtens {
        let w_capacity = round_capacity(written).unwrap();
        let writer = plan(w_capacity, written, written).unwrap();
        let wrote = spans(&writer);
        assert!(writer.iter().all(|r| r.skip == 0), "a writer skips nothing");
        assert_eq!(wrote.last().unwrap().2, layer_bytes(written));
        let tokens_at =
            [0, 1, 3, 4, 5, 4999, 5000, 8192, 8193, written / 2, written - 1, written];
        for tokens in tokens_at.into_iter().filter(|&t| t <= written) {
            // The state `acquire_from_disk` builds (for a prompt as long as
            // the checkpoint), and one already grown to the writer's size.
            for capacity in [round_capacity(tokens).unwrap(), w_capacity] {
                let reader = plan(capacity, tokens, written).unwrap();
                let read = spans(&reader);
                assert_eq!(reader.len(), writer.len());
                for (i, ((r, w), (rs, ws))) in
                    reader.iter().zip(&writer).zip(read.iter().zip(&wrote)).enumerate()
                {
                    let at = format!(
                        "{format:?} region {i} ({:?}), {tokens} of {written} into {capacity}",
                        r.store
                    );
                    assert_eq!(r.store, w.store, "{at}");
                    assert_eq!(rs.0, ws.0, "{at}: starts where the writer's does");
                    assert_eq!(rs.2, ws.2, "{at}: ends where the writer's does");
                    assert!(r.rows <= w.rows, "{at}");
                    // Rows land at the same place within each state's own
                    // buffers: head h at h * capacity.
                    assert_eq!(r.first % capacity, 0, "{at}");
                    assert_eq!(r.first / capacity, w.first / w_capacity, "{at}");
                }
                assert_eq!(read.last().unwrap().2, layer_bytes(written));
                let blocks = reader.last().unwrap();
                assert_eq!(blocks.store, AttnStore::BlkKeys);
                assert_eq!(blocks.rows, tokens.div_ceil(RATIO));
            }
        }
    }
}

/// What the state cannot hold is an error, never a shorter read or write.
#[test]
fn a_prefix_the_state_cannot_hold_is_an_error_not_a_shorter_read() {
    for format in FORMATS {
        let q8 = format == KvFormat::Q8;
        assert!(
            plan(format, 8192, 8193, 9000).is_err(),
            "more tokens than the capacity"
        );
        assert!(plan(format, 8192, 10, 9).is_err(), "a prefix longer than the layout");
        // Fewer block-key rows than the prefix starts blocks.
        assert!(attn_prefix_plan(HEADS, 8192, q8, 1024, RATIO, 5000, 5000).is_err());
        assert!(attn_prefix_plan(HEADS, 8192, q8, 1250, RATIO, 5000, 32_768).is_ok());
        assert!(attn_prefix_plan(HEADS, 8192, q8, 1249, RATIO, 4997, 32_768).is_err());
    }
}

/// A q8 layer persists its scales after the values, a region per head and
/// cache, and nothing else changes: the bf16 layout is the one entries on
/// disk already have.
#[test]
fn a_q8_layout_adds_the_scale_regions_after_the_values() {
    let stores = |format| {
        plan(format, 8192, 100, 100)
            .unwrap()
            .iter()
            .map(|r| r.store)
            .collect::<Vec<_>>()
    };
    use AttnStore::*;
    assert_eq!(stores(KvFormat::Bf16), [K, K, V, V, IdxKeys, BlkKeys]);
    assert_eq!(
        stores(KvFormat::Q8),
        [K, K, V, V, KScales, KScales, VScales, VScales, IdxKeys, BlkKeys]
    );
}

/// Every capacity a state can have holds the block keys of every prefix up
/// to it, the partial last block included, so the plan never refuses a
/// state's own contents.
#[test]
fn every_capacity_holds_the_block_keys_of_its_whole_prefix() {
    let mut capacity = round_capacity(0).unwrap();
    loop {
        for tokens in [capacity - 1, capacity] {
            for format in FORMATS {
                assert!(
                    plan(format, capacity, tokens, tokens).is_ok(),
                    "{tokens} in {capacity}"
                );
            }
        }
        if capacity == MAX_SEQ {
            break;
        }
        capacity = round_capacity(capacity + 1).unwrap();
    }
    // Nothing fed: an empty prefix of empty regions.
    let empty = plan(KvFormat::Q8, 8192, 0, 0).unwrap();
    assert!(empty.iter().all(|r| r.rows == 0 && r.skip == 0));
}

// --- the round trip through a synthetic state -------------------------------

/// A state of `capacity` tokens with the layer mix that matters to the
/// layout: GDN layers (no per-token caches) between two attention layers,
/// and the draft head's attention layer.
fn synthetic_state(
    ctx: &MetalContext,
    format: KvFormat,
    capacity: usize,
) -> DecodeState {
    let capacity = round_capacity(capacity).expect("capacity");
    let attn = || {
        DecodeState::attn_caches(ctx, format, HEADS, HEAD_DIM, RATIO, capacity)
            .expect("caches")
    };
    let gdn = || LayerState::Gdn {
        state: Tensor::zeros(ctx, &[1, 4, 4], DType::F32).expect("state"),
        conv_windows: [
            Tensor::zeros(ctx, &[8, 3], DType::BF16).expect("window"),
            Tensor::zeros(ctx, &[8, 3], DType::BF16).expect("window"),
        ],
    };
    DecodeState {
        pos: 0,
        rope_delta: 0,
        capacity,
        layers: vec![gdn(), attn(), gdn(), attn()],
        conv_slot: 0,
        ple: None,
        mtp: Some(MtpState {
            layer: attn(),
            hidden: Tensor::zeros(ctx, &[8], DType::BF16).expect("hidden"),
        }),
        spec: None,
        kv_format: format,
        kv_heads: HEADS,
        head_dim: HEAD_DIM,
        ratio: RATIO,
    }
}

/// The state's attention layers, the draft head's last: per layer, the
/// per-head stores (K and V values, then a q8 cache's K and V scales) and
/// the indexer and block keys.
#[allow(clippy::type_complexity)]
fn attn_layers(
    state: &DecodeState,
) -> Vec<(Vec<(&'static str, &Tensor)>, &Tensor, &Tensor)> {
    state
        .layers
        .iter()
        .chain(state.mtp.as_ref().map(|m| &m.layer))
        .filter_map(|l| match l {
            LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys } => {
                let mut heads = vec![("K", &k_cache.values), ("V", &v_cache.values)];
                if let (Some(ks), Some(vs)) = (&k_cache.scales, &v_cache.scales) {
                    heads.extend([("K scales", ks), ("V scales", vs)]);
                }
                Some((heads, idx_keys, blk_keys))
            }
            LayerState::Gdn { .. } => None,
        })
        .collect()
}

/// Fills every attention cache with bytes distinct per layer, store and
/// offset, so a region read from another's place cannot compare equal.
fn fill_distinct(state: &DecodeState) {
    for (layer, (heads, idx, blk)) in attn_layers(state).into_iter().enumerate() {
        let stores = heads.into_iter().map(|(_, t)| t).chain([idx, blk]);
        for (store, t) in stores.enumerate() {
            let seed =
                ((layer * 8 + store + 1) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let bytes: Vec<u8> = (0..t.byte_len() / 8)
                .flat_map(|i| {
                    (seed ^ (i as u64))
                        .wrapping_mul(0xBF58_476D_1CE4_E5B9)
                        .to_le_bytes()
                })
                .collect();
            t.write_bytes(&bytes).expect("fill");
        }
    }
}

/// `rows` rows from row `first` of `t` (`[.., width]`).
fn rows(t: &Tensor, first: usize, rows: usize) -> &[u8] {
    let row = t.shape()[t.shape().len() - 1] * t.dtype().size();
    &t.raw_bytes()[first * row..(first + rows) * row]
}

/// Asserts that `dst` holds `src`'s first `tokens` entries in every
/// region of every attention layer, the draft head's included.
fn assert_prefix_equal(src: &DecodeState, dst: &DecodeState, tokens: usize, at: &str) {
    let (s_cap, d_cap) = (src.capacity, dst.capacity);
    let pairs = attn_layers(src).into_iter().zip(attn_layers(dst));
    for (layer, ((sh, si, sb), (dh, di, db))) in pairs.enumerate() {
        assert_eq!(sh.len(), dh.len(), "{at}: layer {layer} stores");
        for ((name, s), (_, d)) in sh.into_iter().zip(dh) {
            for h in 0..HEADS {
                assert!(
                    rows(s, h * s_cap, tokens) == rows(d, h * d_cap, tokens),
                    "{at}: layer {layer} {name} head {h}"
                );
            }
        }
        assert!(
            rows(si, 0, tokens) == rows(di, 0, tokens),
            "{at}: layer {layer} indexer keys"
        );
        let blocks = tokens.div_ceil(RATIO);
        assert!(
            rows(sb, 0, blocks) == rows(db, 0, blocks),
            "{at}: layer {layer} block keys"
        );
    }
}

/// A checkpoint inside a disk entry that spans several capacity steps,
/// restored into a state built for the checkpoint (smaller than the
/// entry's) or grown by the read itself: every region of every attention
/// layer and the draft head's comes back byte for byte, and the read
/// consumes the whole layout. Resumes on block boundaries, inside a block,
/// on and past capacity steps, at both ends.
#[test]
fn a_checkpoint_inside_a_longer_entry_restores_every_region_exactly() {
    let ctx = MetalContext::new().expect("metal context");
    for format in FORMATS {
        checkpoint_inside_a_longer_entry_restores_every_region(&ctx, format);
    }
}

fn checkpoint_inside_a_longer_entry_restores_every_region(
    ctx: &MetalContext,
    format: KvFormat,
) {
    let written = 3 * 8192 + 1234;
    let mut src = synthetic_state(ctx, format, written);
    assert_eq!(src.capacity, 32_768);
    fill_distinct(&src);
    src.pos = written;
    let mut bytes = Vec::new();
    src.write_prefix(written, &mut bytes).expect("write");
    assert_eq!(bytes.len(), 3 * layer_bytes(format, written), "three attention layers");

    for tokens in [0, 1, 4999, 5000, 8191, 8192, 8193, 16_385, written - 1, written] {
        for capacity in [tokens, 1] {
            let mut dst = synthetic_state(ctx, format, capacity);
            let mut cursor = Cursor::new(&bytes);
            dst.read_prefix(ctx, written, tokens, &mut cursor).expect("read");
            assert_eq!(
                cursor.position() as usize,
                bytes.len(),
                "the whole layout read"
            );
            // Grown to the checkpoint, not to the entry: smaller than the
            // writer's state for every resume below the last step.
            assert_eq!(dst.capacity, round_capacity(tokens).unwrap());
            let at = format!(
                "{format:?}: {tokens} of {written} into a state built for {capacity}"
            );
            assert_prefix_equal(&src, &dst, tokens, &at);
        }
    }
}

/// A layout longer or shorter than the reader expects is an error, not a
/// restore from shifted offsets.
#[test]
fn a_layout_of_the_wrong_length_is_refused() {
    let ctx = MetalContext::new().expect("metal context");
    let written = 9000;
    let mut src = synthetic_state(&ctx, KvFormat::Q8, written);
    fill_distinct(&src);
    src.pos = written;
    let mut bytes = Vec::new();
    src.write_prefix(written, &mut bytes).expect("write");

    let mut longer = bytes.clone();
    longer.push(0);
    let mut dst = synthetic_state(&ctx, KvFormat::Q8, 1);
    let error = dst
        .read_prefix(&ctx, written, 5000, &mut Cursor::new(&longer))
        .expect_err("a byte past the end");
    assert!(format!("{error:#}").contains("past its end"), "{error:#}");

    let shorter = &bytes[..bytes.len() - 1];
    assert!(dst.read_prefix(&ctx, written, 5000, &mut Cursor::new(shorter)).is_err());
    // The layout of a shorter entry read as a longer one.
    assert!(
        dst.read_prefix(&ctx, written + 4, 5000, &mut Cursor::new(&bytes)).is_err()
    );
}
