use super::*;
use crate::kernels::Arg;

fn u32s(ctx: &MetalContext, v: &[u32]) -> Tensor {
    let t = Tensor::zeros(ctx, &[v.len()], DType::U32).expect("tensor");
    t.write_bytes(bytemuck::cast_slice(v)).expect("write");
    t
}

/// The accepted count is the number of leading rows whose draw equals the
/// draft after it, and the chain rows' positions and block flags follow it.
#[test]
fn accept_counts_leading_matches_and_describes_the_chain() {
    let ctx = MetalContext::new().expect("metal context");
    let (pos0, ratio, chain) = (13usize, 4usize, 3usize);
    // (draws, ids) -> expected accepted count; ids[0] is the pending token.
    let cases: [(&[u32], &[u32], u32); 5] = [
        (&[5, 6, 7, 8], &[1, 5, 6, 7], 3),
        (&[5, 6, 9, 8], &[1, 5, 6, 7], 2),
        (&[5, 0, 7, 8], &[1, 5, 6, 7], 1),
        (&[4, 6, 7, 8], &[1, 5, 6, 7], 0),
        (&[5], &[1], 0),
    ];
    for (draws, ids, expected) in cases {
        let draws_t = u32s(&ctx, draws);
        let ids_t = u32s(&ctx, ids);
        let ctrl = Tensor::zeros(&ctx, &[ctrl_words(chain)], DType::U32).expect("ctrl");
        let pass = ctx.begin().expect("pass");
        spec_accept(&ctx, &pass, &draws_t, &ids_t, &ctrl, pos0, ratio, chain).expect("accept");
        pass.commit_wait().expect("run");
        let words = ctrl.to_u32().expect("read");
        let slot = |s: usize| words[s * CTRL_STRIDE];
        assert_eq!(slot(SLOT_ACCEPTED), expected, "draws {draws:?} ids {ids:?}");
        assert_eq!(slot(SLOT_KEEP), expected + 1);
        for i in 0..chain {
            let p = pos0 + expected as usize + 1 + i;
            assert_eq!(slot(slot_pos(i)) as usize, p);
            assert_eq!(slot(slot_block(i)) as usize, p / ratio);
            assert_eq!(slot(slot_count(i)), u32::from((p + 1).is_multiple_of(ratio)));
        }
    }
}

/// A row index written by an earlier dispatch of the same pass selects the
/// copied row; an index past the end copies nothing.
#[test]
fn copy_row_reads_a_gpu_supplied_index() {
    let ctx = MetalContext::new().expect("metal context");
    // Three source rows, but up to three drafts can be accepted: the index 3
    // lies past the end.
    let rows = 3usize;
    let words = 300usize;
    let src: Vec<u32> = (0..rows * words).map(|i| i as u32 * 7 + 1).collect();
    let src_t = Tensor::zeros(&ctx, &[rows, words], DType::U32).expect("src");
    src_t.write_bytes(bytemuck::cast_slice(&src)).expect("write");
    for wanted in [0usize, 2, 3, 4] {
        let dst = u32s(&ctx, &vec![0xdead_beef; words]);
        // The index arrives through the accept kernel (a = wanted, capped at
        // the drafts), read by copy_row through Arg::Gpu.
        let m = 4usize;
        let draws: Vec<u32> = (0..m as u32).map(|j| if (j as usize) < wanted { 100 + j } else { 0 }).collect();
        let ids: Vec<u32> = std::iter::once(1).chain((0..m as u32 - 1).map(|j| 100 + j)).collect();
        let ctrl = Tensor::zeros(&ctx, &[ctrl_words(0)], DType::U32).expect("ctrl");
        // Bound by address only: every buffer a pass reads must outlive it.
        let (draws_t, ids_t) = (u32s(&ctx, &draws), u32s(&ctx, &ids));
        let pass = ctx.begin_concurrent().expect("pass");
        spec_accept(&ctx, &pass, &draws_t, &ids_t, &ctrl, 0, 4, 0).expect("accept");
        pass.level_barrier(&[&ctrl]).expect("barrier");
        let index = ctrl_word(&ctrl, SLOT_ACCEPTED).expect("slot");
        copy_row(&ctx, &pass, &src_t, Arg::Gpu(&index), &dst).expect("copy");
        pass.commit_wait().expect("run");
        let a = ctrl.to_u32().expect("ctrl")[0] as usize;
        let expected_a = wanted.min(m - 1);
        assert_eq!(a, expected_a);
        let got = dst.to_u32().expect("dst");
        if a < rows {
            assert_eq!(got, src[a * words..(a + 1) * words].to_vec(), "row {a}");
        } else {
            assert!(got.iter().all(|&w| w == 0xdead_beef), "row {a} must copy nothing");
        }
    }
    // Host constant index.
    let dst = u32s(&ctx, &vec![0; words]);
    let pass = ctx.begin().expect("pass");
    copy_row(&ctx, &pass, &src_t, 1usize, &dst).expect("copy");
    pass.commit_wait().expect("run");
    assert_eq!(dst.to_u32().expect("dst"), src[words..2 * words].to_vec());
}
