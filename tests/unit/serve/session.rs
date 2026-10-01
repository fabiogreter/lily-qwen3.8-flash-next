use super::{
    CachedImage, agreement, boundary_position, common_prefix_len, images_within,
    resume_position, shared_prefix,
};

/// No images: the lineages are text.
const TEXT: &[CachedImage] = &[];

/// An image of `len` placeholders at `start` (a `(2, 2 * len)` grid) whose
/// pixel digest is `seed` repeated.
fn image(start: usize, len: usize, seed: u8) -> CachedImage {
    CachedImage { start, len, grid_h: 2, grid_w: 2 * len, digest: [seed; 32] }
}

/// A prompt of `before` text tokens, `len` placeholders (id 99) and `after`
/// text tokens, distinct from `text_prompt`'s tokens only where asked.
fn image_prompt(before: usize, len: usize, after: &[u32]) -> Vec<u32> {
    (1..=before as u32)
        .chain(std::iter::repeat_n(99, len))
        .chain(after.iter().copied())
        .collect()
}

#[test]
fn agreement_is_the_longest_prefix_over_all_lineages_capped_below_the_prompt_end() {
    let prompt = [1, 2, 3, 4, 5];
    let gpu: &[u32] = &[1, 2, 9, 9];
    let disk: &[u32] = &[1, 2, 3, 7, 7, 7];
    // The disk lineage agrees further, and its tail is what it continued with.
    assert_eq!(
        agreement(&prompt, TEXT, [(gpu, TEXT), (disk, TEXT)]),
        (3, vec![7, 7, 7])
    );
    // Order does not matter.
    assert_eq!(
        agreement(&prompt, TEXT, [(disk, TEXT), (gpu, TEXT)]),
        (3, vec![7, 7, 7])
    );
    // A lineage that contains the whole prompt: capped at prompt_len - 1 like
    // every resume position, tail starts at the cap.
    assert_eq!(
        agreement(&prompt, TEXT, [(&[1, 2, 3, 4, 5, 6][..], TEXT)]),
        (4, vec![5, 6])
    );
    // Nothing shared, or no lineages at all.
    assert_eq!(agreement(&prompt, TEXT, [(&[8, 8][..], TEXT)]), (0, vec![]));
    assert_eq!(
        agreement(&prompt, TEXT, std::iter::empty::<(&[u32], &[CachedImage])>()),
        (0, vec![])
    );
    // The tail is bounded.
    let long: Vec<u32> = std::iter::once(1).chain(std::iter::repeat_n(4, 40)).collect();
    let (pos, tail) = agreement(&prompt, TEXT, [(long.as_slice(), TEXT)]);
    assert_eq!((pos, tail.len()), (1, 16));
}

#[test]
fn agreement_with_images_stops_at_a_different_image_and_the_boundary_follows() {
    // Two agent runs share a 4-token preamble and a 6-placeholder image,
    // then ask different questions.
    let a = image_prompt(4, 6, &[20, 21, 22, 23]);
    let b = image_prompt(4, 6, &[30, 31, 32, 33]);
    let same = [image(4, 6, 1)];
    let other = [image(4, 6, 2)];
    // Same image: they agree through the image, up to the question.
    assert_eq!(agreement(&b, &same, [(a.as_slice(), &same[..])]).0, 10);
    // A different screenshot in the same place: the agreement ends where
    // the image starts, whatever the identical tokens say.
    assert_eq!(agreement(&b, &other, [(a.as_slice(), &same[..])]).0, 4);
    // The durable boundary is the agreement, so a shared preamble plus a
    // shared image is materialised after the image and a different image
    // before it.
    assert_eq!(boundary_position(10, 0, b.len(), 4), Some(10));
    assert_eq!(boundary_position(4, 0, b.len(), 4), Some(4));
    assert_eq!(boundary_position(4, 0, b.len(), 5), None);
}

#[test]
fn boundary_rule_materialises_only_a_long_unresumed_agreement() {
    // Two runs shared 2 000 tokens, nothing was resumable: materialise there.
    assert_eq!(boundary_position(2000, 0, 3000, 1024), Some(2000));
    // Below the threshold: not worth a disk entry.
    assert_eq!(boundary_position(900, 0, 3000, 1024), None);
    // Threshold 0 disables the feature entirely.
    assert_eq!(boundary_position(2000, 0, 3000, 0), None);
    // A pure extension (one growing conversation): agreement == reused.
    assert_eq!(boundary_position(2000, 2000, 3000, 1024), None);
    // Resumed past the agreement never happens, but must not materialise.
    assert_eq!(boundary_position(2000, 2500, 3000, 1024), None);
    // The agreement may sit at the last feedable position ...
    assert_eq!(boundary_position(2999, 0, 3000, 1024), Some(2999));
    // ... but never at or past the prompt end (a token must remain to feed).
    assert_eq!(boundary_position(3000, 0, 3000, 1024), None);
    assert_eq!(boundary_position(1, 0, 0, 1), None);
    // Exactly the threshold counts.
    assert_eq!(boundary_position(1024, 0, 3000, 1024), Some(1024));
    // The threshold is measured from the resume position: a turn that
    // resumed at 74 000 and diverged 240 tokens later is an ordinary fork
    // inside one conversation, not a shared preamble.
    assert_eq!(boundary_position(74_240, 74_000, 90_000, 1024), None);
    assert_eq!(boundary_position(75_023, 74_000, 90_000, 1024), None);
    assert_eq!(boundary_position(75_024, 74_000, 90_000, 1024), Some(75_024));
}

#[test]
fn common_prefix_length() {
    assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 4]), 2);
    assert_eq!(common_prefix_len(&[1, 2], &[1, 2, 3]), 2);
    assert_eq!(common_prefix_len(&[], &[1]), 0);
    assert_eq!(common_prefix_len(&[5], &[1]), 0);
}

#[test]
fn text_only_lineages_share_exactly_their_token_prefix() {
    // `shared_prefix` without spans is `common_prefix_len`.
    assert_eq!(shared_prefix(&[1, 2, 3], TEXT, &[1, 2, 4], TEXT), 2);
    assert_eq!(shared_prefix(&[1, 2], TEXT, &[1, 2, 3], TEXT), 2);
    assert_eq!(shared_prefix(&[], TEXT, &[1], TEXT), 0);
    assert_eq!(shared_prefix(&[5], TEXT, &[1], TEXT), 0);
}

#[test]
fn shared_prefix_cuts_at_the_first_image_the_other_lineage_does_not_match() {
    let a = image_prompt(3, 4, &[10, 11, 12]);
    let b = image_prompt(3, 4, &[10, 11, 13]);
    let one = [image(3, 4, 1)];
    let two = [image(3, 4, 2)];
    // Identical tokens, different digests: cut at the span start.
    assert_eq!(shared_prefix(&a, &one, &a, &two), 3);
    // Equal digests share fully (the tokens decide).
    assert_eq!(shared_prefix(&a, &one, &a, &one), a.len());
    assert_eq!(shared_prefix(&a, &one, &b, &one), 9);
    // The same pixels at another position, or another grid or length for
    // the same pixels, is a different span too.
    assert_eq!(shared_prefix(&a, &one, &a, &[image(2, 4, 1)]), 2);
    let mut wider = image(3, 4, 1);
    wider.grid_h = 4;
    wider.grid_w = 4;
    assert_eq!(shared_prefix(&a, &one, &a, &[wider]), 3);
    // A lineage without a span where the other has one: the placeholders
    // are identical tokens, but only one side has an image there.
    assert_eq!(shared_prefix(&a, &one, &a, TEXT), 3);
    assert_eq!(shared_prefix(&a, TEXT, &a, &one), 3);
    // A span straddling the token divergence point: the tokens diverge
    // inside the placeholders (one image is longer), so neither span
    // matches the other and the cut lands at the span start.
    let longer = image_prompt(3, 6, &[10]);
    assert_eq!(common_prefix_len(&a, &longer), 7);
    assert_eq!(shared_prefix(&a, &one, &longer, &[image(3, 6, 1)]), 3);
    // Spans past the token divergence never matter.
    let c = image_prompt(2, 4, &[10]);
    assert_eq!(shared_prefix(&a, &one, &c, &[image(2, 4, 1)]), 2);
    // Several images: the earliest mismatch decides, in either lineage.
    let d = image_prompt(2, 2, &[5, 99, 99, 99, 99, 6]);
    let d_images = [image(2, 2, 1), image(5, 4, 2)];
    assert_eq!(shared_prefix(&d, &d_images, &d, &d_images), d.len());
    assert_eq!(shared_prefix(&d, &d_images, &d, &[image(2, 2, 1), image(5, 4, 3)]), 5);
    assert_eq!(shared_prefix(&d, &d_images, &d, &[image(2, 2, 7), image(5, 4, 2)]), 2);
    assert_eq!(shared_prefix(&d, &d_images, &d, &[image(2, 2, 1)]), 5);
    assert_eq!(shared_prefix(&d, &[image(2, 2, 1)], &d, &d_images), 5);
}

#[test]
fn images_within_keeps_the_spans_that_end_inside_the_prefix() {
    let images = [image(2, 2, 1), image(5, 4, 2)];
    assert_eq!(images_within(&images, 10), images.to_vec());
    assert_eq!(images_within(&images, 9), images.to_vec());
    assert_eq!(images_within(&images, 8), vec![image(2, 2, 1)]);
    assert_eq!(images_within(&images, 4), vec![image(2, 2, 1)]);
    assert_eq!(images_within(&images, 3), vec![]);
    assert_eq!(images_within(TEXT, 3), vec![]);
}

#[test]
fn resume_rule_prefers_live_end_then_checkpoints() {
    let resume =
        |tokens: &[u32], checkpoints: &[usize], live_end: bool, prompt: &[u32]| {
            resume_position(tokens, TEXT, checkpoints, live_end, prompt, TEXT)
        };
    // Strict prefix: the live end, when resumable.
    assert_eq!(resume(&[1, 2, 3], &[2, 3], true, &[1, 2, 3, 4]), Some(3));
    // Live end not resumable (a disk entry without one): latest checkpoint.
    assert_eq!(resume(&[1, 2, 3], &[2], false, &[1, 2, 3, 4]), Some(2));
    // Divergence after 2: the checkpoint at 2 (not 3).
    assert_eq!(resume(&[1, 2, 3, 4], &[2, 3], true, &[1, 2, 9, 9]), Some(2));
    // Identical prompt: never the last token.
    assert_eq!(resume(&[1, 2, 3], &[2, 3], true, &[1, 2, 3]), Some(2));
    // Nothing shared, or nothing usable.
    assert_eq!(resume(&[5, 6], &[2], true, &[1, 2, 3]), None);
    assert_eq!(resume(&[1, 2, 3], &[3], true, &[1, 2, 9]), None);
}

#[test]
fn resume_rule_with_images_never_reuses_another_images_placeholders() {
    // A served prompt: 3 text tokens, a 4-placeholder image, a question;
    // its checkpoint sits at prompt_len - 1, and there is one at 2 too.
    let served = image_prompt(3, 4, &[10, 11, 12]);
    let ckpts = [2, served.len() - 1];
    let mine = [image(3, 4, 1)];
    // The same image and a longer question: a strict prefix, the live end.
    let extended = image_prompt(3, 4, &[10, 11, 12, 13, 14]);
    assert_eq!(
        resume_position(&served, &mine, &ckpts, true, &extended, &mine),
        Some(served.len())
    );
    // The same image, another question: the checkpoint before the question.
    let other_q = image_prompt(3, 4, &[10, 11, 20, 21]);
    assert_eq!(
        resume_position(&served, &mine, &ckpts, true, &other_q, &mine),
        Some(served.len() - 1)
    );
    // Another screenshot, identical tokens: only the checkpoint before the
    // image is usable, never one inside or after it.
    let theirs = [image(3, 4, 2)];
    assert_eq!(
        resume_position(&served, &mine, &ckpts, true, &served, &theirs),
        Some(2)
    );
    assert_eq!(
        resume_position(&served, &mine, &ckpts, true, &extended, &theirs),
        Some(2)
    );
    // Without a checkpoint before the image there is nothing to resume from.
    let late = [served.len() - 1];
    assert_eq!(resume_position(&served, &mine, &late, true, &extended, &theirs), None);
    // A text prompt whose tokens happen to spell the placeholders (which
    // the API refuses anyway) is not the image either.
    assert_eq!(resume_position(&served, &mine, &ckpts, true, &served, TEXT), Some(2));
    // A durable entry ending exactly at the image start is resumable there
    // for both screenshots: it holds no image.
    let durable = &served[..3];
    assert_eq!(resume_position(durable, TEXT, &[3], true, &served, &mine), Some(3));
    assert_eq!(resume_position(durable, TEXT, &[3], true, &served, &theirs), Some(3));
    // And one ending after the image is resumable only for the same one.
    let durable = &served[..8];
    assert_eq!(resume_position(durable, &mine, &[8], true, &served, &mine), Some(8));
    assert_eq!(resume_position(durable, &mine, &[8], true, &served, &theirs), None);
}

// --- the store, with a model whose state lives in host memory -----------------

mod store {
    use std::io::Read;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use anyhow::{Result, bail};

    use super::super::{Acquired, DiskCopy, Evictions, SessionStore};
    use crate::engine::{
        DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi, Segment,
        SnapshotApi,
    };
    use crate::metal::{EncodedPass, MetalContext};
    use crate::serve::disk::DiskStore;
    use crate::tensor::Tensor;

    /// Budget bytes per token of capacity: a 300-token session is 300 000.
    const BYTES_PER_TOKEN: usize = 1000;

    /// A decode state that keeps one byte per fed token (the token id, cut
    /// to a byte) and a rolling hash of them as its "recurrent" part, so a
    /// restored session can be checked for exactly what it held.
    struct State {
        data: Vec<u8>,
        recurrent: u64,
        capacity: usize,
    }

    struct Snap {
        pos: usize,
        recurrent: u64,
    }

    fn snap_bytes(pos: usize, recurrent: u64) -> Vec<u8> {
        let mut bytes = (pos as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&recurrent.to_le_bytes());
        bytes
    }

    impl SnapshotApi for Snap {
        fn pos(&self) -> usize {
            self.pos
        }
        fn bytes(&self) -> usize {
            0
        }
        fn layout(&self) -> Result<Vec<Segment>> {
            Ok(vec![Segment::Bytes(snap_bytes(self.pos, self.recurrent))])
        }
    }

    impl State {
        fn feed(&mut self, tokens: &[u32]) {
            for &t in tokens {
                self.data.push(t as u8);
                self.recurrent = self.recurrent.wrapping_mul(31).wrapping_add(t.into());
            }
            self.capacity = self.capacity.max(self.data.len());
        }
    }

    impl DecodeStateApi for State {
        type Snapshot = Snap;
        fn pos(&self) -> usize {
            self.data.len()
        }
        fn advance(&mut self, _: usize) {}
        fn reset(&mut self) -> Result<()> {
            self.data.clear();
            Ok(())
        }
        fn capacity(&self) -> usize {
            self.capacity
        }
        fn ensure_capacity(&mut self, _: &MetalContext, tokens: usize) -> Result<()> {
            self.capacity = self.capacity.max(tokens);
            Ok(())
        }
        fn bytes(&self) -> usize {
            self.capacity * BYTES_PER_TOKEN
        }
        fn snapshot(&self, _: &MetalContext) -> Result<Snap> {
            Ok(Snap { pos: self.data.len(), recurrent: self.recurrent })
        }
        fn restore(&mut self, _: &MetalContext, snapshot: &Snap) -> Result<()> {
            self.data.truncate(snapshot.pos);
            self.recurrent = snapshot.recurrent;
            Ok(())
        }
        fn copy_prefix_from(
            &mut self,
            _: &MetalContext,
            from: &Self,
            tokens: usize,
        ) -> Result<()> {
            self.data = from.data[..tokens].to_vec();
            Ok(())
        }
        fn prefix_layout(&self, tokens: usize) -> Result<Vec<Segment>> {
            Ok(vec![Segment::host(&self.data[..tokens])])
        }
        fn live_layout(&self) -> Result<Vec<Segment>> {
            Ok(vec![Segment::Bytes(snap_bytes(self.data.len(), self.recurrent))])
        }
        fn read_prefix(
            &mut self,
            _: &MetalContext,
            written: usize,
            tokens: usize,
            r: &mut dyn Read,
        ) -> Result<()> {
            let mut all = vec![0u8; written];
            r.read_exact(&mut all)?;
            self.data = all[..tokens].to_vec();
            self.capacity = self.capacity.max(tokens);
            Ok(())
        }
    }

    struct Scratch;

    impl ScratchApi for Scratch {
        fn next_token(&self) -> &Tensor {
            unreachable!("no decoding in these tests")
        }
        fn logits(&self) -> &Tensor {
            unreachable!("no decoding in these tests")
        }
        fn begin_request(&self) {}
    }

    struct Model;

    impl LanguageModel for Model {
        type State = State;
        type Scratch = Scratch;
        const MODEL_ID: &'static str = "fake";

        fn load(_: &MetalContext, _: &Path, _: &LoadOptions) -> Result<Self> {
            Ok(Self)
        }
        fn max_position_embeddings(&self) -> usize {
            0
        }
        fn eos_token_ids(&self) -> Vec<u32> {
            Vec::new()
        }
        fn vocab_size(&self) -> usize {
            256
        }
        fn bytes_per_token(&self) -> usize {
            BYTES_PER_TOKEN
        }
        fn read_snapshot(&self, _: &MetalContext, r: &mut dyn Read) -> Result<Snap> {
            let mut b = [0u8; 16];
            r.read_exact(&mut b)?;
            Ok(Snap {
                pos: u64::from_le_bytes(b[..8].try_into()?) as usize,
                recurrent: u64::from_le_bytes(b[8..].try_into()?),
            })
        }
        fn new_state(&self, _: &MetalContext, capacity: usize) -> Result<State> {
            Ok(State { data: Vec::new(), recurrent: 0, capacity })
        }
        fn new_scratch_with_capacity(
            &self,
            _: &MetalContext,
            _: usize,
        ) -> Result<Scratch> {
            Ok(Scratch)
        }
        fn prefill(
            &self,
            _: &MetalContext,
            _: &mut State,
            _: &mut Scratch,
            _: &[u32],
            _: Option<Draw<'_>>,
        ) -> Result<()> {
            bail!("not in these tests")
        }
        fn prepare_step_inputs(
            &self,
            _: &mut State,
            _: &Scratch,
            _: u32,
        ) -> Result<()> {
            bail!("not in these tests")
        }
        fn encode_decode_step<'a>(
            &self,
            _: &'a MetalContext,
            _: &State,
            _: &Scratch,
            _: usize,
            _: usize,
            _: Draw<'_>,
        ) -> Result<EncodedPass<'a>> {
            bail!("not in these tests")
        }
    }

    struct Fixture {
        ctx: MetalContext,
        store: SessionStore<Model>,
        root: PathBuf,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            self.store.drop_all();
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    /// A store of `budget` bytes over a fresh disk tier; the write ahead on
    /// with `floor` bytes when given.
    fn fixture(tag: &str, budget: usize, floor: Option<usize>) -> Fixture {
        let root = std::env::temp_dir()
            .join(format!("lily-session-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let disk = DiskStore::open(&root, "fake", 1 << 30, 0).expect("disk");
        let mut store = SessionStore::new(budget, 16, 3).with_disk(disk);
        if let Some(floor) = floor {
            store = store.with_write_ahead(floor);
        }
        Fixture { ctx: MetalContext::new().expect("metal"), store, root }
    }

    /// A prompt of `len` tokens whose ids start at `seed` (distinct seeds,
    /// distinct lineages).
    fn prompt(seed: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| seed * 1000 + i).collect()
    }

    /// What the engine does with a request: acquire, feed the rest of the
    /// prompt with a checkpoint at `n - 1`, release. Returns the acquire
    /// and what the release evicted.
    fn serve(f: &mut Fixture, prompt: &[u32]) -> (Acquired<Model>, Evictions) {
        let n = prompt.len();
        let mut acquired =
            f.store.acquire(&f.ctx, &Model, prompt, &[], None).expect("acquire");
        let reused = acquired.reused;
        let mut session = std::mem::replace(
            &mut acquired.session,
            super::super::Session::new(State {
                data: vec![],
                recurrent: 0,
                capacity: 0,
            }),
        );
        assert_eq!(session.state.pos(), reused);
        session.state.feed(&prompt[reused..n - 1]);
        let snapshot = session.state.snapshot(&f.ctx).expect("snapshot");
        session.add_checkpoint(snapshot);
        session.state.feed(&prompt[n - 1..]);
        session.tokens.truncate(reused);
        session.tokens.extend_from_slice(&prompt[reused..]);
        let released = f.store.release(&f.ctx, session, &[], None);
        (acquired, released)
    }

    /// Runs the write ahead until nothing is in flight; returns how many
    /// writes finished.
    fn settle(f: &mut Fixture) -> usize {
        let before = f.store.disk().map_or(0, DiskStore::len);
        while f.store.write_ahead(&f.ctx) {
            std::thread::sleep(Duration::from_millis(1));
        }
        f.store.disk().map_or(0, DiskStore::len) - before
    }

    fn copy_of(f: &Fixture, seed: u32) -> Option<DiskCopy> {
        f.store
            .entries
            .iter()
            .find(|s| s.tokens.first() == Some(&(seed * 1000)))
            .map(|s| s.disk.clone())
    }

    fn entry_dirs(f: &Fixture) -> usize {
        std::fs::read_dir(f.root.join("fake")).map_or(0, |d| d.count())
    }

    /// The state a resumed session holds must be exactly the one served.
    fn assert_holds(state: &State, tokens: &[u32]) {
        let mut expected = State { data: vec![], recurrent: 0, capacity: 0 };
        expected.feed(tokens);
        assert_eq!(state.data, expected.data);
        assert_eq!(state.recurrent, expected.recurrent);
    }

    #[test]
    fn a_session_written_ahead_is_evicted_without_a_write_and_resumes_exactly() {
        // Two 300-token sessions fit a 700 000-byte budget, a third does not.
        let mut f = fixture("ahead", 700_000, Some(300_000));
        let a = prompt(1, 300);
        serve(&mut f, &a);
        serve(&mut f, &prompt(2, 300));
        // 100 000 free, a new session needs 300 000: A (least recently used)
        // is written ahead, B (the one that just ran) is not.
        assert_eq!(settle(&mut f), 1);
        assert!(matches!(copy_of(&f, 1), Some(DiskCopy::Written(_))));
        assert_eq!(copy_of(&f, 2), Some(DiskCopy::None));
        // Both still resident: the write released nothing.
        assert_eq!((f.store.len(), f.store.used_bytes()), (2, 600_000));
        // The room now holds a new session: nothing more to write.
        assert!(!f.store.write_ahead(&f.ctx));
        assert_eq!(settle(&mut f), 0);

        let (acquired, _) = serve(&mut f, &prompt(3, 300));
        let e = acquired.evictions;
        assert_eq!((e.evicted, e.written_ahead, e.spilled, e.cancelled), (1, 1, 0, 0));
        assert!(f.store.used_bytes() <= 700_000);

        // A comes back from its copy with exactly what it held.
        let mut longer = a.clone();
        longer.extend(prompt(9, 20));
        let acquired = f.store.acquire(&f.ctx, &Model, &longer, &[], None).expect("a");
        assert_eq!(acquired.reused, 300);
        assert!(acquired.from_disk.is_some());
        assert_holds(&acquired.session.state, &a);
    }

    #[test]
    fn a_prompt_that_resumes_the_session_being_written_cancels_the_write() {
        let mut f = fixture("reclaim", 700_000, Some(300_000));
        let gate = Arc::new(AtomicBool::new(true));
        f.store.write_gate = Some(gate.clone());
        let a = prompt(1, 300);
        serve(&mut f, &a);
        serve(&mut f, &prompt(2, 300));
        assert!(f.store.write_ahead(&f.ctx), "the write of A is held in flight");
        // Parked: counted in the budget, invisible to lookups, nothing indexed.
        assert_eq!((f.store.len(), f.store.used_bytes()), (2, 600_000));
        assert_eq!(f.store.disk().expect("disk").len(), 0);

        let mut longer = a.clone();
        longer.extend(prompt(9, 20));
        let (acquired, _) = serve(&mut f, &longer);
        // The request got A itself, resident, at its live end ...
        assert_eq!((acquired.reused, acquired.forked), (300, false));
        assert!(acquired.from_disk.is_none());
        assert_eq!(acquired.evictions.cancelled, 1);
        // ... and the partly written entry is gone, never indexed.
        assert!(!f.store.writing());
        assert_eq!(f.store.disk().expect("disk").len(), 0);
        assert_eq!(entry_dirs(&f), 0);
        gate.store(false, Ordering::Release);
    }

    #[test]
    fn an_eviction_that_reaches_the_write_in_flight_waits_for_it() {
        let mut f = fixture("wait", 700_000, Some(300_000));
        let gate = Arc::new(AtomicBool::new(true));
        f.store.write_gate = Some(gate.clone());
        let a = prompt(1, 300);
        serve(&mut f, &a);
        serve(&mut f, &prompt(2, 300));
        assert!(f.store.write_ahead(&f.ctx));
        let opener = {
            let gate = gate.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                gate.store(false, Ordering::Release);
            })
        };
        // A new session needs A's room: the eviction waits for A's write and
        // then drops it, no second write.
        let acquired = f
            .store
            .acquire(&f.ctx, &Model, &prompt(3, 300), &[], None)
            .expect("acquire");
        opener.join().expect("opener");
        let e = acquired.evictions;
        assert_eq!((e.evicted, e.written_ahead, e.spilled), (1, 1, 0));
        assert!(e.waited_secs >= 0.02, "{e:?}");
        // The bound: what stays resident plus the new session fits.
        assert_eq!(f.store.used_bytes() + acquired.session.bytes(), 600_000);
        let disk = f.store.disk().expect("disk");
        assert_eq!(disk.len(), 1);
        assert_eq!(disk.entries()[0].tokens, a);
    }

    #[test]
    fn without_a_copy_on_disk_the_eviction_writes_the_session_on_the_spot() {
        // The fallback: the write ahead is off, or had no time.
        let mut f = fixture("fallback", 700_000, None);
        let a = prompt(1, 300);
        serve(&mut f, &a);
        serve(&mut f, &prompt(2, 300));
        assert!(!f.store.write_ahead(&f.ctx), "off");
        let (acquired, _) = serve(&mut f, &prompt(3, 300));
        let e = acquired.evictions;
        assert_eq!((e.evicted, e.written_ahead, e.spilled), (1, 0, 1));
        assert!(e.spill_secs > 0.0);
        let mut longer = a.clone();
        longer.push(5);
        let acquired = f.store.acquire(&f.ctx, &Model, &longer, &[], None).expect("a");
        assert_eq!(acquired.reused, 300);
        assert_holds(&acquired.session.state, &a);
    }

    #[test]
    fn a_copy_the_tier_deleted_is_written_again_or_spilled() {
        let mut f = fixture("vanished", 700_000, Some(300_000));
        serve(&mut f, &prompt(1, 300));
        serve(&mut f, &prompt(2, 300));
        assert_eq!(settle(&mut f), 1);
        let Some(DiskCopy::Written(id)) = copy_of(&f, 1) else { panic!("written") };
        // The budget or the age limit deletes the entry behind the store's back.
        f.store.disk.as_mut().expect("disk").remove(&id);
        // An eviction now does not trust the copy: it spills.
        let (acquired, _) = serve(&mut f, &prompt(3, 300));
        let e = acquired.evictions;
        assert_eq!((e.evicted, e.written_ahead, e.spilled), (1, 0, 1));
        // And between requests the write ahead notices a deleted copy too.
        assert_eq!(settle(&mut f), 1);
        let Some(DiskCopy::Written(id)) = copy_of(&f, 2) else {
            panic!("B written ahead")
        };
        f.store.disk.as_mut().expect("disk").remove(&id);
        assert_eq!(settle(&mut f), 1, "written again");
    }

    #[test]
    fn resuming_a_session_that_was_written_ahead_deletes_the_stale_copy() {
        let mut f = fixture("stale", 700_000, Some(300_000));
        let a = prompt(1, 300);
        serve(&mut f, &a);
        serve(&mut f, &prompt(2, 300));
        assert_eq!(settle(&mut f), 1);
        let mut longer = a.clone();
        longer.extend(prompt(9, 20));
        let (acquired, _) = serve(&mut f, &longer);
        // Resumed resident (ties go to the GPU), not from the copy.
        assert_eq!(acquired.reused, 300);
        assert!(acquired.from_disk.is_none());
        // The copy held A as it was; after the request it is stale and gone.
        assert_eq!(f.store.disk().expect("disk").len(), 0);
        assert_eq!(copy_of(&f, 1), Some(DiskCopy::None));
    }

    #[test]
    fn nothing_is_written_ahead_while_the_room_suffices_nor_the_latest_session() {
        let mut f = fixture("quiet", 2_000_000, Some(300_000));
        serve(&mut f, &prompt(1, 300));
        serve(&mut f, &prompt(2, 300));
        // 1 400 000 free: room enough.
        assert_eq!(settle(&mut f), 0);
        // A lone session filling the budget: it is the one that just ran.
        let mut g = fixture("lone", 300_000, Some(300_000));
        serve(&mut g, &prompt(1, 300));
        assert_eq!(settle(&mut g), 0);
        // Sessions too short for the disk tier are dropped anyway and never
        // written.
        let mut h = fixture("short", 250_000, Some(300_000));
        serve(&mut h, &prompt(1, 100));
        serve(&mut h, &prompt(2, 100));
        assert_eq!(settle(&mut h), 0);
    }

    #[test]
    fn the_headroom_follows_the_largest_recent_new_session() {
        // A 600-token session was created: the next new one is expected to
        // be as large, so two sessions are written ahead, not one.
        let mut f = fixture("headroom", 1_300_000, Some(100_000));
        serve(&mut f, &prompt(1, 300));
        serve(&mut f, &prompt(2, 300));
        serve(&mut f, &prompt(3, 600));
        // 100 000 free; 600 000 wanted: A and B, in eviction order.
        assert_eq!(settle(&mut f), 2);
        assert!(matches!(copy_of(&f, 1), Some(DiskCopy::Written(_))));
        assert!(matches!(copy_of(&f, 2), Some(DiskCopy::Written(_))));
        assert_eq!(copy_of(&f, 3), Some(DiskCopy::None));
    }

    #[test]
    fn a_fault_drops_the_write_in_flight_and_an_unload_finishes_it() {
        let mut f = fixture("drop", 700_000, Some(300_000));
        let gate = Arc::new(AtomicBool::new(true));
        f.store.write_gate = Some(gate.clone());
        serve(&mut f, &prompt(1, 300));
        serve(&mut f, &prompt(2, 300));
        assert!(f.store.write_ahead(&f.ctx));
        assert_eq!(f.store.drop_all(), 2);
        assert!(!f.store.writing());
        assert_eq!(f.store.disk().expect("disk").len(), 0);
        assert_eq!(entry_dirs(&f), 0);

        // The unload: the write in flight completes, the other session spills.
        serve(&mut f, &prompt(3, 300));
        serve(&mut f, &prompt(4, 300));
        assert!(f.store.write_ahead(&f.ctx));
        gate.store(false, Ordering::Release);
        assert_eq!(f.store.spill_all(&f.ctx), (2, 0));
        assert!(f.store.is_empty());
        assert_eq!(f.store.disk().expect("disk").len(), 2);
    }

    #[test]
    fn a_write_ahead_writes_what_a_synchronous_spill_writes() {
        let read = |root: &Path, id: &str| -> Vec<Vec<u8>> {
            let dir = root.join("fake").join(id);
            let mut files: Vec<_> = std::fs::read_dir(&dir)
                .expect("dir")
                .map(|e| e.expect("entry").path())
                .filter(|p| !p.ends_with("meta.json"))
                .collect();
            files.sort();
            files.iter().map(|p| std::fs::read(p).expect("read")).collect()
        };
        let mut ahead = fixture("same-ahead", 700_000, Some(300_000));
        let mut sync = fixture("same-sync", 700_000, None);
        for f in [&mut ahead, &mut sync] {
            serve(f, &prompt(1, 300));
            serve(f, &prompt(2, 300));
            settle(f);
            serve(f, &prompt(3, 300));
        }
        let id = |f: &Fixture| f.store.disk().expect("disk").entries()[0].id.clone();
        assert_eq!(read(&ahead.root, &id(&ahead)), read(&sync.root, &id(&sync)));
        let mut out = Vec::new();
        std::fs::File::open(
            ahead.root.join("fake").join(id(&ahead)).join("prefix.bin"),
        )
        .expect("prefix")
        .read_to_end(&mut out)
        .expect("read");
        assert_eq!(out, prompt(1, 300).iter().map(|&t| t as u8).collect::<Vec<_>>());
    }

    /// What the engine does with a request whose prefill a cancellation
    /// stopped at `at`: acquire, feed up to there, keep that prefix, release.
    fn serve_stopped(f: &mut Fixture, prompt: &[u32], at: usize) -> Acquired<Model> {
        let mut acquired =
            f.store.acquire(&f.ctx, &Model, prompt, &[], None).expect("acquire");
        let reused = acquired.reused;
        let mut session = std::mem::replace(
            &mut acquired.session,
            super::super::Session::new(State {
                data: vec![],
                recurrent: 0,
                capacity: 0,
            }),
        );
        session.state.feed(&prompt[reused..at]);
        // The position must be the state's: anything else is refused.
        assert!(session.stop_at(prompt, reused, at + 1).is_err());
        session.stop_at(prompt, reused, at).expect("stop");
        f.store.release(&f.ctx, session, &[], None);
        acquired
    }

    #[test]
    fn a_prefill_stopped_at_a_chunk_boundary_is_kept_and_its_retry_resumes_there() {
        // 1 000 bytes a token; the write ahead keeps room for a new
        // 300-token session free of writes.
        let mut f = fixture("stopped", 900_000, Some(300_000));
        let a = prompt(1, 300);
        serve(&mut f, &a);
        // The next turn extends A in place, and its client leaves after the
        // prefill reached 400 of 700 tokens.
        let mut turn = a.clone();
        turn.extend(prompt(2, 400));
        let acquired = serve_stopped(&mut f, &turn, 400);
        assert_eq!((acquired.reused, acquired.forked), (300, false));
        assert_eq!(f.store.len(), 1);
        // Kept like any request's session: it just ran, so it is not
        // written ahead (nothing else is resident to write either).
        assert_eq!(settle(&mut f), 0);

        // A different conversation fills the budget; the stopped session is
        // now the least recently used and is written ahead like any other.
        serve(&mut f, &prompt(3, 300));
        assert!(f.store.used_bytes() <= 900_000);
        assert_eq!(settle(&mut f), 1);
        let kept = f.store.entries.iter().find(|s| s.tokens.first() == Some(&1000));
        let kept = kept.expect("the stopped session is still resident");
        assert_eq!(kept.tokens, turn[..400]);
        assert!(matches!(kept.disk, DiskCopy::Written(_)));

        // The retry resumes in place at the boundary with exactly the state
        // the stopped prefill left, and carries on to the end as usual.
        let (retry, _) = serve(&mut f, &turn);
        assert_eq!((retry.reused, retry.forked, retry.from_disk), (400, false, None));
        let resumed = f.store.entries.iter().find(|s| s.tokens.first() == Some(&1000));
        assert_holds(&resumed.expect("resumed").state, &turn);

        // The checkpoint of the first turn still serves a rollback below the
        // stopped position: a regenerated first answer forks from 299.
        let mut regenerate = a.clone();
        regenerate.push(4242);
        let (fork, _) = serve(&mut f, &regenerate);
        assert_eq!((fork.reused, fork.forked), (299, true));
    }

    #[test]
    fn a_prefill_stopped_before_its_first_chunk_keeps_what_it_was_acquired_with() {
        let mut f = fixture("stopped-early", 1_000_000, None);
        let a = prompt(1, 300);
        serve(&mut f, &a);
        let mut turn = a.clone();
        turn.extend(prompt(2, 100));
        let acquired = serve_stopped(&mut f, &turn, 300);
        assert_eq!(acquired.reused, 300);
        assert_eq!(f.store.len(), 1);
        let (retry, _) = serve(&mut f, &turn);
        assert_eq!((retry.reused, retry.forked), (300, false));
    }
}
