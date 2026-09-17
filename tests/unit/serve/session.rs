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
