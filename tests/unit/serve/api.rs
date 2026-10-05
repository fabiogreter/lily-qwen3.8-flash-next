use super::*;

#[test]
fn usage_reports_reasoning_tokens_for_chat() {
    let u = usage(100, 40, 30, Some(12), None);
    assert_eq!(u["prompt_tokens"], 100);
    assert_eq!(u["completion_tokens"], 30);
    assert_eq!(u["total_tokens"], 130);
    assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 40);
    assert_eq!(
        u["completion_tokens_details"],
        serde_json::json!({"reasoning_tokens": 12})
    );
    // No reasoning generated still says so rather than leaving the field out.
    let u = usage(100, 0, 5, Some(0), None);
    assert_eq!(u["completion_tokens_details"]["reasoning_tokens"], 0);
}

#[test]
fn usage_keeps_the_prediction_counts_next_to_reasoning() {
    let u = usage(10, 0, 20, Some(7), Some(Speculation { drafted: 9, accepted: 6 }));
    assert_eq!(
        u["completion_tokens_details"],
        serde_json::json!({
            "reasoning_tokens": 7,
            "accepted_prediction_tokens": 6,
            "rejected_prediction_tokens": 3,
        })
    );
    // Text completions have no reasoning block: only the draft counts.
    let u = usage(10, 0, 20, None, Some(Speculation { drafted: 4, accepted: 4 }));
    assert_eq!(
        u["completion_tokens_details"],
        serde_json::json!({"accepted_prediction_tokens": 4, "rejected_prediction_tokens": 0})
    );
    let u = usage(10, 0, 20, None, None);
    assert!(u.get("completion_tokens_details").is_none());
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for (j, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            out.push(if j <= chunk.len() {
                ALPHABET[(n >> shift & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    out
}

/// A 300 x 200 RGB PNG with a gradient seeded by `seed`, as a data URI.
fn png_data_uri(seed: u8) -> String {
    let (w, h) = (300u32, 200u32);
    let data: Vec<u8> = (0..w * h * 3).map(|i| (i as u8).wrapping_mul(seed)).collect();
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&data).unwrap();
    }
    format!("data:image/png;base64,{}", base64(&out))
}

fn identity(n: u8) -> ImageIdentity {
    ImageIdentity { grid_h: n as usize, grid_w: 2, digest: [n; 32] }
}

#[test]
fn the_image_memo_remembers_by_uri_and_drops_the_oldest_past_capacity() {
    let memo = ImageMemo::default();
    let a = ImageMemo::key("data:image/png;base64,AA==");
    assert_ne!(a, ImageMemo::key("data:image/png;base64,AQ=="));
    assert_eq!(memo.get(&a), None);
    memo.insert(a, identity(1));
    assert_eq!(memo.get(&a), Some(identity(1)));
    // Inserting a known key again neither grows the memo nor its order.
    memo.insert(a, identity(1));
    assert_eq!(memo.len(), 1);
    for n in 0..ImageMemo::CAPACITY {
        memo.insert(ImageMemo::key(&n.to_string()), identity(2));
    }
    assert_eq!(memo.len(), ImageMemo::CAPACITY);
    assert_eq!(memo.get(&a), None, "the oldest entry went first");
    assert_eq!(memo.get(&ImageMemo::key("0")), Some(identity(2)));
}

#[test]
fn a_remembered_image_is_not_decoded_and_its_deferred_rows_match() {
    let policy = ImagePolicy {
        limits: ImageLimits::default(),
        available: Ok(()),
        memo: Arc::default(),
    };
    let url = png_data_uri(7);
    let (first, id1) = prepare_image(url.clone(), 1, &policy).unwrap();
    let ImagePixels::Ready(rows) = &first else { panic!("a new image is decoded") };
    let (second, id2) = prepare_image(url.clone(), 1, &policy).unwrap();
    assert!(
        matches!(second, ImagePixels::Deferred { .. }),
        "a known image is not decoded"
    );
    assert_eq!(id1, id2);
    let image = |pixels, id: ImageIdentity| PreparedImage {
        pixels,
        span: ImageSpan { start: 0, len: 0, grid_h: id.grid_h, grid_w: id.grid_w },
        digest: id.digest,
    };
    let deferred = image(second, id2);
    assert_eq!(deferred.rows().unwrap().as_ref(), rows.as_slice());
    // A deferred image that preprocesses to another identity is refused.
    let wrong = image(
        ImagePixels::Deferred { url, limits: policy.limits },
        ImageIdentity { digest: [0; 32], ..id1 },
    );
    let err = wrong.rows().unwrap_err().to_string();
    assert!(err.contains("preprocessed differently"), "{err}");
    // Another image is decoded again.
    let (third, _) = prepare_image(png_data_uri(9), 2, &policy).unwrap();
    assert!(matches!(third, ImagePixels::Ready(_)));
    assert_eq!(policy.memo.len(), 2);
}
