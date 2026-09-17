//! Image preprocessing against the committed reference goldens
//! (`docs/architecture.md`, "How the tower was verified", comparison 1) and PIL's behaviour: the
//! four test images and the five `hf_vision_preprocess_*.json` goldens run
//! without Python or a GPU, the seeded 4 096-element sample judged with
//! `PREPROCESS_TOLERANCE` and, stricter than the tolerance, counted by how
//! many uint8 levels each element moved. The exact-parity check against
//! PIL's own `Image.resize` output reads raw dumps from
//! `LILY_PIL_RESIZE_DIR` when that is set (tools/reference/VISION.md names
//! the Pillow version) and says so when it is not.

use std::path::{Path, PathBuf};
use std::time::Instant;

use super::*;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

// ---------------------------------------------------------------------------
// Synthetic images

/// An RGB image whose red channel is the column, green the row and blue the
/// sum modulo 256, so a pixel's value names its coordinates.
fn coordinate_image(width: usize, height: usize) -> RgbImage {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            data.extend([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]);
        }
    }
    RgbImage::new(width, height, data).unwrap()
}

fn encode_png(
    width: u32,
    height: u32,
    color: png::ColorType,
    depth: png::BitDepth,
    palette: Option<Vec<u8>>,
    data: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, width, height);
        enc.set_color(color);
        enc.set_depth(depth);
        if let Some(p) = palette {
            enc.set_palette(p);
        }
        let mut writer = enc.write_header().unwrap();
        writer.write_image_data(data).unwrap();
    }
    out
}

/// A PNG signature and IHDR claiming `width x height`, nothing after it.
fn png_ihdr_only(width: u32, height: u32) -> Vec<u8> {
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    out.extend(13u32.to_be_bytes());
    out.extend(b"IHDR");
    out.extend(width.to_be_bytes());
    out.extend(height.to_be_bytes());
    out.extend([8, 2, 0, 0, 0]);
    out.extend([0, 0, 0, 0]); // crc, not checked by the sniffer
    out
}

/// A JPEG start-of-image and a baseline frame header claiming
/// `width x height`, nothing after it.
fn jpeg_sof_only(width: u16, height: u16) -> Vec<u8> {
    let mut out = vec![0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 8];
    out.extend(height.to_be_bytes());
    out.extend(width.to_be_bytes());
    out.push(3);
    out.extend([1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
    out
}

// ---------------------------------------------------------------------------
// Decoding

#[test]
fn rgba_png_drops_alpha_without_compositing() {
    let (w, h) = (5u32, 4u32);
    let mut rgba = Vec::new();
    for i in 0..(w * h) as usize {
        rgba.extend([
            (i * 7 % 256) as u8,
            (i * 13 % 256) as u8,
            (i * 29 % 256) as u8,
            (i * 61 % 256) as u8,
        ]);
    }
    let png = encode_png(w, h, png::ColorType::Rgba, png::BitDepth::Eight, None, &rgba);
    let img = decode_image(&png, &ImageLimits::default()).unwrap();
    assert_eq!((img.width, img.height), (5, 4));
    let expect: Vec<u8> =
        rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    assert_eq!(img.data, expect, "alpha must be dropped, not composited");
}

#[test]
fn greyscale_png_is_replicated_to_rgb() {
    let grey: Vec<u8> = (0..24u8).map(|v| v * 10).collect();
    let png =
        encode_png(6, 4, png::ColorType::Grayscale, png::BitDepth::Eight, None, &grey);
    let img = decode_image(&png, &ImageLimits::default()).unwrap();
    let expect: Vec<u8> = grey.iter().flat_map(|&g| [g, g, g]).collect();
    assert_eq!(img.data, expect);

    // Greyscale with alpha drops the alpha as well.
    let ga: Vec<u8> = grey.iter().flat_map(|&g| [g, 255 - g]).collect();
    let png = encode_png(
        6,
        4,
        png::ColorType::GrayscaleAlpha,
        png::BitDepth::Eight,
        None,
        &ga,
    );
    assert_eq!(decode_image(&png, &ImageLimits::default()).unwrap().data, expect);
}

#[test]
fn palette_png_is_looked_up() {
    let palette = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 200, 210, 220];
    let indices = vec![0u8, 1, 2, 3, 3, 2, 1, 0];
    let png = encode_png(
        4,
        2,
        png::ColorType::Indexed,
        png::BitDepth::Eight,
        Some(palette.clone()),
        &indices,
    );
    let img = decode_image(&png, &ImageLimits::default()).unwrap();
    let expect: Vec<u8> =
        indices.iter().flat_map(|&i| palette[i as usize * 3..][..3].to_vec()).collect();
    assert_eq!(img.data, expect);
}

#[test]
fn sixteen_bit_png_keeps_the_high_byte() {
    // RGB 16: PIL's RGB;16B unpacker takes the high byte, and so does this.
    let samples: Vec<u16> = (0..12).map(|i| (i as u16) * 5000 + 123).collect();
    let be: Vec<u8> = samples.iter().flat_map(|v| v.to_be_bytes()).collect();
    let png = encode_png(2, 2, png::ColorType::Rgb, png::BitDepth::Sixteen, None, &be);
    let img = decode_image(&png, &ImageLimits::default()).unwrap();
    let expect: Vec<u8> = samples.iter().map(|&v| (v >> 8) as u8).collect();
    assert_eq!(img.data, expect);

    // Greyscale 16: the high byte here; PIL 12.3.0 saturates at 255 instead
    // (I;16 -> RGB goes through a clamp), which turns the image white. That
    // departure is deliberate and documented in docs/architecture.md.
    let grey: Vec<u16> = vec![0, 255, 256, 30000, 65535, 40000];
    let be: Vec<u8> = grey.iter().flat_map(|v| v.to_be_bytes()).collect();
    let png =
        encode_png(3, 2, png::ColorType::Grayscale, png::BitDepth::Sixteen, None, &be);
    let img = decode_image(&png, &ImageLimits::default()).unwrap();
    let expect: Vec<u8> = grey.iter().flat_map(|&v| [(v >> 8) as u8; 3]).collect();
    assert_eq!(img.data, expect);
}

/// The two JPEG fixtures were written by Pillow 12.3.0 (`tests/goldens/
/// jpeg_*.jpg`) together with what Pillow decodes them to (`*.pil.rgb`, raw
/// RGB rows). libjpeg-turbo and zune-jpeg are not the same decoder, so the
/// check is a bound on the residual, not an exact match; the counts are
/// printed for the record.
fn jpeg_fixture(stem: &str, width: usize, height: usize) -> (RgbImage, Vec<u8>) {
    let dir = repo().join("tests/goldens");
    let bytes = std::fs::read(dir.join(format!("{stem}.jpg"))).unwrap();
    let header = read_header(&bytes).unwrap();
    assert_eq!(header.format, ImageFormat::Jpeg);
    assert_eq!((header.width as usize, header.height as usize), (width, height));
    let img = decode_image(&bytes, &ImageLimits::default()).unwrap();
    assert_eq!((img.width, img.height), (width, height));
    let pil = std::fs::read(dir.join(format!("{stem}.pil.rgb"))).unwrap();
    assert_eq!(pil.len(), width * height * 3);
    (img, pil)
}

fn level_histogram(a: &[u8], b: &[u8]) -> [usize; 3] {
    let mut counts = [0usize; 3];
    for (&x, &y) in a.iter().zip(b) {
        counts[(i32::from(x) - i32::from(y)).unsigned_abs().min(2) as usize] += 1;
    }
    counts
}

#[test]
fn jpeg_decodes_close_to_pillow() {
    // Measured residuals of zune-jpeg against libjpeg-turbo (Pillow 12.3.0):
    // the greyscale fixture differs on about 1 % of the samples by one level
    // (IDCT rounding); the 4:2:0 colour fixture on 16 % by one level and
    // 10 % by two or three (chroma upsampling), 2 % and 1 % on a 640 x 480
    // photo-like image. The bounds below are those residuals with room; an
    // exact match is not expected from a different decoder.
    for (stem, w, h, max_levels, min_exact_percent) in
        [("jpeg_96x64_q90", 96, 64, 3, 70), ("jpeg_gray_48x32_q90", 48, 32, 1, 98)]
    {
        let (img, pil) = jpeg_fixture(stem, w, h);
        let counts = level_histogram(&img.data, &pil);
        let max = img
            .data
            .iter()
            .zip(&pil)
            .map(|(&x, &y)| (i32::from(x) - i32::from(y)).abs())
            .max()
            .unwrap();
        println!(
            "{stem}: {} exact, {} off by one level, {} by more; max {max}",
            counts[0], counts[1], counts[2]
        );
        assert!(
            max <= max_levels,
            "{stem}: zune-jpeg differs from libjpeg by {max} levels"
        );
        assert!(
            counts[0] * 100 >= img.data.len() * min_exact_percent,
            "{stem}: only {} of {} samples match Pillow exactly",
            counts[0],
            img.data.len()
        );
        if stem.contains("gray") {
            for px in img.data.chunks_exact(3) {
                assert!(
                    px[0] == px[1] && px[1] == px[2],
                    "greyscale JPEG must replicate to RGB"
                );
            }
        }
    }
}

#[test]
fn other_formats_are_refused_by_name() {
    let cases: [(&[u8], &str); 5] = [
        (b"GIF89a\x01\x00\x01\x00", "GIF"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "WebP"),
        (b"BM\x00\x00\x00\x00", "BMP"),
        (b"\x00\x00\x00\x18ftypavif", "HEIF/AVIF"),
        (b"", "empty data"),
    ];
    for (bytes, name) in cases {
        let err = decode_image(bytes, &ImageLimits::default()).unwrap_err().to_string();
        assert!(err.contains(name) && err.contains("unsupported"), "{name}: {err}");
    }
}

#[test]
fn oversized_headers_are_refused_before_decoding() {
    let lim = ImageLimits::default();
    // 100 000 x 100 000: over the side limit and the pixel limit, judged
    // from the 33 header bytes alone.
    let err =
        decode_image(&png_ihdr_only(100_000, 100_000), &lim).unwrap_err().to_string();
    assert!(err.contains("exceeds the limit of 16384"), "{err}");
    // A wide strip under the side limit but over 64 megapixels.
    let err =
        decode_image(&png_ihdr_only(16_384, 8_192), &lim).unwrap_err().to_string();
    assert!(err.contains("over the limit of 67108864"), "{err}");
    // Zero-sized.
    let err = decode_image(&png_ihdr_only(0, 10), &lim).unwrap_err().to_string();
    assert!(err.contains("zero-sized"), "{err}");
    // JPEG frame header claiming the maximum a JPEG can, 65535 squared.
    let err =
        decode_image(&jpeg_sof_only(65_535, 65_535), &lim).unwrap_err().to_string();
    assert!(err.contains("exceeds the limit of 16384"), "{err}");
    // The decoded-size bound counts channels and depth: 16-bit RGBA at the
    // pixel cap is admitted, one byte more per pixel is not.
    let mut hdr = ImageHeader {
        format: ImageFormat::Png,
        width: 8192,
        height: 8192,
        channels: 4,
        bit_depth: 16,
    };
    assert!(check_header(&hdr, &lim).is_ok());
    hdr.channels = 4;
    check_header(&hdr, &ImageLimits { max_decoded_bytes: 8192 * 8192 * 8 - 1, ..lim })
        .unwrap_err();
    // Nothing above reached a decoder; a header that passes and lies about
    // its size is caught by the decoder's own consistency check.
    let mut lying = png_ihdr_only(4, 4);
    lying.truncate(lying.len() - 4);
    assert!(decode_image(&lying, &lim).is_err());
}

/// Deterministic noise (a 32-bit LCG), so the PNG does not compress.
fn noise(len: usize) -> Vec<u8> {
    let mut state = 0x2545_F491u32;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
}

#[test]
fn truncated_png_is_an_error_not_a_panic() {
    let pixels = noise(64 * 64 * 3);
    let full =
        encode_png(64, 64, png::ColorType::Rgb, png::BitDepth::Eight, None, &pixels);
    assert!(full.len() > 10_000, "noise should not compress: {} bytes", full.len());
    // Whole and untruncated it decodes to the pixels written.
    assert_eq!(decode_image(&full, &ImageLimits::default()).unwrap().data, pixels);
    for cut in [0, 8, 20, 33, 40, 100, full.len() / 4, full.len() / 2, full.len() - 40]
    {
        let err = decode_image(&full[..cut], &ImageLimits::default());
        assert!(err.is_err(), "a PNG cut at byte {cut} of {} decoded", full.len());
    }
    let bytes = std::fs::read(repo().join("tests/goldens/jpeg_96x64_q90.jpg")).unwrap();
    for cut in [2, 4, 100, bytes.len() / 2] {
        assert!(
            decode_image(&bytes[..cut], &ImageLimits::default()).is_err(),
            "a JPEG cut at byte {cut} decoded"
        );
    }
}

#[test]
fn header_reads_dimensions_of_the_test_images() {
    let images: serde_json::Value = serde_json::from_slice(
        &std::fs::read(repo().join("tools/reference/images/images.json")).unwrap(),
    )
    .unwrap();
    for (_, entry) in images.as_object().unwrap() {
        let bytes = std::fs::read(
            repo().join("tools/reference/images").join(entry["file"].as_str().unwrap()),
        )
        .unwrap();
        let header = read_header(&bytes).unwrap();
        assert_eq!(header.format, ImageFormat::Png);
        assert_eq!(u64::from(header.width), entry["width"].as_u64().unwrap());
        assert_eq!(u64::from(header.height), entry["height"].as_u64().unwrap());
        assert_eq!((header.channels, header.bit_depth), (3, 8));
    }
}

// ---------------------------------------------------------------------------
// smart_resize

#[test]
fn smart_resize_matches_the_reference_table() {
    let (min, max) = (DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS);
    // VISION.md "The pixel cap", (height, width) -> (h_bar, w_bar).
    assert_eq!(smart_resize(777, 333, min, max).unwrap(), (768, 320));
    assert_eq!(smart_resize(480, 640, min, max).unwrap(), (480, 640));
    assert_eq!(smart_resize(1080, 1920, min, max).unwrap(), (1088, 1920));
    assert_eq!(smart_resize(2160, 3840, min, max).unwrap(), (1056, 1920));
    // The checkpoint's own cap leaves the Retina capture alone.
    assert_eq!(smart_resize(2160, 3840, min, 16_777_216).unwrap(), (2176, 3840));
    // The min branch, values from `hf_vision_reference.smart_resize_reference`.
    assert_eq!(smart_resize(100, 100, min, max).unwrap(), (256, 256));
    assert_eq!(smart_resize(50, 300, min, max).unwrap(), (128, 640));
    assert_eq!(smart_resize(1, 1, min, max).unwrap(), (256, 256));
    assert_eq!(smart_resize(300, 200, min, max).unwrap(), (320, 224));
    // Rounding half to even: 48 / 32 = 1.5 rounds to 2, 16 / 32 = 0.5 to 0,
    // both then lifted by the min branch to 256.
    assert_eq!(smart_resize(48, 48, min, max).unwrap(), (256, 256));
    assert_eq!(smart_resize(16, 16, min, max).unwrap(), (256, 256));
    // Half to even where it decides the answer: 80 / 32 = 2.5 rounds to 2
    // (64), 112 / 32 = 3.5 rounds to 4 (128); with a small min cap nothing
    // lifts them.
    assert_eq!(smart_resize(80, 112, 1, max).unwrap(), (64, 128));
    assert!(
        smart_resize(10, 3000, min, max)
            .unwrap_err()
            .to_string()
            .contains("aspect ratio")
    );
    assert!(smart_resize(0, 10, min, max).is_err());
}

// ---------------------------------------------------------------------------
// Resampling

#[test]
fn identity_resize_copies() {
    let img = coordinate_image(64, 32);
    assert_eq!(resize_bicubic(&img, 64, 32), img);
}

#[test]
fn flat_images_stay_flat_under_resampling() {
    // The normalised kernel sums to one in fixed point too, so a flat colour
    // must come out unchanged whether the axis shrinks or grows.
    for (w, h, tw, th) in [(100, 70, 32, 32), (30, 20, 96, 64), (333, 777, 320, 768)] {
        let img = RgbImage::new(w, h, [200u8, 17, 90].repeat(w * h)).unwrap();
        let out = resize_bicubic(&img, tw, th);
        assert_eq!((out.width, out.height), (tw, th));
        assert!(
            out.data.chunks_exact(3).all(|p| p == [200, 17, 90]),
            "{w}x{h} -> {tw}x{th}"
        );
    }
}

/// Exact parity with `PIL.Image.resize(..., BICUBIC)`: raw RGB dumps named
/// `pil_<stem>_<w>x<h>.rgb` in `LILY_PIL_RESIZE_DIR`, the source being
/// `<stem>.png` in that directory or, failing that, the test image of that
/// name in `tools/reference/images/` (`_uncapped` stripped).
#[test]
fn resize_matches_pillow_dumps_when_present() {
    let Some(dir) = std::env::var_os("LILY_PIL_RESIZE_DIR") else {
        println!("LILY_PIL_RESIZE_DIR unset: PIL parity dumps not checked");
        return;
    };
    let dir = PathBuf::from(dir);
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("pil_").and_then(|s| s.strip_suffix(".rgb"))
        else {
            continue;
        };
        // `pil_<stem>_<w>x<h>.rgb`; other dumps in the directory are skipped.
        let Some((stem, size)) = rest.rsplit_once('_') else { continue };
        let Some((tw, th)) = size.split_once('x') else { continue };
        let (Ok(tw), Ok(th)) = (tw.parse::<usize>(), th.parse::<usize>()) else {
            continue;
        };
        let src = [
            dir.join(format!("{stem}.png")),
            repo()
                .join("tools/reference/images")
                .join(format!("{}.png", stem.trim_end_matches("_uncapped"))),
        ]
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| panic!("no source image for {name}"));
        let img = decode_image(&std::fs::read(&src).unwrap(), &ImageLimits::default())
            .unwrap();
        let out = resize_bicubic(&img, tw, th);
        let pil = std::fs::read(&path).unwrap();
        assert_eq!(pil.len(), out.data.len(), "{name}");
        let counts = level_histogram(&out.data, &pil);
        println!(
            "{name}: {} exact, {} off by one, {} by more",
            counts[0], counts[1], counts[2]
        );
        assert_eq!(out.data, pil, "{name} differs from PIL");
        checked += 1;
    }
    assert!(checked > 0, "no pil_*.rgb dumps in {}", dir.display());
}

// ---------------------------------------------------------------------------
// Patchify

#[test]
fn patch_rows_are_block_major_and_channel_major() {
    // 64 x 32 pixels: grid (2, 4), two merge blocks side by side.
    let img = coordinate_image(64, 32);
    let pv = patchify(&img).unwrap();
    assert_eq!((pv.grid_h, pv.grid_w), (2, 4));
    assert_eq!(pv.resized, (64, 32));
    assert_eq!(pv.patches(), 8);
    assert_eq!(pv.tokens(), 2);
    assert_eq!(pv.data.len(), 8 * PATCH_DIM);
    let gw = pv.grid_w;
    for bh in 0..1 {
        for bw in 0..2 {
            for ih in 0..2 {
                for iw in 0..2 {
                    let r = ((bh * (gw / 2) + bw) * 2 + ih) * 2 + iw;
                    let row = &pv.data[r * PATCH_DIM..][..PATCH_DIM];
                    for py in 0..16 {
                        for px in 0..16 {
                            let x = (2 * bw + iw) * 16 + px;
                            let y = (2 * bh + ih) * 16 + py;
                            let expect = [x as u8, y as u8, (x + y) as u8];
                            for (c, &k) in expect.iter().enumerate() {
                                for t in 0..2 {
                                    let at = ((c * 2 + t) * 16 + py) * 16 + px;
                                    let want = (f32::from(k) - 127.5) / 127.5;
                                    assert_eq!(
                                        row[at], want,
                                        "row {r} c {c} t {t} py {py} px {px}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // The normalisation's range.
    let (lo, hi) = (normalise(0), normalise(255));
    assert_eq!((lo, hi), (-1.0, 1.0));
    assert!(patchify(&coordinate_image(48, 32)).is_err(), "48 is not a multiple of 32");
}

// ---------------------------------------------------------------------------
// The goldens (comparison 1, offline)

struct Golden {
    image: PathBuf,
    max_pixels: usize,
    min_pixels: usize,
    resized_hw: (usize, usize),
    grid: (usize, usize),
    tokens: usize,
    indices: Vec<usize>,
    values: Vec<f64>,
    atol: f64,
    max_atol: f64,
    min_frac: f64,
}

fn load_golden(name: &str) -> Golden {
    let path = repo().join("tools/reference/goldens").join(name);
    let g: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(g["kind"], "preprocess", "{name}");
    let as_usize = |v: &serde_json::Value| v.as_u64().unwrap() as usize;
    Golden {
        image: repo()
            .join("tools/reference/images")
            .join(g["image"]["file"].as_str().unwrap()),
        max_pixels: as_usize(&g["cap"]["max_pixels"]),
        min_pixels: as_usize(&g["cap"]["min_pixels"]),
        resized_hw: (as_usize(&g["resized_hw"][0]), as_usize(&g["resized_hw"][1])),
        grid: (
            as_usize(&g["image_grid_thw"][0][1]),
            as_usize(&g["image_grid_thw"][0][2]),
        ),
        tokens: as_usize(&g["tokens"]),
        indices: g["pixel_values"]["sample"]["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(as_usize)
            .collect(),
        values: g["pixel_values"]["sample"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect(),
        atol: g["tolerance"]["atol"].as_f64().unwrap(),
        max_atol: g["tolerance"]["max_atol"].as_f64().unwrap(),
        min_frac: g["tolerance"]["min_frac"].as_f64().unwrap(),
    }
}

/// One image against its golden: exact size and grid, the sample within
/// `PREPROCESS_TOLERANCE`, and the level histogram (zero, one, more) printed.
fn check_golden(name: &str) {
    let g = load_golden(name);
    let bytes = std::fs::read(&g.image).unwrap();
    let lim = ImageLimits {
        max_pixels: g.max_pixels,
        min_pixels: g.min_pixels,
        ..ImageLimits::default()
    };
    let started = Instant::now();
    let pv = preprocess(&bytes, &lim).unwrap();
    let ms = started.elapsed().as_secs_f64() * 1e3;
    assert_eq!((pv.resized.1, pv.resized.0), g.resized_hw, "{name}: resized (h, w)");
    assert_eq!((pv.grid_h, pv.grid_w), g.grid, "{name}: grid");
    assert_eq!(pv.tokens(), g.tokens, "{name}: tokens");
    assert_eq!(pv.data.len(), g.grid.0 * g.grid.1 * PATCH_DIM);
    let one_level = 2.0 / 255.0;
    let (mut within, mut max_abs, mut levels) = (0usize, 0.0f64, [0usize; 3]);
    for (&i, &want) in g.indices.iter().zip(&g.values) {
        let diff = (f64::from(pv.data[i]) - want).abs();
        max_abs = max_abs.max(diff);
        within += usize::from(diff <= g.atol);
        levels[((diff / one_level).round() as usize).min(2)] += 1;
    }
    let frac = within as f64 / g.indices.len() as f64;
    println!(
        "{name}: {} -> {} x {}, grid ({}, {}), {} tokens, {ms:.1} ms; sample of {}: {} at zero levels, {} at one, {} beyond; within atol {frac:.5}, max |err| {max_abs:.6}",
        Path::new(&g.image).file_stem().unwrap().to_string_lossy(),
        pv.resized.0,
        pv.resized.1,
        pv.grid_h,
        pv.grid_w,
        pv.tokens(),
        g.indices.len(),
        levels[0],
        levels[1],
        levels[2]
    );
    assert!(frac >= g.min_frac, "{name}: {frac} of the sample within {}", g.atol);
    assert!(
        max_abs <= g.max_atol,
        "{name}: max abs error {max_abs} over {}",
        g.max_atol
    );
    assert_eq!(levels[2], 0, "{name}: an element moved by two levels or more");
}

#[test]
fn golden_333x777() {
    check_golden("hf_vision_preprocess_333x777_cap2097152.json");
}

#[test]
fn golden_640x480() {
    check_golden("hf_vision_preprocess_640x480_cap2097152.json");
}

#[test]
fn golden_1920x1080() {
    check_golden("hf_vision_preprocess_1920x1080_cap2097152.json");
}

#[test]
fn golden_3840x2160_capped() {
    check_golden("hf_vision_preprocess_3840x2160_cap2097152.json");
}

#[test]
fn golden_3840x2160_uncapped() {
    check_golden("hf_vision_preprocess_3840x2160_cap16777216.json");
}
