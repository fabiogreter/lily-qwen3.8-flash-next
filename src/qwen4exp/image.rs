//! Host-side image preprocessing (`docs/architecture.md`, "The vision tower"): the
//! bytes of a PNG or JPEG become the `pixel_values` rows the vision tower
//! consumes ([`super::vision::VisionTower::forward`]). The chain is
//! `tools/reference/VISION.md`, "Preprocessing", step by step:
//!
//! 1. decode to 8-bit RGB the way `PIL.Image.convert("RGB")` does: alpha is
//!    dropped without compositing, greyscale is replicated, a palette is
//!    looked up, 16-bit samples keep their high byte;
//! 2. [`smart_resize`]: the target size, a multiple of 32 on both sides,
//!    under the pixel cap;
//! 3. [`resize_bicubic`]: PIL's `Image.resize(..., BICUBIC)` reproduced from
//!    `libImaging/Resample.c` (Pillow 12.3.0): Keys cubic with a = -0.5,
//!    support widened by the downscale factor, coefficients normalised in
//!    f64 and then fixed to 22 fractional bits, the horizontal pass first,
//!    clamp-and-round to uint8 after each pass, and a pass skipped when its
//!    axis keeps its size;
//! 4. [`patchify`]: `(k - 127.5) / 127.5` in f32 into 1 536-wide rows in
//!    block-major patch order with the two temporal frames identical.
//!
//! Decoding is the server's first contact with user-supplied binary data,
//! so the header is read and judged before either decoder allocates
//! anything: an unsupported format is named and refused, the source may not
//! exceed [`ImageLimits::max_side`] on either side or
//! [`ImageLimits::max_source_pixels`] in area, and the decoders run under an
//! allocation bound. Decoder errors come back as errors, never panics.

use std::fmt;
use std::io::Cursor;

use anyhow::{Context as _, Result, bail, ensure};
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

/// The tower's spatial patch (`vision_config.patch_size`).
pub const PATCH_SIZE: usize = 16;
/// Patches merged into one language token per axis (`spatial_merge_size`).
pub const MERGE_SIZE: usize = 2;
/// Frames per patch row (`temporal_patch_size`); a still image is copied.
pub const TEMPORAL_PATCH_SIZE: usize = 2;
/// Colour channels of a patch row.
pub const CHANNELS: usize = 3;
/// Both resized sides are multiples of this: one language token is
/// `FACTOR x FACTOR` pixels.
pub const FACTOR: usize = PATCH_SIZE * MERGE_SIZE;
/// Floats per patch row: `3 x 2 x 16 x 16`.
pub const PATCH_DIM: usize = CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;

/// The server-side pixel cap (VISION.md "The pixel cap"): 2 048 language
/// tokens of 32 x 32 pixels.
pub const DEFAULT_MAX_PIXELS: usize = 2048 * FACTOR * FACTOR;
/// The checkpoint's `size.shortest_edge`: an image is scaled up to at least
/// this many pixels.
pub const DEFAULT_MIN_PIXELS: usize = 65_536;

/// The bounds a request's image is judged against before and during
/// decoding, and the cap `smart_resize` scales it to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageLimits {
    /// `smart_resize` scales the image down until `h_bar * w_bar` fits.
    pub max_pixels: usize,
    /// `smart_resize` scales the image up until `h_bar * w_bar` reaches it.
    pub min_pixels: usize,
    /// The source may not have more pixels than this (header check, before
    /// any allocation).
    pub max_source_pixels: u64,
    /// Neither source side may exceed this (header check).
    pub max_side: u32,
    /// Upper bound on the decoded image and on what the decoders allocate.
    pub max_decoded_bytes: usize,
}

impl Default for ImageLimits {
    fn default() -> Self {
        Self {
            max_pixels: DEFAULT_MAX_PIXELS,
            min_pixels: DEFAULT_MIN_PIXELS,
            max_source_pixels: 64 * 1024 * 1024,
            max_side: 16_384,
            // 64 megapixels of 16-bit RGBA, the largest source the pixel and
            // side limits let through.
            max_decoded_bytes: 64 * 1024 * 1024 * 8,
        }
    }
}

/// The two formats the server decodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Jpeg,
}

impl fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ImageFormat::Png => "PNG",
            ImageFormat::Jpeg => "JPEG",
        })
    }
}

/// What the file header says, before anything is decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageHeader {
    pub format: ImageFormat,
    pub width: u32,
    pub height: u32,
    /// Samples per pixel as stored (1 to 4; a palette counts as 1).
    pub channels: u8,
    /// Bits per sample as stored (PNG 1 to 16; JPEG 8 or 12).
    pub bit_depth: u8,
}

impl ImageHeader {
    /// The bytes one decoded frame takes as stored (before the conversion
    /// to RGB), the quantity the allocation bound judges.
    pub fn decoded_bytes(&self) -> u64 {
        let bytes_per_sample = u64::from(self.bit_depth.div_ceil(8));
        u64::from(self.width)
            * u64::from(self.height)
            * u64::from(self.channels)
            * bytes_per_sample
    }
}

/// An 8-bit RGB image, rows top to bottom, `width * height * 3` bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self> {
        ensure!(
            data.len() == width * height * CHANNELS,
            "RGB data has {} bytes, {width} x {height} needs {}",
            data.len(),
            width * height * CHANNELS
        );
        Ok(Self { width, height, data })
    }
}

/// The tower's input for one image.
#[derive(Clone, Debug, PartialEq)]
pub struct PixelValues {
    /// `[grid_h * grid_w, PATCH_DIM]` f32 in block-major patch order.
    pub data: Vec<f32>,
    /// Patch grid: `resized height / 16`.
    pub grid_h: usize,
    /// Patch grid: `resized width / 16`.
    pub grid_w: usize,
    /// The resized image's `(width, height)`.
    pub resized: (usize, usize),
}

impl PixelValues {
    /// Patch rows, `grid_h * grid_w`.
    pub fn patches(&self) -> usize {
        self.grid_h * self.grid_w
    }

    /// Language tokens the image becomes: one per 2 x 2 merge block.
    pub fn tokens(&self) -> usize {
        self.patches() / (MERGE_SIZE * MERGE_SIZE)
    }
}

/// Decodes, resizes and patchifies one image under `limits`.
pub fn preprocess(bytes: &[u8], limits: &ImageLimits) -> Result<PixelValues> {
    let image = decode_image(bytes, limits)?;
    let (h_bar, w_bar) =
        smart_resize(image.height, image.width, limits.min_pixels, limits.max_pixels)?;
    let resized = resize_bicubic(&image, w_bar, h_bar);
    patchify(&resized)
}

// ---------------------------------------------------------------------------
// Header sniffing and decoding

/// Reads the format and dimensions off the header without decoding; refuses
/// formats other than PNG and JPEG, naming the one detected.
pub fn read_header(bytes: &[u8]) -> Result<ImageHeader> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return png_header(bytes);
    }
    if bytes.starts_with(&[0xFF, 0xD8]) {
        return jpeg_header(bytes);
    }
    bail!(
        "unsupported image format: {} (PNG and JPEG are accepted)",
        detect_other(bytes)
    )
}

/// A name for a header that is not PNG or JPEG, for the error message.
fn detect_other(bytes: &[u8]) -> &'static str {
    let starts = |magic: &[u8]| bytes.starts_with(magic);
    if bytes.is_empty() {
        "empty data"
    } else if starts(b"GIF8") {
        "GIF"
    } else if starts(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        "WebP"
    } else if starts(b"BM") {
        "BMP"
    } else if starts(b"II*\0") || starts(b"MM\0*") {
        "TIFF"
    } else if bytes.get(4..8) == Some(b"ftyp") {
        "HEIF/AVIF"
    } else if starts(b"\0\0\x01\0") {
        "ICO"
    } else if starts(b"%PDF") {
        "PDF"
    } else if starts(b"<?xml") || starts(b"<svg") {
        "SVG/XML"
    } else if starts(b"8BPS") {
        "PSD"
    } else if starts(b"\xFF\x0A") || starts(b"\0\0\0\x0CJXL ") {
        "JPEG XL"
    } else if starts(b"qoif") {
        "QOI"
    } else {
        "unrecognised data"
    }
}

fn png_header(bytes: &[u8]) -> Result<ImageHeader> {
    // Signature (8), IHDR length (4), "IHDR" (4), then 13 bytes of fields.
    ensure!(bytes.len() >= 8 + 8 + 13, "PNG header is truncated");
    ensure!(&bytes[12..16] == b"IHDR", "PNG does not start with an IHDR chunk");
    let be32 = |at: usize| {
        u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    };
    let (width, height) = (be32(16), be32(20));
    let bit_depth = bytes[24];
    let channels = match bytes[25] {
        0 => 1, // greyscale
        2 => 3, // RGB
        3 => 1, // palette
        4 => 2, // greyscale + alpha
        6 => 4, // RGBA
        other => bail!("PNG colour type {other} is not valid"),
    };
    ensure!(
        matches!(bit_depth, 1 | 2 | 4 | 8 | 16),
        "PNG bit depth {bit_depth} is not valid"
    );
    Ok(ImageHeader { format: ImageFormat::Png, width, height, channels, bit_depth })
}

fn jpeg_header(bytes: &[u8]) -> Result<ImageHeader> {
    // Walk the marker segments to the first start-of-frame.
    let mut at = 2;
    loop {
        ensure!(at + 4 <= bytes.len(), "JPEG has no frame header");
        ensure!(bytes[at] == 0xFF, "JPEG marker expected at byte {at}");
        let marker = bytes[at + 1];
        if marker == 0xFF {
            // Fill byte before a marker.
            at += 1;
            continue;
        }
        match marker {
            // Standalone markers without a length field.
            0xD8 | 0x01 | 0xD0..=0xD7 => {
                at += 2;
                continue;
            }
            // Start of scan or end of image before any frame header.
            0xDA | 0xD9 => bail!("JPEG has no frame header before its scan data"),
            _ => {}
        }
        let len = usize::from(u16::from_be_bytes([bytes[at + 2], bytes[at + 3]]));
        ensure!(len >= 2, "JPEG segment length {len} is not valid");
        let is_sof =
            matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xC8 | 0xCC);
        if is_sof {
            ensure!(
                len >= 8 && at + 2 + len <= bytes.len(),
                "JPEG frame header is truncated"
            );
            let f = &bytes[at + 4..at + 2 + len];
            let bit_depth = f[0];
            let height = u32::from(u16::from_be_bytes([f[1], f[2]]));
            let width = u32::from(u16::from_be_bytes([f[3], f[4]]));
            let channels = f[5];
            return Ok(ImageHeader {
                format: ImageFormat::Jpeg,
                width,
                height,
                channels,
                bit_depth,
            });
        }
        at += 2 + len;
    }
}

/// Judges a header against `limits` before anything is allocated.
pub fn check_header(header: &ImageHeader, limits: &ImageLimits) -> Result<()> {
    let (w, h) = (header.width, header.height);
    ensure!(w > 0 && h > 0, "{} image is {w} x {h}: zero-sized", header.format);
    ensure!(
        w <= limits.max_side && h <= limits.max_side,
        "{} image is {w} x {h}: a side exceeds the limit of {}",
        header.format,
        limits.max_side
    );
    let pixels = u64::from(w) * u64::from(h);
    ensure!(
        pixels <= limits.max_source_pixels,
        "{} image is {w} x {h} = {pixels} pixels, over the limit of {}",
        header.format,
        limits.max_source_pixels
    );
    ensure!(
        header.decoded_bytes() <= limits.max_decoded_bytes as u64,
        "{} image is {w} x {h} with {} channels of {} bits: {} bytes decoded, over the limit of {}",
        header.format,
        header.channels,
        header.bit_depth,
        header.decoded_bytes(),
        limits.max_decoded_bytes
    );
    Ok(())
}

/// Decodes a PNG or JPEG to 8-bit RGB as `PIL.Image.convert("RGB")` would,
/// after the header check.
pub fn decode_image(bytes: &[u8], limits: &ImageLimits) -> Result<RgbImage> {
    let header = read_header(bytes)?;
    check_header(&header, limits)?;
    match header.format {
        ImageFormat::Png => decode_png(bytes, &header, limits),
        ImageFormat::Jpeg => decode_jpeg(bytes, &header, limits),
    }
}

fn decode_png(
    bytes: &[u8],
    header: &ImageHeader,
    limits: &ImageLimits,
) -> Result<RgbImage> {
    let mut decoder = png::Decoder::new_with_limits(
        Cursor::new(bytes),
        png::Limits { bytes: limits.max_decoded_bytes },
    );
    // EXPAND looks a palette up and widens 1, 2 and 4-bit greyscale to 8 bits
    // (scaled, as PIL's L;1 / L;2 / L;4 unpackers do); STRIP_16 keeps the
    // high byte of 16-bit samples, which is what PIL's RGB;16B unpacker
    // does for 16-bit RGB and RGBA. For 16-bit greyscale PIL 12.3.0 opens the
    // image as I;16 and convert("RGB") saturates the 16-bit value at 255
    // instead, leaving almost every pixel white; the high byte is taken here
    // too, a deliberate departure noted in docs/architecture.md.
    decoder.set_transformations(
        png::Transformations::EXPAND | png::Transformations::STRIP_16,
    );
    decoder.set_ignore_text_chunk(true);
    let mut reader = decoder.read_info().context("PNG header")?;
    let info = reader.info();
    ensure!(
        info.width == header.width && info.height == header.height,
        "PNG decoder reports {} x {}, the header said {} x {}",
        info.width,
        info.height,
        header.width,
        header.height
    );
    let (color, depth) = reader.output_color_type();
    ensure!(depth == png::BitDepth::Eight, "PNG decoded at {depth:?} bits, expected 8");
    let size = reader.output_buffer_size().context("PNG frame size overflows")?;
    ensure!(
        size <= limits.max_decoded_bytes,
        "PNG frame of {size} bytes exceeds the decoded-size limit of {}",
        limits.max_decoded_bytes
    );
    let mut buf = vec![0u8; size];
    let out = reader.next_frame(&mut buf).context("PNG data")?;
    buf.truncate(out.buffer_size());
    let (w, h) = (header.width as usize, header.height as usize);
    let channels = match color {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => bail!("PNG palette was not expanded"),
    };
    ensure!(
        buf.len() == w * h * channels,
        "PNG frame has {} bytes, expected {}",
        buf.len(),
        w * h * channels
    );
    Ok(to_rgb(&buf, w, h, channels))
}

// TODO: zune-jpeg is not bit-identical to the libjpeg-turbo behind Pillow:
// measured on Pillow-encoded fixtures, greyscale differs on about 1 % of the
// samples by one uint8 level (IDCT rounding) and 4:2:0 colour on 2 to 16 % by
// one level and 1 to 10 % by two or three (chroma upsampling); `jpeg-decoder`
// 0.3 is closer (7 % and 3 % on the small fixture) but not exact either. The
// residual is a property of the decoder crate, outside this item; the
// server's driving input is PNG screenshots, which decode exactly.
fn decode_jpeg(
    bytes: &[u8],
    header: &ImageHeader,
    limits: &ImageLimits,
) -> Result<RgbImage> {
    // `new_safe` keeps to the portable code paths; the output colourspace is
    // RGB whatever the file holds (greyscale is replicated, YCbCr converted).
    // Strict mode makes a truncated scan an error, as PIL's default does
    // (`LOAD_TRUNCATED_IMAGES` off); the lenient mode fills the rest with
    // grey and reports success.
    let options = DecoderOptions::new_safe()
        .jpeg_set_out_colorspace(ColorSpace::RGB)
        .set_max_width(limits.max_side as usize)
        .set_max_height(limits.max_side as usize)
        .set_strict_mode(true);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    decoder.decode_headers().map_err(|e| anyhow::anyhow!("JPEG header: {e}"))?;
    let info = decoder.info().context("JPEG header was not decoded")?;
    ensure!(
        u32::from(info.width) == header.width
            && u32::from(info.height) == header.height,
        "JPEG decoder reports {} x {}, the header said {} x {}",
        info.width,
        info.height,
        header.width,
        header.height
    );
    let (w, h) = (header.width as usize, header.height as usize);
    let size = decoder.output_buffer_size().context("JPEG frame size overflows")?;
    ensure!(
        size == w * h * CHANNELS,
        "JPEG frame of {size} bytes is not {w} x {h} RGB"
    );
    ensure!(
        size <= limits.max_decoded_bytes,
        "JPEG frame of {size} bytes exceeds the decoded-size limit of {}",
        limits.max_decoded_bytes
    );
    let mut buf = vec![0u8; size];
    decoder.decode_into(&mut buf).map_err(|e| anyhow::anyhow!("JPEG data: {e}"))?;
    RgbImage::new(w, h, buf)
}

/// `PIL.Image.convert("RGB")` on 8-bit samples: greyscale replicated, alpha
/// dropped without compositing.
fn to_rgb(buf: &[u8], width: usize, height: usize, channels: usize) -> RgbImage {
    let n = width * height;
    let data = match channels {
        3 => buf.to_vec(),
        1 => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        2 => buf.chunks_exact(2).flat_map(|ga| [ga[0], ga[0], ga[0]]).collect(),
        4 => buf.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect(),
        _ => unreachable!("channel count {channels} was validated"),
    };
    debug_assert_eq!(data.len(), n * CHANNELS);
    RgbImage { width, height, data }
}

// ---------------------------------------------------------------------------
// smart_resize

/// The largest aspect ratio `smart_resize` accepts.
pub const MAX_ASPECT_RATIO: f64 = 200.0;

/// `Qwen2VLImageProcessor.smart_resize` on `(height, width)`: both sides
/// rounded to multiples of 32, then scaled down under `max_pixels` or up over
/// `min_pixels`; returns `(h_bar, w_bar)`. Python's `round` rounds half to
/// even, `floor` and `ceil` are the usual ones, and the floating-point
/// expressions are evaluated in the reference's order.
pub fn smart_resize(
    height: usize,
    width: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    ensure!(height > 0 && width > 0, "image is {width} x {height}: zero-sized");
    let (h, w) = (height as f64, width as f64);
    let ratio = h.max(w) / h.min(w);
    ensure!(
        ratio <= MAX_ASPECT_RATIO,
        "aspect ratio {ratio:.1} of {width} x {height} exceeds {MAX_ASPECT_RATIO}"
    );
    let factor = FACTOR as f64;
    let mut h_bar = (h / factor).round_ties_even() * factor;
    let mut w_bar = (w / factor).round_ties_even() * factor;
    if h_bar * w_bar > max_pixels as f64 {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = factor.max((h / beta / factor).floor() * factor);
        w_bar = factor.max((w / beta / factor).floor() * factor);
    } else if h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / factor).ceil() * factor;
        w_bar = (w * beta / factor).ceil() * factor;
    }
    Ok((h_bar as usize, w_bar as usize))
}

// ---------------------------------------------------------------------------
// PIL's bicubic resampler

/// Keys cubic convolution with a = -0.5, `bicubic_filter` in Resample.c.
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// The filter's support in source pixels at scale 1.
const BICUBIC_SUPPORT: f64 = 2.0;
/// Fractional bits of the fixed-point coefficients in PIL's 8-bit path,
/// `PRECISION_BITS (32 - 8 - 2)`: two bits of headroom for a coefficient sum
/// above one or below zero.
const PRECISION_BITS: u32 = 22;

/// One axis's resampling plan: for output index `i`, taps
/// `coeffs[i * ksize..][..counts[i]]` apply to source indices from
/// `starts[i]`. `precompute_coeffs` and `normalize_coeffs_8bpc` in
/// Resample.c.
struct AxisPlan {
    ksize: usize,
    starts: Vec<usize>,
    counts: Vec<usize>,
    coeffs: Vec<i32>,
}

fn axis_plan(in_size: usize, out_size: usize) -> AxisPlan {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = BICUBIC_SUPPORT * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let inv_filterscale = 1.0 / filterscale;
    let mut starts = Vec::with_capacity(out_size);
    let mut counts = Vec::with_capacity(out_size);
    let mut coeffs = vec![0i32; out_size * ksize];
    let mut k = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        // `(int)(v + 0.5)` in C truncates toward zero; a negative result is
        // clamped to zero either way.
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = (((center + support + 0.5) as i64).max(0) as usize).min(in_size);
        let count = xmax.saturating_sub(xmin);
        let mut ww = 0.0;
        for (x, slot) in k.iter_mut().take(count).enumerate() {
            let w = bicubic(((x + xmin) as f64 - center + 0.5) * inv_filterscale);
            *slot = w;
            ww += w;
        }
        if ww != 0.0 {
            for slot in k.iter_mut().take(count) {
                *slot /= ww;
            }
        }
        let one = f64::from(1u32 << PRECISION_BITS);
        for (x, &w) in k.iter().take(count).enumerate() {
            // Round half away from zero, as `(int)(±0.5 + w * 2^22)` does.
            coeffs[xx * ksize + x] =
                if w < 0.0 { (-0.5 + w * one) as i32 } else { (0.5 + w * one) as i32 };
        }
        starts.push(xmin);
        counts.push(count);
    }
    AxisPlan { ksize, starts, counts, coeffs }
}

/// `clip8` in Resample.c: drop the fractional bits (arithmetic shift) and
/// clamp to a byte.
#[inline]
fn clip8(acc: i32) -> u8 {
    (acc >> PRECISION_BITS).clamp(0, 255) as u8
}

/// The rounding offset every accumulator starts from: one half in the
/// fixed-point scale.
const HALF: i32 = 1 << (PRECISION_BITS - 1);

/// `ImagingResampleHorizontal_8bpc` over every row.
fn resample_horizontal(src: &RgbImage, out_w: usize) -> RgbImage {
    let plan = axis_plan(src.width, out_w);
    let mut data = vec![0u8; out_w * src.height * CHANNELS];
    for (row_in, row_out) in src
        .data
        .chunks_exact(src.width * CHANNELS)
        .zip(data.chunks_exact_mut(out_w * CHANNELS))
    {
        for xx in 0..out_w {
            let (xmin, count) = (plan.starts[xx], plan.counts[xx]);
            let k = &plan.coeffs[xx * plan.ksize..][..count];
            let taps = &row_in[xmin * CHANNELS..][..count * CHANNELS];
            let mut acc = [HALF; CHANNELS];
            for (px, &w) in taps.chunks_exact(CHANNELS).zip(k) {
                for (a, &p) in acc.iter_mut().zip(px) {
                    *a += i32::from(p) * w;
                }
            }
            for (o, &a) in row_out[xx * CHANNELS..][..CHANNELS].iter_mut().zip(&acc) {
                *o = clip8(a);
            }
        }
    }
    RgbImage { width: out_w, height: src.height, data }
}

/// `ImagingResampleVertical_8bpc`: the same plan down the columns, one output
/// row at a time with the taps as the outer loop so the inner loop runs along
/// the row.
fn resample_vertical(src: &RgbImage, out_h: usize) -> RgbImage {
    let plan = axis_plan(src.height, out_h);
    let stride = src.width * CHANNELS;
    let mut data = vec![0u8; stride * out_h];
    let mut acc = vec![HALF; stride];
    for (yy, row_out) in data.chunks_exact_mut(stride).enumerate() {
        let (ymin, count) = (plan.starts[yy], plan.counts[yy]);
        let k = &plan.coeffs[yy * plan.ksize..][..count];
        acc.fill(HALF);
        for (y, &w) in k.iter().enumerate() {
            let row_in = &src.data[(ymin + y) * stride..][..stride];
            for (a, &p) in acc.iter_mut().zip(row_in) {
                *a += i32::from(p) * w;
            }
        }
        for (o, &a) in row_out.iter_mut().zip(&acc) {
            *o = clip8(a);
        }
    }
    RgbImage { width: src.width, height: out_h, data }
}

/// `PIL.Image.resize((width, height), Image.BICUBIC)` on an RGB image: the
/// horizontal pass first, then the vertical one, each rounded to uint8, and
/// an axis that keeps its size is not resampled at all (`ImagingResampleInner`).
pub fn resize_bicubic(src: &RgbImage, width: usize, height: usize) -> RgbImage {
    assert!(width > 0 && height > 0, "resize target {width} x {height} is empty");
    match (width != src.width, height != src.height) {
        (false, false) => src.clone(),
        (true, false) => resample_horizontal(src, width),
        (false, true) => resample_vertical(src, height),
        (true, true) => resample_vertical(&resample_horizontal(src, width), height),
    }
}

// ---------------------------------------------------------------------------
// Normalise and patchify

/// `(k - 127.5) / 127.5` in f32: the reference's fused rescale and normalise
/// with mean = std = 0.5 and rescale 1/255.
#[inline]
fn normalise(k: u8) -> f32 {
    (f32::from(k) - 127.5) / 127.5
}

/// VISION.md steps 5 and 6: the resized image (both sides multiples of 32)
/// as `[gh * gw, 1536]` rows, row `r = ((bh * (gw / 2) + bw) * 2 + ih) * 2 + iw`
/// holding the patch at pixel rows `(2 bh + ih) * 16 ..` and columns
/// `(2 bw + iw) * 16 ..`, laid out `((c * 2 + t) * 16 + py) * 16 + px` with
/// both temporal frames the same still.
pub fn patchify(image: &RgbImage) -> Result<PixelValues> {
    let (w, h) = (image.width, image.height);
    ensure!(
        w.is_multiple_of(FACTOR) && h.is_multiple_of(FACTOR) && w > 0 && h > 0,
        "resized image {w} x {h} is not a positive multiple of {FACTOR} on both sides"
    );
    let (gh, gw) = (h / PATCH_SIZE, w / PATCH_SIZE);
    let (bh_n, bw_n) = (gh / MERGE_SIZE, gw / MERGE_SIZE);
    let mut data = vec![0f32; gh * gw * PATCH_DIM];
    let frame = PATCH_SIZE * PATCH_SIZE;
    for (r, row) in data.chunks_exact_mut(PATCH_DIM).enumerate() {
        let iw = r % MERGE_SIZE;
        let ih = (r / MERGE_SIZE) % MERGE_SIZE;
        let block = r / (MERGE_SIZE * MERGE_SIZE);
        let (bh, bw) = (block / bw_n, block % bw_n);
        debug_assert!(bh < bh_n);
        let y0 = (bh * MERGE_SIZE + ih) * PATCH_SIZE;
        let x0 = (bw * MERGE_SIZE + iw) * PATCH_SIZE;
        for py in 0..PATCH_SIZE {
            let src =
                &image.data[((y0 + py) * w + x0) * CHANNELS..][..PATCH_SIZE * CHANNELS];
            for (px, pixel) in src.chunks_exact(CHANNELS).enumerate() {
                for (c, &k) in pixel.iter().enumerate() {
                    let v = normalise(k);
                    let at = (c * TEMPORAL_PATCH_SIZE * frame) + py * PATCH_SIZE + px;
                    row[at] = v;
                    row[at + frame] = v;
                }
            }
        }
    }
    Ok(PixelValues { data, grid_h: gh, grid_w: gw, resized: (w, h) })
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/image.rs"]
mod tests;
