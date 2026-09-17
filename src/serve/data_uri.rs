//! The one URL form an image part may carry: a `data:` URI with base64
//! payload. The server never fetches an image by URL (a request to an
//! arbitrary address from the server is a request-forgery surface, and the
//! agent clients send data URIs anyway), so anything else is refused here
//! with a message that says so.
//!
//! The decoder takes the standard and the URL-safe alphabets (RFC 4648 §4
//! and §5), with or without `=` padding, and skips ASCII whitespace, which
//! some clients insert when they wrap long lines. No crate for it: the
//! dependency list is deliberately short and this is forty lines.

use anyhow::{Result, bail, ensure};

/// A parsed image data URI: the declared media type (`image/jpg` is
/// reported as `image/jpeg`) and the decoded bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageData {
    pub media_type: &'static str,
    pub bytes: Vec<u8>,
}

/// Parses `data:image/png;base64,...` or `data:image/jpeg;base64,...`.
/// Every other URL is refused: `http(s)`, `file`, other schemes, a
/// non-image or non-base64 data URI.
pub fn parse_image_data_uri(url: &str) -> Result<ImageData> {
    let Some(rest) = url.strip_prefix("data:") else {
        let scheme = url.split(':').next().unwrap_or("").to_ascii_lowercase();
        if scheme == "http" || scheme == "https" || scheme == "file" {
            bail!(
                "image URLs are not fetched ({scheme}: refused); the server accepts data URIs only \
                 (data:image/png;base64,... or data:image/jpeg;base64,...)"
            );
        }
        bail!(
            "the image URL is not a data URI; the server accepts data URIs only \
             (data:image/png;base64,... or data:image/jpeg;base64,...)"
        );
    };
    let Some((header, payload)) = rest.split_once(',') else {
        bail!("the image data URI has no ',' separating its header from the payload");
    };
    let mut params = header.split(';');
    let media = params.next().unwrap_or("").trim().to_ascii_lowercase();
    let media_type = match media.as_str() {
        "image/png" => "image/png",
        "image/jpeg" | "image/jpg" => "image/jpeg",
        "" => bail!(
            "the image data URI names no media type; the server accepts image/png and image/jpeg"
        ),
        other => bail!(
            "image data URIs of type {other:?} are not accepted; the server decodes image/png and image/jpeg"
        ),
    };
    let base64 = params.any(|p| p.trim().eq_ignore_ascii_case("base64"));
    ensure!(base64, "the image data URI is not base64-encoded (`;base64` is missing)");
    let bytes = decode_base64(payload)?;
    ensure!(!bytes.is_empty(), "the image data URI carries no data");
    Ok(ImageData { media_type, bytes })
}

/// Decodes base64 in the standard (`+/`) or URL-safe (`-_`) alphabet, mixed
/// freely, with optional `=` padding; ASCII whitespace is skipped. Refuses
/// any other character, padding in the middle, and a final group of one
/// character (which cannot encode a byte).
pub fn decode_base64(text: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3 + 2);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    let mut chars = 0usize;
    let mut padding = 0usize;
    for (at, byte) in text.bytes().enumerate() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => {
                padding += 1;
                ensure!(
                    padding <= 2,
                    "base64 payload has more than two '=' padding characters"
                );
                continue;
            }
            b' ' | b'\t' | b'\r' | b'\n' => continue,
            _ => bail!(
                "base64 payload has an invalid character {:?} at offset {at}",
                char::from(byte)
            ),
        };
        ensure!(
            padding == 0,
            "base64 payload has data after its '=' padding (offset {at})"
        );
        acc = (acc << 6) | u32::from(value);
        bits += 6;
        chars += 1;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    let rem = chars % 4;
    ensure!(
        rem != 1,
        "base64 payload ends with a lone character: its length is not a multiple of 4"
    );
    if padding > 0 {
        ensure!(
            (rem + padding).is_multiple_of(4),
            "base64 payload has {padding} '=' padding characters for {rem} trailing characters"
        );
    }
    // The two or four bits an incomplete final group leaves over carry no
    // byte; a canonical encoder leaves them zero and a sloppy one is not
    // worth refusing over.
    Ok(out)
}

#[cfg(test)]
#[path = "../../tests/unit/serve/data_uri.rs"]
mod tests;
