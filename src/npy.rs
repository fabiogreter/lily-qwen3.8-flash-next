//! A minimal `.npy` reader and writer for float32 C-order arrays: the
//! interchange format of the vision reference goldens
//! (`tools/reference/goldens/large/`). Format: the magic `\x93NUMPY`, a
//! version byte pair, a little-endian header length, an ASCII dict with
//! `descr`, `fortran_order` and `shape`, then the raw data.

use std::path::Path;

use anyhow::{Context as _, Result, bail, ensure};

const MAGIC: &[u8] = b"\x93NUMPY";

/// A float32 array with its shape.
#[derive(Debug)]
pub struct NpyF32 {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Reads a little-endian float32 C-order `.npy` file.
pub fn read_f32(path: &Path) -> Result<NpyF32> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    parse_f32(&bytes).with_context(|| format!("parsing {}", path.display()))
}

pub fn parse_f32(bytes: &[u8]) -> Result<NpyF32> {
    ensure!(bytes.len() >= 10 && &bytes[..6] == MAGIC, "not a .npy file");
    let (major, minor) = (bytes[6], bytes[7]);
    let (header_len, start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            ensure!(bytes.len() >= 12, "truncated .npy header");
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        _ => bail!("unsupported .npy version {major}.{minor}"),
    };
    let end = start + header_len;
    ensure!(bytes.len() >= end, "truncated .npy header");
    let header =
        std::str::from_utf8(&bytes[start..end]).context("header is not UTF-8")?;
    let descr = dict_value(header, "descr")?;
    ensure!(
        descr == "'<f4'" || descr == "\"<f4\"",
        "dtype {descr} is not little-endian float32"
    );
    let fortran = dict_value(header, "fortran_order")?;
    ensure!(fortran == "False", "Fortran-order arrays are not supported");
    let shape_text = dict_value(header, "shape")?;
    let shape: Vec<usize> = shape_text
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().with_context(|| format!("shape entry {s:?}")))
        .collect::<Result<_>>()?;
    let numel: usize = shape.iter().product();
    let payload = &bytes[end..];
    ensure!(
        payload.len() == numel * 4,
        "payload holds {} bytes, shape {shape:?} needs {}",
        payload.len(),
        numel * 4
    );
    let data = payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(NpyF32 { shape, data })
}

/// The raw text of `key`'s value in the header dict, up to the next
/// top-level comma (tuples keep their parentheses).
fn dict_value(header: &str, key: &str) -> Result<String> {
    let quoted = format!("'{key}'");
    let at = header
        .find(&quoted)
        .or_else(|| header.find(&format!("\"{key}\"")))
        .with_context(|| format!("header has no {key}"))?;
    let rest = &header[at + quoted.len()..];
    let rest =
        rest.trim_start().strip_prefix(':').context("expected ':'")?.trim_start();
    let mut depth = 0i32;
    let mut end = rest.len();
    for (i, c) in rest.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' | '}' if depth <= 0 => {
                end = i;
                break;
            }
            _ => {}
        }
    }
    Ok(rest[..end].trim().to_string())
}

/// Serializes a float32 C-order array as a version 1.0 `.npy` file.
pub fn to_bytes_f32(shape: &[usize], data: &[f32]) -> Result<Vec<u8>> {
    let numel: usize = shape.iter().product();
    ensure!(
        data.len() == numel,
        "data has {} elements, shape {shape:?} needs {numel}",
        data.len()
    );
    let shape_text = match shape {
        [one] => format!("({one},)"),
        _ => format!(
            "({})",
            shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        ),
    };
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_text}, }}");
    // The whole preamble (magic, version, length, header) is padded to a
    // multiple of 64 bytes, the header ending in a newline.
    let preamble = MAGIC.len() + 2 + 2;
    let padded = (preamble + header.len() + 1).div_ceil(64) * 64;
    header.extend(std::iter::repeat_n(' ', padded - preamble - header.len() - 1));
    header.push('\n');
    let mut out = Vec::with_capacity(padded + numel * 4);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

pub fn write_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<()> {
    std::fs::write(path, to_bytes_f32(shape, data)?)
        .with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
#[path = "../tests/unit/npy.rs"]
mod tests;
