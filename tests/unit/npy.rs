use super::*;

#[test]
fn round_trips_through_the_numpy_layout() {
    let data: Vec<f32> = (0..6).map(|i| i as f32 * 0.5 - 1.0).collect();
    let bytes = to_bytes_f32(&[2, 3], &data).expect("serialize");
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    assert_eq!(bytes[6..8], [1, 0]);
    // Preamble plus header pad to a multiple of 64 bytes, newline-terminated.
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    assert_eq!((10 + header_len) % 64, 0);
    assert_eq!(bytes[10 + header_len - 1], b'\n');
    assert_eq!(bytes.len(), 10 + header_len + 6 * 4);
    let back = parse_f32(&bytes).expect("parse");
    assert_eq!(back.shape, vec![2, 3]);
    assert_eq!(back.data, data);
}

#[test]
fn reads_numpys_own_header_style() {
    // numpy 2.x writes exactly this for np.zeros((1, 4), np.float32).
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 4), }";
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    let mut h = header.to_string();
    while !(10 + h.len() + 1).is_multiple_of(64) {
        h.push(' ');
    }
    h.push('\n');
    bytes.extend_from_slice(&(h.len() as u16).to_le_bytes());
    bytes.extend_from_slice(h.as_bytes());
    bytes.extend_from_slice(&[0u8; 16]);
    let arr = parse_f32(&bytes).expect("parse");
    assert_eq!(arr.shape, vec![1, 4]);
    assert_eq!(arr.data, vec![0.0; 4]);
    // A one-dimensional shape keeps its trailing comma.
    let one = to_bytes_f32(&[3], &[1.0, 2.0, 3.0]).expect("1-d");
    let header_len = u16::from_le_bytes([one[8], one[9]]) as usize;
    let header = std::str::from_utf8(&one[10..10 + header_len]).expect("ascii header");
    assert!(header.contains("'shape': (3,)"), "{header}");
    assert_eq!(parse_f32(&one).expect("parse 1-d").shape, vec![3]);
}

#[test]
fn rejects_other_dtypes() {
    let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }\n";
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0u8; 8]);
    let err = parse_f32(&bytes).expect_err("f64 is refused");
    assert!(format!("{err:#}").contains("float32"), "{err:#}");
}
