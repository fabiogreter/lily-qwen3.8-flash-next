use super::*;

#[test]
fn base64_decodes_the_rfc_4648_vectors_in_both_alphabets() {
    assert_eq!(decode_base64("").unwrap(), b"");
    assert_eq!(decode_base64("Zg==").unwrap(), b"f");
    assert_eq!(decode_base64("Zm8=").unwrap(), b"fo");
    assert_eq!(decode_base64("Zm9v").unwrap(), b"foo");
    assert_eq!(decode_base64("Zm9vYg==").unwrap(), b"foob");
    assert_eq!(decode_base64("Zm9vYmE=").unwrap(), b"fooba");
    assert_eq!(decode_base64("Zm9vYmFy").unwrap(), b"foobar");
    // Padding is optional.
    assert_eq!(decode_base64("Zg").unwrap(), b"f");
    assert_eq!(decode_base64("Zm8").unwrap(), b"fo");
    assert_eq!(decode_base64("Zm9vYg").unwrap(), b"foob");
    // Bytes whose encoding uses the 62nd and 63rd symbols: standard `+/`
    // and URL-safe `-_` decode to the same bytes.
    let bytes = [0xfb, 0xff, 0xbf, 0x3e, 0xfe];
    assert_eq!(decode_base64("+/+/Pv4=").unwrap(), bytes);
    assert_eq!(decode_base64("-_-_Pv4").unwrap(), bytes);
    assert_eq!(decode_base64("+_-/Pv4=").unwrap(), bytes);
    // Whitespace from line wrapping is skipped.
    assert_eq!(decode_base64("Zm9v\r\nYmFy\n").unwrap(), b"foobar");
    assert_eq!(decode_base64(" Zm 9v Ym Fy ").unwrap(), b"foobar");
}

#[test]
fn base64_refuses_malformed_payloads_with_the_reason() {
    let err = |s: &str| decode_base64(s).unwrap_err().to_string();
    assert!(
        err("Zm9v*").contains("invalid character '*' at offset 4"),
        "{}",
        err("Zm9v*")
    );
    assert!(err("Z").contains("lone character"), "{}", err("Z"));
    assert!(err("Zm9vY").contains("lone character"), "{}", err("Zm9vY"));
    assert!(err("Zg===").contains("more than two"), "{}", err("Zg==="));
    assert!(err("Zg=x").contains("after its '=' padding"), "{}", err("Zg=x"));
    // Padding that does not complete the final group.
    assert!(err("Zg=").contains("padding characters for"), "{}", err("Zg="));
    assert!(err("Zm9v=").contains("padding characters for"), "{}", err("Zm9v="));
}

#[test]
fn image_data_uris_are_parsed_and_everything_else_is_refused() {
    let png = parse_image_data_uri("data:image/png;base64,iVBORw0KGgo=").unwrap();
    assert_eq!(png.media_type, "image/png");
    assert_eq!(png.bytes, b"\x89PNG\r\n\x1a\n");
    let jpeg = parse_image_data_uri("data:image/jpeg;base64,/9j/4A==").unwrap();
    assert_eq!(
        (jpeg.media_type, jpeg.bytes.as_slice()),
        ("image/jpeg", &b"\xff\xd8\xff\xe0"[..])
    );
    // `image/jpg` is the same thing; parameters and case are tolerated.
    let jpg =
        parse_image_data_uri("data:IMAGE/JPG;charset=binary;BASE64,/9j/4A").unwrap();
    assert_eq!(jpg.media_type, "image/jpeg");

    let err = |s: &str| parse_image_data_uri(s).unwrap_err().to_string();
    assert!(
        err("https://example.com/a.png")
            .contains("image URLs are not fetched (https: refused)"),
        "{}",
        err("https://example.com/a.png")
    );
    assert!(err("http://x/a.png").contains("(http: refused)"));
    assert!(err("file:///tmp/a.png").contains("(file: refused)"));
    assert!(err("ftp://x/a.png").contains("not a data URI"));
    assert!(err("x").contains("not a data URI"));
    assert!(
        err("data:image/gif;base64,R0lGODlh")
            .contains("\"image/gif\" are not accepted")
    );
    assert!(
        err("data:text/plain;base64,aGk=").contains("\"text/plain\" are not accepted")
    );
    assert!(err("data:;base64,aGk=").contains("names no media type"));
    assert!(err("data:image/png,aGk=").contains("`;base64` is missing"));
    assert!(err("data:image/png;base64").contains("no ','"));
    assert!(err("data:image/png;base64,").contains("carries no data"));
    assert!(err("data:image/png;base64,***").contains("invalid character"));
}
