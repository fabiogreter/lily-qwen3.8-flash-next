use super::*;

#[test]
fn sha256_matches_the_fips_vectors() {
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // Two blocks, with the length field crossing into the second.
    let long = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    assert_eq!(
        hex(&sha256(long)),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    // The 896-bit vector (four blocks).
    let longer = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
    assert_eq!(
        hex(&sha256(longer)),
        "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
    );
    // One million 'a's, fed in uneven pieces.
    let million = vec![b'a'; 1_000_000];
    assert_eq!(
        hex(&sha256(&million)),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

#[test]
fn incremental_updates_give_the_same_digest_as_one_slice() {
    let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 % 251) as u8).collect();
    let whole = sha256(&data);
    for pieces in [1usize, 3, 63, 64, 65, 127, 128, 129, 999] {
        let mut h = Sha256::new();
        for chunk in data.chunks(pieces) {
            h.update(chunk);
        }
        assert_eq!(h.finish(), whole, "pieces of {pieces}");
    }
    // Every tail length around the block boundary pads correctly.
    for n in 50..70 {
        let mut h = Sha256::new();
        h.update(&data[..n]);
        assert_eq!(h.finish(), sha256(&data[..n]), "length {n}");
    }
}
