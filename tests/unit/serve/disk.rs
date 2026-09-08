use std::io::Read;

use super::*;

fn temp_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("lily-disk-test-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    root
}

fn store_fake(store: &mut DiskStore, tokens: &[u32], prefix_bytes: usize, key: Option<&str>) -> Option<String> {
    let n = tokens.len();
    store
        .store(
            tokens,
            key,
            &[n],
            &mut |w| Ok(w.write_all(&vec![7u8; prefix_bytes])?),
            &mut |pos, w| Ok(w.write_all(&(pos as u64).to_le_bytes())?),
        )
        .expect("store")
}

#[test]
fn store_reopen_and_read_back() {
    let root = temp_root("roundtrip");
    let mut store = DiskStore::open(&root, "fmt-a", 1 << 20, 0).expect("open");
    assert!(store.is_empty());
    let id = store_fake(&mut store, &[1, 2, 3, 4], 100, Some("conv")).expect("stored");
    assert_eq!(store.len(), 1);
    assert_eq!(store.used_bytes(), 108);

    // A second store instance indexes the entry from disk.
    let store2 = DiskStore::open(&root, "fmt-a", 1 << 20, 0).expect("reopen");
    assert_eq!(store2.len(), 1);
    let e = &store2.entries()[0];
    assert_eq!((e.id.as_str(), e.tokens.as_slice(), e.checkpoints.as_slice(), e.cache_key.as_deref()), (id.as_str(), &[1, 2, 3, 4][..], &[4][..], Some("conv")));
    let mut prefix = Vec::new();
    store2.open_prefix(&id).expect("prefix").read_to_end(&mut prefix).expect("read");
    assert_eq!(prefix, vec![7u8; 100]);
    let mut ckpt = Vec::new();
    store2.open_checkpoint(&id, 4).expect("ckpt").read_to_end(&mut ckpt).expect("read");
    assert_eq!(ckpt, 4u64.to_le_bytes());

    // Another format shares the root but sees nothing.
    let other = DiskStore::open(&root, "fmt-b", 1 << 20, 0).expect("open other");
    assert!(other.is_empty());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn budget_evicts_least_recently_used_and_refuses_oversize() {
    let root = temp_root("budget");
    // Entries are 308 bytes (300 of prefix, 8 of checkpoint): two fit, three do not.
    let mut store = DiskStore::open(&root, "fmt", 700, 0).expect("open");
    let a = store_fake(&mut store, &[1], 300, None).expect("a");
    let b = store_fake(&mut store, &[2], 300, None).expect("b");
    // Touch a so b becomes the oldest.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    store.touch(&a);
    let c = store_fake(&mut store, &[3], 300, None).expect("c");
    let ids: Vec<&str> = store.entries().iter().map(|e| e.id.as_str()).collect();
    assert!(ids.contains(&a.as_str()) && ids.contains(&c.as_str()) && !ids.contains(&b.as_str()), "{ids:?}");
    assert!(!root.join("fmt").join(&b).exists());
    // Larger than the whole budget: not kept, nothing else evicted.
    assert!(store_fake(&mut store, &[4], 2000, None).is_none());
    assert_eq!(store.len(), 2);
    assert!(!root.join("fmt").join("s4").exists());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn corrupt_entries_are_dropped_on_open() {
    let root = temp_root("corrupt");
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    let a = store_fake(&mut store, &[1, 2], 10, None).expect("a");
    let b = store_fake(&mut store, &[3, 4], 10, None).expect("b");
    fs::remove_file(root.join("fmt").join(&b).join("ckpt-2.bin")).expect("remove");
    let reopened = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen");
    let ids: Vec<&str> = reopened.entries().iter().map(|e| e.id.as_str()).collect();
    assert_eq!(ids, vec![a.as_str()]);
    assert!(!root.join("fmt").join(&b).exists());
    // Ids continue past the ones seen on disk.
    let mut reopened = reopened;
    let c = store_fake(&mut reopened, &[5], 10, None).expect("c");
    assert_ne!(c, a);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn remove_deletes_files() {
    let root = temp_root("remove");
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    let a = store_fake(&mut store, &[1], 10, None).expect("a");
    assert!(root.join("fmt").join(&a).is_dir());
    store.remove(&a);
    assert!(store.is_empty());
    assert!(!root.join("fmt").join(&a).exists());
    let _ = fs::remove_dir_all(&root);
}

/// Backdates an entry's last use by rewriting its meta file.
fn backdate(root: &Path, id: &str, seconds: u64) {
    let path = root.join("fmt").join(id).join(META);
    let mut meta: Meta = serde_json::from_slice(&fs::read(&path).expect("meta")).expect("parse");
    meta.last_used -= seconds;
    fs::write(&path, serde_json::to_vec(&meta).expect("json")).expect("write");
}

#[test]
fn entries_expire_after_the_maximum_age() {
    let root = temp_root("ttl");
    let day = 86_400;
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 3 * day).expect("open");
    let old = store_fake(&mut store, &[1], 10, None).expect("old");
    let fresh = store_fake(&mut store, &[2], 10, None).expect("fresh");
    backdate(&root, &old, 4 * day);
    // The in-memory index still holds the old timestamp until reopened; a
    // reopen applies the age limit.
    let reopened = DiskStore::open(&root, "fmt", 1 << 20, 3 * day).expect("reopen");
    let ids: Vec<&str> = reopened.entries().iter().map(|e| e.id.as_str()).collect();
    assert_eq!(ids, vec![fresh.as_str()]);
    assert!(!root.join("fmt").join(&old).exists());
    drop(reopened);
    backdate(&root, &fresh, 4 * day);
    let mut relisted = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen without ttl");
    assert_eq!(relisted.len(), 1, "no age limit keeps it");
    relisted.expire();
    assert_eq!(relisted.len(), 1, "max_age 0 never expires");
    drop(relisted);
    let live = DiskStore::open(&root, "fmt", 1 << 20, 3 * day).expect("reopen");
    assert!(live.is_empty());
    let _ = fs::remove_dir_all(&root);
}
