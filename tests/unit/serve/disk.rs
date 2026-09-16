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

fn store_fake_durable(store: &mut DiskStore, tokens: &[u32], prefix_bytes: usize) -> String {
    let n = tokens.len();
    store
        .store_durable(
            tokens,
            None,
            &[n],
            &mut |w| Ok(w.write_all(&vec![9u8; prefix_bytes])?),
            &mut |pos, w| Ok(w.write_all(&(pos as u64).to_le_bytes())?),
        )
        .expect("store durable")
        .expect("kept")
}

fn ids(store: &DiskStore) -> Vec<String> {
    let mut ids: Vec<String> = store.entries().iter().map(|e| e.id.clone()).collect();
    ids.sort();
    ids
}

#[test]
fn durable_flag_round_trips_and_old_meta_files_load_as_not_durable() {
    let root = temp_root("durable-meta");
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    let plain = store_fake(&mut store, &[1, 2, 3], 10, Some("k")).expect("plain");
    let durable = store_fake_durable(&mut store, &[1, 2, 3, 4], 10);
    assert_eq!((store.len(), store.durable_len()), (2, 1));
    let flag = |store: &DiskStore, id: &str| store.entries().iter().find(|e| e.id == id).map(|e| e.durable);
    assert_eq!((flag(&store, &plain), flag(&store, &durable)), (Some(false), Some(true)));

    // The flag is in meta.json and survives a touch (which rewrites it).
    let meta_path = |id: &str| root.join("fmt").join(id).join(META);
    let read_meta = |id: &str| -> serde_json::Value { serde_json::from_slice(&fs::read(meta_path(id)).expect("meta")).expect("json") };
    assert_eq!(read_meta(&durable)["durable"], serde_json::Value::Bool(true));
    assert_eq!(read_meta(&plain)["durable"], serde_json::Value::Bool(false));
    store.touch(&durable);
    assert_eq!(read_meta(&durable)["durable"], serde_json::Value::Bool(true));

    // Reopening rebuilds the index with the flag.
    let reopened = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen");
    assert_eq!((flag(&reopened, &plain), flag(&reopened, &durable)), (Some(false), Some(true)));
    assert_eq!(reopened.durable_len(), 1);
    drop(reopened);

    // A meta file from before the field existed has no `durable` key: it is
    // an evicted session, not a durable entry.
    let mut old = read_meta(&durable);
    old.as_object_mut().expect("object").remove("durable");
    fs::write(meta_path(&durable), serde_json::to_vec(&old).expect("json")).expect("write");
    let legacy = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen legacy");
    assert_eq!(flag(&legacy, &durable), Some(false));
    assert_eq!(legacy.durable_len(), 0);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn the_durable_cap_evicts_the_least_recently_used_durable_entry_only() {
    let root = temp_root("durable-cap");
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    // An evicted session first: older than everything, yet never a victim of
    // the durable cap.
    let plain = store_fake(&mut store, &[100], 10, None).expect("plain");
    let durables: Vec<String> = (0..DURABLE_MAX_ENTRIES as u32).map(|i| store_fake_durable(&mut store, &[i, i + 1], 10)).collect();
    assert_eq!((store.len(), store.durable_len()), (DURABLE_MAX_ENTRIES + 1, DURABLE_MAX_ENTRIES));
    // A normal store on top does not count toward the cap and evicts nothing.
    let plain2 = store_fake(&mut store, &[200], 10, None).expect("plain2");
    assert_eq!((store.len(), store.durable_len()), (DURABLE_MAX_ENTRIES + 2, DURABLE_MAX_ENTRIES));
    drop(store);

    // Make the third durable entry the least recently used one (the plain
    // entries older still) and reopen so the index carries the timestamps.
    backdate(&root, &durables[2], 60);
    backdate(&root, &plain, 600);
    backdate(&root, &plain2, 600);
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen");
    let newest = store_fake_durable(&mut store, &[7, 7, 7], 10);
    assert_eq!((store.len(), store.durable_len()), (DURABLE_MAX_ENTRIES + 2, DURABLE_MAX_ENTRIES));
    let present = ids(&store);
    assert!(!present.contains(&durables[2]), "the least recently used durable entry goes: {present:?}");
    assert!(!root.join("fmt").join(&durables[2]).exists());
    for keep in durables.iter().filter(|id| **id != durables[2]).chain([&plain, &plain2, &newest]) {
        assert!(present.contains(keep), "{keep} must survive: {present:?}");
    }
    let _ = fs::remove_dir_all(&root);
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
