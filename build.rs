//! Embeds the built dashboard (`dashboard/dist/`, committed) into the binary:
//! writes `$OUT_DIR/dashboard_assets.rs`, a table of every file under it with
//! its path relative to `dist/`, its content type and its bytes
//! (`include_bytes!`), which `src/serve/dashboard.rs` serves. `cargo build`
//! never needs node; a stale `dist/` is caught by the dashboard's own check
//! (`npm --prefix dashboard run check:dist`).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// The content types the dashboard's files may have, by extension. A file
/// with any other extension fails the build: serving it as a guess would be
/// worse than noticing.
const TYPES: &[(&str, &str)] = &[
    ("html", "text/html; charset=utf-8"),
    ("js", "text/javascript; charset=utf-8"),
    ("css", "text/css; charset=utf-8"),
    ("woff2", "font/woff2"),
    ("txt", "text/plain; charset=utf-8"),
    ("svg", "image/svg+xml"),
    ("json", "application/json"),
    ("png", "image/png"),
];

fn main() {
    let root =
        PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let dist = root.join("dashboard").join("dist");
    println!("cargo:rerun-if-changed={}", dist.display());
    if !dist.join("index.html").is_file() {
        panic!(
            "{} is missing: build the dashboard first with `npm --prefix dashboard run build` \
             (dashboard/dist/ is committed; a fresh checkout has it)",
            dist.join("index.html").display()
        );
    }
    let mut files = Vec::new();
    collect(&dist, &mut files);
    files.sort();
    let mut table = String::from(
        "/// Every file of `dashboard/dist/`: path relative to it, content type, bytes.\n\
         pub static FILES: &[(&str, &str, &[u8])] = &[\n",
    );
    for path in &files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path
            .strip_prefix(&dist)
            .expect("under dist")
            .to_str()
            .unwrap_or_else(|| panic!("{} is not UTF-8", path.display()))
            .replace('\\', "/");
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let Some((_, content_type)) = TYPES.iter().find(|(e, _)| *e == extension)
        else {
            panic!(
                "{}: no content type for .{extension} files; add it to build.rs",
                path.display()
            );
        };
        writeln!(
            table,
            "    ({relative:?}, {content_type:?}, include_bytes!({:?})),",
            path.display().to_string()
        )
        .expect("write to a string");
    }
    table.push_str("];\n");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("out dir"));
    std::fs::write(out.join("dashboard_assets.rs"), table)
        .expect("writing dashboard_assets.rs");
}

/// Every regular file below `dir`, directories descended into (rerun on each,
/// so an added or removed file is noticed). Hidden files (a Finder
/// `.DS_Store`) are not part of the build output and are skipped.
fn collect(dir: &Path, files: &mut Vec<PathBuf>) {
    println!("cargo:rerun-if-changed={}", dir.display());
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("listing {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.expect("directory entry");
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            collect(&path, files);
        } else if path.is_file() {
            files.push(path);
        }
    }
}
