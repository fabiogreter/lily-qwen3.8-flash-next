#!/usr/bin/env python3
"""Deterministic synthetic test images for the vision reference goldens.

Flat backgrounds, a few rectangles and large text: they compress to a few tens
of kilobytes as PNG, are unambiguous under bicubic resampling (hard edges make
resampling differences visible, not hidden in noise), and the sizes exercise the
preprocessing rules the goldens must pin down:

  333 x 777   odd on both axes, rounds to 320 x 768 (multiples of 32)
  640 x 480   VGA, already a multiple of 32
  1920 x 1080 a native screenshot, 1080 rounds to 1088, passes the default cap
  3840 x 2160 a Retina capture, halved to 1920 x 1088 by the default cap

Writes the PNGs and `images.json` (size and sha256 per file) into
`tools/reference/images/`. The committed PNGs are the fixture; this script is
the record of how they were made and regenerates them bit for bit under the
same Pillow version (text rendering uses Pillow's bundled default font).

    .venv/bin/python tools/reference/make_images.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent / "images"

# (width, height, background, rectangles as (x0, y0, x1, y1, fill), text lines)
IMAGES = {
    "333x777": (
        333,
        777,
        (30, 60, 120),
        [(20, 40, 300, 200, (240, 200, 40)), (60, 300, 260, 520, (200, 40, 60)), (0, 700, 333, 777, (255, 255, 255))],
        [(24, 560, "lily", (255, 255, 255)), (24, 620, "333x777", (255, 255, 255))],
    ),
    "640x480": (
        640,
        480,
        (245, 245, 245),
        [(0, 0, 640, 60, (40, 40, 40)), (40, 100, 300, 300, (60, 140, 220)), (340, 100, 600, 300, (60, 180, 80))],
        [(40, 340, "Hello 640x480", (20, 20, 20)), (40, 400, "Button", (200, 40, 40))],
    ),
    "1920x1080": (
        1920,
        1080,
        (255, 255, 255),
        [
            (0, 0, 1920, 96, (24, 24, 32)),
            (0, 96, 320, 1080, (236, 238, 244)),
            (400, 180, 1100, 620, (66, 133, 244)),
            (1200, 180, 1840, 400, (219, 68, 55)),
            (1200, 460, 1840, 620, (15, 157, 88)),
        ],
        [(400, 700, "Screenshot 1920x1080", (0, 0, 0)), (400, 820, "Layout check", (90, 90, 90))],
    ),
    "3840x2160": (
        3840,
        2160,
        (18, 18, 18),
        [
            (0, 0, 3840, 160, (250, 250, 250)),
            (200, 300, 1800, 1900, (255, 193, 7)),
            (2000, 300, 3640, 1000, (3, 169, 244)),
            (2000, 1200, 3640, 1900, (156, 39, 176)),
        ],
        [(240, 400, "Retina 3840x2160", (0, 0, 0)), (240, 640, "cap test", (0, 0, 0))],
    ),
}


def render(width: int, height: int, background, rectangles, texts) -> Image.Image:
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    for x0, y0, x1, y1, fill in rectangles:
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=fill)
    font = ImageFont.load_default(size=max(24, height // 12))
    for x, y, text, fill in texts:
        draw.text((x, y), text, fill=fill, font=font)
    return image


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {}
    for name, (width, height, background, rectangles, texts) in IMAGES.items():
        path = OUT_DIR / f"{name}.png"
        render(width, height, background, rectangles, texts).save(path, format="PNG", optimize=True)
        data = path.read_bytes()
        record[name] = {
            "file": path.name,
            "width": width,
            "height": height,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        print(f"{path.name}: {width}x{height}, {len(data)} bytes")
    (OUT_DIR / "images.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
