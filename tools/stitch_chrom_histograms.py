#!/usr/bin/env python3
"""
stitch_chrom_histograms.py

Tile the already-rendered per-chromosome histogram_comparison.png files
into a single stacked image (24 rows x 1 column of 3-panel figures).

No activation recomputation — just PIL paste. Takes seconds.
"""

import argparse
import os
import sys

from PIL import Image

DEFAULT_CHROMS = [
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8",
    "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16",
    "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_dir", required=True,
                   help="dir containing {chrom}/histogram_comparison.png")
    p.add_argument("--output", required=True, help="output PNG path")
    p.add_argument("--chroms", nargs="+", default=DEFAULT_CHROMS)
    p.add_argument("--scale", type=float, default=0.5,
                   help="uniform downscale factor (1.0 = native 2160px wide)")
    args = p.parse_args()

    tiles = []
    missing = []
    for c in args.chroms:
        path = os.path.join(args.source_dir, c, "histogram_comparison.png")
        if not os.path.isfile(path):
            missing.append(c)
            continue
        img = Image.open(path).convert("RGBA")
        if args.scale != 1.0:
            w, h = img.size
            img = img.resize((int(w * args.scale), int(h * args.scale)),
                             Image.LANCZOS)
        tiles.append((c, img))

    if not tiles:
        print("ERROR: no histograms found", file=sys.stderr)
        return 2

    w = max(img.size[0] for _, img in tiles)
    total_h = sum(img.size[1] for _, img in tiles)
    canvas = Image.new("RGBA", (w, total_h), (255, 255, 255, 255))
    y = 0
    for c, img in tiles:
        canvas.paste(img, (0, y))
        y += img.size[1]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    canvas.convert("RGB").save(args.output, optimize=True)
    print(f"Wrote {args.output} ({canvas.size[0]}x{canvas.size[1]}, "
          f"{len(tiles)} chroms)")
    if missing:
        print(f"Missing: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
