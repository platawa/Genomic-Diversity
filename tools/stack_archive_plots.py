#!/usr/bin/env python3
"""
stack_archive_plots.py

Compose the 9 per-locus archive plots (3 entropy + 6 drop-detection methods)
into a single stacked PNG per locus, with a unified header bar showing title
and a color legend for the on-plot exon/CDS/drop indicators.

Archive plot filename patterns:
  - HBB, NPS (newer Feb-8 runs): *.transitions_{method}.png
  - EGFR, ecoli (older Feb-6 runs): *.drops_{method}.png
All loci have: *.entropy_raw.png, *.entropy_smooth.png, *.entropy_boundaries.png

Output: results/_genome_wide/archive_stacked/<locus>_stacked.png
"""

import argparse
import glob
import logging
import os
import sys

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# Method order matches Feb-9 deck (images 2 and 3)
METHOD_ORDER = ["zscore", "mad", "win_shift", "local", "derivative", "cusum"]

ENTROPY_VIEWS = [
    ("entropy_raw", "Raw Entropy"),
    ("entropy_boundaries", "Entropy with Exon Boundaries"),
    ("entropy_smooth", "Smoothed Entropy"),
]

# Default locus → archive subdir + organism
LOCUS_REGISTRY = {
    "NPS":  ("human", "NPS_human_methods_cusum_derivative_local_mad_win_shift_zscore_20260208_150531"),
    "HBB":  ("human", "HBB_human_methods_cusum_derivative_local_mad_win_shift_zscore_20260208_150539"),
    "EGFR": ("human", "EGFR_human_methods_cusum_derivative_local_mad_win_shift_zscore_20260206_103051"),
    "ssrA": ("ecoli", "b2621_ecoli_methods_cusum_derivative_local_mad_win_shift_zscore_20260206_104434"),
    "rnpB": ("ecoli", "b3123_ecoli_methods_cusum_derivative_local_mad_win_shift_zscore_20260206_111951"),
}


def find_plot(plots_dir: str, suffix: str):
    """Find first PNG in plots_dir ending with <suffix>.png."""
    matches = glob.glob(os.path.join(plots_dir, f"*.{suffix}.png"))
    return matches[0] if matches else None


def get_method_plot(plots_dir: str, method: str):
    """Prefer transitions_<method>.png; fall back to drops_<method>.png."""
    p = find_plot(plots_dir, f"transitions_{method}")
    if p:
        return p, "transitions"
    p = find_plot(plots_dir, f"drops_{method}")
    if p:
        return p, "drops"
    return None, None


def _load_font(sizes=(22, 16, 14)):
    """Load a usable font, trying a few common paths. Returns (title, label, small)."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    title = label = small = None
    for path in candidates:
        if os.path.isfile(path):
            try:
                title = ImageFont.truetype(path, sizes[0])
                label = ImageFont.truetype(path, sizes[1])
                small = ImageFont.truetype(path, sizes[2])
                break
            except Exception:
                continue
    if title is None:
        # fall back to PIL default (tiny but works)
        title = label = small = ImageFont.load_default()
    return title, label, small


def render_header(width: int, locus_name: str, method_style: str,
                  height: int = 130) -> Image.Image:
    """Render a header bar with locus title and unified legend."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, label_font, small_font = _load_font()

    # Title
    draw.text((20, 10), f"{locus_name}  —  archive plot compilation",
              fill="black", font=title_font)

    # Legend row
    y = 55
    x = 20
    entries = [
        ("green_block",  "#2ca02c", "Exon regions (GTF track)"),
        ("orange_shade", "#ff7f0e", "CDS shaded interior"),
        ("red_dot",      "#d62728", "Detected drop (per-method)"),
        ("red_gradient", None,      "Drop-score confidence gradient"),
        ("blue_curve",   "#1f77b4", "Entropy signal"),
    ]
    for key, color, label in entries:
        if key == "green_block":
            draw.rectangle([x, y, x + 40, y + 12], fill=color)
        elif key == "orange_shade":
            draw.rectangle([x, y, x + 40, y + 12], fill=color)
        elif key == "red_dot":
            cx, cy, r = x + 20, y + 6, 6
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color,
                         outline="black")
        elif key == "red_gradient":
            for i in range(40):
                shade = int(230 - i * 3.5)
                draw.line([(x + i, y), (x + i, y + 12)],
                          fill=(shade, 30, 30))
        elif key == "blue_curve":
            draw.line([(x, y + 6), (x + 40, y + 6)], fill=color, width=2)

        draw.text((x + 48, y - 2), label, fill="black", font=small_font)
        x += 48 + int(8 * len(label)) + 30

    # Method style annotation
    draw.text((20, height - 28),
              f"Drop panels rendered from source files: {method_style}_<method>.png",
              fill="#444444", font=small_font)
    return img


def stack_locus(archive_dir: str, locus: str, output_dir: str):
    """Compose the 9 panels + header for one locus."""
    if locus not in LOCUS_REGISTRY:
        logger.warning(f"Locus {locus} not in registry. Skipping.")
        return False

    organism, subdir = LOCUS_REGISTRY[locus]
    plots_dir = os.path.join(archive_dir, organism, subdir, "plots")
    if not os.path.isdir(plots_dir):
        logger.warning(f"{locus}: plots dir not found at {plots_dir}")
        return False

    # Resolve the 9 panels
    panels = []
    for suffix, label in ENTROPY_VIEWS:
        path = find_plot(plots_dir, suffix)
        if not path:
            logger.warning(f"{locus}: missing {suffix}.png")
            continue
        panels.append((label, path))
    method_style = None
    for method in METHOD_ORDER:
        path, style = get_method_plot(plots_dir, method)
        if not path:
            logger.warning(f"{locus}: missing {method} plot")
            continue
        if method_style is None:
            method_style = style
        panels.append((f"Drop Detection: {method}", path))

    if not panels:
        logger.error(f"{locus}: no panels found")
        return False

    # Load images
    images = [Image.open(p).convert("RGB") for _, p in panels]
    max_w = max(im.width for im in images)

    # Resize each to max_w (keep aspect)
    resized = []
    for im in images:
        if im.width != max_w:
            new_h = int(im.height * max_w / im.width)
            im = im.resize((max_w, new_h), Image.LANCZOS)
        resized.append(im)

    # Header
    header = render_header(max_w, locus, method_style or "mixed")

    # Stack
    sep_h = 2
    total_h = header.height + sum(im.height + sep_h for im in resized)
    canvas = Image.new("RGB", (max_w, total_h), "white")
    y = 0
    canvas.paste(header, (0, y)); y += header.height
    for im in resized:
        canvas.paste(im, (0, y))
        # separator
        sep = Image.new("RGB", (max_w, sep_h), (200, 200, 200))
        canvas.paste(sep, (0, y + im.height))
        y += im.height + sep_h

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{locus}_stacked.png")
    canvas.save(out_path, format="PNG", optimize=True)
    logger.info(f"{locus}: wrote {out_path} ({canvas.width}×{canvas.height})")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive_dir", default="archive/outputs_single_gene/")
    p.add_argument("--output_dir", default="results/_genome_wide/archive_stacked/")
    p.add_argument("--loci", nargs="+",
                   default=list(LOCUS_REGISTRY.keys()),
                   help="Loci to stack. Default: all registered.")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    n_ok = 0
    for locus in args.loci:
        if stack_locus(args.archive_dir, locus, args.output_dir):
            n_ok += 1
    logger.info(f"Stacked {n_ok}/{len(args.loci)} loci. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
