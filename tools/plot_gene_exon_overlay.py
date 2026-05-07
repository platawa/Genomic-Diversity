#!/usr/bin/env python3
"""
plot_gene_exon_overlay.py

Overlay a gene's exons on a pre-computed SAE latent embedding (tSNE or UMAP).
SAE windows overlapping an exon are colored; all other windows are gray.

Color modes (--color-mode):
  single    : one color for every exon-overlap window (gene-specific).
  position  : three colors — first exon / middle exons / last exon.
              An exon is "first" if it has exon_number=1 in any transcript;
              "last" if exon_number==max(exon_number) in any transcript;
              otherwise "middle". Priority first > last > middle for
              overlaps across isoforms.
  gradient  : continuous colormap (plasma) from the earliest to the latest
              unique exon, ordered by genomic start position.
  numbered  : single color + text annotation at each dot showing the
              1-based rank of the overlapping exon (sorted by genomic start).

Consumes:
  - A loci TSV (columns: organism, chrom, start, end, name, scoring_run)
  - A GRCh38 GTF (for exon interval lookup by gene symbol)
  - A latent-analysis directory with cluster_assignments.tsv +
    embedding_{tsne,umap}.npy

Produces: one PNG per invocation.
"""

import argparse
import logging
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_tsne_by_annotation import CHROM_MAP  # chr<N> -> RefSeq NC_* accession

logger = logging.getLogger(__name__)

# Gene -> color for single-color mode (consistent across all plots)
GENE_COLORS = {
    "HBB":  "#d62728",   # red
    "EGFR": "#1f77b4",   # blue
    "NPS":  "#2ca02c",   # green
}
DEFAULT_GENE_COLOR = "#ff7f0e"  # orange fallback

# Colors for position mode (colorblind-friendly)
POSITION_COLORS = {
    "first":  "#d62728",  # red
    "middle": "#ff7f0e",  # orange
    "last":   "#1f77b4",  # blue
}

GRADIENT_CMAP = "plasma"

TRANSCRIPT_RE = re.compile(r'transcript_id "([^"]+)"')
EXON_NUMBER_RE = re.compile(r'exon_number "?(\d+)"?')


def load_loci(loci_path):
    df = pd.read_csv(loci_path, sep="\t", dtype=str).dropna(how="all")
    required = {"chrom", "start", "end", "name"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"loci file missing columns: {missing}")
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df


def gene_row(loci_df, gene_name):
    hits = loci_df[loci_df["name"] == gene_name]
    if hits.empty:
        raise SystemExit(
            f"Gene '{gene_name}' not found in loci file. "
            f"Available: {sorted(loci_df['name'].unique().tolist())}"
        )
    return hits.iloc[0]


def load_gene_exons(gtf_path, chrom_id, gene_name, gene_start, gene_end):
    """Return a list of exon records for a gene, ordered by genomic start.

    Each record is a dict:
        {"rank": 1-based rank by genomic start,
         "start": int, "end": int,
         "position_class": "first"|"middle"|"last"}

    position_class reflects whether the (start, end) interval appears as
    exon_number=1 in any transcript ("first"), or exon_number=max in any
    transcript ("last"). Priority: first > last > middle.
    """
    per_transcript = defaultdict(list)   # tid -> list[(exon_number, start, end)]
    pad = 50_000
    window_start = max(1, gene_start - pad)
    window_end = gene_end + pad
    # NCBI GTF uses `gene "SYMBOL"`; GENCODE uses `gene_name "SYMBOL"`.
    attr_variants = (f'gene "{gene_name}"', f'gene_name "{gene_name}"')

    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            if parts[0] != chrom_id or parts[2] != "exon":
                continue
            s, e = int(parts[3]), int(parts[4])
            if e < window_start or s > window_end:
                continue
            attrs = parts[8]
            if not any(v in attrs for v in attr_variants):
                continue
            tm = TRANSCRIPT_RE.search(attrs)
            em = EXON_NUMBER_RE.search(attrs)
            if not tm:
                continue
            tid = tm.group(1)
            enum = int(em.group(1)) if em else None
            per_transcript[tid].append((enum, s, e))

    # Classify each unique (start, end) interval.
    flags = {}  # (s, e) -> {"is_first": bool, "is_last": bool}
    for tid, items in per_transcript.items():
        nums = [n for n, _, _ in items if n is not None]
        if nums:
            min_n, max_n = min(nums), max(nums)
        else:
            min_n = max_n = None
        for enum, s, e in items:
            d = flags.setdefault((s, e), {"is_first": False, "is_last": False})
            if enum is not None and enum == min_n:
                d["is_first"] = True
            if enum is not None and enum == max_n:
                d["is_last"] = True

    ordered = sorted(flags.keys())
    exons_out = []
    for rank, (s, e) in enumerate(ordered, start=1):
        d = flags[(s, e)]
        if d["is_first"]:
            pc = "first"
        elif d["is_last"]:
            pc = "last"
        else:
            pc = "middle"
        exons_out.append({
            "rank": rank, "start": s, "end": e, "position_class": pc,
        })
    return exons_out


def regions_primary_exon(region_starts, region_ends, exons):
    """For each region, return (overlap_mask, primary_exon_idx_in_list).

    The "primary" exon for an overlapping region is the first exon in the
    rank-ordered list it intersects (earliest by genomic start).
    `primary_idx` is -1 for non-overlapping regions.
    """
    n = len(region_starts)
    mask = np.zeros(n, dtype=bool)
    primary_idx = np.full(n, -1, dtype=np.int32)
    if not exons:
        return mask, primary_idx
    rs = np.asarray(region_starts, dtype=np.int64)
    re_ = np.asarray(region_ends, dtype=np.int64)
    for i, ex in enumerate(exons):  # iterate in rank order
        hits = (re_ > ex["start"]) & (rs < ex["end"])
        new_hits = hits & (~mask)
        primary_idx[new_hits] = i
        mask |= hits
    return mask, primary_idx


def load_embeddings_and_metadata(data_dir, variant):
    ca_path = os.path.join(data_dir, "cluster_assignments.tsv")
    if not os.path.isfile(ca_path):
        raise SystemExit(f"cluster_assignments.tsv not found: {ca_path}")
    ca = pd.read_csv(ca_path, sep="\t", comment="#")
    for col in ("genomic_start", "genomic_end"):
        if col not in ca.columns:
            raise SystemExit(f"{ca_path} missing required column: {col}")

    emb_path = os.path.join(data_dir, f"embedding_{variant}.npy")
    if os.path.isfile(emb_path):
        coords = np.load(emb_path)
        logger.info(f"Loaded {variant} embedding: {coords.shape}")
    else:
        c1, c2 = f"{variant}_1", f"{variant}_2"
        if c1 in ca.columns and c2 in ca.columns:
            coords = ca[[c1, c2]].values.astype(np.float32)
            logger.info(f"Using {variant} from TSV columns ({coords.shape})")
        else:
            raise SystemExit(f"Embedding not found: {emb_path} (and no {c1}/{c2} in TSV)")

    if len(coords) != len(ca):
        raise SystemExit(
            f"Embedding rows ({len(coords)}) != cluster_assignments rows ({len(ca)})."
        )
    return ca, coords


def _plot_background(ax, coords, mask):
    if not mask.any():
        return
    bg_coords = coords[mask]
    bg_size = 1.5 if len(bg_coords) > 100_000 else 3.0
    ax.scatter(
        bg_coords[:, 0], bg_coords[:, 1],
        c="#cccccc", s=bg_size, alpha=0.25,
        edgecolors="none", rasterized=True,
        label=f"Other ({len(bg_coords):,})",
    )


def plot_overlay(coords, highlight_mask, primary_idx, scope_mask, exons,
                 out_path, gene_name, variant, scope_label, norm_label,
                 chrom_label, color_mode):
    """Render a single latent-space scatter with one of four coloring modes."""
    n_total = int(scope_mask.sum())
    n_hi = int((highlight_mask & scope_mask).sum())
    prefix = variant.upper()

    fig, ax = plt.subplots(figsize=(11, 9))
    _plot_background(ax, coords, scope_mask & (~highlight_mask))

    hi_mask = highlight_mask & scope_mask
    hi_coords = coords[hi_mask]
    hi_primary = primary_idx[hi_mask]

    extra_title = ""

    if len(hi_coords) == 0:
        logger.warning(f"No windows overlap {gene_name} exons in scope={scope_label}")

    elif color_mode == "single":
        color = GENE_COLORS.get(gene_name, DEFAULT_GENE_COLOR)
        ax.scatter(
            hi_coords[:, 0], hi_coords[:, 1],
            c=color, s=28, alpha=0.95,
            edgecolors="black", linewidths=0.3, zorder=5,
            label=f"{gene_name} exon windows ({len(hi_coords):,})",
        )

    elif color_mode == "position":
        # Count per class for legend
        classes = np.array([exons[i]["position_class"] for i in hi_primary])
        for cls in ("first", "middle", "last"):
            m = classes == cls
            if not m.any():
                continue
            ax.scatter(
                hi_coords[m, 0], hi_coords[m, 1],
                c=POSITION_COLORS[cls], s=34, alpha=0.95,
                edgecolors="black", linewidths=0.3, zorder=5,
                label=f"{cls} exon ({int(m.sum())})",
            )
        n_first = int((classes == "first").sum())
        n_mid = int((classes == "middle").sum())
        n_last = int((classes == "last").sum())
        extra_title = f"first={n_first} middle={n_mid} last={n_last}"

    elif color_mode == "gradient":
        ranks = np.array([exons[i]["rank"] for i in hi_primary])
        n_ex = len(exons)
        norm = Normalize(vmin=1, vmax=max(n_ex, 2))
        cmap = plt.get_cmap(GRADIENT_CMAP)
        ax.scatter(
            hi_coords[:, 0], hi_coords[:, 1],
            c=ranks, cmap=cmap, norm=norm,
            s=34, alpha=0.95,
            edgecolors="black", linewidths=0.3, zorder=5,
        )
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(f"Exon rank (1..{n_ex}, by genomic start)", fontsize=10)
        extra_title = f"{len(hi_coords):,} windows, {n_ex} exons (gradient)"

    elif color_mode == "numbered":
        color = GENE_COLORS.get(gene_name, DEFAULT_GENE_COLOR)
        ax.scatter(
            hi_coords[:, 0], hi_coords[:, 1],
            c=color, s=34, alpha=0.95,
            edgecolors="black", linewidths=0.3, zorder=5,
            label=f"{gene_name} exon windows ({len(hi_coords):,})",
        )
        # Annotate with rank of the primary exon
        for j, (x, y) in enumerate(hi_coords):
            rank = exons[hi_primary[j]]["rank"]
            ax.annotate(
                str(rank), (x, y),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, fontweight="bold",
                color="black", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    else:
        raise SystemExit(f"Unknown color_mode: {color_mode}")

    ax.set_xlabel(f"{prefix} 1", fontsize=11)
    ax.set_ylabel(f"{prefix} 2", fontsize=11)
    title = (
        f"{prefix} — {gene_name} exon overlay ({chrom_label})  "
        f"[{color_mode}]\n"
        f"scope={scope_label}, norm={norm_label}, "
        f"{len(exons)} exon intervals, "
        f"{n_hi:,} / {n_total:,} windows highlighted"
    )
    if extra_title:
        title += f"  |  {extra_title}"
    ax.set_title(title, fontsize=11.5)

    # Legend only for modes with categorical labels
    if color_mode in ("single", "position", "numbered"):
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay a gene's exons on a SAE tSNE/UMAP embedding.",
    )
    parser.add_argument("--loci", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--embedding-dir", required=True,
                        help="Directory with cluster_assignments.tsv and embedding_{tsne,umap}.npy")
    parser.add_argument("--variant", required=True, choices=["tsne", "umap"])
    parser.add_argument("--scope", required=True, choices=["whole", "chromosome"])
    parser.add_argument("--gene-name", required=True)
    parser.add_argument("--norm-label", required=True,
                        help="String shown in the plot title (prenorm/postnorm/raw)")
    parser.add_argument("--color-mode", default="single",
                        choices=["single", "position", "gradient", "numbered"])
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    loci = load_loci(args.loci)
    row = gene_row(loci, args.gene_name)
    chrom_label = row["chrom"]
    gene_start, gene_end = int(row["start"]), int(row["end"])
    gtf_chrom = CHROM_MAP.get(chrom_label, chrom_label)
    logger.info(
        f"Gene {args.gene_name}: {chrom_label} ({gtf_chrom}) "
        f"{gene_start:,}-{gene_end:,}"
    )

    exons = load_gene_exons(args.gtf, gtf_chrom, args.gene_name, gene_start, gene_end)
    n_first = sum(1 for e in exons if e["position_class"] == "first")
    n_mid = sum(1 for e in exons if e["position_class"] == "middle")
    n_last = sum(1 for e in exons if e["position_class"] == "last")
    logger.info(
        f"Loaded {len(exons)} unique exon intervals for {args.gene_name} "
        f"(first={n_first}, middle={n_mid}, last={n_last})"
    )

    ca, coords = load_embeddings_and_metadata(args.embedding_dir, args.variant)

    if args.scope == "whole":
        scope_mask = np.ones(len(ca), dtype=bool)
    else:
        if "chrom" not in ca.columns:
            raise SystemExit(
                "scope=chromosome requires a 'chrom' column in cluster_assignments.tsv."
            )
        scope_mask = (ca["chrom"].values == chrom_label)
        if scope_mask.sum() == 0:
            raise SystemExit(
                f"No windows in cluster_assignments.tsv belong to {chrom_label}."
            )

    # Overlap is always computed across only same-chrom windows (the rest
    # physically can't overlap this gene's exons regardless of scope).
    same_chrom_mask = (
        (ca["chrom"].values == chrom_label) if "chrom" in ca.columns
        else np.ones(len(ca), dtype=bool)
    )
    highlight_mask = np.zeros(len(ca), dtype=bool)
    primary_idx = np.full(len(ca), -1, dtype=np.int32)
    if exons:
        idxs = np.where(same_chrom_mask)[0]
        om, pi = regions_primary_exon(
            ca["genomic_start"].values[idxs],
            ca["genomic_end"].values[idxs],
            exons,
        )
        highlight_mask[idxs[om]] = True
        primary_idx[idxs[om]] = pi[om]
    logger.info(
        f"Scope windows: {int(scope_mask.sum()):,} | "
        f"highlight windows: {int(highlight_mask.sum()):,} | "
        f"color-mode: {args.color_mode}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    plot_overlay(
        coords=coords,
        highlight_mask=highlight_mask,
        primary_idx=primary_idx,
        scope_mask=scope_mask,
        exons=exons,
        out_path=args.output,
        gene_name=args.gene_name,
        variant=args.variant,
        scope_label=args.scope,
        norm_label=args.norm_label,
        chrom_label=chrom_label,
        color_mode=args.color_mode,
    )


if __name__ == "__main__":
    main()
