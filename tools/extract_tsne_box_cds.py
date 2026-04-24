#!/usr/bin/env python3
"""
extract_tsne_box_cds.py

Given a t-SNE plot of SAE region fingerprints, extract the CDS-annotated
regions (red dots) falling inside user-specified bounding boxes, and
cross-reference them with GTF gene annotations.

Inputs:
    - results/<chrom>/sae/latent_analysis_*/data/cluster_assignments.tsv
    - results/<chrom>/sae/latent_analysis_*/data/embedding_tsne.npy
    - GTF file for the organism

Outputs (written to --output_dir, default: <latent_dir>/box_extraction/):
    - cds_in_box_<box_idx>.tsv    — per-box TSV of CDS regions with coords + genes
    - cds_in_box_summary.tsv      — combined summary across all boxes
    - tsne_boxes_<chrom>.png      — t-SNE plot with boxes drawn and CDS-in-box highlighted
    - box_extraction.log          — stdout log

A "box" is specified as x1,y1,x2,y2 (t-SNE coords); pass --box once per box.

Example (chr2 boxes from screenshot):
    python tools/extract_tsne_box_cds.py \\
        --latent_dir results/chr2/sae/latent_analysis_postnorm \\
        --gtf /path/to/GRCh38_genomic.gtf \\
        --chrom chr2 \\
        --box -50,-30,-25,15 \\
        --box 15,-70,50,-55
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Reuse GTF / classification helpers from the sibling tool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identify_tsne_clusters import (
    CHROM_MAP,
    load_gtf_features,
    classify_region,
    find_overlapping_genes,
)


def parse_box(spec):
    """Parse 'x1,y1,x2,y2' into a normalized (xmin,ymin,xmax,ymax) tuple."""
    parts = [float(p) for p in spec.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--box expects 'x1,y1,x2,y2', got {spec!r}")
    x1, y1, x2, y2 = parts
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latent_dir", required=True,
                    help="Path to latent_analysis directory "
                         "(e.g. results/chr2/sae/latent_analysis_postnorm)")
    ap.add_argument("--gtf", required=True, help="Path to GTF file")
    ap.add_argument("--chrom", required=True, help="Chromosome name (e.g. chr2)")
    ap.add_argument("--box", action="append", type=parse_box, required=True,
                    help="Bounding box in t-SNE coords as 'x1,y1,x2,y2'. "
                         "Repeat --box for multiple boxes.")
    ap.add_argument("--output_dir", default=None,
                    help="Output dir (default: <latent_dir>/box_extraction/)")
    ap.add_argument("--annotation_filter", default="CDS",
                    help="Annotation class to extract (default: CDS). "
                         "Use 'ALL' to keep every annotation.")
    args = ap.parse_args()

    data_dir = os.path.join(args.latent_dir, "data")
    ca_path = os.path.join(data_dir, "cluster_assignments.tsv")
    emb_path = os.path.join(data_dir, "embedding_tsne.npy")
    for p in (ca_path, args.gtf):
        if not os.path.exists(p):
            print(f"ERROR: missing {p}")
            sys.exit(1)

    ca = pd.read_csv(ca_path, sep="\t", comment="#")
    print(f"Loaded {len(ca)} regions from {ca_path}")

    # Attach t-SNE coords if not already columns
    if "tsne_1" not in ca.columns or "tsne_2" not in ca.columns:
        if not os.path.exists(emb_path):
            print(f"ERROR: no tsne cols and {emb_path} missing")
            sys.exit(1)
        emb = np.load(emb_path)
        if emb.shape[0] != len(ca):
            print(f"ERROR: embedding rows ({emb.shape[0]}) != ca rows ({len(ca)})")
            sys.exit(1)
        ca["tsne_1"] = emb[:, 0]
        ca["tsne_2"] = emb[:, 1]

    out_dir = args.output_dir or os.path.join(args.latent_dir, "box_extraction")
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: union of all boxes -> mask of candidate rows BEFORE classification
    # (huge speedup vs classifying all 50k regions)
    box_mask = np.zeros(len(ca), dtype=bool)
    for (xmin, ymin, xmax, ymax) in args.box:
        box_mask |= (
            (ca["tsne_1"].values >= xmin) & (ca["tsne_1"].values <= xmax) &
            (ca["tsne_2"].values >= ymin) & (ca["tsne_2"].values <= ymax)
        )
    candidates = ca[box_mask].copy()
    print(f"Candidates inside at least one box: {len(candidates)} / {len(ca)} "
          f"({100*len(candidates)/len(ca):.1f}%)", flush=True)

    chrom_id = CHROM_MAP.get(args.chrom, args.chrom)
    print(f"Loading GTF features for {args.chrom} ({chrom_id})...", flush=True)
    intervals, gene_names = load_gtf_features(args.gtf, chrom_id)
    print(f"  {len(intervals['gene'])} genes, {len(intervals['CDS'])} CDS, "
          f"{len(intervals['exon'])} exons", flush=True)

    # Classify ONLY the candidate rows (much smaller than 50k)
    print(f"Classifying {len(candidates)} candidate regions...", flush=True)
    anns = []
    genes_per_row = []
    for _, row in candidates.iterrows():
        s, e = int(row["genomic_start"]), int(row["genomic_end"])
        anns.append(classify_region(s, e, intervals))
        genes_per_row.append(find_overlapping_genes(s, e, intervals, gene_names))
    candidates["annotation"] = anns
    candidates["overlapping_genes"] = ["; ".join(g[:5]) for g in genes_per_row]

    # Filter to target annotation
    if args.annotation_filter.upper() == "ALL":
        target = candidates
        label = "ALL"
    else:
        target = candidates[candidates["annotation"] == args.annotation_filter]
        label = args.annotation_filter
    print(f"{len(target)} candidates classified as {label}", flush=True)

    # Per-box extraction
    summary_rows = []
    per_box_frames = []
    for i, box in enumerate(args.box, start=1):
        xmin, ymin, xmax, ymax = box
        in_box = target[
            (target["tsne_1"] >= xmin) & (target["tsne_1"] <= xmax) &
            (target["tsne_2"] >= ymin) & (target["tsne_2"] <= ymax)
        ].copy()
        in_box.insert(0, "box_id", i)
        per_box_frames.append(in_box)

        # Per-box TSV
        cols_front = ["box_id", "annotation", "overlapping_genes",
                      "genomic_start", "genomic_end", "tsne_1", "tsne_2"]
        cols_extra = [c for c in in_box.columns if c not in cols_front]
        in_box = in_box[cols_front + cols_extra]
        tsv_path = os.path.join(out_dir, f"cds_in_box_{i}.tsv")
        in_box.to_csv(tsv_path, sep="\t", index=False)

        # Top gene counts
        gene_counter = Counter()
        for g_str in in_box["overlapping_genes"]:
            if g_str:
                for g in g_str.split("; "):
                    gene_counter[g] += 1

        print(f"\n--- Box {i}  x=[{xmin:.1f},{xmax:.1f}]  y=[{ymin:.1f},{ymax:.1f}] ---")
        print(f"  {label} regions in box: {len(in_box)}")
        if len(in_box):
            gspan = (in_box['genomic_start'].min(), in_box['genomic_end'].max())
            print(f"  Genomic span: {gspan[0]:,} - {gspan[1]:,}")
            if "cluster" in in_box.columns:
                c_counts = in_box["cluster"].value_counts().head(5)
                print(f"  Top Leiden clusters: " +
                      ", ".join(f"{c}({n})" for c, n in c_counts.items()))
            print(f"  Unique genes: {len(gene_counter)}")
            top_n = min(20, len(gene_counter))
            if top_n:
                print(f"  Top {top_n} genes by region count:")
                for gene, cnt in gene_counter.most_common(top_n):
                    print(f"    {gene}: {cnt}")
        print(f"  -> {tsv_path}")

        summary_rows.append({
            "box_id": i,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "n_regions": len(in_box),
            "n_unique_genes": len(gene_counter),
            "genomic_start_min": int(in_box['genomic_start'].min()) if len(in_box) else -1,
            "genomic_end_max": int(in_box['genomic_end'].max()) if len(in_box) else -1,
            "top_genes": ", ".join(g for g, _ in gene_counter.most_common(10)),
        })

    # Combined summary
    summary_df = pd.DataFrame(summary_rows)
    sum_path = os.path.join(out_dir, "cds_in_box_summary.tsv")
    summary_df.to_csv(sum_path, sep="\t", index=False)
    print(f"\nSummary: {sum_path}")

    # Combined TSV of all boxes
    if per_box_frames:
        combined = pd.concat(per_box_frames, ignore_index=True)
        combo_path = os.path.join(out_dir, "cds_in_box_all.tsv")
        combined.to_csv(combo_path, sep="\t", index=False)
        print(f"Combined TSV: {combo_path}")

    # Plot: t-SNE with boxes + CDS-in-box highlighted
    # Background: all regions in light grey (we didn't classify the full dataset)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(ca["tsne_1"], ca["tsne_2"], c="#cccccc", s=3, alpha=0.35,
               label=f"all regions ({len(ca)})", rasterized=True)
    # Overlay classified candidates by annotation
    ann_colors = {"CDS": "#d62728", "UTR/exon": "#ff7f0e",
                  "Intron": "#1f77b4", "Intergenic": "#aec7e8"}
    for ann, col in ann_colors.items():
        m = candidates["annotation"] == ann
        if m.sum():
            ax.scatter(candidates.loc[m, "tsne_1"], candidates.loc[m, "tsne_2"],
                       c=col, s=6, alpha=0.75,
                       label=f"{ann} in-box ({m.sum()})", rasterized=True)

    # Highlight and draw boxes
    for i, (box, frame) in enumerate(zip(args.box, per_box_frames), start=1):
        xmin, ymin, xmax, ymax = box
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                         fill=False, edgecolor="red", linewidth=2.5)
        ax.add_patch(rect)
        ax.text(xmin, ymax + 2, f"Box {i}: n={len(frame)}",
                fontsize=10, color="red", fontweight="bold")
        if len(frame):
            ax.scatter(frame["tsne_1"], frame["tsne_2"],
                       facecolors="none", edgecolors="black",
                       s=18, linewidths=0.6, alpha=0.9)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"{args.chrom}: {label} regions inside user-defined boxes "
                 f"(N_total={len(ca)})")
    ax.legend(loc="upper right", markerscale=4, framealpha=0.9)
    plot_path = os.path.join(out_dir, f"tsne_boxes_{args.chrom}.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
