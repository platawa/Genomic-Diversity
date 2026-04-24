#!/usr/bin/env python3
"""
identify_tsne_clusters.py

Given a chr12 t-SNE plot with visually identified clusters, extract the
genomic regions in each cluster and cross-reference with GTF annotations.

Works by loading cluster_assignments.tsv (which has tsne_1, tsne_2 coords)
and letting you define bounding ellipses/boxes around your visual clusters.

Usage:
    # Interactive mode — prints cluster_assignments.tsv columns and
    # Leiden cluster distribution so you can map visual clusters to IDs:
    python tools/identify_tsne_clusters.py \
        --latent_dir results/chr12/sae/latent_analysis_postnorm \
        --mode explore

    # Define visual clusters by t-SNE coordinate bounding boxes and
    # cross-reference with GTF:
    python tools/identify_tsne_clusters.py \
        --latent_dir results/chr12/sae/latent_analysis_postnorm \
        --gtf /path/to/genomic.gtf \
        --chrom chr12 \
        --mode annotate

    # Export regions per Leiden cluster to TSV:
    python tools/identify_tsne_clusters.py \
        --latent_dir results/chr12/sae/latent_analysis_postnorm \
        --gtf /path/to/genomic.gtf \
        --chrom chr12 \
        --mode export
"""

import argparse
import os
import sys
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patheffects as pe

# ---------------------------------------------------------------------------
# Chromosome name -> RefSeq accession (for GTF matching)
# ---------------------------------------------------------------------------
CHROM_MAP = {
    "chr1": "NC_000001.11", "chr2": "NC_000002.12", "chr3": "NC_000003.12",
    "chr4": "NC_000004.12", "chr5": "NC_000005.10", "chr6": "NC_000006.12",
    "chr7": "NC_000007.14", "chr8": "NC_000008.11", "chr9": "NC_000009.12",
    "chr10": "NC_000010.11", "chr11": "NC_000011.10", "chr12": "NC_000012.12",
    "chr13": "NC_000013.11", "chr14": "NC_000014.9", "chr15": "NC_000015.10",
    "chr16": "NC_000016.10", "chr17": "NC_000017.11", "chr18": "NC_000018.10",
    "chr19": "NC_000019.10", "chr20": "NC_000020.11", "chr21": "NC_000021.9",
    "chr22": "NC_000022.11", "chrX": "NC_000023.11",
}


def load_gtf_features(gtf_path, chrom_id):
    """Parse GTF and return interval lists for CDS, exon, gene on one chrom."""
    intervals = {"CDS": [], "exon": [], "gene": []}
    gene_names = {}  # (start, end) -> gene_name
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            if parts[0] != chrom_id:
                continue
            feat = parts[2]
            if feat in intervals:
                s, e = int(parts[3]) - 1, int(parts[4])  # 0-based half-open
                intervals[feat].append((s, e))
                if feat == "gene":
                    # Extract gene name from attributes
                    attrs = parts[8]
                    name = None
                    for attr in attrs.split(";"):
                        attr = attr.strip()
                        if attr.startswith("gene_id"):
                            name = attr.split('"')[1] if '"' in attr else attr.split()[1]
                        if attr.startswith("gene ") or 'gene_name' in attr:
                            name = attr.split('"')[1] if '"' in attr else attr.split()[1]
                    if name:
                        gene_names[(s, e)] = name
    # Sort intervals for faster lookup
    for k in intervals:
        intervals[k].sort()
    return intervals, gene_names


def load_repeatmasker_bed(bed_path, chrom_id):
    """Load RepeatMasker BED/OUT file and return interval list with repeat classes.

    Supports:
    - UCSC RepeatMasker .bed (chrom, start, end, name, ...)
    - UCSC RepeatMasker .out parsed to BED (tab-separated: chrom start end name class family)
    - Simple BED with at least 4 columns

    Returns list of (start, end, repeat_name, repeat_class).
    """
    repeats = []
    with open(bed_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("browser") or line.startswith("track"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            chrom = parts[0]
            # Map chr12 -> NC_000012.12 or accept as-is
            if chrom != chrom_id and chrom not in CHROM_MAP.values():
                # Try matching by chr name
                mapped = CHROM_MAP.get(chrom)
                if mapped != chrom_id:
                    continue
            elif chrom != chrom_id:
                continue
            s, e = int(parts[1]), int(parts[2])
            name = parts[3] if len(parts) > 3 else "Unknown"
            rep_class = parts[4] if len(parts) > 4 else "Unknown"
            repeats.append((s, e, name, rep_class))
    repeats.sort()
    print(f"  RepeatMasker: loaded {len(repeats)} repeats for {chrom_id}")
    return repeats


def load_encode_ccre_bed(bed_path, chrom_id):
    """Load ENCODE cCRE BED file.

    Expected format (ENCODE V3 cCREs):
    chrom  start  end  accession  name  classification
    Where classification is one of: PLS, pELS, dELS, CTCF-only, DNase-H3K4me3

    Classification meanings:
    - PLS: Promoter-like signature
    - pELS: Proximal enhancer-like signature
    - dELS: Distal enhancer-like signature
    - CTCF-only: CTCF-bound (insulator)
    - DNase-H3K4me3: DNase + H3K4me3

    Returns list of (start, end, accession, classification).
    """
    ccres = []
    with open(bed_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("browser") or line.startswith("track"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            chrom = parts[0]
            if chrom != chrom_id and chrom not in CHROM_MAP.values():
                mapped = CHROM_MAP.get(chrom)
                if mapped != chrom_id:
                    continue
            elif chrom != chrom_id:
                continue
            s, e = int(parts[1]), int(parts[2])
            accession = parts[3] if len(parts) > 3 else ""
            # Classification can be in column 5 or 6 depending on format
            classification = ""
            for col in parts[4:]:
                if col in ("PLS", "pELS", "dELS", "CTCF-only", "DNase-H3K4me3",
                           "Promoter-like", "Enhancer-like", "CTCF-bound"):
                    classification = col
                    break
            if not classification and len(parts) > 5:
                classification = parts[5]
            ccres.append((s, e, accession, classification))
    ccres.sort()
    print(f"  ENCODE cCREs: loaded {len(ccres)} elements for {chrom_id}")
    return ccres


def classify_repeatmasker(start, end, repeats):
    """Check if region overlaps a repeat element. Returns (repeat_name, repeat_class) or None."""
    mid = (start + end) // 2
    for s, e, name, rep_class in repeats:
        if s > mid + 1000:  # sorted, so we can break early
            break
        if s <= mid <= e:
            return name, rep_class
    return None


def classify_ccre(start, end, ccres):
    """Check if region overlaps an ENCODE cCRE. Returns classification or None."""
    mid = (start + end) // 2
    for s, e, accession, classification in ccres:
        if s > mid + 1000:
            break
        if s <= mid <= e:
            return classification
    return None


def classify_region(start, end, intervals, repeats=None, ccres=None):
    """Classify a region by midpoint overlap.

    Priority: CDS > UTR/exon > Intron > Intergenic.
    For intergenic regions, further classify by RepeatMasker and ENCODE cCREs.
    """
    mid = (start + end) // 2
    for s, e in intervals["CDS"]:
        if s <= mid <= e:
            return "CDS"
    for s, e in intervals["exon"]:
        if s <= mid <= e:
            return "UTR/exon"
    for s, e in intervals["gene"]:
        if s <= mid <= e:
            return "Intron"

    # Intergenic — try to sub-classify
    if ccres:
        ccre_hit = classify_ccre(start, end, ccres)
        if ccre_hit:
            # Map ENCODE labels to readable names
            ccre_map = {
                "PLS": "Promoter (cCRE)",
                "pELS": "Prox. Enhancer (cCRE)",
                "dELS": "Dist. Enhancer (cCRE)",
                "CTCF-only": "CTCF/Insulator (cCRE)",
                "DNase-H3K4me3": "DNase-H3K4me3 (cCRE)",
                "Promoter-like": "Promoter (cCRE)",
                "Enhancer-like": "Enhancer (cCRE)",
                "CTCF-bound": "CTCF/Insulator (cCRE)",
            }
            return ccre_map.get(ccre_hit, f"cCRE:{ccre_hit}")

    if repeats:
        rep_hit = classify_repeatmasker(start, end, repeats)
        if rep_hit:
            name, rep_class = rep_hit
            # Simplify repeat class names
            if "LINE" in rep_class:
                return "Repeat: LINE"
            elif "SINE" in rep_class or "Alu" in rep_class:
                return "Repeat: SINE/Alu"
            elif "LTR" in rep_class:
                return "Repeat: LTR"
            elif "DNA" in rep_class:
                return "Repeat: DNA transposon"
            elif "Simple" in rep_class or "Low" in rep_class:
                return "Repeat: Simple/Low complexity"
            elif "Satellite" in rep_class:
                return "Repeat: Satellite"
            else:
                return f"Repeat: {rep_class}"

    return "Intergenic"


def find_overlapping_genes(start, end, intervals, gene_names):
    """Find gene names overlapping a region."""
    hits = []
    for (gs, ge), name in gene_names.items():
        if start < ge and end > gs:
            hits.append(name)
    return hits


def mode_explore(ca, embedding):
    """Print summary of what's in the data to help identify clusters."""
    print("=" * 70)
    print(f"CLUSTER ASSIGNMENTS: {len(ca)} regions")
    print(f"Columns: {list(ca.columns)}")
    print()

    # Check if tsne coords are in the TSV or need numpy
    has_tsne_cols = "tsne_1" in ca.columns and "tsne_2" in ca.columns
    if has_tsne_cols:
        print("t-SNE coordinates found in cluster_assignments.tsv")
    elif embedding is not None:
        print(f"t-SNE coordinates from embedding_tsne.npy: shape {embedding.shape}")
        ca = ca.copy()
        ca["tsne_1"] = embedding[:, 0]
        ca["tsne_2"] = embedding[:, 1]
    else:
        print("WARNING: No t-SNE coordinates found!")

    # Leiden cluster distribution
    if "cluster" in ca.columns:
        print(f"\n{'=' * 70}")
        print("LEIDEN CLUSTER DISTRIBUTION")
        print(f"{'=' * 70}")
        cluster_counts = ca["cluster"].value_counts().sort_index()
        for cid, count in cluster_counts.items():
            subset = ca[ca["cluster"] == cid]
            if "tsne_1" in subset.columns:
                x_range = f"  t-SNE x: [{subset['tsne_1'].min():.1f}, {subset['tsne_1'].max():.1f}]"
                y_range = f"  y: [{subset['tsne_2'].min():.1f}, {subset['tsne_2'].max():.1f}]"
            else:
                x_range = y_range = ""
            print(f"  Cluster {cid:3d}: {count:5d} regions{x_range}{y_range}")

    # If annotation column exists
    if "annotation" in ca.columns:
        print(f"\n{'=' * 70}")
        print("ANNOTATION DISTRIBUTION")
        print(f"{'=' * 70}")
        print(ca["annotation"].value_counts().to_string())

    # Genomic coordinate ranges
    if "genomic_start" in ca.columns:
        print(f"\nGenomic range: {ca['genomic_start'].min():,} - {ca['genomic_end'].max():,}")

    # Show first few rows
    print(f"\n{'=' * 70}")
    print("FIRST 5 ROWS")
    print(f"{'=' * 70}")
    print(ca.head().to_string())


def mode_annotate(ca, embedding, intervals, gene_names, output_dir, repeats=None, ccres=None):
    """Annotate each region with GTF classification and export per-cluster summaries."""
    # Ensure t-SNE coords are available
    if "tsne_1" not in ca.columns and embedding is not None:
        ca = ca.copy()
        ca["tsne_1"] = embedding[:, 0]
        ca["tsne_2"] = embedding[:, 1]

    # Classify each region
    print("Classifying regions by GTF annotation...")
    annotations = []
    nearest_genes = []
    for _, row in ca.iterrows():
        start = int(row["genomic_start"])
        end = int(row["genomic_end"])
        ann = classify_region(start, end, intervals, repeats, ccres)
        annotations.append(ann)
        genes = find_overlapping_genes(start, end, intervals, gene_names)
        nearest_genes.append("; ".join(genes[:3]) if genes else "")

    ca = ca.copy()
    ca["annotation"] = annotations
    ca["overlapping_genes"] = nearest_genes

    # Per-cluster summary
    if "cluster" in ca.columns:
        print(f"\n{'=' * 70}")
        print("PER-CLUSTER ANNOTATION BREAKDOWN")
        print(f"{'=' * 70}")

        for cid in sorted(ca["cluster"].unique()):
            subset = ca[ca["cluster"] == cid]
            ann_counts = subset["annotation"].value_counts()
            total = len(subset)

            print(f"\n--- Cluster {cid} ({total} regions) ---")
            if "tsne_1" in subset.columns:
                print(f"  t-SNE center: ({subset['tsne_1'].mean():.1f}, {subset['tsne_2'].mean():.1f})")
                print(f"  t-SNE range:  x=[{subset['tsne_1'].min():.1f}, {subset['tsne_1'].max():.1f}]  "
                      f"y=[{subset['tsne_2'].min():.1f}, {subset['tsne_2'].max():.1f}]")

            if "genomic_start" in subset.columns:
                print(f"  Genomic span: {subset['genomic_start'].min():,} - {subset['genomic_end'].max():,}")
                median_len = subset["region_length"].median() if "region_length" in subset.columns else "N/A"
                print(f"  Median region length: {median_len}")

            print("  Annotations:")
            for ann, cnt in ann_counts.items():
                pct = 100 * cnt / total
                print(f"    {ann:15s}: {cnt:4d} ({pct:5.1f}%)")

            # Show top genes
            gene_counts = Counter()
            for genes_str in subset["overlapping_genes"]:
                if genes_str:
                    for g in genes_str.split("; "):
                        gene_counts[g] += 1
            if gene_counts:
                print(f"  Top genes:")
                for gene, cnt in gene_counts.most_common(5):
                    print(f"    {gene}: {cnt} regions")

    # Export annotated TSV
    out_path = os.path.join(output_dir, "cluster_annotations.tsv")
    ca.to_csv(out_path, sep="\t", index=False)
    print(f"\nFull annotated data saved to: {out_path}")

    return ca


def mode_export(ca, embedding, intervals, gene_names, output_dir, repeats=None, ccres=None):
    """Export per-cluster TSV files with full region details."""
    if "tsne_1" not in ca.columns and embedding is not None:
        ca = ca.copy()
        ca["tsne_1"] = embedding[:, 0]
        ca["tsne_2"] = embedding[:, 1]

    # Classify
    annotations = []
    nearest_genes = []
    for _, row in ca.iterrows():
        start = int(row["genomic_start"])
        end = int(row["genomic_end"])
        annotations.append(classify_region(start, end, intervals, repeats, ccres))
        genes = find_overlapping_genes(start, end, intervals, gene_names)
        nearest_genes.append("; ".join(genes[:3]) if genes else "")

    ca = ca.copy()
    ca["annotation"] = annotations
    ca["overlapping_genes"] = nearest_genes

    if "cluster" not in ca.columns:
        print("No 'cluster' column found — exporting all regions as one file")
        out_path = os.path.join(output_dir, "all_regions_annotated.tsv")
        ca.to_csv(out_path, sep="\t", index=False)
        print(f"Saved: {out_path}")
        return

    cluster_dir = os.path.join(output_dir, "per_cluster")
    os.makedirs(cluster_dir, exist_ok=True)

    for cid in sorted(ca["cluster"].unique()):
        subset = ca[ca["cluster"] == cid]
        out_path = os.path.join(cluster_dir, f"cluster_{cid}_regions.tsv")
        subset.to_csv(out_path, sep="\t", index=False)
        ann_summary = subset["annotation"].value_counts().to_dict()
        print(f"Cluster {cid}: {len(subset)} regions — {ann_summary} -> {out_path}")

    print(f"\nAll per-cluster files in: {cluster_dir}")


def mode_plot(ca, embedding, intervals, gene_names, output_dir, chrom, repeats=None, ccres=None):
    """Generate annotated t-SNE plot with cluster labels, top genes, and annotation breakdown."""
    # Ensure t-SNE coords
    if "tsne_1" not in ca.columns and embedding is not None:
        ca = ca.copy()
        ca["tsne_1"] = embedding[:, 0]
        ca["tsne_2"] = embedding[:, 1]

    # Classify each region
    print("Classifying regions by GTF annotation...")
    annotations = []
    gene_lists = []
    for _, row in ca.iterrows():
        start, end = int(row["genomic_start"]), int(row["genomic_end"])
        annotations.append(classify_region(start, end, intervals, repeats, ccres))
        genes = find_overlapping_genes(start, end, intervals, gene_names)
        gene_lists.append(genes)

    ca = ca.copy()
    ca["annotation"] = annotations
    ca["overlapping_genes"] = ["; ".join(g[:3]) for g in gene_lists]

    # --- Color scheme: annotation colors ---
    ann_colors = {
        "CDS": "#d62728",
        "UTR/exon": "#ff7f0e",
        "Intron": "#1f77b4",
        "Intergenic": "#aec7e8",
        # RepeatMasker sub-categories
        "Repeat: LINE": "#8B4513",
        "Repeat: SINE/Alu": "#DAA520",
        "Repeat: LTR": "#CD853F",
        "Repeat: DNA transposon": "#A0522D",
        "Repeat: Simple/Low complexity": "#D2B48C",
        "Repeat: Satellite": "#BC8F8F",
        # ENCODE cCRE sub-categories
        "Promoter (cCRE)": "#2ca02c",
        "Prox. Enhancer (cCRE)": "#98df8a",
        "Dist. Enhancer (cCRE)": "#66c2a5",
        "CTCF/Insulator (cCRE)": "#9467bd",
        "DNase-H3K4me3 (cCRE)": "#c5b0d5",
        "Enhancer (cCRE)": "#98df8a",
    }
    # Base categories always plotted; extra ones only if they appear
    ann_order_base = ["CDS", "UTR/exon", "Intron", "Intergenic"]
    # Determine which categories actually appear
    all_anns = ca["annotation"].unique()
    ann_order = [a for a in ann_order_base if a in all_anns]
    ann_order += sorted([a for a in all_anns if a not in ann_order_base])
    # Assign fallback colors for any unknown categories
    fallback_colors = plt.cm.Set3(np.linspace(0, 1, 12))
    for i, ann in enumerate(ann_order):
        if ann not in ann_colors:
            ann_colors[ann] = matplotlib.colors.to_hex(fallback_colors[i % 12])

    # Dominant-annotation color for ellipses: map each base annotation to an ellipse color
    # (slightly darker/more saturated than the dot colors)
    ellipse_colors = {
        "CDS": "#b71c1c",
        "UTR/exon": "#e65100",
        "Intron": "#0d47a1",
        "Intergenic": "#78909c",
    }
    # For sub-categories, map back to parent
    def get_ellipse_color(top_ann):
        if top_ann in ellipse_colors:
            return ellipse_colors[top_ann]
        if "Repeat" in top_ann:
            return "#6D4C00"
        if "cCRE" in top_ann or "Enhancer" in top_ann or "Promoter" in top_ann:
            return "#1b5e20"
        return "#333333"

    # --- Figure 1: 2-panel overview (left: labeled by cluster, right: clean annotation) ---
    fig, (ax, ax_clean) = plt.subplots(1, 2, figsize=(34, 16))

    # Plot all points colored by annotation on BOTH axes
    for ann in ann_order:
        mask = ca["annotation"] == ann
        count = mask.sum()
        if count == 0:
            continue
        ax.scatter(ca.loc[mask, "tsne_1"], ca.loc[mask, "tsne_2"],
                   c=ann_colors[ann], s=3, alpha=0.4, label=f"{ann} ({count})", rasterized=True)
        ax_clean.scatter(ca.loc[mask, "tsne_1"], ca.loc[mask, "tsne_2"],
                         c=ann_colors[ann], s=3, alpha=0.4, label=f"{ann} ({count})", rasterized=True)

    # Per-cluster: ellipses colored by dominant annotation
    # Only draw ellipses for spatially compact clusters (not spread across the whole plot)
    if "cluster" in ca.columns:
        cluster_ids = sorted(ca["cluster"].unique())

        # Compute the overall t-SNE range to set a compactness threshold
        tsne_x_range = ca["tsne_1"].max() - ca["tsne_1"].min()
        tsne_y_range = ca["tsne_2"].max() - ca["tsne_2"].min()
        # Skip ellipse if std > 25% of the full plot range (too diffuse)
        max_std_x = tsne_x_range * 0.20
        max_std_y = tsne_y_range * 0.20

        for cid in cluster_ids:
            subset = ca[ca["cluster"] == cid]
            if len(subset) < 5:
                continue

            cx = subset["tsne_1"].mean()
            cy = subset["tsne_2"].mean()
            std_x = subset["tsne_1"].std()
            std_y = subset["tsne_2"].std()

            # Determine dominant annotation
            ann_counts = subset["annotation"].value_counts()
            total = len(subset)
            top_ann = ann_counts.index[0]
            top_ann_pct = 100 * ann_counts.iloc[0] / total

            # Color ellipse by dominant annotation
            ec = get_ellipse_color(top_ann)

            # Top 2 genes
            gene_counter = Counter()
            for row_idx in subset.index:
                for g in gene_lists[row_idx]:
                    gene_counter[g] += 1
            top_genes = [g for g, _ in gene_counter.most_common(2)]
            gene_str = ", ".join(top_genes) if top_genes else ""

            # Only draw ellipse if cluster is spatially compact
            is_compact = std_x < max_std_x and std_y < max_std_y
            if is_compact:
                # Use 1.5x std for tighter ellipses
                sx = std_x * 1.5
                sy = std_y * 1.5
                lw = 2.0 if top_ann_pct >= 50 else 1.0

                ellipse = Ellipse((cx, cy), width=sx * 2, height=sy * 2,
                                  fill=False, edgecolor=ec,
                                  linewidth=lw, linestyle="--", alpha=0.6)
                ax.add_patch(ellipse)

            # Label: always show for clusters >= 100 regions
            if total >= 100:
                label = f"C{cid} (n={total})\n{top_ann} {top_ann_pct:.0f}%"
                if gene_str:
                    label += f"\n{gene_str}"
                offset_y = (std_y * 1.5 + 1.5) if is_compact else 2.0
                ax.annotate(label,
                            xy=(cx, cy), xytext=(cx, cy + offset_y),
                            fontsize=6, fontweight="bold", ha="center", va="bottom",
                            color=ec,
                            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                            arrowprops=dict(arrowstyle="-", color=ec,
                                            alpha=0.3, lw=0.5))
            else:
                ax.text(cx, cy, f"C{cid}", fontsize=5, ha="center", va="center",
                        color=ec, alpha=0.7,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(f"t-SNE of SAE Region Fingerprints — {chrom} (N={len(ca)})\n"
                 f"Colored by Annotation, Labeled by Cluster", fontsize=13)
    ax.legend(loc="upper right", fontsize=10, markerscale=4,
              framealpha=0.9, edgecolor="black")

    ax_clean.set_xlabel("t-SNE 1", fontsize=12)
    ax_clean.set_ylabel("t-SNE 2", fontsize=12)
    ax_clean.set_title(f"t-SNE of SAE Region Fingerprints — {chrom} (N={len(ca)})\n"
                       f"Colored by Genomic Annotation", fontsize=13)
    ax_clean.legend(loc="upper right", fontsize=10, markerscale=4,
                    framealpha=0.9, edgecolor="black")
    ax_clean.set_xlim(ax.get_xlim())
    ax_clean.set_ylim(ax.get_ylim())

    out_path = os.path.join(output_dir, f"tsne_annotated_clusters_{chrom}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved annotated cluster plot: {out_path}")

    # --- Figure 2: Per-cluster detail panels (top 20 largest) ---
    if "cluster" in ca.columns and len(cluster_ids) > 0:
        # Sort by size, show top 20
        cluster_sizes = ca["cluster"].value_counts().sort_values(ascending=False)
        top_clusters = cluster_sizes.index[:20].tolist()

        ncols = 4
        nrows = (len(top_clusters) + ncols - 1) // ncols
        fig2, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
        axes = np.array(axes).flatten()

        for idx, cid in enumerate(top_clusters):
            ax2 = axes[idx]
            subset = ca[ca["cluster"] == cid]

            # Plot all points in grey
            ax2.scatter(ca["tsne_1"], ca["tsne_2"], c="#e0e0e0", s=1, alpha=0.3, rasterized=True)

            # Highlight this cluster colored by annotation
            for ann in ann_order:
                mask = (ca["cluster"] == cid) & (ca["annotation"] == ann)
                if mask.sum() > 0:
                    ax2.scatter(ca.loc[mask, "tsne_1"], ca.loc[mask, "tsne_2"],
                                c=ann_colors[ann], s=8, alpha=0.7, label=f"{ann} ({mask.sum()})")

            # Black dashed ellipse on this panel too
            cx = subset["tsne_1"].mean()
            cy = subset["tsne_2"].mean()
            sx = subset["tsne_1"].std() * 2.0
            sy = subset["tsne_2"].std() * 2.0
            ellipse = Ellipse((cx, cy), width=sx * 2, height=sy * 2,
                              fill=False, edgecolor="black",
                              linewidth=1.5, linestyle="--", alpha=0.6)
            ax2.add_patch(ellipse)

            # Annotation breakdown
            ann_counts = subset["annotation"].value_counts()
            total = len(subset)
            ann_str = ", ".join(f"{a}: {100*c/total:.0f}%" for a, c in ann_counts.items())

            # Top genes
            gene_counter = Counter()
            for row_idx in subset.index:
                for g in gene_lists[row_idx]:
                    gene_counter[g] += 1
            top5 = [f"{g} ({c})" for g, c in gene_counter.most_common(5)]

            ax2.set_title(f"Cluster {cid} (n={total})\n{ann_str}", fontsize=8)
            ax2.legend(fontsize=6, markerscale=2, loc="upper right")

            # Gene list text box
            if top5:
                gene_text = "Top genes:\n" + "\n".join(top5)
                ax2.text(0.02, 0.02, gene_text, transform=ax2.transAxes,
                         fontsize=5.5, va="bottom", ha="left",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

        for idx in range(len(top_clusters), len(axes)):
            axes[idx].set_visible(False)

        fig2.suptitle(f"{chrom} — Top 20 Clusters by Size", fontsize=14, y=1.01)
        fig2.tight_layout()
        out_path2 = os.path.join(output_dir, f"tsne_cluster_details_{chrom}.png")
        fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved per-cluster detail panels: {out_path2}")

    # --- Export annotated TSV ---
    out_tsv = os.path.join(output_dir, "cluster_annotations.tsv")
    ca.to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved annotated TSV: {out_tsv}")

    # --- Print summary table ---
    if "cluster" in ca.columns:
        print(f"\n{'=' * 80}")
        print(f"CLUSTER SUMMARY — {chrom}")
        print(f"{'=' * 80}")
        print(f"{'Cluster':>8} {'N':>6} {'Dominant':>25} {'%':>5}  Top Genes")
        print("-" * 80)
        for cid in cluster_ids:
            subset = ca[ca["cluster"] == cid]
            total = len(subset)
            ann_counts = subset["annotation"].value_counts()
            top_ann = ann_counts.index[0]
            top_pct = 100 * ann_counts.iloc[0] / total

            gene_counter = Counter()
            for row_idx in subset.index:
                for g in gene_lists[row_idx]:
                    gene_counter[g] += 1
            top3 = ", ".join(g for g, _ in gene_counter.most_common(3))

            print(f"{cid:>8} {total:>6} {top_ann:>25} {top_pct:>4.0f}%  {top3}")


def main():
    parser = argparse.ArgumentParser(
        description="Identify t-SNE clusters and cross-reference with GTF annotations")
    parser.add_argument("--latent_dir", required=True,
                        help="Path to latent_analysis directory (e.g. results/chr12/sae/latent_analysis_postnorm)")
    parser.add_argument("--gtf", default=None,
                        help="Path to GTF file (required for annotate/export modes)")
    parser.add_argument("--chrom", default=None,
                        help="Chromosome name (e.g. chr12)")
    parser.add_argument("--mode", choices=["explore", "annotate", "export", "plot"],
                        default="explore",
                        help="Mode: explore (inspect data), annotate (classify & summarize), "
                             "export (per-cluster TSV files), plot (annotated t-SNE figure)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: latent_dir/cluster_analysis/)")
    parser.add_argument("--repeatmasker", default=None,
                        help="Path to RepeatMasker BED file (optional, sub-classifies intergenic regions)")
    parser.add_argument("--ccre", default=None,
                        help="Path to ENCODE cCRE BED file (optional, identifies enhancers/promoters/insulators)")
    args = parser.parse_args()

    data_dir = os.path.join(args.latent_dir, "data")

    # Load cluster_assignments.tsv
    ca_path = os.path.join(data_dir, "cluster_assignments.tsv")
    if not os.path.exists(ca_path):
        print(f"ERROR: {ca_path} not found")
        sys.exit(1)
    ca = pd.read_csv(ca_path, sep="\t", comment="#")
    print(f"Loaded {len(ca)} regions from {ca_path}")

    # Load embedding if available
    emb_path = os.path.join(data_dir, "embedding_tsne.npy")
    embedding = np.load(emb_path) if os.path.exists(emb_path) else None

    output_dir = args.output_dir or os.path.join(args.latent_dir, "cluster_analysis")
    os.makedirs(output_dir, exist_ok=True)

    if args.mode == "explore":
        mode_explore(ca, embedding)
        return

    # For annotate/export/plot, need GTF
    if not args.gtf or not args.chrom:
        print("ERROR: --gtf and --chrom required for annotate/export/plot modes")
        sys.exit(1)

    chrom_id = CHROM_MAP.get(args.chrom, args.chrom)
    print(f"Loading GTF features for {args.chrom} ({chrom_id})...")
    intervals, gene_names = load_gtf_features(args.gtf, chrom_id)
    print(f"  GTF loaded: {len(intervals['gene'])} genes, {len(intervals['CDS'])} CDS, {len(intervals['exon'])} exons")

    # Load optional annotation sources
    repeats = None
    if args.repeatmasker:
        print(f"Loading RepeatMasker from {args.repeatmasker}...")
        repeats = load_repeatmasker_bed(args.repeatmasker, chrom_id)

    ccres = None
    if args.ccre:
        print(f"Loading ENCODE cCREs from {args.ccre}...")
        ccres = load_encode_ccre_bed(args.ccre, chrom_id)

    if args.mode == "annotate":
        mode_annotate(ca, embedding, intervals, gene_names, output_dir, repeats, ccres)
    elif args.mode == "export":
        mode_export(ca, embedding, intervals, gene_names, output_dir, repeats, ccres)
    elif args.mode == "plot":
        mode_plot(ca, embedding, intervals, gene_names, output_dir, args.chrom, repeats, ccres)


if __name__ == "__main__":
    main()
