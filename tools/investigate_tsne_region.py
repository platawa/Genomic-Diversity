#!/usr/bin/env python3
"""
investigate_tsne_region.py

Extract every SAE region whose t-SNE coordinates fall inside a user-specified
box (or list of region indices) and report their genomic annotations.

Outputs (under <sae_run>/<variant>/investigations/<tag>/):
  - regions_in_box.tsv      per-region coords, annotation class, nearest gene
  - summary.tsv             class composition in-box vs rest-of-chromosome
  - intergenic_details.tsv  for intergenic rows: ncRNA/pseudogene overlaps,
                            nearest upstream + downstream gene names/distances
  - composition.png         bar chart: in-box vs background
  - tsne_box_overlay.png    original t-SNE with box drawn on top

Usage:
  python tools/investigate_tsne_region.py \
      --chrom chr1 \
      --variant latent_analysis \
      --box 35,90,-25,30 \
      --gtf /path/to/genomic.gtf \
      --tag red_island

  # Or with an explicit index list (e.g., lasso export from the HTML):
  python tools/investigate_tsne_region.py \
      --chrom chr1 --variant latent_analysis \
      --indices-file /tmp/lasso_idx.txt --gtf ... --tag lasso1
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import find_latest_completed
from tools.plot_tsne_by_annotation import CHROM_MAP, classify_region, load_gtf_features
from tools.latent_plot_utils import compute_distance_to_nearest_gene


def _build_interval_index(interval_list):
    """Sort (start,end) intervals and compute running max-end for O(log N) containment."""
    if not interval_list:
        return None
    arr = np.array(sorted(interval_list), dtype=np.int64)
    starts = arr[:, 0]
    ends = arr[:, 1]
    max_end = np.maximum.accumulate(ends)
    return starts, max_end


def _contains_any(index, mids):
    """Vectorized: for each midpoint, is it inside any interval?"""
    if index is None:
        return np.zeros(len(mids), dtype=bool)
    starts, max_end = index
    idx = np.searchsorted(starts, mids, side="right") - 1
    valid = idx >= 0
    out = np.zeros(len(mids), dtype=bool)
    out[valid] = mids[valid] <= max_end[idx[valid]]
    return out


def vectorized_classify(region_starts, region_ends, intervals):
    """Vectorized counterpart of plot_tsne_by_annotation.classify_region.

    Classifies each region by midpoint with priority CDS > UTR/exon > Intron
    > Intergenic, using searchsorted + running-max-end for O(N log M) speed.
    """
    mids = (np.asarray(region_starts) + np.asarray(region_ends)) // 2
    cds = _build_interval_index(intervals.get("CDS", []))
    exon = _build_interval_index(intervals.get("exon", []))
    gene = _build_interval_index(intervals.get("gene", []))

    in_cds = _contains_any(cds, mids)
    in_exon = _contains_any(exon, mids) & ~in_cds
    in_gene = _contains_any(gene, mids) & ~in_cds & ~in_exon
    labels = np.full(len(mids), "Intergenic", dtype=object)
    labels[in_gene] = "Intron"
    labels[in_exon] = "UTR/exon"
    labels[in_cds] = "CDS"
    return labels


def vectorized_nearest_gene_distance(region_starts, region_ends, genes_df):
    """Vectorized upstream/downstream nearest-gene distance via searchsorted.

    Returns (up_dist, down_dist, up_name, down_name) as numpy arrays.
    Regions whose midpoint overlaps a gene get distance = 0.
    """
    n = len(region_starts)
    if genes_df.empty:
        nan = np.full(n, np.nan)
        empty = np.array([""] * n, dtype=object)
        return nan, nan, empty, empty

    mids = (np.asarray(region_starts) + np.asarray(region_ends)) // 2
    gstarts = genes_df.start.values
    gends = genes_df.end.values
    gnames = genes_df.gene_name.values

    # Overlap: midpoint inside any gene — reuse _contains_any
    gene_index = _build_interval_index(list(zip(gstarts, gends)))
    overlap = _contains_any(gene_index, mids)

    up_dist = np.full(n, np.nan)
    down_dist = np.full(n, np.nan)
    up_name = np.array([""] * n, dtype=object)
    down_name = np.array([""] * n, dtype=object)

    # Downstream: first gene with start > mid
    down_idx = np.searchsorted(gstarts, mids, side="right")
    valid_down = down_idx < len(gstarts)
    down_dist[valid_down] = gstarts[down_idx[valid_down]] - mids[valid_down]
    down_name[valid_down] = gnames[down_idx[valid_down]]

    # Upstream: last gene with end < mid. Use searchsorted on ends (sorted by gene start,
    # not end — so scan from idx backwards a small bounded distance for correctness).
    # For speed: sort genes by end separately.
    order_end = np.argsort(gends)
    gends_s = gends[order_end]
    gnames_s = gnames[order_end]
    up_idx = np.searchsorted(gends_s, mids, side="left") - 1
    valid_up = up_idx >= 0
    up_dist[valid_up] = mids[valid_up] - gends_s[up_idx[valid_up]]
    up_name[valid_up] = gnames_s[up_idx[valid_up]]

    # Overlapping midpoints: distance = 0, gene_name = the overlapping gene
    if overlap.any():
        over_idx = np.searchsorted(gstarts, mids, side="right") - 1
        up_dist[overlap] = 0
        down_dist[overlap] = 0
        up_name[overlap] = gnames[over_idx[overlap]]
        down_name[overlap] = gnames[over_idx[overlap]]

    return up_dist, down_dist, up_name, down_name


NCRNA_TYPES = {
    "lnc_RNA", "lncRNA", "miRNA", "snoRNA", "snRNA", "tRNA", "rRNA",
    "ncRNA", "pseudogene", "transcribed_pseudogene",
    "processed_pseudogene", "unprocessed_pseudogene",
    "transcribed_unprocessed_pseudogene",
    "transcribed_processed_pseudogene",
}


def load_gtf_extended(gtf_path, chrom_id):
    """Like load_gtf_features but also returns ncRNA/pseudogene intervals with
    gene names, and nearest-gene lookup (gene_name, start, end, strand)."""
    intervals = defaultdict(list)
    ncrna_rows = []
    gene_rows = []
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[0] != chrom_id:
                continue
            ftype = parts[2]
            start, end = int(parts[3]), int(parts[4])
            strand = parts[6]
            attrs = parts[8]

            def _attr(key):
                token = f'{key} "'
                i = attrs.find(token)
                if i < 0:
                    return ""
                i += len(token)
                j = attrs.find('"', i)
                return attrs[i:j] if j > 0 else ""

            if ftype in ("CDS", "exon", "gene"):
                intervals[ftype].append((start, end))

            if ftype == "gene":
                gene_rows.append({
                    "start": start, "end": end, "strand": strand,
                    "gene_name": _attr("gene") or _attr("gene_name") or _attr("gene_id"),
                    "gene_biotype": _attr("gene_biotype"),
                })

            if ftype in NCRNA_TYPES or "pseudogene" in ftype.lower():
                ncrna_rows.append({
                    "start": start, "end": end, "strand": strand,
                    "feature_type": ftype,
                    "gene_name": _attr("gene") or _attr("gene_name") or _attr("gene_id"),
                })

    genes_df = pd.DataFrame(gene_rows).sort_values("start").reset_index(drop=True) \
        if gene_rows else pd.DataFrame(columns=["start","end","strand","gene_name","gene_biotype"])
    nc_df = pd.DataFrame(ncrna_rows).sort_values("start").reset_index(drop=True) \
        if ncrna_rows else pd.DataFrame(columns=["start","end","strand","feature_type","gene_name"])

    print(f"Loaded for {chrom_id}: CDS={len(intervals['CDS'])}, "
          f"exon={len(intervals['exon'])}, gene={len(intervals['gene'])}, "
          f"genes_df={len(genes_df)}, ncRNA/pseudogene={len(nc_df)}")
    return intervals, genes_df, nc_df


def nearest_gene(mid, genes_df, direction="downstream"):
    """Return (gene_name, distance) for nearest gene on either side."""
    if genes_df.empty:
        return ("", np.nan)
    starts = genes_df.start.values
    ends = genes_df.end.values
    if direction == "downstream":
        idx = np.searchsorted(starts, mid, side="right")
        if idx < len(starts):
            return (genes_df.iloc[idx].gene_name, int(starts[idx] - mid))
        return ("", np.nan)
    # upstream: last gene whose end < mid
    idx = np.searchsorted(ends, mid, side="left") - 1
    while idx >= 0 and ends[idx] >= mid:
        idx -= 1
    if idx >= 0:
        return (genes_df.iloc[idx].gene_name, int(mid - ends[idx]))
    return ("", np.nan)


def find_sae_run(results_dir, chrom, variant):
    """Locate a COMPLETED sae run that contains the requested variant."""
    sae_dir = os.path.join(results_dir, chrom, "sae")
    # Prefer the top-level {variant} dir if present directly under sae/
    top = os.path.join(sae_dir, variant, "data", "cluster_assignments.tsv")
    if os.path.isfile(top):
        return os.path.join(sae_dir, variant), os.path.dirname(os.path.dirname(top))
    # Else fall back to latest COMPLETED sae run with this variant
    run = find_latest_completed(results_dir, chrom, "sae")
    if run is None:
        return None, None
    cand = os.path.join(run, variant, "data", "cluster_assignments.tsv")
    if os.path.isfile(cand):
        return os.path.join(run, variant), run
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chrom", required=True)
    p.add_argument("--variant", default="latent_analysis",
                   choices=["latent_analysis", "latent_analysis_normalized",
                            "latent_analysis_prenorm", "latent_analysis_postnorm"])
    p.add_argument("--results-dir", default="results")
    p.add_argument("--gtf", required=True)
    p.add_argument("--embedding", default="tsne", choices=["tsne", "umap"],
                   help="Which embedding's coordinates the box applies to.")
    p.add_argument("--box", default=None,
                   help="Comma-separated x1,x2,y1,y2 in embedding coordinates.")
    p.add_argument("--indices-file", default=None,
                   help="Text file with region_idx values (one per line) "
                        "for polygon/lasso-style selection.")
    p.add_argument("--tag", required=True, help="Subdir name under investigations/")
    args = p.parse_args()

    if not (args.box or args.indices_file):
        p.error("Provide --box OR --indices-file")

    variant_dir, sae_run = find_sae_run(args.results_dir, args.chrom, args.variant)
    if variant_dir is None:
        print(f"ERROR: could not locate {args.variant} for {args.chrom}")
        sys.exit(1)

    ca_path = os.path.join(variant_dir, "data", "cluster_assignments.tsv")
    regions = pd.read_csv(ca_path, sep="\t", comment="#")
    col1, col2 = f"{args.embedding}_1", f"{args.embedding}_2"
    if col1 not in regions.columns:
        print(f"ERROR: {ca_path} has no {col1} column "
              f"(available: {[c for c in regions.columns if c.startswith(('tsne_','umap_'))]})")
        sys.exit(1)
    print(f"Loaded {len(regions)} regions from {ca_path}")

    # Selection mask
    if args.box:
        x1, x2, y1, y2 = [float(v) for v in args.box.split(",")]
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        mask = ((regions[col1] >= x1) & (regions[col1] <= x2) &
                (regions[col2] >= y1) & (regions[col2] <= y2))
        selection_desc = f"box {col1}∈[{x1},{x2}], {col2}∈[{y1},{y2}]"
    else:
        with open(args.indices_file) as f:
            wanted = {int(line.strip()) for line in f if line.strip()}
        mask = regions.region_idx.isin(wanted)
        selection_desc = f"{len(wanted)} region_idx values from {args.indices_file}"

    n_in = int(mask.sum())
    print(f"Selection: {selection_desc} → {n_in} / {len(regions)} regions")
    if n_in == 0:
        print("No regions in selection; exiting.")
        sys.exit(0)

    # GTF classification
    chrom_id = CHROM_MAP.get(args.chrom, args.chrom)
    intervals, genes_df, nc_df = load_gtf_extended(args.gtf, chrom_id)

    regions["annotation"] = vectorized_classify(
        regions.genomic_start.values, regions.genomic_end.values, intervals)

    up_dist, down_dist, up_name, down_name = vectorized_nearest_gene_distance(
        regions.genomic_start.values, regions.genomic_end.values, genes_df)
    regions["upstream_dist"] = up_dist
    regions["downstream_dist"] = down_dist
    regions["upstream_gene"] = up_name
    regions["downstream_gene"] = down_name

    in_box = regions[mask].copy()
    out_dir = os.path.join(variant_dir, "investigations", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    in_box.to_csv(os.path.join(out_dir, "regions_in_box.tsv"),
                  sep="\t", index=False)
    print(f"Saved: {out_dir}/regions_in_box.tsv")

    # ── Summary: class composition in-box vs rest ──
    classes = ["CDS", "UTR/exon", "Intron", "Intergenic"]
    in_counts = in_box.annotation.value_counts().reindex(classes, fill_value=0)
    bg_counts = regions.loc[~mask, "annotation"].value_counts().reindex(classes, fill_value=0)

    summary = pd.DataFrame({
        "class": classes,
        "in_box": in_counts.values,
        "rest_of_chrom": bg_counts.values,
        "in_box_pct": 100 * in_counts.values / max(n_in, 1),
        "rest_pct": 100 * bg_counts.values / max(int((~mask).sum()), 1),
    })
    summary["enrichment_log2"] = np.log2(
        (summary.in_box_pct + 1e-9) / (summary.rest_pct + 1e-9))
    summary.to_csv(os.path.join(out_dir, "summary.tsv"), sep="\t", index=False)
    print(f"Saved: {out_dir}/summary.tsv")
    print(summary.to_string(index=False))

    # ── Intergenic drill-down ──
    intergenic = in_box[in_box.annotation == "Intergenic"].copy()
    if not intergenic.empty and not nc_df.empty:
        nc_hits = []
        nc_starts = nc_df.start.values
        nc_ends = nc_df.end.values
        for _, r in intergenic.iterrows():
            hit_idx = np.where((nc_starts <= r.genomic_end) &
                               (nc_ends >= r.genomic_start))[0]
            if len(hit_idx) == 0:
                nc_hits.append("")
                continue
            hits = nc_df.iloc[hit_idx]
            summ = ";".join(f"{h.feature_type}:{h.gene_name}"
                            for _, h in hits.iterrows())
            nc_hits.append(summ)
        intergenic["ncRNA_pseudogene_overlaps"] = nc_hits
    else:
        intergenic["ncRNA_pseudogene_overlaps"] = ""

    intergenic.to_csv(os.path.join(out_dir, "intergenic_details.tsv"),
                      sep="\t", index=False)
    print(f"Saved: {out_dir}/intergenic_details.tsv "
          f"({len(intergenic)} intergenic regions, "
          f"{(intergenic.ncRNA_pseudogene_overlaps != '').sum()} with ncRNA/pseudogene hit)")

    # ── Composition bar plot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(classes))
    w = 0.4
    ax.bar(x - w/2, summary.in_box_pct, width=w, label=f"In box (n={n_in})",
           color="#E74C3C")
    ax.bar(x + w/2, summary.rest_pct, width=w, label=f"Rest (n={int((~mask).sum())})",
           color="#BDC3C7")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("% of regions")
    ax.set_title(f"Annotation composition — {args.chrom} / {args.variant} / {args.tag}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "composition.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir}/composition.png")

    # ── Embedding overlay with box drawn on ──
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(regions.loc[~mask, col1], regions.loc[~mask, col2],
               c="#CCCCCC", s=4, alpha=0.4, edgecolors="none", rasterized=True,
               label=f"outside ({(~mask).sum()})")
    ax.scatter(in_box[col1], in_box[col2],
               c="#E74C3C", s=8, alpha=0.8, edgecolors="none", rasterized=True,
               label=f"in box ({n_in})")
    if args.box:
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                               edgecolor="black", fill=False, linewidth=1.5))
    ax.set_xlabel(f"{args.embedding.upper()} 1")
    ax.set_ylabel(f"{args.embedding.upper()} 2")
    ax.set_title(f"{args.chrom} / {args.variant} / {args.embedding} — selection: {args.tag}")
    ax.legend()
    plt.tight_layout()
    overlay_path = os.path.join(out_dir, f"{args.embedding}_box_overlay.png")
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {overlay_path}")


if __name__ == "__main__":
    main()
