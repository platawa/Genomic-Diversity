#!/usr/bin/env python3
"""
cluster_top_genes.py

Rank genes by the number of drop regions overlapping them within each Leiden
cluster. Also computes per-cluster annotation summary (fraction CDS / UTR /
intron / intergenic) for "annotation by inference" downstream use.

Outputs:
  - cluster_top_genes.tsv        — cluster_id × gene × n_regions (sorted)
  - cluster_annotation_summary.tsv — fraction of each annotation class per cluster
  - cluster_majority_annotation.tsv — cluster_id → majority class + fraction
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_utils import write_completed
from plot_tsne_by_annotation import CHROM_MAP

logger = logging.getLogger(__name__)


def parse_gtf_features(gtf_path, chrom_filter=None):
    """Parse exon, CDS, gene entries. Return dict of lists per chrom."""
    out = defaultdict(list)
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            chrom = parts[0]
            if chrom_filter is not None and chrom not in chrom_filter:
                continue
            feat_type = parts[2]
            if feat_type not in {"exon", "CDS", "five_prime_utr", "three_prime_utr", "gene"}:
                continue
            try:
                s, e = int(parts[3]), int(parts[4])
            except ValueError:
                continue
            gname = None
            for field in parts[8].split(";"):
                f_ = field.strip()
                if f_.startswith("gene_name") or f_.startswith("gene_id"):
                    gname = f_.split('"')[1] if '"' in f_ else f_.split()[-1]
                    break
            out[chrom].append((feat_type, s, e, gname))
    return out


def classify(start, end, features):
    """Priority CDS > UTR > exon (non-CDS non-UTR) > gene (intron-ish) > intergenic."""
    types = set()
    genes = set()
    for (ft, s, e, gname) in features:
        if e < start or s > end:
            continue
        types.add(ft)
        if gname:
            genes.add(gname)
    if "CDS" in types:
        klass = "CDS"
    elif "five_prime_utr" in types or "three_prime_utr" in types:
        klass = "UTR"
    elif "exon" in types:
        klass = "exon_other"
    elif "gene" in types:
        klass = "intron"
    else:
        klass = "intergenic"
    return klass, sorted(genes)


def load_regions(scope_dir):
    ca = pd.read_csv(os.path.join(scope_dir, "data", "cluster_assignments.tsv"), sep="\t", comment="#")
    if "cluster_id" not in ca.columns:
        for alt in ["cluster", "leiden", "leiden_cluster"]:
            if alt in ca.columns:
                ca = ca.rename(columns={alt: "cluster_id"})
                break
    return ca


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--scope_dir", required=True)
    p.add_argument("--gtf", required=True)
    p.add_argument("--chrom", default=None,
                   help="If per-chrom scope (regions lack chrom column)")
    p.add_argument("--top_k_genes_per_cluster", type=int, default=20)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    regions = load_regions(args.scope_dir)
    if "chrom" not in regions.columns and args.chrom:
        regions["chrom"] = args.chrom

    chroms_in_scope = set(regions["chrom"].unique())
    # Translate UCSC-style names to NCBI accessions for GTF matching
    gtf_ids = {CHROM_MAP.get(c, c) for c in chroms_in_scope}
    chrom_to_gtf = {c: CHROM_MAP.get(c, c) for c in chroms_in_scope}
    features_by_gtf = parse_gtf_features(args.gtf, chrom_filter=gtf_ids)
    # Re-key so later code can look up by UCSC chrom name
    features = {c: features_by_gtf.get(chrom_to_gtf[c], []) for c in chroms_in_scope}
    logger.info(f"GTF features loaded for {len(features)} chroms (via accession mapping)")

    out_dir = args.output_dir or os.path.join(args.scope_dir, "cluster_top_genes")
    os.makedirs(out_dir, exist_ok=True)

    t0 = __import__("time").time()

    # Per-region classification
    annotations = []
    gene_rows = []
    for i, r in regions.iterrows():
        chrom = r["chrom"]
        feats = features.get(chrom, [])
        klass, genes = classify(int(r["genomic_start"]), int(r["genomic_end"]), feats)
        annotations.append(klass)
        for g in genes:
            gene_rows.append((int(r["cluster_id"]), chrom, g, i))

    regions = regions.copy()
    regions["annotation"] = annotations

    # Annotation fraction per cluster
    annot_table = regions.groupby(["cluster_id", "annotation"]).size().unstack(fill_value=0)
    annot_frac = annot_table.div(annot_table.sum(axis=1), axis=0)
    annot_table.to_csv(os.path.join(out_dir, "cluster_annotation_counts.tsv"), sep="\t")
    annot_frac.to_csv(os.path.join(out_dir, "cluster_annotation_fractions.tsv"), sep="\t")

    majority = annot_frac.idxmax(axis=1)
    majority_frac = annot_frac.max(axis=1)
    pd.DataFrame({
        "cluster_id": majority.index,
        "majority_annotation": majority.values,
        "majority_fraction": majority_frac.values,
    }).to_csv(os.path.join(out_dir, "cluster_majority_annotation.tsv"), sep="\t", index=False)

    # Per-cluster top genes
    gene_df = pd.DataFrame(gene_rows, columns=["cluster_id", "chrom", "gene", "region_idx"])
    top_genes = (
        gene_df.groupby(["cluster_id", "gene"]).size()
        .reset_index(name="n_regions")
        .sort_values(["cluster_id", "n_regions"], ascending=[True, False])
    )
    top_genes = top_genes.groupby("cluster_id").head(args.top_k_genes_per_cluster)
    top_genes.to_csv(os.path.join(out_dir, "cluster_top_genes.tsv"), sep="\t", index=False)

    write_completed(out_dir, "cluster_top_genes.py",
                    __import__("time").time() - t0)
    logger.info(f"Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
