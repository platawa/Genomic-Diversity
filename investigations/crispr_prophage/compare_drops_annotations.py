#!/usr/bin/env python3
"""
compare_drops_annotations.py — Annotate entropy drops with gene/CRISPR/prophage info

Classifies each entropy drop as genic vs intergenic and flags known
CRISPR/prophage loci in E. coli K-12.

Usage:
    python investigations/crispr_prophage/compare_drops_annotations.py \
        --boundaries_tsv results/ecoli_K12/scoring/.../data/drop_boundaries.tsv \
        --gtf /path/to/genomic.gtf \
        --chrom NC_000913.3 \
        --chrom_name ecoli_K12
"""

import os
import sys
import csv
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from results_utils import build_run_dir, write_completed, write_source
from sae_utils import parse_chromosome_drops_tsv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tools'))
from sae_annotation_overlay import parse_gtf_genes, annotate_regions

# Known prophage and CRISPR loci in E. coli K-12 MG1655 (NC_000913.3)
# Coordinates are 0-based, half-open
ECOLI_K12_SPECIAL_LOCI = {
    # Prophages
    "CP4-6":  (262246, 282260),
    "DLP12":  (556698, 582543),
    "e14":    (1195432, 1211059),
    "Rac":    (1408685, 1433217),
    "Qin":    (1629855, 1651856),
    "CP4-44": (2064327, 2078613),
    "CPS-53": (2161314, 2175866),
    "CPZ-55": (2556942, 2563568),
    "CP4-57": (2747020, 2773709),
    "KpLE2":  (3449036, 3467424),
    # CRISPR arrays
    "CRISPR-I":  (2875441, 2876516),
    "CRISPR-II": (2877618, 2878569),
}


def check_special_locus_overlap(drop_start: int, drop_end: int) -> str:
    """Return name of overlapping special locus, or empty string."""
    hits = []
    for name, (locus_start, locus_end) in ECOLI_K12_SPECIAL_LOCI.items():
        if drop_start < locus_end and drop_end > locus_start:
            hits.append(name)
    return ','.join(hits) if hits else ''


def main():
    parser = argparse.ArgumentParser(description="Annotate entropy drops with gene/CRISPR/prophage info")
    parser.add_argument('--boundaries_tsv', required=True, help='Path to drop_boundaries.tsv')
    parser.add_argument('--gtf', required=True, help='Path to GTF annotation file')
    parser.add_argument('--chrom', required=True, help='Chromosome/accession ID in GTF')
    parser.add_argument('--chrom_name', default='ecoli_K12', help='Human-readable name for output dir')
    parser.add_argument('--output_dir', default='results', help='Base output directory')
    parser.add_argument('--min_confidence', type=float, default=0.0, help='Min confidence filter for drops')
    args = parser.parse_args()

    t0 = time.time()

    # Load drops
    drops = parse_chromosome_drops_tsv(args.boundaries_tsv, min_confidence=args.min_confidence)
    print(f"Loaded {len(drops)} drops from {args.boundaries_tsv}")

    # Load GTF features
    gtf_features = parse_gtf_genes(args.gtf, args.chrom)
    print(f"Loaded {len(gtf_features)} GTF features for {args.chrom}")

    # Annotate with gene overlaps
    annotated = annotate_regions(drops, gtf_features)

    # Add special locus column
    for region in annotated:
        start = int(region.get('drop_start', region.get('genomic_start', 0)))
        end = int(region.get('drop_end', region.get('genomic_end', 0)))
        region['special_locus'] = check_special_locus_overlap(start, end)

    # Build output directory
    run_dir = build_run_dir(args.output_dir, args.chrom_name, 'drop_annotations',
                            f"conf{args.min_confidence}")
    data_dir = os.path.join(run_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Write annotated TSV
    tsv_path = os.path.join(data_dir, 'annotated_drops.tsv')
    fieldnames = list(annotated[0].keys()) if annotated else []
    with open(tsv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in annotated:
            writer.writerow(row)
    print(f"Wrote {len(annotated)} annotated drops to {tsv_path}")

    # Compute summary statistics
    n_genic = sum(1 for r in annotated if r.get('is_annotated', False))
    n_intergenic = len(annotated) - n_genic
    n_special = sum(1 for r in annotated if r.get('special_locus', ''))

    biotype_counts = {}
    for r in annotated:
        for bt in r.get('gene_biotype', 'intergenic').split(','):
            biotype_counts[bt] = biotype_counts.get(bt, 0) + 1

    special_hits = {}
    for r in annotated:
        locus = r.get('special_locus', '')
        if locus:
            for name in locus.split(','):
                special_hits[name] = special_hits.get(name, 0) + 1

    summary = {
        'total_drops': len(annotated),
        'n_genic': n_genic,
        'n_intergenic': n_intergenic,
        'n_special_locus': n_special,
        'biotype_counts': biotype_counts,
        'special_locus_hits': special_hits,
        'min_confidence': args.min_confidence,
        'chrom': args.chrom,
    }

    summary_path = os.path.join(data_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {n_genic} genic, {n_intergenic} intergenic, {n_special} in special loci")

    wall_time = time.time() - t0
    write_source(run_dir, boundaries_tsv=args.boundaries_tsv, gtf=args.gtf)
    write_completed(run_dir, os.path.basename(__file__), wall_time)
    print(f"Done in {wall_time:.1f}s. Output: {run_dir}")


if __name__ == '__main__':
    main()
