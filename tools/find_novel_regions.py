#!/usr/bin/env python3
"""
find_novel_regions.py

Identify entropy drop regions that do NOT overlap any GTF annotation.
These are candidate novel functional elements for lab validation.

Usage:
    python tools/find_novel_regions.py \
        --boundaries results/chr22/scoring/.../data/drop_boundaries.tsv \
        --gtf /path/to/genomic.gtf \
        --chrom NC_000022.11 \
        --output novel_regions_chr22.bed \
        [--tolerance 500]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple, Set
from collections import defaultdict


def load_boundaries(tsv_path: str) -> List[dict]:
    """Load drop boundary regions from TSV."""
    regions = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            regions.append({
                'chrom': row.get('chrom', ''),
                'start': int(row.get('drop_start', row.get('genomic_start', 0))),
                'end': int(row.get('drop_end', row.get('genomic_end', 0))),
                'method': row.get('method', ''),
                'confidence': float(row.get('start_confidence', 0)),
                'mean_entropy': float(row.get('mean_entropy', 0)),
                'region_length': int(row.get('region_length', 0)),
            })
    return regions


def load_gtf_intervals(gtf_path: str, chrom: str) -> List[Tuple[int, int, str]]:
    """Load all annotated intervals from GTF for a chromosome.

    Returns list of (start, end, feature_type) tuples.
    """
    intervals = []
    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            if fields[0] != chrom:
                continue
            feat_type = fields[2]
            feat_start = int(fields[3]) - 1  # 0-based
            feat_end = int(fields[4])
            intervals.append((feat_start, feat_end, feat_type))
    return intervals


def find_overlapping_annotations(region_start: int, region_end: int,
                                  annotations: List[Tuple[int, int, str]],
                                  tolerance: int = 500) -> List[str]:
    """Find all annotation types overlapping a region (with tolerance)."""
    overlapping = []
    for ann_start, ann_end, feat_type in annotations:
        # Check overlap with tolerance
        if (region_start - tolerance <= ann_end and
            region_end + tolerance >= ann_start):
            overlapping.append(feat_type)
    return overlapping


def main():
    ap = argparse.ArgumentParser(
        description="Find drop regions not overlapping any GTF annotation"
    )
    ap.add_argument("--boundaries", required=True,
                    help="Drop boundaries TSV from scoring pipeline")
    ap.add_argument("--gtf", required=True,
                    help="GTF annotation file")
    ap.add_argument("--chrom", required=True,
                    help="Chromosome name as in GTF")
    ap.add_argument("--output", required=True,
                    help="Output BED file with novel regions")
    ap.add_argument("--tolerance", type=int, default=500,
                    help="bp tolerance for overlap (default: 500)")
    ap.add_argument("--min_confidence", type=float, default=0,
                    help="Minimum confidence score to include (default: 0)")
    ap.add_argument("--feature_types", nargs='+',
                    default=['gene', 'exon', 'CDS', 'mRNA', 'tRNA', 'rRNA'],
                    help="GTF feature types to consider as 'annotated'")
    args = ap.parse_args()

    # Load data
    print(f"Loading boundaries: {args.boundaries}")
    regions = load_boundaries(args.boundaries)
    print(f"  {len(regions)} drop regions")

    if args.min_confidence > 0:
        regions = [r for r in regions if r['confidence'] >= args.min_confidence]
        print(f"  {len(regions)} after confidence filter >= {args.min_confidence}")

    print(f"Loading GTF: {args.gtf}")
    all_annotations = load_gtf_intervals(args.gtf, args.chrom)
    print(f"  {len(all_annotations)} total features for {args.chrom}")

    # Filter to relevant feature types
    annotations = [(s, e, t) for s, e, t in all_annotations
                    if t in args.feature_types]
    print(f"  {len(annotations)} features of types: {args.feature_types}")

    # Classify regions
    novel = []
    annotated = []
    annotation_type_counts = defaultdict(int)

    for region in regions:
        overlaps = find_overlapping_annotations(
            region['start'], region['end'], annotations, args.tolerance
        )
        if overlaps:
            annotated.append(region)
            for ot in set(overlaps):
                annotation_type_counts[ot] += 1
        else:
            novel.append(region)

    # Sort novel regions by confidence (descending)
    novel.sort(key=lambda r: r['confidence'], reverse=True)

    print(f"\nResults:")
    print(f"  Annotated (overlap GTF): {len(annotated)}")
    print(f"  Novel (no overlap):      {len(novel)}")
    print(f"  Novelty rate:            {len(novel)/len(regions)*100:.1f}%" if regions else "  N/A")

    print(f"\nAnnotation overlap breakdown:")
    for feat_type, count in sorted(annotation_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {feat_type}: {count} regions")

    # Write BED output
    with open(args.output, 'w') as f:
        f.write('#chrom\tstart\tend\tname\tscore\tstrand\tmethod\t'
                'confidence\tmean_entropy\tregion_length\n')
        for i, region in enumerate(novel):
            name = f"novel_{i+1}"
            f.write(f"{region['chrom']}\t{region['start']}\t{region['end']}\t"
                    f"{name}\t{region['confidence']:.2f}\t.\t{region['method']}\t"
                    f"{region['confidence']:.4f}\t{region['mean_entropy']:.4f}\t"
                    f"{region['region_length']}\n")

    print(f"\nNovel regions BED: {args.output}")

    # Write summary
    summary = {
        "boundaries_file": args.boundaries,
        "gtf_file": args.gtf,
        "chrom": args.chrom,
        "tolerance_bp": args.tolerance,
        "total_regions": len(regions),
        "annotated_regions": len(annotated),
        "novel_regions": len(novel),
        "novelty_rate": len(novel) / len(regions) if regions else 0,
        "annotation_overlap_counts": dict(annotation_type_counts),
        "top_novel_regions": [
            {
                "start": r['start'], "end": r['end'],
                "confidence": r['confidence'],
                "mean_entropy": r['mean_entropy'],
                "region_length": r['region_length'],
            }
            for r in novel[:20]
        ],
    }
    summary_path = args.output.replace('.bed', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
