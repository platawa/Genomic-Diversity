#!/usr/bin/env python3
"""
build_ground_truth.py

Parse GTF annotations to extract ground truth boundaries for evaluating
drop detection methods. Outputs BED-format file with position + feature type.

Ground truth features:
  - CDS start/stop boundaries (coding sequence transitions)
  - Exon boundaries (splice sites)
  - Gene start/stop
  - tRNA, rRNA gene boundaries

Usage:
    python tools/build_ground_truth.py \
        --gtf /path/to/genomic.gtf \
        --chrom NC_000022.11 \
        --output ground_truth_chr22.bed \
        [--tolerance 100] \
        [--start 0 --end 50818468]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class GroundTruthFeature:
    """A single ground truth boundary position."""
    chrom: str
    position: int          # 0-based genomic position
    feature_type: str      # e.g., "CDS_start", "exon_boundary", "gene_start"
    strand: str            # +, -, .
    gene_name: str         # parent gene name if available
    gene_type: str         # e.g., "protein_coding", "tRNA", "rRNA"
    expected_direction: str  # "drop" (entropy should decrease) or "rise" (should increase)


def parse_gtf_attribute(attr_str: str, key: str) -> Optional[str]:
    """Extract a value from GTF attribute string."""
    for field in attr_str.split(';'):
        field = field.strip()
        if field.startswith(key):
            parts = field.split('"')
            if len(parts) >= 2:
                return parts[1]
            parts = field.split(' ', 1)
            if len(parts) >= 2:
                return parts[1].strip('"')
    return None


def parse_gtf(gtf_path: str, chrom: str,
              start: int = 0, end: Optional[int] = None) -> List[GroundTruthFeature]:
    """Parse GTF and extract boundary features.

    For each CDS/exon, we extract both boundaries:
    - On + strand: start = "drop" (entering coding), end = "rise" (leaving coding)
    - On - strand: reversed (start = "rise", end = "drop")
    """
    features = []
    seen_positions = set()  # avoid duplicates at same position

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
            feat_start = int(fields[3]) - 1  # GTF is 1-based -> 0-based
            feat_end = int(fields[4])        # GTF end is inclusive -> exclusive
            strand = fields[6]
            attr_str = fields[8]

            if end is not None and feat_start > end:
                continue
            if feat_end < start:
                continue

            gene_name = (parse_gtf_attribute(attr_str, 'gene_name') or
                         parse_gtf_attribute(attr_str, 'gene_id') or
                         'unknown')
            gene_type = (parse_gtf_attribute(attr_str, 'gene_biotype') or
                         parse_gtf_attribute(attr_str, 'gene_type') or
                         'unknown')

            if feat_type == 'CDS':
                # CDS start: entropy should drop (entering highly constrained region)
                # CDS end: entropy should rise (leaving constrained region)
                if strand == '+':
                    start_dir, end_dir = 'drop', 'rise'
                else:
                    start_dir, end_dir = 'rise', 'drop'

                key_s = (feat_start, 'CDS_start')
                if key_s not in seen_positions:
                    features.append(GroundTruthFeature(
                        chrom=chrom, position=feat_start,
                        feature_type='CDS_start', strand=strand,
                        gene_name=gene_name, gene_type=gene_type,
                        expected_direction=start_dir
                    ))
                    seen_positions.add(key_s)

                key_e = (feat_end, 'CDS_end')
                if key_e not in seen_positions:
                    features.append(GroundTruthFeature(
                        chrom=chrom, position=feat_end,
                        feature_type='CDS_end', strand=strand,
                        gene_name=gene_name, gene_type=gene_type,
                        expected_direction=end_dir
                    ))
                    seen_positions.add(key_e)

            elif feat_type == 'exon':
                # Exon boundaries = splice sites
                key_s = (feat_start, 'exon_boundary')
                if key_s not in seen_positions:
                    features.append(GroundTruthFeature(
                        chrom=chrom, position=feat_start,
                        feature_type='exon_boundary', strand=strand,
                        gene_name=gene_name, gene_type=gene_type,
                        expected_direction='drop' if strand == '+' else 'rise'
                    ))
                    seen_positions.add(key_s)

                key_e = (feat_end, 'exon_boundary')
                if key_e not in seen_positions:
                    features.append(GroundTruthFeature(
                        chrom=chrom, position=feat_end,
                        feature_type='exon_boundary', strand=strand,
                        gene_name=gene_name, gene_type=gene_type,
                        expected_direction='rise' if strand == '+' else 'drop'
                    ))
                    seen_positions.add(key_e)

            elif feat_type == 'gene':
                for pos, label, direction in [
                    (feat_start, 'gene_start', 'drop' if strand == '+' else 'rise'),
                    (feat_end, 'gene_end', 'rise' if strand == '+' else 'drop'),
                ]:
                    key = (pos, label)
                    if key not in seen_positions:
                        features.append(GroundTruthFeature(
                            chrom=chrom, position=pos,
                            feature_type=label, strand=strand,
                            gene_name=gene_name, gene_type=gene_type,
                            expected_direction=direction
                        ))
                        seen_positions.add(key)

    features.sort(key=lambda f: f.position)
    return features


def write_bed(features: List[GroundTruthFeature], output_path: str,
              tolerance: int = 100, offset: int = 0):
    """Write features in BED format.

    BED columns: chrom, start, end, name, score, strand
    The start/end span +-tolerance around the feature position.
    Positions are adjusted by offset (subtract genomic_start of the scored region).
    """
    with open(output_path, 'w') as f:
        f.write('#chrom\tstart\tend\tname\tscore\tstrand\tfeature_type\t'
                'gene_name\tgene_type\texpected_direction\n')
        for feat in features:
            pos = feat.position - offset
            bed_start = max(0, pos - tolerance)
            bed_end = pos + tolerance
            name = f"{feat.feature_type}|{feat.gene_name}"
            f.write(f"{feat.chrom}\t{bed_start}\t{bed_end}\t{name}\t0\t"
                    f"{feat.strand}\t{feat.feature_type}\t{feat.gene_name}\t"
                    f"{feat.gene_type}\t{feat.expected_direction}\n")


def write_positions_tsv(features: List[GroundTruthFeature], output_path: str,
                        offset: int = 0):
    """Write simple TSV with array-relative positions for detection evaluation."""
    with open(output_path, 'w') as f:
        f.write('position\tfeature_type\texpected_direction\tgene_name\tgene_type\tstrand\n')
        for feat in features:
            pos = feat.position - offset
            f.write(f"{pos}\t{feat.feature_type}\t{feat.expected_direction}\t"
                    f"{feat.gene_name}\t{feat.gene_type}\t{feat.strand}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Extract ground truth boundaries from GTF for detection evaluation"
    )
    ap.add_argument("--gtf", required=True, help="Path to GTF annotation file")
    ap.add_argument("--chrom", required=True,
                    help="Chromosome/sequence name as it appears in the GTF")
    ap.add_argument("--output", required=True,
                    help="Output BED file path")
    ap.add_argument("--output_tsv", default=None,
                    help="Also write simple TSV with array-relative positions")
    ap.add_argument("--tolerance", type=int, default=100,
                    help="Tolerance window in bp around each boundary (default: 100)")
    ap.add_argument("--start", type=int, default=0,
                    help="Genomic start of scored region (for offset, default: 0)")
    ap.add_argument("--end", type=int, default=None,
                    help="Genomic end of scored region")
    ap.add_argument("--feature_types", nargs='+',
                    default=['CDS_start', 'CDS_end', 'exon_boundary', 'gene_start', 'gene_end'],
                    help="Feature types to include (default: all)")
    args = ap.parse_args()

    print(f"Parsing GTF: {args.gtf}")
    print(f"Chromosome: {args.chrom}")
    print(f"Tolerance: +-{args.tolerance} bp")

    features = parse_gtf(args.gtf, args.chrom, args.start, args.end)
    print(f"Total features extracted: {len(features)}")

    # Filter by feature type
    features = [f for f in features if f.feature_type in args.feature_types]
    print(f"After filtering by type: {len(features)}")

    # Count by type
    type_counts = defaultdict(int)
    direction_counts = defaultdict(int)
    gene_type_counts = defaultdict(int)
    for f in features:
        type_counts[f.feature_type] += 1
        direction_counts[f.expected_direction] += 1
        gene_type_counts[f.gene_type] += 1

    print("\nFeature type breakdown:")
    for ft, count in sorted(type_counts.items()):
        print(f"  {ft}: {count}")

    print(f"\nExpected directions: drop={direction_counts['drop']}, rise={direction_counts['rise']}")

    print(f"\nGene biotype breakdown (top 10):")
    for gt, count in sorted(gene_type_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {gt}: {count}")

    # Write outputs
    write_bed(features, args.output, tolerance=args.tolerance, offset=args.start)
    print(f"\nBED file: {args.output}")

    if args.output_tsv:
        write_positions_tsv(features, args.output_tsv, offset=args.start)
        print(f"TSV file: {args.output_tsv}")

    # Write summary JSON
    summary_path = args.output.replace('.bed', '_summary.json')
    summary = {
        "gtf": args.gtf,
        "chrom": args.chrom,
        "tolerance_bp": args.tolerance,
        "genomic_start": args.start,
        "genomic_end": args.end,
        "total_features": len(features),
        "type_counts": dict(type_counts),
        "direction_counts": dict(direction_counts),
        "gene_type_counts": dict(gene_type_counts),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
