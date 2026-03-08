#!/usr/bin/env python3
"""
sae_annotation_overlay.py

Cross-reference SAE cluster assignments with GTF annotations.
Determines which gene types cluster together and which SAE features
are most associated with specific annotation categories.

Usage:
    python tools/sae_annotation_overlay.py \
        --clusters results/chr22/sae/.../latent_analysis/data/cluster_assignments.tsv \
        --sae_results results/chr22/sae/.../data/sae_results.tsv \
        --gtf /path/to/genomic.gtf \
        --chrom NC_000022.11 \
        [--output_dir results/chr22/sae/.../annotation_overlay/]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter


def load_cluster_assignments(tsv_path: str) -> List[dict]:
    """Load cluster assignments TSV from analyze_sae_regions.py."""
    rows = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    return rows


def load_sae_results(tsv_path: str) -> List[dict]:
    """Load SAE results TSV from run_sae_on_chromosome_drops.py."""
    rows = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    return rows


def parse_gtf_genes(gtf_path: str, chrom: str) -> List[dict]:
    """Parse GTF to extract gene/CDS/exon intervals with biotype info."""
    features = []
    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[0] != chrom:
                continue
            feat_type = fields[2]
            if feat_type not in ('gene', 'CDS', 'exon', 'mRNA', 'tRNA', 'rRNA'):
                continue

            attr_str = fields[8]
            gene_name = None
            gene_type = None
            for attr in attr_str.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_name') or attr.startswith('gene_id'):
                    parts = attr.split('"')
                    if len(parts) >= 2:
                        gene_name = parts[1]
                if attr.startswith('gene_biotype') or attr.startswith('gene_type'):
                    parts = attr.split('"')
                    if len(parts) >= 2:
                        gene_type = parts[1]

            features.append({
                'type': feat_type,
                'start': int(fields[3]) - 1,
                'end': int(fields[4]),
                'strand': fields[6],
                'gene_name': gene_name or 'unknown',
                'gene_type': gene_type or 'unknown',
            })
    return features


def annotate_regions(regions: List[dict], gtf_features: List[dict],
                     tolerance: int = 500) -> List[dict]:
    """Annotate each SAE region with overlapping GTF features."""
    annotated = []
    for region in regions:
        reg_start = int(region.get('genomic_start', region.get('drop_start', 0)))
        reg_end = int(region.get('genomic_end', region.get('drop_end', 0)))

        overlapping_genes = set()
        overlapping_types = set()
        overlapping_biotypes = set()

        for feat in gtf_features:
            if feat['start'] - tolerance <= reg_end and feat['end'] + tolerance >= reg_start:
                overlapping_genes.add(feat['gene_name'])
                overlapping_types.add(feat['type'])
                overlapping_biotypes.add(feat['gene_type'])

        ann = dict(region)
        ann['overlapping_genes'] = ','.join(sorted(overlapping_genes)) if overlapping_genes else 'intergenic'
        ann['overlapping_feat_types'] = ','.join(sorted(overlapping_types)) if overlapping_types else 'none'
        ann['gene_biotype'] = ','.join(sorted(overlapping_biotypes)) if overlapping_biotypes else 'intergenic'
        ann['is_annotated'] = len(overlapping_genes) > 0 and 'unknown' not in overlapping_genes
        annotated.append(ann)

    return annotated


def compute_cluster_enrichment(annotated_regions: List[dict]) -> dict:
    """Compute gene biotype enrichment per cluster."""
    cluster_biotypes = defaultdict(lambda: Counter())
    cluster_sizes = Counter()

    for region in annotated_regions:
        cluster = region.get('cluster', region.get('leiden_cluster', 'unknown'))
        cluster_sizes[cluster] += 1
        for biotype in region['gene_biotype'].split(','):
            cluster_biotypes[cluster][biotype] += 1

    # Compute enrichment relative to overall frequency
    total_count = sum(cluster_sizes.values())
    overall_biotypes = Counter()
    for region in annotated_regions:
        for biotype in region['gene_biotype'].split(','):
            overall_biotypes[biotype] += 1

    enrichment = {}
    for cluster in sorted(cluster_sizes.keys()):
        cluster_size = cluster_sizes[cluster]
        enrichment[str(cluster)] = {
            'n_regions': cluster_size,
            'biotype_counts': dict(cluster_biotypes[cluster]),
            'dominant_biotype': cluster_biotypes[cluster].most_common(1)[0][0] if cluster_biotypes[cluster] else 'unknown',
            'biotype_enrichment': {},
        }
        for biotype, count in cluster_biotypes[cluster].items():
            expected = (overall_biotypes[biotype] / total_count) * cluster_size
            fold = count / expected if expected > 0 else float('inf')
            enrichment[str(cluster)]['biotype_enrichment'][biotype] = round(fold, 2)

    return enrichment


def main():
    ap = argparse.ArgumentParser(
        description="Cross-reference SAE clusters with GTF annotations"
    )
    ap.add_argument("--clusters", required=True,
                    help="Cluster assignments TSV from analyze_sae_regions.py")
    ap.add_argument("--sae_results", required=True,
                    help="SAE results TSV from run_sae_on_chromosome_drops.py")
    ap.add_argument("--gtf", required=True,
                    help="GTF annotation file")
    ap.add_argument("--chrom", required=True,
                    help="Chromosome name as in GTF")
    ap.add_argument("--output_dir", default=None,
                    help="Output directory")
    ap.add_argument("--tolerance", type=int, default=500,
                    help="bp tolerance for overlap (default: 500)")
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.clusters), '..', 'annotation_overlay')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading cluster assignments: {args.clusters}")
    clusters = load_cluster_assignments(args.clusters)
    print(f"  {len(clusters)} regions")

    print(f"Loading SAE results: {args.sae_results}")
    sae_results = load_sae_results(args.sae_results)

    # Merge cluster info into sae_results by region index
    for i, region in enumerate(sae_results):
        if i < len(clusters):
            region.update(clusters[i])

    print(f"Parsing GTF: {args.gtf}")
    gtf_features = parse_gtf_genes(args.gtf, args.chrom)
    print(f"  {len(gtf_features)} features for {args.chrom}")

    # Annotate regions
    print("Annotating regions with GTF overlap...")
    annotated = annotate_regions(sae_results, gtf_features, tolerance=args.tolerance)

    n_annotated = sum(1 for r in annotated if r['is_annotated'])
    n_intergenic = len(annotated) - n_annotated
    print(f"  Annotated: {n_annotated}, Intergenic: {n_intergenic}")

    # Gene biotype distribution
    biotype_counts = Counter()
    for r in annotated:
        for bt in r['gene_biotype'].split(','):
            biotype_counts[bt] += 1

    print("\nGene biotype distribution:")
    for bt, count in biotype_counts.most_common(15):
        print(f"  {bt}: {count}")

    # Cluster enrichment
    enrichment = compute_cluster_enrichment(annotated)

    print("\nCluster enrichment summary:")
    for cluster_id, info in sorted(enrichment.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
        print(f"  Cluster {cluster_id} (n={info['n_regions']}): "
              f"dominant={info['dominant_biotype']}")
        for bt, fold in sorted(info['biotype_enrichment'].items(), key=lambda x: -x[1])[:3]:
            if fold > 1.5:
                print(f"    {bt}: {fold:.1f}x enriched")

    # Save outputs
    ann_tsv = os.path.join(args.output_dir, 'annotated_regions.tsv')
    with open(ann_tsv, 'w') as f:
        if annotated:
            keys = list(annotated[0].keys())
            f.write('\t'.join(keys) + '\n')
            for row in annotated:
                f.write('\t'.join(str(row.get(k, '')) for k in keys) + '\n')
    print(f"\nAnnotated regions: {ann_tsv}")

    enrichment_path = os.path.join(args.output_dir, 'cluster_enrichment.json')
    with open(enrichment_path, 'w') as f:
        json.dump(enrichment, f, indent=2)
    print(f"Enrichment: {enrichment_path}")

    # Generate plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Stacked bar: biotype composition per cluster
        cluster_ids = sorted(enrichment.keys(), key=lambda x: int(x) if x.isdigit() else 999)
        top_biotypes = [bt for bt, _ in biotype_counts.most_common(8)]

        fig, ax = plt.subplots(figsize=(10, 6))
        bottom = np.zeros(len(cluster_ids))
        for bt in top_biotypes:
            values = [enrichment[c]['biotype_counts'].get(bt, 0) for c in cluster_ids]
            ax.bar(cluster_ids, values, bottom=bottom, label=bt)
            bottom += np.array(values, dtype=float)

        ax.set_xlabel('Cluster')
        ax.set_ylabel('Region count')
        ax.set_title('Gene biotype composition per SAE cluster')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'cluster_biotype_composition.png'), dpi=150)
        plt.close()

        # Heatmap: enrichment fold-change
        all_biotypes_in_enrichment = set()
        for info in enrichment.values():
            all_biotypes_in_enrichment.update(info['biotype_enrichment'].keys())
        biotypes_list = sorted(all_biotypes_in_enrichment)

        if len(biotypes_list) > 1 and len(cluster_ids) > 1:
            matrix = np.ones((len(cluster_ids), len(biotypes_list)))
            for i, c in enumerate(cluster_ids):
                for j, bt in enumerate(biotypes_list):
                    matrix[i, j] = enrichment[c]['biotype_enrichment'].get(bt, 0)

            fig, ax = plt.subplots(figsize=(max(8, len(biotypes_list)), max(4, len(cluster_ids) * 0.5)))
            im = ax.imshow(np.log2(matrix + 0.01), cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
            ax.set_xticks(range(len(biotypes_list)))
            ax.set_xticklabels(biotypes_list, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(cluster_ids)))
            ax.set_yticklabels([f'Cluster {c}' for c in cluster_ids])
            plt.colorbar(im, label='log2(fold enrichment)')
            ax.set_title('Gene biotype enrichment per SAE cluster')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'cluster_enrichment_heatmap.png'), dpi=150)
            plt.close()

        print(f"Plots saved to {args.output_dir}")
    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
