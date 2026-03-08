#!/usr/bin/env python3
"""
curate_test_loci.py

Extract and curate 15 test loci (5 per organism) from GTF annotations
for use in detection method evaluation.

Outputs a JSON file with locus definitions including genomic coordinates,
expected features, and evaluation windows.

Usage:
    python tools/curate_test_loci.py \
        --human_gtf /path/to/human/genomic.gtf \
        --ecoli_gtf /path/to/ecoli/genomic.gtf \
        --bacillus_gtf /path/to/bacillus/genomic.gtf \
        --output test_loci.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from typing import Dict, List, Optional
from collections import defaultdict


def parse_gtf_attribute(attr_str: str, key: str) -> Optional[str]:
    """Extract a value from GTF attribute string."""
    for field in attr_str.split(';'):
        field = field.strip()
        if field.startswith(key):
            parts = field.split('"')
            if len(parts) >= 2:
                return parts[1]
    return None


def find_gene(gtf_path: str, chrom: str, gene_name: str) -> Optional[dict]:
    """Find a specific gene by name in GTF, return its coordinates and exon count."""
    gene_info = None
    exon_count = 0
    cds_count = 0

    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[0] != chrom:
                continue

            name = parse_gtf_attribute(fields[8], 'gene_name')
            gid = parse_gtf_attribute(fields[8], 'gene_id')
            if name != gene_name and gid != gene_name:
                continue

            feat_type = fields[2]
            if feat_type == 'gene':
                gene_info = {
                    'gene_name': gene_name,
                    'chrom': chrom,
                    'start': int(fields[3]) - 1,
                    'end': int(fields[4]),
                    'strand': fields[6],
                    'gene_type': parse_gtf_attribute(fields[8], 'gene_biotype') or 'unknown',
                }
            elif feat_type == 'exon':
                exon_count += 1
            elif feat_type == 'CDS':
                cds_count += 1

    if gene_info:
        gene_info['n_exons'] = exon_count
        gene_info['n_cds'] = cds_count
        gene_info['length'] = gene_info['end'] - gene_info['start']
    return gene_info


def find_genes_by_type(gtf_path: str, chrom: str, gene_type: str,
                       max_results: int = 5) -> List[dict]:
    """Find genes of a specific biotype."""
    genes = {}
    exon_counts = defaultdict(int)
    cds_counts = defaultdict(int)

    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[0] != chrom:
                continue

            biotype = parse_gtf_attribute(fields[8], 'gene_biotype')
            if biotype != gene_type:
                continue

            name = (parse_gtf_attribute(fields[8], 'gene_name') or
                    parse_gtf_attribute(fields[8], 'gene_id'))
            if not name:
                continue

            feat_type = fields[2]
            if feat_type == 'gene':
                genes[name] = {
                    'gene_name': name,
                    'chrom': chrom,
                    'start': int(fields[3]) - 1,
                    'end': int(fields[4]),
                    'strand': fields[6],
                    'gene_type': biotype,
                }
            elif feat_type == 'exon':
                exon_counts[name] += 1
            elif feat_type == 'CDS':
                cds_counts[name] += 1

    results = []
    for name, info in genes.items():
        info['n_exons'] = exon_counts.get(name, 0)
        info['n_cds'] = cds_counts.get(name, 0)
        info['length'] = info['end'] - info['start']
        results.append(info)

    results.sort(key=lambda x: -x['length'])
    return results[:max_results]


def curate_human_loci(gtf_path: str, chrom: str = 'NC_000022.11') -> List[dict]:
    """Curate 5 human chr22 test loci."""
    loci = []

    # 1. EWSR1 - complex multi-exon gene (22 exons, ~40kb)
    gene = find_gene(gtf_path, chrom, 'EWSR1')
    if gene:
        gene['rationale'] = 'Complex multi-exon gene (RNA-binding protein), many splice sites'
        gene['expected_features'] = 'Multiple drops at exon boundaries, complex internal structure'
        loci.append(gene)

    # 2. Immunoglobulin lambda locus
    for name in ['IGLV1-44', 'IGLC1', 'IGLJ1']:
        gene = find_gene(gtf_path, chrom, name)
        if gene:
            gene['rationale'] = 'Immunoglobulin lambda variable region - highly variable, interesting entropy pattern'
            gene['expected_features'] = 'Variable entropy due to somatic hypermutation targets'
            loci.append(gene)
            break

    # 3. tRNA gene
    trnas = find_genes_by_type(gtf_path, chrom, 'tRNA')
    if trnas:
        trna = trnas[0]
        trna['rationale'] = 'Small, highly conserved tRNA gene (~75bp)'
        trna['expected_features'] = 'Sharp, narrow drop - strong conservation signal'
        loci.append(trna)

    # 4. miRNA gene
    for biotype in ['miRNA', 'misc_RNA', 'snRNA']:
        mirnas = find_genes_by_type(gtf_path, chrom, biotype)
        if mirnas:
            mirna = mirnas[0]
            mirna['rationale'] = f'Small non-coding RNA ({biotype})'
            mirna['expected_features'] = 'Narrow drop, distinct from protein-coding pattern'
            loci.append(mirna)
            break

    # 5. Simple single-exon gene
    genes = find_genes_by_type(gtf_path, chrom, 'protein_coding')
    for g in genes:
        if g['n_exons'] == 1 and 500 < g['length'] < 5000:
            g['rationale'] = 'Simple single-exon protein-coding gene'
            g['expected_features'] = 'Single drop region with clear boundaries'
            loci.append(g)
            break

    return loci[:5]


def curate_ecoli_loci(gtf_path: str, chrom: str = 'NC_000913.3') -> List[dict]:
    """Curate 5 E. coli K-12 test loci."""
    loci = []

    targets = [
        ('rrsB', 'Ribosomal RNA (16S rRNA in rrnB operon) - highly conserved'),
        ('lacZ', 'lac operon beta-galactosidase - well-studied operon structure'),
        ('ssrA', 'tmRNA (ssrA) - dual-function RNA, unique structure'),
        ('cas1', 'CRISPR-associated protein - part of CRISPR array'),
        ('rpoB', 'RNA polymerase beta subunit - essential, highly conserved'),
    ]

    for name, rationale in targets:
        gene = find_gene(gtf_path, chrom, name)
        if gene:
            gene['rationale'] = rationale
            gene['expected_features'] = 'Strong entropy drop in conserved coding region'
            loci.append(gene)

    # Fill remaining with tRNAs
    if len(loci) < 5:
        trnas = find_genes_by_type(gtf_path, chrom, 'tRNA')
        for t in trnas:
            if len(loci) >= 5:
                break
            t['rationale'] = 'tRNA gene cluster'
            t['expected_features'] = 'Sharp narrow drop'
            loci.append(t)

    return loci[:5]


def curate_bacillus_loci(gtf_path: str, chrom: str = 'NC_000964.3') -> List[dict]:
    """Curate 5 B. subtilis test loci."""
    loci = []

    targets = [
        ('spo0A', 'Master regulator of sporulation - key transcription factor'),
        ('comK', 'Competence transcription factor - DNA uptake regulation'),
        ('rpsA', 'Ribosomal protein S1 - highly conserved'),
        ('dnaA', 'Chromosomal replication initiator - essential gene'),
        ('sigA', 'Primary sigma factor - housekeeping transcription'),
    ]

    for name, rationale in targets:
        gene = find_gene(gtf_path, chrom, name)
        if gene:
            gene['rationale'] = rationale
            gene['expected_features'] = 'Strong entropy drop in conserved coding region'
            loci.append(gene)

    # Fill with rRNA if needed
    if len(loci) < 5:
        rrnas = find_genes_by_type(gtf_path, chrom, 'rRNA')
        for r in rrnas:
            if len(loci) >= 5:
                break
            r['rationale'] = 'Ribosomal RNA - highly conserved'
            r['expected_features'] = 'Very strong entropy drop'
            loci.append(r)

    return loci[:5]


def main():
    ap = argparse.ArgumentParser(
        description="Curate test loci from GTF for detection method evaluation"
    )
    ap.add_argument("--human_gtf", default=None, help="Human GTF file")
    ap.add_argument("--human_chrom", default="NC_000022.11", help="Human chromosome name in GTF")
    ap.add_argument("--ecoli_gtf", default=None, help="E. coli GTF file")
    ap.add_argument("--ecoli_chrom", default="NC_000913.3", help="E. coli chromosome name in GTF")
    ap.add_argument("--bacillus_gtf", default=None, help="B. subtilis GTF file")
    ap.add_argument("--bacillus_chrom", default="NC_000964.3", help="B. subtilis chromosome name in GTF")
    ap.add_argument("--output", required=True, help="Output JSON file")
    args = ap.parse_args()

    all_loci = {}

    if args.human_gtf:
        print(f"Curating human chr22 loci from {args.human_gtf}...")
        human_loci = curate_human_loci(args.human_gtf, args.human_chrom)
        all_loci['human'] = human_loci
        print(f"  Found {len(human_loci)} loci:")
        for l in human_loci:
            print(f"    {l['gene_name']}: {l['chrom']}:{l['start']}-{l['end']} "
                  f"({l['length']}bp, {l['n_exons']} exons)")

    if args.ecoli_gtf:
        print(f"\nCurating E. coli loci from {args.ecoli_gtf}...")
        ecoli_loci = curate_ecoli_loci(args.ecoli_gtf, args.ecoli_chrom)
        all_loci['ecoli'] = ecoli_loci
        print(f"  Found {len(ecoli_loci)} loci:")
        for l in ecoli_loci:
            print(f"    {l['gene_name']}: {l['chrom']}:{l['start']}-{l['end']} "
                  f"({l['length']}bp)")

    if args.bacillus_gtf:
        print(f"\nCurating B. subtilis loci from {args.bacillus_gtf}...")
        bacillus_loci = curate_bacillus_loci(args.bacillus_gtf, args.bacillus_chrom)
        all_loci['bacillus'] = bacillus_loci
        print(f"  Found {len(bacillus_loci)} loci:")
        for l in bacillus_loci:
            print(f"    {l['gene_name']}: {l['chrom']}:{l['start']}-{l['end']} "
                  f"({l['length']}bp)")

    total = sum(len(v) for v in all_loci.values())
    print(f"\nTotal: {total} test loci across {len(all_loci)} organisms")

    with open(args.output, 'w') as f:
        json.dump(all_loci, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
