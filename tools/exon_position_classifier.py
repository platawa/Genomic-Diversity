#!/usr/bin/env python3
"""
exon_position_classifier.py

Classifies each drop region as {first_exon, middle_exon, last_exon, non_exon}
based on the region's overlap with annotated transcripts in a GTF.

Outputs a TSV (region_idx, chrom, start, end, class, transcript_id, exon_index,
n_exons_in_transcript) and a categorical numpy array usable as a colormap input
by enhanced_latent_plots.py.

Rules:
  - If no overlap with any exon → "non_exon"
  - If overlap with an exon, pick the transcript whose exon overlaps the region
    center the most. Use its exon index (1-based) and total exon count.
  - exon_index == 1                → first_exon
  - exon_index == n_exons          → last_exon
  - otherwise                      → middle_exon
  - Single-exon transcripts (n_exons == 1) → "first_exon" (by convention; such
    regions are also the last exon, but first takes precedence).
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
from results_utils import build_run_dir, write_completed
from plot_tsne_by_annotation import CHROM_MAP

logger = logging.getLogger(__name__)

CATEGORIES = ["first_exon", "middle_exon", "last_exon", "non_exon"]


def resolve_chrom_id_for_gtf(chrom):
    """Translate chr22 → NC_000022.11 when the GTF uses NCBI accessions."""
    return CHROM_MAP.get(chrom, chrom)


def parse_gtf_exons(gtf_path, chrom_id):
    """Return dict[transcript_id] -> list of (exon_number, start, end, strand)."""
    transcripts = defaultdict(list)
    # Auto-translate UCSC-style chr names to NCBI accessions used in the GTF
    chrom_id = resolve_chrom_id_for_gtf(chrom_id)

    def _scan(target_id):
        with open(gtf_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 9 or parts[0] != target_id or parts[2] != "exon":
                    continue
                start, end, strand = int(parts[3]), int(parts[4]), parts[6]
                attrs = parts[8]
                tid = None
                enum = None
                for field in attrs.split(";"):
                    field = field.strip()
                    if field.startswith("transcript_id"):
                        tid = field.split('"')[1] if '"' in field else field.split()[-1]
                    elif field.startswith("exon_number"):
                        token = field.split('"')[1] if '"' in field else field.split()[-1]
                        try:
                            enum = int(token)
                        except ValueError:
                            enum = None
                if tid is None:
                    continue
                transcripts[tid].append((enum, start, end, strand))
        return transcripts

    _scan(chrom_id)
    if not transcripts:
        # Try stripping chr prefix
        alt = chrom_id[3:] if chrom_id.startswith("chr") else f"chr{chrom_id}"
        logger.info(f"No exons for {chrom_id}; trying {alt}")
        _scan(alt)

    # Fill in missing exon_number by positional order (transcript-relative, 1-based)
    for tid, exons in transcripts.items():
        if any(e[0] is None for e in exons):
            # Sort by coord + strand
            strand = exons[0][3]
            sorted_exons = sorted(exons, key=lambda e: e[1], reverse=(strand == "-"))
            transcripts[tid] = [(i + 1, s, e, st) for i, (_, s, e, st) in enumerate(sorted_exons)]
        else:
            transcripts[tid] = sorted(exons, key=lambda e: e[0])
    return transcripts


def build_exon_index(transcripts):
    """Flatten to sorted array for fast overlap queries.

    Returns: starts, ends, tids, exon_nums, n_exons_arr
    """
    starts, ends, tids, enums, totals = [], [], [], [], []
    for tid, exons in transcripts.items():
        n = len(exons)
        for enum, s, e, _strand in exons:
            starts.append(s)
            ends.append(e)
            tids.append(tid)
            enums.append(enum)
            totals.append(n)
    idx = np.argsort(starts)
    return (
        np.array(starts)[idx],
        np.array(ends)[idx],
        np.array(tids)[idx],
        np.array(enums)[idx],
        np.array(totals)[idx],
    )


def classify_regions(region_starts, region_ends, transcripts):
    """Return a list of dicts — one per region — with classification fields."""
    ex_starts, ex_ends, tids, enums, totals = build_exon_index(transcripts)
    n = len(region_starts)
    out = []
    for i in range(n):
        rs, re_ = int(region_starts[i]), int(region_ends[i])
        center = (rs + re_) / 2.0
        # Find exons that overlap [rs, re_]
        mask = (ex_starts < re_) & (ex_ends > rs)
        if not mask.any():
            out.append({
                "class": "non_exon",
                "transcript_id": "",
                "exon_index": 0,
                "n_exons": 0,
                "center_dist_to_exon": -1,
            })
            continue
        # Pick the exon whose center is closest to the region center
        candidate_starts = ex_starts[mask]
        candidate_ends = ex_ends[mask]
        centers = (candidate_starts + candidate_ends) / 2.0
        best = int(np.argmin(np.abs(centers - center)))
        enum_i = int(enums[mask][best])
        total_i = int(totals[mask][best])
        tid_i = str(tids[mask][best])
        if total_i == 1 or enum_i == 1:
            klass = "first_exon"
        elif enum_i == total_i:
            klass = "last_exon"
        else:
            klass = "middle_exon"
        out.append({
            "class": klass,
            "transcript_id": tid_i,
            "exon_index": enum_i,
            "n_exons": total_i,
            "center_dist_to_exon": float(abs(centers[best] - center)),
        })
    return out


def classify_to_array(region_starts, region_ends, gtf_path, chrom_id):
    """Convenience wrapper: returns a categorical numpy array of strings."""
    transcripts = parse_gtf_exons(gtf_path, chrom_id)
    logger.info(f"Loaded {len(transcripts)} transcripts from {chrom_id}")
    records = classify_regions(region_starts, region_ends, transcripts)
    return np.array([r["class"] for r in records])


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--chrom", required=True)
    p.add_argument("--gtf", required=True)
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--latent_subdir", default="latent_analysis_prenorm",
                   help="Subdir under results/<chrom>/sae/")
    p.add_argument("--output_dir", default=None,
                   help="Override output directory; default writes alongside cluster_assignments.tsv")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ca_path = os.path.join(args.results_dir, args.chrom, "sae",
                           args.latent_subdir, "data", "cluster_assignments.tsv")
    if not os.path.isfile(ca_path):
        logger.error(f"No cluster_assignments.tsv at {ca_path}")
        return 2

    ca = pd.read_csv(ca_path, sep="\t", comment="#")
    logger.info(f"Loaded {len(ca)} regions from {ca_path}")
    transcripts = parse_gtf_exons(args.gtf, args.chrom)
    logger.info(f"GTF: {len(transcripts)} transcripts on {args.chrom}")

    records = classify_regions(
        ca["genomic_start"].values, ca["genomic_end"].values, transcripts
    )
    df = pd.DataFrame(records)
    df.insert(0, "region_idx", np.arange(len(df)))
    df.insert(1, "chrom", args.chrom)
    df.insert(2, "genomic_start", ca["genomic_start"].values)
    df.insert(3, "genomic_end", ca["genomic_end"].values)

    out_dir = args.output_dir or os.path.dirname(ca_path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exon_position.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    logger.info(f"Wrote {out_path}")

    counts = df["class"].value_counts().to_dict()
    logger.info(f"Class counts: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
