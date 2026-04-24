#!/usr/bin/env python3
"""
make_ncRNA_figures.py

Emits a 6-panel figure per ncRNA locus for the thesis:
  (1) raw entropy around the locus
  (2) smoothed entropy with a ±N bp margin
  (3) z-score drop track
  (4) MAD drop track
  (5) zoom-in showing drop boundaries
  (6) GTF/annotation track (exons + gene name)

Input: TSV with columns (organism, chrom, start, end, name, scoring_run)
  - scoring_run can be "auto" to pick the latest COMPLETED scoring run
Output: one PNG per row in a single run directory + a manifest TSV.

Example locus TSV:
    organism    chrom           start       end         name        scoring_run
    human       chr11           5225464     5229395     HBB         auto
    ecoli       NC_000913.3     2820000     2821000     ssrA        auto
    human       chrX            73820651    73852723    XIST        auto
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, find_latest_completed, write_completed

logger = logging.getLogger(__name__)

# GRCh38 chrom-name aliases: NCBI RefSeq GTFs use NC_... accessions in column 1,
# but our results/ tree and locus TSVs use chr... names. Normalize both.
GRCH38_CHROM_ALIASES = {
    "chr1": "NC_000001.11", "chr2": "NC_000002.12", "chr3": "NC_000003.12",
    "chr4": "NC_000004.12", "chr5": "NC_000005.10", "chr6": "NC_000006.12",
    "chr7": "NC_000007.14", "chr8": "NC_000008.11", "chr9": "NC_000009.12",
    "chr10": "NC_000010.11", "chr11": "NC_000011.10", "chr12": "NC_000012.12",
    "chr13": "NC_000013.11", "chr14": "NC_000014.9",  "chr15": "NC_000015.10",
    "chr16": "NC_000016.10", "chr17": "NC_000017.11", "chr18": "NC_000018.10",
    "chr19": "NC_000019.10", "chr20": "NC_000020.11", "chr21": "NC_000021.9",
    "chr22": "NC_000022.11", "chrX":  "NC_000023.11", "chrY":  "NC_000024.10",
    "chrM":  "NC_012920.1",
}


def chrom_matches(gtf_chrom: str, query_chrom: str) -> bool:
    """True if gtf_chrom and query_chrom refer to the same sequence.
    Handles chr11 <-> NC_000011.10 style RefSeq/UCSC name mismatches."""
    if gtf_chrom == query_chrom:
        return True
    if GRCH38_CHROM_ALIASES.get(query_chrom) == gtf_chrom:
        return True
    # reverse: query has NC_, gtf has chr
    for chr_name, refseq in GRCH38_CHROM_ALIASES.items():
        if refseq == query_chrom and chr_name == gtf_chrom:
            return True
    return False


def resolve_scoring_run(results_dir, chrom, run_spec):
    if run_spec and run_spec != "auto":
        return os.path.join(results_dir, chrom, "scoring", run_spec)
    return find_latest_completed(results_dir, chrom, "scoring")


def load_entropy_region(run_dir, start, end, pad):
    npz_path = os.path.join(run_dir, "data", "entropy.npz")
    if not os.path.isfile(npz_path):
        return None
    nz = np.load(npz_path)
    if "entropy" not in nz.files:
        return None
    ent = nz["entropy"]
    s = max(0, start - pad)
    e = min(len(ent), end + pad)
    return np.arange(s, e), ent[s:e]


def smooth(arr, window):
    if window < 2:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def load_drops(run_dir, method, start, end, pad):
    drops_path = os.path.join(run_dir, "data", "drops.tsv")
    if not os.path.isfile(drops_path):
        return pd.DataFrame()
    df = pd.read_csv(drops_path, sep="\t", comment="#")
    if "method" in df.columns:
        df = df[df["method"] == method]
    if "genomic_pos" in df.columns:
        df = df[(df["genomic_pos"] >= start - pad) & (df["genomic_pos"] <= end + pad)]
    elif "position" in df.columns:
        df = df[(df["position"] >= start - pad) & (df["position"] <= end + pad)]
    return df


def load_gtf_exons(gtf_path, chrom, start, end, pad):
    if not gtf_path or not os.path.isfile(gtf_path):
        return []
    rows = []
    lo, hi = start - pad, end + pad
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            if not chrom_matches(parts[0], chrom):
                continue
            s, e = int(parts[3]), int(parts[4])
            if e < lo or s > hi:
                continue
            gname = None
            for field in parts[8].split(";"):
                f_ = field.strip()
                if f_.startswith("gene_name") or f_.startswith("gene_id"):
                    gname = f_.split('"')[1] if '"' in f_ else f_.split()[-1]
                    break
            rows.append((s, e, parts[6], gname))
    return rows


def plot_locus(row, results_dir, gtf_path, out_path, pad, smooth_window):
    run_dir = resolve_scoring_run(results_dir, row["chrom"], row.get("scoring_run", "auto"))
    if run_dir is None:
        logger.warning(f"{row['name']}: no scoring run for {row['chrom']}")
        return False
    ent_data = load_entropy_region(run_dir, row["start"], row["end"], pad)
    if ent_data is None:
        logger.warning(f"{row['name']}: missing entropy.npz at {run_dir}")
        return False
    pos, ent = ent_data
    smoothed = smooth(ent, smooth_window)
    zscore_drops = load_drops(run_dir, "zscore", row["start"], row["end"], pad)
    mad_drops = load_drops(run_dir, "mad", row["start"], row["end"], pad)
    exons = load_gtf_exons(gtf_path, row["chrom"], row["start"], row["end"], pad)

    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(pos, ent, color="tab:gray", lw=0.6)
    axes[0].set_title(f"{row['name']} raw entropy ({row['chrom']}:{row['start']}-{row['end']})")
    axes[0].set_ylabel("entropy")

    axes[1].plot(pos, smoothed, color="tab:blue", lw=1.0)
    axes[1].set_title(f"smoothed (window={smooth_window})")
    axes[1].set_ylabel("entropy")

    for ax, title, df, color in [
        (axes[2], "Z-score drops", zscore_drops, "tab:red"),
        (axes[3], "MAD drops", mad_drops, "tab:purple"),
    ]:
        ax.plot(pos, smoothed, color="tab:gray", lw=0.6, alpha=0.5)
        pos_col = "genomic_pos" if "genomic_pos" in df.columns else ("position" if "position" in df.columns else None)
        if pos_col and len(df):
            ax.scatter(df[pos_col], np.interp(df[pos_col], pos, smoothed),
                       color=color, s=30, label=f"n={len(df)}")
            ax.legend(loc="upper right")
        ax.set_title(title)
        ax.set_ylabel("entropy")

    # Zoom panel
    zoom_margin = int(0.15 * (row["end"] - row["start"] + 1))
    zlo = max(row["start"] - zoom_margin, pos[0])
    zhi = min(row["end"] + zoom_margin, pos[-1])
    mask = (pos >= zlo) & (pos <= zhi)
    axes[4].plot(pos[mask], smoothed[mask], color="tab:green", lw=1.0)
    axes[4].axvspan(row["start"], row["end"], color="yellow", alpha=0.3)
    axes[4].set_title("zoom")
    axes[4].set_ylabel("entropy")

    # Exon track
    axes[5].set_title("exons (GTF)")
    for (es, ee, strand, gname) in exons:
        axes[5].add_patch(plt.Rectangle((es, 0.2), ee - es, 0.6,
                                        color="tab:orange", alpha=0.7))
        if gname:
            axes[5].text((es + ee) / 2, 0.5, gname, ha="center", va="center",
                         fontsize=6, alpha=0.7)
    axes[5].set_ylim(0, 1)
    axes[5].set_xlabel(f"genomic position ({row['chrom']})")
    axes[5].set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")
    return True


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--loci_tsv", required=True)
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--human_gtf", default=None)
    p.add_argument("--ecoli_gtf", default=None)
    p.add_argument("--bacillus_gtf", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--pad", type=int, default=5000, help="bp padding around locus")
    p.add_argument("--smooth_window", type=int, default=51)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    gtf_map = {
        "human": args.human_gtf,
        "ecoli": args.ecoli_gtf,
        "bacillus": args.bacillus_gtf,
    }

    loci = pd.read_csv(args.loci_tsv, sep="\t")
    required_cols = {"organism", "chrom", "start", "end", "name"}
    missing = required_cols - set(loci.columns)
    if missing:
        logger.error(f"Loci TSV missing columns: {missing}")
        return 2

    if args.output_dir is None:
        out_dir = build_run_dir(args.results_dir, "_genome_wide", "ncRNA_figures",
                                f"n{len(loci)}_loci")
    else:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)

    t0 = __import__("time").time()
    manifest = []
    for _, row in loci.iterrows():
        gtf = gtf_map.get(row["organism"])
        fname = f"{row['organism']}_{row['name']}.png".replace("/", "_")
        ok = plot_locus(row, args.results_dir, gtf,
                        os.path.join(out_dir, fname),
                        args.pad, args.smooth_window)
        manifest.append({**row.to_dict(), "plot": fname, "success": ok})
    pd.DataFrame(manifest).to_csv(os.path.join(out_dir, "manifest.tsv"),
                                  sep="\t", index=False)
    write_completed(out_dir, "make_ncRNA_figures.py",
                    __import__("time").time() - t0)
    logger.info(f"Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
