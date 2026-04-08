#!/usr/bin/env python3
"""Plot t-SNE and UMAP colored by percentage of SAE features activated per region."""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_FEATURES = 32768


def find_latent_dir(results_dir, chrom):
    """Find latent analysis directory."""
    direct = os.path.join(results_dir, chrom, "sae", "latent_analysis")
    if os.path.isdir(os.path.join(direct, "data")):
        return direct
    # Try under latest completed SAE run
    sae_dir = os.path.join(results_dir, chrom, "sae")
    if os.path.isdir(sae_dir):
        for entry in sorted(os.listdir(sae_dir), reverse=True):
            candidate = os.path.join(sae_dir, entry, "latent_analysis")
            if os.path.isdir(os.path.join(candidate, "data")):
                return candidate
    return None


def plot_pct(coords, pct, title, xlabel, ylabel, out_path, s=5, alpha=0.7, dpi=200):
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=pct, cmap="viridis",
                    s=s, alpha=alpha, edgecolors="none", rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("% Features Activated", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def process_chrom(results_dir, chrom):
    latent_dir = find_latent_dir(results_dir, chrom)
    if latent_dir is None:
        print(f"  {chrom}: no latent_analysis found, skipping")
        return

    data_dir = os.path.join(latent_dir, "data")
    plots_dir = os.path.join(latent_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    mp_path = os.path.join(data_dir, "maxpooled_vectors.npy")
    if not os.path.isfile(mp_path):
        print(f"  {chrom}: no maxpooled_vectors.npy, skipping")
        return

    mp = np.load(mp_path)
    n_regions = mp.shape[0]
    pct_activated = np.count_nonzero(mp, axis=1) / N_FEATURES * 100

    print(f"  {chrom}: {n_regions} regions, mean {pct_activated.mean():.2f}% features activated")

    # Load embeddings
    for emb_name in ["tsne", "umap"]:
        emb_path = os.path.join(data_dir, f"embedding_{emb_name}.npy")
        if not os.path.isfile(emb_path):
            continue
        coords = np.load(emb_path)
        if len(coords) != n_regions:
            continue
        prefix = emb_name.upper()
        n_pts = len(coords)
        s = 5 if n_pts < 5000 else (2 if n_pts < 50000 else 0.5)
        alpha = 0.7 if n_pts < 5000 else (0.5 if n_pts < 50000 else 0.3)

        plot_pct(coords, pct_activated,
                 title=f"{prefix} — % Features Activated\n({chrom}, N={n_regions})",
                 xlabel=f"{prefix} 1", ylabel=f"{prefix} 2",
                 out_path=os.path.join(plots_dir, f"{emb_name}_pct_features_activated.png"),
                 s=s, alpha=alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--chrom", default=None, help="Single chromosome")
    parser.add_argument("--all_human", action="store_true")
    parser.add_argument("--organism", default=None, choices=["ecoli", "bacillus"])
    args = parser.parse_args()

    chroms = []
    if args.chrom:
        chroms = [args.chrom]
    elif args.all_human:
        chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    elif args.organism == "ecoli":
        chroms = ["NC_000913.3"]
    elif args.organism == "bacillus":
        chroms = ["NC_000964.3"]

    for chrom in chroms:
        process_chrom(args.results_dir, chrom)

    print("Done.")


if __name__ == "__main__":
    main()
