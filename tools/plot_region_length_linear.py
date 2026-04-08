#!/usr/bin/env python3
"""Plot t-SNE and UMAP colored by region length (linear scale, not log)."""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_length(coords, lengths, title, xlabel, ylabel, out_path, s=0.5, alpha=0.3, dpi=300):
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=lengths, cmap="plasma",
                    s=s, alpha=alpha, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Region Length (bp)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsne_tsv", required=True)
    parser.add_argument("--umap_tsv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dot_size", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # t-SNE
    print(f"Loading {args.tsne_tsv}...")
    df_t = pd.read_csv(args.tsne_tsv, sep="\t")
    lengths_t = (df_t["genomic_end"] - df_t["genomic_start"]).values
    coords_t = df_t[["tsne_1", "tsne_2"]].values
    n = len(df_t)
    plot_length(coords_t, lengths_t,
                title=f"t-SNE of SAE Region Fingerprints (N={n:,})\nColored by Region Length (linear)",
                xlabel="t-SNE 1", ylabel="t-SNE 2",
                out_path=os.path.join(args.output_dir, "tsne_region_length_linear.png"),
                s=args.dot_size, alpha=args.alpha, dpi=args.dpi)

    # UMAP
    print(f"Loading {args.umap_tsv}...")
    df_u = pd.read_csv(args.umap_tsv, sep="\t")
    lengths_u = (df_u["genomic_end"] - df_u["genomic_start"]).values
    coords_u = df_u[["umap_1", "umap_2"]].values
    plot_length(coords_u, lengths_u,
                title=f"UMAP of SAE Region Fingerprints (N={len(df_u):,})\nColored by Region Length (linear)",
                xlabel="UMAP 1", ylabel="UMAP 2",
                out_path=os.path.join(args.output_dir, "umap_region_length_linear.png"),
                s=args.dot_size, alpha=args.alpha, dpi=args.dpi)

    print("Done.")


if __name__ == "__main__":
    main()
