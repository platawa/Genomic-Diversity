#!/usr/bin/env python3
"""Replot 4-panel t-SNE for a single E. coli SAE run."""
import sys, os, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("replot_ecoli")

from analyze_sae_regions import plot_embedding

run_dir = sys.argv[1]
path = os.path.join(run_dir, "latent_analysis", "data", "cluster_assignments.tsv")

region_metadata, clusters, tsne_coords = [], [], []
with open(path) as f:
    lines = [l for l in f if not l.startswith("#")]
reader = csv.DictReader(lines, delimiter="\t")
for row in reader:
    region_metadata.append({
        "genomic_start": int(row.get("genomic_start", 0)),
        "genomic_end": int(row.get("genomic_end", 0)),
        "region_length": int(row.get("region_length", 0)),
        "method": row.get("method", ""),
        "confidence": float(row.get("confidence", row.get("start_confidence", 0))),
    })
    clusters.append(int(row.get("cluster", row.get("cluster_id", 0))))
    tx = row.get("tsne_x") or row.get("tsne_1")
    ty = row.get("tsne_y") or row.get("tsne_2")
    if tx and ty:
        tsne_coords.append([float(tx), float(ty)])

clusters = np.array(clusters)
tsne = np.array(tsne_coords)
out = os.path.join(run_dir, "latent_analysis", "plots", "tsne_4panel.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plot_embedding(tsne, region_metadata, clusters, out, embedding_name="t-SNE", logger=logger)
