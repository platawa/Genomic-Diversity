#!/usr/bin/env python3
"""Lightweight replot of tsne_4panel.png and tsne_by_annotation.png only.

Reads cluster_assignments.tsv (already has t-SNE coords) and replots
without loading the heavy cosine similarity or maxpooled vector matrices.
"""
import os, sys, csv, logging, argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools"))

from analyze_sae_regions import plot_embedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("replot_tsne_only")

GTF = "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"

CHROMS = [
    "chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9",
    "chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17",
    "chr18","chr20","chr21","chr22","chrX",
]


def load_cluster_tsv(path):
    """Load cluster_assignments.tsv and return metadata, clusters, tsne coords."""
    region_metadata = []
    clusters = []
    tsne_coords = []
    with open(path) as f:
        # Skip comment lines starting with #
        lines = [line for line in f if not line.startswith('#')]
    reader = csv.DictReader(lines, delimiter='\t')
    for row in reader:
        region_metadata.append({
            'genomic_start': int(row.get('genomic_start', 0)),
            'genomic_end': int(row.get('genomic_end', 0)),
            'region_length': int(row.get('region_length', 0)),
            'method': row.get('method', ''),
            'confidence': float(row.get('confidence', row.get('start_confidence', 0))),
        })
        clusters.append(int(row.get('cluster', row.get('cluster_id', 0))))
        # Support both column naming conventions
        tx = row.get('tsne_x') or row.get('tsne_1')
        ty = row.get('tsne_y') or row.get('tsne_2')
        if tx is not None and ty is not None:
            tsne_coords.append([float(tx), float(ty)])
    return region_metadata, np.array(clusters), np.array(tsne_coords) if tsne_coords else None


def replot_4panel(chrom, variant):
    """Replot tsne_4panel.png for one chromosome + variant."""
    base = os.path.join("results", chrom, "sae", variant, "data", "cluster_assignments.tsv")
    if not os.path.exists(base):
        return False
    plots_dir = os.path.join("results", chrom, "sae", variant, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    region_metadata, clusters, tsne = load_cluster_tsv(base)
    if tsne is None or len(tsne) == 0:
        logger.warning(f"  No t-SNE coords for {chrom}/{variant}")
        return False

    out = os.path.join(plots_dir, "tsne_4panel.png")
    plot_embedding(tsne, region_metadata, clusters, out, embedding_name="t-SNE", logger=logger)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroms", nargs="*", default=CHROMS)
    parser.add_argument("--skip-annotation", action="store_true",
                        help="Skip tsne_by_annotation (only replot 4-panel)")
    args = parser.parse_args()

    for chrom in args.chroms:
        logger.info(f"=== {chrom} ===")
        for variant in ["latent_analysis", "latent_analysis_normalized"]:
            ok = replot_4panel(chrom, variant)
            if ok:
                logger.info(f"  {variant}/tsne_4panel.png done")
            else:
                logger.info(f"  {variant} skipped (no data)")

    # Annotation plots via plot_tsne_by_annotation.py (handles both variants)
    if not args.skip_annotation:
        logger.info("\n=== Annotation t-SNE plots ===")
        for chrom in args.chroms:
            # Find the SAE run that actually contains latent_analysis/
            sae_base = os.path.join("results", chrom, "sae")
            la_path = os.path.join(sae_base, "latent_analysis", "data", "cluster_assignments.tsv")
            if not os.path.exists(la_path):
                logger.info(f"  {chrom}: no latent_analysis, skipping annotation plot")
                continue
            # The latent_analysis/ is directly under sae/, so pass sae/ as the run
            logger.info(f"  {chrom}...")
            ret = os.system(
                f"python tools/plot_tsne_by_annotation.py --chrom {chrom} --sae_run {sae_base} --gtf {GTF}"
            )
            if ret != 0:
                logger.warning(f"  annotation plot failed for {chrom}")

    logger.info("\nAll done.")


if __name__ == "__main__":
    main()
