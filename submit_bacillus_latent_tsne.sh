#!/bin/bash
#SBATCH -J bacillus_latent_tsne
#SBATCH -p pi_zhang_f
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/bacillus_latent_tsne_%j.out
#SBATCH -e logs/bacillus_latent_tsne_%j.err

set -e

module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

SAE_RUN="results/NC_000964.3/sae/20260324_151121_all_conf8.0_merged4of4"
GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf"

echo "[$(date)] Starting Bacillus latent analysis and t-SNE plots..."

# Step 1: Run latent analysis (compute embeddings, clusters, etc.) with global normalization
echo "[$(date)] Step 1/2: Computing latent embeddings and clusters..."
python tools/analyze_sae_regions.py \
    --input_dir "${SAE_RUN}" \
    --embedding both \
    --leiden_resolution 0.5 \
    --global_stats "results/NC_000964.3/sae_global_stats/20260317_232751_fused_minmax/data/global_sae_stats.npz" \
    2>&1 | tee logs/bacillus_latent_analysis.log

# Step 2: Generate t-SNE plots with genomic annotation
echo "[$(date)] Step 2/2: Generating annotated t-SNE plots..."
python tools/plot_tsne_by_annotation.py \
    --sae_run "${SAE_RUN}" \
    --gtf "${GTF}" \
    --chrom NC_000964.3 \
    2>&1 | tee logs/bacillus_tsne_plots.log

echo "[$(date)] Bacillus latent analysis and t-SNE plots COMPLETED"
