#!/bin/bash
#SBATCH -J tsne_prenorm
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=64
#SBATCH --mem=380G
#SBATCH -t 12:00:00
#SBATCH -o logs/tsne_prenorm_%j.log
#SBATCH -e logs/tsne_prenorm_%j.err

# Genome-wide t-SNE/UMAP on PRE-NORMALIZED vectors (Option A)
# Vectors are already z-scored before max-pooling, so NO --global_stats needed here
set -e
module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

echo "[$(date)] Running genome-wide t-SNE/UMAP on pre-normalized vectors (Option A)..."
python tools/genome_sae_tsne.py \
    --all_human \
    --gtf /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf \
    --results_dir results/ \
    --latent_subdir latent_analysis_prenorm \
    --output_dir results/_genome_wide/sae_tsne_prenorm \
    --embedding both \
    --n_pca 50

echo "[$(date)] Done successfully."
