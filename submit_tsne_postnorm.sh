#!/bin/bash
#SBATCH -J tsne_postnorm
#SBATCH -p mit_preemptable
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/tsne_postnorm_%j.log
#SBATCH -e logs/tsne_postnorm_%j.err

# Genome-wide t-SNE/UMAP on POST-NORMALIZED vectors (Option B)
# Vectors are max-pooled raw then z-scored with chunk-max stats, so NO --global_stats needed here
set -e
module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

echo "[$(date)] Running genome-wide t-SNE/UMAP on post-normalized vectors (Option B)..."
python tools/genome_sae_tsne.py \
    --all_human \
    --gtf /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf \
    --results_dir results/ \
    --latent_subdir latent_analysis_postnorm \
    --output_dir results/_genome_wide/sae_tsne_postnorm \
    --embedding both \
    --n_pca 50

echo "[$(date)] Done successfully."
