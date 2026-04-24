#!/bin/bash
#SBATCH -J norm_gw10
#SBATCH -p mit_preemptable
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/norm_gw10_%j.log
#SBATCH -e logs/norm_gw10_%j.err

set -e
module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

CACHE_DIR=results/_genome_wide/sae_tsne/_cache

# ── Safety: remove corrupted normalized cache if present ─────────────────
if [ -f "$CACHE_DIR/combined_maxpooled_normalized.npy" ]; then
    SIZE_GB=$(du --block-size=1G "$CACHE_DIR/combined_maxpooled_normalized.npy" 2>/dev/null | awk '{print int($1)}')
    if [ "$SIZE_GB" -lt 80 ]; then
        echo "[$(date)] Removing corrupted normalized cache (${SIZE_GB}GB < 80GB expected)"
        rm -f "$CACHE_DIR/combined_maxpooled_normalized.npy"
        rm -f "$CACHE_DIR/combined_metadata_normalized.json"
    fi
fi

echo "[$(date)] Running normalized t-SNE pipeline (--embedding both --n_pca 50)..."
echo "[$(date)] Script will rebuild combined vectors from per-chrom SAE runs + apply z-score normalization"

python tools/genome_sae_tsne.py \
    --all_human \
    --gtf /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf \
    --results_dir results/ \
    --global_stats results/_genome_sae_stats/20260406_235042_corrected_24chroms/data/genome_wide_sae_stats_corrected.npz \
    --latent_subdir latent_analysis_normalized \
    --embedding both \
    --n_pca 50

echo "[$(date)] Done successfully."
