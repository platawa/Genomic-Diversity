#!/bin/bash
# submit_fix_genome_norm.sh — Regenerate normalized combined vectors from
# per-chromosome latent_analysis_normalized, then rerun genome-wide t-SNE
# and UMAP. Does NOT use the SSD symlink trick (that caused data loss).
#
# Usage: bash submit_fix_genome_norm.sh

set -euo pipefail

PROJECT="/orcd/data/zhang_f/001/platawa/jan31_files"
HUMAN_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"
GLOBAL_STATS="${PROJECT}/results/_genome_sae_stats/20260406_235042_corrected_24chroms/data/genome_wide_sae_stats_corrected.npz"
LOGDIR="${PROJECT}/logs"

mkdir -p "$LOGDIR"

# Job 1: Regenerate combined_maxpooled_normalized.npy + run t-SNE
echo "=== Job 1: Genome-wide normalized t-SNE (no SSD trick) ==="
JOB_TSNE=$(sbatch --parsable --job-name="gw_norm_tsne" --partition=pi_zhang_f \
    --cpus-per-task=32 --mem=500G --time=2-00:00:00 \
    --output="${LOGDIR}/gw_norm_tsne_%j.out" --error="${LOGDIR}/gw_norm_tsne_%j.err" \
    --wrap="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/genome_sae_tsne.py \
        --all_human \
        --gtf ${HUMAN_GTF} \
        --results_dir results/ \
        --global_stats ${GLOBAL_STATS} \
        --latent_subdir latent_analysis_normalized \
        --embedding tsne \
        --n_pca 50")
echo "  t-SNE: ${JOB_TSNE}"

# Job 2: UMAP on normalized (depends on Job 1 for combined vectors)
echo "=== Job 2: Genome-wide normalized UMAP (depends on Job 1) ==="
JOB_UMAP=$(sbatch --parsable --dependency=afterok:${JOB_TSNE} \
    --job-name="gw_norm_umap" --partition=pi_zhang_f \
    --cpus-per-task=32 --mem=500G --time=1-00:00:00 \
    --output="${LOGDIR}/gw_norm_umap_%j.out" --error="${LOGDIR}/gw_norm_umap_%j.err" \
    --wrap="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/genome_sae_tsne.py \
        --all_human \
        --gtf ${HUMAN_GTF} \
        --results_dir results/ \
        --global_stats ${GLOBAL_STATS} \
        --latent_subdir latent_analysis_normalized \
        --embedding umap \
        --n_pca 50")
echo "  UMAP: ${JOB_UMAP}"

echo ""
echo "=== SUMMARY ==="
echo "Job 1: ${JOB_TSNE}  — Normalized t-SNE (no deps, rebuilds combined vectors)"
echo "Job 2: ${JOB_UMAP}  — Normalized UMAP (after Job 1)"
echo ""
echo "Monitor: squeue -u platawa"
echo "Logs: tail -f logs/gw_norm_tsne_${JOB_TSNE}.err"
