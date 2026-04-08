#!/bin/bash
# submit_bacteria_normalization.sh — Normalize bacteria SAE features using
# human genome-wide stats, compute normalized embeddings + enhanced plots.
#
# Uses the same 32768-dim SAE feature space as human, so cross-species
# normalization with human genome-wide nuc_mean/nuc_std is valid.
#
# Usage:
#   sbatch submit_bacteria_normalization.sh
#   # or just: bash submit_bacteria_normalization.sh  (if on a compute node)

set -euo pipefail

PROJECT="/orcd/data/zhang_f/001/platawa/jan31_files"
GLOBAL_STATS="${PROJECT}/results/_genome_sae_stats/20260406_235042_corrected_24chroms/data/genome_wide_sae_stats_corrected.npz"
ECOLI_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf"
BACILLUS_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf"
LOGDIR="${PROJECT}/logs"
SETUP="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28"

mkdir -p "$LOGDIR"

echo "=== Bacteria Normalization Pipeline ==="
echo "Global stats: ${GLOBAL_STATS}"
echo ""

# ── E. coli: normalize + compute latent (from input_dir) ──────────────
ECOLI_RUN="results/NC_000913.3/sae/20260309_134205_max1000_conf3.0"
echo "=== Job 1: E. coli normalized latent analysis ==="
JOB_ECOLI=$(sbatch --parsable --job-name="norm_ecoli" --partition=pi_zhang_f \
    --cpus-per-task=4 --mem=32G --time=1:00:00 \
    --output="${LOGDIR}/norm_ecoli_%j.out" --error="${LOGDIR}/norm_ecoli_%j.err" \
    --wrap="${SETUP} && python tools/compute_sae_latent.py \
        --input_dir ${ECOLI_RUN} \
        --chrom NC_000913.3 \
        --results_dir results/ \
        --global_stats ${GLOBAL_STATS} \
        --stage both \
        --embedding both")
echo "  E. coli: ${JOB_ECOLI}"

# ── Bacillus: normalize + compute latent ──────────────────────────────
BACILLUS_RUN="results/NC_000964.3/sae/20260324_151121_all_conf8.0_merged4of4"
echo "=== Job 2: Bacillus normalized latent analysis ==="
JOB_BACILLUS=$(sbatch --parsable --job-name="norm_bacil" --partition=pi_zhang_f \
    --cpus-per-task=4 --mem=32G --time=1:00:00 \
    --output="${LOGDIR}/norm_bacil_%j.out" --error="${LOGDIR}/norm_bacil_%j.err" \
    --wrap="${SETUP} && python tools/compute_sae_latent.py \
        --input_dir ${BACILLUS_RUN} \
        --chrom NC_000964.3 \
        --results_dir results/ \
        --global_stats ${GLOBAL_STATS} \
        --stage both \
        --embedding both")
echo "  Bacillus: ${JOB_BACILLUS}"

# ── Enhanced plots for E. coli (depends on Job 1) ────────────────────
echo "=== Job 3: E. coli enhanced plots (depends on Job 1) ==="
JOB_ECOLI_ENH=$(sbatch --parsable --dependency=afterok:${JOB_ECOLI} \
    --job-name="enh_ecoli" --partition=pi_zhang_f \
    --cpus-per-task=4 --mem=32G --time=1:00:00 \
    --output="${LOGDIR}/enh_ecoli_%j.out" --error="${LOGDIR}/enh_ecoli_%j.err" \
    --wrap="${SETUP} && python tools/enhanced_latent_plots.py \
        --scope organism --organism ecoli \
        --gtf ${ECOLI_GTF} \
        --results_dir results/ \
        --latent_subdir latent_analysis_normalized \
        --plots distance_to_gene,entropy_color,length_stats,firing_counts,firing_thresholds")
echo "  E. coli enhanced: ${JOB_ECOLI_ENH}"

# ── Enhanced plots for Bacillus (depends on Job 2) ───────────────────
echo "=== Job 4: Bacillus enhanced plots (depends on Job 2) ==="
JOB_BACILLUS_ENH=$(sbatch --parsable --dependency=afterok:${JOB_BACILLUS} \
    --job-name="enh_bacil" --partition=pi_zhang_f \
    --cpus-per-task=4 --mem=32G --time=1:00:00 \
    --output="${LOGDIR}/enh_bacil_%j.out" --error="${LOGDIR}/enh_bacil_%j.err" \
    --wrap="${SETUP} && python tools/enhanced_latent_plots.py \
        --scope organism --organism bacillus \
        --gtf ${BACILLUS_GTF} \
        --results_dir results/ \
        --latent_subdir latent_analysis_normalized \
        --plots distance_to_gene,entropy_color,length_stats,firing_counts,firing_thresholds")
echo "  Bacillus enhanced: ${JOB_BACILLUS_ENH}"

# ── Annotation + CRISPR plots for both (depends on Jobs 1,2) ─────────
echo "=== Job 5: E. coli annotation t-SNE (depends on Job 1) ==="
JOB_ECOLI_ANN=$(sbatch --parsable --dependency=afterok:${JOB_ECOLI} \
    --job-name="ann_ecoli" --partition=pi_zhang_f \
    --cpus-per-task=2 --mem=16G --time=0:30:00 \
    --output="${LOGDIR}/ann_ecoli_%j.out" --error="${LOGDIR}/ann_ecoli_%j.err" \
    --wrap="${SETUP} && python tools/plot_tsne_by_annotation.py \
        --chrom NC_000913.3 --auto \
        --gtf ${ECOLI_GTF} \
        --results_dir results/ \
        --sae_run ${ECOLI_RUN}")
echo "  E. coli annotation: ${JOB_ECOLI_ANN}"

echo "=== Job 6: Bacillus annotation t-SNE (depends on Job 2) ==="
JOB_BACILLUS_ANN=$(sbatch --parsable --dependency=afterok:${JOB_BACILLUS} \
    --job-name="ann_bacil" --partition=pi_zhang_f \
    --cpus-per-task=2 --mem=16G --time=0:30:00 \
    --output="${LOGDIR}/ann_bacil_%j.out" --error="${LOGDIR}/ann_bacil_%j.err" \
    --wrap="${SETUP} && python tools/plot_tsne_by_annotation.py \
        --chrom NC_000964.3 --auto \
        --gtf ${BACILLUS_GTF} \
        --results_dir results/ \
        --sae_run ${BACILLUS_RUN}")
echo "  Bacillus annotation: ${JOB_BACILLUS_ANN}"

echo ""
echo "=== SUMMARY ==="
echo "Job 1: ${JOB_ECOLI}  — E. coli normalized latent (no deps)"
echo "Job 2: ${JOB_BACILLUS}  — Bacillus normalized latent (no deps)"
echo "Job 3: ${JOB_ECOLI_ENH}  — E. coli enhanced plots (after Job 1)"
echo "Job 4: ${JOB_BACILLUS_ENH}  — Bacillus enhanced plots (after Job 2)"
echo "Job 5: ${JOB_ECOLI_ANN}  — E. coli annotation plots (after Job 1)"
echo "Job 6: ${JOB_BACILLUS_ANN}  — Bacillus annotation plots (after Job 2)"
echo ""
echo "Total: 6 jobs. Monitor with: squeue -u platawa"
