#!/bin/bash
# Resubmit nuc_stats for chr1-4 which timed out at 6h.
# Extended to 12h. Checkpoints exist so they will resume.

set -euo pipefail

PROJECT="/orcd/data/zhang_f/001/platawa/jan31_files"
FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna"
LOGDIR="${PROJECT}/logs"
mkdir -p "${LOGDIR}"

CHROMS=(chr1 chr2 chr3 chr4)

echo "Resubmitting nuc_stats for ${CHROMS[@]} with 12h time limit..."
echo ""

for CHROM in "${CHROMS[@]}"; do
    JOB_NAME="nuc_stats_${CHROM}"
    echo "  Submitting ${CHROM}..."

    sbatch \
        --job-name="${JOB_NAME}" \
        --partition=mit_preemptable \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=64G \
        --time=12:00:00 \
        --output="${LOGDIR}/${JOB_NAME}_%j.out" \
        --error="${LOGDIR}/${JOB_NAME}_%j.err" \
        --wrap="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python scan_sae_global_stats.py --fasta ${FASTA} --chrom ${CHROM} --output_dir results/"

    echo "    -> submitted"
done

echo ""
echo "All ${#CHROMS[@]} jobs submitted. Monitor with: squeue -u platawa"
