#!/bin/bash
# Submit dedicated scan_sae_global_stats.py GPU jobs for chromosomes that
# are missing or only have fused (under-sampled) stats.
#
# Missing entirely: chr1, chr15
# Fused only (20-71 chunks instead of thousands): chr2, chr3, chr4, chr5, chr6, chr7, chr8, chrY
#
# Each job takes ~20-40 min on a single GPU depending on chromosome size.

set -euo pipefail

PROJECT="/orcd/data/zhang_f/001/platawa/jan31_files"
FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna"
LOGDIR="${PROJECT}/logs"

CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr15 chrY)

echo "Submitting scan_sae_global_stats jobs for ${#CHROMS[@]} chromosomes..."
echo ""

for CHROM in "${CHROMS[@]}"; do
    JOB_NAME="sae_stats_${CHROM}"
    echo "  Submitting ${CHROM}..."

    sbatch \
        --job-name="${JOB_NAME}" \
        --partition=mit_preemptable \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=64G \
        --time=4:00:00 \
        --output="${LOGDIR}/${JOB_NAME}_%j.out" \
        --error="${LOGDIR}/${JOB_NAME}_%j.err" \
        --wrap="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python scan_sae_global_stats.py --fasta ${FASTA} --chrom ${CHROM} --output_dir results/"

    echo "    → submitted"
done

echo ""
echo "All ${#CHROMS[@]} jobs submitted."
echo "Monitor with: squeue -u platawa"
echo ""
echo "When all complete, re-run corrected aggregation:"
echo "  python scan_sae_global_stats.py --aggregate_corrected --all_human --results_dir results/"
