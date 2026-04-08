#!/bin/bash
# Re-run scan_sae_global_stats.py for ALL 24 human chromosomes to compute
# per-nucleotide mean/std (nuc_mean, nuc_std) via Welford streaming.
#
# The existing scans only have chunk_max_mean/chunk_max_std — biased stats
# from per-chunk maxima. The updated script now also computes true
# per-nucleotide mean/var at every position, which is what we need for
# genome-wide z-score normalization.
#
# Each job takes ~2-4 hours on a single GPU depending on chromosome size.
# Checkpointing is enabled, so preempted jobs resume from where they left off.
#
# After all 24 complete, run the aggregation:
#   python scan_sae_global_stats.py --aggregate_corrected --all_human --results_dir results/
#
# Then normalize selected features:
#   python tools/normalize_selected_features.py --all_human --results_dir results/

set -euo pipefail

PROJECT="/orcd/data/zhang_f/001/platawa/jan31_files"
FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna"
LOGDIR="${PROJECT}/logs"
mkdir -p "${LOGDIR}"

CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10
        chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20
        chr21 chr22 chrX chrY)

echo "Submitting scan_sae_global_stats (with nuc_mean/nuc_std) for all ${#CHROMS[@]} chromosomes..."
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
        --time=6:00:00 \
        --output="${LOGDIR}/${JOB_NAME}_%j.out" \
        --error="${LOGDIR}/${JOB_NAME}_%j.err" \
        --wrap="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python scan_sae_global_stats.py --fasta ${FASTA} --chrom ${CHROM} --output_dir results/"

    echo "    -> submitted"
done

echo ""
echo "All ${#CHROMS[@]} jobs submitted."
echo "Monitor with: squeue -u platawa"
echo ""
echo "After ALL complete, run:"
echo "  1. Aggregate: python scan_sae_global_stats.py --aggregate_corrected --all_human --results_dir results/"
echo "  2. Normalize: python tools/normalize_selected_features.py --all_human --results_dir results/"
