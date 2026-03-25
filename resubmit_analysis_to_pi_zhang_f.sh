#!/bin/bash
# Resubmit 23 analysis jobs (one per chromosome) to pi_zhang_f partition
# The script analyze_sae_regions.py is now available in tools/

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs

CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY"

COUNT=0
echo "Submitting analysis jobs to pi_zhang_f..."
echo ""

for CHROM in $CHROMS; do
    COUNT=$((COUNT + 1))
    JOB_NAME="analysis_${CHROM}_hp"

    echo "[$COUNT/24] $CHROM: $JOB_NAME..."

    JOBID=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --partition=pi_zhang_f \
        --cpus-per-task=4 \
        --mem=32G \
        --time=4:00:00 \
        --output="${LOGS}/${JOB_NAME}_%j.out" \
        --error="${LOGS}/${JOB_NAME}_%j.err" \
        --wrap="module load miniforge/24.3.0-0 && conda activate evo2_sep28 && cd ${PROJECT} && python tools/analyze_sae_regions.py --chrom ${CHROM} --auto --output_dir results/")

    echo "  Job ID: $JOBID"
done

echo ""
echo "Done! Submitted $COUNT analysis jobs to pi_zhang_f"
