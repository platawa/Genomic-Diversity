#!/bin/bash
# finalize_feature_normalization.sh — Run after all 24 nuc_stats scan jobs complete.
#
# Steps:
#   1. Verify all 24 chromosomes have completed scans with nuc_mean/nuc_std
#   2. Aggregate per-chromosome stats into genome-wide nuc_mean/nuc_std
#   3. Z-score normalize feature matrices for all chromosomes
#
# Run on the cluster (CPU is fine, no GPU needed):
#   bash finalize_feature_normalization.sh
#
# Or submit via SLURM:
#   sbatch --partition=mit_normal --cpus-per-task=4 --mem=64G --time=2:00:00 \
#       --wrap="cd /orcd/data/zhang_f/001/platawa/jan31_files && bash finalize_feature_normalization.sh"

set -euo pipefail

PROJECT="/orcd/data/zhang_f/001/platawa/jan31_files"
cd "${PROJECT}"

# Ensure conda is available
module load miniforge/24.3.0-0 2>/dev/null || true
conda activate evo2_sep28 2>/dev/null || true

echo "============================================================"
echo "Feature Normalization Pipeline — Finalization"
echo "============================================================"
echo ""

# Step 1: Check that all 24 scans completed with nuc_mean/nuc_std
echo "Step 1: Verifying per-chromosome scan completions..."
MISSING=()
NO_NUC=()
CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10
        chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20
        chr21 chr22 chrX chrY)

for CHROM in "${CHROMS[@]}"; do
    STATS_DIR="results/${CHROM}/sae_global_stats"
    if [ ! -d "${STATS_DIR}" ]; then
        MISSING+=("${CHROM}")
        continue
    fi
    # Find latest completed run
    LATEST=$(find "${STATS_DIR}" -name COMPLETED -not -path "*_checkpoint*" -exec dirname {} \; 2>/dev/null | sort | tail -1)
    if [ -z "${LATEST}" ]; then
        MISSING+=("${CHROM}")
        continue
    fi
    # Check for nuc_mean in the npz
    HAS_NUC=$(python3 -c "
import numpy as np
d = np.load('${LATEST}/data/global_sae_stats.npz')
print('yes' if 'nuc_mean' in d else 'no')
" 2>/dev/null || echo "error")
    if [ "${HAS_NUC}" != "yes" ]; then
        NO_NUC+=("${CHROM}")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "  ERROR: Missing completed scans for: ${MISSING[*]}"
    echo "  Re-submit these chromosomes before running finalization."
    exit 1
fi

if [ ${#NO_NUC[@]} -gt 0 ]; then
    echo "  ERROR: Scans without nuc_mean/nuc_std: ${NO_NUC[*]}"
    echo "  These were run with the old script. Re-submit them."
    exit 1
fi

echo "  All 24 chromosomes have completed scans with nuc_mean/nuc_std."
echo ""

# Step 2: Aggregate into genome-wide stats
echo "Step 2: Aggregating genome-wide nuc_mean/nuc_std across all chromosomes..."
python scan_sae_global_stats.py \
    --aggregate_corrected \
    --all_human \
    --results_dir results/

echo ""
echo "  Aggregation complete."
echo ""

# Step 3: Normalize feature matrices
echo "Step 3: Z-score normalizing feature matrices for all chromosomes..."
python tools/normalize_selected_features.py \
    --all_human \
    --results_dir results/

echo ""
echo "============================================================"
echo "DONE. All feature matrices normalized with genome-wide z-scores."
echo ""
echo "Normalized files: results/<chrom>/sae/<merged>/data/feature_matrices_normalized.npz"
echo "Genome-wide stats: results/_genome_sae_stats/<latest>/data/genome_wide_sae_stats_corrected.npz"
echo "============================================================"
