#!/bin/bash
#SBATCH -J verify_normalize
#SBATCH -p pi_zhang_f
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /orcd/data/zhang_f/001/platawa/jan31_files/logs/verify_normalize_%j.out
#SBATCH -e /orcd/data/zhang_f/001/platawa/jan31_files/logs/verify_normalize_%j.err

set -eo pipefail
cd /orcd/data/zhang_f/001/platawa/jan31_files
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "============================================"
echo "Post-Merge Verification & Normalization"
echo "Started: $(date)"
echo "============================================"

CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr20 chr21 chr22 chrX chrY"
# chr19 excluded — GPU shards empty

###############################################################################
# STEP 1: Verify all chromosomes have valid COMPLETED merges
###############################################################################
echo ""
echo "========== STEP 1: Verifying merges =========="

PASS=0
FAIL=0
FAIL_LIST=""

for chrom in $CHROMS; do
    # Find latest COMPLETED merge dir (newest first)
    MERGE_DIR=""
    for d in $(ls -dr results/$chrom/sae/*merged* 2>/dev/null); do
        if [ -f "$d/COMPLETED" ]; then
            MERGE_DIR="$d"
            break
        fi
    done

    if [ -z "$MERGE_DIR" ]; then
        echo "FAIL  $chrom: no COMPLETED merge found"
        FAIL=$((FAIL + 1))
        FAIL_LIST="$FAIL_LIST $chrom"
        continue
    fi

    NPZ="$MERGE_DIR/data/feature_matrices.npz"
    NORM="$MERGE_DIR/data/feature_norm_stats.npz"

    # Check feature_matrices.npz exists and is valid zip
    if [ ! -f "$NPZ" ]; then
        echo "FAIL  $chrom: feature_matrices.npz missing in $MERGE_DIR"
        FAIL=$((FAIL + 1))
        FAIL_LIST="$FAIL_LIST $chrom"
        continue
    fi

    # Validate zip and count regions
    REGION_COUNT=$(python3 -c "
import zipfile, sys
try:
    zf = zipfile.ZipFile('$NPZ', 'r')
    regions = [n for n in zf.namelist() if n.startswith('region_')]
    print(len(regions))
    zf.close()
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    print(-1)
" 2>&1)

    if [ "$REGION_COUNT" = "-1" ] || echo "$REGION_COUNT" | grep -q "ERROR"; then
        echo "FAIL  $chrom: feature_matrices.npz is corrupted — $REGION_COUNT"
        FAIL=$((FAIL + 1))
        FAIL_LIST="$FAIL_LIST $chrom"
        continue
    fi

    # Check norm stats
    if [ ! -f "$NORM" ]; then
        echo "WARN  $chrom: feature_norm_stats.npz missing (will recompute)"
    fi

    echo "PASS  $chrom: $REGION_COUNT regions — $(basename $MERGE_DIR)"
    PASS=$((PASS + 1))
done

echo ""
echo "---------- Verification Summary ----------"
echo "PASS: $PASS / $(echo $CHROMS | wc -w | tr -d ' ')"
echo "FAIL: $FAIL"
if [ -n "$FAIL_LIST" ]; then
    echo "Failed chromosomes:$FAIL_LIST"
fi

# If too many failures, abort
if [ $PASS -lt 18 ]; then
    echo "ERROR: Only $PASS chromosomes passed. Need at least 18. Aborting."
    exit 1
fi

###############################################################################
# STEP 2: Compute genome-wide global stats
###############################################################################
echo ""
echo "========== STEP 2: Computing genome-wide global stats =========="
echo "Running aggregate_genome_sae_stats.py..."

python tools/aggregate_genome_sae_stats.py \
    --all_human \
    --results_dir results/ \
    --min_chromosomes 18 \
    --force

echo "Global stats complete."

###############################################################################
# STEP 3: Z-score normalization per chromosome
###############################################################################
echo ""
echo "========== STEP 3: Applying z-score normalization =========="

NORM_PASS=0
NORM_FAIL=0

for chrom in $CHROMS; do
    echo "Normalizing $chrom..."
    python tools/normalize_sae_features.py \
        --chrom $chrom \
        --results_dir results/ \
        --auto \
        --method zscore \
        2>&1 || {
            echo "  WARN: normalization failed for $chrom (may lack maxpooled_vectors.npy)"
            NORM_FAIL=$((NORM_FAIL + 1))
            continue
        }
    NORM_PASS=$((NORM_PASS + 1))
done

echo ""
echo "---------- Normalization Summary ----------"
echo "Normalized: $NORM_PASS chromosomes"
echo "Failed: $NORM_FAIL chromosomes"

###############################################################################
# DONE
###############################################################################
echo ""
echo "============================================"
echo "Pipeline complete: $(date)"
echo "  Verified: $PASS chromosomes"
echo "  Normalized: $NORM_PASS chromosomes"
echo "============================================"
