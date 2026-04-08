#!/bin/bash
# submit_remerge_stale.sh
#
# Re-merge chromosomes that have all 36 shards COMPLETED but stale/partial merges.
# Category D: chr8, chr12, chr14, chr16, chr18, chr20, chr21, chrX (stale partial fast merges)
# Category C: chr11, chr15 (only old compressed merges, need fast version)
# Also: chr4 (has duplicate shard in merge, needs clean re-merge)
#
# Each job: merge_sae_shards_fast.py --skip-norm-stats + finish_merges.py

set -e

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
N_SHARDS=36

mkdir -p "${LOGS}"

CHROMS="chr8 chr12 chr13 chr14 chr16 chr18 chr20 chr21 chrX"

echo "Submitting re-merge jobs for stale/partial chromosomes: ${CHROMS}"
echo ""

for CHROM in $CHROMS; do
    JOB_NAME="remerge_${CHROM}"
    SBATCH="${LOGS}/.${JOB_NAME}.sbatch"

    cat > "${SBATCH}" <<'SBATCH'
#!/bin/bash
#SBATCH -J remerge_CHROM
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o LOGS/remerge_CHROM_%j.out
#SBATCH -e LOGS/remerge_CHROM_%j.err

cd PROJECT
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[$(date)] Starting fast re-merge for CHROM"

# Step 1: Merge with ZIP_STORED, skip norm stats
python merge_sae_shards_fast.py \
    --chrom CHROM \
    --n_shards 36 \
    --output_dir results/ \
    --include-partial \
    --skip-norm-stats

echo "[$(date)] Merge done for CHROM, adding norm stats..."

# Step 2: Add norm stats and write COMPLETED
python finish_merges.py \
    --chrom CHROM \
    --output_dir results/

echo "[$(date)] CHROM complete"
SBATCH

    # Replace placeholders
    sed -i "s|CHROM|${CHROM}|g" "${SBATCH}"
    sed -i "s|PROJECT|${PROJECT}|g" "${SBATCH}"
    sed -i "s|LOGS|${LOGS}|g" "${SBATCH}"

    # Submit job
    JID=$(sbatch "${SBATCH}" | awk '{print $NF}')
    echo "  ${CHROM}: job ${JID}"
done

echo ""
echo "All ${#CHROMS[@]} re-merge jobs submitted to mit_normal partition."
echo "Monitor with: squeue -u platawa"
echo "Expected wall time: 30 min - 4 hours per chromosome"
echo ""
echo "After completion, verify with:"
echo "  python check_conf8_status.py --output_dir results/"
