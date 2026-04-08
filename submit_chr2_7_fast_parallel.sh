#!/bin/bash
# submit_chr2_7_fast_parallel.sh
#
# Submit chr2-7 merges as 6 parallel SLURM jobs.
# Each job: merge_sae_shards_fast.py --skip-norm-stats + finish_merges.py
# Expected wall time per chromosome: ~20-30 minutes

set -e

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
N_SHARDS=36

mkdir -p "${LOGS}"

CHROMS="chr2 chr3 chr4 chr5 chr6 chr7"

echo "Submitting fast parallel merges for: ${CHROMS}"
echo ""

for CHROM in $CHROMS; do
    JOB_NAME="merge_fast_${CHROM}"
    SBATCH="${LOGS}/.${JOB_NAME}.sbatch"

    cat > "${SBATCH}" <<'SBATCH'
#!/bin/bash
#SBATCH -J merge_fast_CHROM
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o LOGS/merge_fast_CHROM_%j.out
#SBATCH -e LOGS/merge_fast_CHROM_%j.err

cd PROJECT
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[CHROM] Starting fast merge at $(date)"

# Step 1: Merge with ZIP_STORED, skip norm stats (15-20 min)
python merge_sae_shards_fast.py \
    --chrom CHROM \
    --n_shards 36 \
    --output_dir results/ \
    --skip-norm-stats

echo "[CHROM] Merge completed at $(date), adding global stats..."

# Step 2: Use finish_merges.py to add global stats from global_sae_stats.npz (instant)
python finish_merges.py \
    --chrom CHROM \
    --output_dir results/

echo "[CHROM] All done at $(date)"
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
echo "All 6 merge jobs submitted to mit_normal partition."
echo "Monitor with: squeue -u platawa"
echo "Logs at: ${LOGS}/merge_fast_chr*_*.out"
