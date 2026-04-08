#!/bin/bash
# submit_latent_from_shards.sh
#
# Submit latent analysis jobs reading directly from shard directories.
# Skips the merge step entirely — reads chunk files from each shard in order.
#
# Also submits:
#   - chr22 verification test (compare --from_shards vs merged NPZ)
#   - chr9, chr13, chr17 with SLURM dependency on their GPU extraction jobs

set -e

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs

mkdir -p "${LOGS}"

# ---- Step 1: chr22 verification test ----
echo "=== Submitting chr22 verification test ==="

cat > "${LOGS}/.verify_chr22.sbatch" <<'SBEOF'
#!/bin/bash
#SBATCH -J verify_chr22
#SBATCH -p pi_zhang_f
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH -t 04:00:00
#SBATCH -o LOGS/verify_chr22_%j.out
#SBATCH -e LOGS/verify_chr22_%j.err

cd PROJECT
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[$(date)] Running chr22 verification: --from_shards vs merged NPZ"

python -c "
import sys, os, zipfile, numpy as np
sys.path.insert(0, 'tools')
from analyze_sae_regions import load_and_pool_from_shards, load_and_pool_feature_matrices, setup_logging

logger = setup_logging('INFO')

# Load from shards
logger.info('=== Loading from SHARDS ===')
pooled_shards, n_shards, _ = load_and_pool_from_shards(
    'results', 'chr22', pool_method='max', n_shards=36, logger=logger
)

# Load from merged NPZ
logger.info('=== Loading from MERGED NPZ ===')
npz_path = 'results/chr22/sae/20260327_180121_all_conf8.0_merged36of36_fast/data/feature_matrices.npz'
pooled_npz, n_npz = load_and_pool_feature_matrices(npz_path, pool_method='max', logger=logger)

# Compare
logger.info('=== COMPARISON ===')
logger.info(f'Regions from shards: {n_shards}')
logger.info(f'Regions from NPZ:    {n_npz}')
logger.info(f'Count match: {n_shards == n_npz}')

if n_shards == n_npz:
    max_diff = np.max(np.abs(pooled_shards - pooled_npz))
    mean_diff = np.mean(np.abs(pooled_shards - pooled_npz))
    identical = np.array_equal(pooled_shards, pooled_npz)
    logger.info(f'Vectors identical: {identical}')
    logger.info(f'Max abs diff:  {max_diff}')
    logger.info(f'Mean abs diff: {mean_diff}')
    if identical:
        logger.info('VERIFICATION PASSED — outputs are byte-identical')
    elif max_diff < 1e-6:
        logger.info('VERIFICATION PASSED — outputs match within float32 tolerance')
    else:
        logger.info('VERIFICATION FAILED — outputs differ significantly')
else:
    logger.info('VERIFICATION FAILED — region counts differ')
"

echo "[$(date)] Verification complete"
SBEOF

sed -i "s|PROJECT|${PROJECT}|g" "${LOGS}/.verify_chr22.sbatch"
sed -i "s|LOGS|${LOGS}|g" "${LOGS}/.verify_chr22.sbatch"
VID=$(sbatch "${LOGS}/.verify_chr22.sbatch" | awk '{print $NF}')
echo "  verify_chr22: job ${VID}"
echo ""

# ---- Step 2: Latent analysis for 20 chromosomes with 36/36 COMPLETED shards ----
echo "=== Submitting --from_shards latent analysis ==="

CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr10 chr11 chr12 chr14 chr15 chr16 chr18 chr20 chr21 chr22 chrX"

for CHROM in $CHROMS; do
    JOB_NAME="latent_fs_${CHROM}"
    SBATCH="${LOGS}/.${JOB_NAME}.sbatch"

    cat > "${SBATCH}" <<SBEOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p pi_zhang_f
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH -t 12:00:00
#SBATCH -o ${LOGS}/${JOB_NAME}_%j.out
#SBATCH -e ${LOGS}/${JOB_NAME}_%j.err

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[\$(date)] Starting latent analysis (from_shards) for ${CHROM}"

python tools/analyze_sae_regions.py \\
    --from_shards \\
    --chrom ${CHROM} \\
    --results_dir results/ \\
    --embedding both \\
    --pool_method max

echo "[\$(date)] Latent analysis complete for ${CHROM}"
SBEOF

    JID=$(sbatch "${SBATCH}" | awk '{print $NF}')
    echo "  ${CHROM}: job ${JID}"
done

echo ""
echo "All latent_fs jobs submitted to pi_zhang_f (500G RAM, 8 CPUs, 12h limit)"
echo ""
echo "NOTE: chr9, chr13, chr17 not included — submit after GPU extraction completes"
echo "Monitor with: squeue -u platawa"
