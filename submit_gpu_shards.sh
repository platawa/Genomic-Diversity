#!/bin/bash
# submit_gpu_shards.sh
#
# Submit GPU SAE extraction jobs for incomplete shards.
# These shards were preempted mid-run and need fresh extraction.
#
# chr9:  shards 30, 31, 33, 35
# chr13: shard 35
# chr17: shards 33, 35

set -e

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs

mkdir -p "${LOGS}"

# Define jobs: "chrom:shard_idx"
JOBS="chr9:30 chr9:31 chr9:33 chr9:35 chr13:35 chr17:33 chr17:35"

echo "Submitting GPU SAE extraction jobs for incomplete shards"
echo ""

for JOB in $JOBS; do
    CHROM="${JOB%%:*}"
    SHARD_IDX="${JOB##*:}"
    JOB_NAME="gpu_${CHROM}_s${SHARD_IDX}"
    SBATCH="${LOGS}/.${JOB_NAME}.sbatch"

    cat > "${SBATCH}" <<SBEOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -t 04:00:00
#SBATCH -o ${LOGS}/${JOB_NAME}_%j.out
#SBATCH -e ${LOGS}/${JOB_NAME}_%j.err

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[\$(date)] Starting SAE extraction for ${CHROM} shard ${SHARD_IDX}/36"

python run_sae_fast.py \\
    --chrom ${CHROM} \\
    --auto \\
    --shard ${SHARD_IDX}/36 \\
    --extract_only \\
    --batch_size 32 \\
    --checkpoint_interval 500 \\
    --min_confidence 8.0 \\
    --output_dir results/

echo "[\$(date)] SAE extraction complete for ${CHROM} shard ${SHARD_IDX}/36"
SBEOF

    JID=$(sbatch "${SBATCH}" | awk '{print $NF}')
    echo "  ${CHROM} shard ${SHARD_IDX}/36: job ${JID}"
done

echo ""
echo "All 7 GPU jobs submitted to mit_preemptable (1 GPU, 100G RAM, 4h limit)"
echo "Expected runtime: 30-50 min per shard"
echo ""
echo "After completion:"
echo "  1. Re-merge chr9 and chr17 with: python merge_sae_shards_fast.py --chrom chrN --n_shards 36 --output_dir results/ --skip-norm-stats"
echo "  2. Finalize with: python finish_merges.py --chrom chrN --output_dir results/"
echo "  3. Submit latent analysis for chr9, chr13, chr17"
