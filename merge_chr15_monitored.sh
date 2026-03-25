#!/bin/bash
# merge_chr15_monitored.sh
# Merge chr15 SAE shards with progress monitoring and time estimation

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
CHROM=chr15
N_SHARDS=36

echo "=========================================="
echo "Chr15 SAE Merge (Monitored)"
echo "=========================================="
echo ""
echo "Setup:"
echo "  Chromosome: $CHROM"
echo "  Shards: $N_SHARDS"
echo "  Time limit: 24 hours"
echo "  Memory: 256GB"
echo ""

# Calculate estimated time based on shard sizes
echo "Pre-merge analysis:"
TOTAL_SIZE=$(du -sh ${PROJECT}/results/${CHROM}/sae/*_all_conf8.0_shard*of${N_SHARDS}/data/_chunk* 2>/dev/null | awk '{s+=$1} END {print s}')
echo "  Total chunk data size: $(du -sh ${PROJECT}/results/${CHROM}/sae/*_all_conf8.0_shard*of${N_SHARDS} 2>/dev/null | tail -1 | awk '{print $1}')"
echo "  Estimated time: 4-8 hours (based on previous attempts)"
echo ""
echo "Starting merge at $(date)..."
echo ""

# Submit with 24h time limit and progress monitoring wrapper
JOB_ID=$(sbatch --parsable \
  --job-name="merge_chr15_monitored" \
  --partition=pi_zhang_f \
  --cpus-per-task=8 \
  --mem=256G \
  --time=24:00:00 \
  --output="${LOGS}/merge_chr15_monitored_%j.out" \
  --error="${LOGS}/merge_chr15_monitored_%j.err" \
  --wrap="
set -eo pipefail

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

# Start time
START=\$(date +%s)
echo '[Progress Monitor] Start time: '\$(date)
echo '[Progress Monitor] PID: \$\$'

# Run merge with background monitoring
python merge_sae_shards.py --chrom ${CHROM} --n_shards ${N_SHARDS} --output_dir results/ &
MERGE_PID=\$!

# Monitor progress every 10 seconds
while kill -0 \$MERGE_PID 2>/dev/null; do
  ELAPSED=\$(((\$(date +%s) - START) / 60))
  MERGED_SIZE=\$(du -sh results/${CHROM}/sae/*_all_conf8.0_merged*/data/feature_matrices.npz 2>/dev/null | tail -1 | awk '{print \$1}' || echo 'waiting...')
  echo \"[Progress] \${ELAPSED}min elapsed - merged file: \$MERGED_SIZE - \$(date '+%H:%M:%S')\"
  sleep 10
done

# Wait for completion
wait \$MERGE_PID
MERGE_EXIT=\$?

END=\$(date +%s)
DURATION=\$(((END - START) / 60))
echo '[Progress Monitor] Completed in '\$DURATION' minutes'
echo '[Progress Monitor] Exit code: '\$MERGE_EXIT

exit \$MERGE_EXIT
")

echo "Job submitted: $JOB_ID"
echo ""
echo "Monitor progress with:"
echo "  tail -f ${LOGS}/merge_chr15_monitored_${JOB_ID}.out"
echo ""
