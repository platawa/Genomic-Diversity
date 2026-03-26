#!/bin/bash
# submit_chr15_fast_merge.sh
# Submit fast (uncompressed) chr15 SAE merge using ZIP_STORED
# Complements the slow compressed merge (job 10987923) already running

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs

echo "=========================================="
echo "Chr15 SAE Fast Merge (Uncompressed)"
echo "=========================================="
echo ""
echo "Strategy:"
echo "  - Original job 10987923 uses ZIP_DEFLATED (very slow, ~2+ hours)"
echo "  - This job uses ZIP_STORED (no compression, ~15-20 minutes)"
echo "  - Output: larger file (~30-40GB) but finishes much faster"
echo "  - Unique directory: *_merged36of36_fast (no naming conflicts)"
echo ""
echo "Expected timeline:"
echo "  - Feature matrix merge: ~15 minutes"
echo "  - Normalization stats: ~10-15 minutes"
echo "  - Total: ~25-30 minutes"
echo ""

JOB_ID=$(sbatch --parsable \
  --job-name="merge_chr15_fast" \
  --partition=pi_zhang_f \
  --cpus-per-task=8 \
  --mem=256G \
  --time=1:00:00 \
  --output="${LOGS}/merge_chr15_fast_%j.out" \
  --error="${LOGS}/merge_chr15_fast_%j.err" \
  --wrap="
set -eo pipefail

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo 'Starting fast merge at '$(date)
python merge_sae_shards_fast.py --chrom chr15 --n_shards 36 --output_dir results/
echo 'Completed at '$(date)
")

echo "Job submitted: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  tail -f ${LOGS}/merge_chr15_fast_${JOB_ID}.out"
echo ""
echo "Compare the two runs:"
echo "  ls -lh results/chr15/sae/ | grep merged36"
echo ""
