#!/bin/bash
# Submit confidence 0.0 merges for completed chromosomes (can run in parallel)

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs

echo "Submitting confidence 0.0 merges for complete chromosomes..."
echo ""

# Chromosomes with all 8 shards complete
COMPLETE_CHROMS=(chrY chr21 chr13)

for CHR in "${COMPLETE_CHROMS[@]}"; do
    JOB_ID=$(sbatch --parsable \
      --job-name="merge_${CHR}_conf0" \
      --partition=pi_zhang_f \
      --cpus-per-task=8 \
      --mem=256G \
      --time=6:00:00 \
      --output="${LOGS}/merge_${CHR}_conf0_%j.out" \
      --error="${LOGS}/merge_${CHR}_conf0_%j.err" \
      --wrap="
set -eo pipefail
cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

python merge_sae_shards.py --chrom ${CHR} --n_shards 8 --output_dir results/
")

    echo "  ${CHR}: Job ${JOB_ID}"
done

echo ""
echo "Submitted 3 merges for confidence 0.0 complete chromosomes"
