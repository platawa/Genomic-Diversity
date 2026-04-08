#!/bin/bash
# Submit chr1-14, 16-22 merges with dependency on chr15 (10987923)

CHR15_JOB=10987923
CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr16 chr17 chr18 chr19 chr20 chr21 chr22)
PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs

echo "Submitting remaining chromosome merges with dependency on chr15 (job ${CHR15_JOB})"
echo ""

for CHR in "${CHROMS[@]}"; do
    JOB_ID=$(sbatch --parsable \
      --job-name="merge_${CHR}" \
      --partition=pi_zhang_f \
      --cpus-per-task=8 \
      --mem=256G \
      --time=6:00:00 \
      --dependency=afterok:${CHR15_JOB} \
      --output="${LOGS}/merge_${CHR}_%j.out" \
      --error="${LOGS}/merge_${CHR}_%j.err" \
      --wrap="
set -eo pipefail
cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28
python merge_sae_shards.py --chrom ${CHR} --n_shards 36 --output_dir results/
")

    echo "  ${CHR}: Job ${JOB_ID}"
done

echo ""
echo "All merges queued with dependency on chr15 (10987923)"
