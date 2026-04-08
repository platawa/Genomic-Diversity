#!/bin/bash
# Submit human chromosome analysis jobs dependent on chr15 merge (10987923)

CHR15_JOB=10987923
PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs

echo "Submitting human chromosome analysis jobs (dependent on chr15 merge 10987923)..."
echo ""

CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22)

for CHR in "${CHROMS[@]}"; do
    JOB_ID=$(sbatch --parsable \
      --job-name="analyze_${CHR}" \
      --partition=pi_zhang_f \
      --cpus-per-task=8 \
      --mem=128G \
      --time=4:00:00 \
      --dependency=afterok:${CHR15_JOB} \
      --output="${LOGS}/analyze_${CHR}_%j.out" \
      --error="${LOGS}/analyze_${CHR}_%j.err" \
      --wrap="
set -eo pipefail
cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

# Find latest merge directory for this chromosome
MERGE_DIR=\$(ls -td results/${CHR}/sae/*merged*/ 2>/dev/null | head -1)
if [ -z \"\$MERGE_DIR\" ]; then
  echo \"ERROR: No merge directory found for ${CHR}\"
  exit 1
fi

echo \"Analyzing \${MERGE_DIR}...\"
python tools/analyze_sae_regions.py \\
  --input_dir \"\${MERGE_DIR}\" \\
  --embedding both \\
  --leiden_resolution 1.0
")

    echo "  ${CHR}: Job ${JOB_ID}"
done

echo ""
echo "Submitted ${#CHROMS[@]} per-chromosome analysis jobs"
