#!/bin/bash
# Wave B merges — 7 chromosomes that depend on GPU jobs completing
# Run this after all GPU shard jobs (11063315–11063504) have COMPLETED.
#
# Verify GPU jobs done first:
#   squeue -u platawa -h -o '%i' | grep -qE '1106(3315|3318|3320|3323|3325|3327|3329|3331)' && echo 'STILL RUNNING'

set -e

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
mkdir -p ${LOGS}

# --after cutoff dates per chromosome
# chr4/chr6: include 20260322 shards that have COMPLETED
# chr19: use 20260326 to pick up only new 20260327 GPU run dirs (old 20260322 dirs were empty)
declare -A AFTER_DATES=(
  [chr1]="20260323_000000"
  [chr2]="20260323_000000"
  [chr4]="20260322_000000"
  [chr6]="20260322_000000"
  [chr8]="20260323_000000"
  [chr18]="20260323_000000"
  [chr19]="20260326_000000"
)

echo "Submitting Wave B re-merges (7 chromosomes)..."
echo ""

for CHROM in chr1 chr2 chr4 chr6 chr8 chr18 chr19; do
  AFTER=${AFTER_DATES[$CHROM]}
  SBATCH_FILE=${LOGS}/.wave_b_${CHROM}.sbatch

  cat > "${SBATCH_FILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J merge_wave_b_${CHROM}
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o ${LOGS}/wave_b_${CHROM}_%j.out
#SBATCH -e ${LOGS}/wave_b_${CHROM}_%j.err

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[$(date)] Starting Wave B merge for ${CHROM}"

python merge_sae_shards_fast.py \
  --chrom ${CHROM} \
  --n_shards 36 \
  --output_dir results/ \
  --skip-norm-stats \
  --include-partial \
  --after ${AFTER}

echo "[$(date)] Merge done for ${CHROM}, running finish_merges..."

python finish_merges.py \
  --chrom ${CHROM} \
  --output_dir results/

echo "[$(date)] All done for ${CHROM}"
SBATCH_EOF

  JID=$(sbatch "${SBATCH_FILE}" | awk '{print $NF}')
  echo "  ${CHROM}: job ${JID}"
done

echo ""
echo "Wave B submitted. Monitor with: squeue -u platawa"
echo ""
echo "After Wave B completes, run final normalization:"
echo "  python finish_merges.py --all_human --output_dir results/"
