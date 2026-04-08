#!/bin/bash
# Wave A merges — all 12 chromosomes that don't need GPU
# Pattern: file-based sbatch to avoid heredoc issues

set -e

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
mkdir -p ${LOGS}

declare -A AFTER_DATES=(
  [chr3]="20260323_000000"
  [chr5]="20260322_000000"
  [chr7]="20260322_000000"
  [chr12]="20260323_000000"
  [chr13]="20260323_000000"
  [chr14]="20260323_000000"
  [chr16]="20260323_000000"
  [chr20]="20260323_000000"
  [chr21]="20260323_000000"
  [chr22]="20260323_000000"
  [chrX]="20260323_000000"
  [chrY]="20260323_000000"
)

echo "Submitting Wave A re-merges (12 chromosomes)..."
echo ""

for CHROM in chr3 chr5 chr7 chr12 chr13 chr14 chr16 chr20 chr21 chr22 chrX chrY; do
  AFTER=${AFTER_DATES[$CHROM]}
  SBATCH_FILE=${LOGS}/.wave_a_${CHROM}.sbatch
  
  cat > "${SBATCH_FILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J merge_wave_a_${CHROM}
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o ${LOGS}/wave_a_${CHROM}_%j.out
#SBATCH -e ${LOGS}/wave_a_${CHROM}_%j.err

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[$(date)] Starting Wave A merge for ${CHROM}"

python merge_sae_shards_fast.py \
  --chrom ${CHROM} \
  --n_shards 36 \
  --output_dir results/ \
  --skip-norm-stats \
  --after ${AFTER}

echo "[$(date)] Completed merge for ${CHROM}, running finish_merges..."

python finish_merges.py \
  --chrom ${CHROM} \
  --output_dir results/

echo "[$(date)] All done for ${CHROM}"
SBATCH_EOF

  JID=$(sbatch "${SBATCH_FILE}" | awk '{print $NF}')
  echo "  ${CHROM}: job ${JID}"
done

echo ""
echo "Wave A submitted. Monitor with: squeue -u platawa"
