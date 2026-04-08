#!/bin/bash
# submit_latent_analysis.sh
#
# Submit latent analysis jobs for chromosomes with completed fast merges.
# Uses streaming loader (no memory blowup from np.load caching).

set -e

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs

mkdir -p "${LOGS}"

# 9 chromosomes with completed 36/36 fast merges
CHROMS="chr1 chr2 chr3 chr5 chr6 chr7 chr10 chr15 chr22"

# Map each chrom to its best merge dir
declare -A MERGE_DIRS
MERGE_DIRS[chr1]="20260328_004018_all_conf8.0_merged36of36_fast"
MERGE_DIRS[chr2]="20260328_004019_all_conf8.0_merged36of36_fast"
MERGE_DIRS[chr3]="20260327_163244_all_conf8.0_merged36of36_fast"
MERGE_DIRS[chr5]="20260327_151234_all_conf8.0_merged36of36_fast"
MERGE_DIRS[chr6]="20260328_003943_all_conf8.0_merged36of36_fast"
MERGE_DIRS[chr7]="20260327_154021_all_conf8.0_merged36of36_fast"
MERGE_DIRS[chr10]="20260328_003941_all_conf8.0_merged36of36_fast"
MERGE_DIRS[chr15]="20260325_132132_all_conf8.0_merged36of36"
MERGE_DIRS[chr22]="20260327_180121_all_conf8.0_merged36of36_fast"

echo "Submitting latent analysis jobs for completed chromosomes"
echo ""

for CHROM in $CHROMS; do
    MERGE_DIR="${MERGE_DIRS[$CHROM]}"
    INPUT_DIR="${PROJECT}/results/${CHROM}/sae/${MERGE_DIR}"
    JOB_NAME="latent_${CHROM}"
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

echo "[\$(date)] Starting latent analysis for ${CHROM}"
echo "Input: ${INPUT_DIR}"

python tools/analyze_sae_regions.py \\
    --input_dir ${INPUT_DIR} \\
    --embedding both \\
    --pool_method max

echo "[\$(date)] Latent analysis complete for ${CHROM}"
SBEOF

    JID=$(sbatch "${SBATCH}" | awk '{print $NF}')
    echo "  ${CHROM}: job ${JID} (input: ${MERGE_DIR})"
done

echo ""
echo "All latent analysis jobs submitted to pi_zhang_f (500G RAM, 8 CPUs, 12h limit)"
echo "Monitor with: squeue -u platawa"
