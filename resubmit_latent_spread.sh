#!/bin/bash
# resubmit_latent_spread.sh
# Resubmit pending latent jobs spread across 3 partitions, 200G RAM, 12h

set -e
PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
GSTATS=${PROJECT}/results/_genome_sae_stats/20260323_193732_genome_minmax_22chroms/data/genome_wide_sae_stats.npz

submit_fs() {
    local CHROM=$1 PART=$2
    local JOB_NAME="latent_fs_${CHROM}"
    local SBATCH="${LOGS}/.${JOB_NAME}_r.sbatch"
    cat > "${SBATCH}" <<SBEOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p ${PART}
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -o ${LOGS}/${JOB_NAME}_%j.out
#SBATCH -e ${LOGS}/${JOB_NAME}_%j.err

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28
echo "[\$(date)] Latent (from_shards) ${CHROM} on ${PART}"
python tools/analyze_sae_regions.py --from_shards --chrom ${CHROM} --results_dir results/ --embedding both --pool_method max
echo "[\$(date)] Done ${CHROM}"
SBEOF
    JID=$(sbatch "${SBATCH}" | awk '{print $NF}')
    echo "  ${CHROM}: job ${JID} (${PART})"
}

submit_npz() {
    local CHROM=$1 PART=$2 INPUT=$3
    local JOB_NAME="latent_npz_${CHROM}"
    local SBATCH="${LOGS}/.${JOB_NAME}_r.sbatch"
    cat > "${SBATCH}" <<SBEOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p ${PART}
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -o ${LOGS}/${JOB_NAME}_%j.out
#SBATCH -e ${LOGS}/${JOB_NAME}_%j.err

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28
echo "[\$(date)] Latent (merged NPZ) ${CHROM} on ${PART}"
python tools/analyze_sae_regions.py --input_dir ${INPUT} --embedding both --pool_method max
echo "[\$(date)] Done ${CHROM}"
SBEOF
    JID=$(sbatch "${SBATCH}" | awk '{print $NF}')
    echo "  ${CHROM}: job ${JID} (${PART}, merged NPZ)"
}

submit_norm() {
    local CHROM=$1 PART=$2
    local JOB_NAME="latent_norm_${CHROM}"
    local SBATCH="${LOGS}/.${JOB_NAME}_r.sbatch"
    cat > "${SBATCH}" <<SBEOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p ${PART}
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -o ${LOGS}/${JOB_NAME}_%j.out
#SBATCH -e ${LOGS}/${JOB_NAME}_%j.err

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28
echo "[\$(date)] Latent (normalized, from_shards) ${CHROM} on ${PART}"
python tools/analyze_sae_regions.py --from_shards --chrom ${CHROM} --results_dir results/ --embedding both --pool_method max --global_stats ${GSTATS}
echo "[\$(date)] Done ${CHROM}"
SBEOF
    JID=$(sbatch "${SBATCH}" | awk '{print $NF}')
    echo "  ${CHROM}: job ${JID} (${PART}, normalized)"
}

echo "=== from_shards on mit_normal (10 jobs) ==="
for C in chr4 chr5 chr6 chr7 chr8 chr10 chr11 chr12 chr14 chr15; do
    submit_fs $C mit_normal
done

echo ""
echo "=== from_shards on ou_bcs_low (8 jobs) ==="
for C in chr16 chr18 chr20 chr21 chr22 chrX; do
    submit_fs $C ou_bcs_low
done

echo ""
echo "=== merged NPZ comparison on ou_bcs_normal (3 jobs) ==="
submit_npz chr1 ou_bcs_normal "${PROJECT}/results/chr1/sae/20260328_004018_all_conf8.0_merged36of36_fast"
submit_npz chr2 ou_bcs_normal "${PROJECT}/results/chr2/sae/20260328_004019_all_conf8.0_merged36of36_fast"
submit_npz chr3 ou_bcs_normal "${PROJECT}/results/chr3/sae/20260327_163244_all_conf8.0_merged36of36_fast"

echo ""
echo "=== normalized on ou_bcs_low (2 jobs) ==="
submit_norm chr22 ou_bcs_low
submit_norm chr21 ou_bcs_low

echo ""
echo "Total: 23 jobs across mit_normal / ou_bcs_low / ou_bcs_normal (200G, 12h each)"
