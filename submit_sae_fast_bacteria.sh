#!/usr/bin/env bash
# submit_sae_fast_bacteria.sh
#
# Submit run_sae_fast.py in parallel GPU shards for bacterial genomes.
# Adapted from submit_sae_fast_shards.sh but with:
#   - Correct FASTA/GTF paths per organism
#   - Lower min_confidence (0.0 = ALL detected regions)
#   - ou_bcs_low partition with H100 GPUs
#   - Smaller batch_size (bacterial regions are denser)
#
# Usage:
#   bash submit_sae_fast_bacteria.sh ecoli    [n_shards]  # default 4 shards
#   bash submit_sae_fast_bacteria.sh bacillus [n_shards]
#   bash submit_sae_fast_bacteria.sh all      [n_shards]  # both organisms
#
# After all shards complete, merge with:
#   python merge_sae_shards.py --chrom NC_000913.3 --n_shards 4 --output_dir results/
#   python merge_sae_shards.py --chrom NC_000964.3 --n_shards 4 --output_dir results/
# ---------------------------------------------------------------------------

set -euo pipefail

ORGANISM="${1:?Usage: $0 <ecoli|bacillus|all> [n_shards] [min_confidence]}"
N_SHARDS="${2:-4}"
MIN_CONF="${3:-0.0}"

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
PARTITION="ou_bcs_low"
GPU_TYPE="h100"

# --- organism config ---
declare -A CHROMS FASTAS GTFS
CHROMS[ecoli]="NC_000913.3"
FASTAS[ecoli]="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna"
GTFS[ecoli]="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf"

CHROMS[bacillus]="NC_000964.3"
FASTAS[bacillus]="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/GCF_000009045.1_ASM904v1_genomic.fna"
GTFS[bacillus]="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf"

mkdir -p "${LOGS}"

submit_organism() {
    local ORG="$1"
    local CHROM="${CHROMS[$ORG]}"
    local FASTA="${FASTAS[$ORG]}"
    local GTF="${GTFS[$ORG]}"

    echo "=========================================="
    echo "Submitting ${N_SHARDS} shards for ${ORG} (${CHROM})"
    echo "  min_confidence: ${MIN_CONF} (all regions)"
    echo "  partition: ${PARTITION} (${GPU_TYPE})"
    echo "=========================================="

    for (( IDX=0; IDX<N_SHARDS; IDX++ )); do
        SHARD="${IDX}/${N_SHARDS}"
        JOB_NAME="saef_${ORG}_s${IDX}"
        SBATCH="${LOGS}/.${JOB_NAME}.sbatch"

        cat > "${SBATCH}" <<SBATCH
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p ${PARTITION}
#SBATCH --gres=gpu:${GPU_TYPE}:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH -o ${LOGS}/${JOB_NAME}_%j.out
#SBATCH -e ${LOGS}/${JOB_NAME}_%j.err
#SBATCH --requeue

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "SAE ${ORG} shard ${IDX}/${N_SHARDS} started at \$(date) on \$(hostname)"

python run_sae_fast.py \\
    --auto \\
    --chrom ${CHROM} \\
    --fasta ${FASTA} \\
    --gtf ${GTF} \\
    --output_dir results/ \\
    --shard ${SHARD} \\
    --min_confidence ${MIN_CONF} \\
    --batch_size 64 \\
    --padding 200 \\
    --checkpoint_interval 500 \\
    --extract_only \\
    --skip_notebook

echo "SAE ${ORG} shard ${IDX}/${N_SHARDS} finished at \$(date)"
SBATCH

        JID=$(sbatch "${SBATCH}" | awk '{print $NF}')
        echo "  Shard ${IDX}/${N_SHARDS}: job ${JID} (${JOB_NAME})"
    done

    echo ""
    echo "After completion, merge with:"
    echo "  python merge_sae_shards.py --chrom ${CHROM} --n_shards ${N_SHARDS} --output_dir results/"
    echo ""
}

# --- dispatch ---
if [ "${ORGANISM}" = "all" ]; then
    submit_organism ecoli
    submit_organism bacillus
elif [ "${ORGANISM}" = "ecoli" ] || [ "${ORGANISM}" = "bacillus" ]; then
    submit_organism "${ORGANISM}"
else
    echo "ERROR: Unknown organism '${ORGANISM}'. Use ecoli, bacillus, or all."
    exit 1
fi
