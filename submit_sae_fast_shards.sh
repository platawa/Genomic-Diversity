#!/usr/bin/env bash
# submit_sae_fast_shards.sh
#
# Submit run_sae_fast.py in N_SHARDS parallel GPU jobs for one chromosome.
# Each shard processes a different contiguous slice of the qualifying regions.
#
# Usage:
#   bash submit_sae_fast_shards.sh <chrom> [n_shards]
#
# Examples:
#   bash submit_sae_fast_shards.sh chr1        # 4 shards (default)
#   bash submit_sae_fast_shards.sh chr1 8      # 8 shards
#
# After all shards complete, run:
#   python merge_sae_shards.py --chrom chr1 --output_dir results/
# ---------------------------------------------------------------------------

set -euo pipefail

CHROM="${1:?Usage: $0 <chrom> [n_shards]}"
N_SHARDS="${2:-4}"

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
FASTA=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna
GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf
LOGS=${PROJECT}/logs

mkdir -p "${LOGS}"

echo "Submitting ${N_SHARDS} shards for ${CHROM}..."

for (( IDX=0; IDX<N_SHARDS; IDX++ )); do
    SHARD="${IDX}/${N_SHARDS}"
    JOB_NAME="saef_${CHROM}_s${IDX}"
    SBATCH="${LOGS}/.${JOB_NAME}.sbatch"

    cat > "${SBATCH}" <<SBATCH
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 24:00:00
#SBATCH -o ${LOGS}/${JOB_NAME}_%j.out
#SBATCH -e ${LOGS}/${JOB_NAME}_%j.err
#SBATCH --requeue

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

python run_sae_fast.py \\
    --auto \\
    --chrom ${CHROM} \\
    --fasta ${FASTA} \\
    --gtf ${GTF} \\
    --output_dir results/ \\
    --shard ${SHARD} \\
    --min_confidence 8.0 \\
    --batch_size 64 \\
    --padding 200 \\
    --checkpoint_interval 500 \\
    --extract_only \\
    --skip_notebook
SBATCH

    JID=$(sbatch "${SBATCH}" | awk '{print $NF}')
    echo "  Shard ${IDX}/${N_SHARDS}: job ${JID} (${JOB_NAME})"
done

echo ""
echo "All ${N_SHARDS} shards submitted for ${CHROM}."
echo "Monitor with: squeue -u platawa -n saef_${CHROM}_s0,saef_${CHROM}_s1,..."
echo "After completion, merge with:"
echo "  python merge_sae_shards.py --chrom ${CHROM} --n_shards ${N_SHARDS} --output_dir results/"
