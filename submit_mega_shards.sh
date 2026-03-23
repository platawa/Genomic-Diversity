#!/usr/bin/env bash
# submit_mega_shards.sh
#
# Submit mega-jobs across ou_bcs_normal (H100, non-preemptable),
# mit_preemptable, AND ou_bcs_low simultaneously to maximize GPU parallelism.
#
# Each job loads Evo2 ONCE then processes all 24 chromosomes for one shard index:
#   GPU 0  job: chrY/s0  → chrX/s0  → chr21/s0  → ... → chr1/s0
#   ...
#   GPU 35 job: chrY/s35 → chrX/s35 → chr21/s35 → ... → chr1/s35
#
# Partition split (given QOS limits):
#   Shards 0-15  → ou_bcs_normal  (16 GPU limit, H100, non-preemptable)
#   Shards 16-19 → mit_preemptable (4 GPU limit, various, preemptable+requeue)
#   Shards 20-35 → ou_bcs_low     (64 GPU limit, preemptable+requeue)
#
# Usage:  bash submit_mega_shards.sh [n_shards] [max_regions_per_chrom]
#   n_shards:              default 36  (16 normal + 4 preemptable + 16 bcs_low)
#   max_regions_per_chrom: default 0   (unlimited — all qualifying regions)
# ---------------------------------------------------------------------------
set -euo pipefail

N_SHARDS="${1:-36}"
MAX_REGIONS="${2:-0}"

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
FASTA=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna
GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf
LOGS=${PROJECT}/logs

# Smallest chromosomes first → early results appear fastest
CHROMS="chrY chrX chr21 chr22 chr20 chr19 chr16 chr15 chr14 chr13 chr12 chr11 chr10 chr9 chr8 chr7 chr6 chr5 chr4 chr18 chr17 chr3 chr2 chr1"

mkdir -p "${LOGS}"

echo "Submitting ${N_SHARDS} mega-jobs across ou_bcs_normal + mit_preemptable + ou_bcs_low"
echo "  max_regions per chrom: ${MAX_REGIONS:-unlimited (all conf>=8.0 regions)}"
echo "  shards 0-15  → ou_bcs_normal  (H100, non-preemptable)"
echo "  shards 16-19 → mit_preemptable (various GPU, preemptable+requeue)"
echo "  shards 20-35 → ou_bcs_low     (various GPU, preemptable+requeue)"
echo ""

for (( IDX=0; IDX<N_SHARDS; IDX++ )); do
    JOB_NAME="sae_mega_s${IDX}"
    SBATCH="${LOGS}/.${JOB_NAME}.sbatch"

    # Partition assignment
    if (( IDX < 16 )); then
        PARTITION="ou_bcs_normal"
        GPU_SPEC="gpu:h100:1"
        REQUEUE=""
        WALLTIME="24:00:00"
    elif (( IDX < 20 )); then
        PARTITION="mit_preemptable"
        GPU_SPEC="gpu:1"
        REQUEUE="#SBATCH --requeue"
        WALLTIME="24:00:00"
    else
        PARTITION="ou_bcs_low"
        GPU_SPEC="gpu:1"
        REQUEUE="#SBATCH --requeue"
        WALLTIME="24:00:00"
    fi

    # Build sequential python commands for each chromosome
    PYTHON_CMDS=""
    for CHROM in $CHROMS; do
        PYTHON_CMDS+="
echo \"[mega s${IDX}] ${CHROM} started at \$(date)\"
python run_sae_fast.py \\
    --auto \\
    --chrom ${CHROM} \\
    --fasta ${FASTA} \\
    --gtf ${GTF} \\
    --output_dir results/ \\
    --shard ${IDX}/${N_SHARDS} \\
    --max_regions ${MAX_REGIONS} \\
    --min_confidence 8.0 \\
    --batch_size 32 \\
    --padding 200 \\
    --checkpoint_interval 500 \\
    --extract_only \\
    --skip_notebook || echo \"[mega s${IDX}] WARNING: ${CHROM} failed, continuing\"
echo \"[mega s${IDX}] ${CHROM} done at \$(date)\"
"
    done

    cat > "${SBATCH}" <<SBATCH
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p ${PARTITION}
#SBATCH --gres=${GPU_SPEC}
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t ${WALLTIME}
#SBATCH -o ${LOGS}/${JOB_NAME}_%j.out
#SBATCH -e ${LOGS}/${JOB_NAME}_%j.err
${REQUEUE}

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[mega s${IDX}] Job started on \$(hostname) at \$(date)"
echo "[mega s${IDX}] ${N_SHARDS}-way sharding, index ${IDX}, partition ${PARTITION}"
${PYTHON_CMDS}
echo "[mega s${IDX}] All chromosomes done at \$(date)"
SBATCH

    JID=$(sbatch "${SBATCH}" | awk '{print $NF}')
    echo "  shard ${IDX}: job ${JID}  partition=${PARTITION}"
done

echo ""
echo "All ${N_SHARDS} mega-jobs submitted."
echo ""
echo "Monitor progress:"
echo "  squeue -u platawa"
echo "  tail -f ${LOGS}/sae_mega_s0_*.out"
echo ""
echo "Check completed shards:"
echo "  ls results/*/sae/*/COMPLETED 2>/dev/null | grep shard | wc -l"
echo ""
echo "After all shards finish, merge each chromosome:"
echo "  for chr in ${CHROMS}; do"
echo "    python merge_sae_shards.py --chrom \$chr --n_shards ${N_SHARDS} --output_dir results/"
echo "  done"
