#!/bin/bash
# submit_conf0_gpu_shards.sh
#
# Submit 57 missing conf0.0 GPU shard extraction jobs.
# No dependency on conf8.0 — completely independent.
# Spread across 4 GPU partitions.

set -e

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
mkdir -p "${LOGS}"

# Missing shards per chromosome (from investigation)
declare -A MISSING
MISSING[chr1]="2 4 5 7"
MISSING[chr2]="1 2 7"
MISSING[chr3]="1 4 6 7"
MISSING[chr4]="1 4 5 7"
MISSING[chr5]="1 2 4 5"
MISSING[chr6]="1 4 7"
MISSING[chr7]="1 6 7"
MISSING[chr9]="5 6"
MISSING[chr10]="3 7"
MISSING[chr11]="3 5 6 7"
MISSING[chr12]="5 6 7"
MISSING[chr14]="4 7"
MISSING[chr15]="3 7"
MISSING[chr16]="4 5 6 7"
MISSING[chr17]="2 6 7"
MISSING[chr18]="2 7"
MISSING[chr19]="4 6 7"
MISSING[chr20]="1"
MISSING[chrX]="1 5 6 7"

# Partition rotation to spread jobs
PARTITIONS=(mit_preemptable mit_preemptable mit_preemptable ou_bcs_normal)
PART_IDX=0

echo "Submitting 57 conf0.0 GPU shard extraction jobs"
echo "Output: results/chrN/sae/TIMESTAMP_all_conf0.0_shardIDXof8/"
echo "Partitions: mit_preemptable, ou_bcs_normal"
echo ""

TOTAL=0
for CHROM in chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr9 chr10 chr11 chr12 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chrX; do
    for IDX in ${MISSING[$CHROM]}; do
        PART=${PARTITIONS[$((PART_IDX % ${#PARTITIONS[@]}))]}
        JOB_NAME="c0_${CHROM}_s${IDX}"

        JID=$(sbatch \
            --job-name=${JOB_NAME} \
            -p ${PART} \
            --gres=gpu:1 \
            --cpus-per-task=8 \
            --mem=100G \
            -t 12:00:00 \
            --requeue \
            -o ${LOGS}/${JOB_NAME}_%j.out \
            -e ${LOGS}/${JOB_NAME}_%j.err \
            --wrap="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python run_sae_fast.py --auto --chrom ${CHROM} --shard ${IDX}/8 --min_confidence 0.0 --batch_size 8 --checkpoint_interval 500 --extract_only --skip_notebook --output_dir results/" \
            | awk '{print $NF}')
        echo "  ${CHROM} shard ${IDX}/8: job ${JID} (${PART})"
        TOTAL=$((TOTAL + 1))
        PART_IDX=$((PART_IDX + 1))
    done
done

echo ""
echo "Total: ${TOTAL} GPU jobs submitted"
echo "Monitor: squeue -u platawa | grep c0"
