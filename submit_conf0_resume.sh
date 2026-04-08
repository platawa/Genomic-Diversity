#!/bin/bash
# submit_conf0_resume.sh
# Resume 25 failed/timed-out conf0.0 GPU shard jobs from checkpoints.
# TIMEOUT jobs: --batch_size 8, 48h, mit_preemptable
# OOM jobs: --batch_size 4, 48h, mit_preemptable, request a100 (80GB VRAM)

set -e
PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs

# TIMEOUT shards (16) - resume with same batch_size, longer time
declare -A TIMEOUT_SHARDS
TIMEOUT_SHARDS[chr1_7]="20260330_135856_all_conf0.0_shard7of8"
TIMEOUT_SHARDS[chr2_1]="20260330_170632_all_conf0.0_shard1of8"
TIMEOUT_SHARDS[chr2_2]="20260330_170632_all_conf0.0_shard2of8"
TIMEOUT_SHARDS[chr2_7]="20260330_170821_all_conf0.0_shard7of8"
TIMEOUT_SHARDS[chr3_1]="20260330_142328_all_conf0.0_shard1of8"
TIMEOUT_SHARDS[chr4_1]="20260330_141110_all_conf0.0_shard1of8"
TIMEOUT_SHARDS[chr4_4]="20260330_230629_all_conf0.0_shard4of8"
TIMEOUT_SHARDS[chr4_5]="20260330_234633_all_conf0.0_shard5of8"
TIMEOUT_SHARDS[chr4_7]="20260331_050908_all_conf0.0_shard7of8"
TIMEOUT_SHARDS[chr5_1]="20260330_141110_all_conf0.0_shard1of8"
TIMEOUT_SHARDS[chr5_2]="20260331_002929_all_conf0.0_shard2of8"
TIMEOUT_SHARDS[chr6_1]="20260325_170641_all_conf0.0_shard1of8"
TIMEOUT_SHARDS[chr11_6]="20260330_143243_all_conf0.0_shard6of8"
TIMEOUT_SHARDS[chr12_7]="20260324_042515_all_conf0.0_shard7of8"
TIMEOUT_SHARDS[chr16_4]="20260330_140202_all_conf0.0_shard4of8"
TIMEOUT_SHARDS[chr18_2]="20260330_144810_all_conf0.0_shard2of8"

# OOM shards (9) - resume with smaller batch_size + request a100
declare -A OOM_SHARDS
OOM_SHARDS[chr3_4]="20260330_170820_all_conf0.0_shard4of8"
OOM_SHARDS[chr3_7]="20260330_230631_all_conf0.0_shard7of8"
OOM_SHARDS[chr7_6]="20260330_143243_all_conf0.0_shard6of8"
OOM_SHARDS[chr10_3]="20260330_143243_all_conf0.0_shard3of8"
OOM_SHARDS[chr12_6]="20260330_224148_all_conf0.0_shard6of8"
OOM_SHARDS[chr14_4]="20260331_054838_all_conf0.0_shard4of8"
OOM_SHARDS[chr15_7]="20260330_143547_all_conf0.0_shard7of8"
OOM_SHARDS[chr16_5]="20260330_140202_all_conf0.0_shard5of8"
OOM_SHARDS[chrX_6]="20260330_150344_all_conf0.0_shard6of8"

echo "=== Submitting 16 TIMEOUT resume jobs (batch_size=8, 48h) ==="
for KEY in "${!TIMEOUT_SHARDS[@]}"; do
    CHROM="${KEY%_*}"
    IDX="${KEY#*_}"
    DIR="${TIMEOUT_SHARDS[$KEY]}"
    RESUME_PATH="${PROJECT}/results/${CHROM}/sae/${DIR}"
    JOB_NAME="c0r_${CHROM}_s${IDX}"

    JID=$(sbatch \
        --job-name=${JOB_NAME} \
        -p mit_preemptable \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=100G \
        -t 2-00:00:00 \
        --requeue \
        -o ${LOGS}/${JOB_NAME}_%j.out \
        -e ${LOGS}/${JOB_NAME}_%j.err \
        --wrap="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python run_sae_fast.py --auto --chrom ${CHROM} --shard ${IDX}/8 --min_confidence 0.0 --batch_size 8 --checkpoint_interval 500 --extract_only --skip_notebook --resume_dir ${RESUME_PATH}" \
        | awk '{print $NF}')
    echo "  ${CHROM} s${IDX} (TIMEOUT resume): job ${JID}"
done

echo ""
echo "=== Submitting 9 OOM resume jobs (batch_size=4, 48h, a100) ==="
for KEY in "${!OOM_SHARDS[@]}"; do
    CHROM="${KEY%_*}"
    IDX="${KEY#*_}"
    DIR="${OOM_SHARDS[$KEY]}"
    RESUME_PATH="${PROJECT}/results/${CHROM}/sae/${DIR}"
    JOB_NAME="c0r_${CHROM}_s${IDX}"

    JID=$(sbatch \
        --job-name=${JOB_NAME} \
        -p mit_preemptable \
        --gres=gpu:a100:1 \
        --cpus-per-task=8 \
        --mem=100G \
        -t 2-00:00:00 \
        --requeue \
        -o ${LOGS}/${JOB_NAME}_%j.out \
        -e ${LOGS}/${JOB_NAME}_%j.err \
        --wrap="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python run_sae_fast.py --auto --chrom ${CHROM} --shard ${IDX}/8 --min_confidence 0.0 --batch_size 4 --checkpoint_interval 500 --extract_only --skip_notebook --resume_dir ${RESUME_PATH}" \
        | awk '{print $NF}')
    echo "  ${CHROM} s${IDX} (OOM resume, a100): job ${JID}"
done

echo ""
echo "Total: 25 resume jobs submitted on mit_preemptable (48h limit)"
