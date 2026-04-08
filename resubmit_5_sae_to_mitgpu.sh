#!/bin/bash
# Resubmit 5 confidence 0.0 SAE jobs to mit_preemptable_gpu partition

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
FASTA=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna
GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf
LOGS=${PROJECT}/logs
MIN_CONF=0.0
N_SHARDS=8

echo "Resubmitting 5 SAE jobs to mit_preemptable_gpu..."
echo ""

# Job list: (CHROM SHARD)
JOBS=(
  "chr1 1"
  "chr2 5"
  "chr3 4"
  "chr3 2"
  "chr19 1"
)

for JOB_INFO in "${JOBS[@]}"; do
    CHROM=$(echo $JOB_INFO | awk '{print $1}')
    SHARD=$(echo $JOB_INFO | awk '{print $2}')
    JOB_NAME="saeall_${CHROM}_s${SHARD}"

    JOBID=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --partition=mit_preemptable_gpu \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=200G \
        --time=12:00:00 \
        --output="${LOGS}/${JOB_NAME}_%j.out" \
        --error="${LOGS}/${JOB_NAME}_%j.err" \
        --requeue \
        --wrap="module load miniforge/24.3.0-0 && conda activate evo2_sep28 && cd ${PROJECT} && python run_sae_fast.py --auto --chrom ${CHROM} --fasta ${FASTA} --gtf ${GTF} --output_dir results/ --shard ${SHARD}/${N_SHARDS} --min_confidence ${MIN_CONF} --batch_size 16 --padding 200 --checkpoint_interval 500 --extract_only --skip_notebook")

    echo "  ${JOB_NAME}: Job ${JOBID}"
done

echo ""
echo "Resubmitted 5 jobs to mit_preemptable_gpu"
