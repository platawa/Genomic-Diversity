#!/bin/bash
# Resubmit incomplete SAE shards with CORRECT flags
# Total: 44 jobs (3 chr1 + 1 chr2 + 1 chr4 + 1 chr6 + 1 chr8 + 1 chr18 + 36 chr19)
# Distribution: ou_bcs_normal(15) + mit_preemptable(15) + ou_bcs_low(10) + mit_normal_gpu(4)

set -e

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
mkdir -p ${LOGS}

FASTA=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna
GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf

# Round-robin partition assignment
PARTITIONS=("ou_bcs_normal" "mit_preemptable" "ou_bcs_low" "mit_normal_gpu")
PARTITION_COUNTS=(0 0 0 0)
PARTITION_LIMITS=(15 15 10 4)

get_next_partition() {
  for i in 0 1 2 3; do
    if [ ${PARTITION_COUNTS[$i]} -lt ${PARTITION_LIMITS[$i]} ]; then
      echo "${PARTITIONS[$i]}"
      PARTITION_COUNTS[$i]=$((PARTITION_COUNTS[$i] + 1))
      return 0
    fi
  done
  echo "ERROR: All partitions full!" >&2
  return 1
}

echo "Submitting 44 GPU shard jobs with corrected flags..."
echo ""

JOB_COUNT=0

# Function to submit a single shard
submit_shard() {
  local CHROM=$1 SHARD=$2
  local PART=$(get_next_partition)
  
  JOB_COUNT=$((JOB_COUNT + 1))
  local SBATCH_FILE=${LOGS}/.gpu_${CHROM}_s${SHARD}.sbatch
  
  cat > "${SBATCH_FILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J sae_${CHROM}_s${SHARD}
#SBATCH -p ${PART}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -o ${LOGS}/gpu_${CHROM}_s${SHARD}_%j.out
#SBATCH -e ${LOGS}/gpu_${CHROM}_s${SHARD}_%j.err
#SBATCH --requeue

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[$(date)] Starting SAE extraction: ${CHROM} shard ${SHARD}/36"

python run_sae_fast.py \
  --auto \
  --chrom ${CHROM} \
  --fasta ${FASTA} \
  --gtf ${GTF} \
  --output_dir results/ \
  --shard ${SHARD}/36 \
  --min_confidence 8.0 \
  --batch_size 16 \
  --padding 200 \
  --checkpoint_interval 500 \
  --extract_only \
  --skip_notebook

echo "[$(date)] Completed SAE for ${CHROM} shard ${SHARD}"
SBATCH_EOF

  JID=$(sbatch "${SBATCH_FILE}" | awk '{print $NF}')
  echo "  ${JOB_COUNT}. ${CHROM}/s${SHARD} -> ${PART}: job ${JID}"
}

# Submit individual shards (8 jobs)
echo "=== Individual shards (8 jobs) ==="
submit_shard chr1 28
submit_shard chr1 30
submit_shard chr1 33
submit_shard chr2 34
submit_shard chr4 31
submit_shard chr6 18
submit_shard chr8 34
submit_shard chr18 34

echo ""
echo "=== chr19 all shards (36 jobs) ==="
for SHARD in $(seq 0 35); do
  submit_shard chr19 $SHARD
done

echo ""
echo "All 44 GPU jobs submitted."
echo "Partition distribution: ou_bcs_normal(${PARTITION_COUNTS[0]}), mit_preemptable(${PARTITION_COUNTS[1]}), ou_bcs_low(${PARTITION_COUNTS[2]}), mit_normal_gpu(${PARTITION_COUNTS[3]})"
echo "Monitor with: squeue -u platawa"
