#!/bin/bash
# Resubmit 10 pending saeall jobs from ou_bcs_low to mit_preemptable

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
FASTA=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna
GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf
LOGS=${PROJECT}/logs

echo "Getting 10 pending jobs from ou_bcs_low..."
squeue -u platawa -p ou_bcs_low --states=PENDING -h -o "%.5i %.20j" 2>/dev/null | grep saeall | head -10 > /tmp/jobs_to_resubmit.txt

COUNT=0
while IFS=' ' read -r JOBID JOBNAME; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/10] Processing $JOBNAME (job $JOBID)..."

    # Parse job name: saeall_chrX_sN
    CHROM=$(echo $JOBNAME | cut -d_ -f2)
    SHARD_IDX=$(echo $JOBNAME | cut -d_ -f3 | sed 's/^s//')

    echo "  Cancelling job $JOBID..."
    scancel $JOBID

    echo "  Resubmitting to mit_preemptable: $CHROM shard $SHARD_IDX..."
    NEW_JOBID=$(sbatch --parsable \
        --job-name="${JOBNAME}" \
        --partition=mit_preemptable \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=200G \
        --time=12:00:00 \
        --output="${LOGS}/${JOBNAME}_%j.out" \
        --error="${LOGS}/${JOBNAME}_%j.err" \
        --requeue \
        --wrap="module load miniforge/24.3.0-0 && conda activate evo2_sep28 && cd ${PROJECT} && python run_sae_fast.py --auto --chrom ${CHROM} --fasta ${FASTA} --gtf ${GTF} --output_dir results/ --shard ${SHARD_IDX}/8 --min_confidence 0.0 --batch_size 16 --padding 200 --checkpoint_interval 500 --extract_only --skip_notebook")
    echo "  New job ID: $NEW_JOBID"
done < /tmp/jobs_to_resubmit.txt

echo ""
echo "Done! Resubmitted $COUNT jobs to mit_preemptable"
