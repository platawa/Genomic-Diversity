#!/bin/bash
# submit_conf0_shards.sh
#
# Submit 57 missing conf0.0 GPU shard extraction jobs.
# Depends on genome_tsne jobs having STARTED (--dependency=after:).
#
# Also submits latent analysis for the 5 already-complete chromosomes.

set -e

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf
GSTATS=${PROJECT}/results/_genome_sae_stats/20260323_193732_genome_minmax_22chroms/data/genome_wide_sae_stats.npz

# Genome-wide t-SNE job IDs (conf0.0 jobs wait for these to START)
GENOME_RAW=11162298
GENOME_NORM=11162301

mkdir -p "${LOGS}"

# Missing shards per chromosome
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

echo "=== Submitting 57 conf0.0 GPU shard extraction jobs ==="
echo "  (will start after genome_tsne jobs begin)"
echo ""

for CHROM in "${!MISSING[@]}"; do
    for IDX in ${MISSING[$CHROM]}; do
        JOB_NAME="c0_gpu_${CHROM}_s${IDX}"
        JID=$(sbatch \
            --dependency=after:${GENOME_RAW}:${GENOME_NORM} \
            --job-name=${JOB_NAME} \
            -p mit_preemptable \
            --gres=gpu:1 \
            --cpus-per-task=8 \
            --mem=100G \
            -t 12:00:00 \
            --requeue \
            -o ${LOGS}/${JOB_NAME}_%j.out \
            -e ${LOGS}/${JOB_NAME}_%j.err \
            --wrap="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python run_sae_fast.py --auto --chrom ${CHROM} --shard ${IDX}/8 --min_confidence 0.0 --batch_size 8 --checkpoint_interval 500 --extract_only --skip_notebook --output_dir results/" \
            | awk '{print $NF}')
        echo "  ${CHROM} shard ${IDX}/8: job ${JID}"
    done
done

echo ""
echo "=== Submitting latent analysis for 5 already-complete conf0.0 chromosomes ==="
echo ""

for CHROM in chr8 chr13 chr21 chr22 chrY; do
    # Raw latent
    JID=$(sbatch \
        --dependency=after:${GENOME_RAW} \
        --job-name=c0_latent_${CHROM} \
        -p ou_bcs_normal \
        --cpus-per-task=8 \
        --mem=500G \
        -t 12:00:00 \
        --exclude=node3806 \
        -o ${LOGS}/c0_latent_${CHROM}_%j.out \
        -e ${LOGS}/c0_latent_${CHROM}_%j.err \
        --wrap="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/compute_sae_latent.py --from_shards --chrom ${CHROM} --results_dir results/ --n_shards 8 --output_dir results/${CHROM}/sae/latent_analysis_conf0 --embedding both" \
        | awk '{print $NF}')
    echo "  ${CHROM} raw latent: job ${JID}"

    # Normalized latent
    JID2=$(sbatch \
        --dependency=after:${GENOME_RAW} \
        --job-name=c0_norm_${CHROM} \
        -p ou_bcs_normal \
        --cpus-per-task=8 \
        --mem=500G \
        -t 12:00:00 \
        --exclude=node3806 \
        -o ${LOGS}/c0_norm_${CHROM}_%j.out \
        -e ${LOGS}/c0_norm_${CHROM}_%j.err \
        --wrap="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/compute_sae_latent.py --from_shards --chrom ${CHROM} --results_dir results/ --n_shards 8 --output_dir results/${CHROM}/sae/latent_analysis_conf0_normalized --global_stats ${GSTATS} --embedding both" \
        | awk '{print $NF}')
    echo "  ${CHROM} norm latent: job ${JID2}"

    # Raw plot
    JID3=$(sbatch \
        --dependency=afterok:${JID} \
        --job-name=c0_plot_${CHROM} \
        -p ou_bcs_normal \
        --cpus-per-task=4 \
        --mem=64G \
        -t 01:00:00 \
        -o ${LOGS}/c0_plot_${CHROM}_%j.out \
        -e ${LOGS}/c0_plot_${CHROM}_%j.err \
        --wrap="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/plot_sae_latent.py --chrom ${CHROM} --input_dir results/${CHROM}/sae/latent_analysis_conf0 --gtf ${GTF}" \
        | awk '{print $NF}')
    echo "  ${CHROM} raw plot: job ${JID3}"

    # Norm plot
    JID4=$(sbatch \
        --dependency=afterok:${JID2} \
        --job-name=c0_plotn_${CHROM} \
        -p ou_bcs_normal \
        --cpus-per-task=4 \
        --mem=64G \
        -t 01:00:00 \
        -o ${LOGS}/c0_plotn_${CHROM}_%j.out \
        -e ${LOGS}/c0_plotn_${CHROM}_%j.err \
        --wrap="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/plot_sae_latent.py --chrom ${CHROM} --input_dir results/${CHROM}/sae/latent_analysis_conf0_normalized --gtf ${GTF}" \
        | awk '{print $NF}')
    echo "  ${CHROM} norm plot: job ${JID4}"
done

echo ""
echo "Total: 57 GPU jobs + 10 latent + 10 plot jobs"
echo "All depend on genome_tsne jobs having STARTED"
echo "Monitor: squeue -u platawa | grep c0"
