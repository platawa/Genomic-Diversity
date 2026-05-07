#!/bin/bash
# submit_recompute_prenorm_12chroms.sh
#
# Recompute t-SNE/UMAP for the 12 prenorm chroms blocked by Apr-23 schema regen.
# Uses existing maxpooled_vectors.npy (no re-pooling). Then chains plot jobs.
#
# Per-chrom: 8 CPU / 500 GB / 12h on pi_zhang_f for compute,
#            4 CPU / 16 GB / 1h on mit_preemptable for plot (--dependency=afterok).
#
# Usage:  cd /orcd/data/zhang_f/001/platawa/jan31_files && bash submit_recompute_prenorm_12chroms.sh

set -e
PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
mkdir -p "${LOGS}"

cd ${PROJECT}

GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf
STATS=${PROJECT}/results/_genome_sae_stats/20260406_235042_corrected_24chroms/data/genome_wide_sae_stats_corrected.npz

CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr13 chr17 chr18"

echo "Submitting recompute (cluster stage) + plot for 12 prenorm chroms"
echo

for CHR in $CHROMS; do
    # Compute job — re-runs cluster stage on existing maxpooled_vectors.npy
    COMPUTE_JID=$(sbatch --parsable \
        -p pi_zhang_f --cpus-per-task=8 --mem=500G -t 12:00:00 \
        -J "recomp_${CHR}_prenorm" \
        -o "${LOGS}/recomp_${CHR}_prenorm_%j.out" \
        -e "${LOGS}/recomp_${CHR}_prenorm_%j.err" \
        --wrap "cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/compute_sae_latent.py --from_shards --chrom ${CHR} --results_dir results/ --norm_method prenorm --global_stats ${STATS} --stage cluster --embedding both")

    # Plot job — depends on compute success
    PLOT_JID=$(sbatch --parsable \
        --dependency=afterok:${COMPUTE_JID} \
        -p mit_preemptable --cpus-per-task=4 --mem=16G -t 1:00:00 \
        -J "plot_${CHR}_prenorm" \
        -o "${LOGS}/plot_${CHR}_prenorm_%j.out" \
        -e "${LOGS}/plot_${CHR}_prenorm_%j.err" \
        --wrap "cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/plot_sae_latent.py --chrom ${CHR} --results_dir results/ --input_dir results/${CHR}/sae/latent_analysis_prenorm --plots all --gtf ${GTF}")

    echo "  ${CHR}: compute=${COMPUTE_JID}  plot=${PLOT_JID} (afterok)"
done

echo
echo "Submitted 12 compute + 12 plot = 24 jobs total."
echo "Monitor: squeue -u platawa -n recomp_chr1_prenorm,recomp_chr2_prenorm,..."
echo "Or:      squeue -u platawa | grep -E 'recomp|plot_chr.*prenorm'"
