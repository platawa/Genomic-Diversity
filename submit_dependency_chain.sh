#!/bin/bash
# Submit all remaining jobs with SLURM dependencies
set -euo pipefail

PROJECT="/orcd/data/zhang_f/001/platawa/jan31_files"
HUMAN_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"
GLOBAL_STATS="${PROJECT}/results/_genome_sae_stats/20260406_235042_corrected_24chroms/data/genome_wide_sae_stats_corrected.npz"
LOGDIR="${PROJECT}/logs"
SETUP="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28"

CHRY_GW_JOBID=${1:-11489586}  # pass chrY_gw job ID as argument

echo "=== JOB A: Normalized per-chromosome latent analysis (24 chroms) ==="
NORM_JOBIDS=""
for CHROM in chr{1..22} chrX chrY; do
    JID=$(sbatch --parsable --job-name="norm_${CHROM}" --partition=pi_zhang_f \
        --cpus-per-task=4 --mem=32G --time=2:00:00 \
        --output="${LOGDIR}/norm_${CHROM}_%j.out" --error="${LOGDIR}/norm_${CHROM}_%j.err" \
        --wrap="${SETUP} && python tools/compute_sae_latent.py --from_shards --chrom ${CHROM} --results_dir results/ --global_stats ${GLOBAL_STATS} --stage both --embedding both")
    echo "  ${CHROM}: ${JID}"
    if [ -n "$NORM_JOBIDS" ]; then
        NORM_JOBIDS="${NORM_JOBIDS}:${JID}"
    else
        NORM_JOBIDS="${JID}"
    fi
done
echo "  All 24 norm jobs submitted: ${NORM_JOBIDS}"
echo ""

echo "=== JOB B: Enhanced plots for normalized data (depends on all Job A) ==="
JOB_B=$(sbatch --parsable --dependency=afterok:${NORM_JOBIDS} \
    --job-name="norm_enh" --partition=pi_zhang_f \
    --cpus-per-task=4 --mem=64G --time=6:00:00 \
    --output="${LOGDIR}/norm_enh_%j.out" --error="${LOGDIR}/norm_enh_%j.err" \
    --wrap="${SETUP} && python tools/enhanced_latent_plots.py --scope organism --organism human --gtf ${HUMAN_GTF} --results_dir results/ --latent_subdir latent_analysis_normalized --plots distance_to_gene,entropy_color,length_stats,firing_counts,firing_thresholds")
echo "  Job B: ${JOB_B} (depends on all Job A)"
echo ""

echo "=== JOB C: Genome-wide replot with chrY (depends on chrY_gw ${CHRY_GW_JOBID}) ==="
JOB_C=$(sbatch --parsable --dependency=afterok:${CHRY_GW_JOBID} \
    --job-name="gw_replot" --partition=pi_zhang_f \
    --cpus-per-task=4 --mem=16G --time=1:00:00 \
    --output="${LOGDIR}/gw_replot_%j.out" --error="${LOGDIR}/gw_replot_%j.err" \
    --wrap="${SETUP} && LATEST_TSV=\$(ls -td ${PROJECT}/results/_genome_wide/sae_tsne/20260407_*_24chroms_with_chrY/data/cluster_assignments.tsv 2>/dev/null | head -1) && if [ -z \"\$LATEST_TSV\" ]; then LATEST_TSV=\$(find ${PROJECT}/results/_genome_wide/sae_tsne -name 'cluster_assignments.tsv' -newer ${PROJECT}/results/_genome_wide/sae_tsne/20260406_022505_23chroms_740913regions/data/cluster_assignments.tsv 2>/dev/null | head -1); fi && echo \"Using TSV: \$LATEST_TSV\" && python tools/genome_replot_from_tsv.py --tsv \$LATEST_TSV --output_dir \$(dirname \$(dirname \$LATEST_TSV))/plots_s0.5_perchrom --dot_size 0.5 --alpha 0.3 --dpi 300 --per_chrom")
echo "  Job C: ${JOB_C} (depends on chrY_gw ${CHRY_GW_JOBID})"
echo ""

echo "=== JOB D: Genome-wide normalized UMAP/t-SNE (depends on all Job A) ==="
JOB_D=$(sbatch --parsable --dependency=afterok:${NORM_JOBIDS} \
    --job-name="norm_gw" --partition=pi_zhang_f \
    --cpus-per-task=8 --mem=128G --time=12:00:00 \
    --output="${LOGDIR}/norm_gw_%j.out" --error="${LOGDIR}/norm_gw_%j.err" \
    --wrap="${SETUP} && python tools/genome_sae_tsne.py --all_human --results_dir results/ --global_stats ${GLOBAL_STATS} --latent_subdir latent_analysis_normalized")
echo "  Job D: ${JOB_D} (depends on all Job A)"
echo ""

echo "=== SUMMARY ==="
echo "Job A: 24 normalized latent analysis jobs (no deps)"
echo "Job B: ${JOB_B} — normalized enhanced plots (after all Job A)"
echo "Job C: ${JOB_C} — genome-wide replot with chrY (after chrY_gw ${CHRY_GW_JOBID})"
echo "Job D: ${JOB_D} — genome-wide normalized embeddings (after all Job A)"
echo ""
echo "Total: 27 jobs submitted. Monitor with: squeue -u platawa"
