#!/bin/bash
# Wave 1: All independent jobs in parallel
set -euo pipefail

PROJECT="/orcd/data/zhang_f/001/platawa/jan31_files"
HUMAN_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"
ECOLI_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf"
BACILLUS_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf"
LOGDIR="${PROJECT}/logs"
mkdir -p "${LOGDIR}"

SETUP="cd ${PROJECT} && module load miniforge/24.3.0-0 && conda activate evo2_sep28"

echo "=== WAVE 1: Submitting all independent jobs ==="
echo ""

# 1. ChrY enhanced plots (missing UMAP distance plots)
echo "1. ChrY UMAP distance plots..."
sbatch --job-name=chrY_enh --partition=pi_zhang_f --cpus-per-task=4 --mem=16G --time=1:00:00 \
    --output="${LOGDIR}/chrY_enh_%j.out" --error="${LOGDIR}/chrY_enh_%j.err" \
    --wrap="${SETUP} && python tools/enhanced_latent_plots.py --scope chromosome --chrom chrY --organism human --gtf ${HUMAN_GTF} --results_dir results/ --plots distance_to_gene,entropy_color"
echo "   -> submitted"

# 2. E. coli: Add UMAP embedding first
echo "2. E. coli UMAP embedding..."
sbatch --job-name=ecoli_umap --partition=pi_zhang_f --cpus-per-task=4 --mem=32G --time=2:00:00 \
    --output="${LOGDIR}/ecoli_umap_%j.out" --error="${LOGDIR}/ecoli_umap_%j.err" \
    --wrap="${SETUP} && python tools/add_umap_embedding.py --chrom NC_000913.3 --results_dir results/"
echo "   -> submitted"

# 3. Human UMAP enhanced plots (all 22 chroms that have latent_analysis)
echo "3. Human UMAP enhanced plots (organism-wide)..."
sbatch --job-name=human_enh --partition=pi_zhang_f --cpus-per-task=4 --mem=64G --time=6:00:00 \
    --output="${LOGDIR}/human_enh_%j.out" --error="${LOGDIR}/human_enh_%j.err" \
    --wrap="${SETUP} && python tools/enhanced_latent_plots.py --scope organism --organism human --gtf ${HUMAN_GTF} --results_dir results/ --plots distance_to_gene,entropy_color,length_stats,firing_counts,firing_thresholds"
echo "   -> submitted"

# 4. Genome-wide nuc_stats aggregation (CPU only, fast)
echo "4. Genome-wide nuc_stats aggregation..."
sbatch --job-name=agg_stats --partition=pi_zhang_f --cpus-per-task=4 --mem=32G --time=1:00:00 \
    --output="${LOGDIR}/agg_stats_%j.out" --error="${LOGDIR}/agg_stats_%j.err" \
    --wrap="${SETUP} && python scan_sae_global_stats.py --aggregate_corrected --all_human --results_dir results/"
echo "   -> submitted"

# 5. % features activated plots (all human + bacteria)
echo "5. % features activated plots..."
sbatch --job-name=pct_feat --partition=pi_zhang_f --cpus-per-task=4 --mem=32G --time=2:00:00 \
    --output="${LOGDIR}/pct_feat_%j.out" --error="${LOGDIR}/pct_feat_%j.err" \
    --wrap="${SETUP} && python tools/plot_pct_features_activated.py --all_human --results_dir results/ && python tools/plot_pct_features_activated.py --organism ecoli --results_dir results/ && python tools/plot_pct_features_activated.py --organism bacillus --results_dir results/"
echo "   -> submitted"

# 6. Region length uniformity cross-chromosome summary
echo "6. Region length uniformity summary..."
sbatch --job-name=len_summ --partition=pi_zhang_f --cpus-per-task=4 --mem=16G --time=1:00:00 \
    --output="${LOGDIR}/len_summ_%j.out" --error="${LOGDIR}/len_summ_%j.err" \
    --wrap="${SETUP} && python tools/plot_region_length_summary.py --all_human --include_bacteria --results_dir results/ --output_dir results/_genome_wide/region_length_summary"
echo "   -> submitted"

echo ""
echo "=== All 6 Wave 1 jobs submitted ==="
echo "Monitor with: squeue -u platawa"
