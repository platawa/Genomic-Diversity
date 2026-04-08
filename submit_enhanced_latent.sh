#!/bin/bash
# submit_enhanced_latent.sh
#
# Submit enhanced latent analysis plots for all organisms and chromosomes.
# Runs on ORCD cluster (CPU only, no GPU needed).
#
# Usage:
#   bash submit_enhanced_latent.sh [ecoli|bacillus|human_perchr|human_genome|all]

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_DIR="/orcd/data/zhang_f/001/platawa/jan31_files"
RESULTS_DIR="${PROJECT_DIR}/results"
CONDA_ENV="evo2_sep28"

HUMAN_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"
ECOLI_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf"
BACILLUS_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf"

ECOLI_FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna"
BACILLUS_FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/GCF_000009045.1_ASM904v1_genomic.fna"

ALL_PLOTS="distance_to_gene,entropy_color,length_stats,top_features,firing_counts,firing_thresholds"

# SLURM defaults (CPU only)
PARTITION="mit_preemptable"
TIME="4:00:00"
CPUS=4
MEM="32G"

MODE="${1:-all}"

# ── Helper functions ───────────────────────────────────────────────────────────

submit_job() {
    local job_name="$1"
    local cmd="$2"
    local time_limit="${3:-$TIME}"
    local mem="${4:-$MEM}"

    sbatch \
        --job-name="enh_${job_name}" \
        --partition="${PARTITION}" \
        --time="${time_limit}" \
        --cpus-per-task="${CPUS}" \
        --mem="${mem}" \
        --output="${PROJECT_DIR}/logs/enhanced_latent_%j_${job_name}.out" \
        --error="${PROJECT_DIR}/logs/enhanced_latent_%j_${job_name}.err" \
        --wrap="module load miniforge/24.3.0-0 && conda activate ${CONDA_ENV} && cd ${PROJECT_DIR} && ${cmd}"

    echo "  Submitted: ${job_name}"
}

# ── E. coli ────────────────────────────────────────────────────────────────────

submit_ecoli() {
    echo "=== Submitting E. coli enhanced latent analysis ==="

    submit_job "ecoli" \
        "python tools/enhanced_latent_plots.py \
            --scope organism --organism ecoli \
            --gtf ${ECOLI_GTF} \
            --results_dir ${RESULTS_DIR} \
            --plots ${ALL_PLOTS}" \
        "2:00:00" "16G"
}

# ── Bacillus ───────────────────────────────────────────────────────────────────

submit_bacillus() {
    echo "=== Submitting Bacillus enhanced latent analysis ==="

    submit_job "bacillus" \
        "python tools/enhanced_latent_plots.py \
            --scope organism --organism bacillus \
            --gtf ${BACILLUS_GTF} \
            --results_dir ${RESULTS_DIR} \
            --plots ${ALL_PLOTS}" \
        "2:00:00" "16G"
}

# ── Human per-chromosome ──────────────────────────────────────────────────────

submit_human_perchr() {
    echo "=== Submitting human per-chromosome enhanced latent analysis ==="

    HUMAN_CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 \
chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX"

    for chr in ${HUMAN_CHROMS}; do
        submit_job "${chr}" \
            "python tools/enhanced_latent_plots.py \
                --scope chromosome --chrom ${chr} --organism human \
                --gtf ${HUMAN_GTF} \
                --results_dir ${RESULTS_DIR} \
                --plots ${ALL_PLOTS}" \
            "4:00:00" "32G"
    done
}

# ── Human genome-wide ─────────────────────────────────────────────────────────

submit_human_genome() {
    echo "=== Submitting human genome-wide enhanced latent analysis ==="

    submit_job "genome_wide" \
        "python tools/enhanced_latent_plots.py \
            --scope genome_wide --organism human \
            --gtf ${HUMAN_GTF} \
            --results_dir ${RESULTS_DIR} \
            --plots ${ALL_PLOTS}" \
        "12:00:00" "256G"
}

# ── CRISPRCasFinder (separate step, needed before CRISPR component plots) ────

submit_crisprcasfinder() {
    echo "=== Submitting CRISPRCasFinder for E.coli + Bacillus ==="
    echo "NOTE: CRISPRCasFinder must be installed. If not available, use:"
    echo "  conda install -c bioconda crisprcasfinder"
    echo "  OR use the web version: https://crisprcas.i2bc.paris-saclay.fr/"
    echo ""
    echo "After CRISPRCasFinder runs, use plot_crispr_components.py:"
    echo "  python tools/plot_crispr_components.py \\"
    echo "    --crispr_json <output.json> \\"
    echo "    --latent <cluster_assignments.tsv> \\"
    echo "    --boundaries <drop_boundaries.tsv> \\"
    echo "    --organism ecoli --output_dir <dir>"
}

# ── Main dispatch ──────────────────────────────────────────────────────────────

mkdir -p "${PROJECT_DIR}/logs"

case "${MODE}" in
    ecoli)
        submit_ecoli
        ;;
    bacillus)
        submit_bacillus
        ;;
    human_perchr)
        submit_human_perchr
        ;;
    human_genome)
        submit_human_genome
        ;;
    crispr)
        submit_crisprcasfinder
        ;;
    all)
        submit_ecoli
        submit_bacillus
        submit_human_perchr
        submit_human_genome
        submit_crisprcasfinder
        ;;
    *)
        echo "Usage: $0 [ecoli|bacillus|human_perchr|human_genome|crispr|all]"
        exit 1
        ;;
esac

echo ""
echo "Done. Check job status with: squeue -u \$USER"
