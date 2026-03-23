#!/bin/bash
#SBATCH --job-name=all_plots
#SBATCH --partition=pi_zhang_f
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/orcd/data/zhang_f/001/platawa/jan31_files/results/all_plots_%j.out
#SBATCH --error=/orcd/data/zhang_f/001/platawa/jan31_files/results/all_plots_%j.err

# Usage (pass flags after the script name when submitting):
#   sbatch run_all_plots.sh                        # run everything
#   sbatch run_all_plots.sh --karyotype            # karyotype only
#   sbatch run_all_plots.sh --chrom-analyses       # all 6 chr analyses
#   sbatch run_all_plots.sh --analyses 2 3 5       # specific analyses only
#   sbatch run_all_plots.sh --karyotype --analyses 4

set -e
module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf

# ── Parse flags ────────────────────────────────────────────────────────────────
RUN_KARYOTYPE=false
RUN_CHROM=false
ANALYSES="1 2 3 4 5 6"

# No args → run everything
if [[ $# -eq 0 ]]; then
    RUN_KARYOTYPE=true
    RUN_CHROM=true
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --karyotype)
            RUN_KARYOTYPE=true
            shift ;;
        --chrom-analyses)
            RUN_CHROM=true
            shift ;;
        --analyses)
            RUN_CHROM=true
            shift
            ANALYSES=""
            while [[ $# -gt 0 ]] && [[ "$1" =~ ^[0-9]+$ ]]; do
                ANALYSES="$ANALYSES $1"
                shift
            done ;;
        *)
            echo "Unknown flag: $1"
            echo "Usage: sbatch run_all_plots.sh [--karyotype] [--chrom-analyses] [--analyses 1 2 3 ...]"
            exit 1 ;;
    esac
done

echo "Run karyotype:       $RUN_KARYOTYPE"
echo "Run chrom-analyses:  $RUN_CHROM  (analyses:$ANALYSES)"
echo ""

# ── Step 1: Karyotype ──────────────────────────────────────────────────────────
if [[ "$RUN_KARYOTYPE" == "true" ]]; then
    echo "========================================"
    echo "Karyotype plots (entropy + drops + combined)"
    echo "========================================"
    python tools/plot_genome_karyotype.py \
        --results_dir results/ \
        --all_human \
        --gtf "$GTF"
fi

# ── Step 2: Chromosome-level analyses ─────────────────────────────────────────
if [[ "$RUN_CHROM" == "true" ]]; then
    echo "========================================"
    echo "Chromosome-level analyses:$ANALYSES"
    echo "========================================"
    # shellcheck disable=SC2086
    python tools/chromosome_analysis.py \
        --results_dir results/ \
        --gtf "$GTF" \
        --analyses $ANALYSES
fi

echo "Done."
