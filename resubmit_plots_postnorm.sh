#!/bin/bash
# Resubmit POSTNORM enhanced plots only (clustering already completed)
# Fix: adds --scope chromosome which was missing from the original submission
cd /orcd/data/zhang_f/001/platawa/jan31_files

GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"
CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr20 chr21 chr22 chrX"

echo "=== Submitting POSTNORM plot-only jobs ==="
for chr in $CHROMS; do
    # Check that latent_analysis_postnorm exists for this chrom
    LATENT_DIR="results/${chr}/sae/latent_analysis_postnorm"
    if [ ! -d "$LATENT_DIR" ]; then
        echo "  SKIP $chr: no latent_analysis_postnorm directory"
        continue
    fi

    JOB=$(sbatch -p mit_preemptable --cpus-per-task=4 --mem=16G -t 1:00:00 \
        -J "pnplot_${chr}" \
        -o "logs/pnplot_${chr}_%j.out" \
        -e "logs/pnplot_${chr}_%j.err" \
        --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/enhanced_latent_plots.py --scope chromosome --chrom ${chr} --organism human --gtf ${GTF} --results_dir results/ --latent_subdir latent_analysis_postnorm" \
        2>&1 | awk '{print $4}')
    echo "  ${chr}: ${JOB}"
done
