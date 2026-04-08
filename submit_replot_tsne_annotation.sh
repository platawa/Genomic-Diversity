#!/bin/bash
#SBATCH -J replot_tsne
#SBATCH -p pi_zhang_f
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o logs/replot_tsne_%j.out
#SBATCH -e logs/replot_tsne_%j.err

set -e

module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"

# All human chromosomes with their SAE run directories
declare -A CHROMS=(
    [chr1]="results/chr1/sae"
    [chr2]="results/chr2/sae"
    [chr3]="results/chr3/sae"
    [chr4]="results/chr4/sae"
    [chr5]="results/chr5/sae"
    [chr6]="results/chr6/sae"
    [chr7]="results/chr7/sae"
    [chr8]="results/chr8/sae"
    [chr9]="results/chr9/sae"
    [chr10]="results/chr10/sae"
    [chr11]="results/chr11/sae"
    [chr12]="results/chr12/sae"
    [chr13]="results/chr13/sae"
    [chr14]="results/chr14/sae"
    [chr15]="results/chr15/sae"
    [chr16]="results/chr16/sae"
    [chr17]="results/chr17/sae"
    [chr18]="results/chr18/sae"
    [chr20]="results/chr20/sae"
    [chr21]="results/chr21/sae"
    [chr22]="results/chr22/sae"
    [chrX]="results/chrX/sae"
)

echo "[$(date)] Re-plotting annotation t-SNE for all chromosomes (uniform s=3, alpha=0.5)..."

for chrom in "${!CHROMS[@]}"; do
    sae_dir="${CHROMS[$chrom]}"

    echo "[$(date)] Processing $chrom..."

    # Use --auto to find latest COMPLETED SAE run
    python tools/plot_tsne_by_annotation.py \
        --auto \
        --chrom "$chrom" \
        --gtf "$GTF" \
        --results_dir results/ \
        2>&1 | tee -a logs/replot_tsne.log

    echo "[$(date)]   done $chrom"
done

echo "[$(date)] All chromosomes re-plotted."
