#!/bin/bash
# Submit cluster annotation plots for all chromosomes
# Two versions per chromosome: GTF-only and GTF+RepeatMasker
#
# Runs as a SLURM array job (1 task per chromosome)
# Each task takes ~4-5 minutes, no GPU needed

#SBATCH --job-name=cluster_plots
#SBATCH --partition=mit_normal
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-22
#SBATCH --output=logs/cluster_plots_%a_%j.out
#SBATCH --error=logs/cluster_plots_%a_%j.err

module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr20 chr21 chr22 chrX)
# Note: chr19 excluded (no sae results typically), adjust if needed

# Get chromosome for this array task
IDX=${SLURM_ARRAY_TASK_ID}
if [ $IDX -ge ${#CHROMS[@]} ]; then
    echo "Array index $IDX exceeds chromosome count, exiting"
    exit 0
fi
CHROM=${CHROMS[$IDX]}

GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf
RMSK=data/rmsk_grch38.bed

LATENT_DIR="results/${CHROM}/sae/latent_analysis_postnorm"

# Check if latent_analysis_postnorm exists for this chromosome
if [ ! -d "${LATENT_DIR}/data" ]; then
    echo "SKIP: ${LATENT_DIR}/data does not exist for ${CHROM}"
    exit 0
fi

echo "=== ${CHROM}: GTF-only plot ==="
python tools/identify_tsne_clusters.py \
    --latent_dir "${LATENT_DIR}" \
    --gtf "${GTF}" \
    --chrom "${CHROM}" \
    --mode plot \
    --output_dir "${LATENT_DIR}/cluster_analysis_gtf_only"

echo ""
echo "=== ${CHROM}: GTF + RepeatMasker plot ==="
python tools/identify_tsne_clusters.py \
    --latent_dir "${LATENT_DIR}" \
    --gtf "${GTF}" \
    --chrom "${CHROM}" \
    --repeatmasker "${RMSK}" \
    --mode plot \
    --output_dir "${LATENT_DIR}/cluster_analysis"

echo ""
echo "DONE: ${CHROM}"
