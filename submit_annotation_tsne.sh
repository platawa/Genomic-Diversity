#!/bin/bash
#SBATCH -J annotation_tsne
#SBATCH -p pi_zhang_f
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o logs/annotation_tsne_%j.out
#SBATCH -e logs/annotation_tsne_%j.err

set -e

module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

echo "[$(date)] Starting GTF annotation t-SNE plots for E. coli and Bacillus..."

# E. coli
echo "[$(date)] E. coli: Generating t-SNE plots colored by genomic annotation..."
python tools/plot_tsne_by_annotation.py \
    --sae_run results/NC_000913.3/sae/20260309_132936_max50_conf3.0 \
    --gtf /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf \
    --chrom NC_000913.3 \
    2>&1 | tee logs/ecoli_annotation_tsne.log

echo "[$(date)] Checking if Bacillus latent analysis is complete..."
if [ -f results/NC_000964.3/sae/20260324_151121_all_conf8.0_merged4of4/latent_analysis/data/cluster_assignments.tsv ]; then
    echo "[$(date)] Bacillus: Generating t-SNE plots colored by genomic annotation..."
    python tools/plot_tsne_by_annotation.py \
        --sae_run results/NC_000964.3/sae/20260324_151121_all_conf8.0_merged4of4 \
        --gtf /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf \
        --chrom NC_000964.3 \
        2>&1 | tee logs/bacillus_annotation_tsne.log
else
    echo "[$(date)] Bacillus latent analysis not yet complete, will run annotation plots later"
fi

echo "[$(date)] GTF annotation t-SNE plots COMPLETED"
