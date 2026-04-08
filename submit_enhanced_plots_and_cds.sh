#!/bin/bash
#SBATCH --job-name=enhanced_plots
#SBATCH --output=logs/enhanced_plots_%j.out
#SBATCH --error=logs/enhanced_plots_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Run enhanced latent plots (Tasks 1 & 2) and bacteria CDS investigation (Task 3)
# No GPU needed — CPU-only matplotlib/sklearn work.

cd /orcd/data/zhang_f/001/platawa/jan31_files
module load miniforge/24.3.0-0
set +u
conda activate evo2_sep28
set -eo pipefail

mkdir -p logs

ECOLI_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf"
BACILLUS_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf"
HUMAN_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"

echo "=========================================="
echo "Task 1 & 2: Enhanced latent plots (distance-to-gene + entropy coloring)"
echo "=========================================="

# E. coli
echo "--- E. coli: distance_to_gene + entropy_color ---"
python tools/enhanced_latent_plots.py \
    --scope organism --organism ecoli \
    --gtf "$ECOLI_GTF" \
    --results_dir results/ \
    --plots distance_to_gene,entropy_color \
    --log_level INFO || echo "E. coli enhanced plots failed (may lack data)"

# Bacillus
echo "--- Bacillus: distance_to_gene + entropy_color ---"
python tools/enhanced_latent_plots.py \
    --scope organism --organism bacillus \
    --gtf "$BACILLUS_GTF" \
    --results_dir results/ \
    --plots distance_to_gene,entropy_color \
    --log_level INFO || echo "Bacillus enhanced plots failed (may lack data)"

# Human chr22 (quick test)
echo "--- Human chr22: distance_to_gene + entropy_color ---"
python tools/enhanced_latent_plots.py \
    --scope chromosome --chrom chr22 --organism human \
    --gtf "$HUMAN_GTF" \
    --results_dir results/ \
    --plots distance_to_gene,entropy_color \
    --log_level INFO || echo "Human chr22 enhanced plots failed (may lack data)"

echo ""
echo "=========================================="
echo "Task 3: Bacteria CDS cluster investigation"
echo "=========================================="

# E. coli CDS clusters
echo "--- E. coli CDS clusters ---"
python tools/investigate_bacteria_cds_clusters.py \
    --chrom NC_000913.3 --organism ecoli \
    --gtf "$ECOLI_GTF" \
    --results_dir results/ \
    --n_clusters 2 --n_examples 3 --top_k 5 || echo "E. coli CDS investigation failed"

# Bacillus CDS clusters
echo "--- Bacillus CDS clusters ---"
python tools/investigate_bacteria_cds_clusters.py \
    --chrom NC_000964.3 --organism bacillus \
    --gtf "$BACILLUS_GTF" \
    --results_dir results/ \
    --n_clusters 2 --n_examples 3 --top_k 5 || echo "Bacillus CDS investigation failed"

echo ""
echo "=========================================="
echo "All tasks complete."
echo "=========================================="
