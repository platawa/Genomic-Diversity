#!/bin/bash
#SBATCH -J sae_tools
#SBATCH -t 1:00:00
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -o results/sae_tools_%j.out
#SBATCH -e results/sae_tools_%j.err

module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

FASTA=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna
GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf

echo "=== discover_region_features.py ==="
python tools/discover_region_features.py \
  --fasta $FASTA --chrom NC_000913.3 \
  --target_start 2877618 --target_end 2878569 \
  --bg_from_gtf --bg_flank 5000 --gtf $GTF \
  --chrom_name ecoli_K12 --output_dir results

echo "=== scan_feature_genome.py ==="
python tools/scan_feature_genome.py \
  --fasta $FASTA --chrom NC_000913.3 \
  --features 19745 --gtf $GTF \
  --chrom_name ecoli_K12 --output_dir results
