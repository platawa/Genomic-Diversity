#!/bin/bash
#SBATCH -J sae_chr22
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH -t 1:00:00
#SBATCH -o logs/sae_chr22_%j.out
#SBATCH -e logs/sae_chr22_%j.err

cd /orcd/data/zhang_f/001/platawa/jan31_files/
module load miniforge/24.3.0-0
conda activate evo2_sep28

python run_sae_on_chromosome_drops.py \
    --boundaries chromosome_scores/chr22_full.drop_boundaries.tsv \
    --entropy chromosome_scores/chr22_full.entropy.npz \
    --chrom chr22 \
    --fasta /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna \
    --max_regions 50 \
    --min_confidence 8.0 \
    --output_dir sae_chromosome_results
