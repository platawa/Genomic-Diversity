#!/bin/bash
#SBATCH --job-name=ssrS_score
#SBATCH --output=logs/ssrS_score.out
#SBATCH --error=logs/ssrS_score.err
#SBATCH --time=2:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

# Create log directory
mkdir -p logs

# Load environment
module load miniforge/24.3.0-0
source ~/.bashrc
conda activate evo2_sep28

# Run ssrS (6S RNA) scoring
# Gene: ssrS (b2911)
# Coordinates: NC_000913.3:3055983-3056165 (+), 183 nt
cd /orcd/data/zhang_f/001/platawa/jan22_files
python genome_scoring_jan22.py --organism ecoli --gene_id b2911
