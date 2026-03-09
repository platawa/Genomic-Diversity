#!/bin/bash
#SBATCH --job-name=ssrA_score
#SBATCH --output=logs/ssrA_score.out
#SBATCH --error=logs/ssrA_score.err
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

# Run ssrA (tmRNA) scoring
cd /orcd/data/zhang_f/001/platawa/jan22_files
python genome_scoring_jan22.py --organism ecoli --gene_id b2621
