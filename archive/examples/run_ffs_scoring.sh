#!/bin/bash
#SBATCH --job-name=ffs_score
#SBATCH --output=logs/ffs_score.out
#SBATCH --error=logs/ffs_score.err
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

# Run ffs (SRP RNA / 4.5S RNA) scoring
# Gene: ffs (b0455)
# Coordinates: NC_000913.3:476448-476561 (+), 114 nt
cd /orcd/data/zhang_f/001/platawa/jan22_files
python genome_scoring_jan22.py --organism ecoli --gene_id b0455
