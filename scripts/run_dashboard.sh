#!/bin/bash
#SBATCH -J dashboard_chr22
#SBATCH -p mit_preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -o logs/dashboard_chr22_%j.out
#SBATCH -e logs/dashboard_chr22_%j.err

cd /orcd/data/zhang_f/001/platawa/jan31_files/
module load miniforge/24.3.0-0
conda activate evo2_sep28

python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_full --all_plots
