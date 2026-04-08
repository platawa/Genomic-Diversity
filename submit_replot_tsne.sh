#!/bin/bash
#SBATCH -J replot_tsne_fast
#SBATCH -p mit_normal
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o replot_tsne_fast_%j.out
#SBATCH -e replot_tsne_fast_%j.err

module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

python replot_tsne_only.py
