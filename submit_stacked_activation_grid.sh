#!/bin/bash
#SBATCH -J stacked_act_grid
#SBATCH -p mit_preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o logs/stacked_act_grid_%j.out
#SBATCH -e logs/stacked_act_grid_%j.err
set -euo pipefail
cd /orcd/data/zhang_f/001/platawa/jan31_files
module load miniforge/24.3.0-0
conda activate evo2_sep28

OUT=results/_genome_wide/normalization_distribution/stacked_grid_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

python tools/stacked_chrom_activation_grid.py \
    --results_dir results/ \
    --output_dir "$OUT"
