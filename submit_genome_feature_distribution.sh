#!/bin/bash
#SBATCH -J gfd_all
#SBATCH -p mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH -t 12:00:00
#SBATCH -o logs/gfd_all_%j.out
#SBATCH -e logs/gfd_all_%j.err
set -eo pipefail   # omit -u: conda activate references unbound vars
cd /orcd/data/zhang_f/001/platawa/jan31_files
module load miniforge/24.3.0-0
conda activate evo2_sep28

TS=$(date +%Y%m%d_%H%M%S)
OUT=results/_genome_wide/feature_distribution/${TS}_all_entities
mkdir -p "$OUT"

python tools/genome_feature_distribution.py \
    --results_dir results/ \
    --output_dir "$OUT"

python tools/genome_feature_distribution_plots.py \
    --stats_npz "$OUT/stats.npz" \
    --output_dir "$OUT/plots"

echo "Done: $OUT"
