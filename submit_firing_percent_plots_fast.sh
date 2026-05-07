#!/bin/bash
#SBATCH -J firing_pct_fast_pn
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 6:00:00
#SBATCH -o logs/firing_pct_fast_prenorm_%j.log
#SBATCH -e logs/firing_pct_fast_prenorm_%j.err

# Fast variant: 5 thresholds instead of 14. Writes to firing_percent_fast/
# so it does NOT collide with the in-flight full-sweep jobs.
set -e
module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

THRESHOLDS="0,0.5,2,5,10"
CHUNK_ROWS=10000
PRENORM_DIR="results/_genome_wide/sae_tsne_prenorm/20260423_192324_23chroms_897129regions"

echo "[$(date)] === PRENORM fast: $PRENORM_DIR ==="
python tools/firing_percent_plots.py \
    --input_dir "$PRENORM_DIR" \
    --mode prenorm \
    --output_subdir firing_percent_fast \
    --thresholds "$THRESHOLDS" \
    --chunk_rows "$CHUNK_ROWS"

echo "[$(date)] Done successfully."
