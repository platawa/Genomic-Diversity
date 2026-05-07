#!/bin/bash
#SBATCH -J firing_pct_prenorm
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 10:00:00
#SBATCH -o logs/firing_pct_prenorm_%j.log
#SBATCH -e logs/firing_pct_prenorm_%j.err

# Genome-wide % SAE features firing plots — PRENORM (Option A, per-nt z-score).
# Streams combined_maxpooled.npy chunk-by-chunk; no re-embedding / re-clustering.
# Submit the raw counterpart separately via submit_firing_percent_plots_raw.sh.

set -e
module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

# Wide tau range: prenorm values are z-scores, signals can be >10 sigma.
THRESHOLDS="0,0.01,0.05,0.1,0.25,0.5,1,2,3,5,7,10,15,20"
CHUNK_ROWS=10000

PRENORM_DIR="results/_genome_wide/sae_tsne_prenorm/20260423_192324_23chroms_897129regions"

echo "[$(date)] === PRENORM: $PRENORM_DIR ==="
python tools/firing_percent_plots.py \
    --input_dir "$PRENORM_DIR" \
    --mode prenorm \
    --thresholds "$THRESHOLDS" \
    --chunk_rows "$CHUNK_ROWS"

echo "[$(date)] Done successfully."
