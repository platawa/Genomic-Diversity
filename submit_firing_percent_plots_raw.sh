#!/bin/bash
#SBATCH -J firing_pct_raw
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 10:00:00
#SBATCH -o logs/firing_pct_raw_%j.log
#SBATCH -e logs/firing_pct_raw_%j.err

# Genome-wide % SAE features firing plots — RAW (un-normalized max-pooled).
# Streams combined_maxpooled.npy chunk-by-chunk; no re-embedding / re-clustering.

set -e
module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

# Wide tau range spanning raw activation magnitudes.
THRESHOLDS="0,0.01,0.05,0.1,0.25,0.5,1,2,3,5,7,10,15,20"
CHUNK_ROWS=10000

RAW_DIR="results/_genome_wide/sae_tsne/20260409_093547_23chroms_740967regions"

echo "[$(date)] === RAW: $RAW_DIR ==="
python tools/firing_percent_plots.py \
    --input_dir "$RAW_DIR" \
    --mode raw \
    --thresholds "$THRESHOLDS" \
    --chunk_rows "$CHUNK_ROWS"

echo "[$(date)] Done successfully."
