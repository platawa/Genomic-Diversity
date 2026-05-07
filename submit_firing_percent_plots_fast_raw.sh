#!/bin/bash
#SBATCH -J firing_pct_fast_raw
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 6:00:00
#SBATCH -o logs/firing_pct_fast_raw_%j.log
#SBATCH -e logs/firing_pct_fast_raw_%j.err

# Fast variant (raw): 5 thresholds, separate output subdir.
set -e
module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

THRESHOLDS="0,0.5,2,5,10"
CHUNK_ROWS=10000
RAW_DIR="results/_genome_wide/sae_tsne/20260409_093547_23chroms_740967regions"

echo "[$(date)] === RAW fast: $RAW_DIR ==="
python tools/firing_percent_plots.py \
    --input_dir "$RAW_DIR" \
    --mode raw \
    --output_subdir firing_percent_fast \
    --thresholds "$THRESHOLDS" \
    --chunk_rows "$CHUNK_ROWS"

echo "[$(date)] Done successfully."
