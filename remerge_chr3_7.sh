#!/bin/bash
# Remerge chr3-7 sequentially (runs after chr2 completes)

set -e

cd /orcd/data/zhang_f/001/platawa/jan31_files
module load miniforge/24.3.0-0
conda activate evo2_sep28

for chr in chr3 chr4 chr5 chr6 chr7; do
  echo ""
  echo "=========================================="
  echo "Starting $chr merge at $(date)"
  echo "=========================================="

  python merge_sae_shards.py --chrom $chr --n_shards 36 --output_dir results/

  echo ""
  echo "Verifying $chr NPZ integrity..."
  latest=$(ls -td results/$chr/sae/20260326*/data/feature_matrices.npz 2>/dev/null | head -1)
  if [ -f "$latest" ]; then
    if unzip -t "$latest" > /dev/null 2>&1; then
      echo "✓ $chr NPZ is valid and readable"
    else
      echo "✗ ERROR: $chr NPZ is corrupted!"
      exit 1
    fi
  else
    echo "✗ ERROR: No NPZ file found for $chr"
    exit 1
  fi

  echo "$chr merge COMPLETE at $(date)"
done

echo ""
echo "=========================================="
echo "All merges (chr3-7) completed successfully!"
echo "=========================================="
