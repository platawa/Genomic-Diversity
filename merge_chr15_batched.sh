#!/bin/bash
# merge_chr15_batched.sh
# Alternative: Merge chr15 in smaller batches (may be faster/more stable)
# Process 9 shards at a time instead of all 36

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
CHROM=chr15
N_SHARDS=36
BATCH_SIZE=9

echo "=========================================="
echo "Chr15 SAE Merge (Batched Alternative)"
echo "=========================================="
echo ""
echo "Strategy: Merge in ${BATCH_SIZE}-shard batches"
echo "  Batch 1: shards 0-8"
echo "  Batch 2: shards 9-17"
echo "  Batch 3: shards 18-26"
echo "  Batch 4: shards 27-35"
echo "  Final: Merge 4 batch results"
echo ""
echo "Time estimate: 2-3 hours per batch × 4 = 8-12 hours total"
echo ""

# Create batch merge script
cat > "${LOGS}/.batch_merge_chr15.py" << 'BATCH_SCRIPT'
#!/usr/bin/env python3
"""Batched merge of chr15 SAE shards"""
import os
import sys
import glob
import shutil
import numpy as np
from datetime import datetime

PROJECT = "/orcd/data/zhang_f/001/platawa/jan31_files"
CHROM = "chr15"
BATCH_SIZE = 9
N_SHARDS = 36

sys.path.insert(0, PROJECT)
from merge_sae_shards import find_shard_dirs, merge_feature_matrices, write_completed

def merge_batch(batch_num, start_idx, end_idx):
    """Merge one batch of shards"""
    print(f"\n[Batch {batch_num}] Processing shards {start_idx}-{end_idx}...")
    print(f"[Batch {batch_num}] Start: {datetime.now()}")

    # Find shards in this batch
    sae_root = os.path.join(PROJECT, "results", CHROM, "sae")
    shard_dirs = find_shard_dirs(PROJECT, CHROM, N_SHARDS)

    batch_shards = [d for d in shard_dirs if start_idx <= int(os.path.basename(d).split('shard')[1].split('of')[0]) <= end_idx]
    print(f"[Batch {batch_num}] Found {len(batch_shards)} shards")

    if not batch_shards:
        print(f"[Batch {batch_num}] ERROR: No shards found!")
        return False

    # Merge this batch
    output_dir = os.path.join(PROJECT, "results", CHROM, "sae", f"batch{batch_num}_merged")
    os.makedirs(output_dir, exist_ok=True)

    try:
        merge_feature_matrices(batch_shards, output_dir)
        write_completed(output_dir)
        print(f"[Batch {batch_num}] SUCCESS: {datetime.now()}")
        return True
    except Exception as e:
        print(f"[Batch {batch_num}] ERROR: {e}")
        return False

if __name__ == "__main__":
    print(f"Starting batched merge of {CHROM}")

    results = {}
    for batch in range(1, 5):
        start = (batch - 1) * BATCH_SIZE
        end = min(start + BATCH_SIZE - 1, N_SHARDS - 1)
        results[batch] = merge_batch(batch, start, end)

    print(f"\n\nBatch Results:")
    for batch, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  Batch {batch}: {status}")

    all_success = all(results.values())
    if all_success:
        print(f"\nAll batches merged! Now merge batch results together...")
    else:
        print(f"\nSome batches failed. Fix and retry.")

    sys.exit(0 if all_success else 1)
BATCH_SCRIPT

chmod +x "${LOGS}/.batch_merge_chr15.py"

# Submit batched merge job
JOB_ID=$(sbatch --parsable \
  --job-name="merge_chr15_batched" \
  --partition=pi_zhang_f \
  --cpus-per-task=8 \
  --mem=256G \
  --time=24:00:00 \
  --output="${LOGS}/merge_chr15_batched_%j.out" \
  --error="${LOGS}/merge_chr15_batched_%j.err" \
  --wrap="
cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28
python ${LOGS}/.batch_merge_chr15.py
")

echo "Batched merge job submitted: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  tail -f ${LOGS}/merge_chr15_batched_${JOB_ID}.out"
echo ""
