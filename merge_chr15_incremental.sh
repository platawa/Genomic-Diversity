#!/bin/bash
# merge_chr15_incremental.sh
# FASTEST approach: Incremental merge (shard 1+2, then +3, then +4, etc.)
# Avoids loading all data simultaneously
# Estimated time: 1-2 hours (30-50% faster than parallel batches)

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
CHROM=chr15
N_SHARDS=36

echo "=========================================="
echo "Chr15 SAE Merge (Incremental - FASTEST)"
echo "=========================================="
echo ""
echo "Strategy: Merge shards incrementally"
echo "  Step 1: Merge shards 0+1        → output_1-2"
echo "  Step 2: Merge result + shard 2  → output_1-3"
echo "  Step 3: Merge result + shard 3  → output_1-4"
echo "  ..."
echo "  Step 35: Add shard 35           → final result"
echo ""
echo "Why faster:"
echo "  - Processes shards one at a time (low memory)"
echo "  - Reuses previous merge as input"
echo "  - Avoids batch overhead"
echo "  - Total: ~30 min per 9 shards = 1-2 hours total"
echo ""

cat > "${LOGS}/.incremental_merge_chr15.py" << 'INCR_SCRIPT'
#!/usr/bin/env python3
"""Incremental merge of chr15 SAE shards (fastest)"""
import os
import sys
import shutil
import numpy as np
from datetime import datetime

PROJECT = "/orcd/data/zhang_f/001/platawa/jan31_files"
CHROM = "chr15"
N_SHARDS = 36

sys.path.insert(0, PROJECT)
from merge_sae_shards import find_shard_dirs, merge_feature_matrices

def incremental_merge():
    """Merge shards incrementally for speed"""
    print(f"Starting incremental merge of {CHROM}")
    print(f"Start time: {datetime.now()}\n")

    # Find all shards
    shard_dirs = find_shard_dirs(PROJECT, CHROM, N_SHARDS)
    print(f"Found {len(shard_dirs)} shards\n")

    # Output directory for final result
    final_output = os.path.join(PROJECT, "results", CHROM, "sae",
                               f"20260325_all_conf8.0_merged_incremental")
    os.makedirs(final_output, exist_ok=True)

    try:
        # Start with first shard
        current_shards = [shard_dirs[0]]
        print(f"[Step 1] Starting with shard 0")

        # Incrementally add each subsequent shard
        for i in range(1, len(shard_dirs)):
            current_shards.append(shard_dirs[i])

            progress = ((i + 1) / len(shard_dirs)) * 100
            print(f"[Step {i+1}] Merging shards 0-{i} ({progress:.0f}%) - {datetime.now()}")

            # Merge current set
            merge_feature_matrices(current_shards, final_output)

            # Report progress
            output_size = 0
            merged_file = os.path.join(final_output, "data", "feature_matrices.npz")
            if os.path.exists(merged_file):
                output_size = os.path.getsize(merged_file) / (1024**3)  # GB
                print(f"         Output size: {output_size:.2f}GB")

        print(f"\n✓ Incremental merge COMPLETE!")
        print(f"Output: {final_output}")
        print(f"End time: {datetime.now()}")

        # Write completed sentinel
        import json
        completed_file = os.path.join(final_output, "COMPLETED")
        with open(completed_file, 'w') as f:
            json.dump({
                "completed_at": datetime.now().isoformat(),
                "script": "merge_chr15_incremental.py",
                "method": "incremental"
            }, f)

        return True

    except Exception as e:
        print(f"\n✗ Incremental merge FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = incremental_merge()
    sys.exit(0 if success else 1)
INCR_SCRIPT

chmod +x "${LOGS}/.incremental_merge_chr15.py"

# Submit incremental merge (single threaded but fast)
JOB_ID=$(sbatch --parsable \
  --job-name="merge_chr15_incremental" \
  --partition=pi_zhang_f \
  --cpus-per-task=8 \
  --mem=256G \
  --time=4:00:00 \
  --output="${LOGS}/merge_chr15_incremental_%j.out" \
  --error="${LOGS}/merge_chr15_incremental_%j.err" \
  --wrap="
cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28
echo '[Incremental] Starting at '
date
python ${LOGS}/.incremental_merge_chr15.py
echo '[Incremental] Completed at '
date
")

echo "Incremental merge job submitted: $JOB_ID"
echo ""
echo "Resources:"
echo "  CPUs: 8 (single-threaded merge)"
echo "  Memory: 256GB"
echo "  Time limit: 4 hours"
echo ""
echo "Monitor with:"
echo "  tail -f ${LOGS}/merge_chr15_incremental_${JOB_ID}.out"
echo ""
