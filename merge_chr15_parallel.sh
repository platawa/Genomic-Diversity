#!/bin/bash
# merge_chr15_parallel.sh
# Merge chr15 in parallel batches (FASTEST approach)
# All 4 batches run simultaneously, then merge results
# Total time: ~3-4 hours (vs 8-12 for sequential)

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
CHROM=chr15
N_SHARDS=36
BATCH_SIZE=9

echo "=========================================="
echo "Chr15 SAE Merge (Parallel Batches)"
echo "=========================================="
echo ""
echo "Strategy: Run all 4 batches IN PARALLEL"
echo "  Batch 1: shards 0-8   (parallel job 1)"
echo "  Batch 2: shards 9-17  (parallel job 2)"
echo "  Batch 3: shards 18-26 (parallel job 3)"
echo "  Batch 4: shards 27-35 (parallel job 4)"
echo ""
echo "Timeline:"
echo "  All 4 batches run together: 2-3 hours"
echo "  Merge 4 batch results:      1 hour"
echo "  TOTAL: 3-4 hours (FASTEST)"
echo ""

# Create parallel batch script
cat > "${LOGS}/.parallel_merge_chr15.py" << 'PARALLEL_SCRIPT'
#!/usr/bin/env python3
"""Parallel merge of chr15 SAE shards"""
import os
import sys
import subprocess
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT = "/orcd/data/zhang_f/001/platawa/jan31_files"
CHROM = "chr15"
BATCH_SIZE = 9
N_SHARDS = 36
MAX_WORKERS = 4  # Run 4 batches in parallel

sys.path.insert(0, PROJECT)
from merge_sae_shards import find_shard_dirs, merge_feature_matrices, write_completed

def merge_batch_worker(batch_num, start_idx, end_idx):
    """Worker function to merge one batch (runs in parallel)"""
    try:
        print(f"[Batch {batch_num}] START: {datetime.now()}")

        # Find shards for this batch
        sae_root = os.path.join(PROJECT, "results", CHROM, "sae")
        shard_dirs = find_shard_dirs(PROJECT, CHROM, N_SHARDS)

        batch_shards = [d for d in shard_dirs
                       if start_idx <= int(os.path.basename(d).split('shard')[1].split('of')[0]) <= end_idx]

        print(f"[Batch {batch_num}] Found {len(batch_shards)} shards")

        if not batch_shards:
            print(f"[Batch {batch_num}] ERROR: No shards found!")
            return (batch_num, False, "No shards found")

        # Merge this batch
        output_dir = os.path.join(PROJECT, "results", CHROM, "sae",
                                 f"20260325_parallel_{batch_num:02d}_merged")
        os.makedirs(output_dir, exist_ok=True)

        merge_feature_matrices(batch_shards, output_dir)
        write_completed(output_dir)

        print(f"[Batch {batch_num}] SUCCESS: {datetime.now()}")
        return (batch_num, True, output_dir)

    except Exception as e:
        print(f"[Batch {batch_num}] FAILED: {e}")
        return (batch_num, False, str(e))

def merge_final_results(batch_outputs):
    """Merge the 4 batch results into final output"""
    print(f"\n[Final] Merging 4 batch results...")
    print(f"[Final] START: {datetime.now()}")

    try:
        final_output = os.path.join(PROJECT, "results", CHROM, "sae",
                                   f"20260325_all_conf8.0_merged_final")
        os.makedirs(final_output, exist_ok=True)

        # Batch outputs are directories with feature_matrices.npz
        batch_dirs = [output for _, success, output in batch_outputs if success]

        if len(batch_dirs) < 4:
            print(f"[Final] ERROR: Only {len(batch_dirs)}/4 batches succeeded!")
            return False

        # Merge batch results (same process as merging shards)
        merge_feature_matrices(batch_dirs, final_output)
        write_completed(final_output)

        print(f"[Final] SUCCESS: {datetime.now()}")
        print(f"[Final] Output: {final_output}")
        return True

    except Exception as e:
        print(f"[Final] FAILED: {e}")
        return False

if __name__ == "__main__":
    print(f"Starting PARALLEL merge of {CHROM}")
    print(f"Total workers: {MAX_WORKERS}\n")

    start_time = datetime.now()

    # Submit all 4 batches to parallel executor
    batch_configs = [
        (1, 0, 8),
        (2, 9, 17),
        (3, 18, 26),
        (4, 27, 35),
    ]

    batch_outputs = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all batches
        futures = {executor.submit(merge_batch_worker, batch, start, end): batch
                   for batch, start, end in batch_configs}

        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            batch_outputs.append(result)
            batch_num, success, output = result
            status = "✓" if success else "✗"
            print(f"[Batch {batch_num}] Completed: {status}\n")

    print("\n" + "="*50)
    print("All batches completed!")
    print("="*50)

    # Print summary
    print("\nBatch Results:")
    for batch_num, success, output in sorted(batch_outputs):
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  Batch {batch_num}: {status}")

    # Merge final results if all batches succeeded
    all_success = all(success for _, success, _ in batch_outputs)
    if all_success:
        final_success = merge_final_results(batch_outputs)
        if final_success:
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"\n✓ COMPLETE! Total time: {elapsed:.1f} minutes")
            sys.exit(0)

    print(f"\n✗ Some batches failed. Check logs above.")
    sys.exit(1)
PARALLEL_SCRIPT

chmod +x "${LOGS}/.parallel_merge_chr15.py"

# Submit parallel merge job (single job, but runs 4 batch processes inside)
JOB_ID=$(sbatch --parsable \
  --job-name="merge_chr15_parallel" \
  --partition=pi_zhang_f \
  --cpus-per-task=32 \
  --mem=512G \
  --time=6:00:00 \
  --output="${LOGS}/merge_chr15_parallel_%j.out" \
  --error="${LOGS}/merge_chr15_parallel_%j.err" \
  --wrap="
cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28
echo 'Starting parallel batch merge at '
date
python ${LOGS}/.parallel_merge_chr15.py
echo 'Parallel merge completed at '
date
")

echo "Parallel merge job submitted: $JOB_ID"
echo ""
echo "Resources allocated:"
echo "  CPUs: 32 (for 4 parallel workers × 8 CPUs each)"
echo "  Memory: 512GB (for parallel processing)"
echo "  Time limit: 6 hours"
echo ""
echo "Monitor with:"
echo "  tail -f ${LOGS}/merge_chr15_parallel_${JOB_ID}.out"
echo ""
