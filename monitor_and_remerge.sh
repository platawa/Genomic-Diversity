#!/usr/bin/env bash
# monitor_and_remerge.sh
#
# Monitors GPU shard jobs. When all sae_* jobs finish, submits merge jobs
# for all chromosomes that need it, then submits finish_merges.

set -euo pipefail
cd /orcd/data/zhang_f/001/platawa/jan31_files

module load miniforge/24.3.0-0
conda activate evo2_sep28

LOG="logs/monitor_remerge_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] Monitor started. Waiting for all sae_* GPU jobs to complete..."

# Phase 1: Wait for all sae_* and merge_* jobs to finish
while true; do
    N_SAE=$(squeue -u platawa --format="%j" -h | grep -c "^sae_" || true)
    N_MERGE=$(squeue -u platawa --format="%j" -h | grep -c "^merge_" || true)
    N_FINISH=$(squeue -u platawa --format="%j" -h | grep -c "^finish_" || true)
    echo "[$(date)] Active jobs: ${N_SAE} sae, ${N_MERGE} merge, ${N_FINISH} finish"

    if [ "$N_SAE" -eq 0 ] && [ "$N_MERGE" -eq 0 ] && [ "$N_FINISH" -eq 0 ]; then
        echo "[$(date)] All SAE, merge, and finish jobs done!"
        break
    fi
    sleep 120
done

# Phase 2: Submit merge jobs for all chromosomes that need it
echo "[$(date)] Phase 2: Checking which chromosomes need merging..."

NEED_MERGE=()
for CHR in chr{1..22} chrX chrY; do
    COMPLETED=$(ls results/${CHR}/sae/*merged*/COMPLETED 2>/dev/null | wc -l)
    if [ "$COMPLETED" -gt 0 ]; then
        LATEST_COMPLETED=$(ls -t results/${CHR}/sae/*merged*/COMPLETED 2>/dev/null | head -1)
        NEWER_SHARDS=$(find results/${CHR}/sae/ -maxdepth 1 -name "*shard*" -newer "$LATEST_COMPLETED" 2>/dev/null | wc -l)
        if [ "$NEWER_SHARDS" -gt 0 ]; then
            echo "  ${CHR}: has COMPLETED merge but ${NEWER_SHARDS} newer shards -> re-merge"
            NEED_MERGE+=("$CHR")
        else
            echo "  ${CHR}: merged and up-to-date, skipping"
        fi
    else
        N_SHARDS=$(ls -d results/${CHR}/sae/*shard* 2>/dev/null | wc -l)
        if [ "$N_SHARDS" -gt 0 ]; then
            echo "  ${CHR}: no completed merge, ${N_SHARDS} shards -> merge"
            NEED_MERGE+=("$CHR")
        else
            echo "  ${CHR}: no shard data, skipping"
        fi
    fi
done

if [ ${#NEED_MERGE[@]} -eq 0 ]; then
    echo "[$(date)] No chromosomes need merging!"
else
    echo "[$(date)] Chromosomes needing merge: ${NEED_MERGE[*]}"

    MERGE_JOBIDS=()
    for CHR in "${NEED_MERGE[@]}"; do
        JOBID=$(sbatch --parsable \
            --job-name="remerge_${CHR}" \
            --partition=mit_normal \
            --cpus-per-task=4 \
            --mem=128G \
            --time=8:00:00 \
            --output="logs/remerge_${CHR}_%j.out" \
            --error="logs/remerge_${CHR}_%j.err" \
            --wrap="module load miniforge/24.3.0-0 && conda activate evo2_sep28 && cd /orcd/data/zhang_f/001/platawa/jan31_files && python merge_sae_shards.py --chrom ${CHR} --output_dir results/ --include-partial")
        MERGE_JOBIDS+=("$JOBID")
        echo "  Submitted remerge_${CHR}: job ${JOBID}"
    done

    echo "[$(date)] Submitted ${#MERGE_JOBIDS[@]} merge jobs. Waiting..."

    # Phase 3: Wait for merge jobs
    while true; do
        N_REMERGE=$(squeue -u platawa --format="%j" -h | grep -c "^remerge_" || true)
        echo "[$(date)] Active remerge jobs: ${N_REMERGE}"
        if [ "$N_REMERGE" -eq 0 ]; then
            echo "[$(date)] All merge jobs done!"
            break
        fi
        sleep 60
    done
fi

# Phase 4: Run finish_merges on all chromosomes
echo "[$(date)] Phase 4: Running finish_merges for all chromosomes..."
FINISH_JOBID=$(sbatch --parsable \
    --job-name="final_finish" \
    --partition=mit_normal \
    --cpus-per-task=4 \
    --mem=128G \
    --time=8:00:00 \
    --output="logs/final_finish_%j.out" \
    --error="logs/final_finish_%j.err" \
    --wrap="module load miniforge/24.3.0-0 && conda activate evo2_sep28 && cd /orcd/data/zhang_f/001/platawa/jan31_files && python finish_merges.py --all_human --output_dir results/")
echo "  Submitted final_finish: job ${FINISH_JOBID}"

while true; do
    N_FINISH=$(squeue -u platawa --format="%j" -h | grep -c "^final_finish" || true)
    if [ "$N_FINISH" -eq 0 ]; then
        echo "[$(date)] finish_merges done!"
        break
    fi
    sleep 30
done

# Phase 5: Final status
echo ""
echo "============================================================"
echo "[$(date)] FINAL STATUS"
echo "============================================================"
echo ""
DONE=0
MISSING=0
for CHR in chr{1..22} chrX chrY; do
    COMPLETED=$(ls results/${CHR}/sae/*merged*/COMPLETED 2>/dev/null | wc -l)
    if [ "$COMPLETED" -gt 0 ]; then
        echo "  ${CHR}: DONE"
        DONE=$((DONE + 1))
    else
        echo "  ${CHR}: MISSING"
        MISSING=$((MISSING + 1))
    fi
done
echo ""
echo "Total completed: ${DONE} / 24"
if [ "$MISSING" -gt 0 ]; then
    echo "WARNING: ${MISSING} chromosomes still missing!"
fi
echo "============================================================"
echo "[$(date)] Monitor finished."
