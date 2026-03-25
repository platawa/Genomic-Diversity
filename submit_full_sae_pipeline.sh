#!/usr/bin/env bash
# submit_full_sae_pipeline.sh
#
# Automated end-to-end SAE pipeline for ALL human drop regions (conf >= 0.0).
# Scoring is already done. This handles:
#   Phase 1: SAE extraction (GPU, sharded)
#   Phase 2: Merge shards (CPU)
#   Phase 3: Finish merges / norm stats (CPU)
#   Phase 4: Verify completeness, resubmit failures
#
# Usage:
#   bash submit_full_sae_pipeline.sh [min_confidence] [n_shards_per_chrom]
#   bash submit_full_sae_pipeline.sh 0.0 8     # all regions, 8 shards each
#   bash submit_full_sae_pipeline.sh 4.0 8     # conf >= 4.0, 8 shards each
#
# After completion, run downstream analysis (CPU-only):
#   python tools/chromosome_analysis.py ...
#   python tools/genome_sae_tsne.py ...
# ---------------------------------------------------------------------------
set -eo pipefail

MIN_CONF="${1:-0.0}"
N_SHARDS="${2:-8}"

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
FASTA=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna
GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf
LOGS=${PROJECT}/logs

CHROMS="chrY chr22 chr21 chr20 chr19 chr18 chr17 chr16 chr15 chr14 chr13 chr12 chr11 chr10 chr9 chr8 chr7 chr6 chr5 chr4 chr3 chr2 chr1 chrX"

mkdir -p "${LOGS}"

echo "============================================================"
echo "SAE Full Pipeline: conf >= ${MIN_CONF}, ${N_SHARDS} shards/chrom"
echo "============================================================"
echo "Total jobs: $((24 * N_SHARDS)) SAE extraction + 1 monitor"
echo ""

# Phase 1: Submit SAE extraction jobs
# Split across partitions to maximize throughput
echo "Phase 1: Submitting SAE extraction jobs..."
SAE_JOBIDS=""

for CHROM in $CHROMS; do
    for (( IDX=0; IDX<N_SHARDS; IDX++ )); do
        JOB_NAME="saeall_${CHROM}_s${IDX}"

        # Partition assignment: spread across available GPU partitions
        # Use ou_bcs_low (64 GPU limit) as primary, overflow to ou_bcs_normal (16)
        if (( IDX % 3 == 0 )); then
            PARTITION="ou_bcs_normal"
            GPU_SPEC="gpu:h100:1"
        else
            PARTITION="ou_bcs_low"
            GPU_SPEC="gpu:1"
        fi

        JOBID=$(sbatch --parsable \
            --job-name="${JOB_NAME}" \
            --partition="${PARTITION}" \
            --gres="${GPU_SPEC}" \
            --cpus-per-task=8 \
            --mem=200G \
            --time=12:00:00 \
            --output="${LOGS}/${JOB_NAME}_%j.out" \
            --error="${LOGS}/${JOB_NAME}_%j.err" \
            --requeue \
            --wrap="module load miniforge/24.3.0-0 && conda activate evo2_sep28 && cd ${PROJECT} && python run_sae_fast.py --auto --chrom ${CHROM} --fasta ${FASTA} --gtf ${GTF} --output_dir results/ --shard ${IDX}/${N_SHARDS} --min_confidence ${MIN_CONF} --batch_size 16 --padding 200 --checkpoint_interval 500 --extract_only --skip_notebook")
        SAE_JOBIDS="${SAE_JOBIDS} ${JOBID}"
        echo "  ${JOB_NAME}: job ${JOBID} (${PARTITION})"
    done
done

echo ""
echo "Submitted $((24 * N_SHARDS)) SAE extraction jobs."

# Phase 2: Submit monitor that auto-triggers merge + finish after SAE completes
echo ""
echo "Phase 2: Submitting auto-monitor..."

cat > "${LOGS}/.saeall_monitor.sh" <<'MONITOR'
#!/usr/bin/env bash
set -eo pipefail
cd /orcd/data/zhang_f/001/platawa/jan31_files
module load miniforge/24.3.0-0
conda activate evo2_sep28

MIN_CONF="$1"
N_SHARDS="$2"

LOG="logs/saeall_monitor_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] Monitor started for conf >= ${MIN_CONF}, ${N_SHARDS} shards"

# ---- Wait for all SAE extraction jobs to finish ----
while true; do
    N=$(squeue -u platawa --format="%j" -h | grep -c "^saeall_" || true)
    echo "[$(date)] SAE extraction jobs remaining: ${N}"
    if [ "$N" -eq 0 ]; then
        echo "[$(date)] All SAE extraction jobs done!"
        break
    fi
    sleep 120
done

# ---- Check for failures and resubmit (up to 2 rounds) ----
for ROUND in 1 2; do
    echo "[$(date)] Failure check round ${ROUND}..."
    NEED_RESUBMIT=0

    for CHR in chr{1..22} chrX chrY; do
        for (( S=0; S<N_SHARDS; S++ )); do
            # Find shard dir
            DIR=$(ls -d results/${CHR}/sae/*_conf${MIN_CONF}_shard${S}of${N_SHARDS} 2>/dev/null | tail -1)
            if [ -z "$DIR" ]; then
                echo "  ${CHR} shard ${S}: NO DIR — needs resubmit"
                NEED_RESUBMIT=$((NEED_RESUBMIT + 1))
                sbatch --parsable \
                    --job-name="saeall_${CHR}_s${S}" \
                    --partition=ou_bcs_low \
                    --gres=gpu:1 \
                    --cpus-per-task=8 \
                    --mem=200G \
                    --time=12:00:00 \
                    --output="logs/saeall_${CHR}_s${S}_%j.out" \
                    --error="logs/saeall_${CHR}_s${S}_%j.err" \
                    --requeue \
                    --wrap="module load miniforge/24.3.0-0 && conda activate evo2_sep28 && cd /orcd/data/zhang_f/001/platawa/jan31_files && python run_sae_fast.py --auto --chrom ${CHR} --fasta /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna --gtf /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf --output_dir results/ --shard ${S}/${N_SHARDS} --min_confidence ${MIN_CONF} --batch_size 16 --padding 200 --checkpoint_interval 500 --extract_only --skip_notebook"
                continue
            fi
            # Check for chunk data
            CHUNKS=$(ls "$DIR"/data/_chunk_*.npz 2>/dev/null | wc -l)
            COMPLETED=$(ls "$DIR"/COMPLETED 2>/dev/null | wc -l)
            if [ "$COMPLETED" -eq 0 ] && [ "$CHUNKS" -eq 0 ]; then
                echo "  ${CHR} shard ${S}: no data — needs resubmit"
                NEED_RESUBMIT=$((NEED_RESUBMIT + 1))
                sbatch --parsable \
                    --job-name="saeall_${CHR}_s${S}" \
                    --partition=ou_bcs_low \
                    --gres=gpu:1 \
                    --cpus-per-task=8 \
                    --mem=200G \
                    --time=12:00:00 \
                    --output="logs/saeall_${CHR}_s${S}_%j.out" \
                    --error="logs/saeall_${CHR}_s${S}_%j.err" \
                    --requeue \
                    --wrap="module load miniforge/24.3.0-0 && conda activate evo2_sep28 && cd /orcd/data/zhang_f/001/platawa/jan31_files && python run_sae_fast.py --auto --chrom ${CHR} --fasta /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna --gtf /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf --output_dir results/ --shard ${S}/${N_SHARDS} --min_confidence ${MIN_CONF} --batch_size 16 --padding 200 --checkpoint_interval 500 --extract_only --skip_notebook"
            fi
        done
    done

    if [ "$NEED_RESUBMIT" -eq 0 ]; then
        echo "[$(date)] All shards have data!"
        break
    fi

    echo "[$(date)] Resubmitted ${NEED_RESUBMIT} shards. Waiting..."
    while true; do
        N=$(squeue -u platawa --format="%j" -h | grep -c "^saeall_" || true)
        if [ "$N" -eq 0 ]; then break; fi
        sleep 120
    done
done

# ---- Merge all chromosomes ----
echo "[$(date)] Submitting merge jobs..."
for CHR in chr{1..22} chrX chrY; do
    JOBID=$(sbatch --parsable \
        --job-name="mergeall_${CHR}" \
        --partition=mit_normal \
        --cpus-per-task=4 \
        --mem=200G \
        --time=12:00:00 \
        --output="logs/mergeall_${CHR}_%j.out" \
        --error="logs/mergeall_${CHR}_%j.err" \
        --wrap="module load miniforge/24.3.0-0 && conda activate evo2_sep28 && cd /orcd/data/zhang_f/001/platawa/jan31_files && python merge_sae_shards.py --chrom ${CHR} --n_shards ${N_SHARDS} --output_dir results/ --include-partial")
    echo "  mergeall_${CHR}: job ${JOBID}"
done

echo "[$(date)] Waiting for merge jobs..."
while true; do
    N=$(squeue -u platawa --format="%j" -h | grep -c "^mergeall_" || true)
    echo "[$(date)] Active merge jobs: ${N}"
    if [ "$N" -eq 0 ]; then break; fi
    sleep 60
done

# ---- Finish merges (norm stats + COMPLETED) ----
echo "[$(date)] Running finish_merges..."
JOBID=$(sbatch --parsable \
    --job-name="finishall" \
    --partition=mit_normal \
    --cpus-per-task=4 \
    --mem=200G \
    --time=12:00:00 \
    --output="logs/finishall_%j.out" \
    --error="logs/finishall_%j.err" \
    --wrap="module load miniforge/24.3.0-0 && conda activate evo2_sep28 && cd /orcd/data/zhang_f/001/platawa/jan31_files && python finish_merges.py --all_human --output_dir results/")
echo "  finishall: job ${JOBID}"

while true; do
    N=$(squeue -u platawa --format="%j" -h | grep -c "^finishall" || true)
    if [ "$N" -eq 0 ]; then break; fi
    sleep 30
done

# ---- Final status ----
echo ""
echo "============================================================"
echo "[$(date)] FINAL STATUS — conf >= ${MIN_CONF}"
echo "============================================================"
DONE=0
for CHR in chr{1..22} chrX chrY; do
    C=$(ls results/${CHR}/sae/*merged*/COMPLETED 2>/dev/null | wc -l)
    if [ "$C" -gt 0 ]; then
        echo "  ${CHR}: DONE"
        DONE=$((DONE + 1))
    else
        echo "  ${CHR}: MISSING"
    fi
done
echo ""
echo "Completed: ${DONE} / 24"
echo "============================================================"
echo "[$(date)] Pipeline finished."
MONITOR

MONITOR_JOBID=$(sbatch --parsable \
    --job-name="saeall_monitor" \
    --partition=mit_normal \
    --cpus-per-task=1 \
    --mem=4G \
    --time=4-00:00:00 \
    --output="${LOGS}/saeall_monitor_%j.out" \
    --error="${LOGS}/saeall_monitor_%j.err" \
    --wrap="bash ${LOGS}/.saeall_monitor.sh ${MIN_CONF} ${N_SHARDS}")

echo "Monitor job: ${MONITOR_JOBID} (4 day time limit)"
echo ""
echo "============================================================"
echo "SUBMITTED SUCCESSFULLY"
echo "============================================================"
echo ""
echo "Pipeline will run automatically:"
echo "  1. SAE extraction: $((24 * N_SHARDS)) GPU jobs (conf >= ${MIN_CONF})"
echo "  2. Monitor detects completion, resubmits any failures"
echo "  3. Auto-merges all chromosomes"
echo "  4. Auto-runs finish_merges (norm stats + COMPLETED)"
echo ""
echo "Monitor progress:"
echo "  squeue -u platawa | grep saeall"
echo "  tail -f ${LOGS}/saeall_monitor_*.out"
echo ""
echo "After pipeline completes, run analysis (CPU-only):"
echo "  python tools/chromosome_analysis.py --all_human --results_dir results/"
echo "  python tools/genome_sae_tsne.py --all_human --results_dir results/"
