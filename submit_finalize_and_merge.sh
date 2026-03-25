#!/usr/bin/env bash
# submit_finalize_and_merge.sh
#
# Parallel finalize + merge pipeline. Runs one SLURM job per chromosome:
#   1. finalize_incomplete_shards.py --chrom chrN  (CPU only, vectorized Welford)
#   2. merge_sae_shards.py --chrom chrN --include-partial  (CPU only)
#
# No GPU needed — all expensive extraction was already done by the mega shards.
# Uses CPU-only partitions or any available partition since no GPU is requested.
#
# Usage:
#   bash submit_finalize_and_merge.sh              # all chromosomes
#   bash submit_finalize_and_merge.sh chr1 chr2    # specific chromosomes
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
N_SHARDS=36

mkdir -p "${LOGS}"

# Default: all human chromosomes
if [ $# -gt 0 ]; then
    CHROMS="$@"
else
    CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY"
fi

# Kill any existing finalize process on the login node
echo "Checking for existing finalize processes..."
EXISTING=$(ps aux | grep "finalize_incomplete_shards" | grep python | grep -v grep | awk '{print $2}' || true)
if [ -n "$EXISTING" ]; then
    echo "  Killing existing finalize process: $EXISTING"
    kill $EXISTING 2>/dev/null || true
fi

# Partition rotation — CPU-only jobs, use partitions with most availability
# mit_normal: 12h limit, many idle nodes; mit_preemptable: 2d limit, many nodes
PARTITIONS=(mit_normal mit_preemptable mit_normal mit_preemptable)
P_IDX=0

echo "Submitting finalize+merge jobs for: ${CHROMS}"
echo ""

MERGE_JIDS=""

for CHROM in $CHROMS; do
    PARTITION=${PARTITIONS[$((P_IDX % ${#PARTITIONS[@]}))]}
    P_IDX=$((P_IDX + 1))

    JOB_NAME="finmerge_${CHROM}"
    SBATCH="${LOGS}/.${JOB_NAME}.sbatch"

    cat > "${SBATCH}" <<SBATCH
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p ${PARTITION}
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o ${LOGS}/${JOB_NAME}_%j.out
#SBATCH -e ${LOGS}/${JOB_NAME}_%j.err

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[${CHROM}] Started at \$(date) on \$(hostname)"

# Step 1: Finalize incomplete shards for this chromosome
echo "[${CHROM}] Step 1: Finalizing incomplete shards..."
python finalize_incomplete_shards.py \\
    --output_dir results/ \\
    --chrom ${CHROM} \\
    --n_shards ${N_SHARDS} || echo "[${CHROM}] WARNING: finalize had errors, continuing to merge"

echo "[${CHROM}] Step 1 done at \$(date)"

# Step 2: Merge all shards (complete + partial with chunk data)
echo "[${CHROM}] Step 2: Merging shards..."
python merge_sae_shards.py \\
    --chrom ${CHROM} \\
    --n_shards ${N_SHARDS} \\
    --output_dir results/ \\
    --include-partial || echo "[${CHROM}] WARNING: merge failed"

echo "[${CHROM}] Step 2 done at \$(date)"
echo "[${CHROM}] All done at \$(date)"
SBATCH

    JID=$(sbatch "${SBATCH}" | awk '{print $NF}')
    echo "  ${CHROM}: job ${JID}  partition=${PARTITION}"
    MERGE_JIDS="${MERGE_JIDS}:${JID}"
done

echo ""
echo "All finalize+merge jobs submitted."
echo ""

# Step 3: Submit genome-wide aggregation job that depends on all merges
AGG_NAME="sae_aggregate"
AGG_SBATCH="${LOGS}/.${AGG_NAME}.sbatch"

cat > "${AGG_SBATCH}" <<SBATCH
#!/bin/bash
#SBATCH -J ${AGG_NAME}
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o ${LOGS}/${AGG_NAME}_%j.out
#SBATCH -e ${LOGS}/${AGG_NAME}_%j.err
#SBATCH --dependency=afterany${MERGE_JIDS}

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[aggregate] Started at \$(date) on \$(hostname)"

# Aggregate genome-wide SAE global stats (from whole-chromosome scans)
echo "[aggregate] Aggregating genome-wide sae_global_stats..."
python tools/scan_sae_global_stats.py \\
    --aggregate \\
    --all_human \\
    --results_dir results/ || echo "[aggregate] WARNING: global stats aggregation failed"

# Aggregate genome-wide feature stats (from merged drop-region SAE runs)
echo "[aggregate] Aggregating genome-wide feature stats from merged SAE runs..."
python tools/aggregate_genome_sae_stats.py \\
    --results_dir results/ \\
    --all_human \\
    --force || echo "[aggregate] WARNING: feature stats aggregation failed"

echo "[aggregate] All done at \$(date)"
SBATCH

AGG_JID=$(sbatch "${AGG_SBATCH}" | awk '{print $NF}')
echo "Genome-wide aggregation: job ${AGG_JID} (depends on all merge jobs)"
echo ""
echo "Monitor: squeue -u platawa"
echo "Logs:    tail -f ${LOGS}/finmerge_chr1_*.out"
