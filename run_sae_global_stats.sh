#!/bin/bash
# run_sae_global_stats.sh — Submit genome-wide SAE min/max scan for all chromosomes
#
# Usage:
#   ./run_sae_global_stats.sh              # Submit all 24 human chromosomes
#   ./run_sae_global_stats.sh chr22        # Submit just one chromosome
#   ./run_sae_global_stats.sh --dry-run    # Preview without submitting
#
# Each chromosome runs independently on 1 GPU. After all finish,
# a CPU aggregation job merges per-chromosome stats into genome-wide min/max.

set -euo pipefail

PROJECT_DIR="/orcd/data/zhang_f/001/platawa/jan31_files"
CONDA_ENV="evo2_sep28"
GPU_PARTITION="mit_normal_gpu"
CPU_PARTITION="mit_normal"
RESULTS_DIR="${PROJECT_DIR}/results"
FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna"

# Time estimates by chromosome size category
# Small (chr21,22,Y): ~1h, Medium (chr13-20,X): ~3h, Large (chr1-12): ~6h
SCAN_MEM="50G"
SCAN_CPUS=4

AGGREGATE_TIME="00:30:00"
AGGREGATE_MEM="8G"
AGGREGATE_CPUS=2

ALL_CHROMS="chr21 chr22 chrY chr20 chr19 chr18 chr17 chr16 chr15 chr14 chr13 chrX chr12 chr11 chr10 chr9 chr8 chr7 chr6 chr5 chr4 chr3 chr2 chr1"

# Time limits per chromosome (max 6h on mit_normal_gpu)
get_time_limit() {
    case "$1" in
        chr21|chr22|chrY) echo "02:00:00" ;;
        chr19|chr20)      echo "04:00:00" ;;
        *)                echo "06:00:00" ;;
    esac
}

# ── Parse arguments ──────────────────────────────────────────────────────────
DRY_RUN=false
TARGET="${1:-all}"

if [[ "$TARGET" == "--dry-run" ]]; then
    DRY_RUN=true
    TARGET="all"
elif [[ "${2:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

if [[ "$TARGET" == "all" ]]; then
    CHROMS="$ALL_CHROMS"
else
    CHROMS="$TARGET"
fi

mkdir -p "${PROJECT_DIR}/logs"

# ── Submit per-chromosome scan jobs ──────────────────────────────────────────
SCAN_JOB_IDS=()

for chrom in $CHROMS; do
    time_limit=$(get_time_limit "$chrom")
    job_name="sae_stats_${chrom}"
    sbatch_file="${PROJECT_DIR}/logs/.${job_name}.sbatch"

    cat > "$sbatch_file" <<SBATCH
#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -p ${GPU_PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${SCAN_CPUS}
#SBATCH --mem=${SCAN_MEM}
#SBATCH -t ${time_limit}
#SBATCH -o ${PROJECT_DIR}/logs/${job_name}_%j.out
#SBATCH -e ${PROJECT_DIR}/logs/${job_name}_%j.err

cd ${PROJECT_DIR}
module load miniforge/24.3.0-0
conda activate ${CONDA_ENV}

python tools/scan_sae_global_stats.py \\
    --fasta ${FASTA} \\
    --chrom ${chrom} \\
    --output_dir ${RESULTS_DIR}
SBATCH

    if $DRY_RUN; then
        echo "[DRY-RUN] Would submit: ${job_name} (time=${time_limit})"
    else
        job_id=$(sbatch "$sbatch_file" | awk '{print $NF}')
        echo "Submitted ${job_name} -> Job ${job_id} (time=${time_limit})"
        SCAN_JOB_IDS+=("$job_id")
    fi
done

# ── Submit aggregation job (depends on all scans) ────────────────────────────
if [[ "$TARGET" == "all" ]]; then
    agg_name="sae_stats_aggregate"
    agg_sbatch="${PROJECT_DIR}/logs/.${agg_name}.sbatch"

    cat > "$agg_sbatch" <<SBATCH
#!/bin/bash
#SBATCH -J ${agg_name}
#SBATCH -p ${CPU_PARTITION}
#SBATCH --cpus-per-task=${AGGREGATE_CPUS}
#SBATCH --mem=${AGGREGATE_MEM}
#SBATCH -t ${AGGREGATE_TIME}
#SBATCH -o ${PROJECT_DIR}/logs/${agg_name}_%j.out
#SBATCH -e ${PROJECT_DIR}/logs/${agg_name}_%j.err

cd ${PROJECT_DIR}
module load miniforge/24.3.0-0
conda activate ${CONDA_ENV}

python tools/scan_sae_global_stats.py \\
    --aggregate \\
    --results_dir ${RESULTS_DIR} \\
    --all_human
SBATCH

    if $DRY_RUN; then
        echo ""
        echo "[DRY-RUN] Would submit: ${agg_name} (after all scan jobs)"
    else
        dep_str=$(IFS=:; echo "${SCAN_JOB_IDS[*]}")
        job_id=$(sbatch --dependency=afterok:${dep_str} "$agg_sbatch" | awk '{print $NF}')
        echo ""
        echo "Submitted ${agg_name} -> Job ${job_id} (depends on all scan jobs)"
    fi
fi

echo ""
echo "Monitor with: squeue -u \$USER"
