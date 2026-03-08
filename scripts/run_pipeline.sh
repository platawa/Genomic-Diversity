#!/bin/bash
# run_pipeline.sh — Parameterized pipeline for scoring + analysis + SAE
#
# Usage:
#   ./run_pipeline.sh <chrom>                          # e.g., ./run_pipeline.sh chr22
#   ./run_pipeline.sh <chrom> [--dry-run]              # preview commands without running
#   ./run_pipeline.sh all                              # submit all human autosomes
#   ./run_pipeline.sh all --dry-run                    # preview all submissions
#   ./run_pipeline.sh <chrom> --organism ecoli         # E. coli K-12
#   ./run_pipeline.sh <chrom> --organism bacillus      # B. subtilis
#   ./run_pipeline.sh <chrom> --detection_methods zscore,mad,cusum
#
# Runs up to four SLURM stages with dependencies:
#   1. score_chromosome.py   (GPU, ~2-48h depending on chrom size)
#   2. analyze_scoring_results.py  (CPU, ~30min)
#   3. run_sae_on_chromosome_drops.py  (GPU, ~1h)
#   4. compare_detection_methods.py  (CPU, ~10min) [optional]

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_DIR="/orcd/data/zhang_f/001/platawa/jan31_files"
CONDA_ENV="evo2_sep28"
PARTITION="mit_preemptable"

# Organism-specific paths (default: human)
ORGANISM="human"
DETECTION_METHODS="zscore,mad"

# Human (GRCh38)
HUMAN_FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna"
HUMAN_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"

# E. coli K-12
ECOLI_FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna"
ECOLI_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf"

# Bacillus subtilis
BACILLUS_FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/GCF_000009045.1_ASM904v1_genomic.fna"
BACILLUS_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf"

# Resolved at runtime based on --organism
FASTA=""
GTF=""
OUTPUT_DIR=""
SAE_OUTPUT_DIR=""

# Scoring defaults
N_GPUS=1
SCORE_TIME="48:00:00"
SCORE_MEM="100G"
SCORE_CPUS=8
SCORE_EXTRA_FLAGS="--rc_average --compute_logprobs --organized"

# Analysis defaults
ANALYZE_TIME="00:30:00"
ANALYZE_MEM="16G"
ANALYZE_CPUS=2

# SAE defaults
SAE_TIME="02:00:00"
SAE_MEM="50G"
SAE_CPUS=4
SAE_MAX_REGIONS=50
SAE_MIN_CONFIDENCE=8.0

# All human autosomes + sex chromosomes (ordered smallest to largest for testing)
ALL_CHROMS="chr21 chr22 chr20 chr19 chr18 chr17 chr16 chr15 chr14 chr13 chr12 chr11 chr10 chr9 chr8 chr7 chr6 chr5 chr4 chr3 chr2 chr1 chrX chrY"

# ── Parse arguments ──────────────────────────────────────────────────────────
DRY_RUN=false
CHROM="${1:-}"

if [[ -z "$CHROM" ]]; then
    echo "Usage: $0 <chrom|all> [--dry-run] [--organism human|ecoli|bacillus] [--detection_methods m1,m2,...]"
    echo ""
    echo "Examples:"
    echo "  $0 chr22                                  # Human chr22"
    echo "  $0 chr21 --dry-run                        # Preview commands"
    echo "  $0 all                                    # All human chromosomes"
    echo "  $0 NC_000913.3 --organism ecoli           # E. coli K-12"
    echo "  $0 NC_000964.3 --organism bacillus        # B. subtilis"
    echo "  $0 chr22 --detection_methods zscore,mad,cusum"
    exit 1
fi

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --organism) ORGANISM="$2"; shift 2 ;;
        --detection_methods) DETECTION_METHODS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Resolve organism-specific paths ──────────────────────────────────────────
case "$ORGANISM" in
    human)
        FASTA="$HUMAN_FASTA"
        GTF="$HUMAN_GTF"
        ;;
    ecoli)
        FASTA="$ECOLI_FASTA"
        GTF="$ECOLI_GTF"
        ;;
    bacillus)
        FASTA="$BACILLUS_FASTA"
        GTF="$BACILLUS_GTF"
        ;;
    *)
        echo "Unknown organism: $ORGANISM (expected: human, ecoli, bacillus)"
        exit 1
        ;;
esac
OUTPUT_DIR="${PROJECT_DIR}/results"
SAE_OUTPUT_DIR="${PROJECT_DIR}/results"

# ── Helper: generate sbatch script and submit ────────────────────────────────
submit_job() {
    local job_name="$1"
    local script_content="$2"
    local dependency="${3:-}"

    local sbatch_file="${PROJECT_DIR}/logs/.${job_name}.sbatch"
    echo "$script_content" > "$sbatch_file"

    local dep_flag=""
    if [[ -n "$dependency" ]]; then
        dep_flag="--dependency=afterok:${dependency}"
    fi

    if $DRY_RUN; then
        echo "[DRY-RUN] Would submit: $job_name"
        echo "  sbatch $dep_flag $sbatch_file"
        echo "  ---"
        echo "$script_content" | head -30
        echo "  ..."
        echo ""
        echo "PREVIEW_JOB_ID"
    else
        local job_id
        job_id=$(sbatch $dep_flag "$sbatch_file" | awk '{print $NF}')
        echo "Submitted $job_name -> Job $job_id"
        echo "$job_id"
    fi
}

# ── Pipeline for a single chromosome ────────────────────────────────────────
run_chromosome() {
    local chrom="$1"
    local date_tag
    date_tag=$(date +%Y%m%d)

    echo "========================================"
    echo "Pipeline: $chrom"
    echo "========================================"

    mkdir -p "${PROJECT_DIR}/logs"

    # ── Stage 1: Score chromosome (GPU) ──────────────────────────────────
    local score_script
    score_script=$(cat <<SBATCH
#!/bin/bash
#SBATCH -J score_${chrom}
#SBATCH -p ${PARTITION}
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --cpus-per-task=${SCORE_CPUS}
#SBATCH --mem=${SCORE_MEM}
#SBATCH -t ${SCORE_TIME}
#SBATCH -o ${PROJECT_DIR}/logs/score_${chrom}_%j.out
#SBATCH -e ${PROJECT_DIR}/logs/score_${chrom}_%j.err

cd ${PROJECT_DIR}
module load miniforge/24.3.0-0
conda activate ${CONDA_ENV}

python score_chromosome.py \\
    --chrom ${chrom} \\
    --output_prefix ${chrom} \\
    --fasta ${FASTA} \\
    --output_dir ${OUTPUT_DIR} \\
    --n_gpus ${N_GPUS} \\
    --detection_methods ${DETECTION_METHODS} \\
    ${SCORE_EXTRA_FLAGS}
SBATCH
)

    local score_result
    score_result=$(submit_job "score_${chrom}" "$score_script")
    local score_job_id
    score_job_id=$(echo "$score_result" | tail -1)
    echo "$score_result" | head -n -1

    # ── Find the output run directory ────────────────────────────────────
    # The --organized flag creates: {output_dir}/{chrom}_{date}_{flags}/
    # We use a glob pattern since the exact name depends on flags.
    local run_dir_pattern="${OUTPUT_DIR}/${chrom}_${date_tag}_*"

    # ── Stage 2: Analyze scoring results (CPU only) ──────────────────────
    local analyze_script
    analyze_script=$(cat <<SBATCH
#!/bin/bash
#SBATCH -J analyze_${chrom}
#SBATCH -p ${PARTITION}
#SBATCH --cpus-per-task=${ANALYZE_CPUS}
#SBATCH --mem=${ANALYZE_MEM}
#SBATCH -t ${ANALYZE_TIME}
#SBATCH -o ${PROJECT_DIR}/logs/analyze_${chrom}_%j.out
#SBATCH -e ${PROJECT_DIR}/logs/analyze_${chrom}_%j.err

cd ${PROJECT_DIR}
module load miniforge/24.3.0-0
conda activate ${CONDA_ENV}

# Find the run directory (organized mode creates dated dirs)
RUN_DIR=\$(ls -td ${run_dir_pattern} 2>/dev/null | head -1)
if [[ -z "\$RUN_DIR" ]]; then
    echo "ERROR: No run directory matching ${run_dir_pattern}" >&2
    exit 1
fi

echo "Analyzing: \$RUN_DIR"
python tools/analyze_scoring_results.py \\
    --prefix "\$RUN_DIR/data" \\
    --all_plots
SBATCH
)

    local analyze_result
    analyze_result=$(submit_job "analyze_${chrom}" "$analyze_script" "$score_job_id")
    local analyze_job_id
    analyze_job_id=$(echo "$analyze_result" | tail -1)
    echo "$analyze_result" | head -n -1

    # ── Stage 3: SAE analysis (GPU) ──────────────────────────────────────
    local sae_script
    sae_script=$(cat <<SBATCH
#!/bin/bash
#SBATCH -J sae_${chrom}
#SBATCH -p ${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${SAE_CPUS}
#SBATCH --mem=${SAE_MEM}
#SBATCH -t ${SAE_TIME}
#SBATCH -o ${PROJECT_DIR}/logs/sae_${chrom}_%j.out
#SBATCH -e ${PROJECT_DIR}/logs/sae_${chrom}_%j.err

cd ${PROJECT_DIR}
module load miniforge/24.3.0-0
conda activate ${CONDA_ENV}

# Find the run directory
RUN_DIR=\$(ls -td ${run_dir_pattern} 2>/dev/null | head -1)
if [[ -z "\$RUN_DIR" ]]; then
    echo "ERROR: No run directory matching ${run_dir_pattern}" >&2
    exit 1
fi

echo "SAE analysis on: \$RUN_DIR"
python run_sae_on_chromosome_drops.py \\
    --boundaries "\$RUN_DIR/data/drop_boundaries.tsv" \\
    --entropy "\$RUN_DIR/data/entropy.npz" \\
    --chrom ${chrom} \\
    --fasta ${FASTA} \\
    --max_regions ${SAE_MAX_REGIONS} \\
    --min_confidence ${SAE_MIN_CONFIDENCE} \\
    --output_dir ${SAE_OUTPUT_DIR}/${chrom}
SBATCH
)

    local sae_result
    sae_result=$(submit_job "sae_${chrom}" "$sae_script" "$score_job_id")
    echo "$sae_result" | head -n -1

    # ── Stage 4: Detection comparison (CPU only) ─────────────────────────
    local detect_script
    detect_script=$(cat <<SBATCH
#!/bin/bash
#SBATCH -J detect_${chrom}
#SBATCH -p ${PARTITION}
#SBATCH --cpus-per-task=${ANALYZE_CPUS}
#SBATCH --mem=${ANALYZE_MEM}
#SBATCH -t ${ANALYZE_TIME}
#SBATCH -o ${PROJECT_DIR}/logs/detect_${chrom}_%j.out
#SBATCH -e ${PROJECT_DIR}/logs/detect_${chrom}_%j.err

cd ${PROJECT_DIR}
module load miniforge/24.3.0-0
conda activate ${CONDA_ENV}

# Find the run directory
RUN_DIR=\$(ls -td ${run_dir_pattern} 2>/dev/null | head -1)
if [[ -z "\$RUN_DIR" ]]; then
    echo "ERROR: No run directory matching ${run_dir_pattern}" >&2
    exit 1
fi

echo "Detection comparison on: \$RUN_DIR"
python tools/compare_detection_methods.py \\
    --entropy "\$RUN_DIR/data/entropy.npz" \\
    --gtf ${GTF} \\
    --chrom ${chrom} \\
    --output_dir "\$RUN_DIR/detection_comparison"
SBATCH
)

    local detect_result
    detect_result=$(submit_job "detect_${chrom}" "$detect_script" "$score_job_id")
    echo "$detect_result" | head -n -1

    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────────────
if [[ "$CHROM" == "all" ]]; then
    echo "Submitting pipeline for ALL chromosomes"
    echo "Order: $ALL_CHROMS"
    echo ""
    for c in $ALL_CHROMS; do
        run_chromosome "$c"
    done
    echo "All jobs submitted. Monitor with: squeue -u \$USER"
else
    run_chromosome "$CHROM"
fi
