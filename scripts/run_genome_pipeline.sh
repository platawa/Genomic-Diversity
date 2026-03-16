#!/bin/bash
# run_genome_pipeline.sh — Submit scoring + SAE jobs for multiple chromosomes
#
# Submits two SLURM stages per chromosome with dependencies:
#   1. score_chromosome.py   (GPU)  — entropy scoring + drop detection
#   2. run_sae_on_chromosome_drops.py (GPU) — SAE feature extraction
#      (uses --auto to find the completed scoring run)
#
# All chromosomes run in parallel; SAE waits for its own chromosome's scoring.
#
# Usage:
#   ./scripts/run_genome_pipeline.sh chr22                    # single chromosome
#   ./scripts/run_genome_pipeline.sh chr21 chr22              # specific chromosomes
#   ./scripts/run_genome_pipeline.sh all                      # all 24 human chromosomes
#   ./scripts/run_genome_pipeline.sh all --dry-run            # preview without submitting
#   ./scripts/run_genome_pipeline.sh chr22 --organism ecoli   # E. coli
#   ./scripts/run_genome_pipeline.sh all --skip-scoring       # SAE only (scoring already done)
#   ./scripts/run_genome_pipeline.sh all --skip-sae           # scoring only
#
# After all jobs complete, run the genome-wide t-SNE:
#   python tools/genome_sae_tsne.py --all_human --gtf /path/to/genomic.gtf --embedding both

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_DIR="/orcd/data/zhang_f/001/platawa/jan31_files"
CONDA_ENV="evo2_sep28"
PARTITION="mit_preemptable"

# Human (GRCh38)
HUMAN_FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna"
HUMAN_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"

# E. coli K-12
ECOLI_FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna"
ECOLI_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf"

# Bacillus subtilis
BACILLUS_FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/GCF_000009045.1_ASM904v1_genomic.fna"
BACILLUS_GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf"

# Scoring job resources
SCORE_TIME="48:00:00"
SCORE_MEM="100G"
SCORE_CPUS=8
SCORE_GPUS=1

# SAE job resources
SAE_TIME="04:00:00"
SAE_MEM="50G"
SAE_CPUS=4
SAE_MAX_REGIONS=1000
SAE_MIN_CONFIDENCE=8.0

# All human chromosomes (smallest first for faster initial results)
ALL_CHROMS=(chr21 chr22 chr20 chr19 chr18 chr17 chr16 chr15 chr14 chr13 chr12 chr11 chr10 chr9 chr8 chr7 chr6 chr5 chr4 chr3 chr2 chr1 chrX chrY)

# ── Parse arguments ──────────────────────────────────────────────────────────
DRY_RUN=false
SKIP_SCORING=false
SKIP_SAE=false
ORGANISM="human"
CHROMS=()

usage() {
    echo "Usage: $0 <chrom1> [chrom2 ...] | all [options]"
    echo ""
    echo "Options:"
    echo "  --dry-run         Preview commands without submitting"
    echo "  --skip-scoring    Skip scoring stage (assumes scoring already completed)"
    echo "  --skip-sae        Skip SAE stage (scoring only)"
    echo "  --organism NAME   human (default), ecoli, or bacillus"
    echo ""
    echo "Examples:"
    echo "  $0 chr22"
    echo "  $0 all --dry-run"
    echo "  $0 all --skip-scoring    # run SAE on all chroms with existing scoring"
    exit 1
}

if [[ $# -eq 0 ]]; then
    usage
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=true; shift ;;
        --skip-scoring)  SKIP_SCORING=true; shift ;;
        --skip-sae)      SKIP_SAE=true; shift ;;
        --organism)      ORGANISM="$2"; shift 2 ;;
        --help|-h)       usage ;;
        all)             CHROMS=("${ALL_CHROMS[@]}"); shift ;;
        chr*|NC_*)       CHROMS+=("$1"); shift ;;
        *)               echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ ${#CHROMS[@]} -eq 0 ]]; then
    echo "ERROR: No chromosomes specified"
    usage
fi

# ── Resolve organism paths ───────────────────────────────────────────────────
case "$ORGANISM" in
    human)    FASTA="$HUMAN_FASTA"; GTF="$HUMAN_GTF" ;;
    ecoli)    FASTA="$ECOLI_FASTA"; GTF="$ECOLI_GTF" ;;
    bacillus) FASTA="$BACILLUS_FASTA"; GTF="$BACILLUS_GTF" ;;
    *)        echo "Unknown organism: $ORGANISM"; exit 1 ;;
esac

RESULTS_DIR="${PROJECT_DIR}/results"

# ── Ensure logs directory exists ─────────────────────────────────────────────
mkdir -p "${PROJECT_DIR}/logs"

# ── Submit helper ────────────────────────────────────────────────────────────
submit_job() {
    local job_name="$1"
    local script_content="$2"
    local dependency="${3:-}"

    local sbatch_file="${PROJECT_DIR}/logs/.${job_name}.sbatch"

    if $DRY_RUN; then
        echo "  [DRY-RUN] $job_name"
        echo "$script_content" > "$sbatch_file"
        echo "  Script: $sbatch_file"
        echo "DRYRUN_${job_name}"
        return
    fi

    echo "$script_content" > "$sbatch_file"

    local dep_flag=""
    if [[ -n "$dependency" ]]; then
        dep_flag="--dependency=afterok:${dependency}"
    fi

    local result
    result=$(sbatch $dep_flag "$sbatch_file")
    local job_id
    job_id=$(echo "$result" | awk '{print $NF}')
    echo "  $job_name -> Job $job_id"
    echo "$job_id"
}

# ── Submit jobs for each chromosome ──────────────────────────────────────────
echo "============================================================"
echo "Genome Pipeline: ${#CHROMS[@]} chromosome(s), organism=$ORGANISM"
echo "  Scoring: $( $SKIP_SCORING && echo 'SKIP' || echo 'YES' )"
echo "  SAE:     $( $SKIP_SAE && echo 'SKIP' || echo 'YES' )"
echo "  Dry run: $DRY_RUN"
echo "============================================================"
echo ""

declare -A SCORE_JOBS  # chrom -> job_id (for dependency tracking)

for chrom in "${CHROMS[@]}"; do
    echo "── $chrom ──"

    score_job_id=""

    # ── Stage 1: Scoring (GPU) ───────────────────────────────────────────
    if ! $SKIP_SCORING; then
        score_script=$(cat <<SBATCH
#!/bin/bash
#SBATCH -J score_${chrom}
#SBATCH -p ${PARTITION}
#SBATCH --gres=gpu:${SCORE_GPUS}
#SBATCH --cpus-per-task=${SCORE_CPUS}
#SBATCH --mem=${SCORE_MEM}
#SBATCH -t ${SCORE_TIME}
#SBATCH -o ${PROJECT_DIR}/logs/score_${chrom}_%j.out
#SBATCH -e ${PROJECT_DIR}/logs/score_${chrom}_%j.err

cd ${PROJECT_DIR}
module load miniforge/24.3.0-0
conda activate ${CONDA_ENV}

echo "=== Scoring ${chrom} ==="
echo "Started: \$(date)"

python score_chromosome.py \\
    --chrom ${chrom} \\
    --fasta ${FASTA} \\
    --output_dir ${RESULTS_DIR} \\
    --n_gpus ${SCORE_GPUS} \\
    --detection_methods zscore,mad \\
    --rc_average \\
    --compute_logprobs

echo "Finished: \$(date)"
SBATCH
)

        score_result=$(submit_job "score_${chrom}" "$score_script")
        score_job_id=$(echo "$score_result" | tail -1)
        echo "$score_result" | head -n -1
    fi

    # ── Stage 2: SAE (GPU, depends on scoring) ──────────────────────────
    if ! $SKIP_SAE; then
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

echo "=== SAE on ${chrom} ==="
echo "Started: \$(date)"

python run_sae_on_chromosome_drops.py \\
    --auto \\
    --chrom ${chrom} \\
    --fasta ${FASTA} \\
    --output_dir ${RESULTS_DIR} \\
    --max_regions ${SAE_MAX_REGIONS} \\
    --min_confidence ${SAE_MIN_CONFIDENCE} \\
    --stratified \\
    --run_latent_analysis \\
    --latent_only

echo "Finished: \$(date)"
SBATCH
)

        sae_result=$(submit_job "sae_${chrom}" "$sae_script" "$score_job_id")
        echo "$sae_result" | head -n -1
    fi

    echo ""
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo "All jobs submitted for ${#CHROMS[@]} chromosome(s)."
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Cancel:   scancel -u \$USER -n score_chr22   # cancel specific job by name"
echo ""
echo "After all jobs complete, run the genome-wide t-SNE:"
echo "  python tools/genome_sae_tsne.py \\"
echo "      --all_human \\"
echo "      --gtf ${GTF} \\"
echo "      --results_dir ${RESULTS_DIR} \\"
echo "      --embedding both"
echo "============================================================"
