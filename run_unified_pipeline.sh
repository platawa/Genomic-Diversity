#!/bin/bash
# run_unified_pipeline.sh — Unified scoring + SAE stats + analysis pipeline
#
# Replaces: run_pipeline.sh, run_sae_global_stats.sh, scripts/run_genome_pipeline.sh
#
# Per-chromosome stages (SLURM dependency chain):
#   1. Scoring (GPU): score_chromosome.py — adds --collect_sae_stats only
#      when no existing SAE stats COMPLETED for this chrom
#   2. Analysis/Plotting (CPU, depends on 1): tools/analyze_scoring_results.py --auto
#   3. SAE Drop Analysis (GPU, depends on 1): run_sae_on_chromosome_drops.py --auto
#
# Cross-chromosome stages:
#   4. SAE Stats Aggregation (CPU, depends on all stage 1)
#   5. Genome-wide Visualization (CPU, depends on all stage 3)
#
# Features:
#   - Skip-completed: checks COMPLETED sentinels before submitting
#   - Smart SAE collection: --collect_sae_stats only when needed
#   - Dual-partition: large chroms → mit_preemptable, small → mit_normal_gpu
#   - --dry-run: preview all jobs without submitting
#
# Usage:
#   ./run_unified_pipeline.sh all                         # all 24 human chroms
#   ./run_unified_pipeline.sh all --dry-run               # preview
#   ./run_unified_pipeline.sh chr21 chr22                 # specific chroms
#   ./run_unified_pipeline.sh all --scoring-only          # skip SAE/analysis
#   ./run_unified_pipeline.sh all --no-logprobs --no-rc   # minimal scoring

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_DIR="/orcd/data/zhang_f/001/platawa/jan31_files"
CONDA_ENV="evo2_sep28"
RESULTS_DIR="${PROJECT_DIR}/results"

# Partitions
GPU_PARTITION_LARGE="mit_preemptable"   # 48h limit, 4 GPU max — for large chroms
GPU_PARTITION_SMALL="mit_normal_gpu"     # 6h limit, 2 GPU max — for small/medium
CPU_PARTITION="pi_zhang_f"               # 7 day, CPU only — analysis/plotting

# Reference genome (GRCh38)
FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna"
GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"

# Scoring defaults
SCORE_GPUS=1
SCORE_CPUS=8
SCORE_MEM="100G"
BATCH_SIZE=4

# SAE drop analysis defaults
SAE_TIME="24:00:00"
SAE_MEM="80G"
SAE_CPUS=4
SAE_MAX_REGIONS=999999
SAE_MIN_CONFIDENCE=8.0

# Analysis defaults
ANALYSIS_TIME="01:00:00"
ANALYSIS_MEM="16G"
ANALYSIS_CPUS=4

# Aggregation defaults
AGGREGATE_TIME="00:30:00"
AGGREGATE_MEM="8G"

# All human chromosomes (smallest first for faster initial results)
ALL_CHROMS=(chr21 chr22 chrY chr20 chr19 chr18 chr17 chr16 chr15 chr14 chr13 chrX chr12 chr11 chr10 chr9 chr8 chr7 chr6 chr5 chr4 chr3 chr2 chr1)

# Chromosome size categories (determines partition routing, GPU count, and time limits)
# Large:  >150 Mbp → preemptable, 1 GPU, 48h      (chr1-chr8, chrX)
# Small:  ≤150 Mbp → mit_normal_gpu, 1 GPU, 6h    (chr9-chr22, chrY)
# NOTE: 2-GPU routing removed — multiprocessing deadlocks; 1 GPU is stable
LARGE_CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chrX"

# ── Parse arguments ──────────────────────────────────────────────────────────
DRY_RUN=false
SCORING_ONLY=false
SKIP_SCORING=false
SKIP_SAE_DROPS=false
SKIP_ANALYSIS=false
RC_AVERAGE=true
COMPUTE_LOGPROBS=true
CHROMS=()

usage() {
    echo "Usage: $0 <chrom1> [chrom2 ...] | all [options]"
    echo ""
    echo "Options:"
    echo "  --dry-run          Preview commands without submitting"
    echo "  --scoring-only     Only run scoring (skip SAE drops + analysis)"
    echo "  --skip-scoring     Skip scoring (assumes already done)"
    echo "  --skip-sae-drops   Skip SAE drop analysis"
    echo "  --skip-analysis    Skip CPU analysis/plotting"
    echo "  --no-rc            Disable RC averaging (halves scoring time)"
    echo "  --no-logprobs      Don't compute logprobs (11x smaller entropy files)"
    echo ""
    echo "Examples:"
    echo "  $0 all --dry-run"
    echo "  $0 all --no-logprobs --no-rc    # fastest scoring"
    echo "  $0 chr21 chr22                   # specific chromosomes"
    exit 1
}

if [[ $# -eq 0 ]]; then usage; fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)         DRY_RUN=true; shift ;;
        --scoring-only)    SCORING_ONLY=true; shift ;;
        --skip-scoring)    SKIP_SCORING=true; shift ;;
        --skip-sae-drops)  SKIP_SAE_DROPS=true; shift ;;
        --skip-analysis)   SKIP_ANALYSIS=true; shift ;;
        --no-rc)           RC_AVERAGE=false; shift ;;
        --no-logprobs)     COMPUTE_LOGPROBS=false; shift ;;
        --help|-h)         usage ;;
        all)               CHROMS=("${ALL_CHROMS[@]}"); shift ;;
        chr*|NC_*)         CHROMS+=("$1"); shift ;;
        *)                 echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ ${#CHROMS[@]} -eq 0 ]]; then
    echo "ERROR: No chromosomes specified"
    usage
fi

if $SCORING_ONLY; then
    SKIP_SAE_DROPS=true
    SKIP_ANALYSIS=true
fi

mkdir -p "${PROJECT_DIR}/logs"

# ── Helper functions ─────────────────────────────────────────────────────────

is_large_chrom() {
    local chrom="$1"
    [[ " $LARGE_CHROMS " == *" $chrom "* ]]
}


get_scoring_partition() {
    local chrom="$1"
    if is_large_chrom "$chrom"; then
        echo "$GPU_PARTITION_LARGE"
    else
        echo "$GPU_PARTITION_SMALL"
    fi
}

get_scoring_time() {
    local chrom="$1"
    if is_large_chrom "$chrom"; then
        echo "48:00:00"
    else
        echo "06:00:00"
    fi
}

get_scoring_gpus() {
    echo "$SCORE_GPUS"
}

get_scoring_mem() {
    echo "$SCORE_MEM"
}

has_completed() {
    # Check if a completed run exists for chrom+stage
    local chrom="$1" stage="$2"
    local stage_dir="${RESULTS_DIR}/${chrom}/${stage}"
    [[ -d "$stage_dir" ]] && find "$stage_dir" -name COMPLETED -print -quit 2>/dev/null | grep -q .
}

submit_job() {
    local job_name="$1"
    local script_content="$2"
    local dependency="${3:-}"
    local sbatch_file="${PROJECT_DIR}/logs/.${job_name}.sbatch"

    echo "$script_content" > "$sbatch_file"

    if $DRY_RUN; then
        echo "  [DRY-RUN] $job_name"
        return
    fi

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

# ── Print plan ───────────────────────────────────────────────────────────────
echo "============================================================"
echo "Unified Pipeline: ${#CHROMS[@]} chromosome(s)"
echo "  RC average:   $RC_AVERAGE"
echo "  Logprobs:     $COMPUTE_LOGPROBS"
echo "  SAE drops:    $( $SKIP_SAE_DROPS && echo 'SKIP' || echo 'YES' )"
echo "  Analysis:     $( $SKIP_ANALYSIS && echo 'SKIP' || echo 'YES' )"
echo "  Dry run:      $DRY_RUN"
echo "============================================================"
echo ""

# ── Per-chromosome jobs ──────────────────────────────────────────────────────
declare -a SCORING_JOB_IDS=()

for chrom in "${CHROMS[@]}"; do
    echo "── $chrom ──"

    score_job_id=""

    # ── Stage 1: Scoring (GPU) ───────────────────────────────────────────
    if ! $SKIP_SCORING; then
        if has_completed "$chrom" "scoring"; then
            echo "  [SKIP] scoring: already completed"
        else
            partition=$(get_scoring_partition "$chrom")
            time_limit=$(get_scoring_time "$chrom")
            n_gpus=$(get_scoring_gpus "$chrom")
            score_mem=$(get_scoring_mem "$chrom")

            # Smart SAE stats: collect if no existing sae_global_stats
            sae_flag=""
            if has_completed "$chrom" "sae_global_stats"; then
                echo "  SAE stats: already completed (scoring only)"
            else
                sae_flag="--collect_sae_stats"
                echo "  SAE stats: will collect during scoring (fused)"
            fi

            # Build scoring flags
            score_flags=""
            if $RC_AVERAGE; then score_flags="$score_flags --rc_average"; fi
            if $COMPUTE_LOGPROBS; then score_flags="$score_flags --compute_logprobs"; fi

            score_script=$(cat <<SBATCH
#!/bin/bash
#SBATCH -J score_${chrom}
#SBATCH -p ${partition}
#SBATCH --gres=gpu:${n_gpus}
#SBATCH --cpus-per-task=${SCORE_CPUS}
#SBATCH --mem=${score_mem}
#SBATCH -t ${time_limit}
#SBATCH -o ${PROJECT_DIR}/logs/score_${chrom}_%j.out
#SBATCH -e ${PROJECT_DIR}/logs/score_${chrom}_%j.err
#SBATCH --requeue

cd ${PROJECT_DIR}
module load miniforge/24.3.0-0
conda activate ${CONDA_ENV}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== Scoring ${chrom} (partition=${partition}) ==="
echo "Started: \$(date)"

python score_chromosome.py --chrom ${chrom} --fasta ${FASTA} --output_dir ${RESULTS_DIR} --n_gpus ${n_gpus} --auto_chunk_size --bf16 --batch_size ${BATCH_SIZE} --detection_methods zscore,mad --skip_if_completed ${score_flags} ${sae_flag}

echo "Finished: \$(date)"
SBATCH
)

            result=$(submit_job "score_${chrom}" "$score_script")
            score_job_id=$(echo "$result" | tail -1)
            SCORING_JOB_IDS+=("$score_job_id")
        fi
    fi

    # ── Stage 2: Analysis/Plotting (CPU, depends on scoring) ─────────────
    if ! $SKIP_ANALYSIS && [[ -n "$score_job_id" || $SKIP_SCORING == true ]]; then
        analysis_script=$(cat <<SBATCH
#!/bin/bash
#SBATCH -J analyze_${chrom}
#SBATCH -p ${CPU_PARTITION}
#SBATCH --cpus-per-task=${ANALYSIS_CPUS}
#SBATCH --mem=${ANALYSIS_MEM}
#SBATCH -t ${ANALYSIS_TIME}
#SBATCH -o ${PROJECT_DIR}/logs/analyze_${chrom}_%j.out
#SBATCH -e ${PROJECT_DIR}/logs/analyze_${chrom}_%j.err

cd ${PROJECT_DIR}
module load miniforge/24.3.0-0
conda activate ${CONDA_ENV}

python tools/analyze_scoring_results.py --auto --chrom ${chrom} --results_dir ${RESULTS_DIR} --gtf ${GTF} --all_plots --dashboard
SBATCH
)

        submit_job "analyze_${chrom}" "$analysis_script" "$score_job_id" > /dev/null
    fi

    # ── Stage 3: SAE Drop Analysis (GPU, depends on scoring) ─────────────
    if ! $SKIP_SAE_DROPS && [[ -n "$score_job_id" || $SKIP_SCORING == true ]]; then
        sae_drop_script=$(cat <<SBATCH
#!/bin/bash
#SBATCH -J sae_drops_${chrom}
#SBATCH -p ${GPU_PARTITION_LARGE}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${SAE_CPUS}
#SBATCH --mem=${SAE_MEM}
#SBATCH -t ${SAE_TIME}
#SBATCH -o ${PROJECT_DIR}/logs/sae_drops_${chrom}_%j.out
#SBATCH -e ${PROJECT_DIR}/logs/sae_drops_${chrom}_%j.err
#SBATCH --requeue

cd ${PROJECT_DIR}
module load miniforge/24.3.0-0
conda activate ${CONDA_ENV}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run_sae_on_chromosome_drops.py --auto --chrom ${chrom} --fasta ${FASTA} --gtf ${GTF} --output_dir ${RESULTS_DIR} --max_regions ${SAE_MAX_REGIONS} --min_confidence ${SAE_MIN_CONFIDENCE} --run_latent_analysis --latent_only
SBATCH
)

        submit_job "sae_drops_${chrom}" "$sae_drop_script" "$score_job_id" > /dev/null
    fi

    echo ""
done

# ── Stage 4: SAE Stats Aggregation (CPU, depends on all scoring) ─────────
if [[ ${#SCORING_JOB_IDS[@]} -gt 0 ]] || $SKIP_SCORING; then
    dep_str=""
    if [[ ${#SCORING_JOB_IDS[@]} -gt 0 ]] && ! $DRY_RUN; then
        dep_str=$(IFS=:; echo "${SCORING_JOB_IDS[*]}")
    fi

    agg_script=$(cat <<SBATCH
#!/bin/bash
#SBATCH -J sae_aggregate
#SBATCH -p ${CPU_PARTITION}
#SBATCH --cpus-per-task=2
#SBATCH --mem=${AGGREGATE_MEM}
#SBATCH -t ${AGGREGATE_TIME}
#SBATCH -o ${PROJECT_DIR}/logs/sae_aggregate_%j.out
#SBATCH -e ${PROJECT_DIR}/logs/sae_aggregate_%j.err

cd ${PROJECT_DIR}
module load miniforge/24.3.0-0
conda activate ${CONDA_ENV}

python tools/scan_sae_global_stats.py --aggregate --results_dir ${RESULTS_DIR} --all_human
SBATCH
)

    echo "── Cross-chromosome stages ──"
    submit_job "sae_aggregate" "$agg_script" "$dep_str" > /dev/null
    echo ""
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Pipeline submitted for ${#CHROMS[@]} chromosome(s)."
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Cancel:   scancel -u \$USER -n score_chr22"
echo "============================================================"
