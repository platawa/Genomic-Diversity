#!/bin/bash
#===============================================================================
# run_benchmark.sh
#
# SLURM job script for running genome scoring performance benchmarks
#
# This script tests:
#   - Model loading time
#   - Inference throughput vs chunk size
#   - GPU/RAM memory usage
#   - Optimal chunking strategies
#
# Usage:
#   # Quick benchmark (~5 minutes)
#   sbatch run_benchmark.sh quick
#
#   # Full benchmark (~30 minutes)
#   sbatch run_benchmark.sh full
#
#   # Custom benchmark
#   sbatch --export=SEQLENS="1000,5000",CHUNKS="50,100,1000" run_benchmark.sh custom
#===============================================================================

#SBATCH --job-name=genome_benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G

# Create log directory
mkdir -p logs
mkdir -p benchmark_results

# Load environment
module load miniforge/24.3.0-0
source ~/.bashrc
conda activate evo2_sep28

# Get benchmark mode from command line (default: quick)
MODE=${1:-quick}

# Navigate to working directory
cd /orcd/data/zhang_f/001/platawa/jan22_files

echo "========================================"
echo "Genome Scoring Benchmark"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Mode: $MODE"
echo "Start time: $(date)"
echo ""

# Print GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Run benchmark based on mode
if [ "$MODE" == "quick" ]; then
    echo "Running QUICK benchmark (3 sequences, 4 chunk sizes, ~5 min)"
    python genome_scoring_jan26_benchmark.py \
        --mode quick \
        --output_dir benchmark_results

elif [ "$MODE" == "full" ]; then
    echo "Running FULL benchmark (7 sequences, 8 chunk sizes, ~30 min)"
    python genome_scoring_jan26_benchmark.py \
        --mode full \
        --output_dir benchmark_results

elif [ "$MODE" == "custom" ]; then
    echo "Running CUSTOM benchmark"

    # Parse custom parameters from environment variables
    if [ -z "$SEQLENS" ] || [ -z "$CHUNKS" ]; then
        echo "Error: SEQLENS and CHUNKS environment variables required for custom mode"
        echo "Example: sbatch --export=SEQLENS=\"1000,5000\",CHUNKS=\"50,100\" run_benchmark.sh custom"
        exit 1
    fi

    # Convert comma-separated to space-separated
    SEQLENS_ARRAY=$(echo $SEQLENS | tr ',' ' ')
    CHUNKS_ARRAY=$(echo $CHUNKS | tr ',' ' ')

    python genome_scoring_jan26_benchmark.py \
        --mode custom \
        --sequence_lengths $SEQLENS_ARRAY \
        --chunk_sizes $CHUNKS_ARRAY \
        --n_repeats 3 \
        --output_dir benchmark_results

else
    echo "Error: Invalid mode '$MODE'"
    echo "Valid modes: quick, full, custom"
    exit 1
fi

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Benchmark Complete"
echo "========================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Results saved to: benchmark_results/"
    echo ""
    echo "View results:"
    echo "  cat benchmark_results/benchmark_summary_*.txt"
    echo "  open benchmark_results/benchmark_plots_*.png"
else
    echo "✗ Benchmark failed with exit code $EXIT_CODE"
    echo "Check logs/benchmark_$SLURM_JOB_ID.err for errors"
fi

exit $EXIT_CODE
