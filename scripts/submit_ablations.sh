#!/bin/bash
# submit_ablations.sh — Submit all Phase 1 ablation experiments for chr22
#
# Usage:
#   ./scripts/submit_ablations.sh [--dry-run]
#
# Submits 4 SLURM jobs simultaneously:
#   1.1  No RC averaging (remove --rc_average)
#   1.2a Stitch method: mean
#   1.2b Stitch method: min
#   1.3  Chunk size sweep (multiple --max_chunk_len on 1Mbp region)

set -euo pipefail

PROJECT_DIR="/orcd/data/zhang_f/001/platawa/jan31_files"
FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna"
CONDA_ENV="evo2_sep28"
PARTITION="mit_preemptable"
CHROM="chr22"

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
    esac
done

mkdir -p "${PROJECT_DIR}/logs"

submit() {
    local name="$1"
    local script="$2"
    local sbatch_file="${PROJECT_DIR}/logs/.ablation_${name}.sbatch"
    echo "$script" > "$sbatch_file"
    if $DRY_RUN; then
        echo "[DRY-RUN] $name"
        echo "$script" | grep -E 'python|SBATCH -J' | head -5
        echo ""
    else
        local jid
        jid=$(sbatch "$sbatch_file" | awk '{print $NF}')
        echo "Submitted $name -> Job $jid"
    fi
}

# ── 1.1 No RC averaging ─────────────────────────────────────────────────────
submit "no_rc" "$(cat <<'SBATCH'
#!/bin/bash
#SBATCH -J ablation_no_rc
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -t 08:00:00
#SBATCH -o /orcd/data/zhang_f/001/platawa/jan31_files/logs/ablation_no_rc_%j.out
#SBATCH -e /orcd/data/zhang_f/001/platawa/jan31_files/logs/ablation_no_rc_%j.err

cd /orcd/data/zhang_f/001/platawa/jan31_files
module load miniforge/24.3.0-0
conda activate evo2_sep28

python score_chromosome.py \
    --chrom chr22 \
    --fasta /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna \
    --output_dir ./results \
    --n_gpus 1 \
    --compute_logprobs \
    --auto_chunk_size
SBATCH
)"

# ── 1.2a Stitch method: mean ────────────────────────────────────────────────
submit "stitch_mean" "$(cat <<'SBATCH'
#!/bin/bash
#SBATCH -J ablation_stitch_mean
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -t 08:00:00
#SBATCH -o /orcd/data/zhang_f/001/platawa/jan31_files/logs/ablation_stitch_mean_%j.out
#SBATCH -e /orcd/data/zhang_f/001/platawa/jan31_files/logs/ablation_stitch_mean_%j.err

cd /orcd/data/zhang_f/001/platawa/jan31_files
module load miniforge/24.3.0-0
conda activate evo2_sep28

python score_chromosome.py \
    --chrom chr22 \
    --fasta /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna \
    --output_dir ./results \
    --n_gpus 1 \
    --rc_average --compute_logprobs \
    --auto_chunk_size \
    --stitch_method mean
SBATCH
)"

# ── 1.2b Stitch method: min ─────────────────────────────────────────────────
submit "stitch_min" "$(cat <<'SBATCH'
#!/bin/bash
#SBATCH -J ablation_stitch_min
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -t 08:00:00
#SBATCH -o /orcd/data/zhang_f/001/platawa/jan31_files/logs/ablation_stitch_min_%j.out
#SBATCH -e /orcd/data/zhang_f/001/platawa/jan31_files/logs/ablation_stitch_min_%j.err

cd /orcd/data/zhang_f/001/platawa/jan31_files
module load miniforge/24.3.0-0
conda activate evo2_sep28

python score_chromosome.py \
    --chrom chr22 \
    --fasta /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna \
    --output_dir ./results \
    --n_gpus 1 \
    --rc_average --compute_logprobs \
    --auto_chunk_size \
    --stitch_method min
SBATCH
)"

# ── 1.3 Chunk size sweep (1Mbp region) ──────────────────────────────────────
# Test chunk sizes: 5000, 10000, 15000, 25000, 50000 on a 1Mbp sub-region
# Uses chr22:20000000-21000000 (a well-annotated region)
submit "chunk_sweep" "$(cat <<'SBATCH'
#!/bin/bash
#SBATCH -J ablation_chunk_sweep
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -t 04:00:00
#SBATCH -o /orcd/data/zhang_f/001/platawa/jan31_files/logs/ablation_chunk_sweep_%j.out
#SBATCH -e /orcd/data/zhang_f/001/platawa/jan31_files/logs/ablation_chunk_sweep_%j.err

cd /orcd/data/zhang_f/001/platawa/jan31_files
module load miniforge/24.3.0-0
conda activate evo2_sep28

FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna"

for CHUNK_SIZE in 5000 10000 15000 25000 50000; do
    echo "========================================"
    echo "Chunk size: ${CHUNK_SIZE}"
    echo "========================================"
    python score_chromosome.py \
        --chrom chr22 \
        --start 20000000 --end 21000000 \
        --fasta ${FASTA} \
        --output_dir ./results \
        --n_gpus 1 \
        --rc_average --compute_logprobs \
        --max_chunk_len ${CHUNK_SIZE}
done

echo "Chunk sweep complete. Compare with:"
echo "  python tools/compare_ablations.py --baseline <baseline_entropy.npz> --ablations <chunk_*_entropy.npz>"
SBATCH
)"

echo ""
echo "All ablation jobs submitted."
echo "After completion, compare results with:"
echo "  python tools/compare_ablations.py \\"
echo "    --baseline results/chr22/scoring/<baseline>/data/entropy.npz \\"
echo "    --ablations results/chr22/scoring/<no_rc>/data/entropy.npz \\"
echo "                results/chr22/scoring/<stitch_mean>/data/entropy.npz \\"
echo "                results/chr22/scoring/<stitch_min>/data/entropy.npz \\"
echo "    --labels baseline no_rc stitch_mean stitch_min"
