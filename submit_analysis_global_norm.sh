#!/bin/bash
# Complete analysis pipeline with GLOBAL normalization
# Phase 1: Aggregation (computes global stats)
# Phase 2: Per-chromosome analysis (with global stats)
# Phase 3: Genome-wide analysis (with global stats)

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf

echo "=========================================="
echo "Complete Analysis Pipeline - Global Normalization"
echo "=========================================="
echo ""

# ============================================================================
# Phase 1: Aggregation — Compute global stats across all chromosomes
# ============================================================================
echo "Phase 1: Computing global normalization stats..."

AGGR_JOB=$(sbatch --parsable \
  --job-name="aggregate_stats_global" \
  --partition=pi_zhang_f \
  --cpus-per-task=16 \
  --mem=256G \
  --time=4:00:00 \
  --output="${LOGS}/aggregate_stats_global_%j.out" \
  --error="${LOGS}/aggregate_stats_global_%j.err" \
  --wrap="
set -eo pipefail
cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo 'Computing global SAE statistics for genome-wide normalization...'
python tools/aggregate_genome_sae_stats.py \\
  --results_dir results/ \\
  --all_human \\
  --force
")

echo "  Aggregation: Job ${AGGR_JOB}"

# ============================================================================
# Phase 2: Per-chromosome analysis — with global stats
# ============================================================================
echo ""
echo "Phase 2: Per-chromosome analysis with global normalization..."

CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22)

for CHR in "${CHROMS[@]}"; do
    JOB_ID=$(sbatch --parsable \
      --job-name="analyze_${CHR}_global" \
      --partition=pi_zhang_f \
      --cpus-per-task=8 \
      --mem=128G \
      --time=4:00:00 \
      --dependency=afterok:${AGGR_JOB} \
      --output="${LOGS}/analyze_${CHR}_global_%j.out" \
      --error="${LOGS}/analyze_${CHR}_global_%j.err" \
      --wrap="
set -eo pipefail
cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

# Find latest merge directory
MERGE_DIR=\$(ls -td results/${CHR}/sae/*merged*/ 2>/dev/null | head -1)
if [ -z \"\$MERGE_DIR\" ]; then
  echo \"ERROR: No merge directory found for ${CHR}\"
  exit 1
fi

# Find global stats (computed by aggregation job)
GLOBAL_STATS=\$(ls -t results/*/sae_global_stats/*/data/global_sae_stats.npz 2>/dev/null | head -1)

echo \"Analyzing \${MERGE_DIR} with global normalization...\"
python tools/analyze_sae_regions.py \\
  --input_dir \"\${MERGE_DIR}\" \\
  --embedding both \\
  --leiden_resolution 1.0 \\
  --global_stats \"\${GLOBAL_STATS}\"
")

    echo "  ${CHR}: Job ${JOB_ID}"
done

# ============================================================================
# Phase 3: Genome-wide analysis — with global stats
# ============================================================================
echo ""
echo "Phase 3: Genome-wide analysis with global normalization..."

# Collect all per-chromosome job IDs for dependency
echo "  (will depend on all per-chromosome analysis completing)"

GENOMEWIDE_JOB=$(sbatch --parsable \
  --job-name="genomewide_global" \
  --partition=pi_zhang_f \
  --cpus-per-task=16 \
  --mem=256G \
  --time=6:00:00 \
  --dependency=afterok:${AGGR_JOB} \
  --output="${LOGS}/genomewide_global_%j.out" \
  --error="${LOGS}/genomewide_global_%j.err" \
  --wrap="
set -eo pipefail
cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

# Find global stats
GLOBAL_STATS=\$(ls -t results/*/sae_global_stats/*/data/global_sae_stats.npz 2>/dev/null | head -1)

echo 'Running genome-wide analysis with global normalization...'
python tools/genome_sae_tsne.py \\
  --all_human \\
  --gtf ${GTF} \\
  --results_dir results/human_genome_analysis/ \\
  --embedding both \\
  --global_stats \"\${GLOBAL_STATS}\"
")

echo "  Genome-wide: Job ${GENOMEWIDE_JOB}"

echo ""
echo "=========================================="
echo "Pipeline Summary:"
echo "=========================================="
echo "Phase 1: Aggregation (${AGGR_JOB}) → global stats"
echo "Phase 2: Per-chromosome analysis (${#CHROMS[@]} jobs) → with global normalization"
echo "Phase 3: Genome-wide analysis (${GENOMEWIDE_JOB}) → t-SNE, karyotype, comparisons"
echo ""
echo "All analyses use GLOBAL NORMALIZATION across entire human genome"
