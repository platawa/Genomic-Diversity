#!/bin/bash
# Submit genome aggregation + genome-wide analysis
# Depends on all 22 per-chromosome analysis jobs completing

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs

# Per-chromosome analysis job IDs (11017475-11017496)
PER_CHR_JOBS="11017475:11017476:11017477:11017478:11017479:11017480:11017481:11017482:11017483:11017484:11017485:11017486:11017487:11017488:11017489:11017490:11017491:11017492:11017493:11017494:11017495:11017496"

echo "Submitting aggregation job (depends on all 22 per-chromosome analysis)..."
echo ""

# Aggregation job — computes global stats across all chromosomes
AGGR_JOB=$(sbatch --parsable \
  --job-name="aggregate_genome_stats" \
  --partition=pi_zhang_f \
  --cpus-per-task=16 \
  --mem=256G \
  --time=4:00:00 \
  --dependency=afterok:${PER_CHR_JOBS} \
  --output="${LOGS}/aggregate_genome_stats_%j.out" \
  --error="${LOGS}/aggregate_genome_stats_%j.err" \
  --wrap="
set -eo pipefail
cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo 'Aggregating genome-wide SAE statistics...'
python tools/aggregate_genome_sae_stats.py \\
  --results_dir results/ \\
  --all_human \\
  --force
")

echo "  Aggregation: Job ${AGGR_JOB}"

# Genome-wide analysis — depends on aggregation completing
echo ""
echo "Submitting genome-wide analysis (depends on aggregation)..."

GENOMEWIDE_JOB=$(sbatch --parsable \
  --job-name="genomewide_analysis" \
  --partition=pi_zhang_f \
  --cpus-per-task=16 \
  --mem=256G \
  --time=6:00:00 \
  --dependency=afterok:${AGGR_JOB} \
  --output="${LOGS}/genomewide_analysis_%j.out" \
  --error="${LOGS}/genomewide_analysis_%j.err" \
  --wrap="
set -eo pipefail
cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo 'Running genome-wide analysis (t-SNE, karyotype, comparisons)...'
python tools/genome_sae_tsne.py \\
  --all_human \\
  --output_dir results/human_genome_analysis/
")

echo "  Genome-wide analysis: Job ${GENOMEWIDE_JOB}"

echo ""
echo "Pipeline complete:"
echo "  Phase 1: Per-chromosome analysis (jobs 11017475-11017496) → maxpooled vectors"
echo "  Phase 2: Aggregation (job ${AGGR_JOB}) → global stats"
echo "  Phase 3: Genome-wide analysis (job ${GENOMEWIDE_JOB}) → t-SNE + karyotype"
