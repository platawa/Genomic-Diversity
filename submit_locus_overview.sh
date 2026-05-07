#!/bin/bash
# Render thesis-ready cross-cutting locus overviews for HBB, NPS, EGFR.
# Outputs entropy_overview.png (2 rows: Raw/Smoothed) and
# drops_overview.png (6 rows: Local baseline, Derivative, CUSUM, Z-score,
# Window mean shift, MAD), each spanning the three loci as columns.
# CPU-only; ~3-5 min wall-clock.

set -euo pipefail

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
LOCI=${PROJECT}/loci_thesis.tsv

HUMAN_GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf
ECOLI_GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf
BACILLUS_GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf

mkdir -p "${LOGS}"

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${PROJECT}/results/_genome_wide/locus_overview/${TS}_HBB_NPS_EGFR

JOB_ID=$(sbatch --parsable \
  --job-name="locus_overview" \
  --partition=pi_zhang_f \
  --cpus-per-task=4 \
  --mem=16G \
  --time=0:30:00 \
  --output="${LOGS}/locus_overview_%j.out" \
  --error="${LOGS}/locus_overview_%j.err" \
  --wrap="
set -eo pipefail
cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

python tools/compose_locus_overview.py \\
  --loci_tsv ${LOCI} \\
  --loci HBB NPS EGFR \\
  --results_dir results/ \\
  --human_gtf ${HUMAN_GTF} \\
  --ecoli_gtf ${ECOLI_GTF} \\
  --bacillus_gtf ${BACILLUS_GTF} \\
  --out_dir ${OUT_DIR} \\
  --pad 5000 \\
  --smooth_window 51 \\
  --annotate_top_n 0

echo 'Output: '${OUT_DIR}
")

echo "Submitted locus overview job: ${JOB_ID}"
echo "Output dir: ${OUT_DIR}"
echo "Watch:      tail -f ${LOGS}/locus_overview_${JOB_ID}.err"
