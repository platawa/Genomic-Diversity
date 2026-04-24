#!/bin/bash
# Render thesis Ch 4 locus figures for all 12 loci from the Feb-9 deck.
# Uses current score_chromosome.py entropy data so pipeline vintage matches Ch 5.
# CPU-only; ~15 min wall-clock.

set -euo pipefail

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
LOCI=${PROJECT}/loci_thesis.tsv

HUMAN_GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf
ECOLI_GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf
BACILLUS_GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf

mkdir -p "${LOGS}"

JOB_ID=$(sbatch --parsable \
  --job-name="thesis_figs" \
  --partition=pi_zhang_f \
  --cpus-per-task=4 \
  --mem=32G \
  --time=2:00:00 \
  --output="${LOGS}/thesis_figs_%j.out" \
  --error="${LOGS}/thesis_figs_%j.err" \
  --wrap="
set -eo pipefail
cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

python tools/make_ncRNA_figures.py \\
  --loci_tsv ${LOCI} \\
  --results_dir results/ \\
  --human_gtf ${HUMAN_GTF} \\
  --ecoli_gtf ${ECOLI_GTF} \\
  --bacillus_gtf ${BACILLUS_GTF} \\
  --pad 5000 \\
  --smooth_window 51

# Composed thesis figures (Fig 4.1, 4.2, 4.3)
LATEST=\$(ls -1dt results/_genome_wide/ncRNA_figures/*/ | head -1)
python tools/compose_chapter4_figures.py \\
  --per_locus_dir \"\${LATEST}\" \\
  --results_dir results/ \\
  --human_gtf ${HUMAN_GTF} \\
  --ecoli_gtf ${ECOLI_GTF} \\
  --reference_locus NPS

echo 'Done. Per-locus: '\"\${LATEST}\"'  Composed: '\"\${LATEST}composed/\"
")

echo "Submitted thesis figure job: ${JOB_ID}"
echo "Per-locus PNGs: results/_genome_wide/ncRNA_figures/<timestamp>_thesis/"
echo "Composed figs:  results/_genome_wide/ncRNA_figures/<timestamp>_thesis/composed/"
echo "Watch with: tail -f ${LOGS}/thesis_figs_${JOB_ID}.err"
