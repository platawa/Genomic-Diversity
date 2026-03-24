#!/usr/bin/env bash
# submit_downstream_analysis.sh
#
# Downstream analysis pipeline — runs AFTER finalize+merge completes.
# All steps are CPU-only (no GPU needed).
#
# Pipeline:
#   Phase 1 (per-chromosome, parallel):
#     - analyze_sae_regions.py: max-pool, cosine similarity, Leiden clustering, t-SNE
#     - replot.py: normalized feature activation plots
#   Phase 2 (genome-wide, after all Phase 1 complete):
#     - normalize_sae_features.py per chromosome (needs global stats from aggregation)
#     - genome_sae_tsne.py: genome-wide t-SNE colored by annotation
#     - plot_genome_karyotype.py: karyotype visualization
#     - chromosome_analysis.py: 6 comparative analyses
#
# Usage:
#   bash submit_downstream_analysis.sh                         # run all
#   bash submit_downstream_analysis.sh --after-jobid 10843398  # chain after aggregation job
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs
GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf
FASTA=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna

CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY"

mkdir -p "${LOGS}"

# Parse optional dependency
AFTER_DEP=""
if [ "${1:-}" = "--after-jobid" ] && [ -n "${2:-}" ]; then
    AFTER_DEP="#SBATCH --dependency=afterok:${2}"
    echo "Chaining after job ${2}"
fi

echo "=========================================="
echo "Downstream Analysis Pipeline"
echo "=========================================="
echo ""

# ─────────────────────────────────────────────────────────
# PHASE 1: Per-chromosome latent analysis + replot (parallel)
# ─────────────────────────────────────────────────────────
echo "Phase 1: Per-chromosome latent analysis + replot..."
PHASE1_JIDS=""

P_IDX=0
PARTITIONS=(mit_normal mit_preemptable mit_normal mit_preemptable)

for CHROM in $CHROMS; do
    PARTITION=${PARTITIONS[$((P_IDX % ${#PARTITIONS[@]}))]}
    P_IDX=$((P_IDX + 1))

    JOB_NAME="analysis_${CHROM}"
    SBATCH="${LOGS}/.${JOB_NAME}.sbatch"

    cat > "${SBATCH}" <<SBATCH
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p ${PARTITION}
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o ${LOGS}/${JOB_NAME}_%j.out
#SBATCH -e ${LOGS}/${JOB_NAME}_%j.err
${AFTER_DEP}

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[${CHROM}] Phase 1 started at \$(date) on \$(hostname)"

# Find the latest completed SAE run for this chromosome
SAE_RUN=\$(python -c "
from results_utils import find_latest_completed
r = find_latest_completed('results/', '${CHROM}', 'sae')
print(r or '')
")

if [ -z "\${SAE_RUN}" ]; then
    echo "[${CHROM}] ERROR: No completed SAE run found, skipping"
    exit 1
fi
echo "[${CHROM}] Using SAE run: \${SAE_RUN}"

# Step 1a: Latent analysis (max-pool, cosine sim, clustering, t-SNE/UMAP)
echo "[${CHROM}] Step 1a: Latent analysis..."
if [ ! -f "\${SAE_RUN}/latent_analysis/data/maxpooled_vectors.npy" ]; then
    python analyze_sae_regions.py \\
        --input_dir "\${SAE_RUN}" \\
        --embedding both \\
        --leiden_resolution 1.0 || echo "[${CHROM}] WARNING: latent analysis failed"
else
    echo "[${CHROM}] Latent analysis already exists, skipping"
fi

# Step 1b: Replot with z-score normalization
echo "[${CHROM}] Step 1b: Replot with normalization..."
python tools/replot.py \\
    --run_dir "\${SAE_RUN}" \\
    --normalize both \\
    --gtf ${GTF} || echo "[${CHROM}] WARNING: replot failed"

# Step 1c: t-SNE by annotation (if latent analysis exists)
if [ -f "\${SAE_RUN}/latent_analysis/data/cluster_assignments.tsv" ]; then
    echo "[${CHROM}] Step 1c: t-SNE by annotation..."
    python tools/plot_tsne_by_annotation.py \\
        --sae_run "\${SAE_RUN}" \\
        --gtf ${GTF} || echo "[${CHROM}] WARNING: tsne annotation failed"
fi

echo "[${CHROM}] Phase 1 done at \$(date)"
SBATCH

    JID=$(sbatch "${SBATCH}" | awk '{print $NF}')
    echo "  ${CHROM}: job ${JID}  partition=${PARTITION}"
    PHASE1_JIDS="${PHASE1_JIDS}:${JID}"
done

echo ""

# ─────────────────────────────────────────────────────────
# PHASE 2: Genome-wide analyses (after all Phase 1 complete)
# ─────────────────────────────────────────────────────────
echo "Phase 2: Genome-wide analyses (depends on Phase 1)..."

# 2a: Normalize per-chromosome vectors using global stats
NORM_NAME="normalize_all"
NORM_SBATCH="${LOGS}/.${NORM_NAME}.sbatch"

cat > "${NORM_SBATCH}" <<SBATCH
#!/bin/bash
#SBATCH -J ${NORM_NAME}
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o ${LOGS}/${NORM_NAME}_%j.out
#SBATCH -e ${LOGS}/${NORM_NAME}_%j.err
#SBATCH --dependency=afterany${PHASE1_JIDS}

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[normalize] Started at \$(date)"

for CHROM in ${CHROMS}; do
    echo "[normalize] Processing \${CHROM}..."
    python tools/normalize_sae_features.py \\
        --chrom \${CHROM} \\
        --results_dir results/ \\
        --auto \\
        --method zscore || echo "[normalize] WARNING: \${CHROM} failed"
done

echo "[normalize] Done at \$(date)"
SBATCH

NORM_JID=$(sbatch "${NORM_SBATCH}" | awk '{print $NF}')
echo "  Normalize all: job ${NORM_JID}"

# 2b: Genome-wide t-SNE
GTSNE_NAME="genome_tsne"
GTSNE_SBATCH="${LOGS}/.${GTSNE_NAME}.sbatch"

cat > "${GTSNE_SBATCH}" <<SBATCH
#!/bin/bash
#SBATCH -J ${GTSNE_NAME}
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o ${LOGS}/${GTSNE_NAME}_%j.out
#SBATCH -e ${LOGS}/${GTSNE_NAME}_%j.err
#SBATCH --dependency=afterok:${NORM_JID}

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[genome_tsne] Started at \$(date)"

python tools/genome_sae_tsne.py \\
    --all_human \\
    --gtf ${GTF} \\
    --results_dir results/ \\
    --embedding both || echo "[genome_tsne] WARNING: failed"

echo "[genome_tsne] Done at \$(date)"
SBATCH

GTSNE_JID=$(sbatch "${GTSNE_SBATCH}" | awk '{print $NF}')
echo "  Genome t-SNE: job ${GTSNE_JID} (after normalize)"

# 2c: Karyotype plots (only needs scoring runs, can run in parallel with Phase 1)
KARY_NAME="karyotype"
KARY_SBATCH="${LOGS}/.${KARY_NAME}.sbatch"

cat > "${KARY_SBATCH}" <<SBATCH
#!/bin/bash
#SBATCH -J ${KARY_NAME}
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o ${LOGS}/${KARY_NAME}_%j.out
#SBATCH -e ${LOGS}/${KARY_NAME}_%j.err
${AFTER_DEP}

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[karyotype] Started at \$(date)"

python tools/plot_genome_karyotype.py \\
    --results_dir results/ \\
    --gtf ${GTF} \\
    --all_human || echo "[karyotype] WARNING: failed"

echo "[karyotype] Done at \$(date)"
SBATCH

KARY_JID=$(sbatch "${KARY_SBATCH}" | awk '{print $NF}')
echo "  Karyotype: job ${KARY_JID}"

# 2d: Confidence-ranked drop plots
CONF_NAME="confidence_drops"
CONF_SBATCH="${LOGS}/.${CONF_NAME}.sbatch"

cat > "${CONF_SBATCH}" <<SBATCH
#!/bin/bash
#SBATCH -J ${CONF_NAME}
#SBATCH -p mit_preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o ${LOGS}/${CONF_NAME}_%j.out
#SBATCH -e ${LOGS}/${CONF_NAME}_%j.err
${AFTER_DEP}

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[confidence_drops] Started at \$(date)"

python tools/plot_confidence_drops.py \\
    --results_dir results/ \\
    --gtf ${GTF} \\
    --all_human || echo "[confidence_drops] WARNING: failed"

echo "[confidence_drops] Done at \$(date)"
SBATCH

CONF_JID=$(sbatch "${CONF_SBATCH}" | awk '{print $NF}')
echo "  Confidence drops: job ${CONF_JID}"

# 2e: Cross-organism summary (if bacterial data exists)
CROSS_NAME="cross_organism"
CROSS_SBATCH="${LOGS}/.${CROSS_NAME}.sbatch"

cat > "${CROSS_SBATCH}" <<SBATCH
#!/bin/bash
#SBATCH -J ${CROSS_NAME}
#SBATCH -p mit_preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o ${LOGS}/${CROSS_NAME}_%j.out
#SBATCH -e ${LOGS}/${CROSS_NAME}_%j.err
${AFTER_DEP}

cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

echo "[cross_organism] Started at \$(date)"

python tools/cross_organism_summary.py \\
    --results_dir results/ || echo "[cross_organism] WARNING: failed"

echo "[cross_organism] Done at \$(date)"
SBATCH

CROSS_JID=$(sbatch "${CROSS_SBATCH}" | awk '{print $NF}')
echo "  Cross-organism: job ${CROSS_JID}"

echo ""
echo "=========================================="
echo "All downstream jobs submitted!"
echo "=========================================="
echo ""
echo "Phase 1 (parallel, per-chromosome): 24 jobs"
echo "  - Latent analysis (max-pool, clustering, t-SNE/UMAP)"
echo "  - Replot with z-score + minmax normalization"
echo "  - t-SNE by annotation"
echo ""
echo "Phase 2 (genome-wide, sequential):"
echo "  - Normalize all: ${NORM_JID} (after Phase 1)"
echo "  - Genome t-SNE:  ${GTSNE_JID} (after normalize)"
echo "  - Karyotype:     ${KARY_JID} (independent)"
echo "  - Conf drops:    ${CONF_JID} (independent)"
echo "  - Cross-organism: ${CROSS_JID} (independent)"
echo ""
echo "Monitor: squeue -u platawa"
