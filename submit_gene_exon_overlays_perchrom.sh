#!/bin/bash
# Generate gene-exon overlay plots for HBB (chr11), EGFR (chr7), NPS (chr10)
# against the *per-chromosome* SAE latent embeddings (results/<chrom>/sae/
# latent_analysis{,_normalized,_prenorm,_postnorm}/), as opposed to the
# genome-wide embeddings that submit_gene_exon_overlays.sh targets.
#
# Plot matrix:
#   3 genes  × 4 norms × 2 variants × {single, position}      = 48 plots
#   EGFR only × 4 norms × 2 variants × {gradient, numbered}    = 16 plots
#   Total: 64 plots
#
# Per-chrom cluster_assignments.tsv files have no `chrom` column (the chrom
# is implicit in the path), so we pass --scope whole — every row in the file
# is the gene's chromosome by construction.
#
# Usage (login node): bash submit_gene_exon_overlays_perchrom.sh
# Usage (cluster):    sbatch submit_gene_exon_overlays_perchrom.sh

#SBATCH -J gene_exon_overlays_perchrom
#SBATCH -p mit_preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o logs/gene_exon_overlays_perchrom_%j.out
#SBATCH -e logs/gene_exon_overlays_perchrom_%j.err

set -eo pipefail

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
cd "${PROJECT}"

GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"
LOCI="${PROJECT}/loci_hbb_egfr_nps.tsv"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module load miniforge/24.3.0-0
    conda activate evo2_sep28
fi

TS=$(date +%Y%m%d_%H%M%S)
OUT="${PROJECT}/results/_gene_exon_overlays_perchrom/${TS}"
mkdir -p "${OUT}" logs

# Gene -> chromosome map (matches loci_hbb_egfr_nps.tsv).
declare -A GENE_CHROM
GENE_CHROM[HBB]=chr11
GENE_CHROM[EGFR]=chr7
GENE_CHROM[NPS]=chr10

NORMS=(raw normalized prenorm postnorm)
declare -A NORM_SUBDIR
NORM_SUBDIR[raw]=latent_analysis
NORM_SUBDIR[normalized]=latent_analysis_normalized
NORM_SUBDIR[prenorm]=latent_analysis_prenorm
NORM_SUBDIR[postnorm]=latent_analysis_postnorm

# Provenance.
{
    echo "{"
    echo "  \"generated_at\": \"$(date -Iseconds)\","
    echo "  \"loci\": \"${LOCI}\","
    echo "  \"gtf\": \"${GTF}\","
    echo "  \"color_modes\": [\"single\", \"position\", \"gradient (EGFR only)\", \"numbered (EGFR only)\"],"
    echo "  \"scope\": \"whole (per-chrom data is already chromosome-scoped)\","
    echo "  \"per_chrom_sources\": {"
    first=1
    for gene in HBB EGFR NPS; do
        chrom=${GENE_CHROM[${gene}]}
        for norm in "${NORMS[@]}"; do
            sub=${NORM_SUBDIR[${norm}]}
            d="${PROJECT}/results/${chrom}/sae/${sub}/data"
            (( first )) || echo ","
            first=0
            printf '    "%s_%s": "%s"' "${gene}" "${norm}" "${d}"
        done
    done
    echo ""
    echo "  }"
    echo "}"
} > "${OUT}/source.json"

n_done=0
n_skip=0
n_fail=0

run_plot() {
    local gene=$1 variant=$2 norm=$3 mode=$4
    local chrom=${GENE_CHROM[${gene}]}
    local sub=${NORM_SUBDIR[${norm}]}
    local dir="${PROJECT}/results/${chrom}/sae/${sub}/data"
    local tag="${gene}/${chrom}/${variant}/${norm}/${mode}"
    if [[ ! -f "${dir}/cluster_assignments.tsv" || ! -f "${dir}/embedding_${variant}.npy" ]]; then
        echo "SKIP ${tag}: missing inputs in ${dir}"
        n_skip=$((n_skip + 1))
        return
    fi
    local out_dir="${OUT}/${gene}/${mode}"
    mkdir -p "${out_dir}"
    local out_png="${out_dir}/chromosome_${variant}_${norm}.png"
    echo "RUN  ${tag}  -> ${out_png}"
    if python tools/plot_gene_exon_overlay.py \
            --loci "${LOCI}" \
            --gtf "${GTF}" \
            --embedding-dir "${dir}" \
            --variant "${variant}" \
            --scope whole \
            --gene-name "${gene}" \
            --norm-label "${norm}" \
            --color-mode "${mode}" \
            --output "${out_png}"; then
        n_done=$((n_done + 1))
    else
        echo "FAIL ${tag}"
        n_fail=$((n_fail + 1))
    fi
}

# 1) All three genes × {single, position}
for gene in HBB EGFR NPS; do
    for variant in tsne umap; do
        for norm in "${NORMS[@]}"; do
            for mode in single position; do
                run_plot "${gene}" "${variant}" "${norm}" "${mode}"
            done
        done
    done
done

# 2) EGFR only × {gradient, numbered}
for variant in tsne umap; do
    for norm in "${NORMS[@]}"; do
        for mode in gradient numbered; do
            run_plot "EGFR" "${variant}" "${norm}" "${mode}"
        done
    done
done

cat > "${OUT}/COMPLETED" <<EOF
{
  "completed_at": "$(date -Iseconds)",
  "script": "submit_gene_exon_overlays_perchrom.sh",
  "n_plots_done": ${n_done},
  "n_skipped": ${n_skip},
  "n_failed": ${n_fail}
}
EOF

echo ""
echo "=== Summary ==="
echo "Output dir: ${OUT}"
echo "Plots generated: ${n_done}  skipped: ${n_skip}  failed: ${n_fail}"
