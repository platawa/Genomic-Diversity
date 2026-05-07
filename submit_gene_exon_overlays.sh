#!/bin/bash
# Generate gene-exon overlay plots for HBB (chr11), EGFR (chr7), NPS (chr10)
# with multiple coloring modes.
#
# Plot matrix:
#   All 3 genes × 2 scopes × 2 variants × 3 norms × [single, position] = 72 plots
#   EGFR only   × 2 scopes × 2 variants × 3 norms × [gradient, numbered] = 24 plots
#   Total: 96 plots per run
#
# Usage (login node): bash submit_gene_exon_overlays.sh
# Usage (cluster):    sbatch submit_gene_exon_overlays.sh
#
#SBATCH -J gene_exon_overlays
#SBATCH -p mit_preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o logs/gene_exon_overlays_%j.out
#SBATCH -e logs/gene_exon_overlays_%j.err

set -eo pipefail

PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
cd "${PROJECT}"

GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"
LOCI="${PROJECT}/loci_hbb_egfr_nps.tsv"

# Activate env under SLURM. conda's activate scripts reference a few unset
# shell variables, so we leave `set -u` off (the rest is guarded by -e and
# pipefail).
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module load miniforge/24.3.0-0
    conda activate evo2_sep28
fi

TS=$(date +%Y%m%d_%H%M%S)
OUT="${PROJECT}/results/_gene_exon_overlays/${TS}"
mkdir -p "${OUT}" logs

# Find the most recent COMPLETED run under a parent that has the required file.
find_completed_run() {
    local parent=$1
    local required=$2
    if [[ ! -d "${parent}" ]]; then
        return 0
    fi
    ls -1dt "${parent}"/*/ 2>/dev/null | while read -r d; do
        d=${d%/}
        if [[ -f "${d}/COMPLETED" && -f "${d}/data/${required}" ]]; then
            echo "${d}/data"
            return 0
        fi
    done | head -n1
}

declare -A EMB_DIR_TSNE EMB_DIR_UMAP
for norm in prenorm postnorm raw; do
    case "${norm}" in
        prenorm)  parent="${PROJECT}/results/_genome_wide/sae_tsne_prenorm" ;;
        postnorm) parent="${PROJECT}/results/_genome_wide/sae_tsne_postnorm" ;;
        raw)      parent="${PROJECT}/results/_genome_wide/sae_tsne" ;;
    esac
    EMB_DIR_TSNE[${norm}]=$(find_completed_run "${parent}" embedding_tsne.npy)
    EMB_DIR_UMAP[${norm}]=$(find_completed_run "${parent}" embedding_umap.npy)
    echo "norm=${norm}  tsne=${EMB_DIR_TSNE[${norm}]:-<none>}  umap=${EMB_DIR_UMAP[${norm}]:-<none>}"
done

# Record input provenance.
{
    echo "{"
    echo "  \"generated_at\": \"$(date -Iseconds)\","
    echo "  \"loci\": \"${LOCI}\","
    echo "  \"gtf\": \"${GTF}\","
    echo "  \"color_modes\": [\"single\", \"position\", \"gradient (EGFR only)\", \"numbered (EGFR only)\"],"
    echo "  \"sources\": {"
    first=1
    for norm in prenorm postnorm raw; do
        for variant in tsne umap; do
            var=$( [[ ${variant} == tsne ]] && echo "EMB_DIR_TSNE" || echo "EMB_DIR_UMAP" )
            eval "val=\${${var}[${norm}]:-}"
            [[ ${first} -eq 0 ]] && echo ","
            printf "    \"%s_%s\": \"%s\"" "${norm}" "${variant}" "${val}"
            first=0
        done
    done
    echo ""
    echo "  }"
    echo "}"
} > "${OUT}/source.json"

n_done=0
n_skip=0
n_fail=0

# Helper: run one plot invocation.
run_plot() {
    local gene=$1 scope=$2 variant=$3 norm=$4 mode=$5
    local var=$( [[ ${variant} == tsne ]] && echo "EMB_DIR_TSNE" || echo "EMB_DIR_UMAP" )
    eval "local dir=\${${var}[${norm}]:-}"
    local tag="${gene}/${scope}/${variant}/${norm}/${mode}"
    if [[ -z "${dir}" ]]; then
        echo "SKIP ${tag}: no embedding dir"
        n_skip=$((n_skip + 1))
        return
    fi
    local out_dir="${OUT}/${gene}/${mode}"
    mkdir -p "${out_dir}"
    local out_png="${out_dir}/${scope}_${variant}_${norm}.png"
    echo "RUN  ${tag}  -> ${out_png}"
    if python tools/plot_gene_exon_overlay.py \
            --loci "${LOCI}" \
            --gtf "${GTF}" \
            --embedding-dir "${dir}" \
            --variant "${variant}" \
            --scope "${scope}" \
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
    for scope in whole chromosome; do
        for variant in tsne umap; do
            for norm in prenorm postnorm raw; do
                for mode in single position; do
                    run_plot "${gene}" "${scope}" "${variant}" "${norm}" "${mode}"
                done
            done
        done
    done
done

# 2) EGFR only × {gradient, numbered}
for scope in whole chromosome; do
    for variant in tsne umap; do
        for norm in prenorm postnorm raw; do
            for mode in gradient numbered; do
                run_plot "EGFR" "${scope}" "${variant}" "${norm}" "${mode}"
            done
        done
    done
done

cat > "${OUT}/COMPLETED" <<EOF
{
  "completed_at": "$(date -Iseconds)",
  "script": "submit_gene_exon_overlays.sh",
  "n_plots_done": ${n_done},
  "n_skipped": ${n_skip},
  "n_failed": ${n_fail}
}
EOF

echo ""
echo "=== Summary ==="
echo "Output dir: ${OUT}"
echo "Plots generated: ${n_done}  skipped: ${n_skip}  failed: ${n_fail}"
