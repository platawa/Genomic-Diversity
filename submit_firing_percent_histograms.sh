#!/bin/bash
#SBATCH -J firing_pct_hist
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -o logs/firing_pct_hist_%j.log
#SBATCH -e logs/firing_pct_hist_%j.err

# Generate firing-percent histograms for every COMPLETED firing_percent dir
# (genome-wide and per-chrom). Idempotent: skips dirs whose histogram PNGs
# already exist unless invoked manually with --force.
#
# Also calls tools/firing_percent_latex_tables.py once at the end with the
# three latest genome-wide modes to emit publication-ready .tex snippets.

set -e
module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

n_done=0
n_skip=0
n_fail=0

run_hist() {
    local parent_dir="$1"
    local subdir="$2"
    local data_dir="${parent_dir}/${subdir}/data"
    local plots_dir="${parent_dir}/${subdir}/plots"

    if [ ! -f "${data_dir}/per_point_pct_fired.npz" ]; then
        return 0
    fi
    if [ ! -f "${parent_dir}/${subdir}/COMPLETED" ]; then
        return 0
    fi

    if [ -f "${plots_dir}/histogram_per_region_pct_fired.png" ] && \
       [ -f "${plots_dir}/histogram_per_feature_firing_rate.png" ]; then
        n_skip=$((n_skip + 1))
        return 0
    fi

    echo "[$(date)] HIST ${parent_dir}/${subdir}"
    if python tools/firing_percent_histograms.py \
        --input_dir "${parent_dir}" \
        --firing_percent_subdir "${subdir}"; then
        n_done=$((n_done + 1))
    else
        echo "[$(date)] FAILED ${parent_dir}/${subdir}"
        n_fail=$((n_fail + 1))
    fi
}

# === Genome-wide runs ===
for parent in \
    results/_genome_wide/sae_tsne/20260409_093547_23chroms_740967regions \
    results/_genome_wide/sae_tsne_prenorm/20260423_192324_23chroms_897129regions \
    results/_genome_wide/sae_tsne_postnorm/20260415_181902_23chroms_740967regions; do
    for subdir in firing_percent firing_percent_fast; do
        run_hist "${parent}" "${subdir}"
    done
done

# === Per-chrom runs ===
CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr20 chr21 chr22 chrX chrY)
VARIANTS=(latent_analysis latent_analysis_normalized latent_analysis_prenorm latent_analysis_postnorm)
for chrom in "${CHROMS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        run_hist "results/${chrom}/sae/${variant}" "firing_percent"
    done
done

echo "[$(date)] Histogram pass complete: done=${n_done} skipped=${n_skip} failed=${n_fail}"

# === LaTeX tables for the three genome-wide modes ===
TABLES_OUT="results/_genome_wide/firing_percent_tables"
echo "[$(date)] Generating LaTeX tables -> ${TABLES_OUT}"
python tools/firing_percent_latex_tables.py \
    --inputs \
        "Raw (genome-wide)=results/_genome_wide/sae_tsne/20260409_093547_23chroms_740967regions/firing_percent" \
        "Prenorm (genome-wide)=results/_genome_wide/sae_tsne_prenorm/20260423_192324_23chroms_897129regions/firing_percent" \
        "Postnorm (genome-wide)=results/_genome_wide/sae_tsne_postnorm/20260415_181902_23chroms_740967regions/firing_percent" \
    --output_dir "${TABLES_OUT}" \
    --tau_subset "0,1,2,3,5,7,10"

echo "[$(date)] Done."
if [ "${n_fail}" -gt 0 ]; then
    exit 1
fi
