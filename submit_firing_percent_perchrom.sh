#!/bin/bash
#SBATCH -J firing_pct_perchrom
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o logs/firing_pct_perchrom_%j.log
#SBATCH -e logs/firing_pct_perchrom_%j.err

# Per-chromosome % SAE features firing plots — fills the gap left by the
# genome-wide-only firing_percent runs. Loops serially over (variant, chrom)
# pairs to avoid hammering NFS with 88 parallel jobs (cf. prior parallel-merge
# incident). Each invocation writes results/{chr}/sae/{variant}/firing_percent/.
# Skips any (variant, chrom) that already has firing_percent/COMPLETED.

set -e
module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

# Same wide tau range as genome-wide runs for cross-mode comparison.
THRESHOLDS="0,0.01,0.05,0.1,0.25,0.5,1,2,3,5,7,10,15,20"
CHUNK_ROWS=10000

# Mode label used in plot titles. Keys = latent_analysis subdir name.
declare -A MODE_FOR_VARIANT=(
  ["latent_analysis"]="raw"
  ["latent_analysis_normalized"]="custom"
  ["latent_analysis_prenorm"]="prenorm"
  ["latent_analysis_postnorm"]="postnorm"
)

VARIANTS=("latent_analysis" "latent_analysis_normalized" "latent_analysis_prenorm" "latent_analysis_postnorm")
CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr20 chr21 chr22 chrX chrY)

n_done=0
n_skipped=0
n_failed=0
total=0

for variant in "${VARIANTS[@]}"; do
  mode="${MODE_FOR_VARIANT[$variant]}"
  for chrom in "${CHROMS[@]}"; do
    total=$((total + 1))
    in_dir="results/${chrom}/sae/${variant}"
    data_dir="${in_dir}/data"
    completed="${in_dir}/firing_percent/COMPLETED"

    if [ ! -f "${data_dir}/maxpooled_vectors.npy" ] || \
       [ ! -f "${data_dir}/embedding_tsne.npy" ] || \
       [ ! -f "${data_dir}/embedding_umap.npy" ]; then
      echo "[$(date)] SKIP ${chrom}/${variant}: missing inputs"
      n_skipped=$((n_skipped + 1))
      continue
    fi

    if [ -f "${completed}" ]; then
      echo "[$(date)] SKIP ${chrom}/${variant}: already COMPLETED"
      n_skipped=$((n_skipped + 1))
      continue
    fi

    echo "[$(date)] === ${chrom}/${variant} (mode=${mode}) ==="
    if python tools/firing_percent_plots.py \
        --input_dir "${in_dir}" \
        --mode "${mode}" \
        --thresholds "${THRESHOLDS}" \
        --chunk_rows "${CHUNK_ROWS}"; then
      n_done=$((n_done + 1))
    else
      echo "[$(date)] FAILED ${chrom}/${variant}"
      n_failed=$((n_failed + 1))
    fi
  done
done

echo "[$(date)] Summary: total=${total} done=${n_done} skipped=${n_skipped} failed=${n_failed}"
if [ "${n_failed}" -gt 0 ]; then
  exit 1
fi
echo "[$(date)] All per-chrom firing-percent runs complete."
