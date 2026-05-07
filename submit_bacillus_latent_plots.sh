#!/bin/bash
# Generate enhanced SAE latent visualizations for Bacillus subtilis (NC_000964.3).
# Uses already-computed embeddings + clusters in:
#   results/NC_000964.3/sae/latent_analysis/data/         (legacy/raw norm)
#   results/NC_000964.3/sae/latent_analysis_prenorm/data/ (prenorm)
# Both have empty plots/ dirs as of 2026-04-30; this fills them.
#
# CPU-only, ~16G mem, ~30 min wall per variant.

cd /orcd/data/zhang_f/001/platawa/jan31_files
GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf

mkdir -p logs

submit() {
    local label="$1"; shift
    local subdir="$1"; shift
    local cmd="module load miniforge/24.3.0-0 && conda activate evo2_sep28 && \
cd /orcd/data/zhang_f/001/platawa/jan31_files && \
python tools/enhanced_latent_plots.py \
    --scope organism --organism bacillus \
    --gtf ${GTF} \
    --results_dir results/ \
    --latent_subdir ${subdir}"
    local jid
    jid=$(sbatch -p mit_preemptable --cpus-per-task=4 --mem=24G -t 1:30:00 \
        -J "blat_${label}" \
        -o "logs/blat_${label}_%j.out" \
        -e "logs/blat_${label}_%j.err" \
        --wrap "${cmd}" 2>&1 | awk '{print $4}')
    echo "submitted ${label} -> ${jid}  (subdir=${subdir})"
}

submit raw     latent_analysis
submit prenorm latent_analysis_prenorm
