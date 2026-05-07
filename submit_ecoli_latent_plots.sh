#!/bin/bash
# Generate enhanced SAE latent visualizations for E. coli (NC_000913.3).
# Mirrors submit_bacillus_latent_plots.sh but adds the postnorm subdir
# (bacillus only emits raw + prenorm; we want all three for E. coli).
# Uses already-computed embeddings + clusters in:
#   results/NC_000913.3/sae/latent_analysis/data/          (raw — already populated)
#   results/NC_000913.3/sae/latent_analysis_prenorm/data/  (prenorm — populated by Task 1)
#   results/NC_000913.3/sae/latent_analysis_postnorm/data/ (postnorm — populated by Task 2)
#
# CPU-only, ~24G mem, ~30 min wall per variant.

cd /orcd/data/zhang_f/001/platawa/jan31_files
GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf

mkdir -p logs

submit() {
    local label="$1"; shift
    local subdir="$1"; shift
    local cmd="module load miniforge/24.3.0-0 && conda activate evo2_sep28 && \
cd /orcd/data/zhang_f/001/platawa/jan31_files && \
python tools/enhanced_latent_plots.py \
    --scope organism --organism ecoli \
    --gtf ${GTF} \
    --results_dir results/ \
    --latent_subdir ${subdir}"
    local jid
    jid=$(sbatch -p mit_preemptable --cpus-per-task=4 --mem=24G -t 1:30:00 \
        -J "elat_${label}" \
        -o "logs/elat_${label}_%j.out" \
        -e "logs/elat_${label}_%j.err" \
        --wrap "${cmd}" 2>&1 | awk '{print $4}')
    echo "submitted ${label} -> ${jid}  (subdir=${subdir})"
}

submit raw      latent_analysis
submit prenorm  latent_analysis_prenorm
submit postnorm latent_analysis_postnorm
