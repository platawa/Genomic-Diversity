#!/bin/bash
# Fill the slide-pipeline plot gap:
#   - raw UMAP (22 chroms × 1 variant × 1 embedding)
#   - prenorm  (22 chroms × 1 variant × tSNE+UMAP via --plots all)
#   - postnorm (22 chroms × 1 variant × tSNE+UMAP via --plots all)
#
# 22 chroms × 3 variants = 66 jobs. chr19 + chrY skipped per user direction.
# All compute already done in latent_analysis_*/data/; this is plotting only.
# Existing PNGs (raw tSNE, normalized tSNE+UMAP) are not touched.
#
# Usage:  cd /orcd/data/zhang_f/001/platawa/jan31_files && bash submit_fill_slide_plots.sh

cd /orcd/data/zhang_f/001/platawa/jan31_files
mkdir -p logs

GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf
CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr20 chr21 chr22 chrX)

submit_one() {
    local chrom=$1
    local variant_dir=$2     # e.g. latent_analysis, latent_analysis_prenorm, latent_analysis_postnorm
    local plots=$3           # tsne | umap | all
    local tag=$4             # short label for job name

    local input_dir="results/${chrom}/sae/${variant_dir}"
    local jobname="fill_${chrom}_${tag}"

    sbatch -p mit_preemptable --cpus-per-task=4 --mem=16G -t 1:00:00 \
        -J "${jobname}" \
        -o "logs/${jobname}_%j.out" \
        -e "logs/${jobname}_%j.err" \
        --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/plot_sae_latent.py --chrom ${chrom} --results_dir results/ --input_dir ${input_dir} --plots ${plots} --gtf ${GTF}"
}

for chr in "${CHROMS[@]}"; do
    # Gap A: raw UMAP only (don't re-render existing raw tSNE PNG)
    submit_one "${chr}" "latent_analysis" "umap" "raw_umap"
    # Gap B: prenorm tSNE + UMAP
    submit_one "${chr}" "latent_analysis_prenorm" "all" "prenorm"
    # Gap C: postnorm tSNE + UMAP
    submit_one "${chr}" "latent_analysis_postnorm" "all" "postnorm"
done

echo "Submitted 66 jobs (22 chroms × 3 variants). Monitor with: squeue -u platawa -n fill_*"
