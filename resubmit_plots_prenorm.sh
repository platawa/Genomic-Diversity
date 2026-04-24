#!/bin/bash
# Resubmit PRENORM enhanced plots only (clustering already completed)
# Fix: adds --scope chromosome which was missing from the original submission
# chr1/2/9 depend on prenorm pooling jobs still running
cd /orcd/data/zhang_f/001/platawa/jan31_files

GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"

# Chroms that already have prenorm latent data (no dependency needed)
READY_CHROMS="chr3 chr4 chr5 chr6 chr7 chr8 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr20 chr21 chr22 chrX"

# Chroms still waiting on pooling (prenorm_chr1=11591339, prenorm_chr2=11591340, prenorm_chr9=11591341)
PENDING_CHROMS="chr1 chr2 chr9"
POOL_DEPS="11591339:11591340:11591341"

echo "=== Submitting PRENORM plot-only jobs (ready chroms) ==="
for chr in $READY_CHROMS; do
    LATENT_DIR="results/${chr}/sae/latent_analysis_prenorm"
    if [ ! -d "$LATENT_DIR" ]; then
        echo "  SKIP $chr: no latent_analysis_prenorm directory"
        continue
    fi

    JOB=$(sbatch -p mit_preemptable --cpus-per-task=4 --mem=16G -t 1:00:00 \
        -J "prplot_${chr}" \
        -o "logs/prplot_${chr}_%j.out" \
        -e "logs/prplot_${chr}_%j.err" \
        --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/enhanced_latent_plots.py --scope chromosome --chrom ${chr} --organism human --gtf ${GTF} --results_dir results/ --latent_subdir latent_analysis_prenorm" \
        2>&1 | awk '{print $4}')
    echo "  ${chr}: ${JOB}"
done

echo ""
echo "=== Submitting PRENORM plot-only jobs (pending chroms, depend on pooling) ==="
echo "  Note: These need pooling + clustering to complete first."
echo "  Pooling jobs: prenorm_chr1=11591339, prenorm_chr2=11591340, prenorm_chr9=11591341"
echo "  These plot jobs will need to be submitted manually after clustering completes for chr1/2/9."
echo "  (The prclust_chr1/2/9 jobs are already queued with pooling dependencies.)"
