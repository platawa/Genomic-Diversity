#!/bin/bash
cd /orcd/data/zhang_f/001/platawa/jan31_files
GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf

for chr in chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr20 chr21 chr22 chrX; do
    JOB=$(sbatch -p mit_preemptable --cpus-per-task=4 --mem=16G -t 1:00:00 \
        -J "nplot_${chr}" \
        -o "logs/nplot_${chr}_%j.out" \
        -e "logs/nplot_${chr}_%j.err" \
        --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/plot_sae_latent.py --chrom ${chr} --results_dir results/ --normalized --plots tsne --gtf ${GTF}" \
        2>&1 | awk '{print $4}')
    echo "Submitted ${chr}: ${JOB}"
done
