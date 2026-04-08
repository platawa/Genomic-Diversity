#!/bin/bash
#SBATCH -J replot_reglen
#SBATCH -p mit_normal
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o replot_reglen_%j.out
#SBATCH -e replot_reglen_%j.err

module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf

for CHR in chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr20 chr21 chr22 chrX; do
    echo "=== $CHR ==="
    python tools/plot_tsne_by_annotation.py --chrom $CHR --sae_run results/$CHR/sae --gtf $GTF 2>&1
done

echo "=== ALL DONE ==="
