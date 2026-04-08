#!/bin/bash
#SBATCH -J replot_tsne
#SBATCH -p mit_normal
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o replot_tsne_%j.out
#SBATCH -e replot_tsne_%j.err

module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf

CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr20 chr21 chr22 chrX"

for CHR in $CHROMS; do
    echo "========================================"
    echo "Processing $CHR"
    echo "========================================"

    # Replot tsne_4panel (raw)
    if [ -d results/$CHR/sae/latent_analysis/data ]; then
        echo "  Replotting 4-panel (raw) for $CHR..."
        python tools/replot.py --run_dir results/$CHR/sae/latent_analysis/ \
            --stage latent_analysis 2>&1 | tail -5
    fi

    # Replot tsne_4panel (normalized)
    if [ -d results/$CHR/sae/latent_analysis_normalized/data ]; then
        echo "  Replotting 4-panel (normalized) for $CHR..."
        python tools/replot.py --run_dir results/$CHR/sae/latent_analysis_normalized/ \
            --stage latent_analysis 2>&1 | tail -5
    fi

    # Replot annotation t-SNE (both raw + normalized handled by the script)
    echo "  Replotting annotation t-SNE for $CHR..."
    python tools/plot_tsne_by_annotation.py --chrom $CHR --auto --gtf $GTF 2>&1 | tail -5

    echo ""
done

echo "Done replotting all chromosomes."
