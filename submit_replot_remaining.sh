#!/bin/bash
#SBATCH -J replot_annot
#SBATCH -p mit_normal
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o replot_annot_%j.out
#SBATCH -e replot_annot_%j.err

module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

GTF_HUMAN=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf
GTF_ECOLI=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf
GTF_BACILLUS=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf

# --- Remaining human annotation plots (chr4 onwards) ---
for CHR in chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr20 chr21 chr22 chrX; do
    echo "=== Annotation: $CHR ==="
    python tools/plot_tsne_by_annotation.py --chrom $CHR --sae_run results/$CHR/sae --gtf $GTF_HUMAN 2>&1
done

# --- Bacillus: 4-panel ---
echo "=== Bacillus 4-panel ==="
python replot_ecoli_4panel.py results/NC_000964.3/sae/20260324_151121_all_conf8.0_merged4of4 2>&1

# --- E. coli: 4-panel for all runs ---
echo "=== E. coli 4-panel ==="
for RUN in results/NC_000913.3/sae/20260309_132936_max50_conf3.0 results/NC_000913.3/sae/20260309_133710_max200_conf3.0 results/NC_000913.3/sae/20260309_134205_max1000_conf3.0; do
    if [ -f "$RUN/latent_analysis/data/cluster_assignments.tsv" ]; then
        echo "  Replotting $RUN..."
        python replot_ecoli_4panel.py "$RUN"
    fi
done

# E. coli annotation plots
echo "=== E. coli annotation ==="
for RUN in results/NC_000913.3/sae/20260309_132936_max50_conf3.0 results/NC_000913.3/sae/20260309_133710_max200_conf3.0 results/NC_000913.3/sae/20260309_134205_max1000_conf3.0; do
    if [ -f "$RUN/latent_analysis/data/cluster_assignments.tsv" ]; then
        python tools/plot_tsne_by_annotation.py --chrom NC_000913.3 --sae_run "$RUN" --gtf $GTF_ECOLI 2>&1
    fi
done

echo "=== ALL DONE ==="
