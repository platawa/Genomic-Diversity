#!/bin/bash
#SBATCH -J human_annotation_plots
#SBATCH -p pi_zhang_f
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/human_annotation_plots_%j.out
#SBATCH -e logs/human_annotation_plots_%j.err

set -e

module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

GTF="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"

# Array of all human chromosomes with their latest merged SAE runs
declare -A CHROMS=(
    [chr1]="results/chr1/sae/20260323_201618_all_conf8.0_merged30of36"
    [chr2]="results/chr2/sae/20260323_174554_all_conf8.0_merged31of36"
    [chr3]="results/chr3/sae/20260323_154550_all_conf8.0_merged31of36"
    [chr4]="results/chr4/sae/20260323_150657_all_conf8.0_merged31of36"
    [chr5]="results/chr5/sae/20260323_153638_all_conf8.0_merged31of36"
    [chr6]="results/chr6/sae/20260323_152311_all_conf8.0_merged32of36"
    [chr7]="results/chr7/sae/20260323_163020_all_conf8.0_merged32of36"
    [chr8]="results/chr8/sae/20260323_142201_all_conf8.0_merged32of36"
    [chr10]="results/chr10/sae/20260323_163413_all_conf8.0_merged36of36"
    [chr11]="results/chr11/sae/20260323_141620_all_conf8.0_merged36of36"
    [chr12]="results/chr12/sae/20260323_133553_all_conf8.0_merged37of36"
    [chr13]="results/chr13/sae/20260323_201617_all_conf8.0_merged39of36"
    [chr14]="results/chr14/sae/20260323_165256_all_conf8.0_merged40of36"
    [chr16]="results/chr16/sae/20260323_161936_all_conf8.0_merged45of36"
    [chr17]="results/chr17/sae/20260323_212221_all_conf8.0_merged36of36"
    [chr18]="results/chr18/sae/20260323_133553_all_conf8.0_merged31of36"
    [chr20]="results/chr20/sae/20260323_133932_all_conf8.0_merged45of36"
    [chr21]="results/chr21/sae/20260323_201617_all_conf8.0_merged55of36"
    [chr22]="results/chr22/sae/20260323_153411_all_conf8.0_merged42of36"
)

echo "[$(date)] Starting annotation t-SNE plots for all human chromosomes..."

for chrom in "${!CHROMS[@]}"; do
    sae_run="${CHROMS[$chrom]}"

    if [ ! -d "$sae_run" ]; then
        echo "[$(date)] WARNING: $sae_run not found, skipping $chrom"
        continue
    fi

    echo "[$(date)] Processing $chrom..."

    # Check if latent analysis already exists
    if [ ! -f "$sae_run/latent_analysis/data/cluster_assignments.tsv" ]; then
        echo "[$(date)]   - Computing latent analysis..."
        python tools/analyze_sae_regions.py \
            --input_dir "$sae_run" \
            --embedding both \
            --leiden_resolution 0.5 \
            2>&1 | tee -a logs/human_latent_analysis.log
    else
        echo "[$(date)]   - Latent analysis already complete"
    fi

    # Generate annotation plots
    echo "[$(date)]   - Generating annotation-colored t-SNE plots..."
    python tools/plot_tsne_by_annotation.py \
        --sae_run "$sae_run" \
        --gtf "$GTF" \
        --chrom "$chrom" \
        2>&1 | tee -a logs/human_annotation_tsne.log

    echo "[$(date)]   ✓ $chrom complete"
done

echo "[$(date)] All human chromosome annotation plots COMPLETED"
