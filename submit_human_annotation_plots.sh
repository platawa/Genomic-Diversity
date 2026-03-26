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

# Array of all human chromosomes with their latest merged SAE runs + global stats paths
declare -A CHROMS=(
    [chr1]="results/chr1/sae/20260323_201618_all_conf8.0_merged30of36|"  # No global stats for chr1
    [chr2]="results/chr2/sae/20260323_174554_all_conf8.0_merged31of36|results/chr2/sae_global_stats/20260317_214405_fused_minmax/data/global_sae_stats.npz"
    [chr3]="results/chr3/sae/20260323_154550_all_conf8.0_merged31of36|results/chr3/sae_global_stats/20260317_212253_fused_minmax/data/global_sae_stats.npz"
    [chr4]="results/chr4/sae/20260323_150657_all_conf8.0_merged31of36|results/chr4/sae_global_stats/20260317_205652_fused_minmax/data/global_sae_stats.npz"
    [chr5]="results/chr5/sae/20260323_153638_all_conf8.0_merged31of36|results/chr5/sae_global_stats/20260317_205046_fused_minmax/data/global_sae_stats.npz"
    [chr6]="results/chr6/sae/20260323_152311_all_conf8.0_merged32of36|results/chr6/sae_global_stats/20260317_193955_fused_minmax/data/global_sae_stats.npz"
    [chr7]="results/chr7/sae/20260323_163020_all_conf8.0_merged32of36|results/chr7/sae_global_stats/20260317_193502_fused_minmax/data/global_sae_stats.npz"
    [chr8]="results/chr8/sae/20260323_142201_all_conf8.0_merged32of36|results/chr8/sae_global_stats/20260317_191602_fused_minmax/data/global_sae_stats.npz"
    [chr10]="results/chr10/sae/20260323_163413_all_conf8.0_merged36of36|results/chr10/sae_global_stats/20260322_163713_minmax/data/global_sae_stats.npz"
    [chr11]="results/chr11/sae/20260323_141620_all_conf8.0_merged36of36|results/chr11/sae_global_stats/20260322_163720_minmax/data/global_sae_stats.npz"
    [chr12]="results/chr12/sae/20260323_133553_all_conf8.0_merged37of36|results/chr12/sae_global_stats/20260322_181117_minmax/data/global_sae_stats.npz"
    [chr13]="results/chr13/sae/20260323_201617_all_conf8.0_merged39of36|results/chr13/sae_global_stats/20260322_180126_minmax/data/global_sae_stats.npz"
    [chr14]="results/chr14/sae/20260323_165256_all_conf8.0_merged40of36|results/chr14/sae_global_stats/20260322_175642_minmax/data/global_sae_stats.npz"
    [chr16]="results/chr16/sae/20260323_161936_all_conf8.0_merged45of36|results/chr16/sae_global_stats/20260322_174616_minmax/data/global_sae_stats.npz"
    [chr17]="results/chr17/sae/20260323_212221_all_conf8.0_merged36of36|results/chr17/sae_global_stats/20260316_230123_minmax/data/global_sae_stats.npz"
    [chr18]="results/chr18/sae/20260323_133553_all_conf8.0_merged31of36|results/chr18/sae_global_stats/20260316_225631_minmax/data/global_sae_stats.npz"
    [chr20]="results/chr20/sae/20260323_133932_all_conf8.0_merged45of36|results/chr20/sae_global_stats/20260316_222228_minmax/data/global_sae_stats.npz"
    [chr21]="results/chr21/sae/20260323_201617_all_conf8.0_merged55of36|results/chr21/sae_global_stats/20260316_181513_minmax/data/global_sae_stats.npz"
    [chr22]="results/chr22/sae/20260323_153411_all_conf8.0_merged42of36|results/chr22/sae_global_stats/20260316_182455_minmax/data/global_sae_stats.npz"
)

echo "[$(date)] Starting annotation t-SNE plots for all human chromosomes..."

for chrom in "${!CHROMS[@]}"; do
    IFS='|' read -r sae_run global_stats <<< "${CHROMS[$chrom]}"

    if [ ! -d "$sae_run" ]; then
        echo "[$(date)] WARNING: $sae_run not found, skipping $chrom"
        continue
    fi

    echo "[$(date)] Processing $chrom..."

    # Check if latent analysis already exists
    if [ ! -f "$sae_run/latent_analysis/data/cluster_assignments.tsv" ]; then
        echo "[$(date)]   - Computing latent analysis with global normalization..."

        # Build analyze_sae_regions command with optional global stats
        cmd="python tools/analyze_sae_regions.py --input_dir \"$sae_run\" --embedding both --leiden_resolution 0.5"
        if [ -n "$global_stats" ] && [ -f "$global_stats" ]; then
            cmd="$cmd --global_stats \"$global_stats\""
            echo "[$(date)]     (using global stats: $global_stats)"
        else
            echo "[$(date)]     (no global stats available for $chrom, using per-chromosome normalization)"
        fi

        eval "$cmd" 2>&1 | tee -a logs/human_latent_analysis.log
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
