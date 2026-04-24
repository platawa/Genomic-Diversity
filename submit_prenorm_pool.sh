#!/bin/bash
# Option A: Pre-normalize per nucleotide BEFORE max pooling
# Reads raw feature_ts from shard chunks, applies (feature_ts - nuc_mean) / nuc_std, then max-pools
# Output: results/{chrom}/sae/latent_analysis_prenorm/data/maxpooled_vectors.npy
cd /orcd/data/zhang_f/001/platawa/jan31_files

STATS=results/_genome_sae_stats/20260406_235042_corrected_24chroms/data/genome_wide_sae_stats_corrected.npz
CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr20 chr21 chr22 chrX"

PRENORM_JOBS=""

for chr in $CHROMS; do
    # Determine n_shards for this chromosome
    N_SHARDS=$(find results/$chr/sae/ -name "COMPLETED" -path "*shard*conf8*" 2>/dev/null | \
        grep -oP 'of\K[0-9]+' | sort -u | tail -1)
    [ -z "$N_SHARDS" ] && N_SHARDS=36

    JOB=$(sbatch -p mit_preemptable --cpus-per-task=4 --mem=64G -t 6:00:00 \
        -J "prenorm_${chr}" \
        -o "logs/prenorm_${chr}_%j.out" \
        -e "logs/prenorm_${chr}_%j.err" \
        --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/compute_sae_latent.py --from_shards --chrom ${chr} --results_dir results/ --n_shards ${N_SHARDS} --norm_method prenorm --global_stats ${STATS} --stage pool" \
        2>&1 | awk '{print $4}')
    echo "Submitted prenorm ${chr} (n_shards=${N_SHARDS}): ${JOB}"
    PRENORM_JOBS="${PRENORM_JOBS}:${JOB}"
done

# Remove leading colon
PRENORM_JOBS="${PRENORM_JOBS#:}"
echo ""
echo "All prenorm pool jobs: ${PRENORM_JOBS}"
echo ""
echo "To submit genome-wide t-SNE after these complete:"
echo "  sbatch --dependency=afterok:${PRENORM_JOBS} submit_tsne_prenorm.sh"
