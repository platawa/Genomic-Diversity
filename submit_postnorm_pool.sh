#!/bin/bash
# Option B: Post-max-pool normalization with chunk-max stats (global_mean/global_std)
# Loads existing raw maxpooled_vectors.npy, applies (v - global_mean) / global_std
# Output: results/{chrom}/sae/latent_analysis_postnorm/data/maxpooled_vectors.npy
cd /orcd/data/zhang_f/001/platawa/jan31_files

STATS=results/_genome_sae_stats/20260406_235042_corrected_24chroms/data/genome_wide_sae_stats_corrected.npz
CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr20 chr21 chr22 chrX"

POSTNORM_JOBS=""

for chr in $CHROMS; do
    N_SHARDS=$(find results/$chr/sae/ -name "COMPLETED" -path "*shard*conf8*" 2>/dev/null | \
        grep -oP 'of\K[0-9]+' | sort -u | tail -1)
    [ -z "$N_SHARDS" ] && N_SHARDS=36

    JOB=$(sbatch -p mit_preemptable --cpus-per-task=4 --mem=16G -t 1:00:00 \
        -J "postnorm_${chr}" \
        -o "logs/postnorm_${chr}_%j.out" \
        -e "logs/postnorm_${chr}_%j.err" \
        --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/compute_sae_latent.py --from_shards --chrom ${chr} --results_dir results/ --n_shards ${N_SHARDS} --norm_method postnorm --global_stats ${STATS} --stage pool" \
        2>&1 | awk '{print $4}')
    echo "Submitted postnorm ${chr}: ${JOB}"
    POSTNORM_JOBS="${POSTNORM_JOBS}:${JOB}"
done

POSTNORM_JOBS="${POSTNORM_JOBS#:}"
echo ""
echo "All postnorm pool jobs: ${POSTNORM_JOBS}"
echo ""
echo "To submit genome-wide t-SNE after these complete:"
echo "  sbatch --dependency=afterok:${POSTNORM_JOBS} submit_tsne_postnorm.sh"
