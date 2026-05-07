#!/bin/bash
# E. coli (NC_000913.3) postnorm pool + cluster.
# Reuse raw maxpooled_vectors.npy if present (much cheaper), otherwise pool from shards,
# then z-score with chunk-max stats (global_mean/global_std), then cluster (--stage both).
# Output: results/NC_000913.3/sae/latent_analysis_postnorm/data/{maxpooled_vectors.npy,cluster_assignments.tsv,...}
cd /orcd/data/zhang_f/001/platawa/jan31_files

CHROM=NC_000913.3
STATS=results/${CHROM}/sae_global_stats/20260415_104706_minmax/data/global_sae_stats.npz

N_SHARDS=$(find results/${CHROM}/sae/ -name "COMPLETED" -path "*conf0.0*shard*" 2>/dev/null | \
    grep -oP 'of\K[0-9]+' | sort -u | tail -1)
[ -z "$N_SHARDS" ] && N_SHARDS=2

JOB=$(sbatch -p mit_preemptable --cpus-per-task=4 --mem=16G -t 1:00:00 \
    -J "ecoli_postnorm_pool" \
    -o "logs/ecoli_postnorm_pool_%j.out" \
    -e "logs/ecoli_postnorm_pool_%j.err" \
    --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/compute_sae_latent.py --from_shards --chrom ${CHROM} --results_dir results/ --n_shards ${N_SHARDS} --norm_method postnorm --global_stats ${STATS} --stage both" \
    2>&1 | awk '{print $4}')
echo "Submitted ecoli postnorm pool+cluster (n_shards=${N_SHARDS}): ${JOB}"
