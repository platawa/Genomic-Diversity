#!/bin/bash
# E. coli (NC_000913.3) prenorm pool + cluster.
# Pre-normalize per nucleotide BEFORE max pooling using nuc_mean/nuc_std,
# then run cosine sim + t-SNE/UMAP + Leiden in the same job (--stage both).
# Output: results/NC_000913.3/sae/latent_analysis_prenorm/data/{maxpooled_vectors.npy,cluster_assignments.tsv,...}
cd /orcd/data/zhang_f/001/platawa/jan31_files

CHROM=NC_000913.3
STATS=results/${CHROM}/sae_global_stats/20260415_104706_minmax/data/global_sae_stats.npz

# Auto-discover n_shards from completed conf0.0 shards (E. coli uses conf0.0, not conf8)
N_SHARDS=$(find results/${CHROM}/sae/ -name "COMPLETED" -path "*conf0.0*shard*" 2>/dev/null | \
    grep -oP 'of\K[0-9]+' | sort -u | tail -1)
[ -z "$N_SHARDS" ] && N_SHARDS=2

JOB=$(sbatch -p pi_zhang_f --cpus-per-task=4 --mem=64G -t 24:00:00 \
    -J "ecoli_prenorm_pool" \
    -o "logs/ecoli_prenorm_pool_%j.out" \
    -e "logs/ecoli_prenorm_pool_%j.err" \
    --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/compute_sae_latent.py --from_shards --chrom ${CHROM} --results_dir results/ --n_shards ${N_SHARDS} --norm_method prenorm --global_stats ${STATS} --stage both" \
    2>&1 | awk '{print $4}')
echo "Submitted ecoli prenorm pool+cluster (n_shards=${N_SHARDS}): ${JOB}"
