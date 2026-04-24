#!/bin/bash
# Resubmit E. coli conf0.0 SAE extraction on A100 GPUs with 500GB memory
# Resumes from existing checkpoints (shard0: 9200/12741, shard1: 8400/12741)
# FIX: Added --fasta pointing to E. coli genome (was defaulting to human GRCh38)
cd /orcd/data/zhang_f/001/platawa/jan31_files

SHARD0_DIR="results/NC_000913.3/sae/20260408_125325_max999999_conf0.0_shard0of2"
SHARD1_DIR="results/NC_000913.3/sae/20260408_125325_max999999_conf0.0_shard1of2"
ECOLI_FASTA="/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna"
# Use the March 9 scoring run (25K regions). The March 17 run has EMPTY boundaries.
SCORING_RUN="results/NC_000913.3/scoring/20260309_122646_rc_logprobs_1gpu"
BOUNDARIES="${SCORING_RUN}/data/drop_boundaries.tsv"
ENTROPY="${SCORING_RUN}/data/entropy.npz"

# Shard 0: resume from checkpoint
S0=$(sbatch -p mit_preemptable --gres=gpu:a100:1 --cpus-per-task=8 --mem=500G -t 12:00:00 \
    -J ecoli_s0a \
    -o logs/ecoli_s0a_%j.out \
    -e logs/ecoli_s0a_%j.err \
    --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python run_sae_fast.py --chrom NC_000913.3 --fasta ${ECOLI_FASTA} --boundaries ${BOUNDARIES} --entropy ${ENTROPY} --min_confidence 0.0 --batch_size 4 --checkpoint_interval 200 --extract_only --skip_notebook --output_dir results/ --shard 0/2 --resume_dir ${SHARD0_DIR}" \
    2>&1 | awk '{print $4}')
echo "Shard 0 job: $S0"

# Shard 1: resume from checkpoint
S1=$(sbatch -p mit_preemptable --gres=gpu:a100:1 --cpus-per-task=8 --mem=500G -t 12:00:00 \
    -J ecoli_s1a \
    -o logs/ecoli_s1a_%j.out \
    -e logs/ecoli_s1a_%j.err \
    --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python run_sae_fast.py --chrom NC_000913.3 --fasta ${ECOLI_FASTA} --boundaries ${BOUNDARIES} --entropy ${ENTROPY} --min_confidence 0.0 --batch_size 4 --checkpoint_interval 200 --extract_only --skip_notebook --output_dir results/ --shard 1/2 --resume_dir ${SHARD1_DIR}" \
    2>&1 | awk '{print $4}')
echo "Shard 1 job: $S1"

# Latent analysis (depends on both shards)
LAT=$(sbatch -p mit_preemptable --cpus-per-task=8 --mem=64G -t 4:00:00 \
    -J ecoli_c0lat \
    --dependency=afterok:${S0}:${S1} \
    -o logs/ecoli_c0lat_%j.out \
    -e logs/ecoli_c0lat_%j.err \
    --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/compute_sae_latent.py --from_shards --chrom NC_000913.3 --results_dir results/ --n_shards 2 --embedding both" \
    2>&1 | awk '{print $4}')
echo "Latent job: $LAT"

# Enhanced plots (depends on latent)
ENH=$(sbatch -p mit_preemptable --cpus-per-task=4 --mem=32G -t 1:00:00 \
    -J ecoli_c0enh \
    --dependency=afterok:${LAT} \
    -o logs/ecoli_c0enh_%j.out \
    -e logs/ecoli_c0enh_%j.err \
    --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/plot_sae_latent.py --chrom NC_000913.3 --results_dir results/ --gtf /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf" \
    2>&1 | awk '{print $4}')
echo "Enhanced plots job: $ENH"

echo ""
echo "Dependency chain: S0($S0) + S1($S1) -> Latent($LAT) -> Plots($ENH)"
