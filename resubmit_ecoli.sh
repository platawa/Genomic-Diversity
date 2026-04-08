#!/bin/bash
cd /orcd/data/zhang_f/001/platawa/jan31_files

SAE_JOB=$(sbatch -p ou_bcs_normal --gres=gpu:1 --cpus-per-task=8 --mem=100G -t 6:00:00 -J ecoli_sae_full -o logs/ecoli_sae_full_%j.out -e logs/ecoli_sae_full_%j.err --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python run_sae_fast.py --chrom NC_000913.3 --auto --min_confidence 0.0 --batch_size 8 --checkpoint_interval 500 --extract_only --skip_notebook --output_dir results/" 2>&1 | awk '{print $4}')
echo "SAE job: $SAE_JOB"

LAT_JOB=$(sbatch -p ou_bcs_normal --cpus-per-task=8 --mem=64G -t 4:00:00 -J ecoli_latent_full --dependency=afterok:$SAE_JOB -o logs/ecoli_latent_full_%j.out -e logs/ecoli_latent_full_%j.err --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/compute_sae_latent.py --from_shards --chrom NC_000913.3 --results_dir results/ --n_shards 1 --embedding both" 2>&1 | awk '{print $4}')
echo "Latent job: $LAT_JOB"

PLOT_JOB=$(sbatch -p ou_bcs_normal --cpus-per-task=4 --mem=32G -t 1:00:00 -J ecoli_plot_full --dependency=afterok:$LAT_JOB -o logs/ecoli_plot_full_%j.out -e logs/ecoli_plot_full_%j.err --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/plot_sae_latent.py --chrom NC_000913.3 --results_dir results/ --gtf /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf" 2>&1 | awk '{print $4}')
echo "Plot job: $PLOT_JOB"

IPLOT_JOB=$(sbatch -p ou_bcs_normal --cpus-per-task=4 --mem=32G -t 1:00:00 -J ecoli_iplot_full --dependency=afterok:$LAT_JOB -o logs/ecoli_iplot_full_%j.out -e logs/ecoli_iplot_full_%j.err --wrap "cd /orcd/data/zhang_f/001/platawa/jan31_files && module load miniforge/24.3.0-0 && conda activate evo2_sep28 && python tools/plot_interactive_latent.py --chrom NC_000913.3 --results_dir results/ --gtf /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf" 2>&1 | awk '{print $4}')
echo "Interactive plot job: $IPLOT_JOB"
