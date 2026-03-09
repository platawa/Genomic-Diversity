#!/bin/bash
#===============================================================================
# run_ssrS_scoring_jan24.sh
#
# SLURM job script for scoring the ssrS gene (6S RNA) in E. coli
# using the Evo2 language model via genome_scoring_jan24.py
#
# Gene Information:
#   - Gene ID: b2911 (ssrS)
#   - Function: 6S RNA - regulates RNA polymerase during stationary phase
#   - Coordinates: NC_000913.3:3055983-3056165 (+), 183 nt
#   - This small RNA modulates transcription by binding to sigma-70 holoenzyme
#
# Output Structure:
#   <out_dir>/b2911/
#       data/      - TSV scoring data, drop points, window summaries
#       plots/     - Entropy visualizations
#       fasta/     - Locus and exon sequences
#       metadata/  - Run provenance JSON
#===============================================================================

#SBATCH --job-name=ssrS_score_jan24
#SBATCH --output=logs/ssrS_score_jan24.out
#SBATCH --error=logs/ssrS_score_jan24.err
#SBATCH --time=2:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

# Create log directory if it doesn't exist
mkdir -p logs

# Load conda environment
module load miniforge/24.3.0-0
source ~/.bashrc
conda activate evo2_sep28

# Navigate to working directory
cd /orcd/data/zhang_f/001/platawa/jan22_files

# Run ssrS (6S RNA) scoring
# The --organism ecoli flag automatically uses E. coli reference genome paths
# Outputs will be organized in: <out_dir>/b2911/
echo "Starting ssrS (b2911) scoring at $(date)"
python genome_scoring_jan24.py \
    --organism ecoli \
    --gene_id b2911 \
    --entropy_unit bits \
    --plot_style evodesigner \
    --drop_on rcavg

echo "Finished ssrS scoring at $(date)"
