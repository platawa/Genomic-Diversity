#!/bin/bash
#===============================================================================
# run_ffs_scoring_jan24.sh
#
# SLURM job script for scoring the ffs gene (4.5S RNA / SRP RNA) in E. coli
# using the Evo2 language model via genome_scoring_jan24.py
#
# Gene Information:
#   - Gene ID: b0455 (ffs)
#   - Function: 4.5S RNA component of Signal Recognition Particle (SRP)
#   - Coordinates: NC_000913.3:476448-476561 (+), 114 nt
#   - This small RNA is essential for protein translocation
#
# Output Structure:
#   <out_dir>/b0455/
#       data/      - TSV scoring data, drop points, window summaries
#       plots/     - Entropy visualizations
#       fasta/     - Locus and exon sequences
#       metadata/  - Run provenance JSON
#===============================================================================

#SBATCH --job-name=ffs_score_jan24
#SBATCH --output=logs/ffs_score_jan24.out
#SBATCH --error=logs/ffs_score_jan24.err
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

# Run ffs (4.5S RNA / SRP RNA) scoring
# The --organism ecoli flag automatically uses E. coli reference genome paths
# Outputs will be organized in: <out_dir>/b0455/
echo "Starting ffs (b0455) scoring at $(date)"
python genome_scoring_jan24.py \
    --organism ecoli \
    --gene_id b0455 \
    --entropy_unit bits \
    --plot_style evodesigner \
    --drop_on rcavg

echo "Finished ffs scoring at $(date)"
