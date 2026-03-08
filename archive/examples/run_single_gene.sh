#!/bin/bash
#===============================================================================
# run_ssrA_scoring_jan24.sh
#
# SLURM job script for scoring the ssrA gene (tmRNA) in E. coli
# using the Evo2 language model via genome_scoring_jan24.py
#
# Gene Information:
#   - Gene ID: b2621 (ssrA)
#   - Function: tmRNA (transfer-messenger RNA) - rescues stalled ribosomes
#   - This dual-function RNA acts as both tRNA and mRNA
#   - Essential for translation quality control and ribosome recycling
#
# Output Structure:
#   <out_dir>/b2621/
#       data/      - TSV scoring data, drop points, window summaries
#       plots/     - Entropy visualizations
#       fasta/     - Locus and exon sequences
#       metadata/  - Run provenance JSON
#===============================================================================

#SBATCH --job-name=ssrA_score_jan24
#SBATCH --output=logs/ssrA_score_jan24.out
#SBATCH --error=logs/ssrA_score_jan24.err
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

# Run ssrA (tmRNA) scoring
# The --organism ecoli flag automatically uses E. coli reference genome paths
# Outputs will be organized in: <out_dir>/b2621/
echo "Starting ssrA (b2621) scoring at $(date)"
python genome_scoring_jan24.py \
    --organism ecoli \
    --gene_id b2621 \
    --entropy_unit bits \
    --plot_style evodesigner \
    --drop_on rcavg

echo "Finished ssrA scoring at $(date)"
