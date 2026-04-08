#!/bin/bash
# Submit genome-wide combined analysis (all human chromosomes appended)
# Depends on ALL per-chromosome analysis jobs completing

CHR15_JOB=10987923
PROJECT=/orcd/data/zhang_f/001/platawa/jan31_files
LOGS=${PROJECT}/logs

echo "Submitting genome-wide combined human analysis..."
echo ""

# Create dependency on chr15 merge
COMBINED_JOB=$(sbatch --parsable \
  --job-name="analyze_human_genomewide" \
  --partition=pi_zhang_f \
  --cpus-per-task=16 \
  --mem=256G \
  --time=6:00:00 \
  --dependency=afterok:${CHR15_JOB} \
  --output="${LOGS}/analyze_human_genomewide_%j.out" \
  --error="${LOGS}/analyze_human_genomewide_%j.err" \
  --wrap="
set -eo pipefail
cd ${PROJECT}
module load miniforge/24.3.0-0
conda activate evo2_sep28

python - << 'PYTHON_SCRIPT'
import os
import numpy as np
from pathlib import Path
from io import StringIO

PROJECT = '${PROJECT}'
CHROMS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
          'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
          'chr20', 'chr21', 'chr22']

print('[Genome-wide analysis] Concatenating feature matrices from all chromosomes...')

all_features = []
all_metadata = []
region_offset = 0

for chrom in CHROMS:
    # Find latest merge directory
    merge_dir = sorted(Path(f'{PROJECT}/results/{chrom}/sae').glob('*merged*/'))
    if not merge_dir:
        print(f'  WARNING: No merge found for {chrom}, skipping')
        continue
    merge_dir = str(merge_dir[-1])

    feature_file = os.path.join(merge_dir, 'data/feature_matrices.npz')
    tsv_file = os.path.join(merge_dir, 'data/sae_results.tsv')

    if not os.path.exists(feature_file):
        print(f'  WARNING: No feature_matrices.npz for {chrom}, skipping')
        continue

    # Load features
    data = np.load(feature_file)
    features = [data[key] for key in sorted(data.files)]
    all_features.extend(features)
    print(f'  {chrom}: {len(features)} regions loaded')

    # Load metadata
    if os.path.exists(tsv_file):
        with open(tsv_file) as f:
            lines = f.readlines()
        if len(lines) > 1:
            # Skip header, keep data rows
            all_metadata.extend(lines[1:])

# Save combined feature matrix
print('[Genome-wide analysis] Saving combined feature matrix...')
output_dir = os.path.join(PROJECT, 'results/human_genome_analysis/latent_analysis/data')
os.makedirs(output_dir, exist_ok=True)

combined_array = np.array(all_features, dtype=np.float32)
np.save(os.path.join(output_dir, 'feature_matrices_combined.npy'), combined_array)

# Save combined metadata
if all_metadata:
    with open(os.path.join(output_dir, 'sae_results_combined.tsv'), 'w') as f:
        f.write('region_id\tcoordinate\tmethod\tconfidence\tlength\\n')  # header
        f.writelines(all_metadata)

print(f'[Genome-wide analysis] Combined {len(all_features)} regions across all chromosomes')
print('[Genome-wide analysis] Ready for clustering and visualization')

PYTHON_SCRIPT
")

echo "  human_genomewide: Job ${COMBINED_JOB}"
echo ""
echo "Submitted genome-wide combined analysis job"
