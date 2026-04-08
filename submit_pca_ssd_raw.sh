#!/bin/bash
#SBATCH -J genome_pca_raw_ssd
#SBATCH -p pi_zhang_f
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/genome_pca_raw_ssd_%j.out
#SBATCH -e logs/genome_pca_raw_ssd_%j.err

set -e
module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

CACHE_DIR=results/_genome_wide/sae_tsne/_cache
LOCAL_DIR="/tmp/sae_cache_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DIR"

echo "[$(date)] Local /tmp space:"
df -h /tmp

echo "[$(date)] Staging 91G raw vectors to local SSD..."
time cp "$CACHE_DIR/combined_maxpooled_raw.npy" "$LOCAL_DIR/"
echo "[$(date)] Staging metadata..."
cp "$CACHE_DIR/combined_metadata_raw.json" "$LOCAL_DIR/"

# Copy any existing small caches
for f in "$CACHE_DIR"/embedding_*_raw.npy "$CACHE_DIR"/cluster_assignments_raw.npy; do
    [ -f "$f" ] && cp "$f" "$LOCAL_DIR/" && echo "  Staged $(basename $f)"
done
echo "[$(date)] Staging complete."

# Create a temporary symlink so the script reads from local SSD
# Back up originals, symlink local copies
mv "$CACHE_DIR/combined_maxpooled_raw.npy" "$CACHE_DIR/combined_maxpooled_raw.npy.bak"
mv "$CACHE_DIR/combined_metadata_raw.json" "$CACHE_DIR/combined_metadata_raw.json.bak"
ln -sf "$LOCAL_DIR/combined_maxpooled_raw.npy" "$CACHE_DIR/combined_maxpooled_raw.npy"
ln -sf "$LOCAL_DIR/combined_metadata_raw.json" "$CACHE_DIR/combined_metadata_raw.json"

echo "[$(date)] Running pipeline..."
python tools/genome_sae_tsne.py \
    --all_human \
    --gtf /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf \
    --results_dir results/ \
    --embedding both \
    --n_pca 50
EXIT_CODE=$?

echo "[$(date)] Restoring NFS originals..."
rm -f "$CACHE_DIR/combined_maxpooled_raw.npy" "$CACHE_DIR/combined_metadata_raw.json"
mv "$CACHE_DIR/combined_maxpooled_raw.npy.bak" "$CACHE_DIR/combined_maxpooled_raw.npy"
mv "$CACHE_DIR/combined_metadata_raw.json.bak" "$CACHE_DIR/combined_metadata_raw.json"

# Copy any new checkpoints back to NFS
for f in "$LOCAL_DIR"/*.npy "$LOCAL_DIR"/*.h5ad; do
    [ -f "$f" ] && cp "$f" "$CACHE_DIR/" && echo "  Saved $(basename $f) to NFS"
done

echo "[$(date)] Cleaning up local SSD..."
rm -rf "$LOCAL_DIR"
echo "[$(date)] Done. Exit code: $EXIT_CODE"
exit $EXIT_CODE
