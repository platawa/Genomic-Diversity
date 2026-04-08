#!/bin/bash
#SBATCH -J genome_tsne_raw
#SBATCH -p pi_zhang_f
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH -t 3-00:00:00
#SBATCH -o logs/genome_tsne_%j.log
#SBATCH -e logs/genome_tsne_%j.err

set -e
module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files

CACHE_DIR=results/_genome_wide/sae_tsne/_cache
LOCAL_DIR="/tmp/sae_tsne_cache_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DIR"

# Cleanup handler: restore NFS originals and save checkpoints on ANY exit
cleanup() {
    echo "[$(date)] Cleanup triggered (exit code: $?)"
    # Restore NFS originals if backups exist
    if [ -f "$CACHE_DIR/combined_maxpooled_raw.npy.bak" ]; then
        rm -f "$CACHE_DIR/combined_maxpooled_raw.npy" "$CACHE_DIR/combined_metadata_raw.json"
        mv "$CACHE_DIR/combined_maxpooled_raw.npy.bak" "$CACHE_DIR/combined_maxpooled_raw.npy"
        mv "$CACHE_DIR/combined_metadata_raw.json.bak" "$CACHE_DIR/combined_metadata_raw.json"
        echo "[$(date)] Restored NFS originals from .bak"
    fi
    # Copy any new checkpoints back to NFS
    if [ -d "$LOCAL_DIR" ]; then
        for f in "$LOCAL_DIR"/*.npy "$LOCAL_DIR"/*.h5ad; do
            [ -f "$f" ] && cp "$f" "$CACHE_DIR/" && echo "  Saved $(basename $f) to NFS"
        done
        rm -rf "$LOCAL_DIR"
        echo "[$(date)] Cleaned up local SSD"
    fi
}
trap cleanup EXIT

# Pre-flight: check /tmp space
echo "[$(date)] Local /tmp space:"
df -h /tmp
AVAIL_KB=$(df --output=avail /tmp 2>/dev/null | tail -1 || df /tmp | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_KB / 1048576))
if [ "$AVAIL_GB" -lt 100 ]; then
    echo "ERROR: /tmp has only ${AVAIL_GB}GB free, need 100GB. Aborting."
    exit 1
fi

echo "[$(date)] Staging 91G raw vectors to local SSD..."
time cp "$CACHE_DIR/combined_maxpooled_raw.npy" "$LOCAL_DIR/"
echo "[$(date)] Staging metadata..."
cp "$CACHE_DIR/combined_metadata_raw.json" "$LOCAL_DIR/"

# Copy any existing small caches (embeddings, clusters, PCA, neighbors)
for f in "$CACHE_DIR"/embedding_*_raw.npy "$CACHE_DIR"/cluster_assignments_raw*.npy \
         "$CACHE_DIR"/pca_vectors_raw_pca50.npy "$CACHE_DIR"/neighbors_raw_pca50.h5ad; do
    [ -f "$f" ] && cp "$f" "$LOCAL_DIR/" && echo "  Staged $(basename $f)"
done
echo "[$(date)] Staging complete."

# Create symlinks so the script reads from local SSD
mv "$CACHE_DIR/combined_maxpooled_raw.npy" "$CACHE_DIR/combined_maxpooled_raw.npy.bak"
mv "$CACHE_DIR/combined_metadata_raw.json" "$CACHE_DIR/combined_metadata_raw.json.bak"
ln -sf "$LOCAL_DIR/combined_maxpooled_raw.npy" "$CACHE_DIR/combined_maxpooled_raw.npy"
ln -sf "$LOCAL_DIR/combined_metadata_raw.json" "$CACHE_DIR/combined_metadata_raw.json"

echo "[$(date)] Running t-SNE pipeline (--embedding tsne --n_pca 50)..."
python tools/genome_sae_tsne.py \
    --all_human \
    --gtf /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf \
    --results_dir results/ \
    --embedding tsne \
    --n_pca 50

echo "[$(date)] Done successfully."
