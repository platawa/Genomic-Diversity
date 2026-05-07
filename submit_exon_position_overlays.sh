#!/bin/bash
# Submit per-chromosome and genome-wide jobs that overlay first / middle /
# last exon position on top of existing t-SNE and UMAP embeddings.
#
# Prerequisite: the relevant latent_analysis{,_prenorm,_normalized} and
# _genome_wide/sae_tsne_{prenorm,postnorm,raw} runs must already exist
# (i.e. cluster_assignments.tsv must contain tsne_1/tsne_2 and/or umap_1/umap_2).
#
# CPU-only, small memory. One short job per (chromosome x normalization)
# plus one per genome-wide normalization.

cd /orcd/data/zhang_f/001/platawa/jan31_files
GTF=/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf

CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 \
        chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY)

# Parallel arrays: label -> (per-chrom latent subdir, genome-wide dir)
NORM_LABELS=(raw normalized prenorm postnorm)
NORM_LATENT=(latent_analysis latent_analysis_normalized latent_analysis_prenorm latent_analysis_postnorm)
NORM_GWDIR=(_genome_wide/sae_tsne \
            _genome_wide/sae_tsne \
            _genome_wide/sae_tsne_prenorm \
            _genome_wide/sae_tsne_postnorm)

# Required overlay PNGs per (chrom x variant) — script writes 6 (3 classes x 2 embeddings).
# If all 6 already exist AND are newer than the embedding source, skip the job.
REQUIRED_PNGS=(tsne_exon_first_overlay.png tsne_exon_middle_overlay.png tsne_exon_last_overlay.png \
               umap_exon_first_overlay.png umap_exon_middle_overlay.png umap_exon_last_overlay.png)

mkdir -p logs

needs_rerun() {
    # Returns 0 (true) if the run_dir needs the overlay job submitted, 1 otherwise.
    local run_dir="$1"
    local ca="${run_dir}/data/cluster_assignments.tsv"
    [[ ! -f "$ca" ]] && return 2  # caller treats as skip-no-embedding
    local ca_mtime
    ca_mtime=$(stat -c%Y "$ca")
    for f in "${REQUIRED_PNGS[@]}"; do
        local p="${run_dir}/plots/${f}"
        [[ ! -f "$p" ]] && return 0
        local p_mtime
        p_mtime=$(stat -c%Y "$p")
        (( p_mtime < ca_mtime )) && return 0
    done
    return 1
}

submit() {
    local name="$1"; shift
    local run_dir="$1"; shift
    local cpus="$1"; shift
    local mem="$1"; shift
    local time="$1"; shift
    local cmd="module load miniforge/24.3.0-0 && conda activate evo2_sep28 && \
cd /orcd/data/zhang_f/001/platawa/jan31_files && \
python tools/plot_exon_position_overlay.py --run_dir ${run_dir} --gtf ${GTF}"
    local jid
    jid=$(sbatch -p mit_preemptable --cpus-per-task=${cpus} --mem=${mem} -t ${time} \
        -J "eov_${name}" \
        -o "logs/eov_${name}_%j.out" \
        -e "logs/eov_${name}_%j.err" \
        --wrap "${cmd}" 2>&1 | awk '{print $4}')
    echo "submitted ${name} -> ${jid}  (${run_dir})"
}

# ---- Per-chromosome x per-normalization ----
for i in "${!NORM_LABELS[@]}"; do
    label="${NORM_LABELS[$i]}"
    subdir="${NORM_LATENT[$i]}"
    for chr in "${CHROMS[@]}"; do
        run_dir="results/${chr}/sae/${subdir}"
        needs_rerun "$run_dir"; rc=$?
        if (( rc == 2 )); then
            echo "skip ${chr}_${label}: no ${run_dir}/data/cluster_assignments.tsv"
            continue
        fi
        if (( rc == 1 )); then
            echo "skip ${chr}_${label}: 6/6 PNGs already present and current"
            continue
        fi
        submit "${chr}_${label}" "${run_dir}" 2 16G 1:00:00
    done
done

# ---- Genome-wide x per-normalization ----
# Bigger allocation: genome-wide classification touches ~900K regions and
# can take ~20-30 min of CPU-bound Python on the first (uncached) run.
for i in "${!NORM_LABELS[@]}"; do
    label="${NORM_LABELS[$i]}"
    parent="results/${NORM_GWDIR[$i]}"
    if [[ ! -d "${parent}" ]]; then
        echo "skip gw_${label}: no ${parent}"
        continue
    fi
    # Match only real timestamped run dirs (YYYYMMDD_...), excluding
    # ancillary _cache* / _cache_stale* siblings.
    # Only real timestamped dirs (YYYYMMDD_... with `chroms` suffix); skip _cache* siblings.
    run_dir=$(ls -1d "${parent}"/20*_*chroms*/ 2>/dev/null | sort | tail -n 1)
    run_dir="${run_dir%/}"
    if [[ -z "${run_dir}" || ! -f "${run_dir}/data/cluster_assignments.tsv" ]]; then
        echo "skip gw_${label}: no cluster_assignments.tsv under ${parent}"
        continue
    fi
    submit "gw_${label}" "${run_dir}" 4 32G 4:00:00
done
