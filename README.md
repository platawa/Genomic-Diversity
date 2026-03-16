# Genomic Diversity: Entropy-based functional element detection with Evo 2

This pipeline uses the [Evo 2](https://github.com/ArcInstitute/evo2) DNA language model to identify functional genomic elements through entropy analysis. Evo 2 scores DNA sequences at single-nucleotide resolution; regions where the model shows high confidence (low entropy) often correspond to biologically significant elements such as exons, regulatory regions, and conserved domains.

## Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Score a Chromosome](#1-score-a-chromosome)
  - [2. Analyze Results](#2-analyze-results)
  - [3. SAE Feature Analysis](#3-sae-feature-analysis)
  - [4. SAE Region Clustering](#4-sae-region-clustering)
  - [5. Genome-wide Feature Scanning](#5-genome-wide-feature-scanning)
  - [6. Differential Feature Discovery](#6-differential-feature-discovery)
  - [7. Multi-locus Feature Discovery](#7-multi-locus-feature-discovery)
  - [8. Drop Annotation](#8-drop-annotation)
- [Output Structure](#output-structure)
- [Supported Organisms](#supported-organisms)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

## Overview

Evo 2 is a 7B+ parameter DNA language model trained on 8.8 trillion tokens across all domains of life. When scoring a genome, positions where the model is highly confident (low entropy) tend to be functionally important — the model has "learned" these patterns from evolutionary data.

This pipeline:
1. **Scores** entire chromosomes with Evo 2, producing per-position entropy values
2. **Detects** statistically significant entropy drops using robust methods (MAD, z-score)
3. **Pairs** drop and rise boundaries to define complete low-entropy regions
4. **Analyzes** detected regions with Evo 2's Sparse Autoencoder (SAE) to identify which biological features are activated
5. **Clusters** regions in SAE latent space to group functionally similar elements
6. **Scans** genomes for activation of specific known SAE features (e.g., prophage, CDS)
7. **Discovers** novel SAE features enriched in target regions via contrastive analysis
8. **Visualizes** results with publication-quality plots and dashboards

## Pipeline

```
Chromosome FASTA
       │
       ▼
┌──────────────────┐
│ score_chromosome  │  GPU scoring with Evo 2
│     .py           │  → per-position entropy
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
 drops     entropy
 .tsv      .npz
    │         │
    ▼         ▼
┌──────────────────┐     ┌──────────────────────┐
│ analyze_scoring  │     │ run_sae_on_chromosome │
│ _results.py      │     │ _drops.py             │
│ → plots,         │     │ → SAE feature         │
│   dashboards     │     │   analysis            │
└──────────────────┘     └──────────┬────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ analyze_sae  │ │ scan_feature │ │ discover_    │
            │ _regions.py  │ │ _genome.py   │ │ region_      │
            │ → clustering,│ │ → genome-    │ │ features.py  │
            │   t-SNE      │ │   wide scan  │ │ → enrichment │
            └──────────────┘ └──────────────┘ └──────────────┘
```

## Setup

### Requirements

- Linux with CUDA 12.1+ and a supported NVIDIA GPU
- Python 3.11+ with conda
- [Evo 2](https://github.com/ArcInstitute/evo2) installed (see their repo for instructions)

### Environment Setup

```bash
module load miniforge/24.3.0-0
conda activate evo2_sep28
```

### SLURM (HPC Cluster)

```bash
# Interactive session with GPU
salloc -t 48:00:00 -p mit_preemptable --gres=gpu:1 --cpus-per-task=8 --mem=100G
```

## Usage

### 1. Score a Chromosome

Score an entire chromosome (or a region) to compute per-position entropy and detect drop boundaries.

```bash
# Score human chromosome 22
python score_chromosome.py --chrom NC_000022.11 --output_prefix chr22

# Score a specific region
python score_chromosome.py --chrom NC_000001.11 --start 1000000 --end 2000000

# Score with custom thresholds
python score_chromosome.py --chrom NC_000021.9 \
    --zscore_threshold 3.0 \
    --mad_threshold 3.5
```

**Outputs:** `entropy.npz`, `drop_boundaries.tsv`, `drops.tsv`, `rises.tsv`, `summary.json`

### 2. Analyze Results

Generate visualizations and statistics from scoring results. Can run locally (no GPU needed).

```bash
# Basic text report
python tools/analyze_scoring_results.py --auto chr22

# Entropy profile plots
python tools/analyze_scoring_results.py --auto chr22 --plot

# Full plot suite with dashboard
python tools/analyze_scoring_results.py --auto chr22 --all_plots --dashboard

# Zoom into a specific region
python tools/analyze_scoring_results.py --auto chr22 \
    --plot --plot_start 20000000 --plot_end 25000000
```

### 3. SAE Feature Analysis

Run Evo 2's Sparse Autoencoder on detected drop regions to identify which biological features (from 32K learned features) are activated.

```bash
# Analyze drop regions with SAE
python run_sae_on_chromosome_drops.py --auto chr22

# Custom parameters
python run_sae_on_chromosome_drops.py --auto chr22 \
    --max_regions 50 \
    --confidence_threshold 8.0
```

**Outputs:** Feature activation plots per region, signature features across regions, feature matrices.

### 4. SAE Region Clustering

Cluster entropy drop regions by their SAE fingerprints to find functionally similar groups. No GPU required.

```bash
# Cluster regions with Leiden + t-SNE visualization
python analyze_sae_regions.py \
    --feature_matrices results/chr22/sae/.../data/feature_matrices.npz \
    --sae_results results/chr22/sae/.../data/sae_results.tsv \
    --output_dir results/chr22/sae/.../latent_analysis

# With GTF annotation overlay for colored embeddings
python analyze_sae_regions.py \
    --feature_matrices results/chr22/sae/.../data/feature_matrices.npz \
    --annotation_tsv results/chr22/sae/.../annotation_overlay/annotated_regions.tsv
```

**Outputs:** Cosine similarity heatmap, t-SNE/UMAP scatter plots, cluster assignments, cluster summaries.

### 5. Genome-wide Feature Scanning

Scan an entire genome for activation of specific known SAE features. Processes in memory-safe 8192bp chunks.

```bash
# Scan for prophage feature (f/19745)
python tools/scan_feature_genome.py \
    --fasta /path/to/genome.fna \
    --chrom NC_000913.3 \
    --features 19745 \
    --gtf /path/to/genomic.gtf \
    --output_dir results

# Scan for multiple features at once
python tools/scan_feature_genome.py \
    --fasta /path/to/genome.fna \
    --chrom NC_000913.3 \
    --features 19745 15680 28339 30262 \
    --gtf /path/to/genomic.gtf \
    --output_dir results
```

**Known SAE features:**

| Feature ID | Label | Description |
|-----------|-------|-------------|
| 15680 | CDS | Coding regions |
| 28339 | Intron | Introns |
| 1050 | Exon start | First base of exon following intron |
| 25666 | Exon end | Last base of exon followed by intron |
| 24278 | Frameshift | Mutation-sensitive, frameshifts & premature stops |
| 19745 | Prophage | Prophage regions across prokaryotes |

### 6. Differential Feature Discovery

Find SAE features enriched in a target genomic region vs. a background region using Mann-Whitney U tests (single locus).

```bash
# CRISPR spacer region in E. coli
python tools/discover_region_features.py \
    --fasta /path/to/ecoli.fna \
    --chrom NC_000913.3 \
    --target_start 2877618 --target_end 2878569 \
    --bg_from_gtf --bg_flank 5000 --gtf /path/to/genomic.gtf \
    --output_dir results

# Manual background specification
python tools/discover_region_features.py \
    --fasta /path/to/ecoli.fna \
    --chrom NC_000913.3 \
    --target_start 2877618 --target_end 2878569 \
    --bg_start 2872000 --bg_end 2884000 \
    --output_dir results
```

**Outputs:** Enriched features TSV, volcano plot, top enriched features bar chart.

### 7. Multi-locus Feature Discovery

Auto-detect CRISPR/prophage loci from GTF, run contrastive feature analysis at each, and aggregate to find features *consistently* enriched across multiple sites. Requires GPU.

```bash
# Auto-detect loci from GTF and run multi-locus discovery
python investigations/crispr_prophage/discover_multi_locus_features.py \
    --fasta /path/to/ecoli.fna \
    --chrom NC_000913.3 \
    --gtf /path/to/genomic.gtf \
    --chrom_name ecoli_K12 --output_dir results

# Provide custom target loci via TSV
python investigations/crispr_prophage/discover_multi_locus_features.py \
    --fasta /path/to/ecoli.fna \
    --chrom NC_000913.3 \
    --gtf /path/to/genomic.gtf \
    --targets_tsv my_loci.tsv \
    --output_dir results
```

**Outputs:** Per-locus enrichment TSVs and plots, consensus features TSV (ranked by how many loci each feature is enriched in), consensus heatmap (features x loci).

### 8. Drop Annotation

Classify entropy drops as genic vs. intergenic and flag known CRISPR/prophage loci. No GPU needed.

```bash
python investigations/crispr_prophage/compare_drops_annotations.py \
    --boundaries_tsv results/ecoli_K12/scoring/.../data/drop_boundaries.tsv \
    --gtf /path/to/genomic.gtf \
    --chrom NC_000913.3 \
    --chrom_name ecoli_K12
```

**Outputs:** Annotated drops TSV (with `overlapping_genes`, `gene_biotype`, `special_locus` columns), summary JSON with genic/intergenic counts and per-biotype breakdown.

## Output Structure

All outputs are saved under `results/`, organized by chromosome and pipeline stage:

```
results/
  {chrom_name}/
    scoring/
      {timestamp}_{flags}/
        data/          entropy.npz, drop_boundaries.tsv, drops.tsv, summary.json
        logs/          scoring.log
        COMPLETED      sentinel file (JSON)
    sae/
      {timestamp}_{flags}/
        data/          sae_results.tsv, signature_features.tsv, feature_matrices.npz
        plots/         region per-feature plots, signature summary
        latent_analysis/
          data/        cluster_assignments.tsv, cosine_similarity.npy
          plots/       tsne_4panel.png, cosine_similarity_heatmap.png
        COMPLETED
    sae_differential/
      {timestamp}_{flags}/
        data/          enriched_features.tsv, all_features.tsv, region_definitions.json
        plots/         top_enriched_features.png, enrichment_volcano.png
        COMPLETED
    sae_feature_scan/
      {timestamp}_{flags}/
        data/          feature_activation.npz, activation_regions.tsv
        plots/         genome_profile.png, top_regions_zoom.png
        COMPLETED
    sae_multi_locus_differential/
      {timestamp}_{n}_loci/
        data/
          target_loci.tsv
          consensus_features.tsv
          per_locus/{locus}/  enriched_features.tsv, region_definitions.json
        plots/
          consensus_heatmap.png
          per_locus/{locus}/  top_enriched_features.png, enrichment_volcano.png
        COMPLETED
    drop_annotations/
      {timestamp}_{flags}/
        data/          annotated_drops.tsv, summary.json
        COMPLETED
    visualization/
      {timestamp}_{flags}/
        analysis.png, dashboard.png, zoom_plots/
        COMPLETED
```

Each completed run writes a `COMPLETED` JSON sentinel as its last action. Absence of this file indicates an interrupted run. A `source.json` file records input paths for provenance.

## Detection Methods

| Method | Default Threshold | Description |
|--------|-------------------|-------------|
| **z-score** | 2.5 | Statistical significance (2.5σ ≈ 1% FDR) |
| **MAD** | 3.0 | Robust to outliers (uses median absolute deviation) |
| **local** | 2.0 | Adapts to regional variance |
| **bootstrap** | 0.50 | Highest confidence (100x slower) |

## Supported Organisms

| Organism | Genome Assembly |
|----------|----------------|
| Human | GRCh38 (GCF_000001405.26) |
| *E. coli* K-12 | ASM584v2 (GCF_000005845.2) |
| *B. subtilis* | ASM904v1 (GCF_000009045.1) |

## Repository Structure

```
├── score_chromosome.py              Main GPU scoring script
├── run_sae_on_chromosome_drops.py   SAE analysis on entropy drops
├── analyze_sae_regions.py           SAE region clustering & embedding
├── detection_methods.py             Drop detection algorithms
├── sae_utils.py                     SAE model loading & feature extraction
├── results_utils.py                 Output directory utilities
│
├── tools/                           Analysis and visualization tools
│   ├── analyze_scoring_results.py     Visualization and statistics (no GPU)
│   ├── scan_feature_genome.py         Genome-wide SAE feature scanner
│   ├── discover_region_features.py    Differential SAE feature discovery
│   ├── sae_annotation_overlay.py      GTF annotation overlay for SAE clusters
│   ├── plot_sae_figure4.py            SAE figure generation
│   ├── scan_sae_global_stats.py       Global SAE activation statistics
│   ├── aggregate_genome_sae_stats.py  Aggregate genome-wide SAE stats
│   ├── genome_sae_tsne.py             Genome-wide SAE t-SNE visualization
│   ├── plot_tsne_by_annotation.py     Annotation-colored t-SNE plots
│   ├── normalize_sae_features.py      SAE feature normalization
│   ├── deep_locus_comparison.py       Deep comparison between loci
│   ├── build_ground_truth.py          Ground truth from GTF annotations
│   ├── map_drops_to_exons.py          Map detected drops to known exons
│   ├── find_novel_regions.py          Identify unannotated detections
│   ├── compare_detection_methods.py   Method comparison
│   ├── compare_ablations.py           Ablation study comparison
│   ├── cross_organism_summary.py      Cross-species analysis
│   ├── curate_test_loci.py            Test loci curation
│   ├── optimize_drop_parameters.py    Parameter optimization
│   ├── optimize_consensus_based.py    Consensus-based optimization
│   ├── quick_parameter_test.py        Quick parameter testing
│   ├── benchmark_performance.py       GPU performance benchmarks
│   ├── benchmark_pipeline_timing.py   Pipeline timing benchmarks
│   └── analyze_benchmarks.py          Benchmark result analysis
│
├── investigations/                  Research investigations
│   └── crispr_prophage/               CRISPR/prophage feature discovery
│       ├── compare_drops_annotations.py   Annotate drops with gene/locus info
│       └── discover_multi_locus_features.py  Multi-locus contrastive analysis
│
├── scripts/                         SLURM job scripts and helpers
│   ├── run_pipeline.sh                Full pipeline runner
│   ├── ssh_connect.sh                 SSH connection helper
│   ├── submit_ablations.sh            Ablation study submission
│   └── ...
│
├── run_sae_tools.sbatch             Batch job for SAE tool suite
├── run_sae_analysis.sbatch          Batch job for SAE analysis
├── run_dashboard.sbatch             Batch job for dashboard generation
│
├── archive/                         Previous script versions
│   └── examples/                      Archived example scripts
│
├── examples/                        Example run scripts
│
├── evo2/                            Evo 2 model (submodule/vendored)
│   ├── evo2/                          Core model code
│   ├── notebooks/                     Reference notebooks (BRCA1, SAE, etc.)
│   └── phage_gen/                     Phage generation pipeline
│
└── results/                         Output data (not tracked in git)
```

## Citation

This project builds on:

```bibtex
@article{evo2,
    title   = {Genome modeling and design across all domains of life with Evo 2},
    author  = {Nguyen, Eric and Poli, Michael and Durrant, Matthew G. and others},
    journal = {Nature},
    year    = {2026},
    doi     = {10.1038/s41586-026-10176-5}
}
```
