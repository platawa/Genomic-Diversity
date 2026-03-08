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
5. **Visualizes** results with publication-quality plots and dashboards

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
└──────────────────┘     └──────────────────────┘
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

**Outputs:** Feature activation plots per region, signature features across regions, interactive notebooks.

## Output Structure

All outputs are saved under `results/`, organized by chromosome and pipeline stage:

```
results/
  chr22/
    scoring/
      20260225_143012_rc_logprobs_4gpu/
        data/          entropy.npz, drop_boundaries.tsv, drops.tsv, summary.json
        logs/          scoring.log
        COMPLETED      sentinel file (JSON)
    sae/
      20260305_110000_overlap_max50_conf8.0/
        data/          sae_results.tsv, signature_features.tsv, feature_matrices.npz
        plots/         region per-feature plots, signature summary
        COMPLETED
    visualization/
      20260305_143000_analyze_scoring/
        analysis.png, dashboard.png, zoom_plots/
        COMPLETED
```

Each completed run writes a `COMPLETED` JSON sentinel as its last action. Absence of this file indicates an interrupted run.

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
├── score_chromosome.py             Main GPU scoring script
├── run_sae_on_chromosome_drops.py  SAE analysis pipeline
├── detection_methods.py            Drop detection algorithms
├── sae_utils.py                    SAE helper functions
├── results_utils.py                Output directory utilities
├── analyze_sae_regions.py          SAE region analysis
│
├── tools/                          Analysis and benchmarking tools
│   ├── analyze_scoring_results.py  Visualization and statistics
│   ├── plot_sae_figure4.py         SAE figure generation
│   ├── build_ground_truth.py       Ground truth from GTF annotations
│   ├── map_drops_to_exons.py       Map detected drops to known exons
│   ├── find_novel_regions.py       Identify unannotated detections
│   ├── compare_detection_methods.py Method comparison
│   ├── cross_organism_summary.py   Cross-species analysis
│   ├── benchmark_performance.py    GPU performance benchmarks
│   └── ...
│
├── scripts/                        SLURM job scripts and helpers
│   ├── run_pipeline.sh             Full pipeline runner
│   ├── ssh_connect.sh              SSH connection helper
│   └── ...
│
├── archive/                        Previous script versions
│   ├── genome_scoring_jan8.py
│   ├── genome_scoring_jan22.py
│   ├── genome_scoring_jan24.py
│   └── genome_scoring_jan26_drops.py
│
├── examples/                       Example run scripts
│
└── results/                        Output data (not tracked)
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
