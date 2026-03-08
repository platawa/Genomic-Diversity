# Claude Code Project Configuration — jan31_files

## Project Overview
Evo2 genomic entropy scoring pipeline. Scores chromosomes using the Evo2 model
and detects entropy drop regions that may correspond to functional elements.

## Key Files
- `score_chromosome.py` — Main GPU scoring script (runs on ORCD cluster)
- `tools/analyze_scoring_results.py` — Analysis & plotting (can run locally or on cluster)
- `results_utils.py` — Shared utilities for organized results directory
- `results/` — Output data directory (organized by chromosome, then stage)

## Remote Cluster: ORCD Engaging (MIT)
- Host: `orcd-login001.mit.edu`
- User: `platawa`
- Remote project dir: `/orcd/data/zhang_f/001/platawa/jan31_files/`
- Conda env: `evo2_sep28`

### Setup on Compute Node (run each session)
```bash
module load miniforge/24.3.0-0
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files
```

### SLURM Allocation
```bash
salloc -t 48:00:00 -p mit_preemptable --gres=gpu:1 --cpus-per-task=8 --mem=100G
```

## Reference Data Paths
```
# Human (GRCh38)
FASTA: /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna
GTF:   /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf

# E. coli K-12
FASTA: /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna
GTF:   /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf

# Bacillus subtilis
FASTA: /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/GCF_000009045.1_ASM904v1_genomic.fna
GTF:   /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf
```

## Output Directory Convention
All output goes under `results/`, organized by chromosome then pipeline stage.
Each run directory has a `YYYYMMDD_HHMMSS_{flags}` name.

```
results/
  chr22/
    scoring/
      20260225_143012_rc_logprobs_4gpu/
        data/          entropy.npz, drop_boundaries.tsv, drops.tsv, rises.tsv, summary.json
        logs/          scoring.log
        COMPLETED      JSON sentinel: {completed_at, script, wall_time_s}
    sae/
      20260305_110000_overlap_max50_conf8.0/
        data/          sae_results.tsv, signature_features.tsv, feature_matrices.npz, run_metadata.json
        plots/         region_N_features.png, signature_summary.png, etc.
        latent_analysis/
        source.json    relative paths to upstream scoring run
        COMPLETED
    visualization/
      20260305_143000_analyze_scoring/
        analysis.png, dashboard.png, transitions_*.png, zoom_plots/, random_plots/
        source.json
        COMPLETED
      20260305_150000_figure4c/
        figure4c_*.png, window_features.npz
        source.json
        COMPLETED
  _benchmarks/         benchmark results (not per-chromosome)
```

### Conventions
- **COMPLETED file**: JSON with `completed_at`, `script`, `wall_time_s`. Written as the very last action. Absence = interrupted run.
- **source.json**: Records relative paths to upstream inputs for dependency tracing.
- **`--auto` flag**: SAE and visualization scripts can auto-discover the latest COMPLETED scoring run for a chromosome.
- **`results_utils.py`**: Shared functions `build_run_dir()`, `write_completed()`, `write_source()`, `find_latest_completed()`.

## SSH Connection via ControlMaster
```bash
# Check socket:
ssh -o ControlPath="$HOME/.ssh/platawa@orcd-login001.mit.edu:22" -o ControlMaster=no -O check platawa@orcd-login001.mit.edu

# Run remote command:
ssh -o "ControlPath=$HOME/.ssh/platawa@orcd-login001.mit.edu:22" -o ControlMaster=no platawa@orcd-login001.mit.edu "COMMAND"

# Or source the helper:
source scripts/ssh_connect.sh
remote_cmd "ls results/"
```

## Constraints
- Only edit files within this `jan31_files/` directory
- GPU scripts must run on the cluster (score_chromosome.py)
- Analysis scripts can run locally if matplotlib/numpy are available
- The cluster uses SLURM for job scheduling
