# SAE Feature Normalization Pipeline: Current State & Implementation Guide

## Executive Summary

**Current Status:** The pipeline has **partial support** for feature normalization. It computes chromosome-level statistics but lacks true genome-wide (cross-chromosome) normalization for visualization. Here's what exists and what's needed.

---

## 1. WHAT THE PIPELINE CURRENTLY DOES

### 1.1 Chromosome-Level Statistics (finish_merges.py)
After SAE shards are merged for each chromosome:
- **Input**: `feature_matrices.npz` (all regions for a chromosome)
- **Computation**: Welford's streaming algorithm over all regions
- **Output**: `feature_norm_stats.npz` containing:
  - `mean`: per-feature mean across chromosome
  - `std`: per-feature standard deviation across chromosome
- **Usage**: These stats are fast-path defaults in `finish_merges.py`

### 1.2 Global SAE Statistics (scan_sae_global_stats.py / aggregate_genome_sae_stats.py)
Two separate scripts compute per-chromosome aggregate statistics:
- **scan_sae_global_stats.py**: Computes `chunk_max_mean` and `chunk_max_std` (mean/std of per-chunk maximum activations)
- **aggregate_genome_sae_stats.py**: Collects max-pooled vectors from all 24 chromosomes and computes:
  - `mean`, `std`, `min`, `max`, `median`
  - `q25`, `q75`, `q95`, `q99` (percentiles)
  - `n_nonzero` (how many regions activate each feature)
  - `valid_mask` (features with non-zero std for z-score safety)

### 1.3 Feature Normalization Tool (tools/normalize_sae_features.py)
Applies genome-wide normalization to a single chromosome's features:
- **Input**: Global stats (from aggregate_genome_sae_stats.py)
- **Methods supported**: zscore, minmax, robust (median/IQR)
- **Output**: `normalized_maxpooled_vectors.npy`
- **Current limitation**: Only normalizes max-pooled vectors, NOT individual selected features

---

## 2. WHAT'S MISSING: The Full Feature Normalization Pipeline

Your requirement: **"Global SAE feature stats (min/max per feature across all chromosomes) → z-score normalization of selected features for consistent interpretation scale"**

### 2.1 Gap 1: Feature Selection is Not Integrated with Normalization
Current flow:
```
run_sae_fast.py
  ↓
  Outputs: sae_results.tsv (top features per region)
  ↓
  [NO STEP] → Features are not normalized before selection/visualization
  ↓
  visualization tools (replot.py, etc.)
```

**Missing**: A step that:
1. Takes the selected/signature features from each chromosome
2. Normalizes them using genome-wide stats
3. Produces normalized feature matrices for visualization

### 2.2 Gap 2: Visualization Tools Don't Use Normalized Stats
Scripts like `tools/replot.py` and `tools/plot_sae_figure4.py`:
- Load raw feature matrices
- Plot without normalizing
- Cannot compare features across chromosomes on the same scale

### 2.3 Gap 3: No End-to-End Feature Normalization Workflow
Missing: A single script that:
1. Waits for all SAE runs to complete
2. Computes global genome-wide stats
3. Normalizes ALL selected features across ALL chromosomes
4. Produces a normalized feature matrix for analysis/visualization

---

## 3. CORE PIPELINE STEPS (What Should Happen)

### Step 1: Chromosome Scoring (Already Done)
```bash
python score_chromosome.py --chrom chr22 --output_dir results/
```
**Outputs**: drop_boundaries.tsv, entropy.npz

### Step 2: SAE Feature Extraction (Per-Chromosome)
```bash
python run_sae_fast.py --chrom chr22 --boundaries results/chr22/scoring/.../drop_boundaries.tsv ...
```
**Outputs**:
- `data/sae_results.tsv` (top features per region)
- `data/feature_matrices.npz` (all region activations)
- `latent_analysis/data/maxpooled_vectors.npy`

### Step 3: Merge SAE Shards (Per-Chromosome)
```bash
python merge_sae_shards_fast.py --chrom chr22 --n_shards 36 ...
```
**Outputs**: Merged feature matrices

### Step 4: Finalize with Chromosome-Level Stats
```bash
python finish_merges.py --all_human --output_dir results/
```
**Outputs**: `feature_norm_stats.npz` (chromosome-level mean/std)

### Step 5: **[CURRENTLY MISSING] Compute Genome-Wide Global Statistics**
```bash
python tools/aggregate_genome_sae_stats.py --all_human --results_dir results/
```
**Outputs**:
- `results/_genome_sae_stats/<timestamp>/data/global_feature_stats.npz`
- Contains: mean, std, min, max, median, q25, q75, q95, q99 (each shape: 32768 features)

### Step 6: **[CURRENTLY MISSING] Select and Normalize Features Across All Chromosomes**
**Goal**: For each chromosome, normalize its selected/signature features using genome-wide stats

**Required workflow**:
```
For each chromosome:
  1. Load sae_results.tsv → identify selected features (e.g., top 50 per region)
  2. Load feature_matrices.npz → extract activations for selected features only
  3. Load global_feature_stats.npz → get mean/std for those features
  4. Apply z-score: z = (x - mean) / std for each feature
  5. Save normalized_selected_features.npz
```

**Produces**:
- Per-chromosome normalized feature matrices
- All features on same genome-wide scale
- Ready for visualization/analysis

### Step 7: Visualize with Normalized Features
```bash
python tools/replot.py --chrom chr22 --use_normalized --auto
```
**Uses**: Global-scale normalized features in plots

---

## 4. IMPLEMENTATION PLAN: What Needs to Be Built

### 4.1 Script 1: Enhance `aggregate_genome_sae_stats.py`
**Status**: 90% complete, just needs to be run after all SAE merges complete
**What to do**:
- Already computes all needed stats (mean, std, min, max, percentiles)
- Just needs to be executed: `python tools/aggregate_genome_sae_stats.py --all_human --results_dir results/`
- Outputs to `results/_genome_sae_stats/<timestamp>/data/global_feature_stats.npz`

### 4.2 Script 2: Create `normalize_selected_features.py` (NEW)
**Purpose**: Normalize selected SAE features per chromosome using genome-wide stats

**Input**:
- Global stats file: `results/_genome_sae_stats/<timestamp>/data/global_feature_stats.npz`
- Per-chromosome SAE run (merged): `results/<chrom>/sae/<run>/`

**Process**:
```python
1. Load sae_results.tsv → get selected feature indices
2. Load feature_matrices.npz → extract activations
3. Load global_feature_stats → get mean/std
4. Apply z-score normalization per feature
5. Save normalized_feature_matrices.npz
```

**Output**:
- `results/<chrom>/sae/<run>/normalized_feature_matrices.npz`
- Contains z-scored activations for selected features only

**Pseudo-code**:
```python
def normalize_selected_features(chrom, sae_run_dir, global_stats_path):
    # Load selected features
    selected = load_selected_features_from_sae_results(sae_run_dir)

    # Load chromosome feature matrices
    feature_matrices = load_feature_matrices(sae_run_dir)

    # Load global stats
    global_stats = np.load(global_stats_path)

    # For each selected feature:
    normalized = {}
    for feat_idx in selected:
        mean = global_stats['mean'][feat_idx]
        std = global_stats['std'][feat_idx]

        # Z-score all regions for this feature
        feat_data = feature_matrices[:, feat_idx]  # all regions
        feat_data_zscore = (feat_data - mean) / (std + 1e-6)

        normalized[feat_idx] = feat_data_zscore

    save_normalized_matrices(normalized, output_path)
```

### 4.3 Script 3: Enhance Visualization Tools
**Target files**: `tools/replot.py`, `tools/plot_sae_figure4.py`

**Changes needed**:
- Add `--use_normalized` flag
- When set: load `normalized_feature_matrices.npz` instead of raw matrices
- Scales visualizations to genome-wide range (z-score distribution ≈ mean 0, std 1)

### 4.4 Master Orchestration Script (NEW)
**Purpose**: Run the entire post-merge pipeline in sequence

**Name**: `finalize_feature_normalization.py`

**Pipeline**:
```bash
python finalize_feature_normalization.py --all_human --results_dir results/
```

**Steps**:
1. Verify all chromosome SAE merges are COMPLETED
2. Run `aggregate_genome_sae_stats.py` if not done
3. Run `normalize_selected_features.py` for each chromosome
4. Verify all normalized matrices exist
5. Write summary report

---

## 5. DETAILED CORE STEPS: What You Need to Execute

### Current Status (as of 2026-03-27)
- ✅ Chromosome scoring complete (all 24 chroms)
- ✅ SAE extraction complete (conf8.0)
- ⏳ SAE shard merges: Wave A (12 chroms) in queue, Wave B pending GPU jobs
- ❌ Global feature stats: NOT COMPUTED YET
- ❌ Feature normalization: NOT APPLIED YET
- ❌ Normalized visualizations: NOT CREATED YET

### What Happens Next (Timeline)

**T+9 hours (after Wave B completes)**:
```bash
# All SAE merges for 24 chromosomes are complete

# Step 5: Compute global genome-wide statistics
python tools/aggregate_genome_sae_stats.py \
    --all_human \
    --results_dir results/ \
    --min_chromosomes 24
# Output: results/_genome_sae_stats/<timestamp>/data/global_feature_stats.npz
# Time: ~5-10 minutes
```

**T+9:15 hours (immediately after)**:
```bash
# Step 6: Normalize selected features for each chromosome
# (need to build normalize_selected_features.py first)
for chrom in chr1 chr2 ... chr22 chrX chrY; do
    python normalize_selected_features.py \
        --chrom $chrom \
        --sae_run_dir results/$chrom/sae/<latest_merged> \
        --global_stats_path results/_genome_sae_stats/<timestamp>/data/global_feature_stats.npz \
        --output_dir results/
done
# Time: ~30 minutes total (parallel possible)
```

**T+9:45 hours**:
```bash
# Step 7: Regenerate visualizations with normalized features
# (once replot.py is enhanced)
python tools/replot.py --chrom chr22 --use_normalized --auto
# ... all 24 chromosomes

# Now all features are on the same genome-wide scale!
```

---

## 6. KEY FILES & THEIR ROLES

### Current Implementation
| File | Purpose | Status |
|------|---------|--------|
| `finish_merges.py` | Compute chromosome-level norm stats | ✅ Complete |
| `aggregate_genome_sae_stats.py` | Compute global genome-wide stats | ✅ Code ready, needs to run |
| `tools/normalize_sae_features.py` | Normalize max-pooled vectors | ✅ Complete but limited |
| `run_sae_fast.py` | Extract SAE features per chromosome | ✅ Complete |
| `merge_sae_shards_fast.py` | Merge per-chromosome shards | ✅ Complete |

### Missing Implementation
| File | Purpose | Status |
|------|---------|--------|
| `normalize_selected_features.py` | **[NEEDED]** Normalize selected features with genome-wide stats | ❌ Not implemented |
| `finalize_feature_normalization.py` | **[NEEDED]** Orchestrate the full workflow | ❌ Not implemented |
| `tools/replot.py` (enhanced) | **[NEEDED]** Support normalized feature plotting | ⚠️ Partial |

---

## 7. Z-SCORE NORMALIZATION: Math Specification

### Per-Feature Z-Score Formula
For feature j across all N regions in genome:

```
mean_j = (1/N) * Σ(x_{i,j})                    [mean of feature j]
std_j = sqrt((1/N) * Σ(x_{i,j} - mean_j)²)    [std of feature j]

For any region i in chromosome c:
z_{i,j} = (x_{i,j} - mean_j) / (std_j + ε)    [ε = 1e-6 for numerical stability]
```

### Why This Works
- **Same scale**: All z-scores centered at 0, with std ≈ 1.0
- **Cross-chromosome comparable**: Feature 1234 in chr22 and chr3 on same scale
- **Interpretable**: |z| > 2 = significant activation, |z| > 3 = very significant
- **Robust**: Unaffected by individual feature's overall activity level

### Edge Cases Handled
- **Features with zero variance**: Create `valid_mask` array, skip z-score for those features
- **Numerical stability**: Add 1e-6 to denominator to avoid division by zero
- **Out-of-distribution values**: Can produce |z| > 3; that's fine—indicates extreme activation

---

## 8. WHAT SUCCESS LOOKS LIKE

After completing this pipeline:

### Feature Representation
- ✅ Each SAE feature has a genome-wide mean and std
- ✅ For selected features, activations are z-scored
- ✅ Feature activations are comparable across all 24 chromosomes

### Visualization
- ✅ Feature plots show normalized scale (y-axis: -2 to +3 is typical range)
- ✅ Feature colors/heatmaps represent normalized activation strength
- ✅ Cross-chromosome figures use consistent feature scaling

### Analysis
- ✅ Can directly compare feature 1234's role in chr22 region vs chr3 region
- ✅ Can identify universally-active features vs region-specific ones
- ✅ Can apply global thresholds for feature selection (e.g., "features with mean z > 1.5 in this region")

---

## 9. WORKFLOW SUMMARY

```
START (after SAE merge completion)
  ↓
1. aggregate_genome_sae_stats.py
   └─→ Computes global mean, std, percentiles (32768 features)
   └─→ Outputs: global_feature_stats.npz
  ↓
2. normalize_selected_features.py (per-chromosome, parallelizable)
   └─→ Load selected features from sae_results.tsv
   └─→ Apply z-score using global stats
   └─→ Outputs: normalized_feature_matrices.npz (per-chrom)
  ↓
3. Visualization tools (enhanced)
   └─→ Load normalized matrices
   └─→ Generate plots with genome-wide scale
   └─→ All features interpreted consistently
  ↓
END: All 24 chromosomes with normalized, comparable features
```

---

## 10. NEXT IMMEDIATE ACTIONS

### For You to Execute Now
1. ✅ Monitor Wave A/B merges (already submitted)
2. ⏳ Wait for all SAE merges to complete (~T+9 hours)
3. Then run: `python tools/aggregate_genome_sae_stats.py --all_human --results_dir results/`

### For Claude to Implement
1. Build `normalize_selected_features.py`
2. Build `finalize_feature_normalization.py` orchestration script
3. Enhance `tools/replot.py` with `--use_normalized` flag
4. Create integration tests

---

## Questions?

- **"Will this break existing visualizations?"** No—old ones still work. New ones will have `--use_normalized` flag.
- **"How much slower is normalization?"** Almost no cost—just a few array operations per feature.
- **"What if a feature is dead (zero std)?"** Handled by `valid_mask`—those features are skipped in z-score, left as zeros.
- **"Can I normalize a subset of chromosomes?"** Yes—`aggregate_genome_sae_stats.py` accepts `--chroms chr21 chr22` or `--all_human`.
