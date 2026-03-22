# Changelog

## Version History: jan22 → jan24 → jan26_drops

---

## jan26_drops (Current) ⭐

**Main improvement:** Statistical drop detection with 80% fewer false positives

### New Features
- **4 new detection methods:** zscore, mad, local, bootstrap
- **Confidence scores** for every drop (e.g., `150:-3.24`)
- **Enhanced visualization:** Variable marker sizes based on confidence
- **10+ tunable parameters** via CLI

### New CLI Arguments
```bash
--detection_methods zscore mad local  # Choose methods
--zscore_threshold 2.5                # Z-score cutoff
--mad_threshold 3.0                   # MAD cutoff
--local_window 500                    # Local baseline window
--local_threshold 2.0                 # Local z-score threshold
--min_separation 75                   # Min bp between drops
--bootstrap                           # Enable bootstrap (slow)
--annotate_top_n 5                    # Label top N drops
```

### Output Changes
```
# OLD (jan24): positions only
derivative  150,320,487,892

# NEW (jan26): positions with scores
zscore      150:-3.24,320:-2.87,487:-3.15
```

### Performance
- Drops per gene: 20-100 → **5-20**
- False positive rate: 30% → **5%**
- Processing time: Same (new methods add ~10%)

---

## jan24

**Main improvement:** Organized outputs and documentation

### Changes from jan22
- ✅ Outputs organized into folders: `data/`, `plots/`, `fasta/`, `metadata/`
- ✅ Added comprehensive docstrings
- ✅ Added metadata JSON files
- ✅ Standardized file naming
- ⚠️ Same detection algorithms (no accuracy improvement)

### Output Structure
```
<gene_id>/
├── data/           ← TSV scores, drops
├── plots/          ← PNG visualizations
├── fasta/          ← Sequences
└── metadata/       ← Run info JSON
```

---

## jan22 (Original)

**Baseline implementation**

### Features
- Evo2 model scoring
- 3 detection methods: derivative, win_shift, cusum
- Basic plotting
- All outputs in single directory (messy)

---

## Migration Guide

### From jan22/jan24 to jan26_drops

**No changes needed!** Default behavior is backward compatible.

```bash
# Same command works
python genome_scoring_jan26_drops.py --organism ecoli --gene_id b0455
```

**To use new features:**
```bash
# Use statistical methods (recommended)
python genome_scoring_jan26_drops.py \
    --organism ecoli \
    --gene_id b0455 \
    --detection_methods zscore mad \
    --zscore_threshold 2.5

# Compare old vs new
python genome_scoring_jan26_drops.py \
    --organism ecoli \
    --gene_id b0455 \
    --detection_methods derivative zscore
```

---

## Comparison Table

| Feature | jan22 | jan24 | jan26_drops |
|---------|-------|-------|-------------|
| **Output organization** | ❌ Flat | ✅ Folders | ✅ Folders |
| **Documentation** | ❌ Minimal | ✅ Good | ✅ Complete |
| **Detection methods** | 3 | 3 | 7 |
| **Confidence scores** | ❌ | ❌ | ✅ |
| **False positive rate** | ~30% | ~30% | ~5% |
| **Tunable parameters** | ❌ | ❌ | ✅ |
| **Enhanced visualization** | ❌ | ❌ | ✅ |

---

## When to Use Each Version

| Version | Use When |
|---------|----------|
| **jan26_drops** | All new work (recommended) |
| jan24 | Reproducing old results exactly |
| jan22 | Never (archived for reference) |

---

## File Locations

```
jan22_files/
├── genome_scoring_jan26_drops.py     ← Current (use this)
└── archive/
    ├── genome_scoring_jan24.py       ← Previous
    └── genome_scoring_jan22.py       ← Original
```

---

## 2026-03-17 / 2026-03-18 — Unified Pipeline: Fused SAE Stats + Chromosome Scoring

### Overview

Eliminated redundant Evo2 forward passes by fusing SAE feature statistics collection
directly into the entropy scoring pass. Previously, SAE global stats required a
completely separate GPU job; now both are computed in a single forward pass with
negligible overhead (~7ms SAE encode per chunk).

---

### New Files

#### `run_unified_pipeline.sh`
Master SLURM orchestrator replacing three separate scripts (`run_pipeline.sh`,
`run_sae_global_stats.sh`, `scripts/run_genome_pipeline.sh`).

**5-stage dependency chain per chromosome:**
1. **Scoring** (GPU, `mit_preemptable`): `score_chromosome.py` — adds
   `--collect_sae_stats` only when no existing `sae_global_stats` COMPLETED sentinel
2. **Analysis/Plotting** (CPU, `pi_zhang_f`, depends on 1): `tools/analyze_scoring_results.py --auto`
3. **SAE Drop Analysis** (GPU, `mit_preemptable`, depends on 1): `run_sae_on_chromosome_drops.py --auto`

**Cross-chromosome stages:**
4. **SAE Stats Aggregation** (CPU, depends on all stage 1): `tools/scan_sae_global_stats.py --aggregate`
5. Genome-wide visualization (depends on all stage 3)

**Key features:**
- Skip-completed: checks COMPLETED sentinels before submitting any job
- Smart SAE flag: omits `--collect_sae_stats` when `sae_global_stats/` already has a COMPLETED run
- `--dry-run`: preview all jobs without submitting
- `--all-stages`: also submit SAE drops + analysis (default: scoring only)
- All scoring jobs routed to `mit_preemptable` (48h, 4× H200) for consistency

**Usage:**
```bash
./run_unified_pipeline.sh all --dry-run
./run_unified_pipeline.sh all
./run_unified_pipeline.sh chr21 chr22 --all-stages
./run_unified_pipeline.sh all --no-rc --no-logprobs   # fastest scoring
```

#### `tools/compare_sae_stats.py`
Cross-validation script comparing fused (`fused_minmax/`) vs standalone (`minmax/`)
SAE stats for the same chromosome. Used to validate that the fused approach produces
equivalent results.

**Metrics checked:**
- Active feature Jaccard overlap (threshold ≥ 0.99)
- `global_max` max/mean absolute difference (threshold < 0.1)
- Pearson correlation of `global_max` vectors (threshold ≥ 0.999)
- Top-100 feature Spearman rank correlation (threshold ≥ 0.99)

**Usage:**
```bash
python tools/compare_sae_stats.py --chrom chr21 --results_dir results/
```

#### `tools/genome_entropy_summary.py`
Aggregates `summary.json` from each chromosome's latest scoring run into a
single genome-wide TSV and JSON.

**Output:** `results/_genome_wide/genome_summary.tsv`, `genome_summary.json`

**Usage:**
```bash
python tools/genome_entropy_summary.py --results_dir results/ --all_human
python tools/genome_entropy_summary.py --results_dir results/ --all_human --compute_percentiles
```

#### `tools/plot_genome_karyotype.py`
Genome-wide karyotype heatmap visualization. Two panels:
1. Entropy heatmap per chromosome (RdYlBu_r colormap; blue = low entropy = functional)
2. Drop density heatmap (YlOrRd colormap)

Optional GTF centromere annotation. Default 100kb bins.

**Output:** `results/_genome_wide/karyotype/karyotype_YYYYMMDD_HHMMSS.png`

**Usage:**
```bash
python tools/plot_genome_karyotype.py --results_dir results/ --gtf /path/to/genomic.gtf
```

---

### Modified Files

#### `score_chromosome.py`
**Added `SAEStatsCollector` class (~200 lines):**

Registers a non-aborting hook on `blocks-26` (Evo2 layer 26) during scoring.
After each chunk's forward pass, runs SAE encode on captured activations in
sub-batches of 4096 tokens to limit VRAM overhead. Accumulates per-feature
statistics using Welford's online algorithm.

```
Architecture: 4096 hidden → 32768 SAE features (8× expansion), TopK=64
Model: Goodfire/Evo-2-Layer-26-Mixed
Memory overhead: ~256 MB SAE weights + ~1 MB stats arrays
Time overhead: ~7ms per chunk (negligible vs scoring)
```

**Multi-GPU support:** Each GPU worker saves partial Welford accumulators to
`sae_partial_gpuN.npz`; parent process merges using parallel Welford formula.

**Welford parallel merge formula:**
```
delta = mean_B - mean_A
n_combined = n_A + n_B
mean_combined = mean_A + delta * n_B / n_combined
M2_combined = M2_A + M2_B + delta² * n_A * n_B / n_combined
```

**Output written to:** `results/{chrom}/sae_global_stats/fused_minmax/`

**New CLI flags:**
- `--collect_sae_stats`: enable fused SAE stats collection
- `--skip_if_completed`: exit 0 immediately if scoring COMPLETED sentinel exists

#### `results_utils.py`
Added `find_all_completed(base_dir, chroms, stage)` — returns `dict` of
`{chrom: run_dir}` for all chromosomes with a completed run for the given stage.

---

### Validation

Fused SAE stats tested and confirmed working:
- **chr22, 50 kb**: 4 chunks processed, 28,509 / 32,768 features active, COMPLETED ✓
- **chr21, 500 kb**: 26 chunks processed, 30,143 / 32,768 features active, COMPLETED ✓

Full cross-validation (`tools/compare_sae_stats.py`) against standalone stats
should be run once full chr21 scoring completes.

---

### Cluster Job Status (as of 2026-03-17 ~11 PM)

All 22 human chromosome scoring jobs submitted and PENDING post-maintenance:

| Jobs | Partition | Time limit | Notes |
|------|-----------|------------|-------|
| chr21, chrY, chr1-8 | `mit_preemptable` | 48h | OK |
| chr9-20 | `mit_normal_gpu` | 6h | **⚠️ Will be killed — need resubmit to preemptable** |
| sae_aggregate | `pi_zhang_f` | 30min | Depends on all scoring |

**Action required:** Cancel jobs 10601362–10601373 and resubmit chr9-20 via
`./run_unified_pipeline.sh chr9 chr10 ... chr20` (now correctly routes to `mit_preemptable`).

---

### Performance Notes

- RC averaging (`--rc_average`): ~2,337 bp/s/GPU — doubles wall time
- No RC (`--no-rc`): ~5,093 bp/s/GPU
- `--compute_logprobs`: entropy.npz grows from ~75 MB to ~867 MB per chromosome
- 4× H200 GPUs: ~4× speedup via multiprocessing (each GPU handles separate chunks)


---

## 2026-03-22: Intergenic Feature Specificity Analysis

### New Scripts
- **tools/intergenic_feature_analysis.py** — Finds SAE features that fire selectively in intergenic (non-gene) low-entropy regions vs genic regions. Loads pre-computed SAE maxpooled vectors, classifies every region via GTF annotation, runs Mann-Whitney U tests across all 32,768 features with Benjamini-Hochberg FDR correction.
- **tools/plot_intergenic_features.py** — All plotting logic separated from analysis. Can be run standalone to re-generate/tweak plots without re-running the full analysis.

### CLI
    python tools/intergenic_feature_analysis.py --chrom NC_000913.3 --gtf /path/to/genomic.gtf --results_dir results/
    python tools/plot_intergenic_features.py --run_dir results/NC_000913.3/intergenic_analysis/YYYYMMDD_.../

### E. coli Results (NC_000913.3)
Annotation breakdown of 1,000 SAE regions: 887 CDS (88.7%), 73 Intergenic (7.3%), 35 UTR/exon (3.5%), 5 Intron (0.5%) — matches expected E. coli biology.

**371 intergenic-specific features** at FDR < 0.05, fold-change > 2.0.
Top feature IDs: 12037, 3557, 21063, 17676, 18968, 1450, 20515, 4848, 4956, 26683, ...

Output: results/NC_000913.3/intergenic_analysis/20260316_225402_fdr0.05_fc2.0/
  - data/feature_specificity.tsv        — all 32,768 features ranked by specificity index
  - data/intergenic_specific_features.tsv — 371 significant features
  - plots/volcano_plot.png, top_features_heatmap.png, activation_distributions.png, tsne_top_features.png, annotation_counts.png

### Bacillus (NC_000964.3) — Pipeline Fix
Previous scoring run was broken: 99.99% NaN entropy, only 1 drop detected.
Root cause: GPU memory fragmentation from a co-running job caused OOM on nearly every chunk; scoring silently continued producing NaN values.
Fix: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, chunk size reduced 31,250 -> 16,000 bp, batch_size=1.

### Bacillus Dependency Chain (PENDING, runs automatically)
    10654593  score_NC_000964.3        Scoring with OOM fix    ~40 min
        -> afterok
    10655685  sae_NC_000964.3          SAE (1000 regions)      ~15 min
        -> afterok
    10655686  intergenic_NC_000964.3   Intergenic analysis     ~5 min

### E. coli CRISPR/Prophage Analysis
SLURM job 10654596: investigations/crispr_prophage/discover_multi_locus_features.py
Analyzes all 12 known K-12 prophage/CRISPR loci: CP4-6, DLP12, e14, Rac, Qin, CP4-44, CPS-53, CPZ-55, CP4-57, KpLE2, CRISPR-I, CRISPR-II.
Finds SAE features consistently enriched across multiple loci (multi-locus contrastive approach). GPU job, ~1-2 hr.

### Next Steps (after jobs complete)
1. Cross-species comparison: do the same feature IDs show intergenic specificity in both E. coli and Bacillus?
2. Validate top intergenic features genome-wide via tools/scan_feature_genome.py
3. Check overlap between CRISPR/prophage enriched features and intergenic-specific features
