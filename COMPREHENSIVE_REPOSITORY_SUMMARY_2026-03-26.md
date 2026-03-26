# Comprehensive Repository Summary: jan31_files

**Generated:** 2026-03-26
**Location:** `/Users/parid/Downloads/jan31_files`
**Status:** Active development with parallel SAE pipelines running

---

## 1. Repository Structure

### Root-Level Organization

```
jan31_files/
├── Core Pipeline Code (15 main Python scripts)
│   ├── score_chromosome.py                 - Evo2 scoring engine
│   ├── run_sae_fast.py                     - SAE extraction on scored regions
│   ├── merge_sae_shards_fast.py            - Fast shard merging
│   ├── finish_merges.py                    - Post-merge normalization
│   └── sae_utils.py, results_utils.py      - Utility functions
│
├── Job Submission & Monitoring (22 shell scripts)
│   ├── submit_mega_shards.sh               - High-conf (>=8.0) pipeline
│   ├── submit_sae_fast_shards.sh           - Per-chromosome SAE
│   ├── submit_full_sae_pipeline.sh         - Full-scale (>=0.0) pipeline
│   ├── submit_*_analysis.sh                - Analysis job submission
│   └── merge_chr15_*.sh                    - Chr15 merge variants
│
├── Analysis & Visualization Tools (41 Python scripts in tools/)
│   ├── genome_sae_tsne.py                  - Genome-wide t-SNE
│   ├── chromosome_analysis.py              - Per-chromosome analysis
│   ├── cross_organism_*.py                 - Comparative analysis
│   ├── linear_probe_classifier.py          - Feature classification
│   ├── intergenic_feature_analysis.py      - Intergenic region analysis
│   └── [35+ more specialized analysis tools]
│
├── Documentation (8 markdown files)
│   ├── README.md                           - Project overview
│   ├── CHANGELOG.md                        - Version history (jan22→jan26)
│   ├── CURRENT_PIPELINE_STATUS.md          - Latest queue status
│   ├── PIPELINE_INVESTIGATION.md           - Deep dive into pipeline structure
│   ├── QUEUE_SUBMISSION_GUIDELINES.md      - Queue selection best practices
│   └── [3 more status documents]
│
├── Data & Results
│   ├── local_review/results/               - Local test results
│   ├── sae_chromosome_results/             - Older results
│   ├── logs/                               - 400+ job logs
│   └── evo2/                               - Evo2 model submodule
│
└── Configuration & Support
    ├── examples/                           - Example runs
    ├── scripts/                            - Utility shell scripts
    ├── docs/                               - Technical documentation
    └── .git/                               - Git version control
```

---

## 2. Project Overview

### Purpose
Use the **Evo 2** 7B+ parameter DNA language model to detect functional genomic elements through entropy analysis. Regions where the model shows high confidence (low entropy) correspond to biologically significant elements (exons, regulatory regions, conserved domains).

### Core Pipeline Flow
```
Chromosome FASTA (scored)
    ↓
Detect entropy drop boundaries → Pair boundaries → Define regions
    ↓
Extract region sequences → Run through Evo2 SAE (layer 26, 32K features)
    ↓
Generate feature matrices → Analyze activations → Cluster in latent space
    ↓
Visualize: t-SNE plots, feature heatmaps, cross-organism comparison
```

### Supported Organisms
- **Human:** chr1-22, chrX, chrY (24 chromosomes)
- **Bacteria:** E. coli (NC_000913.3), Bacillus subtilis (NC_000964.3)

---

## 3. Key Scripts & Tools

### 3.1 Core Pipeline Scripts

| Script | Purpose | Key Parameters |
|--------|---------|-----------------|
| `score_chromosome.py` | GPU-based Evo2 entropy scoring | `--organism`, `--chrom`, `--batch_size` |
| `run_sae_fast.py` | Extract SAE features from scored regions | `--chrom`, `--min_confidence`, `--shard`, `--n_shards` |
| `merge_sae_shards_fast.py` | Merge shard outputs into unified matrices | `--chrom`, `--n_shards`, `--include-partial` |
| `finish_merges.py` | Compute normalization stats & final COMPLETED sentinel | `--chrom_list` |
| `detection_methods.py` | 4 entropy drop detection algorithms (zscore, MAD, local, bootstrap) | `--detection_methods`, `--zscore_threshold` |
| `sae_utils.py` | Shared SAE utilities (feature loading, maxpooling, stats) | N/A (library) |
| `results_utils.py` | Results directory management | N/A (library) |

### 3.2 Analysis & Visualization Tools (tools/ directory)

**Genome-wide Analysis:**
- `genome_sae_tsne.py` - Aggregate SAE fingerprints across chromosomes, compute t-SNE
- `genome_entropy_summary.py` - Aggregate per-chromosome scoring results
- `cross_organism_summary.py` - Statistics across all organisms
- `cross_organism_features.py` - Feature activation pattern comparison (human vs bacteria)

**Per-Chromosome Analysis:**
- `chromosome_analysis.py` - 6 analyses: entropy summary, top features, region counts, intergenic fractions, detection method comparison, cross-method consistency
- `analyze_sae_regions.py` - Detailed per-chromosome region analysis
- `aggregate_genome_sae_stats.py` - Collect max-pooled SAE vectors

**Feature & Annotation Analysis:**
- `linear_probe_classifier.py` - Train logistic regression on SAE features to predict annotation types (CDS, exon, intron, intergenic)
- `intergenic_feature_analysis.py` - Find SAE features selective to intergenic regions
- `discover_region_features.py` - Differential SAE feature discovery (target vs background)
- `sae_annotation_overlay.py` - Overlay SAE features with GTF annotations

**Advanced Methods:**
- `feature_guided_mcmc.py` - Sequence generation with MCMC guided by feature activations
- `steered_generation.py` - Steer Evo2 hidden states toward/away from features
- `attribution_analysis.py` - Per-nucleotide attribution for SAE features
- `logit_lens.py` - Layer-wise analysis of feature emergence

**Visualization & Plotting:**
- `plot_tsne_by_annotation.py` - t-SNE colored by annotation type
- `plot_genome_karyotype.py` - Genome-wide karyotype visualization
- `plot_confidence_drops.py` - Entropy drop distribution plots
- `plot_sae_figure4.py` - Figure 4 style SAE visualizations
- `sae_annotation_overlay.py` - Feature + annotation overlay plots

**Data Comparison & Benchmarking:**
- `compare_detection_methods.py` - Compare 6 detection methods on same dataset
- `compare_sae_stats.py` - Compare standalone vs fused SAE stats
- `compare_ablations.py` - Compare entropy.npz files from different runs
- `benchmark_pipeline_timing.py` - End-to-end timing analysis
- `analyze_benchmarks.py` - Multi-run benchmark comparison

**Specialized Tools:**
- `curate_test_loci.py` - Extract 15 test loci (5 per organism) from annotations
- `find_novel_regions.py` - Identify drops not overlapping any GTF annotation
- `map_drops_to_exons.py` - Spatial analysis of drops relative to exons
- `scan_feature_genome.py` - Genome-wide scan for specific feature activation
- `optimize_consensus_based.py` - Optimize detection method parameters
- `optimize_drop_parameters.py` - Parameter tuning for drop detection
- `scan_sae_global_stats.py` - Extract global SAE statistics

---

## 4. Experiments & Analyses Run

### 4.1 Historical Pipeline Versions

**Version `jan22` (Original)**
- Basic Evo2 scoring with 3 detection methods (derivative, win_shift, cusum)
- Single-directory output
- Backward compatible

**Version `jan24` (Organized)**
- Structured output: `data/`, `plots/`, `fasta/`, `metadata/` folders
- Enhanced documentation
- Same algorithms as jan22

**Version `jan26_drops` (Current, High-Quality)**
- 4 statistical drop detection methods (zscore, MAD, local, bootstrap)
- Confidence scores per drop (e.g., `150:-3.24`)
- Variable marker sizes in plots based on confidence
- 10+ tunable CLI parameters
- **Performance:** 20-100 drops/gene → **5-20 drops/gene** (80% fewer false positives)
- **FPR reduction:** 30% → **5%**

### 4.2 Current Active Experiments (As of 2026-03-25)

#### **Pipeline A: High-Confidence (≥8.0) — PRIMARY**

**Status:** ✅ **ACTIVELY MERGING** (chr15)

| Component | Status | Details |
|-----------|--------|---------|
| SAE Extraction | ✅ COMPLETE | 36 shards per chromosome; 24 chromosomes finished (Mar 22-23) |
| Chr15 Merge | 🟡 RUNNING | Job 10987923 (pi_zhang_f); Started 2026-03-25; Phase: Normalization; ETA: 1-2.5 hours |
| Other Merges | ⏳ PENDING | Will start once chr15 finishes |
| Analysis | ⏳ PENDING | Will resubmit 23 per-chromosome analysis jobs |

**Key Outputs Expected:**
- `results/chr1-22/sae/*_conf8.0_merged/data/feature_matrices.npz` (1.5GB per chr)
- `results/chr1-22/sae/*_conf8.0_merged/data/sae_results.tsv` (per-region top features)
- Global t-SNE on high-confidence regions only

**Scientific Rationale:** High-confidence drops are the primary thesis result — most reliable functional elements.

---

#### **Pipeline B: All Regions (≥0.0) — EXPLORATORY**

**Status:** 🟡 **IN PROGRESS** (SAE extraction phase)

| Component | Status | Details |
|-----------|--------|---------|
| SAE Extraction | 🟡 RUNNING | ~80+ jobs queued/running; 25 RUNNING, 95 PENDING (ou_bcs partitions) |
| Auto-Monitor | 🔄 ACTIVE | Job `saeall_monitor` (4-day time limit) managing full pipeline |
| Merges | ⏳ PENDING | Auto-triggered by monitor once extraction completes |
| Analysis | ⏳ PENDING | Auto-triggered by monitor |

**Key Outputs Expected:**
- `results/chr1-22/sae/*_conf0.0_merged/data/feature_matrices.npz` (11GB per chr)
- Comprehensive analysis across all detected regions
- Potential novel functional elements at lower confidence

**Scientific Rationale:** Exploratory analysis to detect additional functional elements at lower thresholds.

---

#### **Bacteria Pipelines**

**E. coli (NC_000913.3):**
- Status: ✅ Scoring complete; SAE merges complete
- Results: `results/NC_000913.3/sae/*_conf8.0_merged/`
- Submitted: `submit_ecoli_latent_tsne.sh` (2026-03-26)
- Current: Awaiting analysis job completion

**Bacillus subtilis (NC_000964.3):**
- Status: ✅ Scoring complete; SAE merges complete
- Results: `results/NC_000964.3/sae/*_conf8.0_merged/`
- TSV Regeneration: Job 11001096 (RUNNING, 90-min allocation, Mar 25)
- Submitted: `submit_bacillus_latent_tsne.sh` (2026-03-26)
- Current: TSV regen in progress

---

### 4.3 Downstream Analyses

**Cross-Organism Comparison** (When data available)
- Feature activation pattern similarity: Human vs E. coli vs Bacillus
- Determines if Evo2 learned universal biological concepts

**t-SNE Visualization** (In progress)
- Genome-wide (all regions in 2D latent space)
- Colored by: annotation type, organism, confidence level
- Jobs: `genome_sae_tsne.py`, per-chromosome plots

**Annotation Classifier**
- Linear probe: Predict CDS/exon/intron/intergenic from SAE features
- Use case: Test if model can reconstruct basic genomic annotations

**Differential Feature Discovery**
- Find SAE features enriched in specific regions (e.g., intergenic)
- Use FDR thresholds for significance

---

## 5. Current Job Status (As of 2026-03-25)

### Active Jobs

| Job ID | Name | Status | Stage | Queue | Runtime | ETA |
|--------|------|--------|-------|-------|---------|-----|
| 10987923 | chr15_merge | 🟡 RUNNING | Normalization | pi_zhang_f | 4h 43m | 1-2.5h |
| 11001096 | regen_bacillus_tsv | 🟡 RUNNING | TSV generation | pi_zhang_f | 4m 27s | 85min |
| saeall_chr*_s* | SAE extraction | 25 RUNNING / 95 PENDING | Shard processing | ou_bcs | varies | 2-4h |
| saeall_monitor | Auto-monitor | 🟡 RUNNING | Waits for extraction | pi_zhang_f | 1d 18h+ | 2-3 days |

### Critical Notes

**Chr15 Merge Phases:**
1. ✅ Feature matrix merge → 8.1 GB compressed NPZ (completed ~17:48 EDT)
2. 🔄 Normalization stats → Computing mean/std for 4096 features (IN PROGRESS)
3. ⏳ Signature features → Re-extract across merged regions
4. ⏳ COMPLETED sentinel → Written when all phases finish

**Why It's Slow:**
- Streaming 36 shard chunk files without loading all into RAM
- ZIP compression is CPU-bound, not I/O-bound
- Numerically stable Welford's algorithm processes all ~500k regions × 4096 features

**Bacillus TSV Regeneration:**
- Previous job (10995430): TIMEOUT after 30 min (insufficient time)
- Current job (11001096): 90-min allocation (3× buffer) — should succeed

---

## 6. Results Directory Structure (Expected)

### Current State
No unified `results/` directory on this local copy (results are on ORCD cluster).

### Expected Structure on ORCD (/orcd/data/zhang_f/001/platawa/jan31_files/)

```
results/
├── NC_000913.3/                     # E. coli
│   └── sae/
│       ├── *_conf8.0_shard0of36/    # Per-shard output
│       │   ├── data/
│       │   │   ├── _chunk_*.npz     # Incremental features
│       │   │   └── _checkpoint_meta.json
│       │   └── COMPLETED
│       └── *_conf8.0_merged/        # Merged result
│           ├── data/
│           │   ├── feature_matrices.npz
│           │   ├── sae_results.tsv
│           │   └── _checkpoint_meta.json
│           └── COMPLETED
│
├── NC_000964.3/                     # Bacillus
│   ├── sae/                         # Same structure as E. coli
│   └── [TSV regeneration in progress]
│
└── chr1-22, chrX, chrY/             # Human chromosomes
    ├── chr1/
    │   └── sae/
    │       ├── 20260324_144121_all_conf8.0_shard0of36/   # Conf 8.0 shards
    │       ├── ...
    │       ├── 20260323_170054_all_conf8.0_merged43of36/  # ~1.5GB (COMPLETE)
    │       ├── 20260324_144121_all_conf8.0_merged43of36/  # ~11GB (conf0.0, mislabeled)
    │       └── [Other chroms follow same pattern]
    └── [22 more chromosomes]
```

### Key Output Files

**Per-Chromosome, Per-Shard:**
- `_chunk_0.npz`, `_chunk_1.npz`, ... — Incremental feature chunks
- `_checkpoint_meta.json` — Checkpoint metadata (N_regions, features extracted)

**Per-Chromosome, Merged:**
- `feature_matrices.npz` — Unified feature matrix (N_regions × 4096)
- `sae_results.tsv` — Per-region top K features (TSV with headers: `region_id`, `feature_id`, `activation_strength`, `rank`)
- `signature_features.tsv` — Features recurring across regions
- `COMPLETED` — Sentinel file (exists only if merge succeeded)

**Normalization & Stats:**
- Computed by `finish_merges.py` post-merge
- Global mean/std for 4096 features
- Per-region z-scored activations

---

## 7. Data Flow & Confidence Levels

### Confidence Score Definition
**Confidence = statistical significance of entropy drop**

- Computed by detection methods (zscore, MAD, local, bootstrap)
- Higher confidence = more significant drop
- Example: `region_at_position_150` has `confidence = -3.24` (z-score)

### Pipeline Splits by Confidence

| Confidence Level | Interpretation | Use Case | # Regions | Size/Chr |
|------------------|-----------------|----------|-----------|----------|
| **≥ 8.0** | High specificity | PRIMARY: High-confidence functional elements | ~5-20 per gene | 1.5 GB |
| **≥ 0.0** | All detected | EXPLORATORY: Comprehensive element discovery | All drops | 11 GB |
| **Intermediate** (4.0, 6.0) | [Tested but not in current pipeline] | Parameter exploration | ~ | ~5-8 GB |

### Naming Convention Quirk
⚠️ **Known Issue:** The 11GB conf0.0 merge is labeled `_conf8.0_` in directory name. This is a bug in the auto-monitor's naming. **File size (11GB) indicates it's actually conf0.0 data.**

---

## 8. Major Issues & Known Limitations

### Critical Issues (2026-03-25)

| Issue | Severity | Status | Impact |
|-------|----------|--------|--------|
| Chr15 merge slowly progressing (4.7h elapsed) | MEDIUM | MONITORING | Blocks downstream analysis |
| Bacillus TSV regen timeout (prev attempt) | MEDIUM | RESOLVED | Resubmitted with 3× time |
| Two parallel pipelines (conf8.0 & 0.0) running | MEDIUM | EXPECTED | Uses queue capacity but intended |
| Dir naming mismatch (conf0.0 labeled as conf8.0) | LOW | DOCUMENTED | Know which file is which by size |

### Historical Issues (Resolved)

- **Analysis jobs reference non-existent script:** Fixed by moving `analyze_sae_regions.py` to tools/
- **Job dependencies unclear:** Now documented in QUEUE_SUBMISSION_GUIDELINES.md
- **Memory bottlenecks in merge:** Solved with streaming architecture in `merge_sae_shards_fast.py`

### Design Limitations

1. **Two-pipeline approach:** Running both conf8.0 and conf0.0 simultaneously uses significant queue resources. Consider prioritizing one.
2. **Sharded SAE extraction:** Requires merging across shards; adds complexity but necessary for genome-scale.
3. **Long merge times:** Feature matrix compression is CPU-bound; ~4-8h per chromosome.
4. **Monitor job dependency:** Auto-monitor stays alive for 4 days; if killed, pipeline stops.

---

## 9. Key Metrics & Timeline

### Processing Metrics
- **Scoring speed:** ~4.5 sec/Mb of sequence (GPU, Evo2)
- **SAE extraction:** 30-60 min per shard (GPU, 8 cpus, 200GB RAM)
- **Merge speed:** 4-8 hours per chromosome (CPU-bound compression)
- **t-SNE computation:** ~1-2 hours per dataset (CPU-heavy)
- **Analysis per chromosome:** 30-60 min (CPU, 4 cores, 32GB RAM)

### Timeline (Expected)

```
2026-03-25 20:30 UTC — Chr15 merge in normalization phase
├─ 2026-03-25 22:30 ← Chr15 merge completes (est. +2h)
├─ 2026-03-25 22:31 ← Resubmit chr1-22 merges (sequential, 8h each)
│  └─ 2026-03-26 06:30 ← Chr1-22 merges complete (~32h total)
├─ 2026-03-26 07:00 ← Analysis jobs resubmitted
│  └─ 2026-03-26 08:00 ← Per-chromosome analysis complete
├─ 2026-03-26 08:30 ← Genome-wide t-SNE + cross-organism comparisons
│  └─ 2026-03-26 10:30 ← Downstream visualizations complete
└─ 2026-03-27 onwards ← Conf0.0 pipeline continues (lower priority)
```

### Job Submission Timeline

| Date | Event |
|------|-------|
| 2026-03-22 | Conf8.0 SAE extraction complete |
| 2026-03-23 | Conf8.0 merges begin; conf0.0 SAE extraction starts |
| 2026-03-24 | Chr15 merge (conf8.0, 1.5GB) completes; chr15 merge (conf0.0, 11GB) submitted |
| 2026-03-25 | 23 analysis jobs cancelled (resubmit later); chr15 merge still running |
| 2026-03-26 | (Planned) Chr15 merge finishes; per-chromosome analysis resubmits |

---

## 10. Running Jobs & Next Steps

### ✅ Now Complete
- Scoring: all 24 chromosomes + 2 bacteria
- Conf8.0 SAE extraction: all shards (Mar 22-23)
- Conf8.0 merges: 24 chromosome shards available

### 🟡 In Progress (3-8 hours)
- Chr15 merge (conf8.0): Job 10987923, normalization phase, ETA 1-2.5h
- Bacillus TSV regen: Job 11001096, 4m elapsed, ETA 85m
- Conf0.0 SAE extraction: ~120 jobs queued, ongoing

### ⏳ Pending (Next 24-48 hours)
1. Chr15 merge finishes → resubmit chr1-22 merges
2. All merges complete → resubmit 23 per-chromosome analysis jobs
3. Analysis complete → generate genome-wide t-SNE
4. Conf0.0 shards complete → merge conf0.0 chromosomes

### 🔮 Future (Days 2-7)
- Cross-organism feature comparison
- Linear probe classifier training
- Intergenic region analysis
- Differential feature discovery
- Advanced visualizations & publications

---

## 11. Key Resources

### Documentation Files
- **README.md** — Full project overview, pipeline diagram, usage examples
- **CHANGELOG.md** — Version history, migration guide (jan22 → jan26_drops)
- **PIPELINE_INVESTIGATION.md** — Deep dive into two-pipeline architecture, confidence level definitions
- **QUEUE_SUBMISSION_GUIDELINES.md** — Queue selection (pi_zhang_f for CPU work, ou_bcs for GPU)
- **CURRENT_PIPELINE_STATUS.md** — Real-time job status snapshot

### Critical Cluster Paths
- **Project root:** `/orcd/data/zhang_f/001/platawa/jan31_files`
- **Results:** `/orcd/data/zhang_f/001/platawa/jan31_files/results/`
- **Logs:** `/orcd/data/zhang_f/001/platawa/jan31_files/logs/`
- **Data (input):** `/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/`

### Local Test Results
- `/Users/parid/Downloads/jan31_files/local_review/results/` — Sample analyses

### Code Organization
- **Pipeline:** `run_sae_fast.py`, `merge_sae_shards_fast.py`, `finish_merges.py`
- **Analysis:** `tools/` directory (41 scripts)
- **Utilities:** `sae_utils.py`, `results_utils.py`

---

## 12. Summary Table

| Aspect | Status | Details |
|--------|--------|---------|
| **Repository Type** | Active Research | DNA language model analysis pipeline |
| **Primary Model** | Evo 2 (7B+) | Pre-trained, scoring only (no fine-tuning) |
| **Main Data** | Human genome + 2 bacteria | 24 chromosomes per organism |
| **Current Pipeline** | Dual-track | Conf≥8.0 (PRIMARY) + Conf≥0.0 (EXPLORATORY) |
| **Analysis Methods** | 6 detection methods + SAE | Statistical significance + unsupervised feature learning |
| **Key Output** | Feature matrices + t-SNE | Per-region SAE activations + visualization |
| **Current Phase** | Merging & Analysis | Chr15 merge in progress; analysis pending |
| **Next 24h** | Active monitoring | Merge completion → analysis resubmission |
| **Compute Resources** | MIT cluster (ORCD) | GPU (ou_bcs_*, mit_preemptable), CPU (pi_zhang_f) |
| **Documentation Level** | Excellent | 8 markdown files + 41 tools with docstrings |
| **Version** | jan26_drops | Statistical drop detection, 80% fewer FPs |

---

**End of Summary** — Repository comprehensively catalogued and documented.
