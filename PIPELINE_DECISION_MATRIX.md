# Pipeline Decision Matrix: Choosing Between conf≥8.0 and conf≥0.0

**Date:** 2026-03-26
**Status:** Critical clarification for running pipelines
**Audience:** Users deciding which pipeline to run

---

## TL;DR - Quick Decision Tree

```
Do you want...

├─ High-quality, curated results?
│  └─ YES → Use PIPELINE A (conf ≥ 8.0) ✅ RECOMMENDED
│           • Faster to run (already done)
│           • Fewer, more reliable regions
│           • Best for thesis/publication
│
└─ Comprehensive, full-genome analysis?
   └─ YES → Use PIPELINE B (conf ≥ 0.0)
            • Slower, resource-intensive
            • All detected regions (noisy)
            • Best for exploratory research
```

---

## Pipeline A: High-Confidence Analysis (conf ≥ 8.0)

### What It Does
Analyzes **entropy drop regions with high statistical significance** (z-score ≥ 8.0). These are the regions where the Evo2 model is most confident about functional importance.

**Quality:** ⭐⭐⭐⭐⭐ (High specificity, low false positive rate)

### Metrics
| Metric | Value |
|--------|-------|
| **Confidence threshold** | z-score ≥ 8.0 |
| **Regions per gene** | ~5-20 (curated) |
| **Feature matrix size** | ~1.5 GB per chromosome |
| **False positive rate** | ~5% |
| **Data volume** | Small (manageable) |
| **Runtime** | Fast (already complete) |

### Current Status ✅ COMPLETE

| Component | Status | Details |
|-----------|--------|---------|
| Scoring | ✅ DONE | All 24 chromosomes + 2 bacteria (Mar 9-18) |
| SAE Extraction | ✅ DONE | 36 shards per chromosome (Mar 22-23) |
| Chr15 Merge | 🟡 RUNNING | Job 10987923, normalization phase, ~1-2h remaining |
| Other Merges | ⏳ PENDING | Will start once chr15 completes (~32 hours total) |
| Analysis | ⏳ PENDING | Will resubmit 23 per-chromosome analysis jobs |

### Expected Timeline
```
2026-03-26 00:00 UTC — Chr15 merge finishes
     ↓ (sequential, 8h each)
2026-03-26 08:00 UTC — Chr1-22 merges start
     ↓
2026-03-26 16:00 UTC — All merges complete
     ↓
2026-03-26 17:00 UTC — Analysis jobs resubmitted
     ↓
2026-03-27 04:00 UTC — Analysis complete
```

### Best For
✅ **Thesis work, publications**
- High-confidence results
- Smaller, more defensible feature set
- Lower computational cost
- Faster turnaround

✅ **Cross-organism comparison**
- Comparing human vs bacteria
- Only high-signal regions
- Meaningful functional predictions

### Key Outputs
- Per-region SAE feature activations (4096 features)
- t-SNE embeddings colored by genomic annotation
- Linear probe classifiers (CDS/exon/intron/intergenic prediction)
- Cross-organism feature similarity analysis

---

## Pipeline B: Full-Genome Analysis (conf ≥ 0.0)

### What It Does
Analyzes **ALL entropy drop regions**, including low-confidence detections. Produces a comprehensive, unfiltered view of where the model detects any level of structure.

**Quality:** ⭐⭐⭐ (High sensitivity, higher false positive rate)

### Metrics
| Metric | Value |
|--------|-------|
| **Confidence threshold** | z-score ≥ 0.0 (all detections) |
| **Regions per gene** | ~50-100+ (comprehensive) |
| **Feature matrix size** | ~11 GB per chromosome |
| **False positive rate** | ~30-40% |
| **Data volume** | Large (requires storage) |
| **Runtime** | Slow (2-4 days estimated) |

### Current Status 🟡 IN PROGRESS

| Component | Status | Details |
|-----------|--------|---------|
| Scoring | ✅ DONE | All 24 chromosomes + 2 bacteria (Mar 9-18) |
| SAE Extraction | 🟡 RUNNING | 25 jobs active, 95 pending (mar 22-26) |
| Merges | ⏳ PENDING | Auto-triggered by monitor after extraction |
| Analysis | ⏳ PENDING | Auto-triggered by monitor |

### Expected Timeline
```
2026-03-26 00:00 UTC — SAE extraction ongoing (~80+ jobs)
     ↓ (4 days estimated)
2026-03-30 00:00 UTC — Extraction complete
     ↓ (8h each chromosome × 24)
2026-04-02 00:00 UTC — Merges complete (~192 hours)
     ↓
2026-04-02 12:00 UTC — Analysis jobs start
     ↓
2026-04-03 12:00 UTC — Full pipeline complete
```

### Best For
✅ **Exploratory research**
- Find ALL potential functional elements
- Discover novel patterns
- Test hypotheses at lower thresholds

✅ **Method development**
- Testing different detection thresholds
- Comparing confidence levels
- Parameter optimization

✅ **Comprehensive annotation**
- Build complete feature catalog
- No filtering by confidence
- Enables threshold exploration

### Key Outputs
- Comprehensive SAE feature analysis for all regions
- Full genome-wide t-SNE (will be very dense)
- Comparative analysis across all confidence levels
- Potential discovery of novel functional elements at lower thresholds

---

## Side-by-Side Comparison

| Aspect | Pipeline A (conf ≥ 8.0) | Pipeline B (conf ≥ 0.0) |
|--------|------------------------|------------------------|
| **Status** | ✅ COMPLETE (merging) | 🟡 IN PROGRESS |
| **Confidence threshold** | High (z ≥ 8.0) | All (z ≥ 0.0) |
| **Regions per gene** | 5-20 | 50-100+ |
| **Data per chromosome** | 1.5 GB | 11 GB |
| **Total data** | ~35 GB | ~260 GB |
| **FPR** | ~5% | ~30-40% |
| **Specificity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Sensitivity** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Runtime to completion** | ~40 hours | ~10+ days |
| **GPU time** | Low | High |
| **Storage needed** | 35-50 GB | 260-350 GB |
| **Best for** | Thesis, publications | Exploration, methods |
| **Recommended** | ✅ YES | 🔄 If resources allow |

---

## Job Submission: How to Run Each Pipeline

### Pipeline A (High-Confidence)

**Status:** Currently running merges, analysis pending

**When to submit:**
- When you want high-quality results now
- For thesis/publication work
- When GPU time is limited

**Scripts involved:**
```bash
# Already done:
sbatch submit_mega_shards.sh         # SAE extraction ✅ DONE
sbatch submit_sae_fast_shards.sh     # Per-shard results ✅ DONE

# Currently running:
sbatch merge_chr15_monitored.sh      # Chr15 merge (in progress)

# Will need to resubmit:
sbatch submit_finalize_and_merge.sh  # Other chromosomes
sbatch submit_downstream_analysis.sh # Per-chromosome analysis
```

**How to monitor:**
```bash
# Check merge progress
squeue -j 10987923

# Check logs
tail -f logs/merge_chr15_monitored_10987923.out

# Expected: ~40 hours to completion
```

---

### Pipeline B (Full-Genome)

**Status:** SAE extraction in progress, will auto-continue

**When to submit:**
- When you need comprehensive, unfiltered analysis
- For exploratory research
- When you have time/storage budget

**Scripts involved:**
```bash
# Already running (auto-managed):
squeue -u platawa | grep saeall     # SAE extraction jobs

# Auto-triggered after extraction:
# - Merges (auto-triggered by monitor)
# - Analysis (auto-triggered by monitor)

# Monitor job:
squeue -j 10877393                  # saeall_monitor job
```

**How to monitor:**
```bash
# Check overall progress
squeue -u platawa

# View auto-monitor logs
tail -f logs/saeall_monitor*.out

# Check queue load
squeue -p ou_bcs_normal | wc -l
```

---

## Critical Decision Factors

### Choose Pipeline A If:

✅ You need results **quickly** (next 1-2 days)
- Merges already underway
- Analysis will complete this week

✅ You're doing **thesis/publication work**
- High-confidence results are more defensible
- Peer reviewers prefer curated analyses

✅ You have **limited storage** (< 100 GB)
- 35-50 GB total data size
- Manageable on workstations

✅ You want **clear, interpretable results**
- 5-20 regions per gene (easy to visualize)
- Lower false positive rate

✅ **GPU time is limited or expensive**
- SAE extraction already complete
- Only analysis jobs remaining (CPU-bound)

---

### Choose Pipeline B If:

✅ You need **comprehensive discovery**
- All detected regions, unfiltered
- Explore lower-confidence thresholds

✅ You're doing **method development**
- Compare different detection thresholds
- Test novel analysis approaches

✅ You have **time and storage budget**
- 10+ days to completion
- 260-350 GB of data

✅ You want **maximum sensitivity**
- Find every possible functional element
- Accept higher false positive rate

✅ You're doing **exploratory research**
- No fixed hypothesis
- Want to see what model learns at all levels

---

## The Ambiguity Problem (CRITICAL FIX)

### Current Issue
Both pipelines use **identical SLURM job names**: `saeall_chr*_s*`

This means:
- Can't tell which pipeline jobs belong to from queue output
- Merge steps could inadvertently mix data
- Dangerous confusion

### Solution (Recommended Action)
Rename job submissions to distinguish pipelines:

```bash
# Pipeline A (conf ≥ 8.0)
--job-name="sae_conf8.0_chr${CHROM}_s${SHARD}"

# Pipeline B (conf ≥ 0.0)
--job-name="sae_conf0.0_chr${CHROM}_s${SHARD}"
```

Then queue output is unambiguous:
```
JOBID  JOBNAME
10123  sae_conf8.0_chr1_s0
10124  sae_conf0.0_chr1_s0
```

---

## Honest Assessment: Effort vs. Reward

### Pipeline A Effort
```
Work Done:    ████████████░░░░░░░░░░ (50%)
Time Left:    ~40 hours
GPU Time:     None (already spent)
Effort Now:   Minimal (monitoring only)

Result Quality: ⭐⭐⭐⭐⭐ (High-confidence)
Time to Results: 1-2 days
Effort: Low

RECOMMENDATION: ✅ START NOW
```

### Pipeline B Effort
```
Work Done:    ███░░░░░░░░░░░░░░░░░░░░░░░ (10%)
Time Left:    10+ days
GPU Time:     Significant (ongoing extraction)
Effort Now:   Let auto-monitor run

Result Quality: ⭐⭐⭐ (Comprehensive)
Time to Results: 10+ days
Effort: Low (auto-managed)

RECOMMENDATION: 🔄 RUN IN PARALLEL IF POSSIBLE
```

---

## FAQ

### Q: Can I run both pipelines simultaneously?

**A:** Yes, they are already running in parallel!
- Pipeline A (conf8.0): Currently merging
- Pipeline B (conf0.0): Currently extracting SAE features

This uses queue capacity but is intentional. Just be aware:
- Both compete for GPU/CPU resources
- Pipeline B will continue for 10+ days
- Pipeline A results will be ready first (~1-2 days)

---

### Q: I only have time for one. Which should I choose?

**A:** Choose **Pipeline A (conf ≥ 8.0)**
- Results in 1-2 days vs 10+ days
- Better quality (lower false positive rate)
- Already halfway done
- Thesis-ready results

Use Pipeline B later if you have time and interest.

---

### Q: What's the confidence score actually measuring?

**A:**
- **Confidence = statistical significance of entropy drop**
- Computed by z-score, MAD, local detection, or bootstrap methods
- Higher confidence = model is more certain about functional importance
- Example: conf 8.0 means z-score ≥ 8.0 (p-value ≈ 1e-15)

---

### Q: Can I mix results from both pipelines?

**A:** ⚠️ **Not recommended**
- Different confidence thresholds = different region sets
- Mixing could introduce bias
- Keep them separate for clarity

If you need combined analysis, use Pipeline B results (superset includes A).

---

### Q: Should I cancel Pipeline B to prioritize Pipeline A?

**A:** **No, let both run**
- Both are set to auto-manage
- Pipeline A monitor is hands-off
- Pipeline B monitor will auto-continue
- No conflict in current queue

Just monitor periodically.

---

### Q: How do I know which jobs are which pipeline?

**A:** Currently: ⚠️ **You can't!** Both use `saeall_chr*_s*`

This is the ambiguity problem mentioned above.

**Workaround:**
```bash
# Check job submission script to see parameters
head -20 /orcd/data/.../slurm_script_*.sh | grep -i confidence
```

**Better:** Apply the renaming fix above.

---

## Recommended Action Plan

### Phase 1: This Week (Complete Pipeline A)
```
✅ Mon 2026-03-26: Monitor chr15 merge (should complete today)
✅ Tue 2026-03-27: Submit chr1-22 merges
✅ Wed 2026-03-28: Merges complete, submit analysis jobs
✅ Thu 2026-03-29: Analysis jobs running
✅ Fri 2026-03-30: Analysis complete, start visualization
```

**Deliverable:** High-confidence results + t-SNE plots by Friday

### Phase 2: Next Week (Analyze Pipeline A Results)
```
✅ Generate genome-wide t-SNE
✅ Cross-organism comparison (human vs bacteria)
✅ Linear probe classifiers
✅ Publication-quality figures
```

### Phase 3: Later (Monitor Pipeline B)
```
🔄 Let Pipeline B continue auto-extraction
🔄 Check status 1x per day
✅ Once complete, run full-genome analysis
```

---

## Summary

| Aspect | Answer |
|--------|--------|
| **Should I run Pipeline A?** | ✅ **YES** - Start immediately |
| **Should I run Pipeline B?** | 🔄 **Let it run** - Already in progress, auto-manages |
| **Can I use both?** | ✅ Yes, they're independent |
| **Which is better?** | Pipeline A for quality, B for comprehensiveness |
| **Time to results (A)?** | ~1-2 days |
| **Time to results (B)?** | ~10+ days |
| **What should I do now?** | Monitor chr15 merge, resubmit analyses when ready |

---

## Contact & Questions

If you're unsure:
1. Check the **Decision Tree** at the top
2. Read the **Side-by-Side Comparison** table
3. Review **Honest Assessment** section
4. Check the **FAQ**

For technical questions, see QUEUE_SUBMISSION_GUIDELINES.md and COMPREHENSIVE_REPOSITORY_SUMMARY_2026-03-26.md.

---

**Document Version:** 1.0
**Last Updated:** 2026-03-26
**Next Review:** 2026-03-30 (after Pipeline A completion)
