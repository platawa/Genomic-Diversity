# Deep Investigation: Two Parallel SAE Pipelines
**Status:** 2026-03-24 — User investigating why conf=6.18 appears instead of conf=8.0 or 0.0
**Goal:** Clarify which pipeline is running, why, and its current state

---

## TL;DR: What's Happening

You are running **TWO PARALLEL HUMAN PIPELINES** that are **not properly isolated**:

| Pipeline | Confidence | Status | Script(s) | Job Pattern | Priority |
|----------|------------|--------|-----------|------------|----------|
| **A: High-Confidence** | >= 8.0 | IN PROGRESS | `submit_mega_shards.sh` or `submit_sae_fast_shards.sh` | `saeall_chr*_s*` | **🔴 HIGH** |
| **B: All-Regions** | >= 0.0 | **LAUNCHED YESTERDAY** | `submit_full_sae_pipeline.sh 0.0` | `saeall_chr*_s*` | 🟡 MEDIUM |

**The Problem:** Both pipelines use **identical job naming** (`saeall_chr*_s*`), which means:
- Current jobs in queue are **ambiguous** — could be either pipeline
- Output directories have different names but queue jobs don't show which pipeline they belong to
- Merge steps could accidentally combine data from different pipelines

**The Confusion:** Logs show `confidence >= 6.18` which is **neither 8.0 nor 0.0** — this needs investigation.

---

## Pipeline A: Confidence >= 8.0 (HIGH PRIORITY)

### What It Does
- Extracts **only high-confidence entropy drop regions** (confidence score ≥ 8.0)
- Runs SAE (Sparse Autoencoder) on these regions
- Produces feature matrices for analysis

### How It Works

**Submission Scripts:**
```bash
# Option 1: Mega-shards (36 jobs across 3 partitions, all chroms in sequence)
bash submit_mega_shards.sh              # defaults to 36 shards
bash submit_mega_shards.sh 36           # explicit

# Option 2: Per-chromosome shards (4 jobs per chromosome)
bash submit_sae_fast_shards.sh chr1     # 4 shards for chr1
bash submit_sae_fast_shards.sh chr1 8   # 8 shards for chr1
```

**Key Code:**
- `submit_mega_shards.sh:78` — hardcoded `--min_confidence 8.0`
- `submit_sae_fast_shards.sh:60` — hardcoded `--min_confidence 8.0`
- Both call: `python run_sae_fast.py --min_confidence 8.0`

**Job Names:**
- Mega: `sae_mega_s0`, `sae_mega_s1`, ..., `sae_mega_s35`
- Per-chr: `saef_chr1_s0`, `saef_chr2_s1`, etc.

**Output Directory Naming:**
- Format: `results/CHR/sae/YYYYMMDD_HHMMSS_all_conf8.0_shardNofM/`
- Example: `results/chr15/sae/20260323_170054_all_conf8.0_merged43of36/` (1.5GB)

**Merge:** `python merge_sae_shards.py --chrom chr1 --n_shards N --output_dir results/`

---

## Pipeline B: Confidence >= 0.0 (LOWER PRIORITY)

### What It Does
- Extracts **ALL detected entropy drop regions** (confidence >= 0.0, i.e., no filtering)
- Runs SAE on entire set
- Produces comprehensive feature matrices for broader analysis

### How It Works

**Submission Script:**
```bash
# Defaults: conf 0.0, 8 shards per chromosome, auto-monitor
bash submit_full_sae_pipeline.sh

# With explicit parameters:
bash submit_full_sae_pipeline.sh 0.0 8   # conf 0.0, 8 shards
bash submit_full_sae_pipeline.sh 0.0 36  # conf 0.0, 36 shards
bash submit_full_sae_pipeline.sh 4.0 8   # conf 4.0, 8 shards (intermediate)
```

**Key Code:**
- `submit_full_sae_pipeline.sh:22` — reads `MIN_CONF="${1:-0.0}"` (default 0.0)
- `submit_full_sae_pipeline.sh:69` — passes `--min_confidence ${MIN_CONF}` to run_sae_fast.py
- Includes **auto-monitor** that:
  - Waits for all SAE jobs
  - Resubmits failures (2 rounds)
  - Auto-runs merge + finish_merges
  - Takes 4 days (time limit)

**Job Names:**
- SAE extraction: `saeall_chr1_s0`, `saeall_chr1_s1`, ..., `saeall_chr22_sN`, `saeall_chrX_sN`, `saeall_chrY_sN`
- Monitor: `saeall_monitor`
- Merge: `mergeall_chr1`, `mergeall_chr2`, etc.
- Finish: `finishall`

**Output Directory Naming:**
- Format: `results/CHR/sae/YYYYMMDD_HHMMSS_all_conf0.0_shardNofM/`
- After merge: `results/CHR/sae/YYYYMMDD_HHMMSS_all_conf0.0_merged*/`
- Example: `results/chr15/sae/20260324_144121_all_conf8.0_merged43of36/` (11GB) ← **This should say conf0.0!**

**Merge:** Auto-run by monitor job via `merge_sae_shards.py`

---

## Current Job Status (as of 2026-03-24 20:30 UTC)

### Running Jobs
| Job Pattern | Count | Partition | GPU | Status | Likely Pipeline | Confidence |
|-------------|-------|-----------|-----|--------|-----------------|------------|
| `saeall_chr*_s*` | 25 RUNNING | mixed | various | Active | **A or B?** | 🔴 **UNCLEAR** |
| `saeall_chr*_s*` | 95 PENDING | mixed | various | Queued | **A or B?** | 🔴 **UNCLEAR** |
| `mergeall_chr*` | ? | ? | ? | ? | **B (auto-monitor)** | 0.0 |
| `sae_mega_s*` | ? | mixed | various | ? | **A** | 8.0 |

### Output Directories (Actual Data)

| Path | Size | Created | Status | Confidence | Which Pipeline? |
|------|------|---------|--------|------------|-----------------|
| `results/chr15/sae/20260323_170054_all_conf8.0_merged43of36/` | 1.5GB | Mar 23 17:00 | COMPLETED | 8.0 | A |
| `results/chr15/sae/20260324_144121_all_conf8.0_merged43of36/` | 11GB | Mar 24 14:41 | **ACTIVELY WRITING** | ??? | **B (but mislabeled!)** |

---

## 🔴 CRITICAL ISSUE #1: Directory Naming Mismatch

**Problem:** The second chr15 merge says `conf8.0` but:
1. **File size suggests it's conf0.0:**
   - conf8.0 should have ~1.5GB (fewer, higher-confidence regions)
   - conf0.0 should have ~11GB (all regions)
   - Actual: 11GB with `conf8.0` label = **MISLABELED**

2. **Possible causes:**
   - `submit_full_sae_pipeline.sh` was called with default parameters (0.0)
   - But output directory was hardcoded to say `conf8.0` instead of reading `${MIN_CONF}`
   - OR: Pipeline B's monitor auto-renamed it incorrectly

**Investigation:** Check the actual sharding in the directory:
- If `_merged43of36`, that's 36 shards (Pipeline B style)
- If `_shard43of36`, that's incomplete

---

## 🔴 CRITICAL ISSUE #2: Confidence = 6.18 in Logs

**Status document says:** "Logs show `confidence >= 6.18` but directories say `conf8.0_shard`"

**Possible explanations:**
1. **Test/exploratory run:** Someone ran a test with conf=6.18 that got mixed into the logs
2. **Intermediate confidence:** Neither pipeline was correctly set up
3. **Log output bug:** `run_sae_fast.py` might be parsing/printing confidence incorrectly
4. **Old pipeline:** From an earlier attempt (before consolidating to 8.0 vs 0.0)

**Need to investigate:**
- Check actual `run_sae_fast.py` logs on cluster: `tail -f logs/saeall_chr*_*.out`
- Look for `--min_confidence` in actual sbatch commands run
- Search for any hardcoded `6.18` in code

---

## 🟡 CRITICAL ISSUE #3: Job Name Collision

Both pipelines use **identical job naming pattern**: `saeall_chr*_s*`

**Current queue state is AMBIGUOUS:**
- 25 running `saeall_chr*_s*` jobs could be from Pipeline A or B
- 95 pending `saeall_chr*_s*` jobs could be from Pipeline A or B
- Merge jobs (if running) won't know which confidence level to expect
- Downstream analysis won't know which data to use

**To disambiguate, you need:**
1. Check exact job commands: `sinfo -l -j <jobid>` or job script contents
2. Check output directory timestamps: when were the shards created?
3. Verify merge job naming: is it `mergeall_*` (Pipeline B) or something else (Pipeline A)?

---

## Why Are These Pipelines Running?

### Pipeline A (conf≥8.0) — Started Before Yesterday
- **Timeline:** Already in progress on 2026-03-23
- **Reason:** High-confidence drops are the **primary thesis result** — most reliable functional elements
- **Priority:** 🔴 **HIGHEST** — finish first
- **Status:** Partially complete (1.5GB chr15 merge finished)

### Pipeline B (conf≥0.0) — Launched Yesterday
- **Timeline:** Submitted 2026-03-23 or 2026-03-24
- **Reason:** Exploratory analysis — see if you can detect additional functional elements at lower confidence
- **Priority:** 🟡 **MEDIUM** — start after Pipeline A finishes
- **Status:** In progress (11GB chr15 merge running, monitor job auto-managing full pipeline)

---

## Naming Conventions

### Directory Naming
```
results/
  <CHROM>/
    sae/
      YYYYMMDD_HHMMSS_all_conf<N.N>_shard<idx>of<total>/
        data/                       ← Per-shard extraction output
        COMPLETED                   ← Written when shard finishes
      YYYYMMDD_HHMMSS_all_conf<N.N>_merged<idx>of<total>/
        data/
          feature_matrices.npz      ← Merged across shards
          sae_results.tsv           ← Combined metadata
        COMPLETED
```

**Example Timeline for chr15, conf=0.0, 36 shards:**
1. Sharding phase: `20260324_144121_all_conf0.0_shard0of36/` through `shard35of36/` created incrementally
2. Merge phase: `20260324_144121_all_conf0.0_merged43of36/` (monitor auto-names it — "43of36" suggests partial merge mid-process or naming bug)
3. Final: Should be `YYYYMMDD_HHMMSS_all_conf0.0_merged/` or similar

**BUG?** `_merged43of36` doesn't match 36 shards (43 > 36). This is suspicious.

---

## Pipeline Execution Flow

### Pipeline A Flow (submit_mega_shards.sh or submit_sae_fast_shards.sh)
```
User: bash submit_mega_shards.sh 36
  ↓
36 jobs submitted: sae_mega_s0 through sae_mega_s35
  ↓
Each processes all 24 chroms sequentially with conf≥8.0
  ↓
Output: results/chr*/sae/*_conf8.0_shard*/
  ↓
Manual: python merge_sae_shards.py --chrom chr1 --n_shards 36
  ↓
Output: results/chr*/sae/*_conf8.0_merged*/
  ↓
Manual: python submit_finalize_and_merge.sh (chains downstream)
```

### Pipeline B Flow (submit_full_sae_pipeline.sh)
```
User: bash submit_full_sae_pipeline.sh 0.0 36
  ↓
$((24 * 36)) SAE jobs submitted: saeall_chr1_s0 through saeall_chrY_s35
  ↓
Monitor job submitted: saeall_monitor (takes 4 days, auto-manages everything)
  ↓
Monitor waits for all SAE jobs
  ↓
Monitor resubmits failures (2 rounds)
  ↓
Monitor triggers: python merge_sae_shards.py for all chroms
  ↓
Output: results/chr*/sae/*_conf0.0_merged*/
  ↓
Monitor triggers: python finish_merges.py
  ↓
Output: Normalization stats, final COMPLETED sentinel
```

---

## Questions Needing Answers

### 1️⃣ **Which pipeline was actually launched yesterday?**
   - Did you run `bash submit_mega_shards.sh` (Pipeline A)?
   - Did you run `bash submit_full_sae_pipeline.sh 0.0 36` (Pipeline B)?
   - Or both?

### 2️⃣ **What are the actual running jobs?**
   - Are the 25 RUNNING `saeall_chr*_s*` jobs from Pipeline A or B?
   - If Pipeline A, why are they named `saeall_*` instead of `sae_mega_*` or `saef_*`?
   - If Pipeline B, are they still running or just queued?

### 3️⃣ **Where did confidence=6.18 come from?**
   - Was this a test/exploratory run?
   - Should we ignore it?
   - Is it polluting the current pipeline?

### 4️⃣ **The chr15 merge mystery:**
   - The 11GB `conf8.0` merge (actively writing) — is it actually conf=0.0?
   - Should I cancel the merge and let Pipeline B's monitor handle it?
   - Or let it finish and move forward?

### 5️⃣ **What's your desired next state?**
   - Finish Pipeline A (conf≥8.0) completely, including downstream analysis?
   - Then start Pipeline B (conf≥0.0)?
   - Or do both in parallel (if you have queue capacity)?

### 6️⃣ **What about bacteria?**
   - Are E. coli and Bacillus still on hold?
   - Or did those finish on Mar 23?

---

## Recommendations (Waiting for Your Answers)

**Immediate (next 10 minutes):**
1. Check cluster job queue: `squeue -u platawa` — send output
2. Check actual log contents: `tail -100 logs/saeall_chr1_s0_*.out` — send last 50 lines
3. Clarify: Which submission script(s) did you actually run?

**After clarification:**
1. If Pipeline A + B are both running: you're OK, just need to understand which is which
2. If only one is running: might need to launch the other separately
3. If conf=6.18 is from an old run: safely ignore
4. Decide: prioritize conf≥8.0 to completion first, then conf≥0.0?

---

**Status:** Awaiting your clarification on the 6 questions above and current queue snapshot.
