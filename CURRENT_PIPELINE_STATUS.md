# Current Pipeline Status Summary — 2026-03-24 20:30 UTC

## Executive Summary
The conf>=8.0 analysis pipeline is in a **confused state** with contradictory job submissions and unclear dependencies. Multiple jobs are running/pending but the actual confidence levels and data flow are uncertain.

---

## Current Job Landscape

### 1. SAE Scoring (Conf Level UNKNOWN)
**Job Family:** saeall_chr*_s* (10877xxx range)
**Status:** 25 RUNNING | 95 PENDING | ~120 total shards
**Progress:** ~50% of shards completed
**What they do:** Score entropy drops with SAE model
**UNCERTAINTY:** Logs show `confidence >= 6.18` but directories say `conf8.0_shard`
**Time to completion:** 2-4 more hours (queue constrained)

### 2. Chr15 Conf=0.0 Merge
**Job ID:** 10943251
**Status:** RUNNING (5:39:23 elapsed)
**Queue:** pi_zhang_f
**What it does:** Merge 8 conf=0.0 shards into unified chr15 result
**Output:** results/chr15/sae/20260324_144121_all_conf8.0_merged43of36/data/feature_matrices.npz (11GB, actively writing)
**Time limit:** 12:00:00
**ETA:** ~30-60 min remaining
**NOTE:** File was JUST modified (20:20:38), so still actively processing

### 3. Chr15 Conf=8.0 Merge (NEW)
**Job ID:** 10952494
**Status:** PENDING (waiting on dependencies)
**Queue:** pi_zhang_f
**What it does:** Merge conf=8.0 shards for chr15
**Dependencies:** afterok:10877209:10877251 (two shard jobs)
**Time limit:** 6:00:00
**Output location:** TBD (will be created when job runs)
**ISSUE:** May not start if shard jobs have already finished

### 4. Analysis Jobs (chr1-22, X, Y, chr15)
**Job IDs:** 10956202-10956212 (23 jobs total)
**Status:** PENDING on pi_zhang_f
**What they call:** `tools/analyze_sae_regions.py`
**CRITICAL PROBLEM:** **Script does not exist** — all 23 jobs WILL FAIL immediately
**Submitted with:** No dependencies (can start immediately) OR depends on chr15 merge
**Time limit:** 4:00:00 each

### 5. Downstream Analysis Jobs
| Job | ID | Status | Purpose | Data Source |
|-----|----|---------|---------|----|
| genome_sae_tsne | 10943291 | RUNNING (5:01h) | Genome-wide t-SNE visualization | **UNKNOWN** |
| confidence_drops | 10943308 | PENDING | Extract high-conf drops | **UNKNOWN** |
| cross_organism | 10943309 | PENDING | Compare bacteria vs human | **UNKNOWN** |
| normalize_all | 10943290 | PENDING | Normalization | **UNKNOWN** |
| karyotype | 10943292 | PENDING | Karyotype plot | **UNKNOWN** |

---

## Chr15 Merged Data — Which One Is Which?

| Directory | Created | Size | Content | Confidence |
|-----------|---------|------|---------|------------|
| 20260323_170054_all_conf8.0_merged43of36 | Mar 23 17:00 | 1.5GB | feature_matrices.npz | **Likely conf=8.0** |
| 20260324_144121_all_conf8.0_merged43of36 | Mar 24 14:41 | **11GB** (updated 20:20) | feature_matrices.npz | **Likely conf=0.0** |

**Naming is misleading:** Both say "conf8.0" but file sizes suggest different confidence levels.

---

## Critical Uncertainties

### 1. What confidence are saeall_* jobs actually running?
- **Directory names say:** `conf8.0_shard`
- **Job logs show:** `confidence >= 6.18`
- **Need to clarify:** Are these conf=8.0, conf=6.18, or something else?

### 2. Should analysis use conf=8.0 or conf=0.0?
- 1.5GB merge (Mar 23) = likely conf=8.0
- 11GB merge (Mar 24) = likely conf=0.0
- **Which should downstream jobs use?**

### 3. Which merged directory do downstream jobs depend on?
- genome_sae_tsne is already running — which chr15 merge is it using?
- confidence_drops, cross_organism, etc. — what are they waiting for?

### 4. Why does the submitted analysis script not exist?
- I submitted 23 jobs calling `tools/analyze_sae_regions.py`
- **This script does not exist locally or on ORCD**
- All 23 jobs will fail immediately

### 5. Should job 10952494 even exist?
- Depends on shards 10877209:10877251
- But shard jobs may have already finished
- May never start due to unfulfilled dependencies

---

## What's Broken

| Issue | Severity | Impact |
|-------|----------|--------|
| Analysis jobs reference non-existent script | **CRITICAL** | 23 jobs will fail |
| Unclear which confidence level is being used | **HIGH** | Cannot validate results |
| Conflicting merges (conf=0.0 vs conf=8.0) | **HIGH** | Wrong data might be analyzed |
| Job 10952494 may have bad dependencies | **MEDIUM** | May never run |
| Downstream jobs unclear on data source | **MEDIUM** | May use wrong merged results |

---

## What's Working

✓ SAE scoring is progressing (saeall_* jobs scoring regions)
✓ Chr15 conf=0.0 merge is running (job 10943251 actively writing)
✓ Queue submissions are succeeding (jobs get created)
✓ Bacillus/E. coli merges completed earlier (Bacillus done, E. coli awaiting analysis)

---

## Timeline (If Everything Worked)

```
NOW (20:30)
  ├─ Shard jobs continue (~2-4 hours)
  │
  ├─ Job 10943251 finishes chr15 conf=0.0 merge (~30-60 min)
  │   └─ 11GB file completed
  │
  ├─ Job 10952494 runs chr15 conf=8.0 merge (IF it starts)
  │   └─ Depends on shard completion
  │
  └─ 23 analysis jobs attempt to run
      └─ ALL FAIL immediately (script doesn't exist)

Result: Conf=0.0 merge completes, conf=8.0 unclear, analysis fails
```

---

## Questions for User

1. **What confidence should the analysis use?**
   - conf=8.0 (high-confidence drops, 1.5GB per chr)?
   - conf=0.0 (all drops, 11GB per chr)?
   - Something else?

2. **What is the correct per-chromosome analysis workflow?**
   - Which script should be called?
   - What parameters?
   - What outputs are expected?

3. **Should I:**
   - Cancel job 10952494 (chr15 conf=8.0 merge)?
   - Cancel the 23 analysis jobs (10956202-10956212)?
   - Wait for job 10943251 to finish first?

4. **For the downstream jobs (genome_tsne, etc.):**
   - What data should they depend on?
   - Should they run now or wait?

---

## Recommendations (Pending Your Input)

**Immediate Actions Needed:**
1. **Clarify confidence levels:** What should saeall_* jobs run at?
2. **Clarify analysis workflow:** What script/process should analyze each chromosome?
3. **Clarify dependencies:** Which merged directory should be canonical?
4. **Cancel broken jobs:** The 23 analysis jobs will definitely fail

**Once Clarified:**
1. Cancel the broken analysis job submissions
2. Resubmit with correct script and dependencies
3. Verify downstream jobs are using correct data sources
4. Monitor job 10943251 to completion

---

**Generated:** 2026-03-24 20:30 UTC
**Status:** Awaiting user clarification
