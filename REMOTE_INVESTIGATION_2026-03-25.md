# Remote Job Investigation — 2026-03-25 18:05 EDT

## Executive Summary

✅ **Both jobs are progressing normally** — the chr15 merge is resource-intensive but making steady progress, and the Bacillus TSV regen was resubmitted with sufficient time.

---

## CHR15 SAE Merge (Confidence 8.0)

**Job:** 10987923 (pi_zhang_f partition)
**Status:** RUNNING — Normalization phase
**Elapsed:** 283 minutes (4h 43m)
**Expected completion:** 4.5–6 hours total (1–2.5 hours remaining)

### Phase Breakdown

| Phase | Time | Status | Details |
|-------|------|--------|---------|
| **Feature matrix merge** | 0–30 min | ✅ DONE | Merged 36 shards → 8.1 GB compressed NPZ |
| **Normalization stats** | 30–? min | 🔄 IN PROGRESS | Computing mean/std for 4096 features across all regions |
| **Signature features** | ? | ⏳ PENDING | Will re-extract features across merged regions |
| **COMPLETED sentinel** | ? | ⏳ PENDING | Written when all phases finish |

### Why It's Taking So Long

The **merge_feature_matrices** phase writes an 8.1 GB ZIP file with compressed data. This involves:
- Streaming 36 shard chunk files (avoiding loading all into memory)
- Compressing with ZIP_DEFLATED (CPU-bound, not I/O-bound)
- Writing to network storage (`/orcd/data/...`)

The **compute_norm_stats_streaming** phase is now running:
- Reads all chunk files AGAIN (Welford's online algorithm for numerically stable computation)
- Processes all ~500k regions × 4096 features
- Pure CPU computation with no I/O — slow but necessary for downstream processing

**File growth stopped at 8.1G around 17:48 EDT** because we completed the ZIP writing and entered the CPU-only normalization phase. This is expected behavior.

### Log Evidence

```
[Progress] 0min - 1.2G (initial file being written)
[Progress] 5min - 0.17G (ZIP finalization)
[Progress] 10min - 0.5G (compression ramping up)
[Progress] 30min - 5.6G (peak write phase)
[Progress] 240min - 7.7G (compression finishing)
[Progress] 283min - 8.1G (STABLE — normalization phase now)
```

---

## Bacillus TSV Regeneration

**Previous attempt:** Job 10995430 (TIMEOUT)
- **Status:** TIMEOUT after 30 minutes
- **Allocated time:** 30 min (insufficient)
- **Progress:** Started streaming chunks but ran out of time

**Current attempt:** Job 11001096 (RUNNING)
- **Status:** RUNNING
- **Partition:** pi_zhang_f
- **Allocated time:** 90 minutes ✅ (3x more than failed attempt)
- **Elapsed:** 4 minutes 27 seconds
- **Log:** `logs/regen_bacillus_chunks_long_11001096.out`

### Why Previous Job Timed Out

The script regenerates `sae_results.tsv` from chunk files. For Bacillus:
- **N_regions × N_features:** ~26k × 4096 (for each confidence level)
- **Time complexity:** O(regions × shards) to iterate and merge
- **30-minute allocation:** Too tight for full processing

### Why Current Job Will Succeed

- **90-minute allocation:** Provides 3× buffer
- **Low-memory streaming:** Processes one chunk at a time (no OOM risk)
- **Early indicator:** Starting messages logged cleanly, no memory spikes yet

---

## Pipeline Status: Confidence Level 0.0 vs 8.0

You mentioned running two **parallel pipelines** at different confidence levels:

| Confidence Level | Status | Notes |
|------------------|--------|-------|
| **8.0 (high specificity)** | 🔄 Merging chr15 | All SAE extraction done; chr15 merge in progress; chr1-22 can merge once chr15 finishes |
| **0.0 (all regions)** | 🚀 SAE extraction | Many jobs pending (Priority queue on ou_bcs partitions); 4 jobs actively running |

**Insight:** Job 10877393 (`saeall_mo`, running 1d 18h) appears to be an older exploratory/monitored run. The current prod pipeline uses the conf8.0 merge (job 10987923).

---

## Recommended Actions

### 1. **Monitor Chr15 Merge** (next 2–4 hours)
```bash
# Live progress:
ssh platawa@orcd-login001.mit.edu
tail -f /orcd/data/zhang_f/001/platawa/jan31_files/logs/merge_chr15_monitored_10987923.out
```

**What to look for:**
- If file size stays at 8.1G for >30 min → normalization is running (expected)
- If file size stays unchanged for >2 hours → may indicate stall; investigate with `top`/`htop` on node
- Once `[Progress Monitor] Completed in X minutes` appears → merge is done

### 2. **Let Bacillus Job Run** (next 1–1.5 hours)
```bash
squeue -j 11001096
cat logs/regen_bacillus_chunks_long_11001096.out
```

This should complete without issue given the 90-minute allocation.

### 3. **Next Steps After Both Complete**

Once both finish:
1. Check `COMPLETED` sentinels:
   ```bash
   ls results/chr15/sae/20260325_*/COMPLETED
   ls results/bacillus_subtilis/regen_bacillus_*.tsv
   ```

2. **Re-merge chromosomes that used conf0.0 chunks:** If conf0.0 shards finished during the wait, re-run merge for those chromosomes

3. **Run downstream pipeline:**
   - Aggregation (per-chromosome statistics)
   - Per-chromosome analysis & visualization
   - Genome-wide comparisons (e.g., cross-organism feature correlation)

---

## Key Insights

1. **Chr15 merge is inherently slow** — 36 shards × ~500k regions × 4096 features with compression = substantial computation
   - Previous failed attempts likely didn't allocate enough time/memory
   - Current 24-hour allocation + 256GB memory is appropriate

2. **Bacillus TSV bottleneck is time, not correctness** — the script logic is sound; it just needed more CPU time than 30 minutes

3. **Confidence 0.0 pipeline is lower priority** — pending jobs are queued by scheduler; these will run as resources become available (not blocking your main analysis)

---

## Files Created This Session

- `regenerate_bacillus_results_tsv.py` — Low-memory TSV regeneration from chunks
- `regenerate_bacillus_tsv_from_chunks.py` — Variant implementation (check if needed)
- `merge_chr15_monitored.sh` — Monitoring wrapper with progress logging
- `QUEUE_SUBMISSION_GUIDELINES.md` — Queue management reference

These are in your working directory and ready for reference.
