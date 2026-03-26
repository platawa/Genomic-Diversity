# Queue Submission Guidelines for jan31_files Pipeline

## Current Status Summary (2026-03-25)

### SAE Extraction Pipeline
- **Pipeline A (conf >= 8.0):** ✅ COMPLETE
  - 36 shards per chromosome, all finished (Mar 22-23)
  - Ready for merge

- **Pipeline B (conf >= 0.0):** 🟡 IN PROGRESS
  - ~80+ jobs currently running/queued
  - Managed by `saeall_monitor` job
  - Shards incomplete for most chromosomes (needs shard completion)

### Merge Operations
- **Chr15 (conf >= 8.0):** 🟡 RUNNING
  - Standard monitored merge job (10987923)
  - Estimated: 4-8 hours from submission
  - Other parallel approaches failed (script issues)

### Analysis Jobs
- **23 analysis jobs (conf >= 8.0):** ❌ CANCELLED
  - Waiting for chr15 merge to complete
  - Will resubmit after merge finishes

---

## Queue Selection Guidelines

Based on observed performance and constraints:

### 1. **SAE Extraction (GPU Jobs)**

#### Use `mit_preemptable` for:
- ✅ Short-running jobs (< 2 hours)
- ✅ Non-critical work that can be restarted
- ✅ Batched jobs (can recover if preempted)
- ❌ Long-running merges/critical work
- **Constraint:** GPU queue is heavily saturated (100+ pending)
- **Recommendation:** Use for parallelized small jobs, not for large merges

#### Use `ou_bcs_low` for:
- ✅ Extended runs (2-12 hours)
- ✅ SAE extraction with moderate resource needs
- ✅ Parallel jobs that need sustained access
- ❌ Critical CPU-only work (use pi_zhang_f instead)
- **Constraint:** 64 GPU limit, moderate queue depth
- **Recommendation:** Primary choice for SAE extraction

#### Use `ou_bcs_normal` for:
- ✅ High-priority GPU work
- ✅ When ou_bcs_low is saturated
- ❌ Small quick jobs (overkill)
- **Constraint:** Only 16 GPUs
- **Recommendation:** Overflow from ou_bcs_low

#### Use `mit_normal_gpu` for:
- ✅ 2 GPU allocation (less common)
- ✅ Multi-GPU training/processing
- ❌ Single-GPU work
- **Constraint:** Very limited availability
- **Recommendation:** Reserve for genuine multi-GPU needs only

---

### 2. **Merge Operations (CPU-Heavy)**

#### Use `pi_zhang_f` for:
- ✅ **BEST CHOICE** for merge jobs
- ✅ Long-running CPU work (4-24 hours)
- ✅ High-priority analysis
- ✅ All analysis/downstream tasks
- **Constraint:** None observed (reliable)
- **Recommendation:** **Default for all CPU work**

#### Do NOT use `mit_preemptable` or `ou_bcs_*` for:
- ❌ Merges (too long, can be preempted mid-merge)
- ❌ Analysis jobs (need guaranteed runtime)
- ❌ Critical downstream work

---

### 3. **Queue Allocation Matrix**

| Job Type | Duration | Use Queue | CPUs | Memory | Time Limit |
|----------|----------|-----------|------|--------|------------|
| SAE Extraction | 30-60 min | ou_bcs_low | 8 | 200G | 2h |
| Batched SAE | 2-3h each | ou_bcs_low | 8 | 200G | 4h |
| Parallel SAE (4×) | 2-3h total | ou_bcs_low | 32 | 512G | 4h |
| **Merge (Standard)** | **4-8h** | **pi_zhang_f** | **8** | **256G** | **12-24h** |
| **Merge (Parallel)** | **3-4h** | **pi_zhang_f** | **32** | **512G** | **6h** |
| **Merge (Incremental)** | **1-2h** | **pi_zhang_f** | **8** | **256G** | **4h** |
| Analysis (per chr) | 30-60 min | pi_zhang_f | 4 | 32G | 4h |
| Cross-chromosome | 1-2h | pi_zhang_f | 4 | 64G | 4h |

---

## Lessons Learned & Recommendations

### What Worked
✅ `pi_zhang_f` for CPU-heavy merge work (most reliable)
✅ Parallel job submission (takes advantage of queue parallelism)
✅ Progress monitoring (catch hangs early)
✅ Multiple merge strategies (incremental was theoretically best)

### What Failed
❌ Using default partitions for long-running merges (timeout/preemption)
❌ Merging all 36 shards at once (memory/I/O bottleneck)
❌ Sequential batch merging (inefficient, takes 8-12 hours)
❌ Not having fallback strategies

### Future Optimization
1. **Always use pi_zhang_f for merges** (CPU work is guaranteed)
2. **Use incremental merge** for large datasets (better than parallel batches)
3. **Set realistic time limits:**
   - SAE extraction: 2-4h per batch
   - Standard merge: 8-12h
   - Incremental merge: 2-4h
   - Analysis: 1-2h per chromosome
4. **Monitor progress actively** (catches hangs before timeout)
5. **Plan for queue saturation:**
   - When GPU queues backed up: prioritize critical SAE jobs
   - Stagger merges (don't submit 24 at once)
   - Use pi_zhang_f exclusively for downstream work

---

## Current Recommendations

### Immediate (Next 2-8 hours)
- [ ] Monitor chr15 merge (Job 10987923)
- [ ] Once complete: resubmit chr15 analysis job (pi_zhang_f)
- [ ] Start chr19 merge: `python merge_sae_shards.py --chrom chr19 --n_shards 36`

### Short-term (Next 12-24 hours)
- [ ] Let conf >= 0.0 SAE jobs complete (saeall_* jobs)
- [ ] Once shards complete: merge each chromosome (pi_zhang_f, 8h each)
- [ ] Stagger merges (don't submit all 24 at once)

### Medium-term (1-3 days)
- [ ] Downstream analysis (cross-organism, t-SNE, classifier)
- [ ] All should go to pi_zhang_f with 4h time limits
- [ ] Prioritize conf >= 8.0 results (high-confidence analysis)

---

**Document Version:** 2026-03-25
**Last Updated:** Based on observed queue behavior during chr15 merge
**Status:** Monitored merge job 10987923 still running
