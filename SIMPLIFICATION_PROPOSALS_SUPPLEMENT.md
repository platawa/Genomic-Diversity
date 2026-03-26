# Additional Repository Simplification Proposals

**Date:** 2026-03-26
**Purpose:** Supplementary simplification recommendations beyond the architectural review
**Audience:** Developers planning repository improvements

---

## Executive Summary

Beyond the 15 recommendations in the architectural review, here are 8 additional proposals to further simplify the repository and improve usability. These focus on:
- Reducing cognitive load when navigating the project
- Automating repetitive tasks
- Improving discoverability
- Eliminating redundancy in documentation

**Time to Implement:** 2-4 weeks (in phases)
**Benefit:** 30-40% reduction in project complexity and cognitive overhead

---

## Proposal 1: Create a Unified Results Manifest

**Problem:**
- Results are scattered across `results/chr*/sae/`, `results/chr*/scoring/`, `results/NC_*/...`
- No single source of truth for what exists and where
- Hard to know which SAE runs are complete vs incomplete
- Difficult to track dependencies between stages

**Solution:**
Create a single `RESULTS_MANIFEST.json` file at the root level:

```json
{
  "generated_at": "2026-03-26T14:00:00Z",
  "chromosomes": {
    "chr1": {
      "scoring": {
        "run_id": "20260225_143012",
        "status": "complete",
        "path": "results/chr1/scoring/20260225_143012_rc_logprobs_4gpu",
        "completed_at": "2026-02-25T20:00:00Z"
      },
      "sae": {
        "run_id": "20260323_201618",
        "status": "complete",
        "confidence": "≥8.0",
        "path": "results/chr1/sae/20260323_201618_all_conf8.0_merged30of36",
        "shards": 36,
        "shards_complete": 36,
        "dependencies": ["scoring:20260225_143012"],
        "completed_at": "2026-03-23T20:16:00Z"
      },
      "analysis": {
        "status": "pending",
        "expected_start": "2026-03-28T00:00:00Z"
      }
    }
  },
  "organisms": {
    "NC_000913.3": {
      "name": "E. coli K-12",
      "sae": {
        "conf3.0": {
          "run_id": "20260309_132936",
          "status": "complete",
          "path": "results/NC_000913.3/sae/20260309_132936_max50_conf3.0",
          "latent_analysis": {
            "status": "complete",
            "completed_at": "2026-03-26T02:06:00Z"
          },
          "annotation_tsne": {
            "status": "complete",
            "completed_at": "2026-03-26T02:11:00Z"
          }
        }
      }
    }
  },
  "pipelines": {
    "pipeline_a": {
      "name": "High-Confidence (conf ≥8.0)",
      "status": "merging",
      "progress": "50%",
      "expected_completion": "2026-03-28T00:00:00Z"
    },
    "pipeline_b": {
      "name": "Full-Genome (conf ≥0.0)",
      "status": "extracting",
      "progress": "10%",
      "expected_completion": "2026-04-03T12:00:00Z"
    }
  }
}
```

**Benefits:**
- Single source of truth for all results
- Dependency tracking (which analyses depend on which results)
- Status at a glance
- Enables programmatic queries (e.g., "list all incomplete jobs")
- Can be auto-generated on job completion

**Implementation:**
- Create `tools/generate_manifest.py` to scan results directory
- Call it as post-processing step after each job
- Version it in git (update on major changes)

**Effort:** 4-6 hours

---

## Proposal 2: Consolidate Tool Documentation into a README

**Problem:**
- Tools are in `tools/` directory but no unified documentation
- Hard to know which tools to run and in what order
- Tool requirements and outputs not documented
- Analysis workflows not clearly described

**Solution:**
Create `tools/README.md` that documents:
1. Each tool's purpose, inputs, outputs
2. Which tools run in sequence
3. Example usage for each pipeline
4. Tool dependencies and requirements

**Example structure:**
```markdown
# Tools Reference Guide

## SAE Extraction Tools
### analyze_sae_regions.py
**Purpose:** Extract SAE features from entropy drop regions
**Inputs:** SAE run directory, confidence threshold
**Outputs:** latent_analysis/data, embeddings, cluster assignments
**Example:** python analyze_sae_regions.py --input_dir results/chr1/sae/... --embedding both

## Visualization Tools
### plot_tsne_by_annotation.py
**Purpose:** Generate t-SNE plots colored by genomic annotation
**Depends on:** analyze_sae_regions.py (must run first)
...

## Workflow Diagrams
### Pipeline A (conf ≥8.0) Workflow
scoring.py → SAE extraction → merge → analyze_sae_regions → plotting
```

**Benefits:**
- New users can quickly understand tool ecosystem
- Reduces time to run analyses
- Clear dependency ordering

**Effort:** 3-4 hours

---

## Proposal 3: Create a Single Configuration File

**Problem:**
- Hardcoded paths scattered throughout scripts (GTF files, FASTA files, reference data)
- Merging/updating reference data requires editing multiple files
- Different scripts use different path conventions

**Solution:**
Create `config.yaml` at root:

```yaml
project:
  name: Evo2 Genomic Analysis Pipeline
  version: 1.0
  work_dir: /orcd/data/zhang_f/001/platawa/jan31_files
  results_base: results

reference_data:
  human:
    name: GRCh38
    fasta: /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/.../GCF_000001405.26_GRCh38_genomic.fna
    gtf: /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/.../genomic.gtf
    chromosomes: [chr1, chr2, ..., chr22]

  ecoli:
    name: K-12 MG1655
    fasta: /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/.../GCF_000005845.2_ASM584v2_genomic.fna
    gtf: /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/.../genomic.gtf

  bacillus:
    name: subtilis 168
    fasta: /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/.../GCF_000009045.1_ASM904v1_genomic.fna
    gtf: /orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/.../genomic.gtf

slurm:
  partition: pi_zhang_f
  default_gpus: 1
  default_cpus: 8
  default_mem: 64G

pipelines:
  pipeline_a:
    confidence_threshold: 8.0
    enabled: true
  pipeline_b:
    confidence_threshold: 0.0
    enabled: true
```

**Benefits:**
- Change paths in one place instead of many
- Scripts become more portable
- Easier to switch between environments (local vs remote)

**Effort:** 2-3 hours (initial) + 1 hour per script to refactor

---

## Proposal 4: Simplify Script Naming Convention

**Problem:**
- Scripts have inconsistent naming: `submit_*.sh`, `regenerate_*.py`, `analyze_*.py`
- Hard to know which scripts are for which organism
- Job names in SLURM are ambiguous (proposal 15 in architecture review)

**Solution:**
Rename all scripts with consistent pattern: `{stage}_{organism}_{variant}.sh`

**Example refactoring:**
```
Current:                              Proposed:
submit_sae_fast_shards.sh        →   extract_human_conf8.0_shards.sh
submit_mega_shards.sh             →   extract_human_conf0.0_shards.sh
submit_finalize_and_merge.sh      →   merge_human_conf8.0.sh
submit_human_annotation_plots.sh  →   analyze_human_annotation_tsne.sh
submit_bacillus_latent_tsne.sh    →   analyze_bacillus_latent_tsne.sh
submit_ecoli_latent_tsne.sh       →   analyze_ecoli_latent_tsne.sh
```

**Benefits:**
- Immediately clear what each script does
- Easier to find scripts by stage or organism
- Makes batch operations easier (`ls extract_*.sh`)

**Effort:** 4-6 hours (refactoring + testing)

---

## Proposal 5: Create a Pipeline State Machine

**Problem:**
- Hard to know what the valid next steps are
- Job interdependencies not explicit
- Easy to accidentally run jobs out of order
- No single source of truth for pipeline state

**Solution:**
Create `pipeline_state.py` that enforces valid state transitions:

```python
class PipelineState(Enum):
    SCORING = "scoring"
    SAE_EXTRACTING = "sae_extracting"
    SAE_MERGING = "sae_merging"
    SAE_COMPLETE = "sae_complete"
    ANALYSIS_RUNNING = "analysis_running"
    ANALYSIS_COMPLETE = "analysis_complete"

class Pipeline:
    def __init__(self, chromosome, confidence):
        self.state = PipelineState.SCORING

    def can_extract_sae(self):
        return self.state == PipelineState.SCORING

    def can_merge_sae(self):
        return self.state == PipelineState.SAE_EXTRACTING

    def can_run_analysis(self):
        return self.state == PipelineState.SAE_COMPLETE
```

Then in submission scripts:
```bash
# Check state before submitting
python -c "from pipeline_state import Pipeline; p = Pipeline('chr15', 8.0); assert p.can_merge_sae()"
sbatch merge_chr15.sh
```

**Benefits:**
- Prevents accidental out-of-order submissions
- Clear error messages when trying invalid transitions
- Self-documenting state requirements

**Effort:** 3-4 hours (initial) + 30 min per submission script

---

## Proposal 6: Archive and Compress Old Results

**Problem:**
- Results directory has 607 files and growing
- Many old runs that are no longer active
- Takes time to navigate and search
- Consumes disk space unnecessarily

**Solution:**
Create archival script that moves completed, unused results:

```bash
# Create archive directory
mkdir -p results/_archive

# Move runs older than N days
find results/chr*/sae -type d -mtime +30 -exec tar czf {} \;
mv results/chr*/sae/*.tar.gz results/_archive/

# Keep only last 2-3 runs per stage per chromosome active
```

**Benefits:**
- Cleaner active results directory
- Faster directory navigation
- Same data, just archived
- Can be restored if needed

**Effort:** 2-3 hours (initial setup)

---

## Proposal 7: Create Experiment Tracking System

**Problem:**
- Hard to compare results across different runs
- No metadata about what changed between runs
- Can't easily ask "what was different between run A and B?"

**Solution:**
Create `experiments.json` to track each analysis:

```json
{
  "experiments": [
    {
      "id": "exp_001_pipeline_a_initial",
      "date": "2026-03-26",
      "pipeline": "a",
      "confidence": 8.0,
      "description": "Initial high-confidence analysis across all chromosomes",
      "notes": "First full pipeline A run",
      "results": {
        "scoring_runs": ["20260225_143012"],
        "sae_runs": ["20260323_201618", ...],
        "analysis_runs": ["20260328_140000", ...]
      },
      "metrics": {
        "total_regions": 45000,
        "regions_per_gene": 8.5,
        "false_positive_rate": 0.05
      }
    }
  ]
}
```

**Benefits:**
- Track what changed between runs
- Easy to compare experiments
- Build audit trail
- Enable reproducibility

**Effort:** 3-4 hours

---

## Proposal 8: Create a Start-Here Guide

**Problem:**
- New users (or future you) don't know where to start
- README.md exists but might be overwhelming
- Multiple entry points to the project
- No clear path from "I just cloned this" to "I'm running analyses"

**Solution:**
Create `START_HERE.md`:

```markdown
# Start Here: Your First Analysis

## 1. Environment Setup (5 minutes)
- SSH to HPC: `ssh platawa@orcd-login001.mit.edu`
- Load conda: `module load miniforge/24.3.0-0`
- Activate env: `conda activate evo2_sep28`
- Navigate: `cd /orcd/data/zhang_f/001/platawa/jan31_files`

## 2. Choose Your Pipeline (10 minutes)
- Read: PIPELINE_DECISION_MATRIX.md
- Decide: conf≥8.0 (fast, high-quality) or conf≥0.0 (slow, comprehensive)

## 3. Check Current Status (5 minutes)
- Run: `python tools/check_pipeline_status.py`
- Review: Results in RESULTS_MANIFEST.json

## 4. Submit Your First Job (10 minutes)
- For Pipeline A: `sbatch extract_human_conf8.0_shards.sh`
- For Pipeline B: `sbatch extract_human_conf0.0_shards.sh`

## 5. Monitor Progress (5 minutes)
- Check queue: `squeue -u platawa`
- Check logs: `tail -f logs/extract_*.out`

## Next: Understanding the Data
- See: tools/README.md for tool descriptions
- See: COMPREHENSIVE_REPOSITORY_SUMMARY_2026-03-26.md for full inventory
```

**Benefits:**
- Onboard new users in 30 minutes
- Clear critical path from setup to analysis
- Reduces decision paralysis

**Effort:** 2-3 hours

---

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1)
1. **Proposal 8:** Create START_HERE.md (~2 hrs)
2. **Proposal 4:** Rename scripts (~4 hrs)
   - Time investment: 6 hours
   - Benefit: Immediately clearer project

### Phase 2: Foundation (Week 2-3)
1. **Proposal 1:** Create RESULTS_MANIFEST.json (~5 hrs)
2. **Proposal 3:** Create config.yaml (~3 hrs)
3. **Proposal 2:** Create tools/README.md (~3 hrs)
   - Time investment: 11 hours
   - Benefit: Unified documentation and configuration

### Phase 3: Advanced (Week 4+)
1. **Proposal 5:** Create pipeline state machine (~4 hrs)
2. **Proposal 7:** Create experiment tracking (~4 hrs)
3. **Proposal 6:** Archive old results (~2 hrs)
   - Time investment: 10 hours
   - Benefit: Robustness and experimentation tracking

**Total Effort:** ~27 hours over 4 weeks
**Benefit:** 30-40% reduction in cognitive overhead

---

## Priority Ranking

| Priority | Proposal | Effort | Benefit | Do This If... |
|----------|----------|--------|---------|--------------|
| 🔴 NOW | #8 START_HERE.md | 2h | HIGH | Anyone else will use this repo |
| 🔴 NOW | #4 Script Naming | 4h | HIGH | You want clarity |
| 🟡 SOON | #1 Results Manifest | 5h | MEDIUM | You have time this week |
| 🟡 SOON | #3 Config.yaml | 3h | MEDIUM | You're tired of editing paths |
| 🟡 SOON | #2 Tools README | 3h | MEDIUM | Others will run tools |
| 🟢 LATER | #5 State Machine | 4h | LOW | You worry about submission order |
| 🟢 LATER | #7 Experiment Tracking | 4h | LOW | You want to compare runs |
| 🟢 LATER | #6 Archive Results | 2h | LOW | Directory gets unwieldy |

---

## Comparison to Architectural Review Recommendations

This supplement covers **supplementary simplification** beyond the architectural review's 15 recommendations.

**Architectural Review focuses on:**
- Fixing critical issues (logs, job naming)
- Consolidating duplicates
- Improving organization

**This supplement focuses on:**
- Improving user experience
- Reducing cognitive load
- Enabling discovery and tracking
- Onboarding new users

**Together they create:**
- A cleaner, better-organized repository (from review)
- A simpler, more usable repository (from this supplement)

---

## Estimated Impact on Repository Complexity

```
Before any changes:        ████████████████ (100%)
After Architectural Fix:   ███████████░░░░░ (68%)
After This Supplement:     ████████░░░░░░░░ (50%)

Cognitive load reduced by:
- Architecture fixes: -32%
- Simplification: -18%
- Combined: -50%
```

---

## Next Steps

1. **Review this document** with your advisors or collaborators
2. **Choose Phase 1 items** from the implementation roadmap
3. **Assign effort** (these are estimated; actual may vary)
4. **Track progress** using one of the existing task systems
5. **Iterate** based on what's most useful in practice

---

**Questions?** See SUMMARY_FILES_INDEX.md for navigation between related documents.

**Version:** 1.0
**Last Updated:** 2026-03-26
