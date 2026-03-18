# Interpretability & Feature Steering Research Reference

## Context
This document maps research methods to the existing Evo2 SAE pipeline in this repo,
identifying what we already have, what's low-hanging fruit, and what would be novel contributions.

---

## Part 1: Interpretability Methods Beyond SAEs

### 1. Integrated Gradients + tf-modisco — HIGH PRIORITY
**What:** Per-position attribution scores showing which nucleotides drive Evo2's predictions.
tf-modisco then clusters these attributions across sequences to discover motifs de novo.

**Why it matters for us:** Our SAE features tell us *what* features activate in a region,
but not *which specific nucleotides* drive those features. Attribution maps fill that gap.

**How to apply:**
- Run integrated gradients on Evo2 for sequences at entropy drop regions
- Feed attributions into tf-modisco to discover motifs
- Cross-reference discovered motifs with SAE feature activations
- Answer: "Does SAE feature #4821 correspond to a specific DNA motif?"

**Integration point:** `sae_utils.py` already has `ObservableEvo2` with hook management.
Add a `compute_attributions()` method that does integrated gradients through the model.
Compare motifs found by tf-modisco against the signature features from `discover_region_features.py`.

**Packages:** `captum` (PyTorch attribution library), `modisco-lite` (tf-modisco reimplementation)

---

### 2. Linear Probes at Each Layer — MEDIUM PRIORITY
**What:** Train simple linear classifiers on Evo2 hidden states to predict known genomic
annotations (exon/intron, CpG island, repeat element, promoter, etc.)

**Why it matters for us:** Shows which layers encode which biological concepts, and
validates that the layer we run SAEs on (Layer 26) is the right choice.

**How to apply:**
- Extract Evo2 activations at layers 0, 6, 12, 18, 24, 26, 30 (subset)
- For each layer, train a linear probe: hidden_state → {exon, intron, intergenic, UTR, ...}
- Plot probe accuracy vs layer → identifies where biological knowledge is concentrated
- If layer 26 is not the peak, consider running SAEs on additional layers

**Integration point:** `ObservableEvo2` already caches activations at any layer.
Use GTF annotations (already loaded in `scan_feature_genome.py`) as labels.

---

### 3. Logit Lens — LOW EFFORT, HIGH INSIGHT
**What:** Project intermediate-layer hidden states into Evo2's 4-nucleotide vocabulary
to see what the model "believes" at each layer.

**Why it matters for us:** At entropy drop regions, the model is very confident.
Logit lens shows *at which layer* that confidence emerges — early layers = sequence
composition, late layers = long-range context.

**How to apply:**
- At each layer, multiply hidden state by the output unembedding matrix
- Compute entropy of the resulting 4-way distribution
- Plot entropy-by-layer for drop regions vs non-drop regions
- Shows whether drops are driven by local sequence features or long-range dependencies

**Integration point:** Add to `score_chromosome.py` or create a new `tools/logit_lens.py`.
Minimal code — just matrix multiply + softmax at each layer.

---

### 4. Sparse Probing — MEDIUM PRIORITY
**What:** Linear probes with L1 regularization to find *individual neurons* that encode
biological concepts (vs SAE's dictionary learning approach).

**Why it matters:** Validates SAE features — if sparse probing finds the same neurons
that SAE features decompose into, it increases confidence in both methods.

**Integration point:** Quick to implement on top of the linear probe infrastructure.

---

### 5. CCA/CKA Representation Similarity — LOW PRIORITY BUT INTERESTING
**What:** Compare Evo2 representations across organisms or genomic contexts.

**Possible experiments:**
- Compare human chr22 representations vs E. coli vs Bacillus → what's universal?
- Compare high-entropy vs low-entropy region representations
- Compare representations across Evo2 layers

---

## Part 2: DNA Feature Steering Methods

### 1. SAE Feature Clamping During Generation — HIGHEST PRIORITY, NOVEL
**What:** During Evo2 autoregressive generation, clamp specific SAE feature activations
to high values to steer the generated DNA toward sequences with that property.

**Based on:** Anthropic's "Golden Gate Bridge" experiment (Templeton et al., 2024) —
clamping an SAE feature during generation caused Claude to talk about that topic.

**How to apply with our pipeline:**
```
1. Identify SAE feature of interest (e.g., feature #X = "promoter-like")
2. During Evo2 autoregressive generation:
   a. Run forward pass for next token
   b. At layer 26, extract SAE encoding: z = SAE.encode(hidden_state)
   c. Clamp z[feature_X] = high_value (e.g., 10x normal activation)
   d. Reconstruct: hidden_state_steered = SAE.decode(z_clamped)
   e. Replace hidden state, continue forward pass
   f. Sample next token from steered logits
3. Generate full sequence, analyze properties
```

**Integration point:** `sae_utils.py` already has `ObservableEvo2` with intervention hooks
and `BatchTopKTiedSAE` with encode/decode. The infrastructure exists — need to add
a `steered_generate()` function that combines them.

**Validation:** Generate sequences with/without steering, compare:
- SAE feature activation profiles
- Sequence composition (GC content, codon usage)
- Alignment to known functional elements (BLAST)
- Predicted function (run through other genomic models)

---

### 2. Feature-Guided MCMC Search — HIGH PRIORITY
**What:** Start from a natural DNA sequence, propose mutations, accept/reject based on
SAE feature activation (plus Evo2 likelihood to maintain sequence plausibility).

**How to apply:**
```
objective(seq) = α * SAE_feature_activation(seq, feature_X) + β * Evo2_loglikelihood(seq)
```
- Propose: single nucleotide mutations (or small indels)
- Accept/reject: Metropolis-Hastings with objective as energy function
- Result: natural-like sequences optimized for a specific SAE feature

**Why better than pure generation:** Stays close to natural sequence space, easier to
validate, can start from a known functional element and optimize.

**Integration point:** `score_chromosome.py` already computes per-position entropy/logprobs.
Need a new `tools/feature_guided_mcmc.py`.

---

### 3. Gradient-Based Feature Optimization — MEDIUM PRIORITY
**What:** Use Gumbel-softmax relaxation to backpropagate through Evo2 and optimize
a DNA sequence to maximize a target SAE feature.

**How to apply:**
- Represent sequence as continuous logits over {A,C,G,T} at each position
- Forward pass through Evo2 with Gumbel-softmax sampling
- Compute SAE feature activation at layer 26
- Backprop, update logits to maximize target feature
- Discretize final logits to get DNA sequence

**Pros:** Much faster than MCMC (gradient-based vs random search)
**Cons:** May produce adversarial sequences that activate the feature but aren't biologically meaningful

---

### 4. Feature-Based Genome Search/Retrieval — ALREADY PARTIALLY IMPLEMENTED
**What we have:** `scan_feature_genome.py` scans genome for activation of specific features.
`discover_region_features.py` finds enriched features in target regions.

**Extension:** Build a feature activation index over the full genome. Given a query
feature profile (e.g., "high feature #100, low feature #200"), retrieve all matching
genomic regions. Essentially a "search engine" over the genome indexed by SAE features.

---

### 5. Multi-Feature Steering — FUTURE WORK
**What:** Simultaneously clamp multiple SAE features to specify complex biological properties.
E.g., "promoter" + "high GC" + "E. coli-like" to generate a GC-rich E. coli promoter.

**Challenge:** Feature interactions — clamping multiple features may produce interference.
Need to understand feature correlations first (partially addressed by our clustering in
`analyze_sae_regions.py`).

---

## Part 3: Recommended Implementation Order

### Phase 1 — Quick Wins (days)
1. **Logit lens** — minimal code, reveals layer-by-layer prediction formation
2. **Feature-based genome retrieval** — extend `scan_feature_genome.py` with multi-feature queries

### Phase 2 — Core Steering (1-2 weeks)
3. **SAE feature clamping during generation** — the main novel contribution
4. **Feature-guided MCMC** — complementary search approach

### Phase 3 — Validation & Analysis (1-2 weeks)
5. **Integrated gradients + tf-modisco** — motif discovery to validate SAE features
6. **Linear probes** — validate layer 26 choice, characterize what each layer encodes

### Phase 4 — Advanced (longer term)
7. **Gradient-based optimization** — faster than MCMC but needs careful validation
8. **Multi-feature steering** — complex biological specifications
9. **Cross-organism representation analysis** — CCA/CKA comparisons

---

## Key References

- Templeton et al. (2024) — "Scaling Monosemanticity" — SAE feature steering on Claude
- Zou et al. (2023) — Representation Engineering — steering vectors
- Nguyen et al. (2024) — Evo — DNA foundation model, generated functional CRISPR systems
- Nisonoff et al. (2024) — Guided discrete diffusion for regulatory DNA design
- Madani et al. (2023) — ProGen — controllable protein generation
- Sundararajan et al. (2017) — Integrated gradients
- Shrikumar et al. (2018) — tf-modisco for motif discovery from attributions
- Dalla-Torre et al. (2023) — Nucleotide Transformer — linear probes on DNA FM
- Li et al. (2023) — Inference-time intervention
- Marks et al. (2024) — Sparse feature circuits
- Kornblith et al. (2019) — CKA for representation similarity
