#!/usr/bin/env python3
"""
attribution_analysis.py — Per-nucleotide attribution for SAE features

Computes which nucleotide positions drive SAE feature activations using:
1. Integrated Gradients (preferred, if backprop through SSM works)
2. In-silico mutagenesis (ISM) fallback (guaranteed to work)

Optionally runs tf-modisco to discover DNA motifs from attribution maps.

Usage:
    # ISM attribution (always works)
    python tools/attribution_analysis.py \
        --fasta /path/to/genome.fna \
        --chrom NC_000022.11 \
        --start 20000000 --end 20001000 \
        --feature 15680 \
        --method ism

    # Integrated gradients (faster if backprop works)
    python tools/attribution_analysis.py \
        --fasta /path/to/genome.fna \
        --chrom NC_000022.11 \
        --start 20000000 --end 20001000 \
        --feature 15680 \
        --method ig --ig_steps 50

    # With tf-modisco motif discovery
    python tools/attribution_analysis.py \
        --fasta /path/to/genome.fna \
        --chrom NC_000022.11 \
        --boundaries results/.../drop_boundaries.tsv \
        --feature 15680 \
        --method ism --run_modisco
"""

import os
import sys
import json
import argparse
import logging
import time
from typing import List, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, write_completed, write_source

NUCLEOTIDES = ['A', 'C', 'G', 'T']
NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("attribution")
    logger.setLevel(getattr(logging, log_level.upper()))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def seq_to_onehot(seq: str) -> np.ndarray:
    """Convert DNA sequence to one-hot encoding (L, 4) for ACGT."""
    onehot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nuc in enumerate(seq.upper()):
        if nuc in NUC_TO_IDX:
            onehot[i, NUC_TO_IDX[nuc]] = 1.0
    return onehot


def compute_ism_attribution(
    seq: str,
    model,
    sae,
    feature_id: int,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Compute per-nucleotide attribution via in-silico mutagenesis.

    For each position, mutate to each alternative nucleotide and measure
    the change in mean feature activation. Attribution = max decrease.

    Args:
        seq: DNA sequence
        model: ObservableEvo2 instance
        sae: BatchTopKTiedSAE instance
        feature_id: Target SAE feature

    Returns:
        Attribution scores, shape (seq_len, 4) — one score per nucleotide per position
    """
    from sae_utils import get_feature_ts

    seq_len = len(seq)

    # Baseline activation
    if logger:
        logger.info(f"Computing baseline activation for {seq_len} bp...")
    baseline_features = get_feature_ts(model, sae, seq)
    baseline_act = baseline_features[:, feature_id].mean()

    if logger:
        logger.info(f"Baseline mean activation: {baseline_act:.4f}")

    # ISM: mutate each position
    attribution = np.zeros((seq_len, 4), dtype=np.float32)
    seq_list = list(seq.upper())

    t0 = time.time()
    for pos in range(seq_len):
        orig_nuc = seq_list[pos]
        orig_idx = NUC_TO_IDX.get(orig_nuc, -1)

        for nuc_idx, nuc in enumerate(NUCLEOTIDES):
            if nuc == orig_nuc:
                attribution[pos, nuc_idx] = 0.0
                continue

            # Mutate
            seq_list[pos] = nuc
            mut_seq = ''.join(seq_list)
            mut_features = get_feature_ts(model, sae, mut_seq)
            mut_act = mut_features[:, feature_id].mean()

            # Attribution = change in activation
            attribution[pos, nuc_idx] = mut_act - baseline_act

            # Restore
            seq_list[pos] = orig_nuc

        if logger and (pos + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (pos + 1) / elapsed
            eta = (seq_len - pos - 1) / rate if rate > 0 else 0
            logger.info(f"  Position {pos+1}/{seq_len} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    if logger:
        logger.info(f"ISM complete in {time.time() - t0:.1f}s")

    return attribution


def compute_ig_attribution(
    seq: str,
    model,
    sae,
    feature_id: int,
    n_steps: int = 50,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Compute per-nucleotide attribution via integrated gradients.

    Interpolates embeddings from zero baseline to actual, accumulates gradients.

    Args:
        seq: DNA sequence
        model: ObservableEvo2 instance
        sae: BatchTopKTiedSAE instance
        feature_id: Target SAE feature
        n_steps: Number of interpolation steps

    Returns:
        Attribution scores, shape (seq_len, d_embed) — gradient * (input - baseline)
    """
    import torch
    from sae_utils import SAE_LAYER_NAME

    toks = model.tokenizer.tokenize(seq)
    toks_tensor = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(model.device)

    if logger:
        logger.info(f"Running integrated gradients with {n_steps} steps...")

    # Get the embedding layer and compute baseline/actual embeddings
    raw_model = model.model
    emb_layer = None
    for name, mod in raw_model.named_modules():
        if 'embed' in name.lower() and hasattr(mod, 'weight'):
            emb_layer = mod
            break

    if emb_layer is None:
        raise RuntimeError("Could not find embedding layer for IG. Use --method ism instead.")

    with torch.no_grad():
        actual_emb = emb_layer(toks_tensor).detach()  # (1, seq_len, d_embed)

    baseline_emb = torch.zeros_like(actual_emb)

    # Accumulate gradients along interpolation path
    accumulated_grads = torch.zeros_like(actual_emb)

    for step in range(n_steps):
        alpha = (step + 0.5) / n_steps  # midpoint rule
        interp_emb = baseline_emb + alpha * (actual_emb - baseline_emb)
        interp_emb = interp_emb.clone().detach().requires_grad_(True)

        # Hook to replace embedding output
        def emb_hook(module, input, output, replacement=interp_emb):
            return replacement

        # Register hook
        handle = emb_layer.register_forward_hook(emb_hook)

        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, acts = model.forward(
                    toks_tensor,
                    cache_activations_at=[SAE_LAYER_NAME],
                )
                sae_device = next(iter(sae.parameters())).device
                features = sae.encode(acts[SAE_LAYER_NAME][0].to(sae_device))

                # Target: mean activation of feature
                target = features[:, feature_id].mean()

            # Backward
            target.backward()

            if interp_emb.grad is not None:
                accumulated_grads += interp_emb.grad.detach()
        except RuntimeError as e:
            if "does not support" in str(e) or "backward" in str(e).lower():
                if logger:
                    logger.warning(f"Backprop failed at step {step}: {e}")
                    logger.warning("SSM layers may not support gradients. Use --method ism.")
                handle.remove()
                raise
            raise
        finally:
            handle.remove()

        if logger and (step + 1) % 10 == 0:
            logger.info(f"  IG step {step+1}/{n_steps}")

    # Integrated gradients = (input - baseline) * mean_gradient
    ig = (actual_emb - baseline_emb) * accumulated_grads / n_steps

    if logger:
        logger.info("Integrated gradients complete")

    return ig[0].cpu().detach().float().numpy()  # (seq_len, d_embed)


def ism_to_effect_scores(attribution: np.ndarray, seq: str) -> np.ndarray:
    """Convert ISM attribution (seq_len, 4) to per-position effect scores.

    Returns the maximum absolute effect at each position (the worst-case mutation).
    """
    effect = np.zeros(len(seq), dtype=np.float32)
    for i, nuc in enumerate(seq.upper()):
        nuc_idx = NUC_TO_IDX.get(nuc, -1)
        # Max absolute change from any mutation at this position
        all_changes = attribution[i, :]
        # Exclude the wildtype (which is 0)
        if nuc_idx >= 0:
            all_changes[nuc_idx] = 0
        effect[i] = np.max(np.abs(all_changes))
    return effect


def run_modisco(
    attribution_scores: np.ndarray,
    onehot_seqs: np.ndarray,
    output_dir: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Run tf-modisco on attribution maps to discover motifs.

    Args:
        attribution_scores: (n_seqs, seq_len, 4) hypothetical importance scores
        onehot_seqs: (n_seqs, seq_len, 4) one-hot encoded sequences
        output_dir: Directory for modisco output

    Returns:
        Path to modisco results HDF5 file, or None if modisco not available
    """
    try:
        import modisco
    except ImportError:
        if logger:
            logger.warning("modisco-lite not installed. Skipping motif discovery. "
                          "Install with: pip install modisco-lite")
        return None

    if logger:
        logger.info("Running tf-modisco motif discovery...")

    t0 = time.time()

    # modisco expects (n_seqs, seq_len, 4) for both
    pos_patterns, neg_patterns = modisco.tfmodisco.TFMoDISco(
        hypothetical_contribs=attribution_scores,
        one_hot=onehot_seqs,
        max_seqlets_per_metacluster=2000,
    )

    output_path = os.path.join(output_dir, "modisco_results.h5")

    if logger:
        n_pos = len(pos_patterns) if pos_patterns else 0
        n_neg = len(neg_patterns) if neg_patterns else 0
        logger.info(f"tf-modisco found {n_pos} positive and {n_neg} negative patterns "
                     f"in {time.time() - t0:.1f}s")

    return output_path


def plot_attribution(
    effect_scores: np.ndarray,
    seq: str,
    feature_id: int,
    output_path: str,
    title_suffix: str = "",
    highlight_top_n: int = 20,
):
    """Plot per-position attribution scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4))

    x = np.arange(len(effect_scores))
    ax.fill_between(x, effect_scores, alpha=0.6, color='#e74c3c', linewidth=0)

    # Highlight top positions
    if highlight_top_n > 0:
        top_idx = np.argsort(effect_scores)[-highlight_top_n:]
        ax.scatter(top_idx, effect_scores[top_idx], color='#c0392b', s=20, zorder=5)

    ax.set_xlabel('Position (bp)')
    ax.set_ylabel('Attribution score')
    ax.set_title(f'Attribution for feature f/{feature_id}{title_suffix}')
    ax.set_xlim(0, len(effect_scores))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_sequence_logo(
    attribution: np.ndarray,
    seq: str,
    center: int,
    window: int,
    output_path: str,
    feature_id: int,
):
    """Plot a sequence-logo-style visualization of ISM attribution around a position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = max(0, center - window)
    end = min(len(seq), center + window)
    local_attr = attribution[start:end, :]
    local_seq = seq[start:end]

    fig, ax = plt.subplots(figsize=(min(20, (end - start) * 0.3), 3))

    colors = {'A': '#2ecc71', 'C': '#3498db', 'G': '#f39c12', 'T': '#e74c3c'}

    for pos in range(len(local_seq)):
        nuc = local_seq[pos].upper()
        # Show mutation effects as bars below
        for nuc_idx, alt_nuc in enumerate(NUCLEOTIDES):
            if alt_nuc == nuc:
                continue
            val = local_attr[pos, nuc_idx]
            if abs(val) > 0.01:
                ax.bar(pos, val, width=0.8, color=colors[alt_nuc], alpha=0.6)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Effect on feature activation')
    ax.set_title(f'ISM attribution for f/{feature_id} (pos {start}-{end})')

    # Add sequence annotation at bottom
    for pos in range(len(local_seq)):
        ax.text(pos, ax.get_ylim()[0], local_seq[pos].upper(),
                ha='center', va='top', fontsize=6, fontfamily='monospace',
                color=colors.get(local_seq[pos].upper(), 'black'))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-nucleotide attribution for SAE features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fasta", required=True, help="Path to genome FASTA")
    parser.add_argument("--chrom", required=True, help="Chromosome/accession")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--boundaries", default=None,
                        help="drop_boundaries.tsv to analyze specific regions")
    parser.add_argument("--max_regions", type=int, default=5,
                        help="Max regions to analyze from boundaries (default: 5)")
    parser.add_argument("--window", type=int, default=500,
                        help="Window around each region for ISM (default: 500)")

    parser.add_argument("--feature", type=int, required=True, help="Target SAE feature ID")
    parser.add_argument("--method", choices=["ism", "ig"], default="ism",
                        help="Attribution method (default: ism)")
    parser.add_argument("--ig_steps", type=int, default=50,
                        help="Number of IG interpolation steps (default: 50)")
    parser.add_argument("--run_modisco", action="store_true",
                        help="Run tf-modisco for motif discovery")

    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--chrom_name", default=None)
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.log_level)
    t_start = time.time()

    # --- Load sequence ---
    logger.info(f"Loading {args.chrom} from {args.fasta}")
    from score_chromosome import load_chromosome_sequence
    full_seq, actual_start, actual_end = load_chromosome_sequence(
        args.fasta, args.chrom, args.start, args.end, logger
    )
    logger.info(f"Loaded {len(full_seq):,} bp")

    # --- Determine regions to analyze ---
    regions_to_analyze = []
    if args.boundaries:
        from sae_utils import parse_chromosome_drops_tsv
        regions = parse_chromosome_drops_tsv(args.boundaries, max_regions=args.max_regions)
        for r in regions:
            center = (r['drop_start'] + r['drop_end']) // 2
            s = max(0, center - args.window)
            e = min(len(full_seq), center + args.window)
            regions_to_analyze.append({
                'start': s, 'end': e,
                'seq': full_seq[s:e],
                'label': f"drop_{r['drop_start']}_{r['drop_end']}",
            })
        logger.info(f"Analyzing {len(regions_to_analyze)} drop regions")
    else:
        # Analyze the full loaded sequence (should be short enough)
        if len(full_seq) > 2000:
            logger.warning(f"Sequence is {len(full_seq)} bp. ISM will be very slow. "
                          "Consider using --boundaries to target specific regions.")
        regions_to_analyze.append({
            'start': 0, 'end': len(full_seq),
            'seq': full_seq,
            'label': f"{args.chrom}_{actual_start}_{actual_end}",
        })

    # --- Initialize model and SAE ---
    logger.info("Initializing Evo2 model and SAE...")
    import torch
    from sae_utils import ObservableEvo2, load_topk_sae_from_hf

    model = ObservableEvo2("evo2_7b")
    sae = load_topk_sae_from_hf(model.d_hidden, model.device, model.dtype)
    logger.info("Model and SAE loaded")

    # --- Build output directory ---
    chrom_name = args.chrom_name or args.chrom.replace(".", "_")
    run_dir = build_run_dir(
        args.output_dir, chrom_name, "attribution",
        f"f{args.feature}_{args.method}"
    )
    data_dir = os.path.join(run_dir, "data")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    # --- Run attribution ---
    all_attributions = {}
    all_effect_scores = {}

    for region in regions_to_analyze:
        label = region['label']
        seq = region['seq']
        logger.info(f"Running {args.method} attribution on {label} ({len(seq)} bp)...")

        if args.method == "ism":
            attribution = compute_ism_attribution(seq, model, sae, args.feature, logger)
            effect_scores = ism_to_effect_scores(attribution, seq)
            all_attributions[label] = attribution
        elif args.method == "ig":
            try:
                ig_attr = compute_ig_attribution(
                    seq, model, sae, args.feature, n_steps=args.ig_steps, logger=logger
                )
                # Reduce to per-position scores by taking L2 norm
                effect_scores = np.linalg.norm(ig_attr, axis=-1)
                all_attributions[label] = ig_attr
            except RuntimeError:
                logger.warning("IG failed, falling back to ISM")
                attribution = compute_ism_attribution(seq, model, sae, args.feature, logger)
                effect_scores = ism_to_effect_scores(attribution, seq)
                all_attributions[label] = attribution

        all_effect_scores[label] = effect_scores

        # Save per-region data
        np.savez_compressed(
            os.path.join(data_dir, f"attribution_{label}.npz"),
            attribution=all_attributions[label],
            effect_scores=effect_scores,
            sequence=np.array(list(seq.upper())),
            start=region['start'],
            end=region['end'],
        )

        # Plot
        plot_attribution(
            effect_scores, seq, args.feature,
            os.path.join(plots_dir, f"attribution_{label}.png"),
            title_suffix=f" ({label})",
        )

        # Plot sequence logo around top position
        if args.method == "ism" and len(seq) > 20:
            top_pos = np.argmax(effect_scores)
            plot_sequence_logo(
                all_attributions[label], seq, top_pos, 30,
                os.path.join(plots_dir, f"logo_{label}_top.png"),
                args.feature,
            )

        logger.info(f"  Top attribution position: {np.argmax(effect_scores)} "
                     f"(score={effect_scores.max():.4f})")

    # --- Optional: tf-modisco ---
    if args.run_modisco and args.method == "ism":
        # Build arrays for modisco
        onehot_seqs = np.array([seq_to_onehot(r['seq']) for r in regions_to_analyze])
        # Use ISM attribution as hypothetical importance
        attr_arrays = np.array([all_attributions[r['label']] for r in regions_to_analyze])
        modisco_path = run_modisco(attr_arrays, onehot_seqs, data_dir, logger)

    # --- Save summary ---
    summary = {
        "feature_id": args.feature,
        "method": args.method,
        "n_regions": len(regions_to_analyze),
        "regions": [],
    }
    for region in regions_to_analyze:
        label = region['label']
        es = all_effect_scores[label]
        top_positions = np.argsort(es)[-10:][::-1].tolist()
        summary["regions"].append({
            "label": label,
            "start": region['start'],
            "end": region['end'],
            "length": len(region['seq']),
            "max_effect": float(es.max()),
            "mean_effect": float(es.mean()),
            "top_positions": top_positions,
            "top_scores": [float(es[p]) for p in top_positions],
        })

    with open(os.path.join(data_dir, "attribution_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    # --- Provenance ---
    write_source(run_dir, fasta=args.fasta, boundaries=args.boundaries)

    wall_time = time.time() - t_start
    write_completed(run_dir, "attribution_analysis.py", wall_time)
    logger.info(f"Done in {wall_time:.1f}s. Output: {run_dir}")


if __name__ == "__main__":
    main()
