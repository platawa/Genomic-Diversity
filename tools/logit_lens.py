#!/usr/bin/env python3
"""
logit_lens.py — Per-layer next-token entropy via logit lens

Projects hidden states at each Evo2 layer through the unembedding matrix to
compute per-layer next-token entropy. Reveals whether entropy drops come from
early layers (local sequence composition) or late layers (long-range context).

Usage:
    python tools/logit_lens.py \
        --fasta /path/to/genome.fna \
        --chrom NC_000022.11 \
        --boundaries results/chr22/scoring/.../data/drop_boundaries.tsv \
        --output_dir results

    # Specific layers
    python tools/logit_lens.py \
        --fasta /path/to/genome.fna \
        --chrom NC_000022.11 \
        --boundaries results/chr22/scoring/.../data/drop_boundaries.tsv \
        --layers 0 6 12 18 24 26 30
"""

import os
import sys
import json
import argparse
import logging
import time
from typing import List, Dict, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, write_completed, write_source

DEFAULT_LAYERS = [0, 6, 12, 18, 24, 26, 30]


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("logit_lens")
    logger.setLevel(getattr(logging, log_level.upper()))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def compute_layer_entropy(
    model,
    W_unembed,
    acgt_ids,
    sequence: str,
    layers: List[int],
    logger: Optional[logging.Logger] = None,
) -> Dict[int, np.ndarray]:
    """Compute per-position entropy at each layer via logit lens.

    For each layer, projects hidden states through the unembedding matrix,
    applies softmax over ACGT tokens, and computes entropy.

    Args:
        model: ObservableEvo2 instance
        W_unembed: Unembedding weight matrix (vocab_size, d_hidden)
        acgt_ids: Token IDs for A, C, G, T
        sequence: DNA sequence string
        layers: List of layer indices to probe

    Returns:
        Dict mapping layer_index -> np.ndarray of per-position entropy (nats)
    """
    import torch

    layer_names = [f'blocks-{i}' for i in layers]

    toks = model.tokenizer.tokenize(sequence)
    toks_tensor = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(model.device)

    if logger:
        logger.info(f"Running forward pass caching {len(layers)} layers...")

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, acts = model.forward(
                toks_tensor,
                cache_activations_at=layer_names,
            )

    # Compute entropy at each layer
    layer_entropy = {}
    for layer_idx, layer_name in zip(layers, layer_names):
        if layer_name not in acts:
            if logger:
                logger.warning(f"Layer {layer_name} not in cached activations, skipping")
            continue

        hidden = acts[layer_name]  # (1, seq_len, d_hidden)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        with torch.inference_mode():
            # Project through unembedding: (1, seq_len, vocab_size)
            layer_logits = hidden.float() @ W_unembed.T.float().to(hidden.device)

            # Softmax over ACGT tokens only
            acgt_logits = layer_logits[:, :, acgt_ids]  # (1, seq_len, 4)
            probs = torch.softmax(acgt_logits, dim=-1)

            # Entropy in nats
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (1, seq_len)

        layer_entropy[layer_idx] = entropy[0].cpu().numpy().astype(np.float32)

        if logger:
            logger.info(f"  Layer {layer_idx:2d}: mean entropy = {layer_entropy[layer_idx].mean():.4f}")

    # Also compute final-layer entropy from actual logits for validation
    with torch.inference_mode():
        final_acgt_logits = logits[:, :, acgt_ids].float()
        final_probs = torch.softmax(final_acgt_logits, dim=-1)
        final_entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum(dim=-1)
    layer_entropy[-1] = final_entropy[0].cpu().numpy().astype(np.float32)

    if logger:
        logger.info(f"  Final logits: mean entropy = {layer_entropy[-1].mean():.4f}")

    return layer_entropy


def analyze_drop_vs_nondrop(
    layer_entropy: Dict[int, np.ndarray],
    regions: List[Dict],
    genome_length: int,
) -> Dict[str, Dict[int, float]]:
    """Compare mean entropy at each layer for drop vs non-drop regions.

    Returns:
        Dict with 'drop_mean' and 'nondrop_mean', each mapping layer -> mean entropy
    """
    # Build a boolean mask for drop regions
    drop_mask = np.zeros(genome_length, dtype=bool)
    for r in regions:
        s = r['drop_start']
        e = min(r['drop_end'], genome_length)
        drop_mask[s:e] = True

    result = {'drop_mean': {}, 'nondrop_mean': {}, 'drop_std': {}, 'nondrop_std': {}}
    for layer_idx, ent in layer_entropy.items():
        length = min(len(ent), genome_length)
        mask = drop_mask[:length]
        ent_trimmed = ent[:length]

        if mask.any():
            result['drop_mean'][layer_idx] = float(np.mean(ent_trimmed[mask]))
            result['drop_std'][layer_idx] = float(np.std(ent_trimmed[mask]))
        if (~mask).any():
            result['nondrop_mean'][layer_idx] = float(np.mean(ent_trimmed[~mask]))
            result['nondrop_std'][layer_idx] = float(np.std(ent_trimmed[~mask]))

    return result


def plot_layer_entropy_comparison(
    comparison: Dict[str, Dict[int, float]],
    output_path: str,
    chrom: str,
):
    """Plot entropy vs layer for drop and non-drop regions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers_sorted = sorted(k for k in comparison['drop_mean'].keys() if k >= 0)

    drop_means = [comparison['drop_mean'].get(l, np.nan) for l in layers_sorted]
    nondrop_means = [comparison['nondrop_mean'].get(l, np.nan) for l in layers_sorted]
    drop_stds = [comparison['drop_std'].get(l, np.nan) for l in layers_sorted]
    nondrop_stds = [comparison['nondrop_std'].get(l, np.nan) for l in layers_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(layers_sorted, drop_means, yerr=drop_stds,
                marker='o', label='Drop regions', color='#e74c3c', capsize=3)
    ax.errorbar(layers_sorted, nondrop_means, yerr=nondrop_stds,
                marker='s', label='Non-drop regions', color='#3498db', capsize=3)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean entropy (nats)', fontsize=12)
    ax.set_title(f'Logit Lens: Per-layer entropy — {chrom}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_layer_entropy_heatmap(
    layer_entropy: Dict[int, np.ndarray],
    output_path: str,
    chrom: str,
    regions: Optional[List[Dict]] = None,
    window: int = 5000,
):
    """Plot heatmap of entropy across layers for a sample region."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers_sorted = sorted(k for k in layer_entropy.keys() if k >= 0)
    seq_len = min(len(v) for v in layer_entropy.values())

    # Pick a region to show (first drop region, or start of sequence)
    if regions and len(regions) > 0:
        center = regions[0]['drop_start']
        start = max(0, center - window // 2)
        end = min(seq_len, start + window)
    else:
        start, end = 0, min(window, seq_len)

    # Build matrix (layers x positions)
    matrix = np.array([layer_entropy[l][start:end] for l in layers_sorted])

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis',
                   extent=[start, end, len(layers_sorted) - 0.5, -0.5])
    ax.set_yticks(range(len(layers_sorted)))
    ax.set_yticklabels([str(l) for l in layers_sorted])
    ax.set_ylabel('Layer')
    ax.set_xlabel('Genomic position (bp)')
    ax.set_title(f'Logit Lens Heatmap — {chrom} ({start:,}-{end:,})')
    plt.colorbar(im, ax=ax, label='Entropy (nats)')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Logit lens: per-layer entropy analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fasta", required=True, help="Path to genome FASTA")
    parser.add_argument("--chrom", required=True, help="Chromosome/accession to analyze")
    parser.add_argument("--boundaries", default=None,
                        help="Path to drop_boundaries.tsv for drop/non-drop comparison")
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                        help=f"Layer indices to probe (default: {DEFAULT_LAYERS})")
    parser.add_argument("--start", type=int, default=None, help="Start position (bp)")
    parser.add_argument("--end", type=int, default=None, help="End position (bp)")
    parser.add_argument("--max_length", type=int, default=8192,
                        help="Max sequence length per chunk (default: 8192)")
    parser.add_argument("--output_dir", default="results", help="Base output directory")
    parser.add_argument("--chrom_name", default=None, help="Friendly chromosome name")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.log_level)
    t_start = time.time()

    # --- Load sequence ---
    logger.info(f"Loading {args.chrom} from {args.fasta}")
    from score_chromosome import load_chromosome_sequence
    sequence, actual_start, actual_end = load_chromosome_sequence(
        args.fasta, args.chrom, args.start, args.end, logger
    )
    logger.info(f"Loaded {len(sequence):,} bp")

    # Truncate if needed for memory
    if len(sequence) > args.max_length:
        logger.info(f"Truncating to {args.max_length} bp for logit lens analysis")
        sequence = sequence[:args.max_length]

    # --- Load drop regions if provided ---
    regions = []
    if args.boundaries:
        from sae_utils import parse_chromosome_drops_tsv
        regions = parse_chromosome_drops_tsv(args.boundaries)
        logger.info(f"Loaded {len(regions)} drop regions from boundaries file")

    # --- Initialize model ---
    logger.info("Initializing Evo2 model...")
    import torch
    from sae_utils import ObservableEvo2, get_unembedding_matrix, get_acgt_token_ids

    model = ObservableEvo2("evo2_7b")
    W_unembed = get_unembedding_matrix(model)
    acgt_ids = get_acgt_token_ids(model)
    logger.info("Model loaded")

    # --- Build output directory ---
    chrom_name = args.chrom_name or args.chrom.replace(".", "_")
    layer_str = f"layers_{'_'.join(str(l) for l in sorted(args.layers))}"
    run_dir = build_run_dir(args.output_dir, chrom_name, "logit_lens", layer_str)
    data_dir = os.path.join(run_dir, "data")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    # --- Compute layer entropy ---
    layer_entropy = compute_layer_entropy(
        model, W_unembed, acgt_ids, sequence, args.layers, logger
    )

    # --- Save data ---
    npz_data = {f"layer_{k}": v for k, v in layer_entropy.items()}
    npz_data['layers'] = np.array(sorted(layer_entropy.keys()))
    np.savez_compressed(os.path.join(data_dir, "layer_entropy.npz"), **npz_data)
    logger.info("Saved layer_entropy.npz")

    # --- Validation: compare final layer to logit output ---
    if -1 in layer_entropy and max(args.layers) in layer_entropy:
        final_from_logits = layer_entropy[-1]
        final_from_lens = layer_entropy[max(args.layers)]
        diff = np.abs(final_from_logits - final_from_lens).mean()
        logger.info(f"Validation: mean |logit_entropy - layer_{max(args.layers)}_entropy| = {diff:.6f}")

    # --- Drop vs non-drop comparison ---
    if regions:
        comparison = analyze_drop_vs_nondrop(layer_entropy, regions, len(sequence))

        with open(os.path.join(data_dir, "drop_vs_nondrop.json"), "w") as f:
            # Convert int keys to str for JSON
            json_safe = {}
            for k, v in comparison.items():
                json_safe[k] = {str(layer): val for layer, val in v.items()}
            json.dump(json_safe, f, indent=2)
            f.write("\n")
        logger.info("Saved drop_vs_nondrop.json")

        # Plot comparison
        plot_layer_entropy_comparison(
            comparison,
            os.path.join(plots_dir, "layer_entropy_comparison.png"),
            chrom_name,
        )
        logger.info("Saved layer_entropy_comparison.png")

    # Plot heatmap
    plot_layer_entropy_heatmap(
        layer_entropy,
        os.path.join(plots_dir, "layer_entropy_heatmap.png"),
        chrom_name,
        regions=regions,
    )
    logger.info("Saved layer_entropy_heatmap.png")

    # --- Provenance ---
    write_source(run_dir, fasta=args.fasta, boundaries=args.boundaries)

    wall_time = time.time() - t_start
    write_completed(run_dir, "logit_lens.py", wall_time)
    logger.info(f"Done in {wall_time:.1f}s. Output: {run_dir}")


if __name__ == "__main__":
    main()
