#!/usr/bin/env python3
"""
steered_generation.py — SAE Feature Clamping During Evo2 Generation

During autoregressive generation, intercepts layer 26, encodes through SAE,
clamps target feature(s) to a high value, decodes back. Steers generated DNA
toward sequences with that biological property.

Based on Anthropic's "Golden Gate Bridge" activation steering experiment.

Usage:
    # Steer toward CDS (feature 15680)
    python tools/steered_generation.py \
        --prompt ACGTACGT... \
        --feature 15680 \
        --clamp_value 5.0 \
        --n_tokens 1000

    # Multi-feature steering
    python tools/steered_generation.py \
        --prompt ACGTACGT... \
        --feature 15680 28339 \
        --clamp_value 5.0 2.0 \
        --n_tokens 1000

    # From FASTA (use region as prompt)
    python tools/steered_generation.py \
        --fasta /path/to/genome.fna \
        --chrom NC_000022.11 \
        --start 20000000 --end 20001000 \
        --feature 15680 --clamp_value 5.0
"""

import os
import sys
import json
import argparse
import logging
import time
from collections import Counter
from typing import List, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, write_completed, write_source


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("steered_gen")
    logger.setLevel(getattr(logging, log_level.upper()))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def build_sae_clamp_intervention(sae, feature_ids: List[int], clamp_values: List[float]):
    """Build an intervention closure that clamps SAE features during generation.

    Args:
        sae: BatchTopKTiedSAE instance
        feature_ids: List of feature indices to clamp
        clamp_values: Corresponding clamp values for each feature

    Returns:
        Closure compatible with ObservableEvo2 intervention interface
    """
    import torch

    def intervention(hidden_state):
        # hidden_state: (batch, seq_len, d_hidden)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sae_device = next(iter(sae.parameters())).device
            h = hidden_state.to(sae_device)

            # Encode through SAE
            features = sae.encode(h)

            # Clamp target features
            for fid, cval in zip(feature_ids, clamp_values):
                features[:, :, fid] = cval

            # Decode back to hidden space
            reconstructed = sae.decode(features)

        return reconstructed.to(hidden_state.device)

    return intervention


def steered_generate(
    model,
    sae,
    prompt_seq: str,
    feature_ids: List[int],
    clamp_values: List[float],
    n_tokens: int = 1000,
    temperature: float = 1.0,
    top_k: int = 4,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, Dict]:
    """Generate DNA sequence with SAE feature clamping.

    Args:
        model: ObservableEvo2 instance
        sae: BatchTopKTiedSAE instance
        prompt_seq: DNA prompt sequence
        feature_ids: Features to clamp
        clamp_values: Values to clamp features to
        n_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        logger: Optional logger

    Returns:
        Tuple of (generated_sequence, metadata_dict)
    """
    from sae_utils import SAE_LAYER_NAME

    if logger:
        features_str = ", ".join(f"f/{fid}={cv}" for fid, cv in zip(feature_ids, clamp_values))
        logger.info(f"Generating {n_tokens} tokens with clamped features: {features_str}")

    clamp_fn = build_sae_clamp_intervention(sae, feature_ids, clamp_values)

    t0 = time.time()
    generated_seq, acts = model.generate(
        [prompt_seq],
        n_tokens=n_tokens,
        temperature=temperature,
        top_k=top_k,
        interventions={SAE_LAYER_NAME: clamp_fn},
    )
    gen_time = time.time() - t0

    if logger:
        logger.info(f"Generated {len(generated_seq)} bp in {gen_time:.1f}s "
                     f"({n_tokens / gen_time:.0f} tokens/s)")

    metadata = {
        "prompt_length": len(prompt_seq),
        "generated_length": len(generated_seq),
        "n_tokens": n_tokens,
        "generation_time_s": round(gen_time, 2),
        "temperature": temperature,
        "top_k": top_k,
        "features_clamped": {str(fid): cv for fid, cv in zip(feature_ids, clamp_values)},
    }

    return generated_seq, metadata


def compute_sequence_stats(seq: str) -> Dict:
    """Compute basic sequence statistics."""
    counts = Counter(seq.upper())
    total = len(seq)
    gc = (counts.get('G', 0) + counts.get('C', 0)) / total if total > 0 else 0

    # k-mer frequencies (dinucleotides)
    dimers = Counter(seq[i:i+2].upper() for i in range(len(seq) - 1))
    total_dimers = sum(dimers.values())
    dimer_freq = {k: v / total_dimers for k, v in sorted(dimers.items())} if total_dimers > 0 else {}

    return {
        "length": total,
        "gc_content": round(gc, 4),
        "base_counts": dict(counts),
        "dinucleotide_freq": {k: round(v, 4) for k, v in dimer_freq.items()},
    }


def evaluate_steering(
    model,
    sae,
    prompt_seq: str,
    steered_seq: str,
    unsteered_seq: str,
    feature_ids: List[int],
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Evaluate steered vs unsteered sequences.

    Compares feature activations, sequence stats, and log-likelihood.

    Returns:
        Dict with evaluation metrics
    """
    from sae_utils import get_feature_ts

    if logger:
        logger.info("Evaluating steered vs unsteered sequences...")

    # Compute feature activations for both
    # Use just the generated portion (after prompt)
    prompt_len = len(prompt_seq)
    steered_gen = steered_seq[prompt_len:]
    unsteered_gen = unsteered_seq[prompt_len:]

    eval_result = {
        "steered_stats": compute_sequence_stats(steered_gen),
        "unsteered_stats": compute_sequence_stats(unsteered_gen),
    }

    # Feature activations (on chunks that fit in memory)
    max_eval_len = 8192
    for label, gen_seq in [("steered", steered_gen), ("unsteered", unsteered_gen)]:
        chunk = gen_seq[:max_eval_len]
        if len(chunk) < 10:
            continue

        features = get_feature_ts(model, sae, chunk)
        for fid in feature_ids:
            feat_act = features[:, fid]
            eval_result[f"{label}_feature_{fid}"] = {
                "mean_activation": float(np.mean(feat_act)),
                "max_activation": float(np.max(feat_act)),
                "fraction_active": float(np.mean(feat_act > 0)),
                "percentile_95": float(np.percentile(feat_act, 95)),
            }

    return eval_result


def plot_steering_comparison(
    eval_result: Dict,
    feature_ids: List[int],
    output_path: str,
):
    """Plot comparison of steered vs unsteered feature activations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_features = len(feature_ids)
    fig, axes = plt.subplots(1, max(n_features, 1), figsize=(6 * n_features, 5))
    if n_features == 1:
        axes = [axes]

    for ax, fid in zip(axes, feature_ids):
        steered = eval_result.get(f"steered_feature_{fid}", {})
        unsteered = eval_result.get(f"unsteered_feature_{fid}", {})

        metrics = ['mean_activation', 'max_activation', 'fraction_active']
        labels = ['Mean Act.', 'Max Act.', 'Frac. Active']

        x = np.arange(len(metrics))
        width = 0.35
        s_vals = [steered.get(m, 0) for m in metrics]
        u_vals = [unsteered.get(m, 0) for m in metrics]

        ax.bar(x - width/2, s_vals, width, label='Steered', color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, u_vals, width, label='Unsteered', color='#3498db', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f'Feature f/{fid}')
        ax.legend()

    fig.suptitle('Steered vs Unsteered Generation', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAE feature clamping during Evo2 generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Prompt source
    parser.add_argument("--prompt", default=None, help="DNA prompt sequence string")
    parser.add_argument("--fasta", default=None, help="Path to genome FASTA")
    parser.add_argument("--chrom", default=None, help="Chromosome for FASTA extraction")
    parser.add_argument("--start", type=int, default=None, help="Start position in FASTA")
    parser.add_argument("--end", type=int, default=None, help="End position in FASTA")

    # Steering parameters
    parser.add_argument("--feature", type=int, nargs="+", required=True,
                        help="SAE feature IDs to clamp")
    parser.add_argument("--clamp_value", type=float, nargs="+", required=True,
                        help="Clamp values (one per feature, or single value for all)")

    # Generation parameters
    parser.add_argument("--n_tokens", type=int, default=1000, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=4, help="Top-k sampling")
    parser.add_argument("--n_unsteered", type=int, default=1,
                        help="Number of unsteered control sequences to generate")

    # Output
    parser.add_argument("--output_dir", default="results", help="Base output directory")
    parser.add_argument("--chrom_name", default=None, help="Friendly name for output dir")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.log_level)
    t_start = time.time()

    # --- Get prompt sequence ---
    if args.prompt:
        prompt_seq = args.prompt
    elif args.fasta and args.chrom:
        from score_chromosome import load_chromosome_sequence
        seq, _, _ = load_chromosome_sequence(
            args.fasta, args.chrom, args.start, args.end, logger
        )
        prompt_seq = seq
    else:
        parser_error = "Must provide --prompt or --fasta/--chrom"
        raise ValueError(parser_error)

    logger.info(f"Prompt sequence: {len(prompt_seq)} bp")

    # --- Parse feature/clamp pairs ---
    feature_ids = args.feature
    if len(args.clamp_value) == 1:
        clamp_values = args.clamp_value * len(feature_ids)
    elif len(args.clamp_value) == len(feature_ids):
        clamp_values = args.clamp_value
    else:
        raise ValueError("--clamp_value must be 1 value or same count as --feature")

    # --- Initialize model and SAE ---
    logger.info("Initializing Evo2 model and SAE...")
    from sae_utils import ObservableEvo2, load_topk_sae_from_hf
    model = ObservableEvo2("evo2_7b")
    sae = load_topk_sae_from_hf(model.d_hidden, model.device, model.dtype)
    logger.info("Model and SAE loaded")

    # --- Build output directory ---
    chrom_name = args.chrom_name or "steered"
    feat_str = "_".join(f"f{fid}" for fid in feature_ids)
    run_dir = build_run_dir(args.output_dir, chrom_name, "steered_generation", feat_str)
    data_dir = os.path.join(run_dir, "data")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    # --- Steered generation ---
    steered_seq, steered_meta = steered_generate(
        model, sae, prompt_seq, feature_ids, clamp_values,
        n_tokens=args.n_tokens, temperature=args.temperature,
        top_k=args.top_k, logger=logger,
    )

    # --- Unsteered control ---
    logger.info("Generating unsteered control sequence...")
    unsteered_seq, _ = model.generate(
        [prompt_seq],
        n_tokens=args.n_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    logger.info(f"Unsteered: {len(unsteered_seq)} bp")

    # --- Save sequences ---
    with open(os.path.join(data_dir, "steered_sequence.fasta"), "w") as f:
        f.write(f">steered_{'_'.join(f'f{fid}' for fid in feature_ids)}\n")
        # Wrap at 80 chars
        for i in range(0, len(steered_seq), 80):
            f.write(steered_seq[i:i+80] + "\n")

    with open(os.path.join(data_dir, "unsteered_sequence.fasta"), "w") as f:
        f.write(">unsteered_control\n")
        for i in range(0, len(unsteered_seq), 80):
            f.write(unsteered_seq[i:i+80] + "\n")

    with open(os.path.join(data_dir, "prompt_sequence.fasta"), "w") as f:
        f.write(">prompt\n")
        for i in range(0, len(prompt_seq), 80):
            f.write(prompt_seq[i:i+80] + "\n")

    # --- Evaluate ---
    eval_result = evaluate_steering(
        model, sae, prompt_seq, steered_seq, unsteered_seq,
        feature_ids, logger=logger,
    )

    # --- Save metadata ---
    run_metadata = {
        **steered_meta,
        "evaluation": eval_result,
    }
    with open(os.path.join(data_dir, "run_metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=2)
        f.write("\n")
    logger.info("Saved run_metadata.json")

    # --- Plot ---
    plot_steering_comparison(
        eval_result, feature_ids,
        os.path.join(plots_dir, "steering_comparison.png"),
    )
    logger.info("Saved steering_comparison.png")

    # --- Log summary ---
    for fid in feature_ids:
        s = eval_result.get(f"steered_feature_{fid}", {})
        u = eval_result.get(f"unsteered_feature_{fid}", {})
        logger.info(f"Feature f/{fid}: steered mean={s.get('mean_activation', 0):.3f}, "
                     f"unsteered mean={u.get('mean_activation', 0):.3f}")

    # --- Provenance ---
    write_source(run_dir, fasta=args.fasta, prompt="inline" if args.prompt else args.fasta)

    wall_time = time.time() - t_start
    write_completed(run_dir, "steered_generation.py", wall_time)
    logger.info(f"Done in {wall_time:.1f}s. Output: {run_dir}")


if __name__ == "__main__":
    main()
