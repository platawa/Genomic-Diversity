#!/usr/bin/env python3
"""
feature_guided_mcmc.py — Feature-Guided MCMC Sequence Search

Starts from natural DNA, proposes single-nucleotide mutations, accepts/rejects
via Metropolis-Hastings with objective:
    objective = alpha * SAE_feature_activation + beta * Evo2_log_likelihood

Usage:
    python tools/feature_guided_mcmc.py \
        --fasta /path/to/genome.fna \
        --chrom NC_000022.11 \
        --start 20000000 --end 20008192 \
        --feature 15680 \
        --alpha 1.0 --beta 0.5 \
        --n_steps 500

    # From inline sequence
    python tools/feature_guided_mcmc.py \
        --sequence ACGTACGT... \
        --feature 15680 \
        --n_steps 1000
"""

import os
import sys
import json
import argparse
import logging
import time
import random
from typing import List, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, write_completed, write_source

NUCLEOTIDES = ['A', 'C', 'G', 'T']


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("mcmc")
    logger.setLevel(getattr(logging, log_level.upper()))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def mcmc_objective(
    seq: str,
    model,
    sae,
    acgt_ids,
    feature_id: int,
    alpha: float,
    beta: float,
) -> Tuple[float, float, float]:
    """Compute MCMC objective for a sequence.

    Args:
        seq: DNA sequence
        model: ObservableEvo2 instance
        sae: BatchTopKTiedSAE instance
        acgt_ids: ACGT token IDs tensor
        feature_id: Target SAE feature
        alpha: Weight for feature activation
        beta: Weight for log-likelihood

    Returns:
        Tuple of (objective, feature_activation, log_likelihood)
    """
    import torch
    from sae_utils import SAE_LAYER_NAME

    toks = model.tokenizer.tokenize(seq)
    toks_tensor = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, acts = model.forward(
                toks_tensor,
                cache_activations_at=[SAE_LAYER_NAME],
            )

            # Log-likelihood: average log prob of true next token
            # Shift logits and targets for next-token prediction
            shift_logits = logits[0, :-1, :]  # (seq_len-1, vocab)
            shift_targets = toks_tensor[0, 1:]  # (seq_len-1,)

            log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
            ll_per_pos = log_probs.gather(1, shift_targets.unsqueeze(1)).squeeze(1)
            mean_ll = ll_per_pos.mean().item()

            # Feature activation
            sae_device = next(iter(sae.parameters())).device
            layer_acts = acts[SAE_LAYER_NAME][0].to(sae_device)
            features = sae.encode(layer_acts)
            feat_act = features[:, feature_id].mean().item()

    objective = alpha * feat_act + beta * mean_ll
    return objective, feat_act, mean_ll


def propose_mutation(seq: str) -> Tuple[str, int, str, str]:
    """Propose a single-nucleotide mutation.

    Returns:
        Tuple of (mutated_seq, position, old_nuc, new_nuc)
    """
    seq_list = list(seq)
    pos = random.randint(0, len(seq_list) - 1)
    old_nuc = seq_list[pos]

    # Pick a different nucleotide
    alternatives = [n for n in NUCLEOTIDES if n != old_nuc.upper()]
    new_nuc = random.choice(alternatives)
    seq_list[pos] = new_nuc

    return ''.join(seq_list), pos, old_nuc, new_nuc


def mcmc_search(
    start_seq: str,
    model,
    sae,
    acgt_ids,
    feature_id: int,
    alpha: float = 1.0,
    beta: float = 0.5,
    n_steps: int = 500,
    temperature: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Run Metropolis-Hastings MCMC search.

    Args:
        start_seq: Starting DNA sequence
        model: ObservableEvo2 instance
        sae: BatchTopKTiedSAE instance
        acgt_ids: ACGT token IDs
        feature_id: Target feature to maximize
        alpha: Feature activation weight
        beta: Log-likelihood weight (keeps sequence "natural")
        n_steps: Number of MCMC steps
        temperature: MH temperature (higher = more exploration)
        logger: Optional logger

    Returns:
        Dict with trajectory and final sequence
    """
    # Compute initial objective
    if logger:
        logger.info(f"Computing initial objective (alpha={alpha}, beta={beta})...")

    current_seq = start_seq
    current_obj, current_feat, current_ll = mcmc_objective(
        current_seq, model, sae, acgt_ids, feature_id, alpha, beta
    )

    if logger:
        logger.info(f"Initial: obj={current_obj:.4f}, feat={current_feat:.4f}, ll={current_ll:.4f}")

    # Trajectory tracking
    trajectory = {
        'objective': [current_obj],
        'feature_activation': [current_feat],
        'log_likelihood': [current_ll],
        'accepted': [],
        'mutations': [],
    }

    best_seq = current_seq
    best_obj = current_obj
    n_accepted = 0

    t0 = time.time()
    for step in range(n_steps):
        # Propose mutation
        proposed_seq, pos, old_nuc, new_nuc = propose_mutation(current_seq)

        # Evaluate proposal
        prop_obj, prop_feat, prop_ll = mcmc_objective(
            proposed_seq, model, sae, acgt_ids, feature_id, alpha, beta
        )

        # Metropolis-Hastings acceptance
        delta = prop_obj - current_obj
        if delta > 0:
            accept = True
        else:
            accept_prob = np.exp(delta / temperature)
            accept = random.random() < accept_prob

        if accept:
            current_seq = proposed_seq
            current_obj = prop_obj
            current_feat = prop_feat
            current_ll = prop_ll
            n_accepted += 1

            if current_obj > best_obj:
                best_seq = current_seq
                best_obj = current_obj

        trajectory['objective'].append(current_obj)
        trajectory['feature_activation'].append(current_feat)
        trajectory['log_likelihood'].append(current_ll)
        trajectory['accepted'].append(accept)
        trajectory['mutations'].append({
            'step': step, 'pos': pos, 'old': old_nuc, 'new': new_nuc,
            'accepted': accept, 'delta': delta,
        })

        if logger and (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            logger.info(
                f"Step {step+1}/{n_steps}: obj={current_obj:.4f}, "
                f"feat={current_feat:.4f}, ll={current_ll:.4f}, "
                f"accepted={n_accepted}/{step+1} ({n_accepted/(step+1):.1%}), "
                f"{rate:.1f} steps/s"
            )

    elapsed = time.time() - t0

    result = {
        'trajectory': trajectory,
        'best_sequence': best_seq,
        'best_objective': best_obj,
        'final_sequence': current_seq,
        'final_objective': current_obj,
        'n_accepted': n_accepted,
        'acceptance_rate': n_accepted / n_steps if n_steps > 0 else 0,
        'n_steps': n_steps,
        'wall_time_s': round(elapsed, 2),
        'steps_per_second': round(n_steps / elapsed, 2) if elapsed > 0 else 0,
    }

    if logger:
        logger.info(f"MCMC complete: {n_steps} steps in {elapsed:.1f}s")
        logger.info(f"  Best objective: {best_obj:.4f}")
        logger.info(f"  Acceptance rate: {n_accepted}/{n_steps} ({result['acceptance_rate']:.1%})")

    return result


def plot_mcmc_trajectory(trajectory: Dict, feature_id: int, output_path: str):
    """Plot MCMC trajectory: objective, feature activation, and log-likelihood."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    steps = range(len(trajectory['objective']))

    axes[0].plot(steps, trajectory['objective'], color='#2c3e50', linewidth=0.8)
    axes[0].set_ylabel('Objective')
    axes[0].set_title(f'MCMC Trajectory — Feature f/{feature_id}')

    axes[1].plot(steps, trajectory['feature_activation'], color='#e74c3c', linewidth=0.8)
    axes[1].set_ylabel(f'Feature f/{feature_id}\nactivation')

    axes[2].plot(steps, trajectory['log_likelihood'], color='#3498db', linewidth=0.8)
    axes[2].set_ylabel('Log-likelihood')
    axes[2].set_xlabel('MCMC Step')

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature-guided MCMC sequence search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sequence", default=None, help="Starting DNA sequence")
    parser.add_argument("--fasta", default=None, help="Path to genome FASTA")
    parser.add_argument("--chrom", default=None, help="Chromosome for FASTA extraction")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)

    parser.add_argument("--feature", type=int, required=True, help="Target SAE feature ID")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight for feature activation (default: 1.0)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Weight for log-likelihood (default: 0.5)")
    parser.add_argument("--n_steps", type=int, default=500, help="Number of MCMC steps")
    parser.add_argument("--temperature", type=float, default=1.0, help="MH temperature")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--chrom_name", default=None)
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.log_level)
    t_start = time.time()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # --- Get starting sequence ---
    if args.sequence:
        start_seq = args.sequence
    elif args.fasta and args.chrom:
        from score_chromosome import load_chromosome_sequence
        start_seq, _, _ = load_chromosome_sequence(
            args.fasta, args.chrom, args.start, args.end, logger
        )
    else:
        raise ValueError("Must provide --sequence or --fasta/--chrom")

    logger.info(f"Starting sequence: {len(start_seq)} bp")

    # --- Initialize model and SAE ---
    logger.info("Initializing Evo2 model and SAE...")
    import torch
    from sae_utils import ObservableEvo2, load_topk_sae_from_hf, get_acgt_token_ids

    model = ObservableEvo2("evo2_7b")
    sae = load_topk_sae_from_hf(model.d_hidden, model.device, model.dtype)
    acgt_ids = get_acgt_token_ids(model)
    logger.info("Model and SAE loaded")

    # --- Build output directory ---
    chrom_name = args.chrom_name or "mcmc"
    run_dir = build_run_dir(
        args.output_dir, chrom_name, "mcmc_search",
        f"f{args.feature}_a{args.alpha}_b{args.beta}_{args.n_steps}steps"
    )
    data_dir = os.path.join(run_dir, "data")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    # --- Run MCMC ---
    result = mcmc_search(
        start_seq, model, sae, acgt_ids,
        feature_id=args.feature,
        alpha=args.alpha,
        beta=args.beta,
        n_steps=args.n_steps,
        temperature=args.temperature,
        logger=logger,
    )

    # --- Save outputs ---
    # Save best and final sequences
    for label, seq_key in [("best", "best_sequence"), ("final", "final_sequence")]:
        with open(os.path.join(data_dir, f"{label}_sequence.fasta"), "w") as f:
            f.write(f">{label}_f{args.feature}_mcmc\n")
            seq = result[seq_key]
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

    with open(os.path.join(data_dir, "start_sequence.fasta"), "w") as f:
        f.write(">start_sequence\n")
        for i in range(0, len(start_seq), 80):
            f.write(start_seq[i:i+80] + "\n")

    # Save trajectory (without full sequences to save space)
    trajectory_data = {
        'objective': result['trajectory']['objective'],
        'feature_activation': result['trajectory']['feature_activation'],
        'log_likelihood': result['trajectory']['log_likelihood'],
        'accepted': result['trajectory']['accepted'],
    }
    np.savez_compressed(
        os.path.join(data_dir, "trajectory.npz"),
        **{k: np.array(v) for k, v in trajectory_data.items()}
    )

    # Save metadata
    metadata = {k: v for k, v in result.items()
                if k not in ('trajectory', 'best_sequence', 'final_sequence')}
    metadata['feature_id'] = args.feature
    metadata['alpha'] = args.alpha
    metadata['beta'] = args.beta
    metadata['temperature'] = args.temperature
    metadata['start_seq_length'] = len(start_seq)
    with open(os.path.join(data_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    logger.info("Saved metadata and trajectory")

    # --- Plot ---
    plot_mcmc_trajectory(
        result['trajectory'], args.feature,
        os.path.join(plots_dir, "mcmc_trajectory.png"),
    )
    logger.info("Saved mcmc_trajectory.png")

    # --- Provenance ---
    write_source(run_dir, fasta=args.fasta, sequence="inline" if args.sequence else args.fasta)

    wall_time = time.time() - t_start
    write_completed(run_dir, "feature_guided_mcmc.py", wall_time)
    logger.info(f"Done in {wall_time:.1f}s. Output: {run_dir}")


if __name__ == "__main__":
    main()
