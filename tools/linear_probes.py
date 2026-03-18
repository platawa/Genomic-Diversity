#!/usr/bin/env python3
"""
linear_probes.py — Linear Probes for Evo2 Hidden State Annotation Prediction

Two-phase workflow:
  Phase A (GPU cluster): Extract activations at multiple layers for subsampled positions
  Phase B (local):       Train sklearn logistic regression per layer, plot accuracy curves

Validates that layer 26 is optimal for SAE-based interpretability by showing
where genomic annotation information appears in the network.

Usage:
    # Phase A: Extract activations (GPU cluster)
    python tools/linear_probes.py extract \
        --fasta /path/to/genome.fna \
        --chrom NC_000022.11 \
        --gtf /path/to/genomic.gtf \
        --layers 0 4 8 12 16 20 24 26 28 30 \
        --n_samples 10000

    # Phase B: Train probes (local, no GPU needed)
    python tools/linear_probes.py train \
        --activations results/.../data/activations.npz \
        --output_dir results
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

DEFAULT_LAYERS = [0, 4, 8, 12, 16, 20, 24, 26, 28, 30]
ANNOTATION_TYPES = ['exon', 'intron', 'intergenic', 'CDS', 'UTR']


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("linear_probes")
    logger.setLevel(getattr(logging, log_level.upper()))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def load_annotations(gtf_path: str, chrom: str, genome_length: int) -> np.ndarray:
    """Load genomic annotations from GTF and create per-position labels.

    Labels: 0=intergenic, 1=intron, 2=exon, 3=CDS, 4=UTR

    Higher-priority labels override lower ones (CDS > exon > intron > intergenic).

    Args:
        gtf_path: Path to GTF file
        chrom: Chromosome accession
        genome_length: Length of chromosome sequence

    Returns:
        np.ndarray of shape (genome_length,) with integer labels
    """
    labels = np.zeros(genome_length, dtype=np.int8)  # 0 = intergenic

    # Track gene extents for intron detection
    gene_regions = []  # (start, end)
    exon_regions = []  # (start, end)
    cds_regions = []
    utr_regions = []

    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 9:
                continue
            seqid, source, ftype, f_start, f_end = fields[0], fields[1], fields[2], int(fields[3]) - 1, int(fields[4])

            if seqid != chrom:
                continue

            f_start = max(0, f_start)
            f_end = min(genome_length, f_end)

            if ftype == 'gene':
                gene_regions.append((f_start, f_end))
            elif ftype == 'exon':
                exon_regions.append((f_start, f_end))
            elif ftype == 'CDS':
                cds_regions.append((f_start, f_end))
            elif ftype in ('five_prime_UTR', 'three_prime_UTR', 'UTR'):
                utr_regions.append((f_start, f_end))

    # Apply labels in priority order (lowest first, highest overwrites)
    # 1. Mark gene body as intron
    for s, e in gene_regions:
        labels[s:e] = 1  # intron

    # 2. Mark exons
    for s, e in exon_regions:
        labels[s:e] = 2  # exon

    # 3. Mark CDS
    for s, e in cds_regions:
        labels[s:e] = 3  # CDS

    # 4. Mark UTR
    for s, e in utr_regions:
        labels[s:e] = 4  # UTR

    return labels


def subsample_positions(
    labels: np.ndarray,
    n_samples_per_class: int = 10000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample balanced positions from each annotation class.

    Args:
        labels: Per-position annotation labels
        n_samples_per_class: Number of positions per class
        seed: Random seed

    Returns:
        Tuple of (positions, labels) arrays for selected positions
    """
    rng = np.random.RandomState(seed)
    all_positions = []
    all_labels = []

    unique_labels = np.unique(labels)
    for label_val in unique_labels:
        positions = np.where(labels == label_val)[0]
        if len(positions) == 0:
            continue

        n = min(n_samples_per_class, len(positions))
        selected = rng.choice(positions, size=n, replace=False)
        all_positions.append(selected)
        all_labels.append(np.full(n, label_val, dtype=np.int8))

    positions = np.concatenate(all_positions)
    labels_out = np.concatenate(all_labels)

    # Shuffle
    shuffle_idx = rng.permutation(len(positions))
    return positions[shuffle_idx], labels_out[shuffle_idx]


def extract_activations(
    model,
    sequence: str,
    positions: np.ndarray,
    layers: List[int],
    chunk_size: int = 8192,
    logger: Optional[logging.Logger] = None,
) -> Dict[int, np.ndarray]:
    """Extract hidden state activations at specified positions and layers.

    Processes sequence in chunks, extracts activations only at requested positions.

    Args:
        model: ObservableEvo2 instance
        sequence: DNA sequence
        positions: Array of positions to extract activations for
        layers: Layer indices to extract from
        chunk_size: Chunk size for processing

    Returns:
        Dict mapping layer_index -> np.ndarray of shape (n_positions, d_hidden)
    """
    import torch

    layer_names = [f'blocks-{i}' for i in layers]
    d_hidden = model.d_hidden

    # Pre-allocate
    activations = {l: np.zeros((len(positions), d_hidden), dtype=np.float32) for l in layers}
    position_done = np.zeros(len(positions), dtype=bool)

    # Sort positions for efficient chunking
    sorted_indices = np.argsort(positions)
    sorted_positions = positions[sorted_indices]

    # Process in chunks
    overlap = 256
    stride = chunk_size - overlap
    n_chunks = max(1, (len(sequence) - overlap + stride - 1) // stride)

    if logger:
        logger.info(f"Extracting activations at {len(positions)} positions, "
                     f"{len(layers)} layers, {n_chunks} chunks")

    t0 = time.time()
    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * stride
        chunk_end = min(chunk_start + chunk_size, len(sequence))
        chunk_seq = sequence[chunk_start:chunk_end]

        if len(chunk_seq) < 10:
            continue

        # Find positions within this chunk (use core region to avoid edge effects)
        core_start = chunk_start + (overlap // 2 if chunk_idx > 0 else 0)
        core_end = chunk_end - (overlap // 2 if chunk_end < len(sequence) else 0)

        # Binary search for positions in core range
        left = np.searchsorted(sorted_positions, core_start, side='left')
        right = np.searchsorted(sorted_positions, core_end, side='left')

        if left >= right:
            continue

        chunk_positions = sorted_positions[left:right]
        chunk_orig_indices = sorted_indices[left:right]

        # Forward pass
        toks = model.tokenizer.tokenize(chunk_seq)
        toks_tensor = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(model.device)

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, acts = model.forward(toks_tensor, cache_activations_at=layer_names)

        # Extract at positions
        for layer_idx, layer_name in zip(layers, layer_names):
            if layer_name not in acts:
                continue
            layer_act = acts[layer_name]
            if isinstance(layer_act, tuple):
                layer_act = layer_act[0]

            # Convert genomic positions to chunk-local positions
            local_positions = chunk_positions - chunk_start
            valid_mask = (local_positions >= 0) & (local_positions < layer_act.shape[1])
            valid_local = local_positions[valid_mask]
            valid_orig = chunk_orig_indices[valid_mask]

            if len(valid_local) > 0:
                extracted = layer_act[0, valid_local, :].cpu().float().numpy()
                activations[layer_idx][valid_orig] = extracted
                position_done[valid_orig] = True

        if logger and (chunk_idx + 1) % 100 == 0:
            n_done = position_done.sum()
            logger.info(f"  Chunk {chunk_idx+1}/{n_chunks}, {n_done}/{len(positions)} positions extracted")

    elapsed = time.time() - t0
    if logger:
        logger.info(f"Extraction complete in {elapsed:.1f}s ({position_done.sum()}/{len(positions)} positions)")

    return activations


def train_probes(
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    logger: Optional[logging.Logger] = None,
) -> Dict[int, Dict]:
    """Train logistic regression probes at each layer.

    Args:
        activations: Dict mapping layer -> (n_samples, d_hidden) arrays
        labels: (n_samples,) annotation labels
        test_size: Fraction for test set
        seed: Random seed

    Returns:
        Dict mapping layer -> {accuracy, f1_macro, confusion_matrix, ...}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler

    results = {}

    for layer_idx in sorted(activations.keys()):
        X = activations[layer_idx]
        y = labels

        # Remove any all-zero rows (positions that weren't extracted)
        valid_mask = np.any(X != 0, axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 100:
            if logger:
                logger.warning(f"Layer {layer_idx}: only {len(X)} valid samples, skipping")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if logger:
            logger.info(f"Training probe for layer {layer_idx} "
                         f"(train={len(X_train)}, test={len(X_test)})...")

        t0 = time.time()
        clf = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)

        # Per-class accuracy
        unique_labels = np.unique(y_test)
        per_class = {}
        for label_val in unique_labels:
            mask = y_test == label_val
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            class_name = ANNOTATION_TYPES[label_val] if label_val < len(ANNOTATION_TYPES) else str(label_val)
            per_class[class_name] = round(float(class_acc), 4)

        results[layer_idx] = {
            'accuracy': round(float(acc), 4),
            'f1_macro': round(float(f1), 4),
            'per_class_accuracy': per_class,
            'confusion_matrix': cm.tolist(),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'train_time_s': round(train_time, 2),
        }

        if logger:
            logger.info(f"  Layer {layer_idx}: accuracy={acc:.3f}, f1={f1:.3f} ({train_time:.1f}s)")

    return results


def plot_accuracy_curves(results: Dict[int, Dict], output_path: str, chrom: str):
    """Plot accuracy and F1 vs layer index."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(results.keys())
    accuracies = [results[l]['accuracy'] for l in layers]
    f1_scores = [results[l]['f1_macro'] for l in layers]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(layers, accuracies, 'o-', color='#3498db', linewidth=2, markersize=8, label='Accuracy')
    ax1.plot(layers, f1_scores, 's-', color='#e74c3c', linewidth=2, markersize=8, label='F1 (macro)')

    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title(f'Linear Probe Performance by Layer — {chrom}', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(layers)

    # Highlight layer 26
    if 26 in results:
        ax1.axvline(x=26, color='#f39c12', linestyle='--', alpha=0.7, label='Layer 26 (SAE)')

    # Random baseline
    n_classes = len(ANNOTATION_TYPES)
    ax1.axhline(y=1.0/n_classes, color='gray', linestyle=':', alpha=0.5, label=f'Random ({1.0/n_classes:.2f})')
    ax1.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_class_curves(results: Dict[int, Dict], output_path: str, chrom: str):
    """Plot per-class accuracy vs layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(results.keys())
    all_classes = set()
    for r in results.values():
        all_classes.update(r['per_class_accuracy'].keys())

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cls_name in enumerate(sorted(all_classes)):
        accs = [results[l]['per_class_accuracy'].get(cls_name, np.nan) for l in layers]
        color = colors[i % len(colors)]
        ax.plot(layers, accs, 'o-', color=color, linewidth=2, markersize=6, label=cls_name)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Per-class Probe Accuracy by Layer — {chrom}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xticks(layers)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear probes for Evo2 hidden states",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Phase to run')

    # Phase A: Extract
    extract_parser = subparsers.add_parser('extract', help='Extract activations (GPU)')
    extract_parser.add_argument("--fasta", required=True)
    extract_parser.add_argument("--chrom", required=True)
    extract_parser.add_argument("--gtf", required=True, help="GTF annotation file")
    extract_parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    extract_parser.add_argument("--n_samples", type=int, default=10000,
                                help="Samples per annotation class (default: 10000)")
    extract_parser.add_argument("--chunk_size", type=int, default=8192)
    extract_parser.add_argument("--output_dir", default="results")
    extract_parser.add_argument("--chrom_name", default=None)
    extract_parser.add_argument("--log_level", default="INFO")

    # Phase B: Train
    train_parser = subparsers.add_parser('train', help='Train probes (local)')
    train_parser.add_argument("--activations", required=True,
                              help="Path to activations.npz from extract phase")
    train_parser.add_argument("--output_dir", default="results")
    train_parser.add_argument("--chrom_name", default=None)
    train_parser.add_argument("--log_level", default="INFO")

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.log_level)

    if args.command == 'extract':
        main_extract(args, logger)
    elif args.command == 'train':
        main_train(args, logger)
    else:
        print("Usage: linear_probes.py {extract,train} ...")
        sys.exit(1)


def main_extract(args, logger):
    t_start = time.time()

    # --- Load sequence ---
    logger.info(f"Loading {args.chrom} from {args.fasta}")
    from score_chromosome import load_chromosome_sequence
    sequence, actual_start, actual_end = load_chromosome_sequence(
        args.fasta, args.chrom, logger=logger
    )
    logger.info(f"Loaded {len(sequence):,} bp")

    # --- Load annotations ---
    logger.info(f"Loading annotations from {args.gtf}")
    labels = load_annotations(args.gtf, args.chrom, len(sequence))
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        name = ANNOTATION_TYPES[u] if u < len(ANNOTATION_TYPES) else str(u)
        logger.info(f"  {name}: {c:,} positions ({c/len(labels):.1%})")

    # --- Subsample ---
    positions, sampled_labels = subsample_positions(labels, n_samples_per_class=args.n_samples)
    logger.info(f"Subsampled {len(positions)} positions ({len(np.unique(sampled_labels))} classes)")

    # --- Initialize model ---
    logger.info("Initializing Evo2 model...")
    from sae_utils import ObservableEvo2
    model = ObservableEvo2("evo2_7b")
    logger.info("Model loaded")

    # --- Extract activations ---
    activations = extract_activations(
        model, sequence, positions, args.layers,
        chunk_size=args.chunk_size, logger=logger,
    )

    # --- Build output directory ---
    chrom_name = args.chrom_name or args.chrom.replace(".", "_")
    layer_str = f"layers_{len(args.layers)}"
    run_dir = build_run_dir(args.output_dir, chrom_name, "linear_probes", f"extract_{layer_str}")
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    # --- Save ---
    npz_data = {f"layer_{l}": activations[l] for l in activations}
    npz_data['positions'] = positions
    npz_data['labels'] = sampled_labels
    npz_data['layers'] = np.array(sorted(activations.keys()))
    np.savez_compressed(os.path.join(data_dir, "activations.npz"), **npz_data)
    logger.info("Saved activations.npz")

    # Save metadata
    metadata = {
        "chrom": args.chrom,
        "genome_length": len(sequence),
        "layers": sorted(activations.keys()),
        "n_positions": len(positions),
        "n_samples_per_class": args.n_samples,
        "label_names": ANNOTATION_TYPES,
    }
    with open(os.path.join(data_dir, "extract_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    write_source(run_dir, fasta=args.fasta, gtf=args.gtf)
    wall_time = time.time() - t_start
    write_completed(run_dir, "linear_probes.py (extract)", wall_time)
    logger.info(f"Extract phase done in {wall_time:.1f}s. Output: {run_dir}")


def main_train(args, logger):
    t_start = time.time()

    # --- Load activations ---
    logger.info(f"Loading activations from {args.activations}")
    data = np.load(args.activations, allow_pickle=True)

    layers = data['layers'].tolist()
    labels = data['labels']
    activations = {l: data[f'layer_{l}'] for l in layers}

    logger.info(f"Loaded {len(labels)} samples, {len(layers)} layers: {layers}")

    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        name = ANNOTATION_TYPES[u] if u < len(ANNOTATION_TYPES) else str(u)
        logger.info(f"  {name}: {c} samples")

    # --- Train probes ---
    results = train_probes(activations, labels, logger=logger)

    # --- Build output directory ---
    chrom_name = args.chrom_name or "probes"
    run_dir = build_run_dir(args.output_dir, chrom_name, "linear_probes", "train")
    data_dir = os.path.join(run_dir, "data")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    # --- Save results ---
    with open(os.path.join(data_dir, "probe_results.json"), "w") as f:
        # Convert int keys to str for JSON
        json_safe = {str(k): v for k, v in results.items()}
        json.dump(json_safe, f, indent=2)
        f.write("\n")
    logger.info("Saved probe_results.json")

    # --- Plot ---
    plot_accuracy_curves(results, os.path.join(plots_dir, "accuracy_by_layer.png"), chrom_name)
    logger.info("Saved accuracy_by_layer.png")

    plot_per_class_curves(results, os.path.join(plots_dir, "per_class_accuracy.png"), chrom_name)
    logger.info("Saved per_class_accuracy.png")

    # --- Summary ---
    if results:
        best_layer = max(results, key=lambda l: results[l]['accuracy'])
        logger.info(f"Best layer: {best_layer} (accuracy={results[best_layer]['accuracy']:.3f})")
        if 26 in results:
            logger.info(f"Layer 26: accuracy={results[26]['accuracy']:.3f}")

    write_source(run_dir, activations=args.activations)
    wall_time = time.time() - t_start
    write_completed(run_dir, "linear_probes.py (train)", wall_time)
    logger.info(f"Train phase done in {wall_time:.1f}s. Output: {run_dir}")


if __name__ == "__main__":
    main()
