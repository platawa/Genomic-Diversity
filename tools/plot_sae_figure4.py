#!/usr/bin/env python3
"""
plot_sae_figure4.py — Figure 4C-style SAE feature activation plots

Creates publication-quality stacked SAE feature traces with GTF gene
annotations overlaid, matching the Evo2 paper Figure 4C style.

Two modes:
  - Live mode (GPU): provide a genomic window → run Evo2+SAE → plot
  - Precomputed mode (no GPU): load saved feature matrix → plot

Usage:
    # Live mode (on GPU cluster)
    python tools/plot_sae_figure4.py \\
        --fasta GENOME.fna --chrom chr22 \\
        --window_start 20000000 --window_end 20100000 \\
        --gtf genomic.gtf \\
        --signature_features sae_chromosome_results/chr22/data/signature_features.tsv \\
        --n_features 8 --output_dir sae_figure4c_results/

    # Precomputed mode (local, no GPU)
    python tools/plot_sae_figure4.py \\
        --precomputed feature_matrices.npz --region_index 0 \\
        --gtf genomic.gtf --chrom chr22 \\
        --window_start 20000000 --window_end 20100000 \\
        --n_features 8 --output_dir sae_figure4c_results/

    # Specify features manually
    python tools/plot_sae_figure4.py \\
        --fasta GENOME.fna --chrom chr22 \\
        --window_start 20000000 --window_end 20100000 \\
        --features 13606,26069,30262,2812 \\
        --output_dir sae_figure4c_results/
"""

import argparse
import logging
import os
import sys

import numpy as np

# Add project root to path for results_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, write_completed, write_source

logger = logging.getLogger("sae_figure4")


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def load_signature_features(tsv_path, n_features=8):
    """Load top N features from signature_features.tsv, sorted by mean_activation.

    Returns list of feature ID ints.
    """
    rows = []
    with open(tsv_path) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cols = line.split('\t')
            if header is None:
                header = cols
                continue
            row = dict(zip(header, cols))
            rows.append(row)

    # Sort by mean_activation descending
    rows.sort(key=lambda r: -float(r.get('mean_activation', 0)))
    feature_ids = [int(r['feature_id']) for r in rows[:n_features]]
    return feature_ids


# =============================================================================
# LIVE SAE EXTRACTION (GPU)
# =============================================================================

def extract_window_features(fasta_path, chrom, window_start, window_end,
                            model_name="evo2_7b_262k", device="cuda:0"):
    """Run Evo2+SAE on a genomic window and return feature matrix.

    Args:
        fasta_path: Path to genome FASTA
        chrom: Chromosome name (e.g. 'chr22')
        window_start: Start position (0-based)
        window_end: End position (0-based, exclusive)
        model_name: Evo2 model identifier
        device: CUDA device

    Returns:
        feature_matrix: np.ndarray of shape (window_len, 32768)
    """
    import torch

    # Use the project's existing loaders
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_sae_on_chromosome_drops import load_chromosome_sequence, CHROM_MAP
    from sae_utils import ObservableEvo2, load_topk_sae_from_hf, get_feature_ts

    logger.info(f"Loading chromosome {chrom} from {fasta_path}")
    chromosome_seq = load_chromosome_sequence(fasta_path, chrom, logger)

    seq = chromosome_seq[window_start:window_end]
    logger.info(f"Window: {window_start:,}-{window_end:,} ({len(seq):,} bp)")

    logger.info(f"Loading Evo2 model ({model_name})...")
    model = ObservableEvo2(model_name)
    logger.info(f"Model loaded. Device: {model.device}")

    logger.info("Loading SAE from HuggingFace...")
    sae = load_topk_sae_from_hf(
        d_hidden=model.d_hidden,
        device=model.device,
        dtype=torch.bfloat16,
    )
    logger.info("SAE loaded (32,768 features, TopK=64)")

    # For long sequences, process in chunks to avoid OOM
    MAX_CHUNK = 8192
    if len(seq) <= MAX_CHUNK:
        logger.info(f"Running SAE forward pass ({len(seq):,} bp)...")
        feature_matrix = get_feature_ts(model, sae, seq)
    else:
        logger.info(f"Processing in chunks of {MAX_CHUNK} bp (sequence={len(seq):,} bp)...")
        overlap = 256  # overlap for continuity
        chunks = []
        pos = 0
        while pos < len(seq):
            chunk_end = min(pos + MAX_CHUNK, len(seq))
            chunk_seq = seq[pos:chunk_end]
            logger.info(f"  Chunk {len(chunks)+1}: positions {pos:,}-{chunk_end:,}")
            chunk_features = get_feature_ts(model, sae, chunk_seq)

            if pos == 0:
                chunks.append(chunk_features)
            else:
                # Skip the overlap region
                chunks.append(chunk_features[overlap:])

            if chunk_end >= len(seq):
                break
            pos = chunk_end - overlap

        feature_matrix = np.concatenate(chunks, axis=0)
        # Trim to exact window length
        feature_matrix = feature_matrix[:len(seq)]

    logger.info(f"Feature matrix shape: {feature_matrix.shape}")
    return feature_matrix


# =============================================================================
# PLOTTING — Figure 4C Style
# =============================================================================

def plot_figure4c(
    feature_matrix,
    feature_ids,
    window_start,
    window_end,
    gtf_features=None,
    entropy=None,
    entropy_start=0,
    output_path="figure4c.png",
    chrom="",
    figsize_width=40,
):
    """Create a Figure 4C-style stacked SAE feature activation plot.

    Args:
        feature_matrix: np.ndarray (seq_len, n_total_features) — SAE activations
        feature_ids: list of int — which features to plot
        window_start: genomic start coordinate
        window_end: genomic end coordinate
        gtf_features: list of feature dicts from load_annotation_features() (optional)
        entropy: np.ndarray of per-position entropy (optional)
        entropy_start: genomic start of the entropy array (for indexing)
        output_path: where to save the figure
        chrom: chromosome name for title
        figsize_width: figure width in inches
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    n_features = len(feature_ids)
    has_entropy = entropy is not None
    has_genes = gtf_features is not None and len(gtf_features) > 0

    # Panel count: N features + optional entropy + gene track
    n_panels = n_features + (1 if has_entropy else 0) + (1 if has_genes else 0)
    height_ratios = [1.0] * n_features
    if has_entropy:
        height_ratios.append(1.0)
    if has_genes:
        height_ratios.append(0.6)

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(figsize_width, max(n_panels, 4)),
        sharex=True,
        gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.05},
    )
    if n_panels == 1:
        axes = [axes]

    # Genomic x-coordinates
    seq_len = feature_matrix.shape[0]
    x = np.linspace(window_start, window_end, seq_len, endpoint=False)

    # GTF feature colors (from analyze_scoring_results.py)
    gff_colors = {
        "CDS":              "#3498db",
        "gene":             "#2ecc71",
        "mRNA":             "#1abc9c",
        "exon":             "#a8e6cf",
        "transcript":       "#1abc9c",
        "five_prime_UTR":   "#e67e22",
        "three_prime_UTR":  "#e74c3c",
        "start_codon":      "#9b59b6",
        "stop_codon":       "#8e44ad",
        "tRNA":             "#662D91",
        "rRNA":             "#7AC8AC",
        "ncRNA":            "#95a5a6",
    }

    # ── Feature trace panels ──
    for i, fid in enumerate(feature_ids):
        ax = axes[i]
        trace = feature_matrix[:, fid]
        ax.plot(x, trace, lw=0.5, color='black', alpha=0.9)
        ax.set_ylim([0, 5])
        ax.set_yticks([0, 5])
        ax.set_ylabel(f"F{fid}", fontsize=8, rotation=0, labelpad=30, va='center')
        ax.tick_params(axis='y', labelsize=7)

        # Gene region shading on feature panels
        if gtf_features:
            for feat in gtf_features:
                if feat["feature_type"] == "gene":
                    s = max(feat["start"], window_start)
                    e = min(feat["end_exclusive"], window_end)
                    ax.axvspan(s, e, alpha=0.08, facecolor="#2ecc71", edgecolor="none")

        # Remove top/right spines for clean look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Only show x labels on bottom panel
        if i < n_features - 1 and not (has_entropy or has_genes):
            pass  # last feature panel will get x label
        ax.tick_params(axis='x', labelbottom=False)

    # ── Entropy panel (optional) ──
    entropy_ax_idx = n_features if has_entropy else None
    if has_entropy:
        ax = axes[entropy_ax_idx]
        # Extract entropy for this window
        ent_start_idx = window_start - entropy_start
        ent_end_idx = window_end - entropy_start
        ent_start_idx = max(0, ent_start_idx)
        ent_end_idx = min(len(entropy), ent_end_idx)
        ent_slice = entropy[ent_start_idx:ent_end_idx]
        ent_x = np.linspace(window_start, window_end, len(ent_slice), endpoint=False)

        ax.plot(ent_x, ent_slice, lw=0.4, color='#2c3e50', alpha=0.8)
        ax.set_ylabel("Entropy", fontsize=8, rotation=0, labelpad=30, va='center')
        ax.tick_params(axis='y', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Gene shading
        if gtf_features:
            for feat in gtf_features:
                if feat["feature_type"] == "gene":
                    s = max(feat["start"], window_start)
                    e = min(feat["end_exclusive"], window_end)
                    ax.axvspan(s, e, alpha=0.08, facecolor="#2ecc71", edgecolor="none")

        ax.tick_params(axis='x', labelbottom=False)

    # ── Gene track panel (bottom) ──
    if has_genes:
        gene_ax = axes[-1]

        # Import and use draw_gene_track from analyze_scoring_results
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from analyze_scoring_results import draw_gene_track
        draw_gene_track(gene_ax, gtf_features, window_start, window_end)

        gene_ax.set_xlim(window_start, window_end)
        gene_ax.tick_params(axis='x', labelbottom=True, labelsize=8)
        gene_ax.set_xlabel(f"Genomic position ({chrom})", fontsize=10)
    else:
        # x label on last panel
        axes[-1].tick_params(axis='x', labelbottom=True, labelsize=8)
        axes[-1].set_xlabel(f"Genomic position ({chrom})", fontsize=10)

    # Set shared x limits
    for ax in axes:
        ax.set_xlim(window_start, window_end)

    # Format x-axis ticks as Mb
    from matplotlib.ticker import FuncFormatter
    def mb_formatter(x, pos):
        return f"{x/1e6:.2f} Mb"
    axes[-1].xaxis.set_major_formatter(FuncFormatter(mb_formatter))

    # Title
    window_kb = (window_end - window_start) / 1000
    fig.suptitle(
        f"SAE Feature Activations — {chrom} "
        f"{window_start:,}-{window_end:,} ({window_kb:.0f} kb)",
        fontsize=13, fontweight='bold', y=1.01,
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved figure: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Figure 4C-style SAE feature activation plots with GTF gene track",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Window specification
    parser.add_argument("--chrom", type=str, required=True,
                        help="Chromosome name (e.g. chr22)")
    parser.add_argument("--window_start", type=int, required=True,
                        help="Window start (0-based genomic coordinate)")
    parser.add_argument("--window_end", type=int, required=True,
                        help="Window end (0-based, exclusive)")

    # Live mode (GPU)
    parser.add_argument("--fasta", type=str, default=None,
                        help="Path to genome FASTA (enables live SAE extraction)")
    parser.add_argument("--model_name", type=str, default="evo2_7b_262k",
                        help="Evo2 model name (default: evo2_7b_262k)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Compute device (default: cuda:0)")

    # Precomputed mode
    parser.add_argument("--precomputed", type=str, default=None,
                        help="Path to .npz with precomputed feature matrices "
                             "(e.g. feature_matrices.npz)")
    parser.add_argument("--region_index", type=int, default=0,
                        help="Which region's feature matrix to use from .npz (default: 0)")

    # Feature selection
    parser.add_argument("--signature_features", type=str, default=None,
                        help="Path to signature_features.tsv (auto-select top N)")
    parser.add_argument("--features", type=str, default=None,
                        help="Comma-separated feature IDs (e.g. 13606,26069,30262)")
    parser.add_argument("--n_features", type=int, default=8,
                        help="Number of features to plot (default: 8)")

    # Annotations
    parser.add_argument("--gtf", type=str, default=None,
                        help="Path to GTF file for gene annotations")

    # Entropy overlay
    parser.add_argument("--entropy", type=str, default=None,
                        help="Path to entropy.npz for entropy trace panel")

    # Output
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Root results directory (default: ./results)")
    parser.add_argument("--save_features", action="store_true",
                        help="Save the 100kb feature matrix for later reuse")

    # Plot options
    parser.add_argument("--figsize_width", type=int, default=40,
                        help="Figure width in inches (default: 40)")

    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Setup logging
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, args.log_level))

    # Validate inputs
    if args.fasta is None and args.precomputed is None:
        parser.error("Must provide either --fasta (live mode) or --precomputed (precomputed mode)")

    if args.features is None and args.signature_features is None:
        parser.error("Must provide either --features or --signature_features")

    import time as _time
    _fig4_wall_start = _time.time()

    # Build organized output directory
    run_dir = build_run_dir(args.output_dir, args.chrom, "visualization", "figure4c")
    args.output_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)

    # Write source.json
    source_kwargs = {}
    if args.fasta:
        source_kwargs["fasta"] = os.path.abspath(args.fasta)
    if args.precomputed:
        source_kwargs["precomputed"] = os.path.abspath(args.precomputed)
    if args.entropy:
        source_kwargs["entropy"] = os.path.abspath(args.entropy)
    if args.signature_features:
        source_kwargs["signature_features"] = os.path.abspath(args.signature_features)
    if source_kwargs:
        write_source(run_dir, **source_kwargs)

    # ── Resolve feature IDs ──
    if args.features:
        feature_ids = [int(x.strip()) for x in args.features.split(',')]
        logger.info(f"Using manually specified features: {feature_ids}")
    else:
        feature_ids = load_signature_features(args.signature_features, args.n_features)
        logger.info(f"Top {len(feature_ids)} features from {args.signature_features}: {feature_ids}")

    # ── Get feature matrix ──
    if args.precomputed:
        logger.info(f"Loading precomputed features from {args.precomputed}")
        data = np.load(args.precomputed, allow_pickle=True)

        # Handle different npz formats
        if 'feature_matrices' in data:
            matrices = data['feature_matrices']
            if isinstance(matrices, np.ndarray) and matrices.dtype == object:
                feature_matrix = matrices[args.region_index]
            else:
                feature_matrix = matrices
        elif 'feature_matrix' in data:
            feature_matrix = data['feature_matrix']
        else:
            # Try first array key
            keys = list(data.keys())
            logger.info(f"Available keys in npz: {keys}")
            feature_matrix = data[keys[0]]

        logger.info(f"Loaded feature matrix: {feature_matrix.shape}")

        # For precomputed: the matrix may not span the full window.
        # The user must ensure window coords match the precomputed data.
    else:
        feature_matrix = extract_window_features(
            args.fasta, args.chrom,
            args.window_start, args.window_end,
            model_name=args.model_name,
            device=args.device,
        )

        if args.save_features:
            feat_path = os.path.join(args.output_dir, 'window_features.npz')
            np.savez_compressed(
                feat_path,
                feature_matrix=feature_matrix,
                chrom=args.chrom,
                window_start=args.window_start,
                window_end=args.window_end,
                feature_ids=np.array(feature_ids),
            )
            logger.info(f"Saved feature matrix: {feat_path}")

    # ── Load GTF annotations ──
    gtf_features = None
    if args.gtf:
        # Add project root to path for imports
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from analyze_scoring_results import load_annotation_features

        # Map chromosome name to RefSeq accession for GTF matching
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from run_sae_on_chromosome_drops import CHROM_MAP
        chrom_id = CHROM_MAP.get(args.chrom, args.chrom)

        logger.info(f"Loading GTF annotations for {chrom_id} in "
                     f"{args.window_start:,}-{args.window_end:,}")
        gtf_features = load_annotation_features(
            args.gtf, chrom_id, args.window_start, args.window_end
        )
        logger.info(f"Found {len(gtf_features)} annotation features")

    # ── Load entropy (optional) ──
    entropy = None
    entropy_start = 0
    if args.entropy:
        ent_data = np.load(args.entropy, allow_pickle=True)
        entropy = ent_data['entropy']
        entropy_start = int(ent_data.get('start', 0))
        logger.info(f"Loaded entropy: {len(entropy):,} positions, start={entropy_start:,}")

    # ── Generate plot ──
    window_kb = (args.window_end - args.window_start) // 1000
    output_name = f"figure4c_{args.chrom}_{args.window_start}_{args.window_end}.png"
    output_path = os.path.join(args.output_dir, output_name)

    plot_figure4c(
        feature_matrix=feature_matrix,
        feature_ids=feature_ids,
        window_start=args.window_start,
        window_end=args.window_end,
        gtf_features=gtf_features,
        entropy=entropy,
        entropy_start=entropy_start,
        output_path=output_path,
        chrom=args.chrom,
        figsize_width=args.figsize_width,
    )

    # Write COMPLETED sentinel
    _fig4_wall_time = _time.time() - _fig4_wall_start
    write_completed(run_dir, "plot_sae_figure4.py", _fig4_wall_time)
    logger.info(f"COMPLETED sentinel written to {run_dir}/COMPLETED")
    logger.info("Done!")


if __name__ == "__main__":
    main()
