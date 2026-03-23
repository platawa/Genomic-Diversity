#!/usr/bin/env python3
"""
plot_confidence_drops.py — Rank and plot entropy drop regions by confidence.

Loads drop_boundaries.tsv from the latest completed scoring run for each
chromosome, ranks ALL drops genome-wide by start_confidence, then generates
zoom plots (same style as analyze_scoring_results zoom_plots) for the top-N
and bottom-N drops for both z-score and MAD detection methods.

No GPU required — reads from existing entropy.npz and drop_boundaries.tsv.

Outputs (in --output_dir):
  confidence_distribution.png    — violin of confidence scores per chrom & method
  confidence_vs_length.png       — scatter: confidence vs region length
  genome_confidence_ranked.tsv   — full ranked table of all drops
  zscore/top/    — zoom plots for top-N highest-confidence z-score drops
  zscore/bottom/ — zoom plots for bottom-N lowest-confidence z-score drops
  zscore/middle/ — zoom plots for median-confidence z-score drops
  mad/top/       — same for MAD
  mad/bottom/
  mad/middle/

Usage:
    python tools/plot_confidence_drops.py \\
        --results_dir results/ \\
        --output_dir results/confidence_analysis/ \\
        --gtf /path/to/genomic.gtf \\
        --n 25
"""

import argparse
import os
import sys
import csv
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results_utils import find_latest_completed

# ---------------------------------------------------------------------------
ALL_HUMAN_CHROMS = [
    'chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10',
    'chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19',
    'chr20','chr21','chr22','chrX','chrY',
]

CHROM_MAP = {
    'chr1':'NC_000001.11','chr2':'NC_000002.12','chr3':'NC_000003.12',
    'chr4':'NC_000004.12','chr5':'NC_000005.10','chr6':'NC_000006.12',
    'chr7':'NC_000007.14','chr8':'NC_000008.11','chr9':'NC_000009.12',
    'chr10':'NC_000010.11','chr11':'NC_000011.10','chr12':'NC_000012.12',
    'chr13':'NC_000013.11','chr14':'NC_000014.9', 'chr15':'NC_000015.10',
    'chr16':'NC_000016.10','chr17':'NC_000017.11','chr18':'NC_000018.10',
    'chr19':'NC_000019.10','chr20':'NC_000020.11','chr21':'NC_000021.9',
    'chr22':'NC_000022.11','chrX':'NC_000023.11', 'chrY':'NC_000024.10',
}

ZOOM_BP   = 5000   # bp shown either side of each drop centre
SMOOTH_W  = 51


def setup_logging(level='INFO'):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=getattr(logging, level))
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (with simple per-chrom caches)
# ---------------------------------------------------------------------------

_entropy_cache: dict = {}
_smooth_cache:  dict = {}
_drops_cache:   dict = {}
_gtf_cache:     dict = {}


def get_entropy(scoring_run: str) -> np.ndarray:
    if scoring_run not in _entropy_cache:
        data = np.load(os.path.join(scoring_run, 'data', 'entropy.npz'))
        _entropy_cache[scoring_run] = data['entropy']
    return _entropy_cache[scoring_run]


def get_smooth(scoring_run: str) -> np.ndarray:
    if scoring_run not in _smooth_cache:
        from analyze_scoring_results import _rolling_mean
        _smooth_cache[scoring_run] = _rolling_mean(get_entropy(scoring_run), SMOOTH_W)
    return _smooth_cache[scoring_run]


def get_chrom_drops(chrom: str) -> list:
    """Return pre-loaded drop list for a chromosome (populated during load_all_drops)."""
    return _drops_cache.get(chrom, [])


def get_gtf_features(chrom: str, gtf_path: str, lo: int, hi: int) -> list:
    """Load GTF features for a window, caching the full chromosome range."""
    if not gtf_path:
        return []
    from analyze_scoring_results import load_annotation_features
    nc_id = CHROM_MAP.get(chrom, chrom)
    cache_key = (chrom, gtf_path)
    if cache_key not in _gtf_cache:
        # Load a very large range (whole chromosome) so we cache once per chrom
        _gtf_cache[cache_key] = load_annotation_features(gtf_path, nc_id, 0, 999_999_999)
    all_feats = _gtf_cache[cache_key]
    return [f for f in all_feats if f['start'] < hi and f['end_exclusive'] > lo]


def load_all_drops(results_dir: str, chroms: list, logger) -> list:
    all_drops = []
    for chrom in chroms:
        run_dir = find_latest_completed(results_dir, chrom, 'scoring')
        if run_dir is None:
            logger.warning(f'  {chrom}: no completed scoring run — skipping')
            continue
        tsv_path = os.path.join(run_dir, 'data', 'drop_boundaries.tsv')
        if not os.path.exists(tsv_path):
            logger.warning(f'  {chrom}: drop_boundaries.tsv not found')
            continue

        chrom_drops = []
        with open(tsv_path) as f:
            reader = csv.DictReader(
                (l for l in f if not l.startswith('#')), delimiter='\t'
            )
            for row in reader:
                chrom_drops.append({
                    'chrom':            chrom,
                    'drop_start':       int(row['drop_start']),
                    'drop_end':         int(row['drop_end']),
                    'genomic_start':    int(row['genomic_start']),
                    'genomic_end':      int(row['genomic_end']),
                    'region_length':    int(row['region_length']),
                    'method':           row['method'],
                    'start_confidence': float(row['start_confidence']),
                    'end_confidence':   float(row['end_confidence']),
                    'mean_entropy':     float(row['mean_entropy']),
                    'min_entropy':      float(row['min_entropy']),
                    '_scoring_run':     run_dir,
                })
        _drops_cache[chrom] = chrom_drops
        logger.info(f'  {chrom}: {len(chrom_drops):,} drops')
        all_drops.extend(chrom_drops)
    return all_drops


# ---------------------------------------------------------------------------
# Zoom plot (reuses _plot_window from analyze_scoring_results.py exactly)
# ---------------------------------------------------------------------------

def save_zoom_plot(drop: dict, rank: int, tier: str, out_dir: str,
                   gtf_path: str, logger):
    """
    Generate one zoom plot for a single drop region using _plot_window,
    the same function used by analyze_scoring_results zoom_plots.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from analyze_scoring_results import _plot_window

    chrom      = drop['chrom']
    sm         = get_smooth(drop['_scoring_run'])
    L          = len(sm)
    center     = (drop['drop_start'] + drop['drop_end']) // 2
    lo         = max(0, center - ZOOM_BP)
    hi         = min(L, center + ZOOM_BP)

    all_regions = get_chrom_drops(chrom)

    gene_feats = get_gtf_features(chrom, gtf_path, lo, hi) if gtf_path else []

    if gene_feats:
        fig, (ax, ax_annot) = plt.subplots(
            2, 1, figsize=(14, 5.5), sharex=True,
            gridspec_kw={'height_ratios': [4, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax_annot = None

    conf   = drop['start_confidence']
    method = drop['method'].upper()
    title  = f'{tier} #{rank} ({method} conf={conf:.2f})'

    _plot_window(
        ax, sm, all_regions, lo, hi,
        start=0, chrom=CHROM_MAP.get(chrom, chrom),
        title=title, mpatches=mpatches,
        highlight_region=drop,
        gene_features=gene_feats if gene_feats else None,
        ax_annot=ax_annot,
    )

    plt.tight_layout()
    fname = f'{rank:03d}_{drop["method"]}_{chrom}_{drop["drop_start"]}.png'
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary distribution plots
# ---------------------------------------------------------------------------

def plot_confidence_distribution(all_drops: list, output_path: str, logger=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    chroms  = [c for c in ALL_HUMAN_CHROMS if any(d['chrom'] == c for d in all_drops)]
    methods = ['zscore', 'mad']
    colors  = {'zscore': '#3498db', 'mad': '#e67e22'}

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(chroms) * 0.7), 5))
    for ax, method in zip(axes, methods):
        data, labels = [], []
        for chrom in chroms:
            vals = [d['start_confidence'] for d in all_drops
                    if d['chrom'] == chrom and d['method'] == method]
            if vals:
                data.append(vals)
                labels.append(chrom)
        if data:
            vp = ax.violinplot(data, showmedians=True, showextrema=False)
            for body in vp['bodies']:
                body.set_facecolor(colors[method])
                body.set_alpha(0.6)
            vp['cmedians'].set_color('black')
            vp['cmedians'].set_linewidth(1.2)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('start_confidence', fontsize=9)
        ax.set_title(f'{method.upper()} confidence distribution', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    if logger:
        logger.info(f'  Saved: {output_path}')


def plot_confidence_scatter(all_drops: list, output_path: str, logger=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    chroms = [c for c in ALL_HUMAN_CHROMS if any(d['chrom'] == c for d in all_drops)]
    cmap   = get_cmap('tab20', len(chroms))
    colors = {c: cmap(i) for i, c in enumerate(chroms)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, method in zip(axes, ['zscore', 'mad']):
        for chrom in chroms:
            pts = [d for d in all_drops if d['chrom'] == chrom and d['method'] == method]
            if pts:
                ax.scatter([d['region_length'] for d in pts],
                           [d['start_confidence'] for d in pts],
                           s=1.5, alpha=0.15, color=colors[chrom],
                           label=chrom, rasterized=True)
        ax.set_xlabel('Region length (bp)', fontsize=9)
        ax.set_ylabel('start_confidence', fontsize=9)
        ax.set_title(f'{method.upper()}: confidence vs length', fontsize=10)
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', markerscale=5,
               fontsize=7, bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    if logger:
        logger.info(f'  Saved: {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Rank and plot entropy drops by confidence across all chromosomes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--results_dir', default='results/')
    parser.add_argument('--output_dir',  default='results/confidence_analysis/')
    parser.add_argument('--gtf', default=None,
                        help='Path to GTF file for gene annotation tracks (optional)')
    parser.add_argument('--chroms', nargs='+', default=None,
                        help='Chromosomes to include (default: all with completed scoring)')
    parser.add_argument('--n', type=int, default=25,
                        help='Number of top AND bottom drops to plot per method (default: 25)')
    parser.add_argument('--min_length', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=0)
    parser.add_argument('--log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    logger.info('=' * 60)
    logger.info('CONFIDENCE DROP ANALYSIS')
    logger.info('=' * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    chroms = args.chroms or ALL_HUMAN_CHROMS
    logger.info(f'Loading drops for {len(chroms)} chromosomes...')
    all_drops = load_all_drops(args.results_dir, chroms, logger)
    logger.info(f'Total drops loaded: {len(all_drops):,}')

    if not all_drops:
        logger.error('No drops found. Check --results_dir.')
        sys.exit(1)

    if args.min_length > 0:
        all_drops = [d for d in all_drops if d['region_length'] >= args.min_length]
    if args.max_length > 0:
        all_drops = [d for d in all_drops if d['region_length'] <= args.max_length]

    # ── Ranked TSV ──────────────────────────────────────────────────────────
    tsv_path = os.path.join(args.output_dir, 'genome_confidence_ranked.tsv')
    all_sorted = sorted(all_drops, key=lambda d: d['start_confidence'], reverse=True)
    with open(tsv_path, 'w') as f:
        f.write('rank\tchrom\tgenomic_start\tgenomic_end\tregion_length\tmethod\t'
                'start_confidence\tend_confidence\tmean_entropy\tmin_entropy\n')
        for rank, d in enumerate(all_sorted, 1):
            f.write(f"{rank}\t{d['chrom']}\t{d['genomic_start']}\t{d['genomic_end']}\t"
                    f"{d['region_length']}\t{d['method']}\t{d['start_confidence']:.4f}\t"
                    f"{d['end_confidence']:.4f}\t{d['mean_entropy']:.6f}\t"
                    f"{d['min_entropy']:.6f}\n")
    logger.info(f'Saved: {tsv_path}')

    # ── Summary plots ────────────────────────────────────────────────────────
    logger.info('Generating summary distribution plots...')
    plot_confidence_distribution(
        all_drops, os.path.join(args.output_dir, 'confidence_distribution.png'), logger)
    plot_confidence_scatter(
        all_drops, os.path.join(args.output_dir, 'confidence_vs_length.png'), logger)

    # ── Per-method zoom plots ────────────────────────────────────────────────
    for method in ['zscore', 'mad']:
        ranked = sorted(
            [d for d in all_drops if d['method'] == method],
            key=lambda d: d['start_confidence'], reverse=True,
        )
        if not ranked:
            continue
        n = min(args.n, len(ranked))
        logger.info(f'{method.upper()}: {len(ranked):,} total drops, '
                    f'plotting top/middle/bottom {n}')

        tiers = {
            'top':    ranked[:n],
            'bottom': list(reversed(ranked[-n:])),
            'middle': ranked[len(ranked)//2 - n//2 : len(ranked)//2 + n//2],
        }

        for tier, drops in tiers.items():
            out_dir = os.path.join(args.output_dir, method, tier)
            os.makedirs(out_dir, exist_ok=True)
            for rank, drop in enumerate(drops, 1):
                try:
                    save_zoom_plot(drop, rank, tier, out_dir, args.gtf, logger)
                    if rank % 5 == 0:
                        logger.info(f'  {method}/{tier}: {rank}/{len(drops)}')
                except Exception as exc:
                    logger.warning(f'  {method}/{tier} rank {rank}: {exc}')

    logger.info('=' * 60)
    logger.info(f'DONE  →  {os.path.abspath(args.output_dir)}')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
