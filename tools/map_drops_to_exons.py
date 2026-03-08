#!/usr/bin/env python3
"""
map_drops_to_exons.py

Map detected drops/rises to exon boundaries and annotate with context.

Usage:
    python tools/map_drops_to_exons.py \
        --data_dir outputs/human/NPS_... \
        --gene_id NPS \
        --tolerance 100 \
        --output NPS_drop_exon_mapping.tsv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict


def load_entropy_tsv(tsv_path: Path) -> pd.DataFrame:
    """Load the main TSV with exon annotations."""
    df = pd.read_csv(tsv_path, sep='\t')
    return df


def load_drops(drops_path: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Load drops from .drops.txt file."""
    drops = {}
    with open(drops_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            method = parts[0]
            entries = parts[1].split(',')
            drop_list = []
            for entry in entries:
                if ':' in entry:
                    pos, score = entry.split(':')
                    drop_list.append((int(pos), float(score)))
                else:
                    drop_list.append((int(entry), 0.0))
            drops[method] = drop_list
    return drops


def load_rises(rises_path: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Load rises from .rises.txt file."""
    return load_drops(rises_path)  # Same format


def find_exon_context(df: pd.DataFrame, position: int, tolerance: int = 100) -> Dict:
    """
    Find exon context for a given position.

    Returns dict with:
        - in_exon: bool
        - exon_id: int or None
        - dist_to_exon_start: float
        - dist_to_exon_end: float
        - near_boundary: str ('exon_start', 'exon_end', 'intron', None)
        - boundary_dist: float (distance to nearest boundary)
    """
    # Find the row for this position
    idx = df['Pos'] == position
    if not idx.any():
        # Find closest position
        closest_idx = (df['Pos'] - position).abs().idxmin()
        row = df.loc[closest_idx]
    else:
        row = df[idx].iloc[0]

    in_exon = bool(row['IsExon'])
    exon_id = int(row['ExonID']) if row['ExonID'] >= 0 else None
    dist_start = row['DistToExonStart']
    dist_end = row['DistToExonEnd']

    # Determine if near a boundary
    near_boundary = None
    boundary_dist = min(abs(dist_start), abs(dist_end))

    if boundary_dist <= tolerance:
        if abs(dist_start) < abs(dist_end):
            near_boundary = 'exon_start'
            boundary_dist = dist_start
        else:
            near_boundary = 'exon_end'
            boundary_dist = dist_end

    return {
        'in_exon': in_exon,
        'exon_id': exon_id,
        'dist_to_exon_start': dist_start,
        'dist_to_exon_end': dist_end,
        'near_boundary': near_boundary,
        'boundary_dist': boundary_dist
    }


def get_exon_boundaries(df: pd.DataFrame) -> List[Dict]:
    """Extract all exon boundaries from the data."""
    boundaries = []

    # Find transitions in IsExon
    is_exon = df['IsExon'].values
    exon_id = df['ExonID'].values
    positions = df['Pos'].values

    for i in range(1, len(is_exon)):
        if is_exon[i] != is_exon[i-1]:
            if is_exon[i] == 1:  # Entering exon
                boundaries.append({
                    'position': positions[i],
                    'type': 'exon_start',
                    'exon_id': exon_id[i]
                })
            else:  # Leaving exon
                boundaries.append({
                    'position': positions[i-1],
                    'type': 'exon_end',
                    'exon_id': exon_id[i-1]
                })

    return boundaries


def map_drops_to_boundaries(
    drops: Dict[str, List[Tuple[int, float]]],
    boundaries: List[Dict],
    df: pd.DataFrame,
    tolerance: int = 100
) -> pd.DataFrame:
    """
    Map each drop to nearest exon boundary.

    Returns DataFrame with columns:
        method, position, score, in_exon, exon_id,
        nearest_boundary_type, nearest_boundary_pos, distance_to_boundary
    """
    rows = []

    for method, drop_list in drops.items():
        for pos, score in drop_list:
            context = find_exon_context(df, pos, tolerance)

            # Find nearest boundary
            nearest_boundary = None
            nearest_dist = float('inf')
            nearest_type = None

            for boundary in boundaries:
                dist = abs(pos - boundary['position'])
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_boundary = boundary['position']
                    nearest_type = f"{boundary['type']} (exon {boundary['exon_id']})"

            rows.append({
                'method': method,
                'position': pos,
                'score': score,
                'in_exon': context['in_exon'],
                'exon_id': context['exon_id'],
                'dist_to_exon_start': context['dist_to_exon_start'],
                'dist_to_exon_end': context['dist_to_exon_end'],
                'nearest_boundary_type': nearest_type,
                'nearest_boundary_pos': nearest_boundary,
                'distance_to_boundary': nearest_dist,
                'within_tolerance': nearest_dist <= tolerance
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Map detected drops/rises to exon boundaries"
    )

    ap.add_argument("--data_dir", type=Path, required=True,
                   help="Directory with genome_scoring outputs (containing data/ subfolder)")

    ap.add_argument("--gene_id", required=True,
                   help="Gene ID (used for finding files)")

    ap.add_argument("--tolerance", type=int, default=100,
                   help="bp tolerance for matching drops to boundaries (default: 100)")

    ap.add_argument("--output", type=Path, default=None,
                   help="Output TSV file (default: <gene_id>_drop_exon_mapping.tsv)")

    ap.add_argument("--include_rises", action="store_true",
                   help="Also map rise points")

    args = ap.parse_args()

    # Find data files
    data_subdir = args.data_dir / "data"
    if not data_subdir.exists():
        data_subdir = args.data_dir  # Try direct path

    # Look for files matching gene_id
    tsv_files = list(data_subdir.glob(f"*{args.gene_id}*.tsv"))
    if not tsv_files:
        tsv_files = list(data_subdir.glob("*.tsv"))

    if not tsv_files:
        print(f"Error: No TSV files found in {data_subdir}")
        return 1

    # Use first matching TSV (not window_summary)
    tsv_path = None
    for f in tsv_files:
        if 'window_summary' not in f.name and 'method_comparison' not in f.name:
            tsv_path = f
            break

    if tsv_path is None:
        print(f"Error: No main TSV file found")
        return 1

    print(f"Loading entropy data from: {tsv_path}")
    df = load_entropy_tsv(tsv_path)
    print(f"  Loaded {len(df)} positions")

    # Get exon boundaries
    boundaries = get_exon_boundaries(df)
    print(f"  Found {len(boundaries)} exon boundaries")

    # Load drops
    drops_path = tsv_path.with_suffix('.drops.txt')
    if not drops_path.exists():
        drops_path = data_subdir / f"{args.gene_id}.drops.txt"

    if drops_path.exists():
        print(f"Loading drops from: {drops_path}")
        drops = load_drops(drops_path)
        total_drops = sum(len(d) for d in drops.values())
        print(f"  Loaded {total_drops} drops from {len(drops)} methods")
    else:
        print(f"Warning: Drops file not found: {drops_path}")
        drops = {}

    # Load rises if requested
    rises = {}
    if args.include_rises:
        rises_path = tsv_path.with_suffix('.rises.txt')
        if not rises_path.exists():
            rises_path = data_subdir / f"{args.gene_id}.rises.txt"

        if rises_path.exists():
            print(f"Loading rises from: {rises_path}")
            rises = load_rises(rises_path)
            # Prefix method names to distinguish
            rises = {f"rise_{k}": v for k, v in rises.items()}
            total_rises = sum(len(r) for r in rises.values())
            print(f"  Loaded {total_rises} rises from {len(rises)} methods")

    # Combine drops and rises
    all_points = {**drops, **rises}

    if not all_points:
        print("No drops or rises to map!")
        return 1

    # Map to boundaries
    print(f"\nMapping drops/rises to exon boundaries (tolerance: {args.tolerance} bp)...")
    result_df = map_drops_to_boundaries(all_points, boundaries, df, args.tolerance)

    # Sort by position
    result_df = result_df.sort_values(['position', 'method'])

    # Summary
    print(f"\n{'='*60}")
    print("MAPPING SUMMARY")
    print(f"{'='*60}")

    within_tol = result_df['within_tolerance'].sum()
    total = len(result_df)
    print(f"Total drops/rises: {total}")
    print(f"Within {args.tolerance}bp of exon boundary: {within_tol} ({100*within_tol/total:.1f}%)")

    print(f"\nBy method:")
    for method in result_df['method'].unique():
        method_df = result_df[result_df['method'] == method]
        method_within = method_df['within_tolerance'].sum()
        method_total = len(method_df)
        print(f"  {method}: {method_within}/{method_total} near boundaries "
              f"({100*method_within/method_total:.1f}%)")

    print(f"\nExon boundaries detected:")
    for boundary in boundaries:
        b_pos = boundary['position']
        b_type = boundary['type']
        b_exon = boundary['exon_id']

        # Find drops near this boundary
        near_drops = result_df[
            (result_df['nearest_boundary_pos'] == b_pos) &
            (result_df['within_tolerance'])
        ]

        if len(near_drops) > 0:
            methods = ', '.join(near_drops['method'].unique())
            print(f"  {b_type} (exon {b_exon}) @ {b_pos}: detected by [{methods}]")
        else:
            print(f"  {b_type} (exon {b_exon}) @ {b_pos}: NOT detected")

    # Save output
    output_path = args.output or Path(f"{args.gene_id}_drop_exon_mapping.tsv")
    result_df.to_csv(output_path, sep='\t', index=False)
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
