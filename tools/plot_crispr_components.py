#!/usr/bin/env python3
"""
plot_crispr_components.py

Annotate CRISPR array sub-components (spacer, repeat, Cas protein) on
tSNE/UMAP latent analysis plots for E. coli and Bacillus.

CRISPR component coordinates are obtained from CRISPRCasFinder output
(JSON format). The script parses the CRISPRCasFinder result, extracts
spacer/repeat/Cas protein coordinates, and overlays them on pre-computed
embedding scatter plots.

Supports:
  - E. coli K-12 MG1655 (NC_000913.3)
  - Bacillus subtilis 168 (NC_000964.3)

Usage:
    # From CRISPRCasFinder JSON output
    python tools/plot_crispr_components.py \\
        --crispr_json /path/to/CRISPRCasFinder/result.json \\
        --latent results/NC_000913.3/sae/.../latent_analysis/data/cluster_assignments.tsv \\
        --boundaries results/NC_000913.3/scoring/.../data/drop_boundaries.tsv \\
        --output_dir plots/ \\
        --organism ecoli

    # From a pre-parsed TSV of CRISPR components
    python tools/plot_crispr_components.py \\
        --crispr_tsv crispr_components_ecoli.tsv \\
        --latent results/NC_000913.3/sae/.../latent_analysis/data/cluster_assignments.tsv \\
        --boundaries results/NC_000913.3/scoring/.../data/drop_boundaries.tsv \\
        --output_dir plots/ \\
        --organism ecoli
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Component type colors
COMPONENT_COLORS = {
    "spacer": "#2ecc71",    # green
    "repeat": "#3498db",    # blue
    "cas_protein": "#e74c3c",  # red
    "leader": "#f39c12",    # orange
    "trailer": "#9b59b6",   # purple
}

COMPONENT_ORDER = ["spacer", "repeat", "cas_protein", "leader", "trailer"]

# Known CRISPR loci (fallback if CRISPRCasFinder output not available)
KNOWN_CRISPR_LOCI = {
    "ecoli": {
        "CRISPR-I":  (2875441, 2876516),
        "CRISPR-II": (2877618, 2878569),
        # cas genes (type I-E)
        "cas3":  (2883256, 2885556),
        "casA":  (2881613, 2883259),
        "casB":  (2880953, 2881615),
        "casC":  (2880273, 2880959),
        "casD":  (2879929, 2880276),
        "casE":  (2879264, 2879935),
    },
    "bacillus": {
        "CRISPR (csn/cas)": (2747000, 2750000),
    },
}


def parse_crisprcasfinder_json(json_path):
    """Parse CRISPRCasFinder JSON output to extract component coordinates.

    Returns a list of dicts: {type, name, start, end}
    where type is one of: spacer, repeat, cas_protein, leader, trailer
    """
    with open(json_path) as f:
        data = json.load(f)

    components = []

    # CRISPRCasFinder output structure:
    # Sequences[].Crisprs[].Regions[]  (for CRISPRs)
    # Sequences[].Cas[].Genes[]         (for Cas proteins)
    for seq in data.get("Sequences", []):
        # CRISPR arrays
        for crispr in seq.get("Crisprs", []):
            array_name = crispr.get("Name", "CRISPR")
            start_pos = crispr.get("Start", 0)

            # Direct repeats
            for dr in crispr.get("Repeat_list", crispr.get("DRs", [])):
                components.append({
                    "type": "repeat",
                    "name": f"{array_name}_DR",
                    "start": dr.get("Start", dr.get("Position", 0)),
                    "end": dr.get("End", dr.get("Position", 0) + dr.get("Length", 30)),
                })

            # Spacers
            for sp in crispr.get("Spacer_list", crispr.get("Spacers", [])):
                components.append({
                    "type": "spacer",
                    "name": f"{array_name}_spacer",
                    "start": sp.get("Start", sp.get("Position", 0)),
                    "end": sp.get("End", sp.get("Position", 0) + sp.get("Length", 30)),
                })

            # Leader/trailer if present
            if "Leader" in crispr:
                leader = crispr["Leader"]
                components.append({
                    "type": "leader",
                    "name": f"{array_name}_leader",
                    "start": leader.get("Start", 0),
                    "end": leader.get("End", 0),
                })

        # Cas genes/proteins
        for cas_system in seq.get("Cas", []):
            for gene in cas_system.get("Genes", []):
                components.append({
                    "type": "cas_protein",
                    "name": gene.get("Sub_type", gene.get("Type", "cas")),
                    "start": gene.get("Start", 0),
                    "end": gene.get("End", 0),
                })

    logger.info(f"Parsed {len(components)} CRISPR components from {json_path}")
    for ctype in COMPONENT_ORDER:
        n = sum(1 for c in components if c["type"] == ctype)
        if n > 0:
            logger.info(f"  {ctype}: {n}")

    return components


def parse_crispr_tsv(tsv_path):
    """Parse a pre-made TSV of CRISPR components.

    Expected columns: type, name, start, end
    """
    df = pd.read_csv(tsv_path, sep="\t", comment="#")
    components = df.to_dict("records")
    logger.info(f"Loaded {len(components)} CRISPR components from {tsv_path}")
    return components


def build_fallback_components(organism):
    """Build component list from known loci (no sub-component detail)."""
    loci = KNOWN_CRISPR_LOCI.get(organism, {})
    components = []
    for name, (start, end) in loci.items():
        ctype = "cas_protein" if name.startswith("cas") else "repeat"
        components.append({"type": ctype, "name": name, "start": start, "end": end})
    logger.info(f"Using fallback loci for {organism}: {len(components)} entries")
    return components


def match_regions_to_components(region_starts, region_ends, components):
    """For each SAE region, find which CRISPR component(s) it overlaps.

    Returns
    -------
    region_components : list of list of str
        Per-region list of overlapping component types.
    region_component_names : list of list of str
        Per-region list of overlapping component names.
    """
    n = len(region_starts)
    region_components = [[] for _ in range(n)]
    region_component_names = [[] for _ in range(n)]

    for comp in components:
        cs, ce = comp["start"], comp["end"]
        for i in range(n):
            # Overlap check
            if region_starts[i] < ce and region_ends[i] > cs:
                region_components[i].append(comp["type"])
                region_component_names[i].append(comp["name"])

    n_hit = sum(1 for rc in region_components if rc)
    logger.info(f"Matched {n_hit}/{n} regions to CRISPR components")
    return region_components, region_component_names


def plot_crispr_overlay(coords, region_components, components, emb_name,
                        title, out_path, point_size=None, alpha=None):
    """Plot embedding with CRISPR components highlighted."""
    n = len(coords)
    if point_size is None:
        point_size = 20 if n < 2000 else (8 if n < 10000 else 3)
    if alpha is None:
        alpha = 0.7 if n < 2000 else (0.6 if n < 10000 else 0.5)

    fig, ax = plt.subplots(figsize=(14, 11))

    # Background: all non-CRISPR points
    non_crispr = [not rc for rc in region_components]
    non_crispr = np.array(non_crispr)
    ax.scatter(coords[non_crispr, 0], coords[non_crispr, 1],
               c="#d0d0d0", s=point_size * 0.7, alpha=0.3,
               edgecolors="none", rasterized=True, label="Other")

    # Overlay CRISPR components by type
    for ctype in COMPONENT_ORDER:
        mask = np.array([ctype in rc for rc in region_components])
        n_hit = mask.sum()
        if n_hit == 0:
            continue
        color = COMPONENT_COLORS[ctype]
        label_name = ctype.replace("_", " ").title()
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=point_size * 3, alpha=0.9,
                   edgecolors="black", linewidths=0.5,
                   label=f"{label_name} (n={n_hit})",
                   zorder=10)

    ax.legend(fontsize=10, markerscale=2, loc="best")
    prefix = emb_name.upper()
    ax.set_xlabel(f"{prefix} 1", fontsize=11)
    ax.set_ylabel(f"{prefix} 2", fontsize=11)
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot CRISPR sub-components on tSNE/UMAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input sources (one required)
    parser.add_argument("--crispr_json", default=None,
                        help="CRISPRCasFinder JSON output")
    parser.add_argument("--crispr_tsv", default=None,
                        help="Pre-parsed TSV of CRISPR components (type, name, start, end)")

    # Latent analysis data
    parser.add_argument("--latent", required=True,
                        help="Path to cluster_assignments.tsv (or annotated variant)")
    parser.add_argument("--boundaries", default=None,
                        help="Path to drop_boundaries.tsv (for real genomic coordinates)")

    # Options
    parser.add_argument("--organism", required=True, choices=["ecoli", "bacillus"],
                        help="Organism")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for plots")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load CRISPR components
    if args.crispr_json:
        components = parse_crisprcasfinder_json(args.crispr_json)
    elif args.crispr_tsv:
        components = parse_crispr_tsv(args.crispr_tsv)
    else:
        logger.warning("No CRISPRCasFinder output provided, using fallback known loci")
        components = build_fallback_components(args.organism)

    if not components:
        logger.error("No CRISPR components found")
        sys.exit(1)

    # Load latent analysis
    latent = pd.read_csv(args.latent, sep="\t", comment="#")
    logger.info(f"Loaded {len(latent)} regions from {args.latent}")

    # Get real genomic coordinates
    if args.boundaries and os.path.isfile(args.boundaries):
        bounds = pd.read_csv(args.boundaries, sep="\t", comment="#")
        n = min(len(bounds), len(latent))
        region_starts = bounds["genomic_start"].values[:n]
        region_ends = bounds["genomic_end"].values[:n]
        latent = latent.iloc[:n]
    elif "genomic_start" in latent.columns:
        region_starts = latent["genomic_start"].values
        region_ends = latent["genomic_end"].values
    else:
        logger.error("No genomic coordinates available. "
                     "Provide --boundaries or ensure latent TSV has genomic_start/genomic_end")
        sys.exit(1)

    # Match regions to CRISPR components
    region_components, region_names = match_regions_to_components(
        region_starts, region_ends, components
    )

    # Generate plots for each embedding type
    organism_name = {"ecoli": "E. coli K-12", "bacillus": "B. subtilis 168"}[args.organism]

    for emb_name, col1, col2 in [("tsne", "tsne_1", "tsne_2"),
                                   ("umap", "umap_1", "umap_2")]:
        if col1 not in latent.columns:
            continue

        coords = latent[[col1, col2]].values

        plot_crispr_overlay(
            coords, region_components, components, emb_name,
            title=f"{emb_name.upper()} — {organism_name}\n"
                  f"CRISPR Array Components (N={len(latent)})",
            out_path=os.path.join(args.output_dir, f"{emb_name}_crispr_components.png"),
        )

    # Save component matching summary
    summary_path = os.path.join(args.output_dir, "crispr_region_matches.tsv")
    with open(summary_path, "w") as f:
        f.write("region_idx\tgenomic_start\tgenomic_end\tcomponent_types\tcomponent_names\n")
        for i in range(len(region_starts)):
            types = ",".join(region_components[i]) if region_components[i] else "none"
            names = ",".join(region_names[i]) if region_names[i] else "none"
            f.write(f"{i}\t{region_starts[i]}\t{region_ends[i]}\t{types}\t{names}\n")
    logger.info(f"Saved component matches: {summary_path}")


if __name__ == "__main__":
    main()
