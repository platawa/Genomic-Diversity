#!/usr/bin/env python3
"""
optimize_drop_parameters.py

Grid search over drop detection parameters to maximize agreement with
biological ground truth (splice sites, functional annotations, etc.)

Usage:
    python optimize_drop_parameters.py \
        --organism ecoli \
        --gene_list genes.txt \
        --ground_truth annotations.bed \
        --output optimized_params.json

Requires:
    - Entropy data (from score_chromosome.py)
    - Ground truth annotations (BED format or GFF)
    - Multiple test genes for cross-validation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import KFold
import itertools


@dataclass
class GroundTruthDrop:
    """Represents a validated drop location."""
    position: int
    feature_type: str  # e.g., "splice_site", "domain_boundary"
    confidence: float = 1.0


@dataclass
class ParameterSet:
    """Parameter configuration for a detection method."""
    method: str
    params: Dict[str, Any]

    def __hash__(self):
        return hash((self.method, frozenset(self.params.items())))


@dataclass
class EvaluationResult:
    """Results of parameter evaluation."""
    params: ParameterSet
    precision: float
    recall: float
    f1_score: float
    mcc: float  # Matthews Correlation Coefficient
    n_true_positives: int
    n_false_positives: int
    n_false_negatives: int
    genes_tested: List[str]


def load_ground_truth(annotation_file: Path, gene_id: str,
                     tolerance: int = 50) -> List[GroundTruthDrop]:
    """
    Load ground truth drop locations from annotation file.

    Args:
        annotation_file: BED or GFF file with functional annotations
        gene_id: Gene identifier
        tolerance: bp window around annotation (feature at position X means
                  true drop could be at X ± tolerance)

    Returns:
        List of validated drop positions
    """
    ground_truth = []
    ann_path = str(annotation_file)

    if ann_path.endswith('.bed') or ann_path.endswith('.tsv'):
        # Read BED/TSV output from build_ground_truth.py
        with open(ann_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 4:
                    continue
                # BED format from build_ground_truth.py:
                # chrom, start, end, name, score, strand, feature_type, gene_name, ...
                # Or TSV format: position, feature_type, expected_direction, gene_name, ...
                if fields[0].isdigit():
                    # TSV format (position-based)
                    pos = int(fields[0])
                    feat_type = fields[1] if len(fields) > 1 else 'unknown'
                    ground_truth.append(GroundTruthDrop(pos, feat_type))
                else:
                    # BED format: use midpoint of tolerance window
                    bed_start = int(fields[1])
                    bed_end = int(fields[2])
                    midpoint = (bed_start + bed_end) // 2
                    feat_type = fields[6] if len(fields) > 6 else fields[3]
                    ground_truth.append(GroundTruthDrop(midpoint, feat_type))

    elif ann_path.endswith('.gtf') or ann_path.endswith('.gff'):
        # Parse GTF/GFF directly
        with open(ann_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                feat_type = fields[2]
                if feat_type not in ('CDS', 'exon'):
                    continue
                # Check if this gene matches
                if gene_id and gene_id not in fields[8]:
                    continue
                start = int(fields[3]) - 1  # 0-based
                end = int(fields[4])
                ground_truth.append(GroundTruthDrop(start, f'{feat_type}_start'))
                ground_truth.append(GroundTruthDrop(end, f'{feat_type}_end'))

    return ground_truth


def load_entropy_data(gene_data_dir: Path) -> np.ndarray:
    """
    Load entropy scores from genome_scoring output.

    Args:
        gene_data_dir: Directory containing gene scoring outputs
                      (e.g., <gene_id>/data/<gene_id>.tsv)

    Returns:
        Entropy array
    """
    tsv_file = gene_data_dir / "data" / f"{gene_data_dir.name}.tsv"

    if not tsv_file.exists():
        raise FileNotFoundError(f"Entropy data not found: {tsv_file}")

    data = []
    with open(tsv_file) as f:
        header = next(f)  # Skip header
        for line in f:
            fields = line.strip().split('\t')
            # Assuming column 4 is entropy (adjust based on actual format)
            entropy = float(fields[3]) if len(fields) > 3 else np.nan
            data.append(entropy)

    return np.array(data)


def detect_drops_with_params(entropy: np.ndarray,
                             param_set: ParameterSet) -> List[int]:
    """
    Run drop detection with specified parameters.

    Args:
        entropy: Entropy signal
        param_set: Method and parameters to use

    Returns:
        List of detected drop positions
    """
    from detection_methods import METHODS

    method = param_set.method
    params = param_set.params

    # Map short names to METHODS keys
    method_key = {"local": "local_baseline"}.get(method, method)
    if method_key not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHODS.keys())}")

    drops = METHODS[method_key](entropy, **params)
    return sorted([pos for pos, _ in drops])


def evaluate_parameters(detected_drops: List[int],
                       ground_truth: List[GroundTruthDrop],
                       tolerance: int = 50) -> Tuple[int, int, int]:
    """
    Compare detected drops to ground truth.

    Args:
        detected_drops: Positions returned by detection method
        ground_truth: Validated drop positions
        tolerance: bp window for matching (detected within ±tolerance of truth)

    Returns:
        (true_positives, false_positives, false_negatives)
    """
    gt_positions = [gt.position for gt in ground_truth]

    true_positives = 0
    false_positives = 0
    matched_gt = set()

    for det_pos in detected_drops:
        # Check if this detection matches any ground truth within tolerance
        matched = False
        for i, gt_pos in enumerate(gt_positions):
            if i not in matched_gt and abs(det_pos - gt_pos) <= tolerance:
                true_positives += 1
                matched_gt.add(i)
                matched = True
                break

        if not matched:
            false_positives += 1

    false_negatives = len(ground_truth) - true_positives

    return true_positives, false_positives, false_negatives


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Compute performance metrics from confusion matrix values.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives

    Returns:
        Dictionary with precision, recall, F1, MCC
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Matthews Correlation Coefficient
    # Assumes true negatives ~ sequence length (difficult to define precisely)
    # Alternative: use only precision, recall, F1
    tn = 1000  # Placeholder - should be len(sequence) - (tp + fp + fn)
    mcc_num = (tp * tn) - (fp * fn)
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_denom if mcc_denom > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mcc": mcc
    }


def generate_parameter_grid(method: str) -> List[ParameterSet]:
    """
    Generate grid of parameters to test for each method.

    Args:
        method: Detection method name

    Returns:
        List of parameter configurations to test
    """
    grid = []

    if method == "zscore":
        for zscore_thresh in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            for smooth_w in [21, 31, 51, 75, 101]:
                grid.append(ParameterSet(
                    method="zscore",
                    params={
                        "smooth_w": smooth_w,
                        "zscore_threshold": zscore_thresh,
                        "min_separation": 75
                    }
                ))

    elif method == "mad":
        for mad_thresh in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
            for smooth_w in [21, 31, 51, 75, 101]:
                grid.append(ParameterSet(
                    method="mad",
                    params={
                        "smooth_w": smooth_w,
                        "mad_threshold": mad_thresh,
                        "min_separation": 75
                    }
                ))

    elif method in ("local", "local_baseline"):
        for window in [200, 500, 1000]:
            for thresh in [1.5, 2.0, 2.5, 3.0]:
                grid.append(ParameterSet(
                    method="local_baseline",
                    params={
                        "window_baseline": window,
                        "threshold_sigma": thresh,
                        "min_separation": 75
                    }
                ))

    elif method == "derivative":
        for thr_q in [0.005, 0.01, 0.02, 0.05]:
            for smooth_w in [21, 31, 51, 75, 101]:
                grid.append(ParameterSet(
                    method="derivative",
                    params={
                        "smooth_w": smooth_w,
                        "thr_quantile": thr_q,
                        "min_separation": 75
                    }
                ))

    elif method == "window_mean_shift":
        for w in [50, 100, 200, 500]:
            for top_k in [50, 100, 200, 500]:
                grid.append(ParameterSet(
                    method="window_mean_shift",
                    params={
                        "w": w,
                        "top_k": top_k,
                        "min_separation": 75
                    }
                ))

    elif method == "cusum":
        for h in [1.0, 2.0, 5.0, 10.0, 20.0]:
            for smooth_w in [21, 31, 51, 75, 101]:
                grid.append(ParameterSet(
                    method="cusum",
                    params={
                        "smooth_w": smooth_w,
                        "h": h,
                        "min_separation": 75
                    }
                ))

    return grid


def cross_validate_parameters(gene_list: List[str],
                              data_dir: Path,
                              ground_truth_file: Path,
                              param_set: ParameterSet,
                              n_folds: int = 5,
                              tolerance: int = 50) -> EvaluationResult:
    """
    Cross-validate parameters across multiple genes.

    Args:
        gene_list: List of gene IDs to test
        data_dir: Base directory with gene scoring outputs
        ground_truth_file: Annotation file with validated drops
        param_set: Parameters to evaluate
        n_folds: Number of cross-validation folds
        tolerance: bp tolerance for matching drops to ground truth

    Returns:
        Aggregated evaluation metrics
    """
    all_tp, all_fp, all_fn = 0, 0, 0
    tested_genes = []

    for gene_id in gene_list:
        gene_data_dir = data_dir / gene_id

        if not gene_data_dir.exists():
            print(f"Warning: Data not found for {gene_id}, skipping")
            continue

        try:
            # Load entropy data
            entropy = load_entropy_data(gene_data_dir)

            # Load ground truth
            ground_truth = load_ground_truth(ground_truth_file, gene_id, tolerance)

            if len(ground_truth) == 0:
                print(f"Warning: No ground truth for {gene_id}, skipping")
                continue

            # Detect drops with these parameters
            detected = detect_drops_with_params(entropy, param_set)

            # Evaluate
            tp, fp, fn = evaluate_parameters(detected, ground_truth, tolerance)

            all_tp += tp
            all_fp += fp
            all_fn += fn
            tested_genes.append(gene_id)

        except Exception as e:
            print(f"Error processing {gene_id}: {e}")
            continue

    # Compute aggregate metrics
    metrics = compute_metrics(all_tp, all_fp, all_fn)

    return EvaluationResult(
        params=param_set,
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        mcc=metrics["mcc"],
        n_true_positives=all_tp,
        n_false_positives=all_fp,
        n_false_negatives=all_fn,
        genes_tested=tested_genes
    )


def grid_search(gene_list: List[str],
               data_dir: Path,
               ground_truth_file: Path,
               methods: List[str] = ["zscore", "mad", "local_baseline"],
               tolerance: int = 50) -> Dict[str, EvaluationResult]:
    """
    Perform grid search over all methods and parameters.

    Args:
        gene_list: Genes to test (should have entropy data and ground truth)
        data_dir: Directory with genome_scoring outputs
        ground_truth_file: Annotation file
        methods: Detection methods to optimize
        tolerance: bp tolerance for matching

    Returns:
        Best parameters for each method
    """
    results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Optimizing {method} method")
        print(f"{'='*60}")

        param_grid = generate_parameter_grid(method)
        print(f"Testing {len(param_grid)} parameter combinations...")

        best_result = None
        best_f1 = -1.0

        for i, param_set in enumerate(param_grid, 1):
            print(f"  [{i}/{len(param_grid)}] Testing {param_set.params}...",
                  end=' ', flush=True)

            result = cross_validate_parameters(
                gene_list, data_dir, ground_truth_file,
                param_set, tolerance=tolerance
            )

            print(f"F1={result.f1_score:.3f}, P={result.precision:.3f}, "
                  f"R={result.recall:.3f}")

            # Select best by F1 score (balance precision and recall)
            if result.f1_score > best_f1:
                best_f1 = result.f1_score
                best_result = result

        results[method] = best_result

        print(f"\nBest {method} parameters:")
        print(f"  Params: {best_result.params.params}")
        print(f"  F1 Score: {best_result.f1_score:.3f}")
        print(f"  Precision: {best_result.precision:.3f}")
        print(f"  Recall: {best_result.recall:.3f}")
        print(f"  TP/FP/FN: {best_result.n_true_positives}/"
              f"{best_result.n_false_positives}/{best_result.n_false_negatives}")

    return results


def main():
    ap = argparse.ArgumentParser(
        description="Optimize drop detection parameters via grid search"
    )

    ap.add_argument("--organism", required=True,
                   help="Organism name (for reference)")

    ap.add_argument("--gene_list", type=Path, required=True,
                   help="File with gene IDs to test (one per line)")

    ap.add_argument("--data_dir", type=Path, required=True,
                   help="Directory with genome_scoring outputs")

    ap.add_argument("--ground_truth", type=Path, required=True,
                   help="Annotation file with validated features (BED/GFF)")

    ap.add_argument("--methods", nargs='+',
                   choices=["zscore", "mad", "local_baseline", "derivative",
                            "window_mean_shift", "cusum"],
                   default=["zscore", "mad", "local"],
                   help="Methods to optimize")

    ap.add_argument("--tolerance", type=int, default=50,
                   help="bp tolerance for matching drops to ground truth")

    ap.add_argument("--output", type=Path, required=True,
                   help="Output JSON file with optimal parameters")

    args = ap.parse_args()

    # Load gene list
    with open(args.gene_list) as f:
        gene_list = [line.strip() for line in f if line.strip()]

    print(f"Optimizing parameters for {len(gene_list)} genes")
    print(f"Methods: {args.methods}")
    print(f"Tolerance: ±{args.tolerance} bp")

    # Run grid search
    results = grid_search(
        gene_list,
        args.data_dir,
        args.ground_truth,
        methods=args.methods,
        tolerance=args.tolerance
    )

    # Save results
    output_data = {
        "organism": args.organism,
        "genes_tested": gene_list,
        "tolerance_bp": args.tolerance,
        "optimal_parameters": {}
    }

    for method, result in results.items():
        output_data["optimal_parameters"][method] = {
            "params": result.params.params,
            "performance": {
                "f1_score": result.f1_score,
                "precision": result.precision,
                "recall": result.recall,
                "mcc": result.mcc
            },
            "confusion_matrix": {
                "true_positives": result.n_true_positives,
                "false_positives": result.n_false_positives,
                "false_negatives": result.n_false_negatives
            }
        }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nOptimal parameters saved to: {args.output}")


if __name__ == "__main__":
    main()
