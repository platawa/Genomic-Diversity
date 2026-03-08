

"""
genome_scoring_jan22.py (Updated Jan 22, 2026)

Multi-organism genome scoring with Evo2 language model.

Supported organisms:
- human    : Homo sapiens (GRCh38)
- bacillus : Bacillus subtilis (ASM904v1)
- ecoli    : Escherichia coli K-12 (ASM584v2)

Features:
- Robust logits extraction for tuple/nested model outputs.
- Chunk scoring with overlap (context stability, fewer boundary artifacts).
- Entropy units: nats or bits.
- Plot styles: plain or evodesigner-like fill.
- Provenance metadata JSON.
- FASTA exports: locus oriented + exon records oriented.
- Plot suite: raw, smooth, boundaries, drops per method, zooms.

Usage:
    python genome_scoring_jan22.py --organism human --transcript_id NM_000546.6
    python genome_scoring_jan22.py --organism ecoli --gene_id b0001
    python genome_scoring_jan22.py --organism bacillus --pick15
    python genome_scoring_jan22.py --list_organisms
"""

import os
import math
import json
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

from Bio import SeqIO
from Bio.Seq import Seq

from evo2 import Evo2

# -----------------------
# DEFAULT PATHS / CONFIG
# -----------------------

# Organism-specific configurations
ORGANISM_CONFIG = {
    "human": {
        "fasta": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna",
        "gtf": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf",
        "out_dir": "/orcd/data/zhang_f/001/platawa/execution_files/logs_gene_locus_exon_aligned/human",
        "buffer_bp": 5000,
        "description": "Homo sapiens (GRCh38)",
    },
    "bacillus": {
        "fasta": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/GCF_000009045.1_ASM904v1_genomic.fna",
        "gtf": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf",
        "out_dir": "/orcd/data/zhang_f/001/platawa/execution_files/logs_gene_locus_exon_aligned/bacillus",
        "buffer_bp": 1000,  # smaller genome, smaller buffer
        "description": "Bacillus subtilis (ASM904v1)",
    },
    "ecoli": {
        "fasta": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna",
        "gtf": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf",
        "out_dir": "/orcd/data/zhang_f/001/platawa/execution_files/logs_gene_locus_exon_aligned/ecoli",
        "buffer_bp": 1000,  # smaller genome, smaller buffer
        "description": "Escherichia coli K-12 (ASM584v2)",
    },
}

# Default organism
DEFAULT_ORGANISM = "human"

# Legacy defaults (for backward compatibility)
FASTA_PATH_DEFAULT = ORGANISM_CONFIG[DEFAULT_ORGANISM]["fasta"]
GTF_PATH_DEFAULT   = ORGANISM_CONFIG[DEFAULT_ORGANISM]["gtf"]
OUT_DIR_DEFAULT    = ORGANISM_CONFIG[DEFAULT_ORGANISM]["out_dir"]

BUFFER_BP_DEFAULT = 5000

# Chunking defaults
MAX_CHUNK_LEN_DEFAULT = 15000
CHUNK_OVERLAP_DEFAULT = 1024

# Drop detection defaults
DROP_SMOOTH_W = 51
DROP_DERIV_Q = 0.01
DROP_SHIFT_W = 200
DROP_SHIFT_TOPK = 20
DROP_CUSUM_H = 1.0

# Plot zoom defaults
ZOOM_BP_DEFAULT = 1000
MAX_ZOOM_PLOTS_DEFAULT = 60  # safety cap

EPS = 1e-12


# -----------------------
# GTF PARSING
# -----------------------
def parse_gtf_attributes(attr_str: str) -> Dict[str, str]:
    out = {}
    for item in attr_str.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(" ", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        val = parts[1].strip().strip('"')
        out[key] = val
    return out


def load_exons_from_gtf(
    gtf_path: str,
    gene_id: Optional[str] = None,
    transcript_id: Optional[str] = None,
) -> Tuple[str, str, List[Tuple[int, int]], Optional[str], Dict[str, str]]:
    """
    Return (chrom, strand, exons_endexcl_1based, gene_name, meta_attrs)

    Exons are merged, 1-based end-exclusive intervals (start, end_excl).

    For bacterial genomes (which lack exon annotations), this function will
    fall back to CDS records, then to gene records if no exons/CDS found.
    """
    assert (gene_id is not None) or (transcript_id is not None), "Provide gene_id or transcript_id."

    # Try feature types in order of preference: exon > CDS > gene
    # Bacteria don't have introns, so they use CDS or gene records
    feature_priority = ["exon", "CDS", "gene"]

    for target_feature in feature_priority:
        chrom = None
        strand = None
        gene_name: Optional[str] = None
        exons: List[Tuple[int, int]] = []
        meta: Dict[str, str] = {}

        with open(gtf_path, "r") as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 9:
                    continue

                seqname, source, feature, start, end, score, st, frame, attrs = fields
                if feature != target_feature:
                    continue

                attrs_d = parse_gtf_attributes(attrs)

                if transcript_id is not None:
                    if attrs_d.get("transcript_id") != transcript_id:
                        continue
                else:
                    if attrs_d.get("gene_id") != gene_id:
                        continue

                if gene_name is None:
                    gene_name = attrs_d.get("gene_name", attrs_d.get("gene", None))

                # capture some useful attrs for provenance
                for k in ("gene_id", "transcript_id", "gene_name", "gene", "Name", "Dbxref"):
                    if k in attrs_d and k not in meta:
                        meta[k] = attrs_d[k]

                s = int(start)          # 1-based inclusive
                e_incl = int(end)       # 1-based inclusive
                e = e_incl + 1          # end-exclusive

                if chrom is None:
                    chrom = seqname
                if strand is None:
                    strand = st

                if seqname != chrom:
                    raise ValueError(f"Multiple chromosomes for locus: {chrom} vs {seqname}")
                if st != strand:
                    raise ValueError(f"Multiple strands for locus: {strand} vs {st}")

                exons.append((s, e))

        # If we found records with this feature type, use them
        if chrom is not None and strand is not None and exons:
            print(f"[INFO] Found {len(exons)} region(s) using '{target_feature}' feature type")
            break

    if chrom is None or strand is None or not exons:
        raise ValueError("No exons/CDS/gene records found. Check IDs or the GTF.")

    # sort + merge overlaps
    exons.sort()
    merged = []
    for s, e in exons:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    return chrom, strand, [(a, b) for a, b in merged], gene_name, meta


def exon_bounds(exons: List[Tuple[int, int]]) -> Tuple[int, int]:
    return min(s for s, _ in exons), max(e for _, e in exons)


# -----------------------
# FASTA HELPERS
# -----------------------
def fetch_chrom_sequence(fasta_path: str, target_chrom: str) -> str:
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.id == target_chrom:
            return str(record.seq).upper()
    raise ValueError(f"Chromosome {target_chrom} not found in FASTA.")


def slice_locus(seq_chr: str, start_1based: int, end_excl_1based: int) -> str:
    if start_1based < 1:
        start_1based = 1
    if end_excl_1based > len(seq_chr) + 1:
        end_excl_1based = len(seq_chr) + 1
    s0 = start_1based - 1
    e0 = end_excl_1based - 1
    return seq_chr[s0:e0]


def write_fasta(path: str, header: str, seq: str, wrap: int = 60) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f">{header}\n")
        for i in range(0, len(seq), wrap):
            f.write(seq[i:i+wrap] + "\n")


# -----------------------
# EXON LABEL OVERLAY
# -----------------------
def build_exon_labels_genomic_order(
    locus_start_1based: int,
    locus_len: int,
    exons_endexcl_1based: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    exon_id = np.full((locus_len,), -1, dtype=np.int32)
    for k, (s, e) in enumerate(exons_endexcl_1based, start=1):
        lo = max(s, locus_start_1based)
        hi = min(e, locus_start_1based + locus_len)
        if hi <= lo:
            continue
        exon_id[(lo - locus_start_1based):(hi - locus_start_1based)] = k
    is_exon = (exon_id != -1).astype(np.int32)
    return is_exon, exon_id


def build_boundary_distance_fields(is_exon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L = len(is_exon)
    prev = np.concatenate(([0], is_exon[:-1]))
    nxt  = np.concatenate((is_exon[1:], [0]))
    exon_starts = np.where((is_exon == 1) & (prev == 0))[0]
    exon_ends   = np.where((is_exon == 1) & (nxt  == 0))[0]

    def dist_to_points(points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.full((L,), np.nan, dtype=np.float32)
        out = np.empty((L,), dtype=np.float32)
        for i in range(L):
            out[i] = float(np.min(np.abs(points - i)))
        return out

    return dist_to_points(exon_starts), dist_to_points(exon_ends)


def get_exon_intervals_oriented(is_exon: np.ndarray, exon_id: Optional[np.ndarray] = None) -> List[Tuple[int, int, int]]:
    """
    Return list of (start, end, exon_id) in oriented coords.
    If exon_id not provided, uses consecutive labeling 1..K.
    """
    L = len(is_exon)
    intervals: List[Tuple[int, int, int]] = []
    i = 0
    k = 0
    while i < L:
        if is_exon[i] == 1:
            j = i + 1
            while j < L and is_exon[j] == 1:
                j += 1
            if exon_id is not None:
                eid = int(exon_id[i])
                if eid < 0:
                    k += 1
                    eid = k
            else:
                k += 1
                eid = k
            intervals.append((i, j, eid))
            i = j
        else:
            i += 1
    return intervals


def get_exon_boundaries_oriented(is_exon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    prev = np.concatenate(([0], is_exon[:-1]))
    nxt  = np.concatenate((is_exon[1:], [0]))
    exon_starts = np.where((is_exon == 1) & (prev == 0))[0]
    exon_ends   = np.where((is_exon == 1) & (nxt  == 0))[0]
    return exon_starts, exon_ends


# -----------------------
# TOKEN / MODEL UTILITIES
# -----------------------
def _bos_id(tok):
    bid = getattr(tok, "bos_id", getattr(tok, "eod_id", None))
    if bid is None:
        raise AssertionError("Tokenizer must provide bos_id or eod_id.")
    return bid


def _extract_logits(model_out):
    """
    Robust recursive extraction of logits tensor from nested outputs.
    Prefers 3D tensor [B, T, V] when available.
    """

    def walk(x):
        if isinstance(x, torch.Tensor):
            yield x
            return

        lg = getattr(x, "logits", None)
        if isinstance(lg, torch.Tensor):
            yield lg

        if isinstance(x, dict):
            if isinstance(x.get("logits"), torch.Tensor):
                yield x["logits"]
            for v in x.values():
                yield from walk(v)
            return

        if isinstance(x, (tuple, list)):
            for v in x:
                yield from walk(v)
            return

    candidates = list(walk(model_out))
    if not candidates:
        raise TypeError(f"Could not extract logits from model output of type {type(model_out)}")

    for t in candidates:
        if t.ndim == 3:
            return t
    return candidates[0]


def id_to_token_str(tok, idx: int) -> str:
    for attr in ("id_to_token", "decode", "detokenize", "convert_ids_to_tokens"):
        fn = getattr(tok, attr, None)
        if callable(fn):
            try:
                out = fn([idx]) if attr in ("decode", "detokenize", "convert_ids_to_tokens") else fn(idx)
                return out[0] if isinstance(out, (list, tuple)) else str(out)
            except Exception:
                pass
    return str(idx)


# -----------------------
# ENTROPY (REFERENCE DEFINITION)
# -----------------------
@torch.inference_mode()
def entropy_like_reference_acgt(
    sequence: str,
    model: Evo2,
    ACGT_IDS: torch.Tensor,
    device: str,
    prepend_bos: bool = True,
    reverse_complement: bool = False,
):
    """
    Matches your older genome-wide scripts:
    - logits at each position -> entropy over A/C/G/T renormalized
    - drop BOS position if prepend_bos
    - optionally compute RC and average (after flipping)
    """
    tok = model.tokenizer
    toks = tok.tokenize(sequence)
    if prepend_bos:
        toks = [_bos_id(tok)] + toks

    input_ids = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
    out = model(input_ids)
    logits = _extract_logits(out)  # [1, T, V] (ideally)

    logits_sub = logits.index_select(-1, ACGT_IDS)  # [1, T, 4]
    logZ = torch.logsumexp(logits_sub, dim=-1, keepdim=True)
    logp = logits_sub - logZ
    H_fwd = -(logp.exp() * logp).sum(dim=-1)  # [1, T]

    if prepend_bos:
        H_fwd = H_fwd[:, 1:]
    H_fwd = H_fwd.squeeze(0).detach().cpu()

    if not reverse_complement:
        H_final = H_fwd
    else:
        seq_rc = str(Seq(sequence).reverse_complement())
        toks_rc = tok.tokenize(seq_rc)
        if prepend_bos:
            toks_rc = [_bos_id(tok)] + toks_rc
        input_ids_rc = torch.tensor(toks_rc, dtype=torch.long, device=device).unsqueeze(0)
        out_rc = model(input_ids_rc)
        logits_rc = _extract_logits(out_rc)

        logits_rc_sub = logits_rc.index_select(-1, ACGT_IDS)
        logZ_rc = torch.logsumexp(logits_rc_sub, dim=-1, keepdim=True)
        logp_rc = logits_rc_sub - logZ_rc
        H_rc = -(logp_rc.exp() * logp_rc).sum(dim=-1)

        if prepend_bos:
            H_rc = H_rc[:, 1:]
        H_rc = H_rc.squeeze(0).detach().cpu()

        H_rc = torch.flip(H_rc, dims=[0])
        H_final = 0.5 * (H_fwd + H_rc)

    PPX = H_final.exp()
    return H_final, PPX


# -----------------------
# NEXT TOKEN (for TSV)
# -----------------------
@torch.inference_mode()
def next_token_logprobs_and_targets_aligned(sequence: str, model: Evo2, device: str):
    tok = model.tokenizer
    toks = [_bos_id(tok)] + tok.tokenize(sequence)
    input_ids = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)

    out = model(input_ids)
    logits = _extract_logits(out).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    logprobs_next = logprobs[:, :-1, :]   # [1, L, V]
    target_next = input_ids[:, 1:]        # [1, L]
    return logprobs_next.squeeze(0), target_next.squeeze(0)


def next_token_probs_subset(logprobs_next: torch.Tensor, subset_ids: torch.Tensor) -> torch.Tensor:
    return logprobs_next.exp().index_select(-1, subset_ids)


# -----------------------
# LOCUS-ALIGNED SCORING (OVERLAP CHUNKS + N SAFE)
# -----------------------
def score_locus_aligned_overlap(
    seq_oriented: str,
    evo2_model: Evo2,
    ACGT_IDS: torch.Tensor,
    device: str,
    max_chunk_len: int,
    chunk_overlap: int,
    compute_rcavg_entropy: bool = True,
):
    """
    Scores full locus in oriented coords using overlapping chunks.
    Writes results only in each chunk's core region to avoid edge artifacts.
    Keeps exact alignment: output arrays length L, Ns remain NaN.
    """
    L = len(seq_oriented)

    entropy_fwd = np.full((L,), np.nan, dtype=np.float32)
    ppx_fwd     = np.full((L,), np.nan, dtype=np.float32)
    entropy_rc  = np.full((L,), np.nan, dtype=np.float32)
    ppx_rc      = np.full((L,), np.nan, dtype=np.float32)

    p4       = np.full((L, 4), np.nan, dtype=np.float32)
    ll_next  = np.full((L,), np.nan, dtype=np.float32)
    true_tok = np.array([""] * L, dtype=object)

    if L == 0:
        return entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next

    step = max(1, max_chunk_len - chunk_overlap)
    starts = list(range(0, L, step))

    for s in starts:
        e = min(L, s + max_chunk_len)

        core_s = s if s == 0 else s + chunk_overlap // 2
        core_e = e if e == L else e - chunk_overlap // 2
        core_s = min(core_s, core_e)

        chunk_seq = seq_oriented[s:e]

        # score contiguous non-N runs in this chunk
        i = 0
        while i < len(chunk_seq):
            if chunk_seq[i] == "N":
                i += 1
                continue
            j = i + 1
            while j < len(chunk_seq) and chunk_seq[j] != "N":
                j += 1

            run_seq = chunk_seq[i:j]

            # entropy forward
            Hf_t, Pf_t = entropy_like_reference_acgt(
                run_seq, evo2_model, ACGT_IDS, device,
                prepend_bos=True, reverse_complement=False
            )
            Hf = Hf_t.float().numpy().astype(np.float32)
            Pf = Pf_t.float().numpy().astype(np.float32)

            # entropy rcavg
            if compute_rcavg_entropy:
                Hr_t, Pr_t = entropy_like_reference_acgt(
                    run_seq, evo2_model, ACGT_IDS, device,
                    prepend_bos=True, reverse_complement=True
                )
                Hr = Hr_t.float().numpy().astype(np.float32)
                Pr = Pr_t.float().numpy().astype(np.float32)
            else:
                Hr = Pr = None

            # next-token quantities (forward-only)
            logprobs_next, target_next = next_token_logprobs_and_targets_aligned(run_seq, evo2_model, device)

            ll = (
                logprobs_next.float()
                .gather(-1, target_next.unsqueeze(-1))
                .squeeze(-1)
                .detach().cpu().numpy().astype(np.float32)
            )
            p4_run = (
                next_token_probs_subset(logprobs_next.float(), ACGT_IDS)
                .detach().cpu().numpy().astype(np.float32)
            )
            tok = evo2_model.tokenizer
            target_ids = target_next.detach().cpu().tolist()
            true_run = [id_to_token_str(tok, int(tid)) for tid in target_ids]

            # write back run results into locus arrays, but only for core region
            for k in range(i, j):
                g = s + k
                if g < core_s or g >= core_e:
                    continue
                rk = k - i
                entropy_fwd[g] = Hf[rk]
                ppx_fwd[g] = Pf[rk]
                if compute_rcavg_entropy:
                    entropy_rc[g] = Hr[rk]
                    ppx_rc[g] = Pr[rk]
                p4[g, :] = p4_run[rk, :]
                ll_next[g] = ll[rk]
                true_tok[g] = true_run[rk]

            i = j

    return entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next


# -----------------------
# DROP DETECTION METHODS
# -----------------------
def _fill_nans_linear(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    isn = np.isnan(y)
    if not np.any(isn):
        return y
    idx = np.arange(len(y))
    good = ~isn
    if good.sum() >= 2:
        y[isn] = np.interp(idx[isn], idx[good], y[good])
    elif good.sum() == 1:
        y[isn] = y[good][0]
    return y


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    y = _fill_nans_linear(x)
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(y, kernel, mode="same")


def detect_drops_derivative(entropy: np.ndarray, smooth_w: int, thr_quantile: float) -> List[int]:
    sm = _rolling_mean(entropy, smooth_w)
    d = np.diff(sm, prepend=sm[0])
    thr = np.quantile(d, thr_quantile)
    candidates = np.where(d <= thr)[0].tolist()

    out = []
    last = -10**9
    min_sep = max(10, smooth_w // 2)
    for i in candidates:
        if i - last >= min_sep:
            out.append(i)
            last = i
    return out


def detect_drops_window_mean_shift(entropy: np.ndarray, w: int, top_k: int) -> List[int]:
    x = _fill_nans_linear(entropy)
    L = len(x)
    scores = np.full((L,), np.nan, dtype=np.float32)

    min_len = max(5, w // 10)
    for i in range(L):
        a0, a1 = max(0, i - w), i
        b0, b1 = i, min(L, i + w)
        if (a1 - a0) < min_len or (b1 - b0) < min_len:
            continue
        scores[i] = float(np.mean(x[b0:b1]) - np.mean(x[a0:a1]))

    good = ~np.isnan(scores)
    if good.sum() == 0:
        return []

    idx_good = np.where(good)[0]
    order = np.argsort(scores[good])
    picks = idx_good[order][:top_k].tolist()

    out = []
    for i in picks:
        if all(abs(i - j) > w // 2 for j in out):
            out.append(i)
    return out


def detect_drops_cusum(entropy: np.ndarray, smooth_w: int, h: float) -> List[int]:
    x = _rolling_mean(entropy, smooth_w)
    x = _fill_nans_linear(x)
    mu = float(np.mean(x))

    out = []
    s = 0.0
    last = -10**9
    min_sep = max(25, smooth_w)

    for i, xi in enumerate(x):
        s = max(0.0, s + (mu - float(xi)))
        if s > h and (i - last) > min_sep:
            out.append(i)
            last = i
            s = 0.0
    return out


# -----------------------
# PLOTTING (DECOMPOSED SUITE)
# -----------------------
def shade_exons(ax, exon_intervals: List[Tuple[int, int, int]], alpha: float = 0.12) -> None:
    for (s, e, _) in exon_intervals:
        ax.axvspan(s, e, alpha=alpha)


def evodesigner_fill(ax, x, y, low_quantile: float = 0.10) -> None:
    # Grey fill everywhere
    ax.fill_between(x, y, 0, alpha=0.18)
    # Highlight low-entropy positions
    if np.any(~np.isnan(y)):
        thr = np.nanquantile(y, low_quantile)
        mask = y <= thr
        ax.fill_between(x, y, 0, where=mask, alpha=0.35)


def _save_fig(path: str, dpi: int = 200) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_suite(
    out_prefix: str,
    entropy_main: np.ndarray,
    is_exon: np.ndarray,
    drop_points: Dict[str, List[int]],
    title_prefix: str,
    smooth_w: int = 51,
    zoom_bp: int = 0,
    max_zoom_plots: int = 60,
    plot_style: str = "plain",   # plain | evodesigner
    unit: str = "nats",
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    x = np.arange(len(entropy_main))
    exon_intervals = get_exon_intervals_oriented(is_exon)
    exon_starts, exon_ends = get_exon_boundaries_oriented(is_exon)
    sm = _rolling_mean(entropy_main, smooth_w)

    def maybe_style(ax, xx, yy):
        if plot_style == "evodesigner":
            evodesigner_fill(ax, xx, yy, low_quantile=0.10)

    # 1) Raw + shading
    plt.figure(figsize=(16, 4))
    ax = plt.gca()
    shade_exons(ax, exon_intervals, alpha=0.12)
    ax.plot(x, entropy_main, linewidth=0.8, label="Entropy(main)")
    maybe_style(ax, x, entropy_main)
    ax.set_title(f"{title_prefix} | raw")
    ax.set_xlabel("OrientedIdx (5'→3')")
    ax.set_ylabel(f"Entropy ({unit})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=8)
    _save_fig(out_prefix + ".entropy_raw.png")

    # 2) Smoothed + shading + boundaries
    plt.figure(figsize=(16, 4))
    ax = plt.gca()
    shade_exons(ax, exon_intervals, alpha=0.12)
    ax.plot(x, sm, linewidth=1.2, label=f"Entropy(main) rolling_mean(w={smooth_w})")
    for s in exon_starts:
        ax.axvline(s, linestyle="--", linewidth=0.7, alpha=0.75)
    for e in exon_ends:
        ax.axvline(e, linestyle=":", linewidth=0.7, alpha=0.75)
    maybe_style(ax, x, sm)
    ax.set_title(f"{title_prefix} | smoothed + exon boundaries")
    ax.set_xlabel("OrientedIdx (5'→3')")
    ax.set_ylabel(f"Entropy ({unit})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=8)
    _save_fig(out_prefix + ".entropy_smooth.png")

    # 3) Boundary-only view
    plt.figure(figsize=(16, 4))
    ax = plt.gca()
    shade_exons(ax, exon_intervals, alpha=0.12)
    ax.plot(x, entropy_main, linewidth=0.8, label="Entropy(main)")
    for s in exon_starts:
        ax.axvline(s, linestyle="--", linewidth=0.8, alpha=0.85)
    for e in exon_ends:
        ax.axvline(e, linestyle=":", linewidth=0.8, alpha=0.85)
    maybe_style(ax, x, entropy_main)
    ax.set_title(f"{title_prefix} | boundary-only")
    ax.set_xlabel("OrientedIdx (5'→3')")
    ax.set_ylabel(f"Entropy ({unit})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=8)
    _save_fig(out_prefix + ".entropy_boundaries.png")

    # 4) One plot per drop method
    for method, pts in drop_points.items():
        plt.figure(figsize=(16, 4))
        ax = plt.gca()
        shade_exons(ax, exon_intervals, alpha=0.12)
        ax.plot(x, sm, linewidth=1.2, label="Smoothed entropy")
        maybe_style(ax, x, sm)
        if pts:
            ys = sm[pts]
            ax.scatter(pts, ys, s=22, label=f"drops:{method}")
        ax.set_title(f"{title_prefix} | drops={method}")
        ax.set_xlabel("OrientedIdx (5'→3')")
        ax.set_ylabel(f"Entropy ({unit})")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc="best", fontsize=8)
        _save_fig(out_prefix + f".drops_{method}.png")

    # 5) Optional zoom plots around each boundary
    if zoom_bp and zoom_bp > 0:
        boundaries = [(int(s), "start") for s in exon_starts] + [(int(e), "end") for e in exon_ends]
        boundaries.sort(key=lambda t: t[0])

        count = 0
        L = len(entropy_main)
        for idx, kind in boundaries:
            if count >= max_zoom_plots:
                break
            lo = max(0, idx - zoom_bp)
            hi = min(L, idx + zoom_bp)

            plt.figure(figsize=(14, 4))
            ax = plt.gca()

            for (s, e, _) in exon_intervals:
                ss = max(s, lo)
                ee = min(e, hi)
                if ee > ss:
                    ax.axvspan(ss, ee, alpha=0.15)

            xx = np.arange(lo, hi)
            yy = sm[lo:hi]
            ax.plot(xx, yy, linewidth=1.3, label="Smoothed entropy")
            maybe_style(ax, xx, yy)

            ax.axvline(idx, linestyle="--" if kind == "start" else ":", linewidth=1.2, alpha=0.9, label=f"exon_{kind}")

            ax.set_title(f"{title_prefix} | zoom {kind} @ {idx} (±{zoom_bp}bp)")
            ax.set_xlabel("OrientedIdx (5'→3')")
            ax.set_ylabel(f"Entropy ({unit})")
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.legend(loc="best", fontsize=8)
            _save_fig(out_prefix + f".zoom_{kind}_{idx}.png")
            count += 1


# -----------------------
# WINDOW SUMMARY
# -----------------------
def write_window_summary(out_path: str, entropy: np.ndarray, is_exon: np.ndarray, win: int = 200, step: int = 50):
    with open(out_path, "w") as f:
        f.write("WinStartOriented\tWinEndOriented\tMeanEntropy\tMeanEntropyExon\tMeanEntropyIntron\tFracExon\n")
        L = len(entropy)
        for s in range(0, L - win + 1, step):
            e = s + win
            ent_w = entropy[s:e]
            ex = is_exon[s:e].astype(bool)

            mean_all = float(np.nanmean(ent_w)) if np.any(~np.isnan(ent_w)) else np.nan
            mean_ex  = float(np.nanmean(ent_w[ex])) if np.any(ex) else np.nan
            mean_in  = float(np.nanmean(ent_w[~ex])) if np.any(~ex) else np.nan
            frac_ex  = float(np.mean(ex))

            f.write(f"{s}\t{e}\t{mean_all:.6f}\t{mean_ex:.6f}\t{mean_in:.6f}\t{frac_ex:.4f}\n")


# -----------------------
# MAIN LOCUS RUN
# -----------------------
def run_one_locus(
    fasta_path: str,
    gtf_path: str,
    out_dir: str,
    gene_id: Optional[str],
    transcript_id: Optional[str],
    buffer_bp: int,
    max_chunk_len: int,
    chunk_overlap: int,
    drop_on: str,        # "rcavg" or "fwd"
    entropy_unit: str,   # "nats" or "bits"
    plot_style: str,     # "plain" or "evodesigner"
    zoom_bp: int = 0,
    max_zoom_plots: int = 60,
):
    os.makedirs(out_dir, exist_ok=True)

    tag = transcript_id if transcript_id else gene_id
    if not tag:
        raise ValueError("Provide gene_id or transcript_id.")

    # 1) Load exons (+ gene_name + meta)
    chrom, strand, exons, gene_name, gtf_meta = load_exons_from_gtf(
        gtf_path, gene_id=gene_id, transcript_id=transcript_id
    )
    exon_start, exon_end_excl = exon_bounds(exons)

    # 2) Define locus with buffer
    locus_start = max(1, exon_start - buffer_bp)
    locus_end_excl = exon_end_excl + buffer_bp
    locus_len = locus_end_excl - locus_start

    print(f"[INFO] tag={tag}" + (f" | gene_name={gene_name}" if gene_name else ""))
    print(f"[INFO] chrom={chrom}, strand={strand}")
    print(f"[INFO] exon span  [{exon_start}, {exon_end_excl})")
    print(f"[INFO] locus span [{locus_start}, {locus_end_excl})  len={locus_len}  buffer={buffer_bp}")

    # 3) Fetch locus sequence genomic
    print("[INFO] Loading chromosome sequence from FASTA...")
    chr_seq = fetch_chrom_sequence(fasta_path, chrom)
    locus_seq_genomic = slice_locus(chr_seq, locus_start, locus_end_excl).upper()
    if len(locus_seq_genomic) != locus_len:
        raise RuntimeError("Locus slice length mismatch (check coordinate logic).")

    # 4) Exon labels genomic
    is_exon_g, exon_id_g = build_exon_labels_genomic_order(locus_start, locus_len, exons)

    # 5) Orient to 5'→3'
    if strand == "+":
        locus_seq = locus_seq_genomic
        is_exon = is_exon_g
        exon_id = exon_id_g
        pos = np.arange(locus_start, locus_end_excl, dtype=np.int64)
        map_str = "identity"
    elif strand == "-":
        locus_seq = str(Seq(locus_seq_genomic).reverse_complement())
        is_exon = is_exon_g[::-1].copy()
        exon_id = exon_id_g[::-1].copy()
        pos = np.arange(locus_end_excl - 1, locus_start - 1, -1, dtype=np.int64)
        map_str = "reverse_complement"
    else:
        raise ValueError(f"Unexpected strand: {strand}")

    dist_to_exon_start, dist_to_exon_end = build_boundary_distance_fields(is_exon)

    # Base name
    base_name = f"{tag}_{chrom}_{locus_start}_{locus_end_excl}_strand{strand}"
    out_prefix = os.path.join(out_dir, base_name)

    # 5b) Export FASTA
    locus_hdr = f"{tag}|{gene_name or 'NA'}|{chrom}:{locus_start}-{locus_end_excl}|strand={strand}|oriented_5to3|map={map_str}"
    write_fasta(out_prefix + ".locus_oriented.fa", locus_hdr, locus_seq)
    print("[INFO] Wrote locus FASTA:", out_prefix + ".locus_oriented.fa")

    exon_intervals = get_exon_intervals_oriented(is_exon, exon_id=exon_id)
    with open(out_prefix + ".exons_oriented.fa", "w") as f:
        for (s, e, eid) in exon_intervals:
            exon_seq = locus_seq[s:e]
            f.write(f">{tag}|{gene_name or 'NA'}|exon{eid}|orientedIdx={s}-{e}\n")
            for i in range(0, len(exon_seq), 60):
                f.write(exon_seq[i:i+60] + "\n")
    print("[INFO] Wrote exons FASTA:", out_prefix + ".exons_oriented.fa")

    # 6) Model init
    print("[INFO] Initializing Evo2 model...")
    evo2_model = Evo2("evo2_7b")
    device = str(getattr(evo2_model, "device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    if hasattr(evo2_model, "eval"):
        evo2_model.eval()
    print("[INFO] device:", device)

    # Token ids for A/C/G/T
    idx_A = evo2_model.tokenizer.tokenize("A")[0]
    idx_C = evo2_model.tokenizer.tokenize("C")[0]
    idx_G = evo2_model.tokenizer.tokenize("G")[0]
    idx_T = evo2_model.tokenizer.tokenize("T")[0]
    ACGT_IDS = torch.tensor([idx_A, idx_C, idx_G, idx_T], dtype=torch.long, device=device)

    # 7) Score locus aligned (overlap)
    print("[INFO] Scoring locus (aligned arrays, overlap chunks)...")
    entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next = score_locus_aligned_overlap(
        locus_seq,
        evo2_model,
        ACGT_IDS,
        device,
        max_chunk_len=max_chunk_len,
        chunk_overlap=chunk_overlap,
        compute_rcavg_entropy=True,
    )

    # Choose entropy for drop detection
    if drop_on == "rcavg":
        entropy_main = entropy_rc
        name_main = "Entropy_RCavg"
    else:
        entropy_main = entropy_fwd
        name_main = "Entropy_fwd"

    # Units conversion
    if entropy_unit == "bits":
        scale = 1.0 / math.log(2.0)
        entropy_fwd_u = entropy_fwd * scale
        entropy_rc_u  = entropy_rc  * scale
        entropy_main_u = entropy_main * scale
        unit = "bits"
        ylim = (0.0, 2.05)
    else:
        entropy_fwd_u = entropy_fwd
        entropy_rc_u  = entropy_rc
        entropy_main_u = entropy_main
        unit = "nats"
        ylim = (0.0, 1.45)

    # 8) Drop detection
    print(f"[INFO] Drop detection on: {name_main} ({unit})")
    drops = {
        "derivative": detect_drops_derivative(entropy_main_u, smooth_w=DROP_SMOOTH_W, thr_quantile=DROP_DERIV_Q),
        "win_shift":  detect_drops_window_mean_shift(entropy_main_u, w=DROP_SHIFT_W, top_k=DROP_SHIFT_TOPK),
        "cusum":      detect_drops_cusum(entropy_main_u, smooth_w=DROP_SMOOTH_W, h=DROP_CUSUM_H),
    }

    # 8b) Metadata JSON provenance
    meta_path = out_prefix + ".meta.json"
    meta = {
        "tag": tag,
        "gene_name": gene_name,
        "gtf_meta_attrs": gtf_meta,
        "chrom": chrom,
        "strand": strand,
        "exons_genomic_1based_endexcl": exons,
        "locus_genomic_1based_endexcl": [locus_start, locus_end_excl],
        "orientation_map": map_str,
        "buffer_bp": buffer_bp,
        "max_chunk_len": max_chunk_len,
        "chunk_overlap": chunk_overlap,
        "entropy_unit": unit,
        "drop_on": drop_on,
        "plot_style": plot_style,
        "exon_intervals_oriented_idx_endexcl": [(s, e, int(eid)) for (s, e, eid) in exon_intervals],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("[INFO] Wrote metadata:", meta_path)

    # 9) Write TSV (keep your old columns)
    out_tsv = out_prefix + ".tsv"
    print("[INFO] Writing TSV:", out_tsv)
    with open(out_tsv, "w") as f:
        f.write("Pos\tEntropy(nats)\tPerplexity(e)\tP(A)\tP(C)\tP(G)\tP(T)\tTrueToken\tLL_next(nats)")
        f.write("\tEntropy_RCavg(nats)\tPerplexity_RCavg(e)")
        f.write("\tBase\tOrientedIdx\tIsExon\tExonID\tDistToExonStart\tDistToExonEnd\n")

        for i in range(locus_len):
            ent = float(entropy_fwd[i]) if not np.isnan(entropy_fwd[i]) else np.nan
            px  = float(ppx_fwd[i]) if not np.isnan(ppx_fwd[i]) else np.nan
            ll  = float(ll_next[i]) if not np.isnan(ll_next[i]) else np.nan

            if not np.isnan(p4[i, 0]):
                a, c, g, t = p4[i, :].tolist()
            else:
                a = c = g = t = np.nan

            ent_rc = float(entropy_rc[i]) if not np.isnan(entropy_rc[i]) else np.nan
            px_rc  = float(ppx_rc[i]) if not np.isnan(ppx_rc[i]) else np.nan

            f.write(
                f"{int(pos[i])}\t"
                f"{ent:.6f}\t{px:.6f}\t"
                f"{a:.6f}\t{c:.6f}\t{g:.6f}\t{t:.6f}\t"
                f"{true_tok[i]}\t{ll:.6f}\t"
                f"{ent_rc:.6f}\t{px_rc:.6f}\t"
                f"{locus_seq[i]}\t{i}\t{int(is_exon[i])}\t{int(exon_id[i])}\t"
                f"{dist_to_exon_start[i]:.1f}\t{dist_to_exon_end[i]:.1f}\n"
            )

    # 10) Plot suite
    title_prefix = f"{tag} {chrom}:{locus_start}-{locus_end_excl} strand {strand} (5'→3') | drop_on={drop_on}"
    print("[INFO] Writing plot suite:", out_prefix + ".*.png")
    plot_suite(
        out_prefix=out_prefix,
        entropy_main=entropy_main_u,
        is_exon=is_exon,
        drop_points=drops,
        title_prefix=title_prefix,
        smooth_w=DROP_SMOOTH_W,
        zoom_bp=zoom_bp,
        max_zoom_plots=max_zoom_plots,
        plot_style=plot_style,
        unit=unit,
        ylim=ylim,
    )

    # 11) Drop point list
    out_drops = out_prefix + ".drops.txt"
    print("[INFO] Writing drop points:", out_drops)
    with open(out_drops, "w") as f:
        for k, pts in drops.items():
            f.write(f"{k}\t" + ",".join(map(str, pts)) + "\n")

    # 12) Window summary (use entropy_main in chosen units)
    out_summary = out_prefix + ".window_summary.tsv"
    print("[INFO] Writing window summary:", out_summary)
    write_window_summary(out_summary, entropy_main_u, is_exon, win=200, step=50)

    print("[DONE]")


# -----------------------
# PICK-15 HELPER (GTF ONLY)
# -----------------------
def parse_transcript_exons_from_gtf(gtf_path: str) -> Dict[str, Dict]:
    tx_map: Dict[str, Dict] = {}
    with open(gtf_path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            seqname, source, feature, start, end, score, st, frame, attrs = fields
            if feature != "exon":
                continue

            attrs_d = parse_gtf_attributes(attrs)
            tx_id = attrs_d.get("transcript_id")
            if tx_id is None:
                continue

            s = int(start)
            e = int(end) + 1

            rec = tx_map.get(tx_id)
            if rec is None:
                tx_map[tx_id] = {"chrom": seqname, "strand": st, "exons": [(s, e)]}
            else:
                if rec["chrom"] != seqname or rec["strand"] != st:
                    continue
                rec["exons"].append((s, e))

    for tx_id, rec in tx_map.items():
        exons = rec["exons"]
        exons.sort()
        merged = []
        for s, e in exons:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        exm = [(a, b) for a, b in merged]
        rec["exons"] = exm
        rec["exon_count"] = len(exm)
        mn = min(a for a, _ in exm)
        mx = max(b for _, b in exm)
        rec["exon_span"] = (mn, mx)
        rec["length_bp"] = mx - mn
    return tx_map


def pick_15_transcripts_recipe(gtf_path: str) -> List[str]:
    tx_map = parse_transcript_exons_from_gtf(gtf_path)
    items = [(tx, rec["chrom"], rec["strand"], rec["exon_count"], rec["length_bp"]) for tx, rec in tx_map.items()]
    items_valid = [x for x in items if x[3] >= 2 and x[4] >= 500]

    cat1 = [x for x in items_valid if (10000 <= x[4] <= 50000) and (5 <= x[3] <= 15)]
    cat1.sort(key=lambda z: (abs(z[4] - 25000), abs(z[3] - 10)))
    pick1 = [tx for tx, *_ in cat1[:5]]

    cat2 = [x for x in items_valid if x[4] >= 200000]
    cat2.sort(key=lambda z: (-z[3], -z[4]))
    pick2 = [tx for tx, *_ in cat2[:4]]

    cat3 = [x for x in items_valid if x[4] <= 5000 and x[3] <= 3]
    cat3.sort(key=lambda z: (z[4], z[3]))
    pick3 = [tx for tx, *_ in cat3[:3]]

    cat5 = [x for x in items_valid if x[4] >= 100000 and x[3] <= 3]
    cat5.sort(key=lambda z: (-z[4], z[3]))
    pick5 = [tx for tx, *_ in cat5[:2]]

    picks: List[str] = []
    for group in (pick1, pick2, pick3, pick5):
        for tx in group:
            if tx not in picks:
                picks.append(tx)
            if len(picks) >= 15:
                return picks

    remaining = [x for x in items_valid if x[0] not in picks]
    remaining.sort(key=lambda z: (-z[3], -z[4]))
    for tx, *_ in remaining:
        picks.append(tx)
        if len(picks) >= 15:
            break
    return picks[:15]


# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Genome scoring with Evo2 for multiple organisms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported organisms:
  human    - Homo sapiens (GRCh38)
  bacillus - Bacillus subtilis (ASM904v1)
  ecoli    - Escherichia coli K-12 (ASM584v2)

Examples:
  # Score a human transcript
  python genome_scoring_jan22.py --organism human --transcript_id NM_000546.6

  # Score an E. coli gene
  python genome_scoring_jan22.py --organism ecoli --gene_id b0001

  # List available transcripts for bacillus
  python genome_scoring_jan22.py --organism bacillus --pick15
        """
    )

    # Organism selection (new!)
    ap.add_argument("--organism", choices=list(ORGANISM_CONFIG.keys()), default=DEFAULT_ORGANISM,
                    help=f"Organism to analyze. Choices: {list(ORGANISM_CONFIG.keys())}. Default: {DEFAULT_ORGANISM}")
    ap.add_argument("--list_organisms", action="store_true",
                    help="List available organisms and their configurations.")

    # Path overrides (optional - will use organism defaults if not specified)
    ap.add_argument("--fasta", default=None,
                    help="Override FASTA path (default: use organism config)")
    ap.add_argument("--gtf", default=None,
                    help="Override GTF path (default: use organism config)")
    ap.add_argument("--out_dir", default=None,
                    help="Override output directory (default: use organism config)")

    ap.add_argument("--gene_id", default=None)
    ap.add_argument("--transcript_id", default=None)

    ap.add_argument("--buffer_bp", type=int, default=None,
                    help="Buffer bp around locus (default: organism-specific)")

    ap.add_argument("--max_chunk_len", type=int, default=MAX_CHUNK_LEN_DEFAULT)
    ap.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP_DEFAULT)

    ap.add_argument("--drop_on", choices=["rcavg", "fwd"], default="rcavg")

    ap.add_argument("--entropy_unit", choices=["nats", "bits"], default="bits")
    ap.add_argument("--plot_style", choices=["plain", "evodesigner"], default="evodesigner")

    ap.add_argument("--zoom_bp", type=int, default=ZOOM_BP_DEFAULT,
                    help="If >0, save zoom plots around exon boundaries (±zoom_bp).")
    ap.add_argument("--max_zoom_plots", type=int, default=MAX_ZOOM_PLOTS_DEFAULT,
                    help="Cap number of zoom plots.")

    ap.add_argument("--pick15", action="store_true", help="Print 15 transcript_ids selected from the GTF.")

    args = ap.parse_args()

    # Handle --list_organisms
    if args.list_organisms:
        print("\nAvailable organisms:\n")
        for org, cfg in ORGANISM_CONFIG.items():
            print(f"  {org}:")
            print(f"    Description: {cfg['description']}")
            print(f"    FASTA:       {cfg['fasta']}")
            print(f"    GTF:         {cfg['gtf']}")
            print(f"    Output dir:  {cfg['out_dir']}")
            print(f"    Buffer bp:   {cfg['buffer_bp']}")
            print()
        return

    # Get organism config
    org_cfg = ORGANISM_CONFIG[args.organism]
    print(f"[INFO] Organism: {args.organism} ({org_cfg['description']})")

    # Resolve paths: use explicit args if provided, else organism defaults
    fasta_path = args.fasta if args.fasta else org_cfg["fasta"]
    gtf_path = args.gtf if args.gtf else org_cfg["gtf"]
    out_dir = args.out_dir if args.out_dir else org_cfg["out_dir"]
    buffer_bp = args.buffer_bp if args.buffer_bp is not None else org_cfg["buffer_bp"]

    if args.pick15:
        txs = pick_15_transcripts_recipe(gtf_path)
        print(f"\n15 selected transcripts for {args.organism}:\n")
        print("\n".join(txs))
        return

    if (args.gene_id is None) and (args.transcript_id is None):
        raise SystemExit("Provide --gene_id or --transcript_id, or use --pick15")

    run_one_locus(
        fasta_path=fasta_path,
        gtf_path=gtf_path,
        out_dir=out_dir,
        gene_id=args.gene_id,
        transcript_id=args.transcript_id,
        buffer_bp=buffer_bp,
        max_chunk_len=args.max_chunk_len,
        chunk_overlap=args.chunk_overlap,
        drop_on=args.drop_on,
        entropy_unit=args.entropy_unit,
        plot_style=args.plot_style,
        zoom_bp=args.zoom_bp,
        max_zoom_plots=args.max_zoom_plots,
    )


if __name__ == "__main__":
    main()











# """
# genome_scoring_jan8.py (Updated Jan 2026)

# Core locus-centric scorer for exon-boundary signal mining.

# Given a gene_id or transcript_id:
# 1) Parse exon intervals from GTF (1-based inclusive -> 1-based end-exclusive).
# 2) Define locus = [min_exon_start-buffer, max_exon_end+buffer).
# 3) Fetch locus from genome FASTA.
# 4) Orient locus 5'->3' (reverse-complement if strand == '-') and reverse annotations.
# 5) Score per-position with STRONG alignment (arrays length = locus length; N positions remain NaN):
#    - Forward-only: Entropy(nats), Perplexity(e), P(A/C/G/T), TrueToken, LL_next(nats)
#    - RC-averaged: Entropy_RCavg(nats), Perplexity_RCavg(e)  (matches genome-wide scripts exactly)
# 6) Overlay exon labels: IsExon, ExonID, distances to exon start/end.
# 7) Run drop/change-point detection on entropy (default: RC-avg).
# 8) Save TSV + drop-point list + window summary.
# 9) Write plot SUITE (decomposed plots):
#    - raw entropy(main) with exon shading
#    - smoothed entropy(main) with exon shading + boundaries
#    - boundary-only view
#    - one plot per drop method (derivative / win_shift / cusum)
#    - optional zoom plots around each exon boundary
# 10) Export FASTA:
#    - full locus sequence oriented 5'->3'
#    - exon-only sequences (each merged exon as a record) in oriented coords

# Notes:
# - Next-token quantities (P(A/C/G/T), TrueToken, LL_next) are forward-only by definition.
# - RC-avg is applied only to entropy/perplexity, identical to genome_scoring_parse.py / gene_parse.py.
# - Exons are parsed from the same GTF you provide; correctness depends on using the matching annotation set.
# """

# import os
# import math
# import argparse
# from typing import List, Tuple, Optional, Dict

# import numpy as np
# import torch
# import matplotlib.pyplot as plt

# from Bio import SeqIO
# from Bio.Seq import Seq

# from evo2 import Evo2


# # -----------------------
# # DEFAULT PATHS / CONFIG
# # -----------------------
# FASTA_PATH_DEFAULT = "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna"
# GTF_PATH_DEFAULT   = "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf"
# OUT_DIR_DEFAULT    = "/orcd/data/zhang_f/001/platawa/execution_files/logs_gene_locus_exon_aligned"

# BUFFER_BP_DEFAULT = 5000
# MAX_CHUNK_LEN_DEFAULT = 15000

# # Drop detection defaults
# DROP_SMOOTH_W = 51
# DROP_DERIV_Q = 0.01
# DROP_SHIFT_W = 200
# DROP_SHIFT_TOPK = 20
# DROP_CUSUM_H = 1.0

# # Plot zoom defaults
# ZOOM_BP_DEFAULT = 1000
# MAX_ZOOM_PLOTS_DEFAULT = 60  # safety cap

# EPS = 1e-12


# # -----------------------
# # GTF PARSING
# # -----------------------
# def parse_gtf_attributes(attr_str: str) -> Dict[str, str]:
#     """
#     Parse the 9th GTF field into a dict.
#     Typical: gene_id "ENSG..."; transcript_id "ENST..."; gene_name "TP53";
#     """
#     out = {}
#     for item in attr_str.strip().split(";"):
#         item = item.strip()
#         if not item:
#             continue
#         parts = item.split(" ", 1)
#         if len(parts) != 2:
#             continue
#         key = parts[0].strip()
#         val = parts[1].strip().strip('"')
#         out[key] = val
#     return out


# def load_exons_from_gtf(
#     gtf_path: str,
#     gene_id: Optional[str] = None,
#     transcript_id: Optional[str] = None,
# ) -> Tuple[str, str, List[Tuple[int, int]], Optional[str]]:
#     """
#     Return (chrom, strand, exons, gene_name) for a gene/transcript.

#     Exons returned as 1-based end-exclusive intervals (start, end_excl).
#     GTF start/end are 1-based inclusive, so end_excl = end_incl + 1.
#     """
#     assert (gene_id is not None) or (transcript_id is not None), "Provide gene_id or transcript_id."

#     chrom = None
#     strand = None
#     gene_name: Optional[str] = None
#     exons: List[Tuple[int, int]] = []

#     with open(gtf_path, "r") as f:
#         for line in f:
#             if not line or line.startswith("#"):
#                 continue
#             fields = line.rstrip("\n").split("\t")
#             if len(fields) != 9:
#                 continue

#             seqname, source, feature, start, end, score, st, frame, attrs = fields
#             if feature != "exon":
#                 continue

#             attrs_d = parse_gtf_attributes(attrs)
#             if transcript_id is not None:
#                 if attrs_d.get("transcript_id") != transcript_id:
#                     continue
#             else:
#                 if attrs_d.get("gene_id") != gene_id:
#                     continue

#             if gene_name is None:
#                 gene_name = attrs_d.get("gene_name", attrs_d.get("gene", None))

#             s = int(start)          # 1-based inclusive
#             e_incl = int(end)       # 1-based inclusive
#             e = e_incl + 1          # end-exclusive

#             if chrom is None:
#                 chrom = seqname
#             if strand is None:
#                 strand = st

#             if seqname != chrom:
#                 raise ValueError(f"Multiple chromosomes for locus: {chrom} vs {seqname}")
#             if st != strand:
#                 raise ValueError(f"Multiple strands for locus: {strand} vs {st}")

#             exons.append((s, e))

#     if chrom is None or strand is None or not exons:
#         raise ValueError("No exons found. Check IDs or the GTF.")

#     # sort + merge overlaps
#     exons.sort()
#     merged = []
#     for s, e in exons:
#         if not merged or s > merged[-1][1]:
#             merged.append([s, e])
#         else:
#             merged[-1][1] = max(merged[-1][1], e)
#     return chrom, strand, [(a, b) for a, b in merged], gene_name


# def exon_bounds(exons: List[Tuple[int, int]]) -> Tuple[int, int]:
#     return min(s for s, _ in exons), max(e for _, e in exons)


# # -----------------------
# # FASTA HELPERS
# # -----------------------
# def fetch_chrom_sequence(fasta_path: str, target_chrom: str) -> str:
#     """
#     Scan multi-record FASTA and return the record whose record.id == target_chrom.
#     """
#     for record in SeqIO.parse(fasta_path, "fasta"):
#         if record.id == target_chrom:
#             return str(record.seq).upper()
#     raise ValueError(f"Chromosome {target_chrom} not found in FASTA.")


# def slice_locus(seq_chr: str, start_1based: int, end_excl_1based: int) -> str:
#     """
#     Extract [start_1based, end_excl_1based) from a chromosome string.
#     """
#     if start_1based < 1:
#         start_1based = 1
#     if end_excl_1based > len(seq_chr) + 1:
#         end_excl_1based = len(seq_chr) + 1
#     s0 = start_1based - 1
#     e0 = end_excl_1based - 1
#     return seq_chr[s0:e0]


# def write_fasta(path: str, header: str, seq: str, wrap: int = 60) -> None:
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "w") as f:
#         f.write(f">{header}\n")
#         for i in range(0, len(seq), wrap):
#             f.write(seq[i:i+wrap] + "\n")


# # -----------------------
# # EXON LABEL OVERLAY
# # -----------------------
# def build_exon_labels_genomic_order(
#     locus_start_1based: int,
#     locus_len: int,
#     exons_endexcl_1based: List[Tuple[int, int]],
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Build exon labels aligned to the locus in genomic order (increasing coords).

#     Returns:
#       is_exon[L] in {0,1}
#       exon_id[L] in {-1, 1..N} where N = number of merged exons in genomic order.
#     """
#     exon_id = np.full((locus_len,), -1, dtype=np.int32)
#     for k, (s, e) in enumerate(exons_endexcl_1based, start=1):
#         lo = max(s, locus_start_1based)
#         hi = min(e, locus_start_1based + locus_len)
#         if hi <= lo:
#             continue
#         exon_id[(lo - locus_start_1based):(hi - locus_start_1based)] = k
#     is_exon = (exon_id != -1).astype(np.int32)
#     return is_exon, exon_id


# def build_boundary_distance_fields(is_exon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     For each position i (in ORIENTED coordinates), compute distance to nearest exon start and exon end.

#     exon start positions: indices where is_exon transitions 0->1
#     exon end positions:   indices of the last base inside an exon (1->0 boundary, using last exonic base)

#     Returns:
#       dist_to_exon_start[L], dist_to_exon_end[L] (float32; NaN if no exons)
#     """
#     L = len(is_exon)
#     prev = np.concatenate(([0], is_exon[:-1]))
#     nxt  = np.concatenate((is_exon[1:], [0]))

#     exon_starts = np.where((is_exon == 1) & (prev == 0))[0]
#     exon_ends   = np.where((is_exon == 1) & (nxt  == 0))[0]

#     def dist_to_points(points: np.ndarray) -> np.ndarray:
#         if points.size == 0:
#             return np.full((L,), np.nan, dtype=np.float32)
#         out = np.empty((L,), dtype=np.float32)
#         # O(L*B) but B is tiny (num exons) and L is manageable for locus-size runs.
#         for i in range(L):
#             out[i] = float(np.min(np.abs(points - i)))
#         return out

#     return dist_to_points(exon_starts), dist_to_points(exon_ends)


# def get_exon_intervals_oriented(is_exon: np.ndarray) -> List[Tuple[int, int]]:
#     """
#     Return list of half-open intervals [start, end) where is_exon==1 in oriented coords.
#     """
#     L = len(is_exon)
#     intervals: List[Tuple[int, int]] = []
#     i = 0
#     while i < L:
#         if is_exon[i] == 1:
#             j = i + 1
#             while j < L and is_exon[j] == 1:
#                 j += 1
#             intervals.append((i, j))
#             i = j
#         else:
#             i += 1
#     return intervals


# def get_exon_boundaries_oriented(is_exon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     prev = np.concatenate(([0], is_exon[:-1]))
#     nxt  = np.concatenate((is_exon[1:], [0]))
#     exon_starts = np.where((is_exon == 1) & (prev == 0))[0]
#     exon_ends   = np.where((is_exon == 1) & (nxt  == 0))[0]
#     return exon_starts, exon_ends


# # -----------------------
# # TOKEN / MODEL UTILITIES
# # -----------------------
# def _bos_id(tok):
#     # matches your previous scripts: bos_id or eod_id
#     bid = getattr(tok, "bos_id", getattr(tok, "eod_id", None))
#     if bid is None:
#         raise AssertionError("Tokenizer must provide bos_id or eod_id.")
#     return bid


# def _extract_logits(model_out):
#     # exactly your robust extractor pattern
#     if isinstance(model_out, torch.Tensor):
#         return model_out
#     if hasattr(model_out, "logits") and isinstance(model_out.logits, torch.Tensor):
#         return model_out.logits
#     if isinstance(model_out, dict) and isinstance(model_out.get("logits"), torch.Tensor):
#         return model_out["logits"]
#     if isinstance(model_out, (tuple, list)):
#         first = model_out[0]
#         if isinstance(first, torch.Tensor):
#             return first
#         if isinstance(first, (tuple, list)) and len(first) > 0 and isinstance(first[0], torch.Tensor):
#             return first[0]
#         if hasattr(first, "logits") and isinstance(first.logits, torch.Tensor):
#             return first.logits
#         if isinstance(first, dict) and isinstance(first.get("logits"), torch.Tensor):
#             return first["logits"]
#         for x in model_out:
#             if isinstance(x, torch.Tensor):
#                 return x
#             if hasattr(x, "logits") and isinstance(x.logits, torch.Tensor):
#                 return x.logits
#             if isinstance(x, dict) and isinstance(x.get("logits"), torch.Tensor):
#                 return x["logits"]
#             if isinstance(x, (tuple, list)):
#                 for y in x:
#                     if isinstance(y, torch.Tensor):
#                         return y
#                     if hasattr(y, "logits") and isinstance(y.logits, torch.Tensor):
#                         return y.logits
#                     if isinstance(y, dict) and isinstance(y.get("logits"), torch.Tensor):
#                         return y["logits"]
#     raise TypeError(f"Could not extract logits from model output of type {type(model_out)}")


# def id_to_token_str(tok, idx: int) -> str:
#     """
#     Same as your scripts: best-effort token->string conversion.
#     """
#     for attr in ("id_to_token", "decode", "detokenize", "convert_ids_to_tokens"):
#         fn = getattr(tok, attr, None)
#         if callable(fn):
#             try:
#                 out = fn([idx]) if attr in ("decode", "detokenize", "convert_ids_to_tokens") else fn(idx)
#                 return out[0] if isinstance(out, (list, tuple)) else str(out)
#             except Exception:
#                 pass
#     return str(idx)


# # -----------------------
# # ENTROPY (MATCHES YOUR GENOME-WIDE SCRIPTS)
# # -----------------------
# @torch.inference_mode()
# def entropy_like_reference_acgt(
#     sequence: str,
#     model: Evo2,
#     ACGT_IDS: torch.Tensor,
#     device: str,
#     prepend_bos: bool = True,
#     reverse_complement: bool = False,
# ):
#     """
#     EXACT SAME DEFINITION as your genome-wide scripts.

#     Returns:
#       H_final (torch.Tensor shape [L]), PPX (torch.Tensor shape [L])
#     """
#     tok = model.tokenizer
#     toks = tok.tokenize(sequence)
#     if prepend_bos:
#         toks = [_bos_id(tok)] + toks
#     input_ids = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
#     out = model(input_ids)
#     logits = _extract_logits(out)  # [1, T, V]

#     logits_sub = logits.index_select(-1, ACGT_IDS)   # [1, T, 4]
#     logZ = torch.logsumexp(logits_sub, dim=-1, keepdim=True)
#     logp = logits_sub - logZ
#     H_fwd = -(logp.exp() * logp).sum(dim=-1)         # [1, T]

#     if prepend_bos:
#         H_fwd = H_fwd[:, 1:]
#     H_fwd = H_fwd.squeeze(0).detach().cpu()

#     if not reverse_complement:
#         H_final = H_fwd
#     else:
#         seq_rc = str(Seq(sequence).reverse_complement())
#         toks_rc = tok.tokenize(seq_rc)
#         if prepend_bos:
#             toks_rc = [_bos_id(tok)] + toks_rc
#         input_ids_rc = torch.tensor(toks_rc, dtype=torch.long, device=device).unsqueeze(0)
#         out_rc = model(input_ids_rc)
#         logits_rc = _extract_logits(out_rc)

#         logits_rc_sub = logits_rc.index_select(-1, ACGT_IDS)
#         logZ_rc = torch.logsumexp(logits_rc_sub, dim=-1, keepdim=True)
#         logp_rc = logits_rc_sub - logZ_rc
#         H_rc = -(logp_rc.exp() * logp_rc).sum(dim=-1)

#         if prepend_bos:
#             H_rc = H_rc[:, 1:]
#         H_rc = H_rc.squeeze(0).detach().cpu()

#         # flip RC to align with forward coordinates
#         H_rc = torch.flip(H_rc, dims=[0])
#         H_final = 0.5 * (H_fwd + H_rc)

#     PPX = H_final.exp()
#     return H_final, PPX


# # -----------------------
# # NEXT TOKEN
# # -----------------------
# @torch.inference_mode()
# def next_token_logprobs_and_targets_aligned(sequence: str, model: Evo2, device: str):
#     """
#     Returns:
#       logprobs_next: [L, V]
#       target_next:   [L]
#     """
#     tok = model.tokenizer
#     toks = [_bos_id(tok)] + tok.tokenize(sequence)
#     input_ids = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
#     out = model(input_ids)
#     logits = _extract_logits(out).float()  # force float32 to avoid bf16->numpy issues
#     logprobs = torch.log_softmax(logits, dim=-1)

#     logprobs_next = logprobs[:, :-1, :]       # [1, L, V]
#     target_next = input_ids[:, 1:]            # [1, L]
#     return logprobs_next.squeeze(0), target_next.squeeze(0)


# def next_token_probs_subset(logprobs_next: torch.Tensor, subset_ids: torch.Tensor) -> torch.Tensor:
#     p_next = logprobs_next.exp()
#     return p_next.index_select(-1, subset_ids)


# # -----------------------
# # LOCUS-ALIGNED SCORING (STRONG ALIGNMENT)
# # -----------------------
# def score_locus_aligned(
#     seq_oriented: str,
#     evo2_model: Evo2,
#     ACGT_IDS: torch.Tensor,
#     device: str,
#     max_chunk_len: int,
#     compute_rcavg_entropy: bool = True,
# ):
#     """
#     Returns:
#       entropy_fwd[L], ppx_fwd[L],
#       entropy_rcavg[L], ppx_rcavg[L],
#       p4[L,4], true_tok[L], ll_next[L]
#     """
#     L = len(seq_oriented)

#     entropy_fwd = np.full((L,), np.nan, dtype=np.float32)
#     ppx_fwd     = np.full((L,), np.nan, dtype=np.float32)

#     entropy_rc  = np.full((L,), np.nan, dtype=np.float32)
#     ppx_rc      = np.full((L,), np.nan, dtype=np.float32)

#     p4       = np.full((L, 4), np.nan, dtype=np.float32)
#     ll_next  = np.full((L,), np.nan, dtype=np.float32)
#     true_tok = np.array([""] * L, dtype=object)

#     mask_idx = [i for i, b in enumerate(seq_oriented) if b != "N"]
#     if not mask_idx:
#         return entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next

#     num_chunks = math.ceil(len(mask_idx) / max_chunk_len)

#     for chunk_i in range(num_chunks):
#         a = chunk_i * max_chunk_len
#         b = min((chunk_i + 1) * max_chunk_len, len(mask_idx))
#         idx_slice = mask_idx[a:b]
#         chunk_seq = "".join(seq_oriented[i] for i in idx_slice)

#         # --- Entropy / PPX forward-only over A/C/G/T
#         H_fwd_t, PPX_fwd_t = entropy_like_reference_acgt(
#             chunk_seq, evo2_model, ACGT_IDS, device,
#             prepend_bos=True,
#             reverse_complement=False
#         )
#         H_fwd = H_fwd_t.float().numpy()
#         PPX_fwd = PPX_fwd_t.float().numpy()

#         # --- RC-averaged entropy
#         if compute_rcavg_entropy:
#             H_rc_t, PPX_rc_t = entropy_like_reference_acgt(
#                 chunk_seq, evo2_model, ACGT_IDS, device,
#                 prepend_bos=True,
#                 reverse_complement=True
#             )
#             H_rc = H_rc_t.float().numpy()
#             PPX_rc = PPX_rc_t.float().numpy()
#         else:
#             H_rc = None
#             PPX_rc = None

#         # --- Next-token / probabilities (forward-only)
#         logprobs_next, target_next = next_token_logprobs_and_targets_aligned(chunk_seq, evo2_model, device)

#         ll = (
#             logprobs_next.float()
#             .gather(-1, target_next.unsqueeze(-1))
#             .squeeze(-1)
#             .detach()
#             .cpu()
#             .numpy()
#             .astype(np.float32)
#         )

#         p4_chunk = (
#             next_token_probs_subset(logprobs_next.float(), ACGT_IDS)
#             .detach()
#             .cpu()
#             .numpy()
#             .astype(np.float32)
#         )

#         target_ids = target_next.detach().cpu().tolist()
#         tok = evo2_model.tokenizer
#         true_tok_chunk = [id_to_token_str(tok, int(tid)) for tid in target_ids]

#         # --- write back aligned
#         for k, orig_idx in enumerate(idx_slice):
#             entropy_fwd[orig_idx] = H_fwd[k]
#             ppx_fwd[orig_idx] = PPX_fwd[k]
#             if compute_rcavg_entropy:
#                 entropy_rc[orig_idx] = H_rc[k]
#                 ppx_rc[orig_idx] = PPX_rc[k]
#             p4[orig_idx, :] = p4_chunk[k, :]
#             ll_next[orig_idx] = ll[k]
#             true_tok[orig_idx] = true_tok_chunk[k]

#     return entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next


# # -----------------------
# # DROP DETECTION METHODS
# # -----------------------
# def _fill_nans_linear(x: np.ndarray) -> np.ndarray:
#     y = x.astype(np.float32, copy=True)
#     isn = np.isnan(y)
#     if not np.any(isn):
#         return y
#     idx = np.arange(len(y))
#     good = ~isn
#     if good.sum() >= 2:
#         y[isn] = np.interp(idx[isn], idx[good], y[good])
#     elif good.sum() == 1:
#         y[isn] = y[good][0]
#     return y


# def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
#     if w <= 1:
#         return x.copy()
#     y = _fill_nans_linear(x)
#     kernel = np.ones(w, dtype=np.float32) / float(w)
#     return np.convolve(y, kernel, mode="same")


# def detect_drops_derivative(entropy: np.ndarray, smooth_w: int, thr_quantile: float) -> List[int]:
#     sm = _rolling_mean(entropy, smooth_w)
#     d = np.diff(sm, prepend=sm[0])
#     thr = np.quantile(d, thr_quantile)
#     candidates = np.where(d <= thr)[0].tolist()

#     out = []
#     last = -10**9
#     min_sep = max(10, smooth_w // 2)
#     for i in candidates:
#         if i - last >= min_sep:
#             out.append(i)
#             last = i
#     return out


# def detect_drops_window_mean_shift(entropy: np.ndarray, w: int, top_k: int) -> List[int]:
#     x = _fill_nans_linear(entropy)
#     L = len(x)
#     scores = np.full((L,), np.nan, dtype=np.float32)

#     min_len = max(5, w // 10)
#     for i in range(L):
#         a0, a1 = max(0, i - w), i
#         b0, b1 = i, min(L, i + w)
#         if (a1 - a0) < min_len or (b1 - b0) < min_len:
#             continue
#         scores[i] = float(np.mean(x[b0:b1]) - np.mean(x[a0:a1]))

#     good = ~np.isnan(scores)
#     if good.sum() == 0:
#         return []

#     idx_good = np.where(good)[0]
#     order = np.argsort(scores[good])  # ascending => most negative first
#     picks = idx_good[order][:top_k].tolist()

#     out = []
#     for i in picks:
#         if all(abs(i - j) > w // 2 for j in out):
#             out.append(i)
#     return out


# def detect_drops_cusum(entropy: np.ndarray, smooth_w: int, h: float) -> List[int]:
#     x = _rolling_mean(entropy, smooth_w)
#     x = _fill_nans_linear(x)
#     mu = float(np.mean(x))

#     out = []
#     s = 0.0
#     last = -10**9
#     min_sep = max(25, smooth_w)

#     for i, xi in enumerate(x):
#         s = max(0.0, s + (mu - float(xi)))
#         if s > h and (i - last) > min_sep:
#             out.append(i)
#             last = i
#             s = 0.0
#     return out


# # -----------------------
# # PLOTTING (DECOMPOSED SUITE)
# # -----------------------
# def shade_exons(ax, exon_intervals: List[Tuple[int, int]], alpha: float = 0.12) -> None:
#     for (s, e) in exon_intervals:
#         ax.axvspan(s, e, alpha=alpha)


# def _save_fig(path: str, dpi: int = 200) -> None:
#     plt.tight_layout()
#     plt.savefig(path, dpi=dpi)
#     plt.close()


# def plot_suite(
#     out_prefix: str,
#     entropy_main: np.ndarray,
#     is_exon: np.ndarray,
#     drop_points: Dict[str, List[int]],
#     title_prefix: str,
#     smooth_w: int = 51,
#     zoom_bp: int = 0,
#     max_zoom_plots: int = 60,
# ) -> None:
#     x = np.arange(len(entropy_main))
#     exon_intervals = get_exon_intervals_oriented(is_exon)
#     exon_starts, exon_ends = get_exon_boundaries_oriented(is_exon)
#     sm = _rolling_mean(entropy_main, smooth_w)

#     # 1) Raw entropy(main) + exon shading
#     plt.figure(figsize=(16, 4))
#     ax = plt.gca()
#     shade_exons(ax, exon_intervals, alpha=0.12)
#     ax.plot(x, entropy_main, linewidth=0.8, label="Entropy(main)")
#     ax.set_title(f"{title_prefix} | raw")
#     ax.set_xlabel("OrientedIdx (5'→3')")
#     ax.set_ylabel("Entropy (nats)")
#     ax.legend(loc="best", fontsize=8)
#     _save_fig(out_prefix + ".entropy_raw.png")

#     # 2) Smoothed entropy + exon shading + boundaries
#     plt.figure(figsize=(16, 4))
#     ax = plt.gca()
#     shade_exons(ax, exon_intervals, alpha=0.12)
#     ax.plot(x, sm, linewidth=1.2, label=f"Entropy(main) rolling_mean(w={smooth_w})")
#     for s in exon_starts:
#         ax.axvline(s, linestyle="--", linewidth=0.7, alpha=0.75)
#     for e in exon_ends:
#         ax.axvline(e, linestyle=":", linewidth=0.7, alpha=0.75)
#     ax.set_title(f"{title_prefix} | smoothed + exon boundaries")
#     ax.set_xlabel("OrientedIdx (5'→3')")
#     ax.set_ylabel("Entropy (nats)")
#     ax.legend(loc="best", fontsize=8)
#     _save_fig(out_prefix + ".entropy_smooth.png")

#     # 3) Boundary-only view (no drops)
#     plt.figure(figsize=(16, 4))
#     ax = plt.gca()
#     shade_exons(ax, exon_intervals, alpha=0.12)
#     ax.plot(x, entropy_main, linewidth=0.8, label="Entropy(main)")
#     for s in exon_starts:
#         ax.axvline(s, linestyle="--", linewidth=0.8, alpha=0.85)
#     for e in exon_ends:
#         ax.axvline(e, linestyle=":", linewidth=0.8, alpha=0.85)
#     ax.set_title(f"{title_prefix} | boundary-only")
#     ax.set_xlabel("OrientedIdx (5'→3')")
#     ax.set_ylabel("Entropy (nats)")
#     ax.legend(loc="best", fontsize=8)
#     _save_fig(out_prefix + ".entropy_boundaries.png")

#     # 4) One plot per drop method (clean separation)
#     for method, pts in drop_points.items():
#         plt.figure(figsize=(16, 4))
#         ax = plt.gca()
#         shade_exons(ax, exon_intervals, alpha=0.12)
#         ax.plot(x, sm, linewidth=1.2, label="Smoothed entropy")
#         if pts:
#             ys = sm[pts]
#             ax.scatter(pts, ys, s=22, label=f"drops:{method}")
#         ax.set_title(f"{title_prefix} | drops={method}")
#         ax.set_xlabel("OrientedIdx (5'→3')")
#         ax.set_ylabel("Entropy (nats)")
#         ax.legend(loc="best", fontsize=8)
#         _save_fig(out_prefix + f".drops_{method}.png")

#     # 5) Optional zoom plots around each boundary (±zoom_bp)
#     if zoom_bp and zoom_bp > 0:
#         boundaries = [(int(s), "start") for s in exon_starts] + [(int(e), "end") for e in exon_ends]
#         boundaries.sort(key=lambda t: t[0])

#         count = 0
#         L = len(entropy_main)
#         for idx, kind in boundaries:
#             if count >= max_zoom_plots:
#                 break
#             lo = max(0, idx - zoom_bp)
#             hi = min(L, idx + zoom_bp)

#             plt.figure(figsize=(14, 4))
#             ax = plt.gca()

#             # shade exon segments intersecting zoom window
#             for (s, e) in exon_intervals:
#                 ss = max(s, lo)
#                 ee = min(e, hi)
#                 if ee > ss:
#                     ax.axvspan(ss, ee, alpha=0.15)

#             ax.plot(np.arange(lo, hi), sm[lo:hi], linewidth=1.3, label="Smoothed entropy")
#             ax.axvline(idx, linestyle="--" if kind == "start" else ":", linewidth=1.2, alpha=0.9, label=f"exon_{kind}")

#             ax.set_title(f"{title_prefix} | zoom {kind} @ {idx} (±{zoom_bp}bp)")
#             ax.set_xlabel("OrientedIdx (5'→3')")
#             ax.set_ylabel("Entropy (nats)")
#             ax.legend(loc="best", fontsize=8)
#             _save_fig(out_prefix + f".zoom_{kind}_{idx}.png")
#             count += 1


# # -----------------------
# # WINDOW SUMMARY (OPTIONAL)
# # -----------------------
# def write_window_summary(
#     out_path: str,
#     entropy: np.ndarray,
#     is_exon: np.ndarray,
#     win: int = 200,
#     step: int = 50,
# ):
#     with open(out_path, "w") as f:
#         f.write("WinStartOriented\tWinEndOriented\tMeanEntropy\tMeanEntropyExon\tMeanEntropyIntron\tFracExon\n")
#         L = len(entropy)
#         for s in range(0, L - win + 1, step):
#             e = s + win
#             ent_w = entropy[s:e]
#             ex = is_exon[s:e].astype(bool)

#             mean_all = float(np.nanmean(ent_w)) if np.any(~np.isnan(ent_w)) else np.nan
#             mean_ex  = float(np.nanmean(ent_w[ex])) if np.any(ex) else np.nan
#             mean_in  = float(np.nanmean(ent_w[~ex])) if np.any(~ex) else np.nan
#             frac_ex  = float(np.mean(ex))

#             f.write(f"{s}\t{e}\t{mean_all:.6f}\t{mean_ex:.6f}\t{mean_in:.6f}\t{frac_ex:.4f}\n")


# # -----------------------
# # MAIN LOCUS RUN
# # -----------------------
# def run_one_locus(
#     fasta_path: str,
#     gtf_path: str,
#     out_dir: str,
#     gene_id: Optional[str],
#     transcript_id: Optional[str],
#     buffer_bp: int,
#     max_chunk_len: int,
#     drop_on: str,        # "rcavg" or "fwd"
#     zoom_bp: int = 0,    # set >0 to output zoom plots
#     max_zoom_plots: int = 60,
# ):
#     os.makedirs(out_dir, exist_ok=True)

#     tag = transcript_id if transcript_id else gene_id
#     if not tag:
#         raise ValueError("Provide gene_id or transcript_id.")

#     # 1) Load exons (+ optional gene_name)
#     chrom, strand, exons, gene_name = load_exons_from_gtf(gtf_path, gene_id=gene_id, transcript_id=transcript_id)
#     exon_start, exon_end_excl = exon_bounds(exons)

#     # 2) Define locus with buffer
#     locus_start = max(1, exon_start - buffer_bp)
#     locus_end_excl = exon_end_excl + buffer_bp
#     locus_len = locus_end_excl - locus_start

#     print(f"[INFO] tag={tag}" + (f" | gene_name={gene_name}" if gene_name else ""))
#     print(f"[INFO] chrom={chrom}, strand={strand}")
#     print(f"[INFO] exon span  [{exon_start}, {exon_end_excl})")
#     print(f"[INFO] locus span [{locus_start}, {locus_end_excl})  len={locus_len}  buffer={buffer_bp}")

#     # 3) Fetch locus sequence in genomic order
#     print("[INFO] Loading chromosome sequence from FASTA...")
#     chr_seq = fetch_chrom_sequence(fasta_path, chrom)
#     locus_seq_genomic = slice_locus(chr_seq, locus_start, locus_end_excl).upper()
#     if len(locus_seq_genomic) != locus_len:
#         raise RuntimeError("Locus slice length mismatch (check coordinate logic).")

#     # 4) Exon labels in genomic order
#     is_exon_g, exon_id_g = build_exon_labels_genomic_order(locus_start, locus_len, exons)

#     # 5) Orient to 5'→3' and define Pos array so TSV walks 5'→3'
#     if strand == "+":
#         locus_seq = locus_seq_genomic
#         is_exon = is_exon_g
#         exon_id = exon_id_g
#         pos = np.arange(locus_start, locus_end_excl, dtype=np.int64)
#     elif strand == "-":
#         locus_seq = str(Seq(locus_seq_genomic).reverse_complement())
#         is_exon = is_exon_g[::-1].copy()
#         exon_id = exon_id_g[::-1].copy()
#         pos = np.arange(locus_end_excl - 1, locus_start - 1, -1, dtype=np.int64)
#     else:
#         raise ValueError(f"Unexpected strand: {strand}")

#     dist_to_exon_start, dist_to_exon_end = build_boundary_distance_fields(is_exon)

#     # Base name for outputs
#     base_name = f"{tag}_{chrom}_{locus_start}_{locus_end_excl}_strand{strand}"
#     out_prefix = os.path.join(out_dir, base_name)

#     # 5b) Export oriented locus + exon-only FASTA (for reference / debugging)
#     locus_hdr = f"{tag}|{gene_name or 'NA'}|{chrom}:{locus_start}-{locus_end_excl}|strand={strand}|oriented_5to3"
#     write_fasta(out_prefix + ".locus_oriented.fa", locus_hdr, locus_seq)
#     print("[INFO] Wrote locus FASTA:", out_prefix + ".locus_oriented.fa")

#     exon_intervals = get_exon_intervals_oriented(is_exon)
#     with open(out_prefix + ".exons_oriented.fa", "w") as f:
#         for k, (s, e) in enumerate(exon_intervals, start=1):
#             exon_seq = locus_seq[s:e]
#             f.write(f">{tag}|{gene_name or 'NA'}|exon{k}|orientedIdx={s}-{e}\n")
#             for i in range(0, len(exon_seq), 60):
#                 f.write(exon_seq[i:i+60] + "\n")
#     print("[INFO] Wrote exons FASTA:", out_prefix + ".exons_oriented.fa")

#     # 6) Model init
#     print("[INFO] Initializing Evo2 model...")
#     evo2_model = Evo2("evo2_7b")
#     device = str(getattr(evo2_model, "device", "cuda:0" if torch.cuda.is_available() else "cpu"))
#     if hasattr(evo2_model, "eval"):
#         evo2_model.eval()
#     print("[INFO] device:", device)

#     # Token ids for A/C/G/T
#     idx_A = evo2_model.tokenizer.tokenize("A")[0]
#     idx_C = evo2_model.tokenizer.tokenize("C")[0]
#     idx_G = evo2_model.tokenizer.tokenize("G")[0]
#     idx_T = evo2_model.tokenizer.tokenize("T")[0]
#     ACGT_IDS = torch.tensor([idx_A, idx_C, idx_G, idx_T], dtype=torch.long, device=device)

#     # 7) Score locus aligned
#     print("[INFO] Scoring locus (aligned arrays)...")
#     entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next = score_locus_aligned(
#         locus_seq,
#         evo2_model,
#         ACGT_IDS,
#         device,
#         max_chunk_len=max_chunk_len,
#         compute_rcavg_entropy=True,
#     )

#     # Choose entropy to run drop detection on
#     if drop_on == "rcavg":
#         entropy_main = entropy_rc
#         name_main = "Entropy_RCavg"
#     else:
#         entropy_main = entropy_fwd
#         name_main = "Entropy_fwd"

#     # 8) Drop detection
#     print(f"[INFO] Drop detection on: {name_main}")
#     drops = {
#         "derivative": detect_drops_derivative(entropy_main, smooth_w=DROP_SMOOTH_W, thr_quantile=DROP_DERIV_Q),
#         "win_shift":  detect_drops_window_mean_shift(entropy_main, w=DROP_SHIFT_W, top_k=DROP_SHIFT_TOPK),
#         "cusum":      detect_drops_cusum(entropy_main, smooth_w=DROP_SMOOTH_W, h=DROP_CUSUM_H),
#     }

#     # 9) Write TSV
#     out_tsv = out_prefix + ".tsv"
#     print("[INFO] Writing TSV:", out_tsv)

#     with open(out_tsv, "w") as f:
#         f.write("Pos\tEntropy(nats)\tPerplexity(e)\tP(A)\tP(C)\tP(G)\tP(T)\tTrueToken\tLL_next(nats)")
#         f.write("\tEntropy_RCavg(nats)\tPerplexity_RCavg(e)")
#         f.write("\tBase\tOrientedIdx\tIsExon\tExonID\tDistToExonStart\tDistToExonEnd\n")

#         for i in range(locus_len):
#             ent = float(entropy_fwd[i]) if not np.isnan(entropy_fwd[i]) else np.nan
#             px  = float(ppx_fwd[i]) if not np.isnan(ppx_fwd[i]) else np.nan
#             ll  = float(ll_next[i]) if not np.isnan(ll_next[i]) else np.nan

#             if not np.isnan(p4[i, 0]):
#                 a, c, g, t = p4[i, :].tolist()
#             else:
#                 a = c = g = t = np.nan

#             ent_rc = float(entropy_rc[i]) if not np.isnan(entropy_rc[i]) else np.nan
#             px_rc  = float(ppx_rc[i]) if not np.isnan(ppx_rc[i]) else np.nan

#             f.write(
#                 f"{int(pos[i])}\t"
#                 f"{ent:.6f}\t{px:.6f}\t"
#                 f"{a:.6f}\t{c:.6f}\t{g:.6f}\t{t:.6f}\t"
#                 f"{true_tok[i]}\t{ll:.6f}\t"
#                 f"{ent_rc:.6f}\t{px_rc:.6f}\t"
#                 f"{locus_seq[i]}\t{i}\t{int(is_exon[i])}\t{int(exon_id[i])}\t"
#                 f"{dist_to_exon_start[i]:.1f}\t{dist_to_exon_end[i]:.1f}\n"
#             )

#     # 10) Plot suite (NO entropy_fwd overlay; decomposed plots instead)
#     title_prefix = f"{tag} {chrom}:{locus_start}-{locus_end_excl} strand {strand} (5'→3') | drop_on={drop_on}"
#     print("[INFO] Writing plot suite:", out_prefix + ".*.png")
#     plot_suite(
#         out_prefix=out_prefix,
#         entropy_main=entropy_main,
#         is_exon=is_exon,
#         drop_points=drops,
#         title_prefix=title_prefix,
#         smooth_w=DROP_SMOOTH_W,
#         zoom_bp=zoom_bp,
#         max_zoom_plots=max_zoom_plots,
#     )

#     # 11) Drop point list
#     out_drops = out_prefix + ".drops.txt"
#     print("[INFO] Writing drop points:", out_drops)
#     with open(out_drops, "w") as f:
#         for k, pts in drops.items():
#             f.write(f"{k}\t" + ",".join(map(str, pts)) + "\n")

#     # 12) Window summary
#     out_summary = out_prefix + ".window_summary.tsv"
#     print("[INFO] Writing window summary:", out_summary)
#     write_window_summary(out_summary, entropy_main, is_exon, win=200, step=50)

#     print("[DONE]")


# # -----------------------
# # PICK-15 HELPER (GTF ONLY)
# # -----------------------
# def parse_transcript_exons_from_gtf(gtf_path: str) -> Dict[str, Dict]:
#     """
#     Build a transcript-level index from the GTF exon lines only.

#     Returns dict:
#       tx_id -> {
#         chrom, strand,
#         exons: [(s, e_endexcl), ...] merged+sorted,
#         exon_count: int,
#         exon_span: (min_s, max_e),
#         length_bp: max_e - min_s,
#       }
#     """
#     tx_map: Dict[str, Dict] = {}

#     with open(gtf_path, "r") as f:
#         for line in f:
#             if not line or line.startswith("#"):
#                 continue
#             fields = line.rstrip("\n").split("\t")
#             if len(fields) != 9:
#                 continue
#             seqname, source, feature, start, end, score, st, frame, attrs = fields
#             if feature != "exon":
#                 continue

#             attrs_d = parse_gtf_attributes(attrs)
#             tx_id = attrs_d.get("transcript_id")
#             if tx_id is None:
#                 continue

#             s = int(start)
#             e = int(end) + 1

#             rec = tx_map.get(tx_id)
#             if rec is None:
#                 tx_map[tx_id] = {"chrom": seqname, "strand": st, "exons": [(s, e)]}
#             else:
#                 if rec["chrom"] != seqname or rec["strand"] != st:
#                     continue
#                 rec["exons"].append((s, e))

#     # merge exons per transcript
#     for tx_id, rec in tx_map.items():
#         exons = rec["exons"]
#         exons.sort()
#         merged = []
#         for s, e in exons:
#             if not merged or s > merged[-1][1]:
#                 merged.append([s, e])
#             else:
#                 merged[-1][1] = max(merged[-1][1], e)
#         exm = [(a, b) for a, b in merged]
#         rec["exons"] = exm
#         rec["exon_count"] = len(exm)
#         mn = min(a for a, _ in exm)
#         mx = max(b for _, b in exm)
#         rec["exon_span"] = (mn, mx)
#         rec["length_bp"] = mx - mn

#     return tx_map


# def pick_15_transcripts_recipe(gtf_path: str) -> List[str]:
#     """
#     Produce a balanced list of 15 transcript_ids using ONLY GTF geometry.
#     """
#     tx_map = parse_transcript_exons_from_gtf(gtf_path)
#     items = [(tx, rec["chrom"], rec["strand"], rec["exon_count"], rec["length_bp"]) for tx, rec in tx_map.items()]

#     items_valid = [x for x in items if x[3] >= 2 and x[4] >= 500]  # >=2 exons, >=500bp

#     # Category 1
#     cat1 = [x for x in items_valid if (10000 <= x[4] <= 50000) and (5 <= x[3] <= 15)]
#     cat1.sort(key=lambda z: (abs(z[4] - 25000), abs(z[3] - 10)))
#     pick1 = [tx for tx, *_ in cat1[:5]]

#     # Category 2 (long & many exons)
#     cat2 = [x for x in items_valid if x[4] >= 200000]
#     cat2.sort(key=lambda z: (-z[3], -z[4]))
#     pick2 = [tx for tx, *_ in cat2[:4]]

#     # Category 3 (short)
#     cat3 = [x for x in items_valid if x[4] <= 5000 and x[3] <= 3]
#     cat3.sort(key=lambda z: (z[4], z[3]))
#     pick3 = [tx for tx, *_ in cat3[:3]]

#     # Category 5 tricky: long but low exon_count
#     cat5 = [x for x in items_valid if x[4] >= 100000 and x[3] <= 3]
#     cat5.sort(key=lambda z: (-z[4], z[3]))
#     pick5 = [tx for tx, *_ in cat5[:2]]

#     picks: List[str] = []
#     for group in (pick1, pick2, pick3, pick5):
#         for tx in group:
#             if tx not in picks:
#                 picks.append(tx)
#             if len(picks) >= 15:
#                 return picks

#     remaining = [x for x in items_valid if x[0] not in picks]
#     remaining.sort(key=lambda z: (-z[3], -z[4]))
#     for tx, *_ in remaining:
#         picks.append(tx)
#         if len(picks) >= 15:
#             break

#     return picks[:15]


# # -----------------------
# # CLI
# # -----------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--fasta", default=FASTA_PATH_DEFAULT)
#     ap.add_argument("--gtf", default=GTF_PATH_DEFAULT)
#     ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)

#     ap.add_argument("--gene_id", default=None)
#     ap.add_argument("--transcript_id", default=None)

#     ap.add_argument("--buffer_bp", type=int, default=BUFFER_BP_DEFAULT)
#     ap.add_argument("--max_chunk_len", type=int, default=MAX_CHUNK_LEN_DEFAULT)

#     ap.add_argument("--drop_on", choices=["rcavg", "fwd"], default="rcavg")

#     # plotting extras
#     ap.add_argument("--zoom_bp", type=int, default=ZOOM_BP_DEFAULT, help="If >0, save zoom plots around exon boundaries (±zoom_bp).")
#     ap.add_argument("--max_zoom_plots", type=int, default=MAX_ZOOM_PLOTS_DEFAULT, help="Cap number of zoom plots.")

#     # helper mode
#     ap.add_argument("--pick15", action="store_true", help="Print 15 transcript_ids selected from the GTF.")

#     args = ap.parse_args()

#     if args.pick15:
#         txs = pick_15_transcripts_recipe(args.gtf)
#         print("\n".join(txs))
#         return

#     if (args.gene_id is None) and (args.transcript_id is None):
#         raise SystemExit("Provide --gene_id or --transcript_id, or use --pick15")

#     run_one_locus(
#         fasta_path=args.fasta,
#         gtf_path=args.gtf,
#         out_dir=args.out_dir,
#         gene_id=args.gene_id,
#         transcript_id=args.transcript_id,
#         buffer_bp=args.buffer_bp,
#         max_chunk_len=args.max_chunk_len,
#         drop_on=args.drop_on,
#         zoom_bp=args.zoom_bp,
#         max_zoom_plots=args.max_zoom_plots,
#     )


# if __name__ == "__main__":
#     main()

