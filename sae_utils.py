#!/usr/bin/env python3
"""
sae_utils.py - Sparse Autoencoder utilities for Evo2 feature extraction

This module provides classes and functions for extracting interpretable features
from Evo2's internal representations using a pre-trained sparse autoencoder (SAE).

The SAE decomposes layer 26 activations into ~32K sparse features that correspond
to biological concepts (splice sites, promoters, structural motifs, etc.).

Key Components:
    - ModelScope: Hook management for PyTorch models
    - ObservableEvo2: Evo2 wrapper with activation caching
    - BatchTopKTiedSAE: TopK sparse autoencoder with tied weights
    - load_topk_sae: Load pre-trained SAE weights from HuggingFace

Usage:
    from sae_utils import ObservableEvo2, load_topk_sae, get_feature_ts

    # Initialize model and SAE
    model = ObservableEvo2("evo2_7b")
    sae = load_topk_sae_from_hf(model.d_hidden, model.device, model.dtype)

    # Extract features for a sequence
    features = get_feature_ts(model, sae, "ACGTACGT...")
    # features shape: (seq_len, 32768)

Based on: evo2/notebooks/sparse_autoencoder/sparse_autoencoder.ipynb
SAE weights: Goodfire/Evo-2-Layer-26-Mixed
"""

from typing import List, Optional, Callable, Dict, Any, Tuple
from collections import defaultdict
import contextlib

import numpy as np
import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download

# Disable FP8 on GPUs with compute capability < 8.9 (A100=8.0, H100=9.0).
# FP8 calibration allows the first ~16 batches to pass, but then crashes on A100.
# Patch te.fp8_autocast to a no-op so the model runs in bf16 on non-H100 nodes.
try:
    import transformer_engine.pytorch as te
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if not (cc[0] > 8 or (cc[0] == 8 and cc[1] >= 9)):
            te.fp8_autocast = lambda *args, **kwargs: contextlib.nullcontext()
except Exception:
    pass

# Import Evo2 - handle different import paths
try:
    from evo2 import Evo2
except ImportError:
    try:
        from evo2.evo2 import Evo2
    except ImportError:
        Evo2 = None
        print("[WARNING] Could not import Evo2. SAE analysis will require manual model setup.")


# =============================================================================
# CONSTANTS
# =============================================================================

# Default SAE layer for feature extraction
SAE_LAYER_NAME = 'blocks-26'

# SAE model configuration
SAE_REPO_ID = "Goodfire/Evo-2-Layer-26-Mixed"
SAE_FILENAME = "sae-layer26-mixed-expansion_8-k_64.pt"
SAE_EXPANSION_FACTOR = 8  # 4096 -> 32768 features
SAE_TOPK = 64  # Number of features active per position


# =============================================================================
# MODEL SCOPE - Hook Management
# =============================================================================

class ModelScope:
    """
    Class for adding, using, and removing PyTorch hooks with a model.

    Provides functionality for:
    - Activation caching at any layer
    - Activation overriding for interventions
    - Clean hook management

    Attributes:
        model: The PyTorch model to attach hooks to
        hooks: Dictionary of registered hook handles
        activations_cache: Cached activations by module name
        override_store: Override tensors by module name
    """

    def __init__(self, model: nn.Module):
        """
        Initialize ModelScope with a model.

        Args:
            model: PyTorch model to manage hooks for
        """
        self.model = model
        self.hooks: Dict[str, Any] = {}
        self.activations_cache: Dict[str, List[torch.Tensor]] = {}
        self.override_store: Dict[str, Optional[torch.Tensor]] = {}
        self._build_module_dict()

    def _build_module_dict(self) -> None:
        """Walks the model's module tree and builds a name: module map."""
        self._module_dict: Dict[str, nn.Module] = {}

        def recurse(module: nn.Module, prefix: str = '') -> None:
            """Recursive tree walk to build self._module_dict."""
            for name, child in module.named_children():
                self._module_dict[prefix + name] = child
                recurse(child, prefix=prefix + name + '-')

        recurse(self.model)

    def list_modules(self) -> List[str]:
        """Lists all modules in the module dictionary."""
        return list(self._module_dict.keys())

    def add_hook(
        self,
        hook_fn: Callable,
        module_str: str,
        hook_name: str
    ) -> None:
        """
        Add a hook_fn to the module given by module_str.

        Args:
            hook_fn: Hook function to register
            module_str: Module path (e.g., 'blocks-26')
            hook_name: Name for this hook handle
        """
        module = self._module_dict[module_str]
        hook_handle = module.register_forward_hook(hook_fn)
        self.hooks[hook_name] = hook_handle

    def _build_caching_hook(self, module_str: str) -> Callable:
        """Build a hook function that caches activations."""
        self.activations_cache[module_str] = []

        def hook_fn(model, input, output):
            self.activations_cache[module_str].append(output)

        return hook_fn

    def add_caching_hook(self, module_str: str) -> None:
        """Adds an activations caching hook at the location in module_str."""
        hook_fn = self._build_caching_hook(module_str)
        self.add_hook(hook_fn, module_str, 'cache-' + module_str)

    def clear_cache(self, module_str: str) -> None:
        """Clears the activations cache corresponding to module_str."""
        if module_str not in self.activations_cache:
            raise KeyError(f'No activations cache for {module_str}.')
        self.activations_cache[module_str] = []

    def clear_all_caches(self) -> None:
        """Clear all activation caches."""
        for module_str in self.activations_cache:
            self.clear_cache(module_str)

    def remove_cache(self, module_str: str) -> None:
        """Remove the cache for module_str."""
        del self.activations_cache[module_str]

    def remove_all_caches(self) -> None:
        """Remove all caches."""
        caches = list(self.activations_cache.keys())
        for cache_str in caches:
            self.remove_cache(cache_str)

    def _build_override_hook(self, module_str: str) -> Callable:
        """Build a hook that overrides output with stored tensor."""
        self.override_store[module_str] = None

        def hook_fn(model, input, output):
            return self.override_store[module_str]

        return hook_fn

    def add_override_hook(self, module_str: str) -> None:
        """Adds hook to override output of module_str using override_store."""
        hook_fn = self._build_override_hook(module_str)
        self.add_hook(hook_fn, module_str, 'override-' + module_str)

    def override(self, module_str: str, override_tensor: torch.Tensor) -> None:
        """Sets the override tensor for module_str."""
        self.override_store[module_str] = override_tensor

    def clear_override(self, module_str: str) -> None:
        """Clear override hook so it won't affect forward pass."""
        self.override_store[module_str] = None

    def clear_all_overrides(self) -> None:
        """Clear all override hooks."""
        for override in list(self.override_store.keys()):
            self.clear_override(override)

    def remove_hook(self, hook_name: str) -> None:
        """Remove a hook with name hook_name from the model."""
        self.hooks[hook_name].remove()
        del self.hooks[hook_name]

    def remove_all_hooks(self) -> None:
        """Remove all hooks from the model."""
        hooks = list(self.hooks.keys())
        for hook_name in hooks:
            self.remove_hook(hook_name)


# Type alias for intervention functions
INTERVENTION_INTERFACE = Callable[[torch.Tensor], torch.Tensor]


# =============================================================================
# OBSERVABLE EVO2 - Model Wrapper with Activation Caching
# =============================================================================

class ObservableEvo2:
    """
    Wrapper around Evo2 model that enables activation extraction and intervention.

    This class wraps an Evo2 model and provides:
    - Forward pass with activation caching at specified layers
    - Intervention capabilities for activation manipulation
    - Access to tokenizer and model properties

    Example:
        model = ObservableEvo2("evo2_7b")
        logits, acts = model.forward(tokens, cache_activations_at=['blocks-26'])
        # acts['blocks-26'] contains layer 26 activations

    Attributes:
        model_name: Name of the Evo2 model
        evo_model: The underlying Evo2 model instance
        scope: ModelScope for hook management
        tokenizer: Evo2 tokenizer
        model: Raw PyTorch model
        d_hidden: Hidden dimension size (4096 for evo2_7b)
    """

    def __init__(self, model_name: str = "evo2_7b"):
        """
        Initialize ObservableEvo2 with a model.

        Args:
            model_name: Evo2 model name (e.g., "evo2_7b", "evo2_7b_262k")
        """
        if Evo2 is None:
            raise ImportError("Evo2 not available. Please install the evo2 package.")

        self.model_name = model_name
        self.evo_model = Evo2(model_name)
        self.scope = ModelScope(self.evo_model.model)
        self.tokenizer = self.evo_model.tokenizer
        self.model = self.evo_model.model
        self.d_hidden = 4096

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.evo_model.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the model's dtype."""
        return next(self.evo_model.model.parameters()).dtype

    def list_modules(self) -> List[str]:
        """List all hookable modules in the model."""
        return self.scope.list_modules()

    def forward(
        self,
        toks: torch.Tensor,
        cache_activations_at: Optional[List[str]] = None,
        interventions: Optional[Dict[str, INTERVENTION_INTERFACE]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with optional activation caching and interventions.

        Args:
            toks: Input token IDs tensor, shape (batch, seq_len)
            cache_activations_at: List of layer names to cache activations for
            interventions: Dict mapping layer names to intervention functions

        Returns:
            Tuple of:
            - model_outputs: Logits tensor from the model
            - cached_activations: Dict mapping layer names to activation tensors
        """
        if not interventions:
            interventions = {}

        if not cache_activations_at:
            cache_activations_at = []

        output_cache: Dict[str, torch.Tensor] = {}
        layers = list(set(list(interventions.keys()) + cache_activations_at))

        if layers:
            for layer in layers:
                def _intervene(model, input, output, layer=layer):
                    acts = output[0] if isinstance(output, tuple) else output

                    if layer in interventions:
                        acts = interventions[layer](acts)

                    if layer in cache_activations_at:
                        output_cache[layer] = acts.detach()

                    return (acts, output[1]) if isinstance(output, tuple) else acts

                self.scope.add_hook(_intervene, layer, f'intervene-{layer}')

        try:
            model_outputs = self.model(toks)
            cached_activations = {
                layer: act.clone() for layer, act in output_cache.items()
            }
        finally:
            self.scope.remove_all_hooks()
            self.scope.clear_all_caches()

        return model_outputs[0], cached_activations

    def generate(
        self,
        prompt_seqs: List[str],
        n_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.,
        batched: bool = True,
        cached_generation: bool = False,
        verbose: int = 0,
        cache_activations_at: Optional[List[str]] = None,
        interventions: Optional[Dict[str, INTERVENTION_INTERFACE]] = None,
    ) -> Tuple[str, Dict[str, torch.Tensor]]:
        """
        Generate sequence with optional activation caching.

        Args:
            prompt_seqs: List of prompt sequences
            n_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            batched: Whether to use batched generation
            cached_generation: Whether to use KV cache
            verbose: Verbosity level
            cache_activations_at: Layers to cache activations for
            interventions: Intervention functions by layer

        Returns:
            Tuple of (generated_sequence, cached_activations)
        """
        if not interventions:
            interventions = {}

        if not cache_activations_at:
            cache_activations_at = []

        output_cache: Dict[str, List[torch.Tensor]] = {}
        layers = list(set(list(interventions.keys()) + cache_activations_at))

        if layers:
            for layer in layers:
                def _intervene(model, input, output, layer=layer):
                    acts = output[0]

                    if layer in interventions:
                        acts = interventions[layer](acts)

                    if layer in cache_activations_at:
                        if output_cache.get(layer) is None:
                            output_cache[layer] = [acts]
                        else:
                            output_cache[layer].append(acts)

                    if len(output) == 2:
                        return (acts, output[1])
                    else:
                        return acts

                self.scope.add_hook(_intervene, layer, f'intervene-{layer}')

        try:
            output = self.evo_model.generate(
                prompt_seqs,
                n_tokens=n_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                batched=batched,
                cached_generation=cached_generation,
                verbose=verbose,
            )
        finally:
            self.scope.remove_all_hooks()
            self.scope.clear_all_caches()

        acts_cache = {
            layer: torch.cat(acts, dim=1).clone().detach()
            for layer, acts in output_cache.items()
        }

        return ''.join(output[0]), acts_cache


# =============================================================================
# BATCH TOPK TIED SAE - Sparse Autoencoder
# =============================================================================

class BatchTopKTiedSAE(nn.Module):
    """
    TopK Sparse Autoencoder with tied encoder/decoder weights.

    This SAE uses a TopK activation function to enforce sparsity:
    - Only the top K features (out of ~32K) are active per position
    - Encoder and decoder share the same weight matrix (tied)

    Architecture:
        Input: (batch, seq, d_in=4096)
        Encoded: (batch, seq, d_hidden=32768) with TopK sparsity
        Decoded: (batch, seq, d_in=4096)

    Attributes:
        d_in: Input dimension (4096 for Evo2)
        d_hidden: Number of SAE features (32768)
        k: Number of active features per position (64)
        W: Shared encoder/decoder weight matrix
        b_enc: Encoder bias
        b_dec: Decoder bias
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        k: int,
        device: torch.device,
        dtype: torch.dtype,
        tiebreaker_epsilon: float = 1e-6
    ):
        """
        Initialize the TopK SAE.

        Args:
            d_in: Input dimension
            d_hidden: Number of SAE features
            k: Number of active features per position
            device: Device to place the model on
            dtype: Data type for model parameters
            tiebreaker_epsilon: Small epsilon for tie-breaking
        """
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k

        # Initialize weight matrix with small random values
        W_mat = torch.randn((d_in, d_hidden))
        W_mat = 0.1 * W_mat / torch.linalg.norm(W_mat, dim=0, ord=2, keepdim=True)

        self.W = nn.Parameter(W_mat)
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))

        self.device = device
        self.dtype = dtype
        self.tiebreaker_epsilon = tiebreaker_epsilon
        self.tiebreaker = torch.linspace(0, tiebreaker_epsilon, d_hidden)

        self.to(self.device, self.dtype)

    def encoder_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pre-activation encoder output."""
        return x @ self.W + self.b_enc

    def encode(
        self,
        x: torch.Tensor,
        tiebreak: bool = False
    ) -> torch.Tensor:
        """
        Encode input to sparse feature activations.

        Args:
            x: Input tensor, shape (..., d_in)
            tiebreak: Whether to break ties deterministically

        Returns:
            Sparse feature activations, shape (..., d_hidden)
        """
        f = torch.nn.functional.relu(self.encoder_pre(x))
        return self._batch_topk(f, self.k, tiebreak=tiebreak)

    def _batch_topk(
        self,
        f: torch.Tensor,
        k: int,
        tiebreak: bool = False
    ) -> torch.Tensor:
        """Apply batch TopK to enforce sparsity."""
        from math import prod

        if tiebreak:
            f = f + self.tiebreaker.broadcast_to(f)

        *input_shape, _ = f.shape
        numel = k * prod(input_shape)

        f_topk = torch.topk(f.flatten(), numel, dim=-1)
        f_topk = torch.zeros_like(f.flatten()).scatter(
            -1, f_topk.indices, f_topk.values
        ).reshape(f.shape)

        return f_topk

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to input space."""
        return f @ self.W.T + self.b_dec

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.

        Returns:
            Tuple of (reconstruction, features)
        """
        f = self.encode(x)
        return self.decode(f), f


# =============================================================================
# SAE LOADING FUNCTIONS
# =============================================================================

def load_topk_sae(
    sae_path: str,
    d_hidden: int,
    device: torch.device,
    dtype: torch.dtype,
    expansion_factor: int = SAE_EXPANSION_FACTOR,
    k: int = SAE_TOPK,
) -> BatchTopKTiedSAE:
    """
    Load a pre-trained TopK SAE from a file path.

    Args:
        sae_path: Path to the SAE weights file (.pt)
        d_hidden: Hidden dimension of the model (4096)
        device: Device to load the SAE onto
        dtype: Data type for the SAE
        expansion_factor: Expansion factor (default: 8x)
        k: TopK value (default: 64)

    Returns:
        Loaded BatchTopKTiedSAE model
    """
    sae_dict = torch.load(sae_path, weights_only=True, map_location="cpu")

    # Clean up state dict keys (remove DDP/compile prefixes)
    new_dict = {}
    for key, item in sae_dict.items():
        new_key = key.replace("_orig_mod.", "").replace("module.", "")
        new_dict[new_key] = item

    sae = BatchTopKTiedSAE(
        d_hidden,
        d_hidden * expansion_factor,
        k,
        device,
        dtype,
    )
    sae.load_state_dict(new_dict)

    return sae


def load_topk_sae_from_hf(
    d_hidden: int = 4096,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    repo_id: str = SAE_REPO_ID,
    filename: str = SAE_FILENAME,
    expansion_factor: int = SAE_EXPANSION_FACTOR,
    k: int = SAE_TOPK,
) -> BatchTopKTiedSAE:
    """
    Load the pre-trained SAE from HuggingFace Hub.

    Downloads the SAE weights from the Goodfire model repository and
    loads them into a BatchTopKTiedSAE model.

    Args:
        d_hidden: Hidden dimension (4096 for Evo2)
        device: Device to load onto (default: cuda if available)
        dtype: Data type (default: bfloat16)
        repo_id: HuggingFace repo ID
        filename: SAE weights filename
        expansion_factor: Expansion factor (8x)
        k: TopK value (64)

    Returns:
        Loaded SAE model ready for feature extraction
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[SAE] Downloading SAE weights from {repo_id}...")
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model"
    )
    print(f"[SAE] Loading SAE from {file_path}")

    return load_topk_sae(
        sae_path=file_path,
        d_hidden=d_hidden,
        device=device,
        dtype=dtype,
        expansion_factor=expansion_factor,
        k=k,
    )


# =============================================================================
# FEATURE EXTRACTION FUNCTIONS
# =============================================================================

def get_feature_ts(
    model: ObservableEvo2,
    sae: BatchTopKTiedSAE,
    seq: str,
    layer_name: str = SAE_LAYER_NAME,
) -> np.ndarray:
    """
    Extract SAE feature time series for a DNA sequence.

    This function:
    1. Tokenizes the sequence
    2. Runs Evo2 forward pass, caching layer 26 activations
    3. Encodes activations through the SAE
    4. Returns sparse feature activations as numpy array

    Args:
        model: ObservableEvo2 model instance
        sae: Loaded BatchTopKTiedSAE instance
        seq: DNA sequence string
        layer_name: Layer to extract activations from (default: 'blocks-26')

    Returns:
        Feature activations, shape (seq_len, 32768)
        Each position has exactly 64 non-zero features (TopK=64)
    """
    # Tokenize sequence
    toks = model.tokenizer.tokenize(seq)
    toks = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(model.device)

    # Compile SAE encoder once on first call for fused kernels
    if not getattr(sae, '_encode_compiled', False):
        try:
            sae.encode = torch.compile(sae.encode)
            sae._encode_compiled = True
        except Exception:
            sae._encode_compiled = True  # Don't retry on failure

    # Forward pass with activation caching (inference_mode required
    # because model tensors are created under inference mode)
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, acts = model.forward(toks, cache_activations_at=[layer_name])

            # Encode through SAE (move acts to SAE device — layer 26 may be on a different GPU)
            sae_device = next(iter(sae.parameters())).device
            features = sae.encode(acts[layer_name][0].to(sae_device))

    return features.cpu().detach().float().numpy()


def get_feature_ts_batch(
    model: ObservableEvo2,
    sae: BatchTopKTiedSAE,
    seqs: List[str],
    layer_name: str = SAE_LAYER_NAME,
) -> List[np.ndarray]:
    """
    Extract SAE feature time series for a batch of DNA sequences in one forward pass.

    Sequences are right-padded to the same length within the batch. Because Evo2
    uses causal (autoregressive) attention, padding tokens appended after a sequence
    never affect activations at earlier positions, so this is equivalent to running
    each sequence individually.

    Args:
        model: ObservableEvo2 instance
        sae: BatchTopKTiedSAE instance
        seqs: List of DNA sequence strings (variable length)
        layer_name: Layer to extract activations from (default: 'blocks-26')

    Returns:
        List of numpy arrays, one per sequence, each shape (seq_len_i, 32768).
        Order matches the input seqs list.
    """
    if len(seqs) == 1:
        return [get_feature_ts(model, sae, seqs[0], layer_name)]

    # Compile SAE encoder once on first call
    if not getattr(sae, '_encode_compiled', False):
        try:
            sae.encode = torch.compile(sae.encode)
            sae._encode_compiled = True
        except Exception:
            sae._encode_compiled = True

    # Tokenize all sequences; track real lengths for slicing after forward pass
    tokenized = [model.tokenizer.tokenize(seq) for seq in seqs]
    lengths = [len(t) for t in tokenized]
    max_len = max(lengths)

    # Right-pad with token 0 (safe for causal models: padding after real content
    # is never attended to by any real position)
    padded = [t + [0] * (max_len - len(t)) for t in tokenized]
    toks = torch.tensor(padded, dtype=torch.long).to(model.device)  # (B, max_len)

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, acts = model.forward(toks, cache_activations_at=[layer_name])

            sae_device = next(iter(sae.parameters())).device
            layer_acts = acts[layer_name].to(sae_device)  # (B, max_len, d_hidden)
            features = sae.encode(layer_acts)              # (B, max_len, 32768)

    features_cpu = features.cpu().detach().float().numpy()

    # Slice out only real (non-padded) positions for each sequence
    return [features_cpu[i, :lengths[i], :] for i in range(len(seqs))]


def get_feature_ts_via_generate(
    model: ObservableEvo2,
    sae: BatchTopKTiedSAE,
    seq: str,
    layer_name: str = SAE_LAYER_NAME,
) -> np.ndarray:
    """
    Extract SAE features using generation (slower but more stable).

    Alternative to get_feature_ts that uses cached generation.
    May be more memory-stable for very long sequences.

    Args:
        model: ObservableEvo2 model instance
        sae: Loaded SAE instance
        seq: DNA sequence string
        layer_name: Layer to extract from

    Returns:
        Feature activations, shape (seq_len, 32768)
    """
    logits, acts = model.generate(
        [seq],
        n_tokens=1,
        cached_generation=True,
        cache_activations_at=[layer_name]
    )
    features = sae.encode(acts[layer_name][0])
    return features.cpu().detach().float().numpy()


# =============================================================================
# DROP ANALYSIS FUNCTIONS
# =============================================================================

def parse_drops_file(drops_file: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Parse drop positions from genome_scoring output file.

    Reads the .drops.txt file format produced by genome_scoring_jan26_drops.py

    Args:
        drops_file: Path to the drops file

    Returns:
        Dict mapping method names to list of (position, score) tuples
    """
    drops = {}
    with open(drops_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue

            method = parts[0].strip()
            positions_str = parts[1].strip()

            if not positions_str:
                drops[method] = []
                continue

            positions = []
            for item in positions_str.split(','):
                item = item.strip()
                if ':' in item:
                    pos, score = item.split(':')
                    positions.append((int(pos), float(score)))
                else:
                    positions.append((int(item), 0.0))

            drops[method] = positions

    return drops


def parse_chromosome_drops_tsv(
    boundaries_file: str,
    min_confidence: float = 0.0,
    max_regions: int = 0,
) -> List[Dict[str, Any]]:
    """
    Parse .drop_boundaries.tsv from score_chromosome.py.

    Reads the paired drop-rise region format and returns structured data
    compatible with the SAE analysis pipeline.

    Args:
        boundaries_file: Path to .drop_boundaries.tsv file
        min_confidence: Minimum start_confidence to include (0 = no filter)
        max_regions: Maximum number of regions to return, sorted by confidence
                     (0 = no cap)

    Returns:
        List of region dicts with keys:
        - chrom, drop_start, drop_end, genomic_start, genomic_end
        - region_length, method, start_confidence, end_confidence
        - mean_entropy, min_entropy
    """
    regions = []

    with open(boundaries_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            # Header line detection
            if parts[0] == 'chrom':
                continue

            if len(parts) < 11:
                continue

            region = {
                'chrom': parts[0],
                'drop_start': int(parts[1]),
                'drop_end': int(parts[2]),
                'genomic_start': int(parts[3]),
                'genomic_end': int(parts[4]),
                'region_length': int(parts[5]),
                'method': parts[6],
                'start_confidence': float(parts[7]),
                'end_confidence': float(parts[8]),
                'mean_entropy': float(parts[9]),
                'min_entropy': float(parts[10]),
            }

            if region['start_confidence'] >= min_confidence:
                regions.append(region)

    # Sort by confidence (highest first)
    regions.sort(key=lambda r: -r['start_confidence'])

    # Cap at max_regions if specified
    if max_regions > 0 and len(regions) > max_regions:
        regions = regions[:max_regions]

    return regions


def extract_regions_around_drops(
    full_sequence: str,
    drop_positions: List[Tuple[int, float]],
    window: int = 500,
) -> List[Dict[str, Any]]:
    """
    Extract sequence windows around drop positions.

    Args:
        full_sequence: Complete DNA sequence
        drop_positions: List of (position, score) tuples
        window: Number of bp to include on each side

    Returns:
        List of region dicts with keys:
        - pos: Original drop position
        - score: Drop confidence score
        - seq: Sequence window
        - local_pos: Position of drop within window
        - start: Window start in original sequence
        - end: Window end in original sequence
    """
    regions = []
    for pos, score in drop_positions:
        start = max(0, pos - window)
        end = min(len(full_sequence), pos + window)

        regions.append({
            'pos': pos,
            'score': score,
            'seq': full_sequence[start:end],
            'local_pos': pos - start,
            'start': start,
            'end': end,
        })

    return regions


def analyze_drops_with_sae(
    regions: List[Dict[str, Any]],
    model: ObservableEvo2,
    sae: BatchTopKTiedSAE,
    layer_name: str = SAE_LAYER_NAME,
) -> List[Dict[str, Any]]:
    """
    Run SAE analysis on extracted drop regions.

    For each region around a drop:
    1. Extract SAE features for the sequence window
    2. Identify which features are active at the drop position
    3. Record feature IDs and activation strengths

    Args:
        regions: List of region dicts from extract_regions_around_drops
        model: ObservableEvo2 model instance
        sae: Loaded SAE instance
        layer_name: Layer to extract from

    Returns:
        List of result dicts, each containing:
        - pos: Original drop position
        - score: Drop confidence score
        - active_features: List of (feature_id, activation) tuples
        - feature_ts: Full feature time series for the window
        - local_pos: Position within window
    """
    results = []

    for i, region in enumerate(regions):
        print(f"[SAE] Analyzing region {i+1}/{len(regions)} at position {region['pos']}...")

        # Get features for this region
        feature_ts = get_feature_ts(model, sae, region['seq'], layer_name)

        # Get features active at the drop position
        local_pos = region['local_pos']
        if local_pos < len(feature_ts):
            drop_features = feature_ts[local_pos, :]

            # Find non-zero features
            active_idx = np.where(drop_features > 0)[0]
            active_vals = drop_features[active_idx]

            # Sort by activation strength
            sorted_order = np.argsort(active_vals)[::-1]
            active_features = [
                (int(active_idx[i]), float(active_vals[i]))
                for i in sorted_order
            ]
        else:
            active_features = []

        results.append({
            'pos': region['pos'],
            'score': region['score'],
            'active_features': active_features,
            'feature_ts': feature_ts,
            'local_pos': local_pos,
            'window_start': region['start'],
            'window_end': region['end'],
        })

    return results


def find_signature_features(
    results: List[Dict[str, Any]],
    min_prevalence: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Find features that consistently activate at drop positions.

    Aggregates across all analyzed drops to find features that appear
    frequently, suggesting they may be biologically significant.

    Args:
        results: List of result dicts from analyze_drops_with_sae
        min_prevalence: Minimum fraction of drops where feature must appear

    Returns:
        List of signature feature dicts, sorted by mean activation:
        - feature_id: SAE feature index
        - count: Number of drops where feature is active
        - prevalence: Fraction of drops with this feature
        - mean_activation: Average activation when present
        - max_activation: Maximum activation observed
        - positions: List of positions where feature appeared
    """
    feature_stats: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {'activations': [], 'positions': []}
    )

    for result in results:
        for feat_id, activation in result['active_features']:
            feature_stats[feat_id]['activations'].append(activation)
            feature_stats[feat_id]['positions'].append(result['pos'])

    n_drops = len(results)
    min_count = max(1, int(n_drops * min_prevalence))

    signature_features = []
    for feat_id, stats in feature_stats.items():
        count = len(stats['activations'])
        if count >= min_count:
            signature_features.append({
                'feature_id': feat_id,
                'count': count,
                'prevalence': count / n_drops,
                'mean_activation': np.mean(stats['activations']),
                'max_activation': np.max(stats['activations']),
                'positions': stats['positions'],
            })

    # Sort by mean activation (descending)
    signature_features.sort(key=lambda x: -x['mean_activation'])

    return signature_features


def write_sae_analysis_output(
    results: List[Dict[str, Any]],
    signature_features: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Write SAE analysis results to a TSV file.

    Creates a file with one row per drop, showing the top features
    active at each drop position.

    Args:
        results: Analysis results from analyze_drops_with_sae
        signature_features: Signature features from find_signature_features
        output_path: Path for output file
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write("# SAE Feature Analysis of High-Confidence Drops\n")
        f.write(f"# Total drops analyzed: {len(results)}\n")
        f.write(f"# Signature features (>30% prevalence): {len(signature_features)}\n")
        f.write("#\n")

        # Write signature features summary
        f.write("# Top Signature Features:\n")
        for sf in signature_features[:20]:
            f.write(f"#   Feature {sf['feature_id']}: "
                   f"prevalence={sf['prevalence']:.1%}, "
                   f"mean_act={sf['mean_activation']:.2f}\n")
        f.write("#\n")

        # Write per-drop analysis
        f.write("position\tscore\ttop_features\n")
        for result in results:
            pos = result['pos']
            score = result['score']

            # Format top 10 features as feature_id:activation pairs
            top_features = result['active_features'][:10]
            features_str = ','.join(
                f"{fid}:{act:.2f}" for fid, act in top_features
            )

            f.write(f"{pos}\t{score:.4f}\t{features_str}\n")

    print(f"[SAE] Wrote analysis to {output_path}")


# =============================================================================
# CONVENIENCE FUNCTION FOR FULL ANALYSIS
# =============================================================================

def analyze_drops_from_file(
    sequence: str,
    drops_file: str,
    output_path: str,
    model: Optional[ObservableEvo2] = None,
    sae: Optional[BatchTopKTiedSAE] = None,
    method: str = "zscore",
    window: int = 500,
    max_drops: int = 100,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run complete SAE analysis on drops from a file.

    Convenience function that performs the full analysis pipeline:
    1. Parse drops file
    2. Extract regions around drops
    3. Initialize model and SAE if not provided
    4. Run SAE analysis
    5. Find signature features
    6. Write output file

    Args:
        sequence: Full DNA sequence
        drops_file: Path to drops file
        output_path: Path for output file
        model: Optional pre-initialized ObservableEvo2
        sae: Optional pre-initialized SAE
        method: Detection method to analyze (default: "zscore")
        window: Window size around drops (default: 500)
        max_drops: Maximum number of drops to analyze (default: 100)

    Returns:
        Tuple of (results, signature_features)
    """
    # Parse drops
    print(f"[SAE] Parsing drops from {drops_file}...")
    all_drops = parse_drops_file(drops_file)

    if method not in all_drops:
        available = list(all_drops.keys())
        raise ValueError(f"Method '{method}' not found. Available: {available}")

    drop_positions = all_drops[method]
    print(f"[SAE] Found {len(drop_positions)} drops using method '{method}'")

    # Limit number of drops
    if len(drop_positions) > max_drops:
        print(f"[SAE] Limiting to top {max_drops} drops by score")
        drop_positions = sorted(drop_positions, key=lambda x: x[1])[:max_drops]

    # Extract regions
    print(f"[SAE] Extracting ±{window}bp windows around drops...")
    regions = extract_regions_around_drops(sequence, drop_positions, window)

    # Initialize model/SAE if needed
    if model is None:
        print("[SAE] Initializing ObservableEvo2...")
        model = ObservableEvo2("evo2_7b")

    if sae is None:
        print("[SAE] Loading SAE from HuggingFace...")
        sae = load_topk_sae_from_hf(
            d_hidden=model.d_hidden,
            device=model.device,
            dtype=torch.bfloat16,
        )

    # Run analysis
    print("[SAE] Running SAE analysis on drop regions...")
    results = analyze_drops_with_sae(regions, model, sae)

    # Find signature features
    print("[SAE] Finding signature features...")
    signature_features = find_signature_features(results)
    print(f"[SAE] Found {len(signature_features)} signature features")

    # Write output
    write_sae_analysis_output(results, signature_features, output_path)

    return results, signature_features


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("SAE Utils Module")
    print("================")
    print(f"SAE Layer: {SAE_LAYER_NAME}")
    print(f"SAE Model: {SAE_REPO_ID}")
    print(f"Expansion Factor: {SAE_EXPANSION_FACTOR}x")
    print(f"TopK: {SAE_TOPK}")
    print()
    print("To use this module:")
    print("  from sae_utils import ObservableEvo2, load_topk_sae_from_hf, get_feature_ts")
    print("  ")
    print("  model = ObservableEvo2('evo2_7b')")
    print("  sae = load_topk_sae_from_hf(model.d_hidden, model.device, model.dtype)")
    print("  features = get_feature_ts(model, sae, 'ACGTACGT...')")
