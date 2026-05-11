"""
aggregation.py — hidden-state aggregation for the SMILES submission.

The default path is intentionally hidden-state-only and keeps the public
``solution.py`` contract unchanged.  Stable ablation variants can be selected
with ``SMILES_EXPERIMENT_VARIANT`` when reproducing the report experiments:

    final                  mid/late layers with response-tail pools
    baseline_last_token    starter-style final-layer last-token feature
    final_last_token_lr    same feature shape as baseline, for linear probing
    tail_no_second         final minus second-last-token pools
    tail_with_geometry     final plus small norm/cosine/length scalars
    tail_with_var          default; final plus response-tail standard-deviation pools
    tail_minmax            final plus response-tail min/max pools

The variants are deliberately few: they are report-worthy checkpoints, not a
general experiment framework.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


HIDDEN_DIM = 896
SELECTED_LAYERS: tuple[int, ...] = (12, 16, 20, 24)
TAIL_WINDOWS: tuple[int, ...] = (16, 32, 64)
EXPERIMENT_VARIANT = os.getenv("SMILES_EXPERIMENT_VARIANT", "tail_with_var").strip().lower()

_BASELINE_VARIANTS = {"baseline_last_token", "final_last_token_lr"}
_TAIL_VARIANTS = {
    "final",
    "tail_no_second",
    "tail_with_geometry",
    "tail_with_var",
    "tail_minmax",
}
_VALID_VARIANTS = _BASELINE_VARIANTS | _TAIL_VARIANTS


def _validate_variant() -> str:
    if EXPERIMENT_VARIANT not in _VALID_VARIANTS:
        allowed = ", ".join(sorted(_VALID_VARIANTS))
        raise ValueError(
            f"Unknown SMILES_EXPERIMENT_VARIANT={EXPERIMENT_VARIANT!r}. "
            f"Expected one of: {allowed}."
        )
    return EXPERIMENT_VARIANT


def _real_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    idx = attention_mask.to(torch.bool).nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return torch.tensor([0], dtype=torch.long, device=attention_mask.device)
    return idx


def _last_real_token(layer: torch.Tensor, real_idx: torch.Tensor) -> torch.Tensor:
    return layer[real_idx[-1].to(layer.device)]


def _second_last_real_token(layer: torch.Tensor, real_idx: torch.Tensor) -> torch.Tensor:
    pos = real_idx[-2] if real_idx.numel() >= 2 else real_idx[-1]
    return layer[pos.to(layer.device)]


def _masked_mean(layer: torch.Tensor, real_idx: torch.Tensor) -> torch.Tensor:
    return layer.index_select(0, real_idx.to(layer.device)).mean(dim=0)


def _tail_mean(layer: torch.Tensor, real_idx: torch.Tensor, window: int) -> torch.Tensor:
    # Drop the terminal EOS/chat marker when possible.  Its representation is
    # often dominated by special-token behavior rather than answer content.
    usable = real_idx[:-1] if real_idx.numel() > 1 else real_idx
    tail = usable[-min(window, usable.numel()) :]
    return layer.index_select(0, tail.to(layer.device)).mean(dim=0)


def _tail_std(layer: torch.Tensor, real_idx: torch.Tensor, window: int) -> torch.Tensor:
    usable = real_idx[:-1] if real_idx.numel() > 1 else real_idx
    tail = usable[-min(window, usable.numel()) :]
    states = layer.index_select(0, tail.to(layer.device))
    return states.std(dim=0, unbiased=False)


def _tail_min(layer: torch.Tensor, real_idx: torch.Tensor, window: int) -> torch.Tensor:
    usable = real_idx[:-1] if real_idx.numel() > 1 else real_idx
    tail = usable[-min(window, usable.numel()) :]
    states = layer.index_select(0, tail.to(layer.device))
    return states.min(dim=0).values


def _tail_max(layer: torch.Tensor, real_idx: torch.Tensor, window: int) -> torch.Tensor:
    usable = real_idx[:-1] if real_idx.numel() > 1 else real_idx
    tail = usable[-min(window, usable.numel()) :]
    states = layer.index_select(0, tail.to(layer.device))
    return states.max(dim=0).values


def _tail_feature_names() -> tuple[str, ...]:
    variant = _validate_variant()
    base = ["mean", "last", "lastK16", "lastK32", "lastK64"]
    if variant != "tail_no_second":
        base.insert(2, "second_last")
    if variant == "tail_with_var":
        base.extend(["stdK16", "stdK32", "stdK64"])
    if variant == "tail_minmax":
        base.extend(["minK32", "maxK32"])
    return tuple(base)


def feature_dim() -> int:
    """Feature dimension emitted by ``aggregate`` for the active variant."""
    variant = _validate_variant()
    if variant in _BASELINE_VARIANTS:
        return HIDDEN_DIM
    dim = len(SELECTED_LAYERS) * len(_tail_feature_names()) * HIDDEN_DIM
    if variant == "tail_with_geometry":
        dim += geometric_feature_dim()
    return dim


def geometric_feature_dim() -> int:
    """Small scalar geometry block used only by the geometry ablation."""
    # Per selected layer: mean norm, last norm, tail32 norm, mean/last cosine.
    per_layer = 4 * len(SELECTED_LAYERS)
    # Adjacent selected-layer cosine drift for mean-pooled states.
    across_layers = len(SELECTED_LAYERS) - 1
    # Length-like scalars: n_real, log n_real, tail fraction.
    length = 3
    return per_layer + across_layers + length


def pool_indices(pool_name: str) -> list[int]:
    """Column indices for all selected layers for one pool in tail variants."""
    pools = _tail_feature_names()
    if pool_name not in pools:
        raise KeyError(pool_name)
    pool_pos = pools.index(pool_name)
    cols: list[int] = []
    block = len(pools) * HIDDEN_DIM
    for layer_pos in range(len(SELECTED_LAYERS)):
        start = layer_pos * block + pool_pos * HIDDEN_DIM
        cols.extend(range(start, start + HIDDEN_DIM))
    return cols


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into one feature vector."""
    variant = _validate_variant()
    real_idx = _real_indices(attention_mask)

    if variant in _BASELINE_VARIANTS:
        return _last_real_token(hidden_states[-1], real_idx)

    features: list[torch.Tensor] = []
    include_second = variant != "tail_no_second"
    for layer_idx in SELECTED_LAYERS:
        layer = hidden_states[layer_idx]
        features.append(_masked_mean(layer, real_idx))
        features.append(_last_real_token(layer, real_idx))
        if include_second:
            features.append(_second_last_real_token(layer, real_idx))
        for window in TAIL_WINDOWS:
            features.append(_tail_mean(layer, real_idx, window))
        if variant == "tail_with_var":
            for window in TAIL_WINDOWS:
                features.append(_tail_std(layer, real_idx, window))
        if variant == "tail_minmax":
            features.append(_tail_min(layer, real_idx, 32))
            features.append(_tail_max(layer, real_idx, 32))

    if variant == "tail_with_geometry":
        features.append(extract_geometric_features(hidden_states, attention_mask))

    return torch.cat(features, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return a compact norm/cosine/length block for the geometry ablation."""
    real_idx = _real_indices(attention_mask)
    mean_vectors: list[torch.Tensor] = []
    scalars: list[torch.Tensor] = []

    for layer_idx in SELECTED_LAYERS:
        layer = hidden_states[layer_idx]
        mean_vec = _masked_mean(layer, real_idx)
        last_vec = _last_real_token(layer, real_idx)
        tail_vec = _tail_mean(layer, real_idx, 32)
        mean_vectors.append(mean_vec)
        scalars.extend(
            [
                mean_vec.norm().unsqueeze(0),
                last_vec.norm().unsqueeze(0),
                tail_vec.norm().unsqueeze(0),
                F.cosine_similarity(mean_vec, last_vec, dim=0).unsqueeze(0),
            ]
        )

    for left, right in zip(mean_vectors, mean_vectors[1:], strict=False):
        scalars.append(F.cosine_similarity(left, right, dim=0).unsqueeze(0))

    n_real = float(real_idx.numel())
    length = hidden_states.new_tensor(
        [n_real / 512.0, torch.log1p(hidden_states.new_tensor(n_real)).item(), min(n_real, 64.0) / max(n_real, 1.0)]
    )
    scalars.append(length)
    return torch.cat(scalars, dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append legacy geometric features.

    The default solution keeps ``USE_GEOMETRIC=False`` in ``solution.py``.  The
    ``tail_with_geometry`` ablation includes its scalar features inside
    ``aggregate`` so it can be reproduced without editing fixed infrastructure.
    """
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric and _validate_variant() != "tail_with_geometry":
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
