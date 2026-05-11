"""
splitting.py — reproducible CV splits for the SMILES probe.

Default behavior is 5-fold stratified CV with a small validation carve-out from
each training side for threshold tuning.  Set ``SMILES_SPLIT_REPEATS`` to a
larger integer when producing final robustness tables for ``SOLUTION.md``.
Set ``SMILES_SPLIT_MODE=context_group`` for a leakage-aware diagnostic split by
source context.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split


def _extract_context(prompt: str) -> str:
    marker = "Given the context, answer the question in a single brief but complete sentence."
    if marker in prompt:
        prompt = prompt.split(marker, 1)[1]
    end_marker = "Note that your answer"
    if end_marker in prompt:
        return prompt.split(end_marker, 1)[0].strip()
    question_marker = "Here is the question:"
    if question_marker in prompt:
        return prompt.split(question_marker, 1)[0].strip()
    return str(prompt).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Return stratified train/validation/test folds.

    Args are kept compatible with the starter template.  ``test_size`` is not
    used by the default k-fold splitter because each fold acts as the held-out
    local test split; ``val_size`` controls the threshold-tuning carve-out from
    the fold's training side.
    """
    del test_size

    y = np.asarray(y, dtype=np.int32)
    idx = np.arange(len(y))
    n_splits = min(_env_int("SMILES_N_FOLDS", 5), int(np.bincount(y).min()))
    n_repeats = _env_int("SMILES_SPLIT_REPEATS", 1)
    if n_splits < 2:
        idx_train, idx_test = train_test_split(
            idx,
            test_size=0.2,
            random_state=random_state,
            stratify=y,
        )
        return [(idx_train, None, idx_test)]

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    split_mode = os.getenv("SMILES_SPLIT_MODE", "stratified").strip().lower()
    for repeat in range(n_repeats):
        seed = random_state + repeat
        if split_mode == "stratified":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            fold_iter = cv.split(idx, y)
        elif split_mode == "context_group":
            if df is None or "prompt" not in df.columns:
                raise ValueError("SMILES_SPLIT_MODE=context_group requires df['prompt'].")
            groups = df["prompt"].map(_extract_context).to_numpy()
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            fold_iter = cv.split(idx, y, groups=groups)
        else:
            raise ValueError(
                f"Unknown SMILES_SPLIT_MODE={split_mode!r}. "
                "Expected 'stratified' or 'context_group'."
            )

        for idx_train_val, idx_test in fold_iter:
            train_val_indices = idx[idx_train_val]
            if val_size <= 0:
                splits.append((train_val_indices, None, idx[idx_test]))
                continue
            idx_train, idx_val = train_test_split(
                train_val_indices,
                test_size=val_size,
                random_state=seed,
                stratify=y[train_val_indices],
            )
            splits.append((idx_train, idx_val, idx[idx_test]))
    return splits
