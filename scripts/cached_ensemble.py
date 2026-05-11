"""Cached probability ensembling over existing SMILES variant feature caches.

This is an experiment runner, not the official submission entrypoint.  It loads
feature caches produced by ``scripts/cached_solution.py``, trains one
``HallucinationProbe`` per variant on each split, averages probabilities, and
calibrates a single threshold.

Examples:

    python scripts/cached_ensemble.py --variants final,tail_with_var
    python scripts/cached_ensemble.py --variants final,tail_with_var,tail_minmax --threshold-mode train_prior
    SMILES_SPLIT_MODE=context_group python scripts/cached_ensemble.py --variants final,tail_with_var
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import MAX_LENGTH


DATA_FILE = Path("data/dataset.csv")
CACHE_DIR = Path("artifacts/feature_cache")
ENSEMBLE_DIR = Path("artifacts/ensembles")
RANDOM_STATE = 42


@dataclass
class VariantModel:
    variant: str
    model: object


def _cache_path(variant: str) -> Path:
    return CACHE_DIR / f"features__{variant}__maxlen-{MAX_LENGTH}.npz"


def _load_cache(variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = _cache_path(variant)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cache for {variant!r}: {path}. "
            "Run scripts/cached_solution.py for that variant first."
        )
    data = np.load(path, allow_pickle=False)
    return data["X"], data["X_test"], data["y"].astype(int), data["test_ids"]


def _threshold_by_metric(probs: np.ndarray, y: np.ndarray, mode: str) -> float:
    mode = "f1" if mode == "auto" else mode
    if mode == "train_prior":
        target_pos = float(np.mean(y))
        if target_pos <= 0.0:
            return float(np.nextafter(probs.max(), np.inf))
        if target_pos >= 1.0:
            return float(np.nextafter(probs.min(), -np.inf))
        return float(np.quantile(probs, 1.0 - target_pos))

    candidates = np.unique(np.concatenate([probs, np.linspace(0.05, 0.95, 181)]))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        pred = (probs >= threshold).astype(int)
        if mode == "f1":
            score = f1_score(y, pred, zero_division=0)
        elif mode == "accuracy":
            score = accuracy_score(y, pred)
        else:
            raise ValueError(f"Unsupported threshold mode for ensemble: {mode}")
        if score > best_score or (
            score == best_score
            and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def _fit_variant(variant: str, X: np.ndarray, y: np.ndarray):
    os.environ["SMILES_EXPERIMENT_VARIANT"] = variant
    import aggregation
    import probe

    importlib.reload(aggregation)
    importlib.reload(probe)
    return probe.HallucinationProbe().fit(X, y)


def _fit_ensemble(
    variants: list[str],
    train_indices: np.ndarray,
    caches: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> list[VariantModel]:
    models = []
    for variant in variants:
        X, _X_test, y, _ids = caches[variant]
        models.append(VariantModel(variant, _fit_variant(variant, X[train_indices], y[train_indices])))
    return models


def _predict_proba(
    models: list[VariantModel],
    caches: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    indices: np.ndarray | None = None,
    test: bool = False,
) -> np.ndarray:
    probs = []
    for item in models:
        X, X_test, _y, _ids = caches[item.variant]
        matrix = X_test if test else X
        view = matrix if indices is None else matrix[indices]
        probs.append(item.model.predict_proba(view)[:, 1])
    return np.stack(probs, axis=0).mean(axis=0)


def _evaluate(
    variants: list[str],
    threshold_mode: str,
    repeats: int,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    os.environ["SMILES_SPLIT_REPEATS"] = str(repeats)
    import splitting

    importlib.reload(splitting)

    caches = {variant: _load_cache(variant) for variant in variants}
    y = next(iter(caches.values()))[2]
    ids = next(iter(caches.values()))[3]
    df = pd.read_csv(DATA_FILE)
    splits = splitting.split_data(y, df)
    fold_results = []

    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(splits, start=1):
        models = _fit_ensemble(variants, idx_train, caches)
        if idx_val is not None:
            val_probs = _predict_proba(models, caches, idx_val)
            threshold = _threshold_by_metric(val_probs, y[idx_val], threshold_mode)
        else:
            train_probs = _predict_proba(models, caches, idx_train)
            threshold = _threshold_by_metric(train_probs, y[idx_train], threshold_mode)

        result = {"fold": fold_idx, "threshold": threshold}
        for split_name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
            if idx is None:
                continue
            probs = _predict_proba(models, caches, idx)
            pred = (probs >= threshold).astype(int)
            result[f"{split_name}_accuracy"] = accuracy_score(y[idx], pred)
            result[f"{split_name}_f1"] = f1_score(y[idx], pred, zero_division=0)
            result[f"{split_name}_auroc"] = roc_auc_score(y[idx], probs)
        fold_results.append(result)

    idx_non_test = np.unique(
        np.concatenate(
            [
                np.concatenate([idx_train, idx_val]) if idx_val is not None else idx_train
                for idx_train, idx_val, _idx_test in splits
            ]
        )
    )
    final_models = _fit_ensemble(variants, idx_non_test, caches)
    train_probs = _predict_proba(final_models, caches, idx_non_test)
    final_threshold = _threshold_by_metric(train_probs, y[idx_non_test], threshold_mode)
    test_probs = _predict_proba(final_models, caches, test=True)
    test_pred = (test_probs >= final_threshold).astype(int)
    return fold_results, ids, test_pred


def _summary(fold_results: list[dict]) -> dict:
    out = {"folds": fold_results}
    for split in ["train", "val", "test"]:
        for metric in ["accuracy", "f1", "auroc"]:
            key = f"{split}_{metric}"
            values = [row.get(key, np.nan) for row in fold_results]
            out[f"avg_{key}"] = float(np.nanmean(values))
    return out


def _run(args: argparse.Namespace) -> None:
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if len(variants) < 2:
        raise ValueError("Use at least two variants for an ensemble.")

    fold_results, ids, test_pred = _evaluate(
        variants=variants,
        threshold_mode=args.threshold_mode,
        repeats=args.repeats,
    )
    result = _summary(fold_results)
    result["variants"] = variants
    result["threshold_mode"] = args.threshold_mode
    result["split_mode"] = os.getenv("SMILES_SPLIT_MODE", "stratified").strip().lower()
    result["n_folds"] = len(fold_results)

    name = args.name or "__".join(variants)
    suffix = f"threshold-{args.threshold_mode}.repeats-{args.repeats}"
    split_mode = result["split_mode"]
    if split_mode != "stratified":
        suffix = f"{split_mode}.{suffix}"
    out_dir = ENSEMBLE_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"results.{suffix}.json"
    pred_path = out_dir / f"predictions.{suffix}.csv"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame({"id": ids, "label": test_pred}).to_csv(pred_path, index=False)

    print(
        f"Ensemble {name}: "
        f"Acc={result['avg_test_accuracy']:.4f} "
        f"F1={result['avg_test_f1']:.4f} "
        f"AUROC={result['avg_test_auroc']:.4f} "
        f"pred_pos={int(test_pred.sum())}"
    )
    print(f"Saved {result_path}")
    print(f"Saved {pred_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--threshold-mode",
        default="auto",
        choices=("auto", "f1", "accuracy", "train_prior"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    _run(_parse_args())
