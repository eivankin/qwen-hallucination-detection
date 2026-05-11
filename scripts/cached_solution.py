"""Cached experiment runner for SMILES ablations.

This script intentionally does not replace the official `solution.py`.  Use it
for local experiments when you want to avoid repeated Qwen forward passes:

    python scripts/cached_solution.py --variant final
    python scripts/cached_solution.py --variant tail_no_second --repeats 3
    python scripts/cached_solution.py --variant tail_with_var --threshold-mode train_prior
    python scripts/cached_solution.py --variant final --rebuild-cache

It caches aggregated feature matrices under `artifacts/feature_cache/`.  This
is much smaller than caching raw hidden states and is enough for the stable
ablation variants supported by `aggregation.py` / `probe.py`.
"""

from __future__ import annotations

import argparse
import json
import importlib
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import aggregation
from evaluate import print_summary, run_evaluation, save_predictions, save_results
from model import MAX_LENGTH, get_model_and_tokenizer


DATA_FILE = Path("data/dataset.csv")
TEST_FILE = Path("data/test.csv")
CACHE_DIR = Path("artifacts/feature_cache")
ABLATION_DIR = Path("artifacts/ablations")
OUTPUT_FILE = "results.json"
PREDICTIONS_FILE = "predictions.csv"
DEFAULT_VARIANTS = (
    "final",
    "baseline_last_token",
    "final_last_token_lr",
    "tail_no_second",
    "tail_with_geometry",
    "tail_with_var",
    "tail_minmax",
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cache_path(variant: str) -> Path:
    return CACHE_DIR / f"features__{variant}__maxlen-{MAX_LENGTH}.npz"


def _set_variant(variant: str) -> None:
    os.environ["SMILES_EXPERIMENT_VARIANT"] = variant
    aggregation.EXPERIMENT_VARIANT = variant


def _texts_and_labels() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    df = pd.read_csv(DATA_FILE)
    texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    labels = np.array([int(float(label)) for label in df["label"]], dtype=np.int32)
    return df, texts, labels


def _test_texts() -> tuple[pd.DataFrame, list[str]]:
    df_test = pd.read_csv(TEST_FILE)
    texts = [f"{row['prompt']}{row['response']}" for _, row in df_test.iterrows()]
    return df_test, texts


def _aggregate_batch_for_variants(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    variants: tuple[str, ...],
    buckets: dict[str, list[torch.Tensor]],
) -> None:
    for sample_idx in range(hidden.size(0)):
        for variant in variants:
            _set_variant(variant)
            feat = aggregation.aggregation_and_feature_extraction(
                hidden[sample_idx],
                mask[sample_idx],
                use_geometric=False,
            )
            buckets[variant].append(feat.cpu())


def _extract_features(
    texts: list[str],
    variants: tuple[str, ...],
    batch_size: int,
    desc: str,
) -> tuple[dict[str, np.ndarray], float]:
    device = _device()
    print(f"Device: {device}")
    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    buckets: dict[str, list[torch.Tensor]] = {variant: [] for variant in variants}
    start_time = time.time()

    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch_texts = texts[start : start + batch_size]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        _aggregate_batch_for_variants(
            hidden=hidden,
            mask=attention_mask.cpu(),
            variants=variants,
            buckets=buckets,
        )

    elapsed = time.time() - start_time
    matrices = {
        variant: np.vstack([feat.numpy() for feat in features])
        for variant, features in buckets.items()
    }
    return matrices, elapsed


def _build_cache(variants: tuple[str, ...], batch_size: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _df, train_texts, labels = _texts_and_labels()
    df_test, test_texts = _test_texts()
    all_texts = train_texts + test_texts
    matrices, extract_time = _extract_features(
        texts=all_texts,
        variants=variants,
        batch_size=batch_size,
        desc="Extracting cached features",
    )
    n_train = len(train_texts)
    for variant, matrix in matrices.items():
        path = _cache_path(variant)
        np.savez_compressed(
            path,
            X=matrix[:n_train],
            X_test=matrix[n_train:],
            y=labels,
            test_ids=df_test.index.to_numpy(dtype=np.int64),
            extract_time=np.array([extract_time], dtype=np.float64),
            variant=np.array([variant]),
            max_length=np.array([MAX_LENGTH], dtype=np.int32),
        )
        print(f"Cached {variant}: {matrix.shape} -> {path}")


def _load_or_build_cache(
    variant: str,
    variants_to_extract: tuple[str, ...],
    batch_size: int,
    rebuild: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    path = _cache_path(variant)
    if rebuild or not path.exists():
        _build_cache(variants_to_extract, batch_size=batch_size)
    if not path.exists():
        raise FileNotFoundError(f"Feature cache was not created: {path}")
    data = np.load(path, allow_pickle=False)
    return (
        data["X"],
        data["X_test"],
        data["y"],
        data["test_ids"],
        float(data["extract_time"][0]),
    )


def _run_cached_solution(
    variant: str,
    repeats: int,
    batch_size: int,
    rebuild_cache: bool,
    extract_variants: tuple[str, ...],
    threshold_mode: str,
) -> None:
    if variant not in extract_variants:
        extract_variants = tuple(dict.fromkeys((*extract_variants, variant)))
    _set_variant(variant)
    X, X_test, y, test_ids, extract_time = _load_or_build_cache(
        variant=variant,
        variants_to_extract=extract_variants,
        batch_size=batch_size,
        rebuild=rebuild_cache,
    )

    os.environ["SMILES_SPLIT_REPEATS"] = str(repeats)
    if threshold_mode:
        os.environ["SMILES_THRESHOLD_MODE"] = threshold_mode
    import probe
    import splitting

    importlib.reload(probe)
    importlib.reload(splitting)

    df = pd.read_csv(DATA_FILE)
    print(f"Variant: {variant}")
    print(f"Feature matrix: {X.shape}; test: {X_test.shape}")
    splits = splitting.split_data(y, df)
    print(f"Splits: {len(splits)}")
    fold_results = run_evaluation(splits, X, y, probe.HallucinationProbe)
    print_summary(fold_results, X.shape[1], len(X), extract_time)
    save_results(fold_results, X.shape[1], len(X), extract_time, OUTPUT_FILE)

    idx_non_test = np.unique(
        np.concatenate(
            [
                np.concatenate([idx_tr, idx_va]) if idx_va is not None else idx_tr
                for idx_tr, idx_va, _ in splits
            ]
        )
    )
    final_probe = probe.HallucinationProbe()
    final_probe.fit(X[idx_non_test], y[idx_non_test])
    save_predictions(final_probe, X_test, test_ids, PREDICTIONS_FILE)

    run_dir = ABLATION_DIR / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    split_mode = os.getenv("SMILES_SPLIT_MODE", "stratified").strip().lower()
    threshold_mode = os.getenv("SMILES_THRESHOLD_MODE", "auto").strip().lower()
    split_suffix = "" if split_mode == "stratified" else f".{split_mode}"
    threshold_suffix = "" if threshold_mode == "auto" else f".threshold-{threshold_mode}"
    results_copy = run_dir / f"results{split_suffix}{threshold_suffix}.repeats-{repeats}.json"
    predictions_copy = run_dir / f"predictions{split_suffix}{threshold_suffix}.repeats-{repeats}.csv"
    metadata_copy = run_dir / f"metadata{split_suffix}{threshold_suffix}.repeats-{repeats}.json"
    shutil.copyfile(OUTPUT_FILE, results_copy)
    shutil.copyfile(PREDICTIONS_FILE, predictions_copy)
    metadata = {
        "variant": variant,
        "repeats": repeats,
        "feature_dim": int(X.shape[1]),
        "n_train": int(X.shape[0]),
        "n_test": int(X_test.shape[0]),
        "split_mode": split_mode,
        "threshold_mode": threshold_mode,
        "cache_file": str(_cache_path(variant)),
        "results_file": str(results_copy),
        "predictions_file": str(predictions_copy),
    }
    metadata_copy.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Copied ablation artifacts to {run_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=os.getenv("SMILES_EXPERIMENT_VARIANT", "final"))
    parser.add_argument("--repeats", type=int, default=int(os.getenv("SMILES_SPLIT_REPEATS", "1")))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--threshold-mode",
        default=os.getenv("SMILES_THRESHOLD_MODE", "auto"),
        choices=("auto", "f1", "accuracy", "balanced_accuracy", "youden", "train_prior"),
        help="Decision-threshold calibration mode. 'auto' preserves probe defaults.",
    )
    parser.add_argument(
        "--extract-variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants to cache in one Qwen pass.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    variants = tuple(v.strip() for v in args.extract_variants.split(",") if v.strip())
    _run_cached_solution(
        variant=args.variant.strip().lower(),
        repeats=args.repeats,
        batch_size=args.batch_size,
        rebuild_cache=args.rebuild_cache,
        extract_variants=variants,
        threshold_mode=args.threshold_mode.strip().lower(),
    )
