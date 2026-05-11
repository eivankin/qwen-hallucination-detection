"""
probe.py — regularized hidden-state probes for SMILES.

``HallucinationProbe`` keeps the starter-code public interface, but the default
implementation is a small probability-averaged ensemble of linear models.  This
matches the main lesson from the public boundary-clean baselines and recent
activation-probing work: with 689 labels, aggregation and regularization matter
more than probe depth.

Stable ablation variants are selected with ``SMILES_EXPERIMENT_VARIANT``:

    final                  PCA + LR/Ridge/LinearSVC ensemble
    baseline_last_token    starter MLP on final-layer last token
    final_last_token_lr    L2 logistic regression on final-layer last token
    tail_no_second         same probe as final, fewer aggregation pools
    tail_with_geometry     same probe as final, geometry scalar ablation
    tail_with_var          default; final plus response-tail variance/std ablation
    tail_minmax            final plus response-tail min/max ablation
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from aggregation import pool_indices
except Exception:  # pragma: no cover - keeps import robust during static checks.
    pool_indices = None

warnings.filterwarnings("ignore", category=ConvergenceWarning)


EXPERIMENT_VARIANT = os.getenv("SMILES_EXPERIMENT_VARIANT", "tail_with_var").strip().lower()
THRESHOLD_MODE = os.getenv("SMILES_THRESHOLD_MODE", "train_prior").strip().lower()
RANDOM_STATE = 42


def _as_float_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got shape {X.shape}.")
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return X


def _safe_pca_components(n_samples: int, n_features: int, requested: int) -> int | None:
    max_allowed = max(1, min(n_samples - 1, n_features, requested))
    if max_allowed < 2:
        return None
    return max_allowed


def _threshold_by_metric(probs: np.ndarray, y: np.ndarray, metric: str) -> float:
    if metric == "auto":
        metric = "f1"
    if metric == "train_prior":
        target_pos = float(np.mean(y))
        if target_pos <= 0.0:
            return float(np.nextafter(probs.max(), np.inf))
        if target_pos >= 1.0:
            return float(np.nextafter(probs.min(), -np.inf))
        return float(np.quantile(probs, 1.0 - target_pos))

    candidates = np.unique(np.concatenate([probs, np.linspace(0.05, 0.95, 181)]))
    best_t = 0.5
    best_score = -1.0
    for threshold in candidates:
        pred = (probs >= threshold).astype(int)
        if metric == "f1":
            score = f1_score(y, pred, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(y, pred)
        elif metric in {"balanced_accuracy", "youden"}:
            score = balanced_accuracy_score(y, pred)
        else:
            raise ValueError(metric)
        if score > best_score or (
            score == best_score and abs(float(threshold) - 0.5) < abs(best_t - 0.5)
        ):
            best_t = float(threshold)
            best_score = float(score)
    return best_t


@dataclass(frozen=True)
class _MemberSpec:
    name: str
    pools: tuple[str, ...] | None
    pca_components: int | None
    classifier: str
    params: dict


def _columns_for_pools(pools: tuple[str, ...] | None, n_features: int) -> np.ndarray:
    if pools is None:
        return np.arange(n_features, dtype=np.int64)
    if pool_indices is None:
        return np.arange(n_features, dtype=np.int64)
    cols: list[int] = []
    for pool in pools:
        try:
            cols.extend(pool_indices(pool))
        except KeyError:
            return np.arange(n_features, dtype=np.int64)
    cols_arr = np.asarray(cols, dtype=np.int64)
    return cols_arr[cols_arr < n_features]


def _make_classifier(kind: str, params: dict, y: np.ndarray):
    if kind == "logreg":
        return LogisticRegression(**params)
    if kind == "ridge":
        return RidgeClassifier(**params)
    if kind == "svc":
        class_counts = np.bincount(y.astype(int), minlength=2)
        cv = max(2, min(3, int(class_counts.min())))
        return CalibratedClassifierCV(
            estimator=LinearSVC(**params),
            method="sigmoid",
            cv=cv,
        )
    raise ValueError(f"Unknown classifier kind: {kind}")


class _LinearMember:
    def __init__(self, spec: _MemberSpec) -> None:
        self.spec = spec
        self.columns: np.ndarray | None = None
        self.pipeline: Pipeline | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LinearMember":
        self.columns = _columns_for_pools(self.spec.pools, X.shape[1])
        X_view = X[:, self.columns]

        steps: list[tuple[str, object]] = [("scale", StandardScaler())]
        if self.spec.pca_components is not None:
            n_components = _safe_pca_components(
                n_samples=X_view.shape[0],
                n_features=X_view.shape[1],
                requested=self.spec.pca_components,
            )
            if n_components is not None:
                steps.append(
                    (
                        "pca",
                        PCA(
                            n_components=n_components,
                            whiten=False,
                            svd_solver="randomized",
                            random_state=RANDOM_STATE,
                        ),
                    )
                )
        steps.append(("clf", _make_classifier(self.spec.classifier, self.spec.params, y)))
        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X_view, y)
        return self

    def proba_pos(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None or self.columns is None:
            raise RuntimeError("Member is not fitted.")
        X_view = X[:, self.columns]
        clf = self.pipeline
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X_view)[:, 1]
        scores = clf.decision_function(X_view)
        return 1.0 / (1.0 + np.exp(-scores))


class _SklearnProbe:
    def __init__(self) -> None:
        self.members: list[_LinearMember] = []
        self.threshold = 0.5
        self._fit_X: np.ndarray | None = None
        self._fit_y: np.ndarray | None = None
        self._threshold_tuned = False

    @staticmethod
    def _member_specs() -> tuple[_MemberSpec, ...]:
        common_lr = dict(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
        specs = [
            _MemberSpec(
                name="tail32_lr",
                pools=("mean", "last", "lastK32"),
                pca_components=64,
                classifier="logreg",
                params=dict(C=1.0, **common_lr),
            ),
            _MemberSpec(
                name="tail64_lr",
                pools=("second_last", "lastK32", "lastK64"),
                pca_components=64,
                classifier="logreg",
                params=dict(C=0.5, **common_lr),
            ),
            _MemberSpec(
                name="ridge_tail",
                pools=("mean", "last", "lastK32"),
                pca_components=96,
                classifier="ridge",
                params=dict(alpha=100.0, class_weight="balanced"),
            ),
            _MemberSpec(
                name="svc_all",
                pools=None,
                pca_components=128,
                classifier="svc",
                params=dict(
                    C=0.3,
                    dual=False,
                    max_iter=50000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
        if EXPERIMENT_VARIANT == "tail_with_var":
            specs.extend(
                [
                    _MemberSpec(
                        name="tail_std_lr",
                        pools=("stdK16", "stdK32", "stdK64"),
                        pca_components=64,
                        classifier="logreg",
                        params=dict(C=0.5, **common_lr),
                    ),
                    _MemberSpec(
                        name="mean_var_svc",
                        pools=("mean", "lastK32", "stdK32"),
                        pca_components=96,
                        classifier="svc",
                        params=dict(
                            C=0.2,
                            dual=False,
                            max_iter=50000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
        if EXPERIMENT_VARIANT == "tail_minmax":
            specs.extend(
                [
                    _MemberSpec(
                        name="tail_minmax_lr",
                        pools=("lastK32", "minK32", "maxK32"),
                        pca_components=96,
                        classifier="logreg",
                        params=dict(C=0.3, **common_lr),
                    ),
                    _MemberSpec(
                        name="tail_range_ridge",
                        pools=("minK32", "maxK32"),
                        pca_components=64,
                        classifier="ridge",
                        params=dict(alpha=150.0, class_weight="balanced"),
                    ),
                ]
            )
        return tuple(specs)

    @staticmethod
    def _last_token_lr_spec() -> tuple[_MemberSpec, ...]:
        return (
            _MemberSpec(
                name="last_token_lr",
                pools=None,
                pca_components=None,
                classifier="logreg",
                params=dict(
                    C=0.05,
                    max_iter=5000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SklearnProbe":
        X = _as_float_2d(X)
        y = np.asarray(y, dtype=np.int32)
        self._fit_X = X
        self._fit_y = y
        self._threshold_tuned = False
        specs = self._last_token_lr_spec() if EXPERIMENT_VARIANT == "final_last_token_lr" else self._member_specs()
        self.members = [_LinearMember(spec).fit(X, y) for spec in specs]
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "_SklearnProbe":
        probs = self.predict_proba(X_val)[:, 1]
        metric = "f1" if THRESHOLD_MODE == "auto" else THRESHOLD_MODE
        self.threshold = _threshold_by_metric(probs, np.asarray(y_val, dtype=np.int32), metric)
        self._threshold_tuned = True
        return self

    def _ensure_threshold(self) -> None:
        if self._threshold_tuned or self._fit_X is None or self._fit_y is None:
            return
        y = self._fit_y
        class_counts = np.bincount(y.astype(int), minlength=2)
        n_splits = min(5, int(class_counts.min()))
        if n_splits < 2:
            self.threshold = 0.5
            self._threshold_tuned = True
            return

        oof = np.zeros(len(y), dtype=np.float64)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for idx_fit, idx_holdout in splitter.split(self._fit_X, y):
            fold_probe = _SklearnProbe().fit(self._fit_X[idx_fit], y[idx_fit])
            oof[idx_holdout] = fold_probe.predict_proba(self._fit_X[idx_holdout])[:, 1]
        metric = "accuracy" if THRESHOLD_MODE == "auto" else THRESHOLD_MODE
        self.threshold = _threshold_by_metric(oof, y, metric)
        self._threshold_tuned = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._ensure_threshold()
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _as_float_2d(X)
        if not self.members:
            raise RuntimeError("Probe is not fitted.")
        probs = np.stack([member.proba_pos(X) for member in self.members], axis=0).mean(axis=0)
        return np.stack([1.0 - probs, probs], axis=1)


class _StarterMlpProbe(nn.Module):
    """Starter-style MLP retained for the overfitting ablation."""

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None
        self._scaler = StandardScaler()
        self._threshold = 0.5

    def _build_network(self, input_dim: int) -> None:
        self._net = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Call fit() before forward().")
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_StarterMlpProbe":
        torch.manual_seed(RANDOM_STATE)
        X_scaled = self._scaler.fit_transform(_as_float_2d(X))
        y_arr = np.asarray(y, dtype=np.float32)
        self._build_network(X_scaled.shape[1])
        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y_arr)
        n_pos = int(y_arr.sum())
        n_neg = len(y_arr) - n_pos
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        self.train()
        for _ in range(200):
            optimizer.zero_grad()
            loss = criterion(self(X_t), y_t)
            loss.backward()
            optimizer.step()
        self.eval()
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "_StarterMlpProbe":
        probs = self.predict_proba(X_val)[:, 1]
        metric = "f1" if THRESHOLD_MODE == "auto" else THRESHOLD_MODE
        self._threshold = _threshold_by_metric(probs, np.asarray(y_val, dtype=np.int32), metric)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(_as_float_2d(X))
        X_t = torch.from_numpy(X_scaled).float()
        with torch.no_grad():
            probs = torch.sigmoid(self(X_t)).numpy()
        return np.stack([1.0 - probs, probs], axis=1)


class HallucinationProbe(nn.Module):
    """Public probe wrapper used by ``solution.py``."""

    def __init__(self) -> None:
        super().__init__()
        if EXPERIMENT_VARIANT == "baseline_last_token":
            self._impl: _StarterMlpProbe | _SklearnProbe = _StarterMlpProbe()
        else:
            self._impl = _SklearnProbe()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self._impl, _StarterMlpProbe):
            return self._impl.forward(x)
        raise RuntimeError("The sklearn probe does not implement torch forward().")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        self._impl.fit(X, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        self._impl.fit_hyperparameters(X_val, y_val)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._impl.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._impl.predict_proba(X)
