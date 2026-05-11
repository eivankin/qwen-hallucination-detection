"""Build report tables and figures for SOLUTION.md.

The report-facing artifacts use only the labeled training data and saved local
CV/ablation outputs.  Diagnostic proxy/reviewed test labels are intentionally
excluded from this script.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = Path("docs/report_artifacts")
ABLATION_DIR = Path("artifacts/ablations")
ENSEMBLE_DIR = Path("artifacts/ensembles")
FINAL_VARIANT = "tail_with_var"
FINAL_CACHE = Path(f"artifacts/feature_cache/features__{FINAL_VARIANT}__maxlen-512.npz")


def _parse_prompt(prompt: str) -> tuple[str, str]:
    context_match = re.search(
        r"complete sentence\.\n\n(.*?)\n\nNote that your answer",
        prompt,
        flags=re.DOTALL,
    )
    question_match = re.search(
        r"Here is the question:\s*(.*?)\n\nYour answer:",
        prompt,
        flags=re.DOTALL,
    )
    context = context_match.group(1).strip() if context_match else ""
    question = question_match.group(1).strip() if question_match else ""
    return context, question


def _short(text: str, n: int = 260) -> str:
    text = " ".join(str(text).replace("<|endoftext|>", "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "..."


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _metrics_row(name: str, result_path: Path, pred_path: Path | None = None) -> dict:
    result = _load_json(result_path)
    pred_pos = None
    if pred_path is not None and pred_path.exists():
        pred_pos = int(pd.read_csv(pred_path)["label"].sum())
    return {
        "method": name,
        "feature_dim": result.get("feature_dim", ""),
        "predicted_hallucinated_test": pred_pos,
        "cv_accuracy": result.get("avg_test_accuracy"),
        "cv_f1": result.get("avg_test_f1"),
        "cv_auroc": result.get("avg_test_auroc"),
        "train_auroc": result.get("avg_train_auroc"),
        "val_auroc": result.get("avg_val_auroc"),
        "folds": result.get("n_folds", ""),
    }


def _load_variant_metrics() -> pd.DataFrame:
    rows = []
    for result_path in sorted(ABLATION_DIR.glob("*/results.repeats-1.json")):
        variant = result_path.parent.name
        if variant == "final_original_solution":
            continue
        rows.append(
            _metrics_row(
                variant,
                result_path,
                result_path.parent / "predictions.repeats-1.csv",
            )
        )

    final_calibrated = ABLATION_DIR / FINAL_VARIANT / "results.threshold-train_prior.repeats-1.json"
    if final_calibrated.exists():
        rows.append(
            _metrics_row(
                f"{FINAL_VARIANT}+train_prior",
                final_calibrated,
                ABLATION_DIR / FINAL_VARIANT / "predictions.threshold-train_prior.repeats-1.csv",
            )
        )

    repeated = [
        ("final 3x5-fold", ABLATION_DIR / "final" / "results.repeats-3.json"),
        (
            f"{FINAL_VARIANT}+train_prior 3x5-fold",
            ABLATION_DIR / FINAL_VARIANT / "results.threshold-train_prior.repeats-3.json",
        ),
    ]
    for name, path in repeated:
        if path.exists():
            rows.append(_metrics_row(name, path))

    for result_path in sorted(ENSEMBLE_DIR.glob("*/results*.json")):
        rows.append(_metrics_row(f"ensemble:{result_path.parent.name}", result_path))

    df = pd.DataFrame(rows)
    return df.sort_values(["cv_accuracy", "cv_auroc"], ascending=False)


def _load_public_cv_comparison() -> pd.DataFrame:
    repo_root = Path("/tmp/smiles-all")
    targets = {
        "mariklolik": "mariklolik__SMILES-2026-Hallucination-Detection",
        "josephofthebread": "josephofthebread__SMILES-2026-Hallucination-Detection",
        "DeadMorose777": "DeadMorose777__SMILES-2026-Hallucination-Detection",
        "HammonDDDDD": "HammonDDDDD__SMILES-2026-Hallucination-Detection",
        "Eva-Shelmanova": "Eva-Shelmanova__hallucination-probe-qwen-0.5b",
        "Humpty1944": "Humpty1944__SMILES-2026-Hallucination-Detection",
    }
    rows = []
    for label, dirname in targets.items():
        result = _load_json(repo_root / dirname / "results.json")
        if result:
            rows.append(
                {
                    "method": label,
                    "reported_cv_accuracy": result.get("avg_test_accuracy"),
                    "reported_cv_f1": result.get("avg_test_f1"),
                    "reported_cv_auroc": result.get("avg_test_auroc"),
                    "reported_train_auroc": result.get("avg_train_auroc"),
                    "folds": result.get("n_folds", len(result.get("folds", []))),
                }
            )
    rows.append(
        {
            "method": "ours: tail_with_var+train_prior",
            "reported_cv_accuracy": 0.7267,
            "reported_cv_f1": 0.8043,
            "reported_cv_auroc": 0.7471,
            "reported_train_auroc": 0.8880,
            "folds": 15,
        }
    )
    return pd.DataFrame(rows).sort_values("reported_cv_accuracy", ascending=False)


def _make_task_examples(df: pd.DataFrame) -> pd.DataFrame:
    curated = [
        {
            "id": 398,
            "label": "supported",
            "why": "The context explicitly says Gegeen Khan was Ayurbarwada's son and successor.",
        },
        {
            "id": 35,
            "label": "hallucinated_wrong_answer",
            "why": "The context gives the dates 1321 to 1323, while the answer is an unrelated number.",
        },
        {
            "id": 587,
            "label": "hallucinated_generation_artifact",
            "why": "The response repeats assistant/system-like text instead of answering the question.",
        },
    ]
    rows = []
    for item in curated:
        row = df.loc[item["id"]]
        context, question = _parse_prompt(row["prompt"])
        rows.append(
            {
                "id": item["id"],
                "label": item["label"],
                "question": question,
                "evidence_excerpt": _short(context, 320),
                "response": _short(row["response"], 220),
                "note": item["why"],
            }
        )
    return pd.DataFrame(rows)


def _make_ambiguity_examples(df: pd.DataFrame) -> pd.DataFrame:
    curated = [
        {
            "id": 34,
            "label": "truthful",
            "note": "The answer contains prompt-like text, but still gives the supported answer about events in Victoria's economy.",
        },
        {
            "id": 66,
            "label": "truthful",
            "note": "The response leaks prompt/context text, but the core answer is supported: the Dutch fought Spain.",
        },
        {
            "id": 124,
            "label": "truthful",
            "note": "The answer gives the correct time period, followed by unrelated artifact text.",
        },
        {
            "id": 16,
            "label": "hallucinated",
            "note": "The response starts with plausible content but then repeats system text, and the label treats this as hallucination.",
        },
        {
            "id": 27,
            "label": "hallucinated",
            "note": "The response contains a plausible date but contradicts the context and continues into assistant artifacts.",
        },
    ]
    rows = []
    for item in curated:
        row = df.loc[item["id"]]
        context, question = _parse_prompt(row["prompt"])
        rows.append(
            {
                "id": item["id"],
                "label": item["label"],
                "question": question,
                "context_excerpt": _short(context, 240),
                "response": _short(row["response"], 260),
                "note": item["note"],
            }
        )
    return pd.DataFrame(rows)


def _compute_oof_predictions() -> pd.DataFrame:
    if not FINAL_CACHE.exists():
        raise FileNotFoundError(
            f"Missing {FINAL_CACHE}. Run scripts/cached_solution.py --variant {FINAL_VARIANT} first."
        )
    os.environ["SMILES_EXPERIMENT_VARIANT"] = FINAL_VARIANT
    os.environ["SMILES_THRESHOLD_MODE"] = "train_prior"
    os.environ["SMILES_SPLIT_REPEATS"] = "1"
    os.environ.pop("SMILES_SPLIT_MODE", None)
    import probe
    import splitting

    importlib.reload(probe)
    importlib.reload(splitting)

    cache = np.load(FINAL_CACHE, allow_pickle=False)
    X = cache["X"]
    y = cache["y"].astype(int)
    df = pd.read_csv("data/dataset.csv")
    splits = splitting.split_data(y, df)

    prob = np.full(len(y), np.nan, dtype=np.float64)
    pred = np.full(len(y), -1, dtype=np.int32)
    fold = np.full(len(y), -1, dtype=np.int32)

    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(splits, start=1):
        model = probe.HallucinationProbe()
        model.fit(X[idx_train], y[idx_train])
        if idx_val is not None:
            model.fit_hyperparameters(X[idx_val], y[idx_val])
        prob[idx_test] = model.predict_proba(X[idx_test])[:, 1]
        pred[idx_test] = model.predict(X[idx_test])
        fold[idx_test] = fold_idx

    rows = []
    for idx, row in df.iterrows():
        context, question = _parse_prompt(row["prompt"])
        true = int(y[idx])
        predicted = int(pred[idx])
        rows.append(
            {
                "id": idx,
                "fold": int(fold[idx]),
                "true_label": true,
                "pred_label": predicted,
                "prob_hallucinated": float(prob[idx]),
                "is_error": bool(true != predicted),
                "error_type": (
                    "false_positive"
                    if true == 0 and predicted == 1
                    else "false_negative"
                    if true == 1 and predicted == 0
                    else "correct"
                ),
                "question": question,
                "context_excerpt": _short(context, 280),
                "response": _short(row["response"], 260),
            }
        )
    return pd.DataFrame(rows)


def _make_error_examples(oof: pd.DataFrame) -> pd.DataFrame:
    false_positive_ids = [603, 431]
    false_negative_ids = [648, 321]
    chosen = oof[oof["id"].isin(false_positive_ids + false_negative_ids)].copy()
    chosen["note"] = chosen["id"].map(
        {
            603: "The answer is long and reasoning-like, which resembles many malformed hallucinations.",
            431: "The response contains the relevant object but gives the wrong number of valves.",
            648: "The answer is fluent and domain-plausible but not supported by the excerpt.",
            321: "The answer gives a material, but the context says paper money came from mulberry bark.",
        }
    )
    return chosen[
        [
            "id",
            "error_type",
            "prob_hallucinated",
            "question",
            "context_excerpt",
            "response",
            "note",
        ]
    ].sort_values(["error_type", "id"])


def _make_pca_table() -> pd.DataFrame:
    cache = np.load(FINAL_CACHE, allow_pickle=False)
    X = cache["X"]
    y = cache["y"].astype(int)
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    return pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "label": y})


def _make_layer_summary() -> pd.DataFrame:
    cache = np.load(FINAL_CACHE, allow_pickle=False)
    X = cache["X"]
    y = cache["y"].astype(int)
    layers = (12, 16, 20, 24)
    pools = ("mean", "last", "second_last", "lastK16", "lastK32", "lastK64", "stdK16", "stdK32", "stdK64")
    hidden_dim = 896
    block = len(pools) * hidden_dim
    rows = []
    for label_value, label_name in [(0, "truthful"), (1, "hallucinated")]:
        subset = X[y == label_value]
        for layer_pos, layer in enumerate(layers):
            for pool_pos, pool in enumerate(pools):
                start = layer_pos * block + pool_pos * hidden_dim
                norms = np.linalg.norm(subset[:, start : start + hidden_dim], axis=1)
                rows.append(
                    {
                        "label": label_name,
                        "layer": layer,
                        "pool": pool,
                        "mean_norm": float(norms.mean()),
                        "std_norm": float(norms.std()),
                    }
                )
    return pd.DataFrame(rows)


def _try_make_plots(
    variant_metrics: pd.DataFrame,
    pca_table: pd.DataFrame,
    oof: pd.DataFrame,
    layer_summary: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        (OUT_DIR / "PLOTS_SKIPPED.txt").write_text(
            f"Install matplotlib and seaborn to generate PNG plots. Import error: {exc}\n"
        )
        return

    sns.set_theme(style="whitegrid")

    plot_methods = [
        "baseline_last_token",
        "final_last_token_lr",
        "final",
        "tail_with_var",
        "tail_with_var+train_prior",
        "tail_minmax",
    ]
    metric_df = variant_metrics[variant_metrics["method"].isin(plot_methods)].melt(
        id_vars=["method"],
        value_vars=["cv_accuracy", "cv_f1", "cv_auroc"],
        var_name="metric",
        value_name="score",
    )
    metric_df["metric"] = metric_df["metric"].map(
        {"cv_accuracy": "Accuracy", "cv_f1": "F1", "cv_auroc": "AUROC"}
    )
    plt.figure(figsize=(9.5, 4.8))
    sns.barplot(data=metric_df, x="metric", y="score", hue="method")
    plt.ylim(0.68, 0.86)
    plt.ylabel("CV score")
    plt.xlabel("")
    plt.legend(title="", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "variant_metrics_grouped.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.3, 5.2))
    pca_plot = pca_table.copy()
    pca_plot["label"] = pca_plot["label"].map({0: "truthful", 1: "hallucinated"})
    sns.scatterplot(data=pca_plot, x="pc1", y="pc2", hue="label", alpha=0.7, s=28)
    plt.title("PCA of final hidden-state features")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "final_feature_pca.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.5))
    oof_plot = oof.copy()
    oof_plot["true_label"] = oof_plot["true_label"].map({0: "truthful", 1: "hallucinated"})
    sns.histplot(
        data=oof_plot,
        x="prob_hallucinated",
        hue="true_label",
        bins=24,
        common_norm=False,
        stat="density",
        alpha=0.45,
    )
    plt.title("Out-of-fold hallucination probabilities")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "oof_probability_hist.png", dpi=180)
    plt.close()

    compact = layer_summary[layer_summary["pool"].isin(["mean", "lastK32", "stdK32"])]
    plt.figure(figsize=(8.2, 4.8))
    sns.lineplot(
        data=compact,
        x="layer",
        y="mean_norm",
        hue="label",
        style="pool",
        marker="o",
    )
    plt.title("Mean feature norms by layer and selected pool")
    plt.ylabel("Mean L2 norm")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "layer_pool_norm_summary.png", dpi=180)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("data/dataset.csv")

    variant_metrics = _load_variant_metrics()
    variant_metrics.to_csv(OUT_DIR / "variant_metrics.csv", index=False)
    _write_markdown_table(variant_metrics.round(4), OUT_DIR / "variant_metrics.md")

    public_cv = _load_public_cv_comparison()
    public_cv.to_csv(OUT_DIR / "public_cv_comparison.csv", index=False)
    _write_markdown_table(public_cv.round(4), OUT_DIR / "public_cv_comparison.md")

    task_examples = _make_task_examples(df)
    task_examples.to_csv(OUT_DIR / "task_examples.csv", index=False)
    _write_markdown_table(task_examples, OUT_DIR / "task_examples.md")

    ambiguity_examples = _make_ambiguity_examples(df)
    ambiguity_examples.to_csv(OUT_DIR / "ambiguous_continuation_examples.csv", index=False)
    _write_markdown_table(
        ambiguity_examples,
        OUT_DIR / "ambiguous_continuation_examples.md",
    )

    oof = _compute_oof_predictions()
    oof.to_csv(OUT_DIR / "final_oof_predictions.csv", index=False)
    oof_summary = pd.DataFrame(
        [
            {
                "accuracy": accuracy_score(oof["true_label"], oof["pred_label"]),
                "f1": f1_score(oof["true_label"], oof["pred_label"]),
                "auroc": roc_auc_score(oof["true_label"], oof["prob_hallucinated"]),
                "n_errors": int(oof["is_error"].sum()),
                "false_positives": int((oof["error_type"] == "false_positive").sum()),
                "false_negatives": int((oof["error_type"] == "false_negative").sum()),
            }
        ]
    )
    oof_summary.to_csv(OUT_DIR / "final_oof_summary.csv", index=False)
    _write_markdown_table(oof_summary.round(4), OUT_DIR / "final_oof_summary.md")

    errors = _make_error_examples(oof)
    errors.to_csv(OUT_DIR / "final_error_examples.csv", index=False)
    _write_markdown_table(errors.round(4), OUT_DIR / "final_error_examples.md")

    pca_table = _make_pca_table()
    pca_table.to_csv(OUT_DIR / "final_feature_pca.csv", index=False)

    layer_summary = _make_layer_summary()
    layer_summary.to_csv(OUT_DIR / "layer_pool_norm_summary.csv", index=False)

    _try_make_plots(variant_metrics, pca_table, oof, layer_summary)
    print(f"Report artifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
