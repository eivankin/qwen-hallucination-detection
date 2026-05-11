# 🔍 Hallucination Detection

Detect whether a small language
model's answer is *hallucinated* (fabricated) or *truthful* using the model's
own internal representations (hidden states).

## Overview

Large (and small) language models sometimes *hallucinate* — they generate
plausible-sounding text that is factually incorrect.  This competition asks you
to build a **lightweight binary classifier** (called a *probe*) that reads the
model's internal hidden states and predicts whether a given response is
truthful (`label = 0`) or hallucinated (`label = 1`).

The language model used throughout is **[Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)** — a
decoder-only causal transformer with 24 layers and a hidden dimension of 896.
It fits comfortably on a free Google Colab T4 GPU.

**Primary ranking metric:** Accuracy on the held-out `test.csv`.

## Repository Structure

```
HALLUCINATION-DETECTION/
├── data/
│   ├── dataset.csv        # Labelled training data (prompt, response, label)
│   └── test.csv           # Unlabelled competition test set
│
├── solution.py            # Main script - run to create a 
│
│   ── Files you implement ──────────────────────────────────────────────
├── aggregation.py         # Layer selection, token pooling, geometric features
├── probe.py               # HallucinationProbe — the binary classifier
├── splitting.py           # Train / validation / test split strategy
│
│   ── Fixed infrastructure (do not edit) ───────────────────────────────
├── model.py               # Loads Qwen2.5-0.5B and exposes get_model_and_tokenizer()
├── evaluate.py            # Evaluation loop, metrics, summary table, JSON output
│
├── requirements.txt       # Python dependencies
└── LICENSE
```


## Quick Start

### Setup with uv

```bash
uv sync
uv run python solution.py
```

### Setup with pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python solution.py
```

CUDA is recommended. Running on CPU is possible, but hidden-state extraction is
slow.

### Reproducing Experiments

The default command runs the final selected solution:

```bash
python solution.py
```

Stable ablations can be reproduced by setting environment variables:

```bash
SMILES_EXPERIMENT_VARIANT=baseline_last_token python solution.py
SMILES_EXPERIMENT_VARIANT=final_last_token_lr python solution.py
SMILES_EXPERIMENT_VARIANT=final python solution.py
SMILES_EXPERIMENT_VARIANT=tail_no_second python solution.py
SMILES_EXPERIMENT_VARIANT=tail_with_geometry python solution.py
SMILES_EXPERIMENT_VARIANT=tail_minmax python solution.py
SMILES_EXPERIMENT_VARIANT=tail_with_var python solution.py
```

Threshold calibration can also be changed:

```bash
SMILES_THRESHOLD_MODE=auto python solution.py
SMILES_THRESHOLD_MODE=train_prior python solution.py
SMILES_THRESHOLD_MODE=accuracy python solution.py
```

For repeated cross-validation:

```bash
SMILES_SPLIT_REPEATS=3 python solution.py
```

For a context-grouped diagnostic split:

```bash
SMILES_SPLIT_MODE=context_group python solution.py
```

Report illustrations can be regenerated from cached experiment artifacts with:

```bash
python scripts/make_report_artifacts.py
```

The script writes draft plots/tables under `docs/report_artifacts/`. The final
figures used in `SOLUTION.md` are stored separately in `figures/`.

## Dataset

`data/dataset.csv` contains 689 labelled samples with three columns:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Full ChatML-formatted conversation context fed to Qwen |
| `response` | str | The model's generated response |
| `label` | float | `1.0` = hallucinated · `0.0` = truthful |

The `prompt` uses the **ChatML** template built into Qwen models:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Given the context, answer the question …<|im_end|>
<|im_start|>assistant
```


`data/test.csv` is structured identically but the `label` column is null - these are the samples you submit predictions for via a `predictions.csv` generated file.


## What You Implement

You are expected to edit **three files**:  
- `aggregation.py`
- `probe.py`
- `splitting.py`

The rest of the codebase shall remain untouched.

**Feature Engineering & Dimensionality Reduction**: Applicants are encouraged to experiment with adding hand-crafted features during the aggregation step, drawing on geometrical or topological methods to enrich the representation of probe outputs. Additionally, you may apply dimensionality reduction techniques within probe.py to compress or refine the feature space. 

## Evaluation

For each fold `evaluate.py` reports four numbers:

| # | Checkpoint | Metrics |
|---|-----------|---------|
| 1 | Majority-class baseline | Accuracy, F1 |
| 2 | `HallucinationProbe` on **training** split | Accuracy, F1, AUROC |
| 3 | `HallucinationProbe` on **validation** split | Accuracy, F1, AUROC |
| 4 | `HallucinationProbe` on **test** split | Accuracy, F1, AUROC |

**Accuracy on the `test.csv` is the primary competition metric.**

Results are averaged across folds (if using k-fold) and saved to
`results.json`.


[//]: # (# What is expected from the applicant of SMILES-2026 ?)

[//]: # ()
[//]: # (**Q1:** What must the applicant submit in the application form ?<br>)

[//]: # (**A1:** Submit: )

[//]: # (1. A link to your Github repository)

[//]: # (2. A link to your `predictions.csv` publicly available file on some cloud storage)

[//]: # ()
[//]: # (**Q2:** What the applicants must include in the repository ?<br>)

[//]: # (**A2:** Your repository must contain: )

[//]: # (1. `results.json` - produced by the official `solution.py`)

[//]: # (2. Report file in Markdown format `SOLUTION.md`. )

[//]: # ()
[//]: # (**Q3:** Report requirements &#40;`SOLUTION.md`&#41;<br>)

[//]: # (**A3:** Your report must include:<br>)

[//]: # (- Reproducibility instructions: exact commands to run your solution and acquire the same `predictions.csv`, required environment &#40;if any&#41;, any important implementation details needed to reproduce your result.)

[//]: # (- Final solution description: What components you modified ? What your final approach is ? Why you made these choices ? What contributed most to improving the metric ?)

[//]: # (- Experiments and failed attempts: What ideas you tried but did not include in the final solution ? Why they did not work or were discarded ?)

[//]: # ()
[//]: # (**Q4:** Reproducibility<br>)

[//]: # (**A4:** The repository must be self-contained and runnable with the provided `solution.py` file. Your solution must not require changes to the fixed infrastructure files. Running `solution.py` must generate your submitted `predictions.csv`.)
