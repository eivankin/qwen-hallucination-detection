#!/usr/bin/env bash
set -euo pipefail

# Run one stable report ablation without editing solution.py.
#
# Usage:
#   scripts/run_ablation_variant.sh final
#   scripts/run_ablation_variant.sh tail_no_second 3
#
# The first argument selects SMILES_EXPERIMENT_VARIANT.
# The optional second argument selects SMILES_SPLIT_REPEATS.

variant="${1:-final}"
repeats="${2:-1}"

export SMILES_EXPERIMENT_VARIANT="$variant"
export SMILES_SPLIT_REPEATS="$repeats"

python solution.py

mkdir -p "artifacts/ablations/${variant}"
cp results.json "artifacts/ablations/${variant}/results.repeats-${repeats}.json"
cp predictions.csv "artifacts/ablations/${variant}/predictions.repeats-${repeats}.csv"
