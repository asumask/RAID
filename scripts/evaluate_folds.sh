#!/usr/bin/env bash
## evaluate the CV fold models trained by train_folds.sh:
##   each fold on its own val years -> <output_dir>/<fold>/eval_val
##   each fold on the test years    -> <output_dir>/<fold>/eval_test
##   the ensemble of both folds     -> <output_dir>/ensemble/eval_test
## extra arguments are passed to every evaluate.py run, e.g. --output_dir (not --run_dir):
## CUDA_VISIBLE_DEVICES=1 bash scripts/evaluate_folds.sh configs/step1b_unet.yaml
## CUDA_VISIBLE_DEVICES=1 bash scripts/evaluate_folds.sh configs/step1b_unet.yaml --output_dir /some/where
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <config.yaml> [evaluate.py args...]" >&2
    exit 1
fi
config=$1
shift
evaluate="$(dirname "$0")/evaluate.py"

for fold in val2024 val2025; do
    for split in val test; do
        echo "=== fold $fold, split $split ==="
        python3 "$evaluate" --config "$config" --fold "$fold" --split "$split" "$@"
    done
done

echo "=== ensemble, split test ==="
python3 "$evaluate" --config "$config" --fold ensemble "$@"
