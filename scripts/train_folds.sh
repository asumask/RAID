#!/usr/bin/env bash
## train one model per CV fold (val2024, val2025), then the final model on all non-test years (full),
## one after the other.
## runs go to <output_dir>/<var>/<fold>, <var> = the config's target variable(s), e.g. temp
## extra arguments are passed to every train.py run, e.g. --output_dir or --epochs:
## CUDA_VISIBLE_DEVICES=1 bash scripts/train_folds.sh configs/step1b_unet.yaml
## CUDA_VISIBLE_DEVICES=1 bash scripts/train_folds.sh configs/step1b_unet.yaml --output_dir /some/where
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <config.yaml> [train.py args...]" >&2
    exit 1
fi
config=$1
shift

for fold in val2024 val2025 full; do
    echo "=== fold $fold ==="
    python3 "$(dirname "$0")/train.py" --config "$config" --fold "$fold" "$@"
done
