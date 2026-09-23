#!/usr/bin/env bash
set -euo pipefail

for lambda in 0.0001 0.0003 0.00086 0.001 0.003 0.01; do
  tag="pilot_lambda_${lambda}"
  PYTHONPATH=. python scripts/train_sae_level4.py \
    --corruption blur \
    --model base \
    --sae-type vanilla \
    --train-samples 1000 \
    --val-samples 200 \
    --epochs 5 \
    --patience 2 \
    --l1-coefficient "$lambda" \
    --output-tag "$tag"
done
