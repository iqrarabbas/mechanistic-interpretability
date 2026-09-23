#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

python scripts/train_sae_level4.py \
  --corruption noise \
  --model base \
  --sae-type vanilla \
  --train-samples 10000 \
  --val-samples 1000 \
  --epochs 15 \
  --learning-rate 1e-3 \
  --l1-coefficient 1e-3 \
  --expansion-factor 32 \
  --seed 0

python scripts/train_sae_level4.py \
  --corruption noise \
  --model fine_tuned \
  --sae-type vanilla \
  --train-samples 10000 \
  --val-samples 1000 \
  --epochs 15 \
  --learning-rate 1e-3 \
  --l1-coefficient 1e-3 \
  --expansion-factor 32 \
  --seed 0

python scripts/compare_sae_level4.py \
  --corruption noise \
  --sae-type vanilla \
  --model-scope both \
  --samples 1000 \
  --start-index 11000 \
  --corruption-seed 0
