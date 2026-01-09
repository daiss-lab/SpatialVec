#!/usr/bin/env bash
set -e

PYTHONPATH=src python -m cnf2vec.main \
  --gpkg_path "data/NYC_total_data.gpkg" \
  --layer "NYC_total_data" \
  --out_root "outputs/nyc" \
  --n_per_type 2000 \
  --grid_n 64 \
  --max_refine 2000 \
  --epochs 10 \
  --batch_size 4 \
  --lr 1e-3 \
  --sample_ratio 0.5
