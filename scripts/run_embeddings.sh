#!/bin/bash
#SBATCH --job-name=spatialvec_embed
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=batch,ai

set -euo pipefail

export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

INPUT="${INPUT:?set INPUT to the source vector file}"
OUTPUT="${OUTPUT:-outputs/embeddings}"
SAMPLE_WORKERS="${SAMPLE_WORKERS:-8}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"

if [ -n "${VENV:-}" ]; then
  source "$VENV/bin/activate"
fi

python generate_embeddings.py \
  --input "$INPUT" \
  --out "$OUTPUT" \
  --sample-workers "$SAMPLE_WORKERS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE"
