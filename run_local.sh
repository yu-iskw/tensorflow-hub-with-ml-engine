#!/usr/bin/env bash

TRAIN_DATA=$1
EVAL_DATA=$2

MODEL_DIR="./model"

# Remove a directory to store saved model.
rm -fr "${MODEL_DIR}"

# Train a model.
gcloud ml-engine local train \
  --package-path trainer \
  --module-name trainer.task \
  -- \
    --model_dir "${MODEL_DIR}" \
    --train_data "${TRAIN_DATA}" \
    --eval_data "${EVAL_DATA}" \
    --batch_size 32 \
    --max_steps 100 \
    --eval_steps 10
