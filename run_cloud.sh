#!/usr/bin/env bash

PROJECT_ID=$1
GCS_BUCKET=$2
MODEL_DIR=$3
TRAIN_DATA=$4
EVAL_DATA=$5

JOB_NAME="retrain_flower_$(date '+%Y%m%d%H%M%S')"
gcloud beta ml-engine jobs submit training ${JOB_NAME} \
  --project ${PROJECT_ID} \
  --package-path trainer \
  --module-name trainer.task \
  --staging-bucket gs://${GCS_BUCKET}/ \
  --region us-central1 \
  --config config.yml \
  --runtime-version 1.7 \
  -- \
    --model_dir "${MODEL_DIR}" \
    --train_data "${TRAIN_DATA}" \
    --eval_data "${EVAL_DATA}" \
    --max_steps 10000 \
    --eval_steps 50 \
    --batch_size 32
