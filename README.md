# Retrain flower data with tensorflow-hub

This is an example code to reproduce an issue of `tensorflow-hub` with Google ML Engine.
The reason why I made the repository is to reproduce the issue that `tensorflow-hub` doesn't work on Google ML Engine at the time I am making the repository.

- [How to use TF Hub on a distributed setting? · Issue \#48 · tensorflow/hub](https://github.com/tensorflow/hub/issues/48)
- [Tensorflow\-Hub doesn't work on the runtime version 1\.7\. \[78898344\] \- Visible to Public \- Issue Tracker](https://issuetracker.google.com/issues/78898344)

```
OP_REQUIRES failed at save_restore_tensor.cc:170 : Invalid argument: Unsuccessful TensorSliceReader constructor: Failed to get matching files on /tmp/tfhub_modules/11d9faf945d073033780fd924b2b09ff42155763/variables/variables: Not found: /tmp/tfhub_modules/11d9faf945d073033780fd924b2b09ff42155763/variables; No such file or directory
```

## Requirements
- Anaconda
- Google Cloud SDK

## Prepare for the python environment
```
# Create conda environment
make create-conda

# Remove conda environment
make remove-conda
```

## Hot wo run

1. Download flower data.
The script downloads the flower data in `data/flower_photos`
```
bash ./dev/prepare-dataset.sh
```

2. Create TFRecord data.
```
python crete_tfrecord.py \
  --input ./data/flower_photos \
  --train_output ./train.tfrecord \
  --eval_output ./eval.tfrecord
```

3. Upload TFRecord data to Google Cloud Storage.
```
PROJECT_ID=...
GCS_BUCKET=...

# Create a bucket if necessary.
gsutil mb -p "$PROJECT_ID" "gs://${GCS_BUCKET}"

# Copy TFRecord files.
gsutil cp -p train.tfrecord "gs://${GCS_BUCKET}/train.tfrecord"
gsutil cp -p eval.tfrecord "gs://${GCS_BUCKET}/eval.tfrecord"

# Check the uploaded files.
gsutil ls "gs://${GCS_BUCKET}/train.tfrecord"
gsutil ls "gs://${GCS_BUCKET}/eval.tfrecord"
```

4. Train a model.
```
# Run on your local machine to test.
GCS_BUCKET=...
TRAIN_DATA="gs://${GCS_BUCKET}/train.tfrecord"
EVAL_DATA="gs://${GCS_BUCKET}/eval.tfrecord"
bash run_local.sh "${TRAIN_DATA}" "${EVAL_DATA}"

# Run on ML Engine
PROJECT_ID=...
GCS_BUCKET=...
MODEL_DIR="gs://${GCS_BUCKET}/model/"
TRAIN_DATA="gs://${GCS_BUCKET}/train.tfrecord"
EVAL_DATA="gs://${GCS_BUCKET}/eval.tfrecord"
bash run_cloud.sh \
  "$PROJECT_ID" \
  "$GCS_BUCKET" \
  "$MODEL_DIR" \
  "$TRAIN_DATA" \
  "$EVAL_DATA"

# Monitor the training
tensorboard --logdir="$MODEL_DIR"
```
