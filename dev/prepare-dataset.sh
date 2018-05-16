#!/bin/bash -c

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
SCRIPT_NAME=`basename "$0"`
PROJECT_HOME="${SCRIPT_DIR}/../"

cd "${PROJECT_HOME}/data/"
curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
