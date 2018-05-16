import argparse
import multiprocessing

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3
from tensorflow.python.estimator.warm_starting_util import WarmStartSettings

tf.app.flags.DEFINE_string('model_dir', None, 'model direcotry')
tf.app.flags.DEFINE_string('train_data', None, 'path to training data')
tf.app.flags.DEFINE_string('eval_data', None, 'path to evaluating data')
tf.app.flags.DEFINE_integer('max_steps', 10000, 'max training steps')
tf.app.flags.DEFINE_integer('eval_steps', 50, 'evaluating steps')
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = FLAGS.batch_size
MAX_STEPS = FLAGS.max_steps
EVAL_STEPS = FLAGS.eval_steps

NUM_CLASSES = 5
IMAGE_SIZE = 299
IMAGE_SIZE_WITH_MARGIN = 299


def model_fn(features, labels, mode):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    is_eval = (mode == tf.estimator.ModeKeys.EVAL)

    # Input tensor.
    input = features['image']

    # Inceotion V3
    module_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"
    module = hub.Module(module_url, trainable=False)
    prelogits = module(input)

    # Put additional layers for transfer learning
    dense_256 = tf.layers.dense(prelogits, 256, activation=tf.nn.relu,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=1e-5))
    dense_128 = tf.layers.dense(dense_256, 128, activation=tf.nn.relu,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=1e-5))
    dense_64 = tf.layers.dense(dense_128, 64, activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=1e-5))
    dense_32 = tf.layers.dense(dense_64, 32, activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=1e-5))
    logits = tf.layers.dense(dense_32, NUM_CLASSES, activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=1e-5))

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # Loss function
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Optimizers
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=1e-7, decay=0.9, momentum=0.9, epsilon=1.0, centered=False)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            },
            train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    accuracy = tf.metrics.accuracy(labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"])
    precision = tf.metrics.precision(labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"])
    recall = tf.metrics.recall(labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"])
    eval_metric_ops = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        },
        eval_metric_ops=eval_metric_ops)


def parse_train(record):
    tf.logging.info(record)
    keys_to_features = {
        "image": tf.FixedLenFeature((), tf.string, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64, default_value=-1),
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    tf.logging.info(parsed)
    image = tf.image.decode_jpeg(parsed["image"], channels=3)
    image = tf.image.resize_images(image, [IMAGE_SIZE_WITH_MARGIN, IMAGE_SIZE_WITH_MARGIN])
    return {"image": image}, tf.one_hot(parsed["label"], depth=NUM_CLASSES)


def train_input_fn():
    data_path = FLAGS.train_data
    files = tf.data.Dataset.list_files(data_path)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=2)
    dataset = dataset.map(parse_train, num_parallel_calls=multiprocessing.cpu_count())
    dataset = dataset.shuffle(512).repeat(10).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def parse_eval(record):
    keys_to_features = {
        "image": tf.FixedLenFeature((), tf.string, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64, default_value=-1),
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.image.decode_jpeg(parsed["image"], channels=3)
    image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])
    return {"image": image}, tf.one_hot(parsed["label"], depth=NUM_CLASSES)


def eval_input_fn():
    data_path = FLAGS.eval_data
    files = tf.data.Dataset.list_files(data_path)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=2)
    dataset = dataset.map(parse_eval, num_parallel_calls=multiprocessing.cpu_count())
    dataset = dataset.shuffle(128).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def serving_input_receiver_fn():
    receiver_tensors = {
        "image_bytes": tf.placeholder(dtype=tf.string, shape=[None], name='image_bytes'),
    }

    def decode_and_resize(image_str_tensor):
        img = tf.image.decode_jpeg(image_str_tensor, channels=3)
        img = tf.image.resize_image_with_crop_or_pad(img, 299, 299)
        img.set_shape([299, 299, 3])
        img = tf.cast(img, tf.float32)
        return img

    img = tf.map_fn(decode_and_resize, receiver_tensors["image_bytes"],
                    back_prop=False, dtype=tf.float32)
    features = {"image": img}
    serving_input_receiver = tf.estimator.export.ServingInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors
    )
    return serving_input_receiver


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_steps=1000,
        save_summary_steps=1000,
        session_config=tf.ConfigProto(log_device_placement=False),
        model_dir=FLAGS.model_dir)
    clf = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        config=estimator_config)
    latest_exporter = tf.estimator.LatestExporter(
        name="models",
        serving_input_receiver_fn=serving_input_receiver_fn,
        exports_to_keep=5)

    # Train spec
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=MAX_STEPS)

    # Evaluation spec
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        throttle_secs=180,
        steps=EVAL_STEPS,
        exporters=latest_exporter)
    tf.estimator.train_and_evaluate(clf, train_spec=train_spec, eval_spec=eval_spec)


if __name__ == '__main__':
    tf.app.run()
