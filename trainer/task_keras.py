import argparse
import multiprocessing

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3
from tensorflow.python.keras._impl.keras.applications.inception_v3 import preprocess_input

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


def get_keras_model():
    keras = tf.keras

    base_model = keras.applications.InceptionV3(weights='imagenet',
                                                include_top=False,
                                                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    model = keras.Sequential()
    model.add(base_model)
    model.add(keras.layers.MaxPool2D((8, 8)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    base_model.trainable = True
    return model


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
    image = preprocess_input(image)
    return dict(zip(['inception_v3_input'], [image])), tf.one_hot(parsed['label'], NUM_CLASSES)


def train_input_fn():
    data_path = FLAGS.train_data
    files = tf.data.Dataset.list_files(data_path)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=2)
    dataset = dataset.map(parse_train, num_parallel_calls=multiprocessing.cpu_count())
    dataset = dataset.shuffle(512).repeat(1000).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def parse_eval(record):
    keys_to_features = {
        "image": tf.FixedLenFeature((), tf.string, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64, default_value=-1),
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.image.decode_jpeg(parsed["image"], channels=3)
    image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = preprocess_input(image)
    return dict(zip(['inception_v3_input'], [image])), tf.one_hot(parsed['label'], NUM_CLASSES)


def eval_input_fn():
    data_path = FLAGS.eval_data
    files = tf.data.Dataset.list_files(data_path)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=2)
    dataset = dataset.map(parse_eval, num_parallel_calls=multiprocessing.cpu_count())
    dataset = dataset.shuffle(128).repeat(100).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def serving_input_receiver_fn():
    receiver_tensors = {
        "image_bytes": tf.placeholder(dtype=tf.string, shape=[None], name='image_bytes'),
    }

    def decode_and_resize(image_str_tensor):
        img = tf.image.decode_jpeg(image_str_tensor, channels=3)
        img = tf.image.resize_image_with_crop_or_pad(img, IMAGE_SIZE, IMAGE_SIZE)
        img.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        img = tf.cast(img, tf.float32)
        return img

    img = tf.map_fn(decode_and_resize, receiver_tensors["image_bytes"],
                    back_prop=False, dtype=tf.float32)
    features = {"inception_v3_input": img}
    serving_input_receiver = tf.estimator.export.ServingInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors
    )
    return serving_input_receiver


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    # next_batch = train_input_fn()
    # with tf.Session() as sess:
    #     result = sess.run(next_batch)
    #     print(result)

    # Get kears model.
    model = get_keras_model()
    print(model.summary())

    # Compile keras model.
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(lr=1e-5),
                  metrics=['acc'])

    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_steps=1000,
        save_summary_steps=1000,
        session_config=tf.ConfigProto(log_device_placement=False),
        model_dir=FLAGS.model_dir)
    clf = tf.keras.estimator.model_to_estimator(
        keras_model=model,
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
