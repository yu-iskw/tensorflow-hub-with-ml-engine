import os
import sys
from random import random, shuffle

import tensorflow as tf

tf.app.flags.DEFINE_string('input', None, 'path to flower data')
tf.app.flags.DEFINE_string('train_output', None, 'path to training data')
tf.app.flags.DEFINE_string('eval_output', None, 'path to evaluating data')
tf.app.flags.DEFINE_float('train_size', 0.8, 'train size ratio')
FLAGS = tf.app.flags.FLAGS

# Map from class to label.
classes = {
    "daisy": 0,
    "dandelion": 1,
    "roses": 2,
    "sunflowers": 3,
    "tulips": 4,
}


def create_tfrecord(file_name, label):
    """
    Convert to TFRecord

    :param file_name: path to an image
    :param label: label
    :return: serialized TFRecord
    """
    with tf.gfile.GFile(file_name, 'rb') as f:
        # Read an image.
        image_data = f.read()
        # Create a feature.
        feature = {
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
        }
        # Create an example protocol buffer.
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        return example.SerializeToString()


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    # Create a list of file names and labels.
    dataset = []
    for k, label in classes.items():
        path = os.path.join(FLAGS.input, k, "*.jpg")
        file_names = tf.gfile.Glob(path)
        pairs = [(file_name, label) for file_name in file_names]
        dataset.extend(pairs)

    # Shuffle the elements.
    shuffle(dataset)

    # Open the TFRecords file.
    train_writer = tf.python_io.TFRecordWriter(FLAGS.train_output)
    eval_writer = tf.python_io.TFRecordWriter(FLAGS.eval_output)

    # Write TFRecords.
    for x in dataset:
        image_path = x[0]
        label = x[1]
        # Convert to TFRecord.
        tfrecord = create_tfrecord(image_path, label)
        # Split train ane eval.
        if random() <= FLAGS.train_size:
            train_writer.write(tfrecord)
        else:
            eval_writer.write(tfrecord)

    # Close TFRecord writers.
    train_writer.close()
    eval_writer.close()
    sys.stdout.flush()


if __name__ == '__main__':
    tf.app.run()
