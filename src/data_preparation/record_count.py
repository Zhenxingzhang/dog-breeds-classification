import argparse
import tensorflow as tf

from src.utils import helper
from src.data_preparation import dataset


if __name__ == '__main__':
    slim = tf.contrib.slim

    parser = argparse.ArgumentParser(description='Default argument')
    parser.add_argument('-c',
                        dest="config_filename", type=str, required=True,
                        help='the config file name must be provide')
    args = parser.parse_args()

    config = helper.parse_config_file(args.config_filename)

    with tf.Graph().as_default() as graph, tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        # First create the dataset and load one batch
        images_batch, _ = dataset.load_batch(
            config.TRAIN_TF_RECORDS,
            config.TRAIN_BATCH_SIZE,
            config.INPUT_WIDTH,
            config.INPUT_WIDTH,
            is_training=False,
            num_epochs=1)

        sess.run(tf.local_variables_initializer())

        total = 0
        try:
            with slim.queues.QueueRunners(sess):
                while True:
                    images = sess.run(images_batch)
                    print("Loading batch size {}".format(images.shape[0]))
                    total += images.shape[0]
        except tf.errors.OutOfRangeError:
            print('End of Tfrecords, total records: {}'.format(total))
