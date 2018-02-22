import os

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse

from src.utils import helper
from src.data_preparation import dataset
from src.common import paths

import sys
sys.path.append("/data/slim/models/research/slim/")

from nets import nets_factory


def infer_test(config_):
    model_path = os.path.join(paths.CHECKPOINT_DIR, config_.MODEL_NAME, str(config_.TRAIN_LEARNING_RATE))
    if not os.path.exists(model_path):
        print("Model not exist: {}".format(model_path))
        exit()

    output_path = os.path.join(config_.TEST_OUTPUT, config_.MODEL_NAME)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    test_output = os.path.join(output_path, "results.csv")

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        # First create the dataset and load one batch
        images, ids = dataset.load_test_batch(config_.TEST_TF_RECORDS,
                                            config_.TEST_BATCH_SIZE,
                                            config_.INPUT_WIDTH,
                                            config_.INPUT_WIDTH, num_epochs=1)
        # Create the model inference
        net_fn = nets_factory.get_network_fn(
            config_.PRETAIN_MODEL,
            dataset.num_classes,
            is_training=False)

        _, end_points = net_fn(images)
        predictions = end_points['Predictions']

        # Define the scopes that you want to exclude for restoration
        variables_to_restore = slim.get_variables_to_restore()

        # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        checkpoint_path = tf.train.latest_checkpoint(model_path)
        init_fn = slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore)

        _, decoder, _ = dataset.sparse_label_coder()
        breeds = decoder(np.identity(dataset.num_classes))

        with tf.Session() as sess, open(test_output, "w") as output:

            print("writing results to {}".format(test_output))
            init_fn(sess)
            print("Restore model from: {}".format(model_path))

            sess.run(tf.local_variables_initializer())

            output.write("id,{}\n".format(",".join(str(dog) for dog in breeds)))

            try:
                with slim.queues.QueueRunners(sess):
                    while True:
                        test_ids, preds = sess.run([ids, predictions])
                        for (prob_list, id_) in zip(preds, test_ids):
                            output.writelines("{},{}\n".format(id_, ",".join(str(prob_) for prob_ in prob_list)))
            except tf.errors.OutOfRangeError:
                print('')

        print('predictions saved to %s' % test_output)


if __name__ == '__main__':
    slim = tf.contrib.slim

    parser = argparse.ArgumentParser(description='Default argument')
    parser.add_argument('-c',
                        dest="config_filename", type=str, required=True,
                        help='the config file name must be provide')
    args = parser.parse_args()

    arg_config = helper.parse_config_file(args.config_filename)
    infer_test(arg_config)







