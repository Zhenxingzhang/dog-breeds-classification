"""
Restores variable values from previously trained MNIST ConvNet model, re runs
inference process and evaluates accuracy against test data
"""

from src.common import paths
import tensorflow as tf
import argparse
import os
import yaml

from src.models import conv_net
from src.data_preparation import dataset


# Silence compile warnings
def main(model_name, l_rate, input_h, input_w, test_bz, categories, output_path):
    _, decoder, _ = dataset.sparse_label_coder()
    dataset.IMAGE_HEIGHT = input_h
    dataset.IMAGE_WIDTH = input_w
    # Get latest checkpoint file from dir
    latest_checkpoint = os.path.join(paths.CHECKPOINTS_DIR, model_name, str(l_rate), "model.ckpt")

    # Compute forward pass
    # Note: if you are not restoring Graph, you need to create
    # variables before you can restore their values from checkpoint
    with tf.name_scope("input"):
        input_images = tf.placeholder(tf.float32, shape=[None, input_h, input_w, 3])

    with tf.name_scope('dropout_keep_prob'):
        keep_prob_tensor = tf.placeholder(tf.float32)

    logits = conv_net.conv_net_small(input_images, categories, keep_prob_tensor)
    prediction = tf.argmax(logits, 1)
    # Add ops to restore values of the variables created from forward pass
    # from checkpoints
    saver = tf.train.Saver(tf.trainable_variables())

    test_tfrecord_file = paths.TEST_TF_RECORDS

    # Start session
    with tf.Session() as sess:
        next_test_batch = dataset.get_test_data_iter(sess, [test_tfrecord_file], batch_size=test_bz)

        # Restore previously trained variables from disk
        # Variables constructed in forward_pass() will be initialised with
        # values restored from variables with the same name
        # Note: variable names MUST match for it to work
        print("Restoring Saved Variables from Checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

        with open(os.path.join(paths.OUTPUT_PATH, "prediction.txt"), "w") as output:
            try:
                while True:
                    test_batch_examples = sess.run(next_test_batch)
                    test_images = test_batch_examples["image_shape"]
                    test_filename = test_batch_examples["filename"]

                    prediction_ = sess.run(prediction, {input_images: test_images, keep_prob_tensor: 1.0})
                    pred_label = decoder(prediction_)

                    for (p_label, filename) in zip(pred_label, test_filename):
                        output.writelines("{} {}\n".format(filename, p_label))

            except tf.errors.OutOfRangeError:
                print('End of the dataset')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Default argument')
    parser.add_argument('-c', dest="config_filename", type=str, required=True,
                        help='the config file name must be provide')
    args = parser.parse_args()

    with open(args.config_filename, 'r') as yml_file:
        cfg = yaml.load(yml_file)

    MODEL_NAME = str(cfg["MODEL"]["MODEL_NAME"])
    INPUT_HEIGHT = int(cfg["MODEL"]["INPUT_HEIGHT"])
    INPUT_WIDTH = int(cfg["MODEL"]["INPUT_WIDTH"])
    CATEGORIES = int(cfg["MODEL"]["CLASSES"])

    TRAIN_LEARNING_RATE = float(cfg["TRAIN"]["LEARNING_RATE"])

    TEST_BATCH_SIZE = cfg["TEST"]["BATCH_SIZE"]
    OUTPUT_PATH = str(cfg["TEST"]["OUTPUT_PATH"])

    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    main(MODEL_NAME, TRAIN_LEARNING_RATE, INPUT_HEIGHT, INPUT_WIDTH, TEST_BATCH_SIZE, CATEGORIES, OUTPUT_PATH)