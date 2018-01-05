import tensorflow as tf
import argparse
import os
import yaml
import numpy as np

from src.common import paths
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
    probability = tf.contrib.layers.softmax(logits)

    # Add ops to restore values of the variables created from forward pass
    # from checkpoints
    saver = tf.train.Saver(tf.trainable_variables())

    test_tfrecord_file = paths.TEST_TF_RECORDS

    breeds = decoder(np.identity(categories))

    # Start session
    with tf.Session() as sess:
        next_test_batch = dataset.get_test_data_iter(sess, [test_tfrecord_file], batch_size=test_bz)

        # Restore previously trained variables from disk
        # Variables constructed in forward_pass() will be initialised with
        # values restored from variables with the same name
        # Note: variable names MUST match for it to work
        print("Restoring Saved Variables from Checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

        output_csv = os.path.join(output_path, "prediction.txt")
        print("Write results to: {}".format(output_csv))
        with open(output_csv, "w") as output:
            output.write("id,{}\n".format(",".join(str(dog) for dog in breeds)))
            try:
                iter = 0
                while True:
                    test_batch_examples = sess.run(next_test_batch)
                    test_images = test_batch_examples["image_resize"]
                    test_ids = test_batch_examples["id"]

                    prediction_ = sess.run(probability, {input_images: test_images, keep_prob_tensor: 1.0})

                    for (prob_list, id_) in zip(prediction_, test_ids):
                        output.writelines("{},{}\n".format(id_, ",".join(str(prob_) for prob_ in prob_list)))
                    iter += 1
                    print("processing {} test records".format(iter * test_bz))
            except tf.errors.OutOfRangeError:
                print('End of the data set')

        print("Inference finished")


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
    OUTPUT_PATH = os.path.join(str(cfg["TEST"]["OUTPUT_PATH"]), MODEL_NAME)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    main(MODEL_NAME, TRAIN_LEARNING_RATE, INPUT_HEIGHT, INPUT_WIDTH, TEST_BATCH_SIZE, CATEGORIES, OUTPUT_PATH)