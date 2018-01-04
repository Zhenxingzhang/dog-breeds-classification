import tensorflow as tf
import numpy as np
from tqdm import tqdm
import scipy.misc
from src.common import paths
import os
import re


def grey_to_rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def csv_to_record(csv_file, tfrecord_file, is_train=True):
    with open(csv_file) as f:
        lines = f.readlines()
        np.random.shuffle(lines)

    # iterate over each example
    # wrap with tqdm for a progress bar
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for line in tqdm(lines):
            path = str(line.split(',')[0])
            image = np.array(scipy.misc.imread(path, flatten=False, mode='RGB'))
            height = image.shape[0]
            width = image.shape[1]
            image = grey_to_rgb(image) if image.shape[2] == 1 else image
            image_raw = image.tostring()

            text_label = re.sub(r"[\n\t\s]*", "", line.split(',')[1])
            label = -1 if (text_label == '' or text_label is None) else text_label

            # construct the Example proto object
            if is_train:
                example = tf.train.Example(
                    # Example contains a Features proto object
                    features=tf.train.Features(feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
                        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                    }))
            else:
                example = tf.train.Example(
                    # Example contains a Features proto object
                    features=tf.train.Features(feature={
                        'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(label)])),
                        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                    }))

            # use the proto object to serialize the example to a string
            serialized = example.SerializeToString()
            # write the serialized object to disk
            writer.write(serialized)


if __name__ == '__main__':

    np.random.seed(0)

    # create TFRecords from csv files if necessary
    train_csv = paths.TRAIN_CSV_FILE
    train_tfrecord = paths.TRAIN_TF_RECORDS

    val_csv = paths.VAL_CSV_FILE
    val_tfrecord = paths.VAL_TF_RECORDS

    test_csv = paths.TEST_CSV_RECORDS
    test_tfrecord = paths.TEST_TF_RECORDS

    if not os.path.exists(train_tfrecord):
        print('Creating TFRecord from csv files for set: {}'.format('train'))
        csv_to_record(train_csv, train_tfrecord)
    else:
        print('TFRecord exists, nothing to do: {}'.format(train_tfrecord))

    if not os.path.exists(val_tfrecord):
        print('Creating TFRecord from csv files for set: {}'.format('val'))
        csv_to_record(val_csv, val_tfrecord)
    else:
        print('TFRecord exists, nothing to do: {}'.format(val_tfrecord))

    if not os.path.exists(test_tfrecord):
        print('Creating TFRecord from csv files for set: {}'.format('test'))
        csv_to_record(test_csv, test_tfrecord, False)
    else:
        print('TFRecord exists, nothing to do: {}'.format(test_tfrecord))

