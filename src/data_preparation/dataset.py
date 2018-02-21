import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from sklearn import preprocessing
from src.common import consts
from src.common import paths

sys.path.append("/data/slim/models/research/slim/")
from preprocessing import inception_preprocessing

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384

num_samples = 20580
num_classes = 120


def get_int64_feature(example, name):
    return int(example.features.feature[name].int64_list.value[0])


def get_float_feature(example, name):
    return int(example.features.feature[name].float_list.value)


def get_bytes_feature(example, name):
    return example.features.feature[name].bytes_list.value[0]


def read_tf_record(record):
    features = tf.parse_single_example(
        record,
        features={
            consts.IMAGE_RAW_FIELD: tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            consts.LABEL_ONE_HOT_FIELD: tf.FixedLenFeature([120], tf.float32),
            consts.INCEPTION_OUTPUT_FIELD: tf.FixedLenFeature([2048], tf.float32)
        })
    return features


def read_test_tf_record(record):
    features = tf.parse_single_example(
        record,
        features={
            'id': tf.FixedLenFeature([], tf.string),
            consts.IMAGE_RAW_FIELD: tf.FixedLenFeature([], tf.string),
            consts.INCEPTION_OUTPUT_FIELD: tf.FixedLenFeature([2048], tf.float32)
        })
    return features


def features_dataset():
    filenames_ = tf.placeholder(tf.string)
    ds_ = tf.contrib.data.TFRecordDataset(filenames_, compression_type='') \
        .map(read_tf_record)

    return ds_, filenames_


def test_features_dataset():
    filenames = tf.placeholder(tf.string)
    ds = tf.contrib.data.TFRecordDataset(filenames, compression_type='') \
        .map(read_test_tf_record)

    return ds, filenames


def read_train_image_record(record):
    with tf.device('/cpu:0'):

        features = tf.parse_single_example(
            record,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        height_ = tf.cast(features['height'], tf.int32)
        width_ = tf.cast(features['width'], tf.int32)
        image_shape = tf.stack([height_, width_, 3])
        image = tf.reshape(image, image_shape)

        aug_image = tf.image.resize_images(
                image, [IMAGE_HEIGHT + int(IMAGE_HEIGHT/5), IMAGE_WIDTH + int(IMAGE_WIDTH/5)])
        aug_image = tf.image.resize_image_with_crop_or_pad(
            aug_image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
        aug_image = tf.image.random_flip_left_right(aug_image)
        # aug_image = tf.image.random_hue(aug_image, 0.05)
        # aug_image = tf.image.random_saturation(aug_image, 0.5, 2.0)
        aug_image = tf.cast(aug_image, tf.uint8)
        features["image_resize"] = aug_image

    return features


def read_val_image_record(record):
    with tf.device('/cpu:0'):

        features = tf.parse_single_example(
            record,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        height_ = tf.cast(features['height'], tf.int32)
        width_ = tf.cast(features['width'], tf.int32)
        image_shape = tf.stack([height_, width_, 3])
        image = tf.reshape(image, image_shape)

        aug_image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
        aug_image = tf.cast(aug_image, tf.uint8)
        features["image_resize"] = aug_image

    return features


def read_test_image_record(record):
    with tf.device('/cpu:0'):

        features = tf.parse_single_example(
            record,
            features={
                'id': tf.FixedLenFeature([], tf.string),
                'width': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        height_ = tf.cast(features['height'], tf.int32)
        width_ = tf.cast(features['width'], tf.int32)
        image_shape = tf.stack([height_, width_, 3])
        image = tf.reshape(image, image_shape)

        aug_image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
        aug_image = tf.cast(aug_image, tf.uint8)
        features["image_resize"] = aug_image
    return features


def load_test_batch(tf_record_name, batch_size, width, height, num_epochs=1, capacity=2000, min_after_dequeue=1000):
    # this function return images_batch and labels_batch op that can be executed using sess.run

    def read_and_decode_single_example(filename):
        # first construct a queue containing a list of filenames.
        # this lets a user split up there dataset in multiple files to keep
        # size down
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
        # Unlike the TFRecordWriter, the TFRecordReader is symbolic
        reader = tf.TFRecordReader()
        # One can read a single serialized example from a filename
        # serialized_example is a Tensor of type string.
        _, serialized_example = reader.read(filename_queue)
        # The serialized example is converted back to actual values.
        # One needs to describe the format of the objects to be returned
        features = tf.parse_single_example(
            serialized_example,
            features={
                'id': tf.FixedLenFeature([], tf.string),
                'width': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)
            })
        # now return the converted data
        image_ = tf.decode_raw(features['image'], tf.uint8)
        height_ = tf.cast(features['height'], tf.int32)
        width_ = tf.cast(features['width'], tf.int32)
        image_shape = tf.stack([height_, width_, 3])
        image_ = tf.reshape(image_, image_shape)

        return features['id'], image_

    # returns symbolic label and image
    id_, image_raw = read_and_decode_single_example(tf_record_name)

    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=False)

    # groups examples into batches randomly
    images_batch, ids_batch = tf.train.shuffle_batch([image, id_],
                                                     batch_size=batch_size,
                                                     capacity=capacity,
                                                     min_after_dequeue=min_after_dequeue,
                                                     allow_smaller_final_batch=True)

    return images_batch, ids_batch

# def load_batch(split, batch_size, width, height, buffer_size=4000, is_training=False):
#     assert split == "train" or "eval"
#     if is_training:
#         read_image_record = read_train_image_record
#     else:
#         read_image_record = read_val_image_record
#     file_names_ = tf.placeholder(tf.string)
#     IMAGE_WIDTH = width
#     IMAGE_HEIGHT = height
#     _ds = tf.contrib.data.TFRecordDataset(file_names_).map(read_image_record)
#     ds_iter_ = _ds.shuffle(buffer_size).repeat().batch(batch_size).make_initializable_iterator()
#     return file_names_, ds_iter_

def load_batch(tf_record_name,  batch_size, width, height,
               is_training=False, num_epochs=10, capacity=2000, min_after_dequeue=1000):
    # this function return images_batch and labels_batch op that can be executed using sess.run

    def read_and_decode_single_example(filename):
        # first construct a queue containing a list of filenames.
        # this lets a user split up there dataset in multiple files to keep
        # size down
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
        # Unlike the TFRecordWriter, the TFRecordReader is symbolic
        reader = tf.TFRecordReader()
        # One can read a single serialized example from a filename
        # serialized_example is a Tensor of type string.
        _, serialized_example = reader.read(filename_queue)
        # The serialized example is converted back to actual values.
        # One needs to describe the format of the objects to be returned
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)
            })
        # now return the converted data
        image_ = tf.decode_raw(features['image'], tf.uint8)
        height_ = tf.cast(features['height'], tf.int32)
        width_ = tf.cast(features['width'], tf.int32)
        image_shape = tf.stack([height_, width_, 3])
        image_ = tf.reshape(image_, image_shape)

        return features['label'], image_

    # returns symbolic label and image
    label, image_raw = read_and_decode_single_example(tf_record_name)

    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)

    # groups examples into batches randomly
    images_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)

    return images_batch, labels_batch


def get_data_iter(sess_, tf_records_paths_, phase, buffer_size=4000, batch_size=64):
    if phase == "train":
        read_image_record = read_train_image_record
    elif phase == "val":
        read_image_record = read_val_image_record
    elif phase == "test":
        read_image_record = read_test_image_record
    else:
        raise ValueError('The phase value should be: train/val/test')

    _file_names = tf.placeholder(tf.string)
    _ds = tf.contrib.data.TFRecordDataset(_file_names).map(read_image_record)
    if phase == "train" or phase == "val":
        ds_iter_ = _ds.shuffle(buffer_size).repeat().batch(batch_size).make_initializable_iterator()
    else:
        ds_iter_ = _ds.batch(batch_size).make_initializable_iterator()
    sess_.run(ds_iter_.initializer, feed_dict={_file_names: tf_records_paths_})
    return ds_iter_.get_next()


def one_hot_label_encoder():
    dog_breeds_csv = pd.read_csv("./data/breeds.csv", dtype={'breed': np.str})
    lb = preprocessing.LabelBinarizer()
    lb.fit(dog_breeds_csv['breed'])

    def encode(label_):
        return np.asarray(lb.transform(label_), dtype=np.float32)

    def decode(one_hots):
        return np.asarray(lb.inverse_transform(one_hots), dtype=np.str)

    return encode, decode


def sparse_label_coder():
    dog_breeds_csv = pd.read_csv("./data/breeds.csv", dtype={'breed': np.str})
    _lb = preprocessing.LabelBinarizer()
    _lb.fit(dog_breeds_csv['breed'])

    def find_max_idx(lb_vec):
        _lb_vector = lb_vec.reshape(-1).tolist()
        return _lb_vector.index(max(_lb_vector))

    def encode(lbs_str):
        _lbs_vector = np.asarray(_lb.transform(lbs_str), dtype=np.float32)
        return np.apply_along_axis(find_max_idx, 1, _lbs_vector)

    def decode(one_hots):
        return _lb.inverse_transform(np.array(one_hots))

    def decode_text(label_):
        one_hot_label_ = [0]*(len(_lb.classes_))
        one_hot_label_[label_] = 1
        return _lb.inverse_transform(np.array([one_hot_label_]))

    return encode, decode, decode_text


if __name__ == '__main__':
    encoder, decoder, text_decoder = sparse_label_coder()
    print(encoder(['african_hunting_dog'])[0])

    # with tf.Graph().as_default() as g, tf.Session().as_default() as sess:
    #     ds, filenames = features_dataset()
    #     ds_iter = ds.shuffle(buffer_size=1000, seed=1).batch(10).make_initializable_iterator()
    #     next_record = ds_iter.get_next()
    #
    #     sess.run(ds_iter.initializer, feed_dict={filenames: paths.TRAIN_TF_RECORDS})
    #     features = sess.run(next_record)
    #
    #     _, one_hot_decoder = one_hot_label_encoder()
    #
    #     print(one_hot_decoder(features['inception_output']))
    #     print(features['label'])
    #     print(features['inception_output'].shape)

    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128

    with tf.Graph().as_default() as g, tf.Session().as_default() as sess:

        next_record = get_data_iter(sess, [paths.TEST_TF_RECORDS], "test")

        batch_examples = sess.run(next_record)
        images = batch_examples["image_resize"]
        # label = batch_examples["label"]

        print(images.shape)

        plt.imshow(images[2])
        plt.show()

        # idx = 0
        # try:
        #     while idx < 10:
        #         batch_examples = sess.run(next_record)
        #         images = batch_examples["image_resize"]
        #         label = batch_examples["label"]
        #         idx += 1
        #         print(images.shape)
        #
        # except tf.errors.OutOfRangeError:
        #     print('End of the dataset')
