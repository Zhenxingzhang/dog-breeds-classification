import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from src.common import consts
from src.common import paths

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384


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

        # features["image_resize"] = tf.image.resize_image_with_crop_or_pad(
        #     image=image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
        aug_image = tf.cast(
            tf.image.resize_images(
                image, [IMAGE_HEIGHT + int(IMAGE_HEIGHT/5), IMAGE_WIDTH+int(IMAGE_WIDTH/5)]), tf.uint8)
        aug_image = tf.random_crop(aug_image, np.array([IMAGE_HEIGHT, IMAGE_WIDTH, 3]))
        aug_image = tf.image.random_flip_left_right(aug_image)
        # aug_image = tf.image.random_hue(aug_image, 0.05)
        # aug_image = tf.image.random_saturation(aug_image, 0.5, 2.0)
        features["image_resize"] = aug_image

    return features


def train_images_dataset():
    filenames_ = tf.placeholder(tf.string)
    ds_ = tf.contrib.data.TFRecordDataset(filenames_, compression_type='').map(read_train_image_record)

    return ds_, filenames_


def get_train_val_data_iter(sess_, tf_records_paths_, buffer_size=4000, batch_size=64):
    ds_, file_names_ = train_images_dataset()
    ds_iter_ = ds_.shuffle(buffer_size).repeat().batch(batch_size).make_initializable_iterator()
    sess_.run(ds_iter_.initializer, feed_dict={file_names_: tf_records_paths_})
    return ds_iter_.get_next()


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

        # features['image_array'] = tf.decode_raw(features['image'], tf.uint8)
        height_ = tf.cast(features['height'], tf.int32)
        width_ = tf.cast(features['width'], tf.int32)
        image_shape = tf.stack([height_, width_, 3])
        image = tf.reshape(image, image_shape)

        features["image_resize"] = tf.image.resize_image_with_crop_or_pad(
            image=image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
        # features["image_resize"] = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return features


def test_images_dataset():
    filenames_ = tf.placeholder(tf.string)
    ds_ = tf.contrib.data.TFRecordDataset(filenames_, compression_type='').map(read_test_image_record)

    return ds_, filenames_


def get_test_data_iter(sess_, tf_records_paths_, buffer_size=4000, batch_size=64):
    ds_, file_names_ = test_images_dataset()
    ds_iter_ = ds_.shuffle(buffer_size).repeat().batch(batch_size).make_initializable_iterator()
    sess_.run(ds_iter_.initializer, feed_dict={file_names_: tf_records_paths_})
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
        ds, filenames = train_images_dataset()
        ds_iter = ds.batch(10).make_initializable_iterator()
        next_record = ds_iter.get_next()

        sess.run(ds_iter.initializer, feed_dict={filenames: paths.VAL_TF_RECORDS})
        batch_examples = sess.run(next_record)
        images = batch_examples["image_resize"]
        label = batch_examples["label"]

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
