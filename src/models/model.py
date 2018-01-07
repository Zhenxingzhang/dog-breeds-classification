import tensorflow as tf
import numpy as np


def mnist_net(x_input, classes, dropout_keep_prob):
    """Model function for CNN."""
    # Input Layer
    x_input = (x_input - 128.0) / 128.0

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=x_input,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    print(pool1.shape)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    print(pool2.shape)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    print(pool3.shape)

    # Dense Layer
    input_shape = pool3.get_shape()
    ndims = np.int(np.product(input_shape[1:]))
    print(ndims)

    pool3_flat = tf.reshape(pool3, [-1, ndims])
    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=dropout_keep_prob)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=classes)
    return logits


def conv_net(x_input, classes, dropout_keep_prob):
    # Input Layer
    x_input = (x_input - 128.0) / 128.0

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=x_input,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    print(conv1.shape)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    print(pool2.shape)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    print(pool3.shape)

    # Convolutional Layer #3 and Pooling Layer #3
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    print(pool4.shape)

    # Dense Layer
    input_shape = pool4.get_shape()
    ndims = np.int(np.product(input_shape[1:]))
    print(ndims)

    pool_flat = tf.reshape(pool4, [-1, ndims])
    dense1 = tf.layers.dense(inputs=pool_flat, units=512, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=dropout_keep_prob)

    dense2 = tf.layers.dense(inputs=dropout1, units=256, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=dense2, rate=dropout_keep_prob)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=classes)
    return logits


if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    keep_prob = tf.placeholder(tf.float32)
    logits = conv_net(x, 120, keep_prob)
