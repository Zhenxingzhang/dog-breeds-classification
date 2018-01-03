import tensorflow as tf
from src.common import paths
from src.data_preparation import dataset
from src.models import mnist_net
from src.common import consts
import os
import datetime

if __name__ == "__main__":
    BATCH_SIZE = 128
    NUM_STEPS = 30001
    LEARNING_RATE = 1e-3

    with tf.name_scope("input"):
        input_images = tf.placeholder(tf.float32, shape=[None, consts.IMAGE_HEIGHT, consts.IMAGE_WIDTH, 3])
        label = tf.placeholder(tf.int64)
        input_images_summary = tf.summary.image('images', input_images)

    with tf.name_scope('dropout_keep_prob'):
        keep_prob_tensor = tf.placeholder(tf.float32)

    logits = mnist_net(input_images, consts.CLASSES_COUNT, keep_prob_tensor)

    print(logits.shape)
    # for monitoring
    with tf.name_scope('loss'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
        loss_mean = tf.reduce_mean(loss)
        tf.summary.scalar('loss', loss_mean)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), label)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_mean)

    summary_op = tf.summary.merge_all()

    variables_to_store = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(var_list=variables_to_store)

    train_tfrecord_file = paths.TRAIN_TF_RECORDS
    val_tfrecord_file = paths.VAL_TF_RECORDS

    with tf.Session() as sess:
        next_train_batch = dataset.get_train_val_data_iter(sess, [train_tfrecord_file], batch_size=BATCH_SIZE)
        next_val_batch = dataset.get_train_val_data_iter(sess, [val_tfrecord_file], batch_size=BATCH_SIZE)

        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(
            os.path.join(paths.TRAIN_SUMMARY_DIR,
                         str(LEARNING_RATE),
                         datetime.datetime.now().strftime("%Y%m%d-%H%M")),
            sess.graph)
        val_writer = tf.summary.FileWriter(
            os.path.join(paths.VAL_SUMMARY_DIR,
                         str(LEARNING_RATE),
                         datetime.datetime.now().strftime("%Y%m%d-%H%M")),
            sess.graph)

        print("Start training with ")

        checkpoint_dir = os.path.join(paths.CHECKPOINTS_DIR, str(LEARNING_RATE))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        for i in range(NUM_STEPS):
            train_batch_examples = sess.run(next_train_batch)
            train_images = train_batch_examples["image_resize"]
            train_labels = train_batch_examples["label"]

            _, step_loss, step_summary = sess.run([train_op, loss_mean, summary_op],
                                                  feed_dict={input_images: train_images,
                                                             label: train_labels,
                                                             keep_prob_tensor: 1.0})
            train_writer.add_summary(step_summary, i)
            print("Step {}, train loss: {}".format(i, step_loss))

            if i % 10 == 0:
                saver.save(sess, os.path.join(checkpoint_dir, "model.ckpt"))

                val_batch_examples = sess.run(next_val_batch)
                val_images = val_batch_examples["image_resize"]
                val_labels = val_batch_examples["label"]

                val_step_loss, val_step_summary = sess.run([loss_mean, summary_op],
                                                           feed_dict={input_images: val_images,
                                                                      label: val_labels,
                                                                      keep_prob_tensor: 1.0})
                val_writer.add_summary(val_step_summary, i)
            #     print("Step {}, val loss: {}".format(i, val_step_loss))

    print("Finish training from scratch...")
