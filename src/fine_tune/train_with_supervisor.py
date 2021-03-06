from tensorflow.contrib import slim

import sys
sys.path.append("/data/slim/models/research/slim/")
from src.utils import helper
from src.data_preparation import dataset
from src.common import paths
from nets import nets_factory

import tensorflow as tf
import os
import time
import argparse


def run(config):

    # This might take a few minutes.
    train_summary_dir = os.path.join(paths.TRAIN_SUMMARY_DIR, config.MODEL_NAME, str(config.TRAIN_LEARNING_RATE))
    checkpoint_dir = os.path.join(paths.CHECKPOINT_DIR, config.MODEL_NAME, str(config.TRAIN_LEARNING_RATE))

    # Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # ======================= TRAINING PROCESS =========================
    # Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        # First create the dataset and load one batch
        _, images, labels = dataset.load_batch(
            config.TRAIN_TF_RECORDS,
            config.TRAIN_BATCH_SIZE,
            config.INPUT_WIDTH,
            config.INPUT_WIDTH,
            is_training=True,
            num_epochs=config.TRAIN_EPOCHS_COUNT)

        tf.summary.image('images/train', images)

        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / config.TRAIN_BATCH_SIZE)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(config.TRAIN_EPOCHS_BEFORE_DECAY * num_steps_per_epoch)

        # Create the model inference
        net_fn = nets_factory.get_network_fn(
            config.PRETAIN_MODEL,
            dataset.num_classes,
            weight_decay=config.L2_WEIGHT_DECAY,
            is_training=True)

        logits, end_points = net_fn(images)

        # Define the scopes that you want to exclude for restoration
        variables_to_restore = slim.get_variables_to_restore(exclude=arg_config.EXCLUDE_NODES)

        # Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        tf.summary.scalar('losses/entropy_loss', entropy_loss)

        regular_loss = tf.losses.get_regularization_loss()
        tf.summary.scalar('losses/regular_loss', regular_loss)

        # obtain the regularization losses as well
        total_loss = tf.reduce_mean(tf.losses.get_total_loss())
        tf.summary.scalar('losses/total_loss', total_loss)

        # # Specify the loss function, this will add regulation loss as well:
        # one_hot_labels = slim.one_hot_encoding(labels, 5)
        # slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        # total_loss = slim.losses.get_total_loss()

        # Create the global step for monitoring the learning_rate and training.
        global_step = tf.train.get_or_create_global_step()

        # Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate=config.TRAIN_LEARNING_RATE,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=config.TRAIN_RATE_DECAY_FACTOR,
            staircase=True)

        # Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        if config.TRAINABLE_SCOPES is not None:
            scopes = [scope.strip() for scope in config.TRAINABLE_SCOPES]
            training_variables = []
            for scope in scopes:
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                training_variables.extend(variables)
        else:
            training_variables = None

        # Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=training_variables)

        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.metrics.accuracy(labels, predictions)
        metrics_op = tf.group(accuracy_update, probabilities)

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        # precision, recall, f1, _ = score(labels, predictions)
        # tf.summary.scalar('precision', np.mean(precision))
        # tf.summary.scalar('Recall', np.mean(recall))
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()

        # Now we need to create a training step function that runs both the train_op,
        # metrics_op and updates the global_step concurrently.
        def train_step(sess_, train_op_, global_step_):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            # Check the time for each sess run
            start_time = time.time()
            total_loss_, global_step_count, _ = sess_.run([train_op_, global_step_, metrics_op])
            time_elapsed = time.time() - start_time

            # Run the logging to print some results
            print('global step {}: loss: {:.4f} ({:.2f} sec/step)'.format(global_step_count, total_loss_, time_elapsed))

            return total_loss_, global_step_count

        # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess_):
            return saver.restore(sess_, config.PRETAIN_MODEL_PATH)

        train_summ_writer = tf.summary.FileWriter(train_summary_dir, graph)

        # Define your supervisor for running a managed session.
        # Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(
            save_model_secs=300,
            logdir=checkpoint_dir,
            summary_op=None,
            init_fn=restore_fn,
            summary_writer=train_summ_writer)

        # Run the managed session
        with sv.managed_session() as sess:

            for step in xrange(num_steps_per_epoch * config.TRAIN_EPOCHS_COUNT):
                # At the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:
                    print('Epoch {}/{}'.format(step / num_batches_per_epoch + 1, config.TRAIN_EPOCHS_COUNT))
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    print('Current Learning Rate: {:f}'.format(learning_rate_value))
                    print('Current Streaming Accuracy: {:f}'.format(accuracy_value))

                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                        [logits, probabilities, predictions, labels])
                    print 'predictions: \n', predictions_value
                    print 'Labels:\n', labels_value

                # Log the summaries every 10 step.
                if step % 10 == 0:
                    loss, _ = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

                # If not, simply run the training step
                else:
                    loss, _ = train_step(sess, train_op, sv.global_step)

            # We log the final training loss and accuracy
            print('Final Loss: {:f}'.format(loss))
            print('Final Accuracy: {:f}'.format(sess.run(accuracy)))

            # Once all the training has been done, save the log files and checkpoint model
            print('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Default argument')
    parser.add_argument('-c',
                        dest="config_filename", type=str, required=True,
                        help='the config file name must be provide')
    args = parser.parse_args()

    arg_config = helper.parse_config_file(args.config_filename)
    run(arg_config)
