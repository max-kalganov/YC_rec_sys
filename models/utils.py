import os.path

import gin
import tensorflow as tf


@gin.configurable
def triplet_loss(y_true, y_pred, embedding_size, alpha):
    anchor = y_pred[:, :embedding_size]
    positive = y_pred[:, embedding_size:2 * embedding_size]
    negative = y_pred[:, 2 * embedding_size:]

    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)


@gin.configurable
def triplet_pos_dist(y_true, y_pred, embedding_size):
    anchor = y_pred[:, :embedding_size]
    positive = y_pred[:, embedding_size:2 * embedding_size]

    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    return positive_dist


@gin.configurable
def triplet_neg_dist(y_true, y_pred, embedding_size):
    anchor = y_pred[:, :embedding_size]
    negative = y_pred[:, 2 * embedding_size:]

    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return negative_dist


@gin.configurable
def triplet_pos_neg_compare(y_true, y_pred, embedding_size):
    anchor = y_pred[:, :embedding_size]
    positive = y_pred[:, embedding_size:2 * embedding_size]
    negative = y_pred[:, 2 * embedding_size:]

    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return positive_dist - negative_dist


@gin.configurable
def get_tensorboard_callback(log_dir, experiment_name: str):
    log_dir = os.path.join(log_dir, experiment_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    return tensorboard_callback