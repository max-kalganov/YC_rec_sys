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
