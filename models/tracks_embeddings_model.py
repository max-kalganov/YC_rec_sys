import gin
import tensorflow as tf

from models.utils import triplet_loss, triplet_pos_dist, triplet_pos_neg_compare, triplet_neg_dist, \
    get_tensorboard_callback


@gin.configurable
def get_artists_embed_model(embedding_size: int, max_track_id: int):
    embedding_model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(max_track_id, embedding_size, name="tracks_embedding_by_artists_layer")
    ])

    embedding_model.summary()
    return embedding_model


@gin.configurable
def get_artists_triple_loss_model(embedding_size: int):
    input_anchor = tf.keras.layers.Input(shape=(embedding_size,))
    input_positive = tf.keras.layers.Input(shape=(embedding_size,))
    input_negative = tf.keras.layers.Input(shape=(embedding_size,))

    embedding_model = get_artists_embed_model(embedding_size=embedding_size)

    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)

    output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

    net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)
    net.summary()
    return net


@gin.configurable
def train_model(epochs, batch_size, steps_per_epoch, data_generator):
    model = get_artists_triple_loss_model()

    model.compile(optimizer='adam', loss=triplet_loss, metrics=[triplet_pos_dist,
                                                                triplet_neg_dist,
                                                                triplet_pos_neg_compare])

    tensorboard_callback = get_tensorboard_callback()
    model.fit(data_generator, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch,
              callbacks=[tensorboard_callback])
    return model


@gin.configurable
def save_model(model, save_dir: str):
    model.save(save_dir)
