import gin
import tensorflow as tf

from models.utils import triplet_loss
from preprocessing.dataset_generator import ArtistTriplesDatasetGenerator
from preprocessing.utils import get_tracks_to_artists, str_to_int_dict


@gin.configurable
def get_artists_embed_model(embedding_size: int, max_track_id: int):
    embedding_model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(max_track_id, embedding_size, name="Tracks embedding by artists layer")
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
def train_model(epochs, batch_size, data_generator):
    model = get_artists_triple_loss_model()

    model.compile(optimizer='adam', loss=triplet_loss)
    model.fit(data_generator, batch_size=batch_size, epochs=epochs)
    return model
