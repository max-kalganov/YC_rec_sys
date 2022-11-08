import gin
import logging

from models.tracks_embeddings_model import train_model
import tensorflow as tf

from preprocessing.dataset_generator import ArtistTriplesDatasetGenerator
from preprocessing.utils import get_tracks_to_artists, str_to_int_dict

logging.getLogger().setLevel(logging.INFO)


def get_artists_tracks_dataset_generator():
    tracks_to_artists = get_tracks_to_artists()
    tracks_to_artists = str_to_int_dict(tracks_to_artists)

    dataset_generator = ArtistTriplesDatasetGenerator(tracks_to_artists)

    dataset = tf.data.Dataset.from_generator(dataset_generator, output_types=((tf.int8, tf.int8, tf.int8), tf.float64))
    return dataset


if __name__ == '__main__':
    gin.parse_config_file('configs/artists_tracks_embed_model_config.gin')

    logging.info("\nStarting tracks embeddings training...")
    data_generator = get_artists_tracks_dataset_generator()
    model = train_model()
