import logging
import os

import gin
from tensorboard.plugins import projector
import tensorflow as tf

from models.tracks_embeddings_model import load_model
from preprocessing.dataset_generator import ArtistTriplesDatasetGenerator
from preprocessing.utils import get_tracks_to_artists, str_to_int_dict
from scripts.train_tracks_artists_embeddings import get_artists_tracks_dataset_generator


@gin.configurable
def dump_embeddings(model, data_generator: ArtistTriplesDatasetGenerator, max_track_id: int, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for possible_track_id in range(max_track_id):
            f.write(f"{possible_track_id} "
                    f"- {str(data_generator._tracks_to_artist.get(possible_track_id, 'Unknown'))}\n")

    # Save the weights we want to analyze as a variable. Note that the first
    # value represents any unknown word, which is not in the metadata, here
    # we will remove this value.
    embedding_layer = model.layers[3].layers[0]
    weights = tf.Variable(embedding_layer.get_weights()[0][1:])
    # Create a checkpoint from embedding, the filename and key are the
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Set up config.
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)


if __name__ == '__main__':
    gin.parse_config_file('configs/artists_tracks_embed_model_config.gin')

    logging.info("\nChecking embeddings...")
    model = load_model()
    # data_generator = get_artists_tracks_dataset_generator()
    tracks_to_artists = get_tracks_to_artists()
    tracks_to_artists = str_to_int_dict(tracks_to_artists)

    dataset_generator = ArtistTriplesDatasetGenerator(tracks_to_artists)

    dump_embeddings(model, dataset_generator)
