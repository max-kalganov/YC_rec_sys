import logging

import gin

from preprocessing.dataset_generator import ArtistTriplesDatasetGenerator, UsersTriplesDatasetGenerator
from preprocessing.utils import get_tracks_to_artists, str_to_int_dict, get_train_val_tracks_ids, str_to_int_list

logging.getLogger().setLevel(logging.INFO)


def iterate_example(dataset_generator):
    for i in range(3):
        (np_anchors, np_positives, np_negatives), label = next(dataset_generator)
        logging.info(f"\nGenerated: {np_anchors.shape=}, {np_positives.shape=}, {np_negatives.shape=}, {label.shape=}\n"
                     f"{[np_anchors, np_positives, np_negatives]}")


def run_artist_triples_dataset_generation():
    tracks_to_artists = get_tracks_to_artists()
    tracks_to_artists = str_to_int_dict(tracks_to_artists)

    dataset_generator = ArtistTriplesDatasetGenerator(tracks_to_artists, 5)
    iterate_example(dataset_generator)


def run_users_triples_dataset_generation():
    train_users_tracks, val_users_tracks = get_train_val_tracks_ids()
    train_users_tracks = str_to_int_list(train_users_tracks)

    train_dataset_generator = UsersTriplesDatasetGenerator(train_users_tracks, 5)
    iterate_example(train_dataset_generator)


if __name__ == '__main__':
    gin.parse_config_file('configs/dataset_generation.gin')
    logging.info("\n\nRunning 'run_artist_triples_dataset_generation'")
    run_artist_triples_dataset_generation()

    logging.info("\n\nRunning 'run_users_triples_dataset_generation'")
    run_users_triples_dataset_generation()
