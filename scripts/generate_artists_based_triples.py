import logging

import gin

from preprocessing.dataset_generator import DatasetGenerator
from preprocessing.utils import get_tracks_to_artists, str_to_int_dict

logging.getLogger().setLevel(logging.INFO)


@gin.configurable
def run_artist_triples_dataset_generation(tracks_root_folder: str, batch_size: int):
    tracks_to_artists = get_tracks_to_artists()
    tracks_to_artists = str_to_int_dict(tracks_to_artists)

    dataset_generator = DatasetGenerator()
    for np_anchors, np_positives, np_negatives in dataset_generator.generate_artist_triples(tracks_to_artists,
                                                                                            batch_size):
        logging.info(f"Generated: {len(np_anchors)=}, {len(np_positives)=}, {len(np_negatives)=}\n"
                     f"{list(zip(np_anchors[:3], np_positives[:3], np_negatives[:3]))}")


if __name__ == '__main__':
    gin.parse_config_file('configs/dataset_generation.gin')
    run_artist_triples_dataset_generation()
