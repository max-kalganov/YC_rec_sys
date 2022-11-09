import logging

import gin

from preprocessing.utils import get_tracks_to_artists, str_to_int_dict

if __name__ == '__main__':
    gin.parse_config_file('configs/dataset_generation.gin')
    logging.info("\nRunning 'calc_num_of_artists'...")

    tracks_to_artists = get_tracks_to_artists()
    tracks_to_artists = str_to_int_dict(tracks_to_artists)
    all_artists = set(tracks_to_artists.values())
    print(f"{max(all_artists)=}, {min(all_artists)=}, {len(all_artists)=}")
