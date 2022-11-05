"""Loads wav files for each track, runs feature extractors for each track storing the results into csv files"""
import os.path

import gin
from typing import List

from preprocessors.tracks_loader import WAVTracksLoader
from preprocessors.tracks_feature_extractors import TracksFeatureExtractors
from preprocessors.utils import get_train_val_tracks_ids, get_test_tracks_ids


def dump_a_subset(subset_track_ids: List[List[str]],
                  tracks_root_folder: str,
                  wav_subset_folder: str,
                  features_subset_filename: str):
    subset_root_folder = os.path.join(tracks_root_folder, wav_subset_folder)
    features_path = os.path.join(subset_root_folder, features_subset_filename)
    # Load wav tracks into subset_root_folder (loaded only 30s of each track)
    tracks_loader = WAVTracksLoader(tracks_root_folder=subset_root_folder)
    tracks_loader.load_tracks(users_tracks_ids=subset_track_ids)
    tracks_download_info = tracks_loader.get_download_info(users_tracks_ids=subset_track_ids)

    # Initialize TracksFeatureExtractors which will run all feature extraction tools
    tracks_feature_extractors = TracksFeatureExtractors()

    tracks_features_df = tracks_feature_extractors.process(tracks_ids=subset_track_ids,
                                                           tracks_download_info=tracks_download_info,
                                                           tracks_root_folder=tracks_root_folder)
    tracks_features_df.to_csv(features_path, index=False)


@gin.configurable
def run_tracks_preload_features(tracks_root_folder: str):
    train_tracks_ids, val_tracks_ids = get_train_val_tracks_ids()
    dump_a_subset(subset_track_ids=train_tracks_ids, tracks_root_folder=tracks_root_folder,
                  wav_subset_folder='train_wav', features_subset_filename='train_features.csv')
    dump_a_subset(subset_track_ids=val_tracks_ids, tracks_root_folder=tracks_root_folder,
                  wav_subset_folder='val_wav', features_subset_filename='val_features.csv')

    test_tracks_ids = get_test_tracks_ids()
    dump_a_subset(subset_track_ids=test_tracks_ids, tracks_root_folder=tracks_root_folder,
                  wav_subset_folder='test_wav', features_subset_filename='test_features.csv')


if __name__ == '__main__':
    gin.parse_config_file('configs/fe_preloader_config.gin')
    run_tracks_preload_features()
