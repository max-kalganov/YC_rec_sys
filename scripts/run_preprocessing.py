import os.path

import gin
import numpy as np
from typing import List, Optional

from preprocessors.tracks_loader import WAVTracksLoader
from preprocessors.tracks_preprocessor import TracksPreprocessor


@gin.configurable
def preprocess_a_subset(subset_track_ids: List[str],
                        tracks_root_folder: str,
                        wav_subset_folder: str,
                        features_subset_folder: str,
                        batch_size: Optional[int],
                        batch_filename: str = "batch_features"):
    subset_root_folder = os.path.join(tracks_root_folder, wav_subset_folder)

    tracks_donwload_info = WAVTracksLoader(tracks_root_folder=subset_root_folder).load_tracks(
        tracks_ids=subset_track_ids)
    tracks_preprocessor = TracksPreprocessor()
    for i, batch_features in enumerate(tracks_preprocessor.process(tracks_ids=subset_track_ids,
                                                                   tracks_download_info=tracks_donwload_info,
                                                                   tracks_root_folder=tracks_root_folder,
                                                                   batch_size=batch_size)):
        batch_file = os.path.join(tracks_root_folder, features_subset_folder, f"{batch_filename}_{i}.npy")
        np.save(batch_file, batch_features)


@gin.configurable
def run_tracks_preprocessing(tracks_root_folder: str, batch_size: Optional[int]):
    train_tracks_ids, test_tracks_ids = get_train_test_tracks_ids()
    preprocess_a_subset(train_tracks_ids, tracks_root_folder, 'train_wav', 'train', batch_size)
    preprocess_a_subset(test_tracks_ids, tracks_root_folder, 'test_wav', 'test', batch_size=None)


if __name__ == '__main__':
    run_tracks_preprocessing()
