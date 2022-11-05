from typing import List, Iterable, Optional, Tuple
import gin
import librosa
import pandas as pd
import yandex_music as ym
from preprocessors.feature_extractors.abstract_feature_extractor import FeatureExtractor
from preprocessors.utils import get_track_file_path
import numpy as np


@gin.configurable
class TracksPreprocessor:
    feature_extractors: List[FeatureExtractor] = []

    def __init__(self, feature_extractors: List[FeatureExtractor]):
        self.feature_extractors = feature_extractors

    def process_track(
            self,
            track_id: str,
            track_download_info: ym.DownloadInfo,
            tracks_postfix: str,
            tracks_root_folder: str
    ) -> pd.Series:
        track_filepath = get_track_file_path(track_id=track_id,
                                             postfix=tracks_postfix,
                                             track_root_folder=tracks_root_folder)
        track_waveform, sr = librosa.load(track_filepath)

        features = {
            feature_extractor.feature_name: feature_extractor.get_feature(track_download_info, track_waveform)
            for feature_extractor in self.feature_extractors
        }
        return pd.Series(features)

    def process(
            self,
            tracks_ids: List[List[str]],
            tracks_download_info: List[List[ym.DownloadInfo]],
            tracks_root_folder: str,
            tracks_postfix: str = ""
    ) -> Iterable[Tuple[int, List]]:
        assert len(tracks_ids) > 0, "no tracks found"
        assert len(tracks_ids) == len(tracks_download_info), f"#download_info has to be equal #tracks_ids"
        for user_ind, (user_track_ids, user_track_download_info) in enumerate(zip(tracks_ids, tracks_download_info)):
            user_features = []
            for track_id, track_download_info in zip(user_track_ids, user_track_download_info):
                user_features.append(self.process_track(track_id, track_download_info,
                                                        tracks_postfix, tracks_root_folder))
            yield user_ind, user_features
