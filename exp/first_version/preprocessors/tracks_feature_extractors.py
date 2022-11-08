from typing import List
import gin
import librosa
import pandas as pd
import yandex_music as ym
from exp.first_version.preprocessors.feature_extractors.abstract_feature_extractor import FeatureExtractor
from exp.first_version.preprocessors.utils import get_track_file_path


@gin.configurable
class TracksFeatureExtractors:
    """Runs feature extractions for each track composing the results into a single pd.DataFrame"""

    feature_extractors: List[FeatureExtractor] = []

    def __init__(self, feature_extractors: List[FeatureExtractor]):
        self.feature_extractors = feature_extractors
        self.__cache = {}

    def _get_track_waveform_and_sr(self, track_id, user_ind, tracks_postfix, tracks_root_folder):
        if track_id not in self.__cache:

            track_file_path = get_track_file_path(track_id=track_id,
                                                  postfix=tracks_postfix,
                                                  track_root_folder=tracks_root_folder,
                                                  user_ind=user_ind)
            track_waveform, sr = librosa.load(track_file_path)
            self.__cache[track_id] = (track_waveform, sr)
        else:
            track_waveform, sr = self.__cache[track_id]
        return track_waveform, sr

    def process_track(
            self,
            track_id: str,
            track_download_info: ym.DownloadInfo,
            tracks_postfix: str,
            tracks_root_folder: str,
            user_ind: int
    ) -> pd.Series:
        track_waveform, sr = self._get_track_waveform_and_sr(track_id, user_ind, tracks_postfix, tracks_root_folder)

        features = {
            feature_extractor.feature_name: feature_extractor.get_feature(track_download_info, track_waveform)
            for feature_extractor in self.feature_extractors
        }
        features['user_ind'] = user_ind
        features['track_id'] = track_id
        return pd.Series(features)

    def process(
            self,
            tracks_ids: List[List[str]],
            tracks_download_info: List[List[ym.DownloadInfo]],
            tracks_root_folder: str,
            tracks_postfix: str = ""
    ) -> pd.DataFrame:
        assert len(tracks_ids) > 0, "no tracks found"
        assert len(tracks_ids) == len(tracks_download_info), f"#download_info has to be equal #tracks_ids"
        all_features = []
        for user_ind, (user_track_ids, user_track_download_info) in enumerate(zip(tracks_ids, tracks_download_info)):
            for track_id, track_download_info in zip(user_track_ids, user_track_download_info):
                all_features.append(self.process_track(track_id, track_download_info,
                                                        tracks_postfix, tracks_root_folder, user_ind=user_ind))

        return pd.concat(all_features)
