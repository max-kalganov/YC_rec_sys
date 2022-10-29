from typing import List, Iterable, Optional
import gin
import librosa
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
    ) -> np.ndarray:
        track_filepath = get_track_file_path(track_id=track_id,
                                             postfix=tracks_postfix,
                                             track_root_folder=tracks_root_folder)
        track_waveform, sr = librosa.load(track_filepath)

        features = np.array([
            feature_extractor.get_feature(
                track_download_info,
                track_waveform
            ) for feature_extractor in self.feature_extractors
        ])
        return features

    def process(
            self,
            tracks_ids: List[str],
            tracks_download_info: List[ym.DownloadInfo],
            tracks_postfix: str,
            tracks_root_folder: str,
            batch_size: Optional[int] = None
    ) -> Iterable[np.ndarray]:
        assert batch_size is None or batch_size >= 1, f"incorrect batch size = {batch_size}"
        assert len(tracks_ids) > 0, "no tracks found"
        assert len(tracks_ids) == len(tracks_download_info), f"#download_info has to be equal #tracks_ids"
        batch_features = []
        for i, (track_id, track_download_info) in enumerate(zip(tracks_ids, tracks_download_info)):
            batch_features.append(self.process_track(track_id, track_download_info, tracks_postfix, tracks_root_folder))
            if batch_size is not None and i % batch_size == 0:
                batch_features_np = np.concatenate(batch_features)
                yield batch_features_np
                batch_features = []

        batch_features_np = np.concatenate(batch_features)
        yield batch_features_np
