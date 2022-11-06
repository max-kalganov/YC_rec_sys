import abc
import yandex_music as ym
import numpy as np


class FeatureExtractor(abc.ABC):
    feature_name: str = "default feature name"

    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    @abc.abstractmethod
    def get_feature(self, download_info: ym.DownloadInfo, waveform: np.ndarray) -> float:
        pass
