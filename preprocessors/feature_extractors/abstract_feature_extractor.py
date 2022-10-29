import abc
import yandex_music as ym
import numpy as np


class FeatureExtractor(abc.ABC):

    @abc.abstractmethod
    def get_feature(self, download_info: ym.DownloadInfo, waveform: np.ndarray) -> float:
        pass
