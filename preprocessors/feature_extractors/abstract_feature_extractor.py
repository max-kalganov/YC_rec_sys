import abc


class FeatureExtractor(abc.ABC):
    @abc.abstractmethod
    def get_feature(self):
        pass
