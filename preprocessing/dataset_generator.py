import abc
import random
from collections import defaultdict
from copy import copy
from itertools import cycle, chain
from typing import List, Dict, Union, Optional

import gin
import numpy as np


class BaseDatasetGenerator(abc.ABC):
    def __init__(self, all_tracks: List[int], embedding_size: int):
        self._all_tracks = copy(all_tracks)
        random.shuffle(self._all_tracks)
        self._all_tracks_iterator = cycle(self._all_tracks)

        self._embedding_size = embedding_size

    @staticmethod
    def _select_not_equal_value(value, seq):
        selected_value = value
        while selected_value == value:
            selected_value = random.choice(seq)
        return selected_value

    def _get_positive(self,
                      track_id: int,
                      group_id: Union[int, List[int]],
                      group_id_to_tracks: Dict[int, List[int]]) -> Optional[int]:
        current_positives = group_id_to_tracks[group_id] if isinstance(group_id, int) \
            else list(set(chain(*[group_id_to_tracks[single_group_id] for single_group_id in group_id])))
        selected_positive = None
        if len(current_positives) > 1:
            selected_positive = self._select_not_equal_value(track_id, current_positives)
        return selected_positive

    def _get_negative(self,
                      group_id: Union[int, List[int]],
                      group_id_to_tracks: Dict[int, List[int]]) -> int:
        if isinstance(group_id, int):
            current_negative_group_id = self._select_not_equal_value(group_id, list(group_id_to_tracks.keys()))
        else:
            negative_group_ids = set(group_id_to_tracks.keys()) - set(group_id)
            current_negative_group_id = random.choice(list(negative_group_ids))
        current_negative = group_id_to_tracks[current_negative_group_id]
        assert len(current_negative) > 0
        return random.choice(current_negative)

    def _generate_single_triple(self, anchor_track_id, tracks_to_group_id, group_id_to_tracks):
        current_anchor = anchor_track_id
        current_positive = self._get_positive(track_id=anchor_track_id,
                                              group_id=tracks_to_group_id[anchor_track_id],
                                              group_id_to_tracks=group_id_to_tracks)
        current_negative = self._get_negative(group_id=tracks_to_group_id[anchor_track_id],
                                              group_id_to_tracks=group_id_to_tracks)
        if current_positive is None:
            current_anchor = current_negative = None
        else:
            current_anchor = int(current_anchor)
            current_positive = int(current_positive)
            current_negative = int(current_negative)

        return current_anchor, current_positive, current_negative

    def _get_label(self):
        return np.zeros((1, 3 * self._embedding_size))

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self):
        pass


@gin.configurable
class ArtistTriplesDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, tracks_to_artist: Dict[int, int], embedding_size):
        super().__init__(all_tracks=list(tracks_to_artist.keys()), embedding_size=embedding_size)
        self._tracks_to_artist = tracks_to_artist
        self._artists_to_tracks = self._map_artists_to_tracks(tracks_to_artist)

    @staticmethod
    def _map_artists_to_tracks(tracks_to_artists: Dict[int, int]) -> Dict[int, List[int]]:
        artists_to_tracks = defaultdict(list)
        for track, artist in tracks_to_artists.items():
            artists_to_tracks[artist].append(track)
        return dict(artists_to_tracks)  # to remove default behaviour

    def __next__(self):
        current_anchor, current_positive, current_negative = None, None, None
        while current_anchor is None:
            current_anchor = next(self._all_tracks_iterator)
            current_anchor, current_positive, current_negative = self._generate_single_triple(
                anchor_track_id=current_anchor,
                tracks_to_group_id=self._tracks_to_artist,
                group_id_to_tracks=self._artists_to_tracks
            )

        anchor_np = np.array([current_anchor])
        positive_np = np.array([current_positive])
        negative_np = np.array([current_negative])

        return [anchor_np, positive_np, negative_np], self._get_label()


@gin.configurable
class UsersTriplesDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, all_users_tracks: List[List[int]], embedding_size):
        self._user_to_tracks = self._map_users_to_tracks(all_users_tracks)
        self._tracks_to_users = self._map_tracks_to_users(self._user_to_tracks)

        super().__init__(all_tracks=list(self._tracks_to_users.keys()), embedding_size=embedding_size)

    @staticmethod
    def _map_users_to_tracks(all_users_tracks: List[List[int]]) -> Dict[int, List[int]]:
        return dict(zip(range(len(all_users_tracks)), all_users_tracks))

    @staticmethod
    def _map_tracks_to_users(users_to_tracks: Dict[int, List[int]]) -> Dict[int, List[int]]:
        tracks_to_users = defaultdict(list)
        for user, tracks in users_to_tracks.items():
            for track_id in tracks:
                tracks_to_users[track_id].append(user)
        return dict(tracks_to_users)

    def __next__(self):
        current_anchor, current_positive, current_negative = None, None, None
        while current_anchor is None:
            current_anchor = next(self._all_tracks_iterator)
            current_anchor, current_positive, current_negative = self._generate_single_triple(
                anchor_track_id=current_anchor,
                tracks_to_group_id=self._tracks_to_users,
                group_id_to_tracks=self._user_to_tracks
            )

        anchor_np = np.array([current_anchor])
        positive_np = np.array([current_positive])
        negative_np = np.array([current_negative])

        return [anchor_np, positive_np, negative_np], self._get_label()
