import random
from collections import defaultdict
from typing import List, Iterable, Tuple, Dict, Union, Optional

import numpy as np
from tqdm import tqdm


class DatasetGenerator:
    # TODO: add triples per anchor parameter --- self.triples_per_anchor = triples_per_anchor

    # UTILS

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
            else list(set(group_id_to_tracks[single_group_id] for single_group_id in group_id))
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

    def _generate_tracks_triples(self,
                                 group_id_to_tracks: Dict[int, List[int]],
                                 tracks_to_group_id: Dict[int, Union[int, List[int]]],
                                 batch_size: int,
                                 description: str):
        all_tracks = list(tracks_to_group_id.keys())
        random.shuffle(all_tracks)

        batch_anchors, batch_positives, batch_negatives = [], [], []
        for anchor_track_id in tqdm(all_tracks, desc=description):
            current_anchor = anchor_track_id
            current_positive = self._get_positive(track_id=anchor_track_id,
                                                  group_id=tracks_to_group_id[anchor_track_id],
                                                  group_id_to_tracks=group_id_to_tracks)
            current_negative = self._get_negative(group_id=tracks_to_group_id[anchor_track_id],
                                                  group_id_to_tracks=group_id_to_tracks)
            if current_positive is None:
                continue
            batch_anchors.append(int(current_anchor))
            batch_positives.append(int(current_positive))
            batch_negatives.append(int(current_negative))
            if len(batch_anchors) == batch_size:
                yield np.array(batch_anchors), np.array(batch_positives), np.array(batch_negatives)
                batch_anchors, batch_positives, batch_negatives = [], [], []

        yield np.array(batch_anchors), np.array(batch_positives), np.array(batch_negatives)

    # ARTISTS TRIPLES GENERATION

    @staticmethod
    def _map_artists_to_tracks(tracks_to_artists: Dict[int, int]) -> Dict[int, List[int]]:
        artists_to_tracks = defaultdict(list)
        for track, artist in tracks_to_artists.items():
            artists_to_tracks[artist].append(track)
        return dict(artists_to_tracks)  # to remove default behaviour

    def generate_artist_triples(
            self,
            tracks_to_artist: Dict[int, int],
            batch_size: int
    ) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        artists_to_tracks = self._map_artists_to_tracks(tracks_to_artist)
        yield from self._generate_tracks_triples(group_id_to_tracks=artists_to_tracks,
                                                 tracks_to_group_id=tracks_to_artist,
                                                 batch_size=batch_size,
                                                 description="Generating artist triples: ")

    # USERS TRIPLES GENERATION

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

    def generate_user_triples(
            self,
            all_users_tracks: List[List[int]],
            batch_size: int
    ) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        user_to_tracks = self._map_users_to_tracks(all_users_tracks)
        tracks_to_users = self._map_tracks_to_users(user_to_tracks)
        yield from self._generate_tracks_triples(group_id_to_tracks=user_to_tracks,
                                                 tracks_to_group_id=tracks_to_users,
                                                 batch_size=batch_size,
                                                 description="Generating users triples: ")
