import random
from collections import defaultdict
from typing import List, Iterable, Tuple, Dict

import numpy as np
from tqdm import tqdm


class DatasetGenerator:
    # TODO: add triples per anchor parameter --- self.triples_per_anchor = triples_per_anchor

    def generate_user_triples(
            self,
            all_users_tracks: List[List[str]],
            batch_size: int
    ) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        pass

    @staticmethod
    def _map_artists_to_tracks(tracks_to_artists: Dict[str, str]) -> Dict[str, List[str]]:
        artists_to_tracks = defaultdict(list)
        for track, artist in tracks_to_artists.items():
            artists_to_tracks[artist].append(track)
        return dict(artists_to_tracks)  # to remove default behaviour

    @staticmethod
    def _select_not_equal_value(value, seq):
        selected_value = value
        while selected_value == value:
            selected_value = random.choice(seq)
        return selected_value

    def _get_artist_positive(self,
                             track_id: str,
                             artist_id: str,
                             artists_to_tracks: Dict[str, List[str]]) -> str:
        current_positives = artists_to_tracks[artist_id]
        assert len(current_positives) > 1
        return self._select_not_equal_value(track_id, current_positives)

    def _get_artist_negative(self,
                             artist_id: str,
                             artists_to_tracks: Dict[str, List[str]]) -> str:
        current_negative_artist = self._select_not_equal_value(artist_id, artists_to_tracks.keys())

        current_negative = artists_to_tracks[current_negative_artist]
        assert len(current_negative) > 1
        return random.choice(current_negative)

    def generate_artist_triples(
            self,
            tracks_to_artist: Dict[str, str],
            batch_size: int
    ) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        artists_to_tracks = self._map_artists_to_tracks(tracks_to_artist)
        all_tracks = list(tracks_to_artist.keys())
        random.shuffle(all_tracks)

        batch_anchors, batch_positives, batch_negatives = [], [], []
        for anchor_track_id in tqdm(all_tracks, desc="Generating artist triples: "):
            batch_anchors.append(int(anchor_track_id))
            batch_positives.append(int(self._get_artist_positive(track_id=anchor_track_id,
                                                                 artist_id=tracks_to_artist[anchor_track_id],
                                                                 artists_to_tracks=artists_to_tracks)))
            batch_negatives.append(int(self._get_artist_negative(artist_id=tracks_to_artist[anchor_track_id],
                                                                 artists_to_tracks=artists_to_tracks)))
            if len(batch_anchors) == batch_size:
                yield np.array(batch_anchors), np.array(batch_positives), np.array(batch_negatives)
                batch_anchors, batch_positives, batch_negatives = [], [], []

        yield np.array(batch_anchors), np.array(batch_positives), np.array(batch_negatives)