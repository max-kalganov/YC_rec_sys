import os
from typing import Tuple, List, Dict

import gin


def get_track_file_path(track_id: str, track_root_folder: str, user_ind: int, postfix: str = None):
    user_ind_postfix = f"user_{user_ind}"
    postfix = f"_{postfix}" if postfix else ""
    file_name = f"{track_id}{postfix}_{user_ind_postfix}.wav"
    return os.path.join(track_root_folder, file_name)


def load_file_tracks(input_file_path: str) -> List[List[str]]:
    with open(input_file_path) as f:
        lines = f.readlines()
        all_tracks = [user_tracks_str.strip().split(' ') for user_tracks_str in lines]
    return all_tracks


@gin.configurable
def get_tracks_to_artists(input_file_path: str) -> Dict[str, str]:
    with open(input_file_path) as f:
        lines = f.readlines()
        tracks_to_artists = {}
        for track_with_artist in lines[1:]:
            track, artist = track_with_artist.split(',')
            tracks_to_artists[track] = artist
        return tracks_to_artists


def str_to_int_dict(d: Dict[str, str]) -> Dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def str_to_int_list(outer_list: List[List[str]]) -> List[List[int]]:
    return [[int(inner_list_value) for inner_list_value in inner_list] for inner_list in outer_list]


@gin.configurable
def get_train_val_tracks_ids(input_file_path: str,
                             train_val_split: float = 0.7) -> Tuple[List[List[str]], List[List[str]]]:
    """Returns track ids for train and val"""
    all_tracks = load_file_tracks(input_file_path)
    num_of_train_users = int(len(all_tracks) * train_val_split)
    train_tracks = all_tracks[:num_of_train_users]
    val_tracks = all_tracks[num_of_train_users:]
    return train_tracks, val_tracks


@gin.configurable
def get_test_tracks_ids(input_file_path: str) -> List[List[str]]:
    return load_file_tracks(input_file_path)
