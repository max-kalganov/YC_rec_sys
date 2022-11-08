from typing import List

from preprocessing.utils import load_file_tracks, str_to_int_list


def get_max_value(tracks: List[List[str]]) -> int:
    int_tracks = str_to_int_list(tracks)
    max_value = 0
    for user_tracks in int_tracks:
        user_max = max(user_tracks)
        max_value = max(user_max, max_value)
    return max_value


def calc_max_track_id():
    train_tracks = load_file_tracks('data/likes_data/train')
    test_tracks = load_file_tracks('data/likes_data/test')

    train_max = get_max_value(train_tracks)
    test_max = get_max_value(test_tracks)

    print(f"Train max: {train_max}, test max: {test_max}, total max: {max(train_max, test_max)}")


if __name__ == '__main__':
    calc_max_track_id()
