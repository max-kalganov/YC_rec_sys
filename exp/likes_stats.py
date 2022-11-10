import os
from collections import Counter
from typing import List, Optional, Dict

from tqdm import tqdm

from exp.baseline3 import store_all_users_tracks, store_test_users_tracks
import numpy as np
import plotly.express as pe

from exp.improved_baseline import get_popular_tracks, aggregate_track_stats
from preprocessing.utils import get_tracks_to_artists


def likes_len_stats(users_tracks: List[List[str]], desc: str = ""):
    users_likes_len = [len(single_user_tracks) for single_user_tracks in users_tracks]
    print(f"{desc}{max(users_likes_len)=}, {min(users_likes_len)=}, {np.mean(users_likes_len)=}")


def check_triples(users_tracks: List[List[str]], tracks_to_artists: Dict[str, str]):
    triples = {}
    triples_by_artists = {}
    for single_user_tracks in tqdm(users_tracks, desc="Storing triples"):
        for i in range(1, len(single_user_tracks)-2):
            # triples.setdefault((single_user_tracks[i - 1], single_user_tracks[i]), Counter())
            triples.setdefault((single_user_tracks[i - 1], single_user_tracks[i]), list())
            triples[(single_user_tracks[i - 1], single_user_tracks[i])].append(single_user_tracks[i+1])

            triples_by_artists.setdefault((tracks_to_artists[single_user_tracks[i - 1]],
                                           tracks_to_artists[single_user_tracks[i]]), list())
            triples_by_artists[(tracks_to_artists[single_user_tracks[i - 1]],
                                tracks_to_artists[single_user_tracks[i]])].append(
                tracks_to_artists[single_user_tracks[i + 1]])

    has_predictions_in_triples = 0
    has_pair_in_triples = 0

    has_predictions_in_triples_artist = 0
    has_pair_in_triples_artist = 0

    for single_user_tracks in tqdm(users_tracks, desc="checking predictions"):
        last_pair = (single_user_tracks[-3], single_user_tracks[-2])
        last_track = single_user_tracks[-1]
        if last_pair in triples:
            has_pair_in_triples += 1
            if last_track in triples[last_pair]:
                has_predictions_in_triples += 1

        last_pair_artists = (tracks_to_artists[single_user_tracks[-3]], tracks_to_artists[single_user_tracks[-2]])
        last_track_artist = tracks_to_artists[single_user_tracks[-1]]
        if last_pair_artists in triples_by_artists:
            has_pair_in_triples_artist += 1
            if last_track_artist in triples_by_artists[last_pair_artists]:
                has_predictions_in_triples_artist += 1

    print(f"All number of users: {len(users_tracks)}. "
          f"\nPredictions are in triples for: {has_predictions_in_triples}, "
          f"triples stored for: {has_pair_in_triples}\n"
          f"\nPredictions are in triples by artists for: {has_predictions_in_triples_artist}, "
          f"triples by artists stored for: {has_pair_in_triples_artist}")


def tracks_pre_post_intersections(users_tracks: List[List[str]], window_size: Optional[int], all_tracks: List[str], desc: str = ""):
    all_counters = {track: (Counter(), Counter()) for track in all_tracks}
    for single_user_tracks in tqdm(users_tracks):
        for i, track in enumerate(single_user_tracks):
            track_pre_counter, track_post_counter = all_counters[track]

            start_pre_index = max(i-window_size, 0) if window_size is not None else 0
            end_post_index = min(i + window_size + 1, len(single_user_tracks)) if window_size is not None \
                else len(single_user_tracks)
            pre_tracks = single_user_tracks[start_pre_index:i]
            post_tracks = single_user_tracks[i+1:end_post_index]

            track_pre_counter.update(pre_tracks)
            track_post_counter.update(post_tracks)

    print()


def prediction_by_popularity(users_tracks, tracks_to_popularity, html_path: str):
    mean_users_pop_position = []
    last_track_pop_position = []

    for single_user_tracks in tqdm(users_tracks):
        pop_positions = [tracks_to_popularity[track] for track in single_user_tracks[:-1]]
        last_track_position = tracks_to_popularity[single_user_tracks[-1]]

        mean_users_pop_position.append(np.mean(pop_positions))
        last_track_pop_position.append(last_track_position)

    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    pe.scatter(x=mean_users_pop_position, y=last_track_pop_position, opacity=0.1).write_html(html_path)


def run_stats():
    tracks_to_artists = get_tracks_to_artists(input_file_path="data/likes_data/track_artists.csv")

    train_users_tracks = store_all_users_tracks()
    test_users_tracks = store_test_users_tracks()

    # train_popularity_tracks = get_popular_tracks(aggregate_track_stats('data/likes_data/train'), top100=False)
    # test_popularity_tracks = get_popular_tracks(aggregate_track_stats('data/likes_data/test'), top100=False)

    # Calculating num of likes per person
    # likes_len_stats(train_users_tracks, "Train users likes len: ")
    # likes_len_stats(test_users_tracks, "Test users likes len: ")

    # Check intersections
    # tracks_pre_post_intersections(train_users_tracks, 1, list(tracks_to_artists.keys()))

    # Check popularity
    # train_tracks_to_popularity_map = {track: i for i, track in enumerate(train_popularity_tracks)}
    # test_tracks_to_popularity_map = {track: i for i, track in enumerate(test_popularity_tracks)}
    #
    # prediction_by_popularity(train_users_tracks, train_tracks_to_popularity_map,
    #                          html_path="data/stats/train_pop_predictions.html")
    # prediction_by_popularity(test_users_tracks, test_tracks_to_popularity_map,
    #                          html_path="data/stats/test_pop_predictions.html")

    # Check triples
    check_triples(train_users_tracks, tracks_to_artists)


if __name__ == '__main__':
    run_stats()
