import os
from collections import Counter
from typing import List, Optional, Dict

from tqdm import tqdm

from exp.baseline2 import get_tracks_to_users_map
from exp.baseline3 import store_all_users_tracks, store_test_users_tracks
import numpy as np
import plotly.express as pe

from exp.improved_baseline import get_popular_tracks, aggregate_track_stats
from preprocessing.utils import get_tracks_to_artists


def likes_len_stats(users_tracks: List[List[str]], desc: str = ""):
    users_likes_len = [len(single_user_tracks) for single_user_tracks in users_tracks]
    print(f"{desc}{max(users_likes_len)=}, {min(users_likes_len)=}, {np.mean(users_likes_len)=}")


def check_triples(users_tracks: List[List[str]], tracks_to_artists: Dict[str, str], top100_tracks: List[str]):
    top10_tracks = set(top100_tracks[:10])
    top100_tracks = set(top100_tracks)
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

    print(f"Triples mean len - {np.mean([len(v) for v in triples.values()])}")
    print(f"Triples by artist mean len - {np.mean([len(v) for v in triples_by_artists.values()])}")

    has_predictions_in_triples = 0
    has_pair_in_triples = 0

    has_predictions_in_triples_artist = 0
    has_pair_in_triples_artist = 0

    in_top100 = 0
    in_top10 = 0
    in_user_artists = 0
    in_last_user_artist = 0
    users_covered = 0
    covered = False
    for single_user_tracks in tqdm(users_tracks, desc="checking predictions"):
        covered = False
        if single_user_tracks[-1] in top10_tracks:
            covered = True
            in_top10 += 1

        if single_user_tracks[-1] in top100_tracks:
            covered = True
            in_top100 += 1

        user_artists = set([tracks_to_artists[track] for track in single_user_tracks[:-1]])

        if tracks_to_artists[single_user_tracks[-1]] in user_artists:
            covered = True
            in_user_artists += 1

        if tracks_to_artists[single_user_tracks[-1]] == tracks_to_artists[single_user_tracks[-2]]:
            covered = True
            in_last_user_artist += 1

        last_pair = (single_user_tracks[-3], single_user_tracks[-2])
        last_track = single_user_tracks[-1]
        if last_pair in triples:
            has_pair_in_triples += 1
            if last_track in triples[last_pair]:
                covered = True
                has_predictions_in_triples += 1

        last_pair_artists = (tracks_to_artists[single_user_tracks[-3]], tracks_to_artists[single_user_tracks[-2]])
        last_track_artist = tracks_to_artists[single_user_tracks[-1]]
        if last_pair_artists in triples_by_artists:
            has_pair_in_triples_artist += 1
            if last_track_artist in triples_by_artists[last_pair_artists]:
                covered = True
                has_predictions_in_triples_artist += 1
        if covered:
            users_covered += 1
    print(f"All number of users: {len(users_tracks)}, {users_covered=}. "
          f"-- {in_user_artists=}, {in_top100=}, {in_top10=}, {in_last_user_artist=}"
          f"\nPredictions are in triples for: {has_predictions_in_triples}, "
          f"triples stored for: {has_pair_in_triples}\n"
          f"\nPredictions are in triples by artists for: {has_predictions_in_triples_artist}, "
          f"triples by artists stored for: {has_pair_in_triples_artist}")


def calc_stat_for_nearest_user(train_users_tracks, track_to_artist, pop_100_tracks):
    # tracks_to_users_map = get_tracks_to_users_map(train_users_tracks)

    track_in_nearest_user = 0
    nearest_user_track_index = []

    with open("data/stats/nearest_users", 'r') as f:
        nearest_users = f.readlines()

    for single_user_tracks, cur_user_nearest_users in tqdm(zip(train_users_tracks[:100000], nearest_users)):

        top10_nearest_train_users = [user_info.split('_') for user_info in cur_user_nearest_users.strip().split(",")]

        single_user_tracks_set = set(single_user_tracks[:-1])
        new_top10_users_tracks = Counter()
        for nearest_user, intersec in top10_nearest_train_users[1:]:
            new_top10_users_tracks.update(set(train_users_tracks[int(nearest_user)]) - single_user_tracks_set)

        most_common_tracks = new_top10_users_tracks.most_common(20)

        current_user_result_tracks = [track for track, _ in most_common_tracks]
        most_common_tracks_set = set(current_user_result_tracks)

        for pop_track in pop_100_tracks:
            if pop_track not in most_common_tracks_set and len(current_user_result_tracks) < 100:
                current_user_result_tracks.append(pop_track)
            else:
                break

        if single_user_tracks[-1] in current_user_result_tracks:
            track_in_nearest_user += 1
            nearest_user_track_index.append(current_user_result_tracks.index(single_user_tracks[-1]))

        #
        # for i, (user_id, intersection) in enumerate(top10_nearest_train_users):
        #     if i == 0:
        #         continue
        #     nearest_users_tracks = train_users_tracks[int(user_id)]
        #     if single_user_tracks[-1] in nearest_users_tracks:
        #         track_in_nearest_user[i] += 1
        #         nearest_user_track_index.append(nearest_users_tracks.index(single_user_tracks[-1]))
        #         break
        #     else:
        #         n_user_artists = [track_to_artist[t] for t in nearest_users_tracks]
        #         last_track_artist = track_to_artist[single_user_tracks[-1]]
        #         if last_track_artist in n_user_artists:
        #             not_in_nearest_user_but_in_his_artists[i] += 1
        #             artist_track_index.append(n_user_artists.index(last_track_artist))
        #             break

    print(f"all users: {len(train_users_tracks)}, \n"
          f"{track_in_nearest_user=}, \n"
          f"{np.mean(nearest_user_track_index)=}")# , \n"
          # f"{not_in_nearest_user_but_in_his_artists=}, \n"
          # f"{np.mean(artist_track_index)=}")


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


def mean_number_of_songs():
    pass

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

    train_popularity_tracks = get_popular_tracks(aggregate_track_stats('data/likes_data/train'), top100=True)
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
    # check_triples(train_users_tracks, tracks_to_artists,
    #               top100_tracks=get_popular_tracks(aggregate_track_stats('data/likes_data/train'), top100=True))

    calc_stat_for_nearest_user(train_users_tracks, tracks_to_artists, train_popularity_tracks)


if __name__ == '__main__':
    run_stats()
