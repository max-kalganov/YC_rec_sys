import os.path
import pickle
import random
from collections import Counter, defaultdict
from functools import wraps
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from exp.improved_baseline import write_results
from preprocessing.utils import load_file_tracks, get_tracks_to_artists
from exp import constants as ct


def dumper(data, file):
    # open a file, where you ant to store the data
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def loader(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def cacher(file_name: str):
    """Caches func results"""

    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            full_file_name = f"data/cache/{file_name}"
            if os.path.exists(full_file_name):
                print(f"Loading from {full_file_name}")
                data = loader(full_file_name)
            else:
                os.makedirs('data/cache', exist_ok=True)
                print(f"Dumping into {full_file_name}")
                data = fn(*args, **kwargs)
                dumper(data, full_file_name)
            return data

        return wrapper

    return decorate


def calc_prediction():
    @cacher("_get_pop_tracks")
    def _get_pop_tracks(train_users_tracks, test_users_tracks):
        """returns pop tracks"""
        pop_tracks = Counter()
        all_users_tracks = train_users_tracks + test_users_tracks
        for user_tracks in tqdm(all_users_tracks, desc="Running _get_pop_tracks"):
            pop_tracks.update(user_tracks)
        return pop_tracks.most_common()

    @cacher("_get_artists_to_tracks")
    def _get_artists_to_tracks(tracks_to_artists):
        artists_to_tracks = defaultdict(list)
        for track, artist in tqdm(tracks_to_artists.items(), desc="Running _get_artists_to_tracks"):
            artists_to_tracks[artist].append(track)
        return dict(artists_to_tracks)

    @cacher("_get_pop_artists")
    def _get_pop_artists(train_users_tracks, test_users_tracks, tracks_to_artists):
        """returns dict artist -> 10 most_common artists | by train + test"""
        pop_artists = {}
        all_users_tracks = train_users_tracks + test_users_tracks
        for user_tracks in tqdm(all_users_tracks, desc="Running _get_pop_artists"):
            user_artists = [tracks_to_artists[track] for track in user_tracks]
            for artist in user_artists:
                pop_artists.setdefault(artist, Counter())
                pop_artists[artist].update(user_artists)

        pop_artists = {artist: nearest_artists.most_common(20) for artist, nearest_artists in pop_artists.items()}
        return pop_artists

    @cacher("_get_user_pop_artists")
    def _get_user_pop_artists(test_users_tracks,
                              tracks_to_artists,
                              artists_to_nearest_artists: Dict[str, List[Tuple[str, int]]]):
        """returns list of lists: each list contains artists [10 most common unseen] + users artists | by test"""
        users_pop_artists = []

        for users_tracks in tqdm(test_users_tracks, desc="Running _get_user_pop_artists"):
            users_artists = [tracks_to_artists[track] for track in users_tracks]
            users_artists_set = set(users_artists)
            users_unseen_artists = Counter()
            for artist in users_artists:
                unseen_nearest_artists = {nearest_most_common_artist: artist_intersection
                                          for nearest_most_common_artist, artist_intersection in
                                          artists_to_nearest_artists[artist]
                                          if nearest_most_common_artist not in users_artists_set}
                users_unseen_artists.update(unseen_nearest_artists)
            top10_unseen_artists = [artist for artist, _ in users_unseen_artists.most_common(10)]
            users_pop_artists.append(users_artists + top10_unseen_artists)

        return users_pop_artists

    @cacher("_get_user_artists_songs")
    def _get_user_artists_songs(users_artists, artists_to_tracks):
        """returns list of lists: each list contains songs from artists | by test"""
        users_prediction_songs = []
        for user_artists in tqdm(users_artists, desc="Running _get_user_artists_songs"):
            user_songs = list(chain(*[artists_to_tracks[artist] for artist in user_artists]))
            users_prediction_songs.append(user_songs)
        return users_prediction_songs

    @cacher("_get_user_songs_popularity")
    def _get_user_songs_popularity(users_songs, popular_songs: List[Tuple[str, int]]):
        """returns list of tuples: each tuple contains (sum of songs popularity, num of songs)"""
        users_songs_popularity = []
        songs_to_popularity = dict(popular_songs)

        for user_songs in tqdm(users_songs, desc="Running _get_user_songs_popularity"):
            popularity_values = [songs_to_popularity.get(song, 0) for song in user_songs]
            users_songs_popularity.append((sum(popularity_values), len(popularity_values)))
        return users_songs_popularity

    @cacher("_get_all_users_mean_popularity_train")
    def _get_all_users_mean_popularity(train_users_tracks, test_users_tracks, popularity_tracks):
        """returns mean popularity over all dataset"""
        all_users_tracks = train_users_tracks  # + test_users_tracks
        all_users_mean_tracks_popularity = []
        pop_tracks = dict(popularity_tracks)
        for users_tracks in tqdm(all_users_tracks, desc="Running _get_all_users_mean_popularity"):
            user_tracks_popularity = [pop_tracks[track] for track in users_tracks]
            all_users_mean_tracks_popularity.append(sum(user_tracks_popularity) / len(user_tracks_popularity))

        return np.mean(all_users_mean_tracks_popularity), np.quantile(all_users_mean_tracks_popularity, 0.25)

    @cacher("_get_user_songs_predictions")
    def _get_user_songs_predictions():
        """returns songs selected by popularity"""

    @cacher("_sort_user_predictions")
    def _sort_user_predictions(users_predictions,
                               mean_popularity,
                               users_tracks_popularity,
                               tracks_to_popularity_most_common):
        """returns predictions sorted by songs popularity to get average popularity """

        def _get_centered_popularity(current_track_pop, sum_pop, num_tracks):
            return abs((sum_pop + current_track_pop) / (num_tracks + 1) - mean_popularity)

        tracks_to_popularity = dict(tracks_to_popularity_most_common)
        results = []
        for user_id in tqdm(range(len(users_predictions)), desc="Running _sort_user_predictions"):
            user_preds = users_predictions[user_id]
            sum_tracks_poopularity, num_of_tracks = users_tracks_popularity[user_id]

            cur_user_sorted_predictions = sorted(user_preds,
                                                 key=lambda track_id: _get_centered_popularity(
                                                     current_track_pop=tracks_to_popularity[track_id],
                                                     sum_pop=sum_tracks_poopularity,
                                                     num_tracks=num_of_tracks
                                                 ))[:100]
            results.append(' '.join(cur_user_sorted_predictions) + '\n')
        return results

    train_data = load_file_tracks(ct.KEY_TO_FILENAME[ct.TRAIN_KEY])
    test_data = load_file_tracks(ct.KEY_TO_FILENAME[ct.TEST_KEY])
    tracks_to_artists = get_tracks_to_artists(ct.KEY_TO_FILENAME[ct.TRACKS_TO_ARTISTS_KEY])

    artists_to_nearest_artists = _get_pop_artists(train_data, test_data, tracks_to_artists)
    test_users_pop_artists = _get_user_pop_artists(test_data, tracks_to_artists, artists_to_nearest_artists)

    artists_to_tracks = _get_artists_to_tracks(tracks_to_artists)
    test_users_artists_songs = _get_user_artists_songs(test_users_pop_artists, artists_to_tracks)

    pop_tracks_most_common = _get_pop_tracks(train_data, test_data)
    users_songs_popularity = _get_user_songs_popularity(test_users_artists_songs, pop_tracks_most_common)

    all_users_mean_popularity, q1_popularity = _get_all_users_mean_popularity(train_data, test_data,
                                                                              pop_tracks_most_common)

    results = _sort_user_predictions(test_users_artists_songs,
                                     all_users_mean_popularity,
                                     users_songs_popularity,
                                     pop_tracks_most_common)

    #
    # results = []
    # pop_tracks_dict = dict(pop_tracks_most_common)
    # for users_tracks in tqdm(test_users_artists_songs, desc="Making predictions"):
    #     user_tracks_predictions = sorted(users_tracks, reverse=True, key=lambda x: pop_tracks_dict[x])[:100]
    #     user_tracks_predictions = list(reversed(user_tracks_predictions))
    #
    #     results.append(' '.join(user_tracks_predictions) + '\n')

    write_results(result=results, filename="baseline4_sorted_by_mean_popularity")


if __name__ == '__main__':
    calc_prediction()

#
# def backup():
#     all_users_mean_coverage = []
#     num_of_artists_to_check = 5
#
#     tqdm_obj = tqdm(range(len(test_data)), desc="Running test")
#     for test_user_id in tqdm_obj:
#         user_tracks = test_data[test_user_id]
#         user_coverage = 0
#         for i, track in enumerate(user_tracks):
#             cur_user_tracks = user_tracks[:i] + user_tracks[i + 1:]
#             user_artists = set([tracks_to_artists[track] for track in cur_user_tracks])
#             track_artist = tracks_to_artists[track]
#
#             if track_artist not in user_artists:
#                 found = False
#                 for artist in user_artists:
#                     for n_artist, intersec in pop_artists_groups[artist]:
#                         if track_artist == n_artist:
#                             user_coverage += 1
#                             found = True
#                             break
#                     if found is True:
#                         break
#
#             else:
#                 user_coverage += 1
#
#         mean_user_coverage = user_coverage / len(user_tracks)
#         all_users_mean_coverage.append(mean_user_coverage)
#         if test_user_id % 1000 == 0:
#             tqdm_obj.set_description(
#                 f"{test_user_id} - current all_users_mean_coverage = {sum(all_users_mean_coverage) / len(all_users_mean_coverage)}")
#     px.box(x=all_users_mean_coverage).show()
