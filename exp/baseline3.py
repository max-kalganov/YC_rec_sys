import time
from collections import Counter, defaultdict
from tqdm import tqdm
from typing import Set, List, Dict
from heapq import nlargest

from exp.improved_baseline import aggregate_track_stats, get_popular_tracks, write_results


def store_all_users_tracks():
    users_tracks = []
    with open('data/likes_data/train') as f:
        lines = f.readlines()
        for line in lines:
            users_tracks.append(list(line.strip().split(' ')))
    return users_tracks


def store_test_users_tracks():
    users_tracks = []
    with open('data/likes_data/test') as f:
        lines = f.readlines()
        for line in lines:
            users_tracks.append(list(line.strip().split(' ')))
    return users_tracks


def get_tracks_to_users_map(all_users_tracks: List[Set[str]]) -> Dict[str, Set[str]]:
    tracks_to_users = defaultdict(set)
    for i, user_tracks in enumerate(all_users_tracks):
        for track in user_tracks:
            tracks_to_users[track].add(i)
    return tracks_to_users


def get_likes_tree(users_tracks: List[List[str]], tree: Dict[str, Set[str]]) -> None:
    for single_user_tracks in tqdm(users_tracks, desc="Get likes tree making..."):
        previous_track = single_user_tracks[0]
        for track in single_user_tracks[1:]:
            tree[previous_track].add(track)
            previous_track = track


def get_predictions(train_users_tracks, test_users_tracks, popularity_tracks):
    tracks_likes_tree = defaultdict(set)
    get_likes_tree(train_users_tracks, tracks_likes_tree)
    get_likes_tree(test_users_tracks, tracks_likes_tree)

    sorted_tracks_likes_tree = {}
    for k, v in tqdm(tracks_likes_tree.items(), desc="Sorting..."):
        v = list(v)
        v.sort(key=lambda i: popularity_tracks.index(i))
        sorted_tracks_likes_tree[k] = v

    results = []
    for single_user_tracks in tqdm(test_users_tracks, desc="Making predictions"):
        user_results = []
        singe_user_tracks_set = set(single_user_tracks)
        for track in reversed(single_user_tracks):
            not_seen_tracks = []
            for track in sorted_tracks_likes_tree[track]:
                if track not in singe_user_tracks_set:
                    not_seen_tracks.append(track)

                    if len(user_results) + len(not_seen_tracks) >= 100:
                        break

            user_results.extend(not_seen_tracks)
            if len(user_results) >= 100:
                break

        results.append(' '.join(user_results) + '\n')
    return results


def run_baseline3():
    train_track_stats = aggregate_track_stats()

    train_popular_tracks = get_popular_tracks(train_track_stats, top100=False)

    train_users_tracks = store_all_users_tracks()
    test_users_tracks = store_test_users_tracks()
    # tracks_to_users_map = get_tracks_to_users_map(train_users_tracks)
    #
    # results = get_predictions(train_users_tracks, train_100_popular_tracks,
    #                           tracks_to_users_map, n_most_common_tracks=20, n_most_common_nearest_users=10)
    results = get_predictions(train_users_tracks, test_users_tracks, train_popular_tracks)
    write_results(results, "baseline3_results_sort_in_groups")


def crop_results():
    cropped_results = []
    with open('data/results/baseline3_results') as f:
        lines = f.readlines()
        for line in lines:
            cropped_line = ' '.join(line.strip().split(' ')[:100]) + '\n'
            cropped_results.append(cropped_line)
    write_results(cropped_results, "baseline3_results")


if __name__ == '__main__':
    run_baseline3()
    # crop_results()