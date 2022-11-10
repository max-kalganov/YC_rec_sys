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
            users_tracks.append(set(line.strip().split(' ')))
    return users_tracks


def get_tracks_to_users_map(all_users_tracks: List[Set[str]]) -> Dict[str, Set[str]]:
    tracks_to_users = defaultdict(set)
    for i, user_tracks in enumerate(all_users_tracks):
        for track in user_tracks:
            tracks_to_users[track].add(i)
    return tracks_to_users


def get_predictions(train_users_tracks: List[Set[str]], train_100_popular_tracks,
                    tracks_to_users_map: Dict[str, Set[str]], n_most_common_tracks: int = 20,
                    n_most_common_nearest_users: int = 3):
    results = []
    with open('data/likes_data/test') as f:
        lines = f.readlines()

        for line in tqdm(lines):
            current_test_user_tracks = set(line.strip().split(' '))

            nearest_train_users = Counter()
            for track in current_test_user_tracks:
                nearest_train_users.update(tracks_to_users_map[track])

            top3_nearest_train_users = [user_id for user_id, _ in nearest_train_users.most_common(n_most_common_nearest_users)]

            new_top3_users_tracks = Counter()
            for nearest_user in top3_nearest_train_users:
                new_top3_users_tracks.update(train_users_tracks[nearest_user] - current_test_user_tracks)

            most_common_tracks = new_top3_users_tracks.most_common(n_most_common_tracks)

            current_user_result_tracks = [track for track, _ in most_common_tracks]
            most_common_tracks_set = set(current_user_result_tracks)

            for pop_track in train_100_popular_tracks:
                if pop_track not in most_common_tracks_set and len(current_user_result_tracks) < 100:
                    current_user_result_tracks.append(pop_track)
                else:
                    break

            results.append(' '.join(current_user_result_tracks) + '\n')

    return results


def run_baseline2():
    t1 = time.time()
    train_track_stats = aggregate_track_stats()
    train_100_popular_tracks = get_popular_tracks(train_track_stats)
    t2 = time.time()
    print("Time: ", t2 - t1)
    train_users_tracks = store_all_users_tracks()
    tracks_to_users_map = get_tracks_to_users_map(train_users_tracks)
    t3 = time.time()
    print("Time: ", t3 - t2)
    results = get_predictions(train_users_tracks, train_100_popular_tracks,
                              tracks_to_users_map, n_most_common_tracks=20, n_most_common_nearest_users=10)
    write_results(results, "improved_baseline_result_20_most_common_10_nearest_users")

    t4 = time.time()
    print(t4 - t3)


if __name__ == '__main__':
    run_baseline2()
