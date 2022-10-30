import time
from collections import Counter
from tqdm import tqdm
from typing import Set, List

from exp.improved_baseline import aggregate_track_stats, get_popular_tracks, write_results


def store_all_users_tracks():
    users_tracks = []
    with open('../data/likes_data/train') as f:
        lines = f.readlines()
        for line in lines:
            users_tracks.append(set(line.strip().split(' ')))
    return users_tracks


def get_predictions(train_users_tracks: List[Set[str]], train_100_popular_tracks,
                    selected_lines_from: int, selected_lines_to: int, n_most_common: int = 20):
    results = []
    with open('../data/likes_data/test') as f:
        lines = f.readlines()
        t21f = 0
        t32f = 0
        t43f = 0
        t54f = 0
        t65f = 0

        for line in tqdm(lines[selected_lines_from: selected_lines_to]):
            t1 = time.time()
            current_test_user_tracks = set(line.strip().split(' '))
            t2 = time.time()


            top3_nearest_train_users = [0, 1, 2]
            max_intersec = [0, 0, 0]
            maxminval = 0
            maxminpos = 2
            for track_ind, tracks in enumerate(train_users_tracks[3:]):
                intersec_len = len(current_test_user_tracks.intersection(tracks))
                if intersec_len > maxminval:
                    max_intersec.pop(maxminpos)
                    top3_nearest_train_users.pop(maxminpos)

                    max_intersec.append(intersec_len)
                    top3_nearest_train_users.append(track_ind + 3)

                    maxminval = min(max_intersec)
                    maxminpos = max_intersec.index(maxminval)

            top3_nearest_train_users = [train_users_tracks[t_ind] for t_ind in top3_nearest_train_users]
            t3 = time.time()
            new_top3_users_tracks = Counter()
            for nearest_user in top3_nearest_train_users:
                new_top3_users_tracks.update(nearest_user - current_test_user_tracks)

            t4 = time.time()
            most_common_tracks = new_top3_users_tracks.most_common(n_most_common)

            current_user_result_tracks = [track for track, _ in most_common_tracks]
            t5 = time.time()
            most_common_tracks_set = set(current_user_result_tracks)

            for pop_track in train_100_popular_tracks:
                if pop_track not in most_common_tracks_set and len(current_user_result_tracks) < 100:
                    current_user_result_tracks.append(pop_track)
                else:
                    break
            t6 = time.time()
            t21f += t2 - t1
            t32f += t3 - t2
            t43f += t4 - t3
            t54f += t5 - t4
            t65f += t6 - t5

            results.append(' '.join(current_user_result_tracks) + '\n')
    print(f"Time: {t21f}, {t32f}, {t43f}, {t54f}, {t65f}")

    return results


def run_baseline2():
    t1 = time.time()
    train_track_stats = aggregate_track_stats()
    train_100_popular_tracks = get_popular_tracks(train_track_stats)
    t2 = time.time()
    print("Time: ", t2 - t1)
    train_users_tracks = store_all_users_tracks()
    t3 = time.time()
    print("Time: ", t3 - t2)
    results = get_predictions(train_users_tracks, train_100_popular_tracks,
                              selected_lines_from=0, selected_lines_to=100)
    write_results(results, "improved_baseline_result_0_100")
    t4 = time.time()
    print(t4 - t3)


if __name__ == '__main__':
    run_baseline2()
