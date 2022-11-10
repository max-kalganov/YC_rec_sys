def aggregate_track_stats(file: str = 'data/likes_data/train'):
    track_stats = {}

    print(f"process {file} data")
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            tracks = line.strip().split(' ')
            for track in tracks:
                if track not in track_stats:
                    track_stats[track] = 0
                track_stats[track] += 1
    return track_stats


def download_test_data():
    print("process test data")
    with open('data/likes_data/test') as f:
        test = f.readlines()
    return test


def get_popular_tracks(track_stats, top100: bool = True):
    popular_tracks = sorted(track_stats.items(), key=lambda item: item[1], reverse=True)
    if top100:
        popular_tracks = popular_tracks[:100]
    popular_tracks_list = [x[0] for x in popular_tracks]
    return popular_tracks_list


def get_top_tracks(track_stats):
    top_tracks = sorted(track_stats.items(), key=lambda item: item[1], reverse=True)[:1000]
    top_tracks_set = set([x[0] for x in top_tracks])
    return top_tracks_set


def get_global_track_score(track_stats):
    print("calc score")
    top_tracks = sorted(track_stats.items(), key=lambda item: item[1], reverse=True)[:1000]
    global_track_score = {}
    for track in top_tracks:
        global_track_score[track[0]] = track_stats[track[0]] ** 0.5
    return global_track_score


def get_filtered_train_track_count(top_tracks_set):
    print("calc score continue")
    track_count = {}
    with open('data/likes_data/train') as f:
        lines = f.readlines()
        for (i, line) in enumerate(lines):
            tracks = line.strip().split(' ')
            filtered_tracks = []
            for track in tracks:
                if track in top_tracks_set:
                    filtered_tracks.append(track)
            for i in range(len(filtered_tracks)):
                track1 = filtered_tracks[i]
                for j in range(len(filtered_tracks)):
                    if i != j:
                        track2 = filtered_tracks[j]
                        if track1 not in track_count:
                            track_count[track1] = {}
                        current_count = track_count[track1]
                        if track2 not in current_count:
                            current_count[track2] = 0
                        current_count[track2] += 1
    return track_count


def get_test_results(test, track_count, global_track_score, popular_tracks_list):
    print("calc test continue")
    result = []
    empty_track_score = 0
    for query in test:
        test_tracks = query.strip().split(' ')
        track_score = {}
        for track in test_tracks:
            if track in track_count:
                for track_id in track_count[track]:
                    score = track_count[track][track_id]
                    if track_id not in track_score:
                        track_score[track_id] = 0
                    track_score[track_id] += score / global_track_score[track] / global_track_score[track_id]
        if len(track_score) == 0:
            result.append(' '.join(popular_tracks_list) + '\n')
            empty_track_score += 1
        else:
            best_tracks = sorted(track_score.items(), key=lambda item: item[1], reverse=True)[:100]
            result.append(' '.join([x[0] for x in best_tracks]) + '\n')
    return result


def write_results(result, filename: str):
    print("dump results")
    with open(f'data/results/{filename}', 'w') as f:
        f.writelines(result)


if __name__ == '__main__':
    train_track_stats = aggregate_track_stats()

    train_popular_tracks_list = get_popular_tracks(train_track_stats)
    top_tracks_set = get_top_tracks(train_track_stats)
    global_track_score = get_global_track_score(train_track_stats)

    track_count = get_filtered_train_track_count(top_tracks_set)
    test = download_test_data()
    result = get_test_results(test, track_count, global_track_score, train_popular_tracks_list)
    write_results(result)
