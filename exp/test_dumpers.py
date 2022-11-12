import logging

import exp.dump_and_load_main_stats as storage
import exp.constants as ct

logging.getLogger().setLevel(logging.DEBUG)

if __name__ == '__main__':
    # tracks_to_users, all_users_tracks = storage.TracksToUsersStorage(mapped_users_name=[ct.TRAIN_KEY]).load()
    # pop_tracks = storage.PopularTracksStorage(users_to_process=[ct.TRAIN_KEY, ct.TEST_KEY]).load()
    nearest_users, all_users_tracks = storage.NearestUsersStorage(key_users_name=ct.TEST_KEY,
                                                                  to_be_sorted_users_name=[ct.TRAIN_KEY,
                                                                                           ct.TEST_KEY]).calculate_and_dump()
