import os

import gin


def get_track_file_path(track_id: str, track_root_folder: str, user_ind: int, postfix: str = None):
    user_ind_postfix = f"user_{user_ind}"
    postfix = f"_{postfix}" if postfix else ""
    file_name = f"{track_id}{postfix}_{user_ind_postfix}.wav"
    return os.path.join(track_root_folder, file_name)


@gin.configurable
def features_to_numpy():
    pass


def get_train_val_tracks_ids():
    pass


def get_test_tracks_ids():
    pass
