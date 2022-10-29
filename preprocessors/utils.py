import os


def get_track_file_path(track_id: str, track_root_folder: str, postfix: str = None):
    postfix = f"_{postfix}" if postfix else ""
    file_name = f"{track_id}{postfix}.wav"
    return os.path.join(track_root_folder, file_name)
