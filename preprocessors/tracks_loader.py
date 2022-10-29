import logging
import os
from typing import List, Optional
import yandex_music as ym
from pydub import AudioSegment

from preprocessors.utils import get_track_file_path

logger = logging.getLogger()


class WAVTracksLoader:
    """Loads tracks from internet and stores in wav format in local storage"""
    tracks_ids: List[str] = []
    tracks_root_folder: Optional[str] = None

    def __init__(self, tracks_ids: List[str], tracks_root_folder: str) -> None:
        self.tracks_ids = tracks_ids
        self.tracks_root_folder = tracks_root_folder

    def download_track(self, yandex_client: ym.Client, track_id: str, track_output_file: str) -> None:
            tempo_filepath = "data/tempo.mp3"

            track_download_info = yandex_client.tracks_download_info(track_id)
            if len(track_download_info) > 1:
                logger.warning(f"Found {len(track_download_info)} download info for track {track_id}")
            track_download_info[0].download(tempo_filepath)

            track_mp3 = AudioSegment.from_mp3(tempo_filepath)
            track_mp3.export(track_output_file, format="wav")

            os.remove(tempo_filepath)

    def load_tracks(self, postfix: str = ""):
        os.makedirs(self.tracks_root_folder, exist_ok=True)
        client = ym.Client()
        client.init()

        for track_id in self.tracks_ids:
            track_output_file = get_track_file_path(track_id=track_id,
                                                    postfix=postfix,
                                                    track_root_folder=self.tracks_root_folder)
            if not os.path.exists(track_output_file):
                logger.info(f"Loading track with id = {track_id} into {track_output_file}")
                self.download_track(yandex_client=client, track_id=track_id, track_output_file=track_output_file)
            else:
                logger.info(f"Track with id = {track_id} is already loaded in {track_output_file}")
