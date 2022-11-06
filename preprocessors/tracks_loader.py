import logging
import os
from typing import List, Optional

import gin
import yandex_music as ym
from pydub import AudioSegment
from tqdm import tqdm

from preprocessors.utils import get_track_file_path


@gin.configurable
class WAVTracksLoader:
    """Loads tracks from internet and stores in wav format in local storage"""
    tracks_root_folder: Optional[str] = None

    def __init__(self, tracks_root_folder: str, api_token: str) -> None:
        self.tracks_root_folder = tracks_root_folder
        self.__cnt_errors = 0
        self.__api_token = api_token

    def __clean_cnt_errors(self):
        self.__cnt_errors = 0

    def get_single_download_info(self, yandex_client: ym.Client, track_id: str) -> Optional[ym.DownloadInfo]:
        try:
            track_download_info = yandex_client.tracks_download_info(track_id)
            if len(track_download_info) > 1:
                logging.warning(f"Found {len(track_download_info)} download info for track {track_id}")
            return track_download_info[0]
        except Exception as e:
            logging.warning(e)
            self.__cnt_errors += 1
            logging.warning(f"Couldn't download download_info for track {track_id} -- cnt {self.__cnt_errors}")
        return None

    def download_track(self, yandex_client: ym.Client, track_id: str, track_output_file: str) -> None:
        tempo_filepath = "data/tempo.mp3"

        track_download_info = self.get_single_download_info(yandex_client, track_id)

        if track_download_info is not None:
            track_download_info.download(tempo_filepath)

            track_mp3 = AudioSegment.from_mp3(tempo_filepath)
            track_mp3.export(track_output_file, format="wav")

            os.remove(tempo_filepath)

    def get_download_info(self, users_tracks_ids: List[List[str]]) -> List[List[ym.DownloadInfo]]:
        self.__clean_cnt_errors()
        client = ym.Client(token=self.__api_token)
        client.init()

        tracks_download_info = []
        for user_ind, single_user_tracks_ids in enumerate(users_tracks_ids):
            user_tracks_download_info = []
            for track_id in single_user_tracks_ids:
                singe_download_info = self.get_single_download_info(client, track_id)
                if singe_download_info is not None:
                    user_tracks_download_info.append(singe_download_info)
            tracks_download_info.append(user_tracks_download_info)
        return tracks_download_info

    def load_tracks(self, users_tracks_ids: List[List[str]], postfix: str = "") -> None:
        self.__clean_cnt_errors()
        os.makedirs(self.tracks_root_folder, exist_ok=True)
        client = ym.Client()
        client.init()
        processed_tracks = set()
        for user_ind, single_user_tracks_ids in enumerate(tqdm(users_tracks_ids)):
            for track_id in single_user_tracks_ids:
                if track_id in processed_tracks:
                    continue
                track_output_file = get_track_file_path(track_id=track_id,
                                                        postfix=postfix,
                                                        user_ind=user_ind,
                                                        track_root_folder=self.tracks_root_folder)
                if not os.path.exists(track_output_file):
                    logging.info(f"Loading track with id = {track_id} into {track_output_file}")
                    self.download_track(yandex_client=client,
                                        track_id=track_id,
                                        track_output_file=track_output_file)
                processed_tracks.add(track_id)