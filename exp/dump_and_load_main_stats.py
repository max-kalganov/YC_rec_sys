import abc
import logging
import os.path
from collections import Counter, defaultdict
from itertools import chain
from typing import List, Dict, Any, Union, Iterable

from tqdm import tqdm

import exp.constants as cnst
from preprocessing.utils import load_file_tracks, get_tracks_to_artists


class StatsStorage(abc.ABC):
    def __init__(self, name: str = "StatsStorage", file_name: str = "stats_storage_tempo", storage_root_dir: str = "data/stats"):
        self._storage_root_dir = storage_root_dir
        self.name = name
        self._key_to_load_map = {
            cnst.TRAIN_KEY: self.load_train,
            cnst.TEST_KEY: self.load_test,
            cnst.TRACKS_TO_ARTISTS_KEY: self.load_tracks_to_artists
        }
        self.file_path = self._get_file_path(file_name)


    @staticmethod
    def load_train():
        return load_file_tracks(cnst.KEY_TO_FILENAME[cnst.TRAIN_KEY])

    @staticmethod
    def load_test():
        return load_file_tracks(cnst.KEY_TO_FILENAME[cnst.TEST_KEY])

    @staticmethod
    def load_tracks_to_artists():
        return get_tracks_to_artists(cnst.KEY_TO_FILENAME[cnst.TRACKS_TO_ARTISTS_KEY])

    def _get_file_path(self, file_name: str) -> str:
        return os.path.join(self._storage_root_dir, file_name)

    def dump_lines_in_file(self, list_of_lines: List[str]):
        with open(self.file_path, 'w') as f:
            f.writelines(list_of_lines)
        logging.debug(f"Dumped into {self.file_path} by {self.name}")

    def load_lines_from_file(self, file: str):
        with open(file, 'r') as f:
            lines = f.readlines()
        logging.debug(f"Loaded from {file} by {self.name}")
        return lines

    @staticmethod
    def _dict_to_list_of_str(dict_to_dump: Dict[Union[int, str], Iterable]) -> List[str]:
        converted_lines = []
        for key, value in dict_to_dump.items():
            converted_value = ','.join([str(v) for v in value])
            converted_pair = f"{key}-{converted_value}\n"
            converted_lines.append(converted_pair)
        return converted_lines

    @staticmethod
    def _list_of_str_to_dict(list_of_str: List[str]) -> Dict[str, List[str]]:
        result_dict = {}
        for line in list_of_str:
            k, v = line.strip().split('-')
            parse_v = v.split(',')
            result_dict[k] = parse_v
        return result_dict

    @staticmethod
    def _list_of_lists_to_list_of_str(lists: List[List]) -> List[str]:
        formatted_lists = []
        for l in lists:
            formatted_l = ' '.join([str(v) for v in l])
            formatted_lists.append(formatted_l + '\n')
        return formatted_lists

    @staticmethod
    def _list_of_str_to_list_of_lists(lines: List[str]) -> List[List[str]]:
        lists = []
        for line in lines:
            formatted_line = line.strip().split(' ')
            lists.append(formatted_line)
        return lists

    @abc.abstractmethod
    def load(self) -> Any:
        pass

    @abc.abstractmethod
    def calculate_and_dump(self, *args, **kwargs):
        pass


class TracksToUsersStorage(StatsStorage):
    def __init__(self, mapped_users_name: List[str]):
        assert all(name in cnst.ALL_KEYS for name in mapped_users_name)
        super().__init__(name="Tracks To Users Storage",
                         file_name=f"tracks_to_users_storage_{'|'.join(mapped_users_name)}")
        self._mapped_users_name = mapped_users_name

    def get_all_users_lists(self):
        logging.debug(f"{self.name}: Getting all users lists")
        all_users_datasets = [self._key_to_load_map[key_name]() for key_name in self._mapped_users_name]
        all_users_lists = chain(*all_users_datasets)
        return list(all_users_lists)

    def calculate_and_dump(self, *args, **kwargs):
        tracks_to_users = defaultdict(list)
        all_users_lists = self.get_all_users_lists()
        for i, single_user_tracks in tqdm(enumerate(all_users_lists), desc=f"{self.name} running: "):
            for track in single_user_tracks:
                tracks_to_users[track].append(i)

        lines = self._dict_to_list_of_str(tracks_to_users)
        self.dump_lines_in_file(lines)
        return tracks_to_users, all_users_lists

    def load(self):
        if os.path.exists(self.file_path):
            logging.info(f"{self.name}: Loading from {self.file_path}")
            lines = self.load_lines_from_file(self.file_path)
            tracks_to_users = self._list_of_str_to_dict(lines)
            all_users_lists = self.get_all_users_lists()
        else:
            logging.info(f"{self.name}: Calculating and dumping into {self.file_path}")
            tracks_to_users, all_users_lists = self.calculate_and_dump()
        return tracks_to_users, all_users_lists


class NearestUsersStorage(StatsStorage):
    def __init__(self, key_users_name: str, to_be_sorted_users_name: List[str]):
        assert all(name in cnst.ALL_KEYS for name in to_be_sorted_users_name)
        assert key_users_name in cnst.ALL_KEYS

        super().__init__(name="Nearest Users Storage",
                         file_name=f"nearest_users_storage_{key_users_name};{'|'.join(to_be_sorted_users_name)}")
        self._key_user = key_users_name
        self._to_be_sorted_users = to_be_sorted_users_name
        self.track_to_users_loader = TracksToUsersStorage(mapped_users_name=self._to_be_sorted_users)

    def calculate_and_dump(self, *args, **kwargs):
        key_users = self._key_to_load_map[self._key_user]()
        tracks_to_users, all_users_lists = self.track_to_users_loader.load()

        all_nearest_users = []
        for single_user_tracks in tqdm(key_users, desc=f"{self.name} running: "):
            nearest_users = Counter()
            for track in single_user_tracks:
                nearest_users.update(tracks_to_users[track])

            current_user_nearest = [user_id for user_id, intersection in nearest_users.most_common() if intersection > 0]
            all_nearest_users.append(current_user_nearest)

        formatted_nearest_users = self._list_of_lists_to_list_of_str(all_nearest_users)
        self.dump_lines_in_file(formatted_nearest_users)
        return all_nearest_users, all_users_lists

    def load(self):
        if os.path.exists(self.file_path):
            logging.info(f"{self.name}: Loading from {self.file_path}")
            lines = self.load_lines_from_file(self.file_path)
            nearest_users = self._list_of_str_to_list_of_lists(lines)
            all_users_lists = self.track_to_users_loader.get_all_users_lists()
        else:
            logging.info(f"{self.name}: Calculating and dumping into {self.file_path}")
            nearest_users, all_users_lists = self.calculate_and_dump()
        return nearest_users, all_users_lists


class PopularTracksStorage(StatsStorage):
    def __init__(self, users_to_process: List[str]):
        assert all(user in cnst.ALL_KEYS for user in users_to_process)
        super().__init__(name="Popular Tracks Storage",
                         file_name=f"popular_tracks_storage_{'|'.join(users_to_process)}")
        self._users_to_process = users_to_process

    @staticmethod
    def _pop_tracks_to_str(pop_tracks: List[str]) -> List[str]:
        return [pop_track + '\n' for pop_track in pop_tracks]

    @staticmethod
    def _format_pop_tracks(pop_tracks: List[str]) -> List[str]:
        return [pop_track.strip() for pop_track in pop_tracks]

    def calculate_and_dump(self, *args, **kwargs):
        datasets_users = [self._key_to_load_map[user]() for user in self._users_to_process]

        tracks_stats = Counter()
        for users_tracks, users_tracks_name in zip(datasets_users, self._users_to_process):
            for single_user_tracks in tqdm(users_tracks, desc=f"{self.name} running ({users_tracks_name}): "):
                tracks_stats.update(single_user_tracks)

        popular_tracks = [track for track, popularity in tracks_stats.most_common()]

        popular_tracks_to_write = self._pop_tracks_to_str(popular_tracks)
        self.dump_lines_in_file(popular_tracks_to_write)

        return popular_tracks

    def load(self):
        if os.path.exists(self.file_path):
            logging.info(f"{self.name}: Loading from {self.file_path}")
            lines = self.load_lines_from_file(self.file_path)
            pop_tracks = self._format_pop_tracks(lines)
        else:
            logging.info(f"{self.name}: Calculating and dumping into {self.file_path}")
            pop_tracks = self.calculate_and_dump()
        return pop_tracks
