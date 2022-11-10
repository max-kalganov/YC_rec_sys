import os
from typing import List

from pydub import AudioSegment


def get_track_file_path(track_id: str, postfix: str = None):
    postfix = f"_{postfix}" if postfix else ""
    path = os.path.join('data', 'tracks', f"{track_id}{postfix}.mp3")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def remove_vocals(track_ids: List[str]):
    def remove_track_vocals(track_filepath: str):
        # read in audio file and get the two mono tracks
        sound_stereo = AudioSegment.from_file(track_filepath, format="mp3")
        sound_monoL = sound_stereo.split_to_mono()[0]
        sound_monoR = sound_stereo.split_to_mono()[1]

        # Invert phase of the Right audio file
        sound_monoR_inv = sound_monoR.invert_phase()

        # Merge two L and R_inv files, this cancels out the centers
        sound_CentersOut = sound_monoL.overlay(sound_monoR_inv)

        return sound_CentersOut

    all_tracks_melodies = []

    for t_id in track_ids:
        all_tracks_melodies.append(remove_track_vocals(get_track_file_path(t_id)))

    return all_tracks_melodies


def write_melodies(track_ids, all_melodies):
    for t_id, melody in zip(track_ids, all_melodies):
        fh = melody.export(get_track_file_path(t_id, postfix="melody"), format="mp3")
        ...


if __name__ == '__main__':
    # sys.path.append('/opt/homebrew/bin/ffmpeg')
    # sys.path.append('/opt/homebrew/bin/ffprobe')

    no_voice = ['454758', '362253', '436737']
    with_voice = ['382341', '240893', '21451', '66462']

    # download_tracks(no_voice)
    # download_tracks(with_voice)

    no_voice_melodies = remove_vocals(no_voice)
    with_voice_melodies = remove_vocals(with_voice)

    write_melodies(no_voice, no_voice_melodies)
    write_melodies(with_voice, with_voice_melodies)
