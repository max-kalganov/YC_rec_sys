import librosa
from pydub import AudioSegment
import numpy as np
import plotly.express as px


def librosa_example():
    # 1. Get the file path to an included audio example
    filename = librosa.example('nutcracker')

    sound = AudioSegment.from_mp3('../data/tracks/21451.mp3')
    sound.export('data/tracks/21451.wav', format="wav")

    # 2. Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    y, sr = librosa.load('../data/tracks/21451.wav')

    # 3. Run the default beat tracker
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

    # 4. Convert the frame indices of beat events into timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)


def librosa_example_2():

    y, sr = librosa.load('../data/tracks/21451.wav')
    hop_length = 512

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr = sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr = sr)
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate = np.median)
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
    print("end example 2")


def librosa_example_3():
    y, sr = librosa.load('../data/tracks/21451.wav')

    for i in range(30, 33, 1):
        y_h, y_p = librosa.effects.hpss(y, margin=i)
        tempo, beat_frames = librosa.beat.beat_track(y=y_p, sr=sr)
        print(f"{i} - {tempo}")

    bt = librosa.beat.beat_track(y=y, sr=sr)
    p = librosa.beat.plp(y=y, sr=sr)
    t = librosa.beat.tempo(y=y, sr=sr)
    print(bt,p,t)


if __name__ == '__main__':
    librosa_example_3()
