import librosa
import numpy as np


def add_noise(audio: np.array) -> np.array:
    noise_amp = 0.035 * np.random.uniform() * np.amax(audio)
    audio = audio + noise_amp * np.random.normal(size=audio.shape[0])
    return audio


def stretch(audio: np.array, rate: float = 0.8) -> np.array:
    return librosa.effects.time_stretch(y=audio, rate=rate)


def pitch(audio: np.array, sampling_rate: float = 16000, n_steps: float = 2) -> np.ndarray:
    return librosa.effects.pitch_shift(y=audio, sr=sampling_rate, n_steps=n_steps)


def resize_audio_features(features: np.array, mfcc_size: int = 30, duration_size: int = 150) -> np.array:
    new_matrix = np.zeros((mfcc_size, duration_size))
    for i in range(mfcc_size):
        for j in range(duration_size):
            try:
                new_matrix[i][j] = features[i][j]
            except IndexError:
                pass
    return new_matrix


def extract_features(audio: np.array, sampling_rate: float) -> np.array:
    return librosa.feature.mfcc(y=audio, sr=sampling_rate, fmin=50, n_mfcc=30)
