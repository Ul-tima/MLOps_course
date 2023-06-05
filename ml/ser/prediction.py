import functools
import pathlib

import librosa
import numpy as np
from keras.saving.saving_api import load_model

from ml.ser import audio_processing
from ml.ser.model import load_from_registry

_current_dir = pathlib.Path(__file__).parent


@functools.lru_cache
def get_emotions() -> np.ndarray:
    path = _current_dir / "data" / "classes.npy"
    return np.load(path)


@functools.lru_cache
def get_scaling_params() -> tuple[np.ndarray, np.ndarray]:
    mean = np.load(_current_dir / "data" / "tr_mean.npy")
    std = np.load(_current_dir / "data" / "tr_std.npy")
    return mean, std


def scale_feature(feature: np.ndarray) -> np.ndarray:
    # Get mean and standard deviation from the training set
    mean, std = get_scaling_params()
    feature = (feature - mean) / std
    return feature


def feature_extraction(file_path, scale):
    features = []
    audio, s_r = librosa.load(file_path, sr=16000)
    # without augmentation
    res1 = audio_processing.extract_features(audio, s_r)
    features.append(audio_processing.resize_audio_features(res1))
    if scale:
        features = scale_feature(features)
    features = np.expand_dims(features, axis=-1)
    return features


class Predictor:
    def __init__(self, model_name: str):
        local_dir = model_name
        load_from_registry(model_name, local_dir)
        local_path = list(pathlib.Path(local_dir).glob("*.h5"))[0]
        self.model = load_model(local_path)

    def predict(self, file_path, scale=True) -> dict[str, float]:
        features = feature_extraction(file_path, scale)
        y_predict = self.model.predict(features)
        result = dict(zip(get_emotions(), y_predict[0].T))
        return result
