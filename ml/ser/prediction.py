import pathlib

import librosa
import numpy as np
from keras.saving.saving_api import load_model

from ml.ser import audio_processing
from ml.ser.model import load_from_registry


def scale_feature(feature: np.ndarray) -> np.ndarray:
    # Get mean and standard deviation from the training set
    cur_dir = pathlib.Path(__file__).parent
    tr_mean = np.load(cur_dir / "data/tr_mean.npy")
    tr_std = np.load(cur_dir / "data/tr_std.npy")

    feature = (feature - tr_mean) / tr_std

    return feature


def feture_extraction(file_path, scale):
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

    def predict(self, file_path, scale=True):
        features = feture_extraction(file_path, scale)
        y_predict = self.model.predict(features)
        return y_predict
