import pathlib

import librosa
import numpy as np
from keras.saving.saving_api import load_model

from ml.ser import audio_processing
from ml.ser.model import load_from_registry


class Predictor:
    def __init__(self, model_name: str):
        local_dir = "model"
        load_from_registry(model_name, local_dir)
        local_path = list(pathlib.Path(local_dir).glob("*.h5"))[0]
        self.model = load_model(local_path)

    def predict(self, file_path):
        features = []
        audio, s_r = librosa.load(file_path, sr=16000)

        # without augmentation
        res1 = audio_processing.extract_features(audio, s_r)
        features.append(audio_processing.resize_audio_features(res1))
        features = np.expand_dims(features, axis=-1)
        y_predict = self.model.predict(features)
        return y_predict
