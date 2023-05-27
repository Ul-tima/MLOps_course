from pathlib import Path
from typing import List
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
import wandb
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from wandb.integration.keras import WandbCallback
from wandb.integration.keras import WandbMetricsLogger

from ml.ser import audio_processing
from ml.ser.evaluation import plot_confusion_matrix
from ml.ser.load_datasets import get_dataset
from ml.ser.load_datasets import load_saved_data
from ml.ser.model import create_model
from ml.ser.model import save_model_to_registry


def scale_data(train: np.ndarray, valid: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Get mean and standard deviation from the training set
    tr_mean = np.mean(train, axis=0)
    tr_std = np.std(train, axis=0)

    np.save("data/tr_mean.npy", tr_mean)
    np.save("data/tr_std.npy", tr_std)
    # Apply data scaling
    train = (train - tr_mean) / tr_std
    valid = (valid - tr_mean) / tr_std
    test = (test - tr_mean) / tr_std
    return train, valid, test


def train(use_saved_data: bool = False) -> None:
    wandb.init(
        project="ser",
        config={
            "optimizer": "Adam",
            "loss": "categorical_crossentropy",
            "metric": ["accuracy"],
            "epochs": 30,
            "batch_size": 64,
        },
    )

    if use_saved_data:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_saved_data("data/")
    else:
        data = get_dataset(True, True, True)
        x, y = prepare_training_data(data)

        x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(x, y)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        x_valid = np.expand_dims(x_valid, axis=-1)

        np.save("data/x_train_ex.npy", x_train)
        np.save("data/x_valid_ex.npy", x_valid)
        np.save("data/x_test_ex.npy", x_test)

        np.save("data/y_train.npy", y_train)
        np.save("data/y_valid.npy", y_valid)
        np.save("data/y_test.npy", y_test)
    model_path = f"data/model_{wandb.run.id}.h5"
    model = train_model(wandb.config, x_train, x_valid, y_train, y_valid, model_path)

    save_model_to_registry(f"cnn", model_path)

    wandb.finish()

    y_predict = model.predict(x_test)
    matrix = confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1))
    target = np.load("data/classes.npy")
    plot_confusion_matrix(matrix, target, "data/confusion_matrix_test.png")

    wandb.finish()


def split_data(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Create train, validation, and test sets
    # Create train, validation and test set
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(x), np.array(y), train_size=0.8, shuffle=True, random_state=0
    )
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=0)
    x_train, x_valid, x_test = scale_data(x_train, x_valid, x_test)
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def train_model(
    config: dict, x_train: np.ndarray, x_valid: np.ndarray, y_train: np.ndarray, y_valid: np.ndarray, model_path: Path
) -> keras.Model:
    model = create_model(x_train.shape[1:])
    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=config["metric"])
    callbacks = [keras.callbacks.EarlyStopping(patience=5), WandbMetricsLogger(), WandbCallback()]
    model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
    )
    model.save(model_path)
    return model


def prepare_training_data(df: pd.DataFrame, sr: float = 16000) -> Tuple[List[np.ndarray], np.ndarray]:
    X = []
    Y = []
    for i in range(len(df)):
        features = []
        audio, s_r = librosa.load(df["Path"].iloc[i], sr=sr)

        # without augmentation
        res1 = audio_processing.extract_features(audio, s_r)
        features.append(audio_processing.resize_audio_features(res1))

        # with noise
        noise_data = audio_processing.add_noise(audio)
        res2 = audio_processing.extract_features(noise_data, s_r)
        features.append(audio_processing.resize_audio_features(res2))

        # with stretching and pitching
        new_data = audio_processing.stretch(audio)
        data_stretch_pitch = audio_processing.pitch(new_data)
        res3 = audio_processing.extract_features(data_stretch_pitch, s_r)
        features.append(audio_processing.resize_audio_features(res3))

        for j in features:
            X.append(j)
            Y.append(df["Emotion"].iloc[i])

    lb = LabelEncoder()
    Y = np_utils.to_categorical(lb.fit_transform(Y))
    np.save("data/classes.npy", lb.classes_)

    return X, Y


if __name__ == "__main__":
    train(True)
