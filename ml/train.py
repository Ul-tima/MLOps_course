import librosa
import numpy as np
import pandas as pd
import wandb
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from wandb.integration.keras import WandbMetricsLogger
from wandb.integration.keras import WandbModelCheckpoint

from ml import audio_processing
from ml.load_datasets import get_dataset
from ml.model import create_model


def scale_data(train, valid, test):
    # Get mean and standard deviation from the training set
    tr_mean = np.mean(train, axis=0)
    tr_std = np.std(train, axis=0)
    # Apply data scaling
    train = (train - tr_mean) / tr_std
    valid = (valid - tr_mean) / tr_std
    test = (test - tr_mean) / tr_std
    return train, valid, test


def train():
    wandb.init(
        project="ser",
        config={
            "optimizer": "Adam",
            "loss": "categorical_crossentropy",
            "metric": ["accuracy"],
            "epochs": 8,
            "batch_size": 64,
        },
    )

    config = wandb.config

    data = get_dataset(False, False, True)
    x, y = prepare_training_data(data)

    # Create train, validation and test set
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(x), np.array(y), train_size=0.8, shuffle=True, random_state=0
    )
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=0)
    x_train, x_valid, x_test = scale_data(x_train, x_valid, x_test)

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    x_valid = np.expand_dims(x_valid, axis=-1)

    model = create_model(x_train.shape[1:])
    model.compile(loss=config.loss, optimizer=config.optimizer, metrics=config.metric)

    callbacks = [keras.callbacks.EarlyStopping(patience=5), WandbMetricsLogger()]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
    )

    y_predict = model.predict(x_test)
    matrix = confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1))
    print(matrix)

    wandb.finish()


def prepare_training_data(df: pd.DataFrame, sr: float = 16000):
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

    return X, Y


if __name__ == "__main__":
    train()
