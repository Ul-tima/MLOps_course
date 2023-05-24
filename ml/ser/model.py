from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPool2D
from tensorflow import keras


def create_model(input_shape: tuple[int, ...]) -> keras.Sequential:
    model = keras.Sequential()
    model.add(
        Conv2D(
            filters=64,
            kernel_size=5,
            strides=(2, 2),
            activation="relu",
            input_shape=input_shape,
        )
    )
    model.add(MaxPool2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=4, strides=(2, 1), activation="relu"))
    model.add(MaxPool2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=7, activation="softmax"))
    return model


from pathlib import Path

import wandb


def save_model_to_registry(model_name: str, model_path: Path):
    with wandb.init() as _:
        model_art = wandb.Artifact(model_name, type="model")
        model_art.add_file(model_path)
        wandb.log_artifact(model_art)
        wandb.link_artifact(model_art, "model-registry/My Registered Model")


def load_from_registry(model_name: str, model_path: Path):
    with wandb.init() as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact.download(root=model_path)
