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
