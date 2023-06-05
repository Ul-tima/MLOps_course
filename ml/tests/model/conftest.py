import pathlib
import random

import numpy as np
import pytest

from ml.ser.model import create_model


@pytest.fixture()
def model():
    # Set up the model
    model = create_model(input_shape=(30, 150, 1))
    return model


@pytest.fixture()
def data():
    samples_count = 200
    x = np.random.randn(samples_count, 30, 150, 1)
    y = np.zeros((samples_count, 7))

    for i in range(samples_count):
        row_sum = 0
        for j in range(7):
            if row_sum < 1:
                y[i][j] = np.random.randint(0, 2)
                row_sum += y[i][j]
            else:
                break

    return x, y


@pytest.fixture()
def feature():
    x = np.random.randn(30, 150)
    return x


@pytest.fixture()
def file_path():
    dir = pathlib.Path(__file__).parents[3]
    samples_dir = dir / "dataset" / "sample"
    path = random.choice(list(samples_dir.iterdir()))
    return path
