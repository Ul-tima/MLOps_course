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


def test_model_input_shape(model):
    # Check that the model output shape is as expected
    assert model.input_shape == (None, 30, 150, 1)


def test_model_output_shape(model):
    # Check that the model output shape is as expected
    assert model.output_shape == (None, 7)


def test_loss_decreases(model, data):
    # Check that the loss decreases after one batch of training
    x, y = data
    model.compile(loss="categorical_crossentropy")
    loss_before = model.evaluate(x, y)
    model.train_on_batch(x, y)
    loss_after = model.evaluate(x, y)
    assert loss_after < loss_before


def test_overfit_on_batch(model, data):
    x, y = data
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=50, verbose=0)
    loss, acc = model.evaluate(x, y, verbose=0)

    assert acc > 0.9
