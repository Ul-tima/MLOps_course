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
