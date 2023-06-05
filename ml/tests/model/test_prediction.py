import os

import numpy as np
import pytest

from ml.ser import prediction


def test_get_emotions():
    emotions = prediction.get_emotions()
    assert isinstance(emotions, np.ndarray)
    assert len(emotions) == 7


def test_get_scaling_params():
    mean, std = prediction.get_scaling_params()
    assert isinstance(mean, np.ndarray)
    assert isinstance(std, np.ndarray)
    assert mean.shape == std.shape


def test_scale_feature(feature):
    scaled_feature = prediction.scale_feature(feature)
    assert isinstance(scaled_feature, np.ndarray)
    assert scaled_feature.shape == feature.shape


def test_feature_extraction(file_path):
    features = prediction.feature_extraction(file_path, scale=True)
    assert isinstance(features, np.ndarray)
    assert len(features.shape) == 4


def test_predict(file_path):
    predictor = prediction.Predictor("cnn")
    result = predictor.predict(file_path, scale=True)
    assert isinstance(result, dict)
    assert len(result) == 7
    assert sum(result.values()) == pytest.approx(1.0)
