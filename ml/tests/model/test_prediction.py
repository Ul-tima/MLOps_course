import os

import numpy as np
import pytest

from ml.ser.model import load_from_registry
from ml.ser.prediction import Predictor
from ml.ser.prediction import feature_extraction
from ml.ser.prediction import get_emotions
from ml.ser.prediction import get_scaling_params
from ml.ser.prediction import scale_feature


def test_get_emotions():
    emotions = get_emotions()
    assert isinstance(emotions, np.ndarray)
    assert len(emotions) == 7


def test_get_scaling_params():
    mean, std = get_scaling_params()
    assert isinstance(mean, np.ndarray)
    assert isinstance(std, np.ndarray)
    assert mean.shape == std.shape


def test_scale_feature(feature):
    scaled_feature = scale_feature(feature)
    assert isinstance(scaled_feature, np.ndarray)
    assert scaled_feature.shape == feature.shape


def test_feature_extraction(file_path):
    features = feature_extraction(file_path, scale=True)
    assert isinstance(features, np.ndarray)
    assert len(features.shape) == 4


def test_predict(file_path):
    predictor = Predictor("cnn")
    result = predictor.predict(file_path, scale=True)
    assert isinstance(result, dict)
    assert len(result) == 7
    assert sum(result.values()) == pytest.approx(1.0)
