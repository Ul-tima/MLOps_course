import numpy as np
import pytest

from ml.ser import audio_processing


@pytest.fixture
def sample_audio():
    # Generate a random audio signal
    return np.random.rand(16000)


def test_add_noise(sample_audio):
    # Test that add_noise adds noise to the audio signal
    noisy_audio = audio_processing.add_noise(sample_audio)
    assert np.any(sample_audio != noisy_audio)
    assert sample_audio.shape == noisy_audio.shape


def test_stretch(sample_audio):
    # Test that stretch changes the duration of the audio signal
    stretched_audio = audio_processing.stretch(sample_audio)
    assert len(stretched_audio) != len(sample_audio)


def test_pitch(sample_audio):
    # Test that pitch changes the pitch of the audio signal
    pitched_audio = audio_processing.pitch(sample_audio)
    assert np.any(sample_audio != pitched_audio)
    assert sample_audio.shape == pitched_audio.shape


def test_resize_audio_features():
    # Test that resize_audio_features resizes the input feature matrix
    features = np.ones((30, 200))
    resized_features = audio_processing.resize_audio_features(features, mfcc_size=30, duration_size=150)
    assert resized_features.shape == (30, 150)


def test_extract_features(sample_audio):
    # Test that extract_features returns a feature matrix
    features = audio_processing.extract_features(sample_audio, sampling_rate=16000)
    assert features.ndim == 2
