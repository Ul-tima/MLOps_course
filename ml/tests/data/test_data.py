import os

import librosa
import pytest

dir_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_PATH = os.path.join(dir_path, "dataset", "sample")


@pytest.mark.parametrize("audio_file", [os.path.join(DATA_PATH, file) for file in os.listdir(DATA_PATH)])
def test_audio_file_format(audio_file):
    _, file_ext = os.path.splitext(audio_file)
    assert file_ext == ".wav"


@pytest.mark.parametrize("audio_file", [os.path.join(DATA_PATH, file) for file in os.listdir(DATA_PATH)])
class TestAudioData:
    def test_audio_file_format(self, audio_file):
        _, file_ext = os.path.splitext(audio_file)
        assert file_ext == ".wav"

    def test_audio_file_valid(self, audio_file):
        # Test that the audio file can be loaded by librosa without errors
        y, sr = librosa.load(audio_file, sr=None)
        assert y is not None

    def test_audio_duration(self, audio_file):
        # Check if the duration of the audio file is up to 5 sec
        y, sr = librosa.load(audio_file)
        duration = librosa.get_duration(y=y, sr=sr)
        assert duration <= 5
