import pathlib

import librosa
import pytest

current_dir = pathlib.Path(__file__).parent
tests_dir = current_dir.parent
package_dir = tests_dir.parent
project_root_dir = package_dir.parent
samples_dir = project_root_dir / "dataset" / "sample"
audio_files = [sample for sample in samples_dir.iterdir()]


@pytest.mark.parametrize("audio_file", audio_files)
class TestAudioData:
    def test_audio_file_format(self, audio_file: pathlib.Path):
        assert audio_file.suffix == ".wav"

    def test_audio_file_valid(self, audio_file: pathlib.Path):
        # Test that the audio file can be loaded by librosa without errors
        y, sr = librosa.load(audio_file, sr=None)
        assert y is not None

    def test_audio_duration(self, audio_file: pathlib.Path):
        # Check if the duration of the audio file is up to 5 sec
        y, sr = librosa.load(audio_file)
        duration = librosa.get_duration(y=y, sr=sr)
        assert duration <= 5
