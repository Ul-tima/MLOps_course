import pathlib

import pytest


def test__predict_emotion__ok(client):
    project_root = pathlib.Path(__file__).parents[5]
    audio_file = project_root / "dataset" / "sample" / "03-01-01-01-01-01-01.wav"

    with open(audio_file, "rb") as file:
        response = client.post("/v1/predictions/", files={"audio": file})

    assert response.status_code == 200, response.text
    assert response.json() == {
        "prediction": pytest.approx(
            {
                "neutral": 0.491,
                "sad": 0.216,
                "disgust": 0.102,
                "happy": 0.07,
                "fear": 0.06,
                "surprise": 0.043,
                "angry": 0.019,
            },
            abs=0.001,
        )
    }
