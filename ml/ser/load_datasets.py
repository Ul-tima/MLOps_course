import json
import os
import re
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd


def read_crema_data(path_dir: str) -> pd.DataFrame:
    emotion_df = []

    files = os.listdir(path_dir)
    for wav in files:
        info = wav.partition(".wav")[0].split("_")
        emotion = info[2]
        emotion_df.append((emotion, os.path.join(path_dir, wav)))

    crema_df = pd.DataFrame.from_dict(emotion_df)
    crema_df.rename(columns={1: "Path", 0: "Emotion"}, inplace=True)
    crema_df.Emotion.replace(
        {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fear", "DIS": "disgust"}, inplace=True
    )
    return crema_df


def read_savee_data(path_dir: str) -> pd.DataFrame:
    root = os.listdir(path_dir)

    emotion_df = []

    for dir in root:
        next_dir = os.path.join(path_dir, dir)
        if not os.path.isdir(next_dir):
            continue

        actor = os.listdir(next_dir)
        for wav in actor:
            emotion = re.sub("[^a-zA-Z]+", "", wav.partition(".wav")[0])
            emotion_df.append((emotion, os.path.join(path_dir, dir, wav)))

    savee_df = pd.DataFrame.from_dict(emotion_df)
    savee_df.rename(columns={1: "Path", 0: "Emotion"}, inplace=True)
    savee_df.Emotion.replace(
        {"n": "neutral", "h": "happy", "sa": "sad", "a": "angry", "f": "fear", "d": "disgust", "su": "surprise"},
        inplace=True,
    )
    return savee_df


def read_ravdess_data(path_dir: str) -> pd.DataFrame:
    root = os.listdir(path_dir)

    emotion_df = []

    for dir in root:
        next_dir = os.path.join(path_dir, dir)
        if not os.path.isdir(next_dir):
            continue

        actor = os.listdir(next_dir)
        for wav in actor:
            info = wav.partition(".wav")[0].split("-")
            emotion = int(info[2])
            emotion_df.append((emotion, os.path.join(path_dir, dir, wav)))

    ravdess_df = pd.DataFrame.from_dict(emotion_df)
    ravdess_df.rename(columns={1: "Path", 0: "Emotion"}, inplace=True)
    ravdess_df.Emotion.replace(
        {1: "neutral", 2: "neutral", 3: "happy", 4: "sad", 5: "angry", 6: "fear", 7: "disgust", 8: "surprise"},
        inplace=True,
    )
    return ravdess_df


def get_dataset(ravdess: bool, crema: bool, savee: bool) -> pd.DataFrame:
    dataset = pd.DataFrame({"Path": pd.Series(dtype="str"), "Emotion": pd.Series(dtype="str")})
    par_dir = Path(__file__).resolve().parents[1]
    if ravdess:
        path = os.path.join(par_dir, "dataset", "ravdess")
        dataset = pd.concat([dataset, read_ravdess_data(path)], ignore_index=True)
    if crema:
        path = os.path.join(par_dir, "dataset", "crema")
        dataset = pd.concat([dataset, read_crema_data(path)], ignore_index=True)
    if savee:
        path = os.path.join(par_dir, "dataset", "savee")
        dataset = pd.concat([dataset, read_savee_data(path)], ignore_index=True)
    return dataset


def load_saved_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.load(os.path.join(data_dir, "x_train_ex.npy"))
    x_test = np.load(os.path.join(data_dir, "x_test_ex.npy"))
    x_valid = np.load(os.path.join(data_dir, "x_valid_ex.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    y_valid = np.load(os.path.join(data_dir, "y_valid.npy"))
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def load_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r") as f:
        config = json.safe_load(f)
    return config
