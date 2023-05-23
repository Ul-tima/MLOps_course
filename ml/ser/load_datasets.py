import os
import re

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
    par_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
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
