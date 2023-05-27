import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ml.ser.prediction import Predictor

CAT7 = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


COLOR_DICT = {
    "neutral": "grey",
    "happy": "green",
    "surprise": "orange",
    "fear": "purple",
    "angry": "red",
    "sad": "lightblue",
    "disgust": "brown",
}


st.set_page_config(page_title="SER web-app", page_icon=":speech_balloon:", layout="wide")


@st.cache_resource
def get_model() -> Predictor:
    return Predictor("cnn")


# @st.cache
def save_audio(file):
    if file.size > 4000000:
        return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "audio"
    # clear the folder to avoid storage overload
    os.makedirs(folder, exist_ok=True)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0


def check_audio(audio_file):
    if audio_file is not None:
        if not os.path.exists("audio"):
            os.makedirs("audio")
        path = os.path.join("audio", audio_file.name)
        if_save_audio = save_audio(audio_file)
        if if_save_audio == 1:
            st.warning("File size is too large. Try another file.")
    return path


@st.cache_data
def get_title(predictions, categories=CAT7):
    title = f"Detected emotion: {categories[predictions.argmax()]} \
    - {predictions.max() * 100:.2f}%"
    return title


def plot_colored_polar(fig, predictions, categories, title="", colors=COLOR_DICT):
    N = len(predictions)
    ind = predictions.argmax()

    COLOR = colors[categories[ind]]
    sector_colors = [colors[i] for i in categories]

    fig.set_facecolor("#d1d1e0")
    ax = plt.subplot(111, polar="True")

    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    for sector in range(predictions.shape[0]):
        radii = np.zeros_like(predictions)
        radii[sector] = predictions[sector] * 10
        width = np.pi / 1.8 * predictions
        c = sector_colors[sector]
        ax.bar(theta, radii, width=width, bottom=0.0, color=c, alpha=0.25)

    angles = [i / float(N) * 2 * np.pi for i in range(N)]
    angles += angles[:1]

    data = list(predictions)
    data += data[:1]
    plt.polar(angles, data, color=COLOR, linewidth=2)
    plt.fill(angles, data, facecolor=COLOR, alpha=0.25)

    ax.spines["polar"].set_color("lightgrey")
    ax.set_theta_offset(np.pi / 3)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], color="grey", size=8)

    plt.suptitle(title, color="darkblue", size=10)
    plt.ylim(0, 1)
    plt.subplots_adjust(top=0.75)


def main():
    st.header("Speech emotion recognition demo")
    st.markdown("## Upload the file")
    audio_file = st.file_uploader("Upload audio file", type=["wav"])

    if audio_file is not None:
        is_saved = save_audio(audio_file)
        st.audio(audio_file, format="audio/wav", start_time=0)

        st.markdown("## Analyzing...")
        st.sidebar.subheader("Audio file")
        file_details = {"Filename": audio_file.name, "FileSize": audio_file.size}
        st.sidebar.write(file_details)
        path = os.path.join("audio", audio_file.name)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("## Predictions")
            predictor = get_model()
            pred = predictor.predict(path)[0]
            txt = "Emotions\n" + get_title(pred, CAT7)
            fig3 = plt.figure(figsize=(3, 3))
            plot_colored_polar(fig3, predictions=pred, categories=CAT7, title=txt, colors=COLOR_DICT)
            st.pyplot(fig3)


if __name__ == "__main__":
    main()
