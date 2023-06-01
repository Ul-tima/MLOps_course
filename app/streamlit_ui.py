import operator

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ml.ser.prediction import Predictor

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


@st.cache_data
def get_title(predictions):
    max_key, max_value = max(predictions.items(), key=operator.itemgetter(1))
    title = f"Detected emotion: {max_key} \
    - {max_value * 100:.2f}%"
    return title


def plot_colored_polar(fig, predictions, title="", colors=COLOR_DICT):
    categories = list(predictions.keys())
    N = len(categories)
    ind = max(predictions, key=predictions.get)
    color = colors[ind]
    sector_colors = [colors[i] for i in categories]

    fig.set_facecolor("#d1d1e0")
    ax = plt.subplot(111, polar=True)

    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    for sector in range(N):
        radii = np.zeros(N)
        radii[sector] = predictions[categories[sector]] * 10
        width = np.pi / 1.8 * predictions[categories[sector]]
        c = sector_colors[sector]
        ax.bar(theta, radii, width=width, bottom=0.0, color=c, alpha=0.25)

    angles = [i / float(N) * 2 * np.pi for i in range(N)]
    angles += angles[:1]

    data = list(predictions.values())
    data += data[:1]
    plt.polar(angles, data, color=color, linewidth=2)
    plt.fill(angles, data, facecolor=color, alpha=0.25)

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
        st.audio(audio_file, format="audio/wav", start_time=0)
        st.sidebar.subheader("Audio file")
        file_details = {"Filename": audio_file.name, "FileSize": audio_file.size}
        st.sidebar.write(file_details)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("## Predictions")
            predictor = get_model()
            pred = predictor.predict(audio_file)
            txt = "Emotions\n" + get_title(pred)
            fig3 = plt.figure(figsize=(3, 3))
            plot_colored_polar(fig3, predictions=pred, title=txt, colors=COLOR_DICT)
            st.pyplot(fig3)


if __name__ == "__main__":
    main()
