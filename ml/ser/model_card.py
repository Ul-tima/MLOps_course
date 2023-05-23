import numpy as np
from keras.models import load_model
from keras.utils import plot_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from ml.ser.evaluation import plot_classification_report
from ml.ser.evaluation import plot_confusion_matrix


class ModelCardGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.content = []
        self.sections = {
            "Model Overview": [],
            "Intended Use": [],
            "Limitations": [],
            "Datasets": [],
            "Training Information": [],
            "Evaluation Metrics": [],
        }
        self.charts = []

    def add_section(self, title, content):
        self.sections[title] = content

    def add_chart(self, title, filename):
        self.charts.append(f"## {title}\n\n![{title}]({filename})\n")

    def generate_md(self, filename):
        self.content.append(f"# {self.model_name} Model Card\n")
        for section_title in self.sections:
            section_content = self.sections[section_title]
            if section_content:
                self.content.append(f"## {section_title}\n")
                self.content.append(section_content)

        content = "\n".join(self.content + self.charts)
        with open(filename, "w") as f:
            f.write(content)


if __name__ == "__main__":
    # create an instance of the model card generator
    model_card = ModelCardGenerator("Speech Emotion Recognition")

    # add model overview section
    model_summary = "The model is a Keras deep learning model trained to predict the emotion of a given speech sample."

    model_card.add_section("Model Overview", model_summary)

    # add intended use section
    intended_use = "This model is intended to be used for classifying the emotion in speech samples in real-time. It can be integrated into a mobile or web application for user feedback, emotion recognition in customer service interactions, or for educational or therapeutic purposes."
    model_card.add_section("Intended Use", intended_use)

    # add limitations section
    limitations = "This model may not perform well on speech samples from speakers with different age, gender, or language backgrounds than those represented in the training data. It may also struggle with recognizing emotions that are not well-represented in the training data."
    model_card.add_section("Limitations", limitations)

    # add training information section
    training_info = "The model was trained on a dataset of speech samples labeled with the corresponding emotion. The dataset was splited into 80% training, 10% validation, and 10% testing sets. The model was trained using the categorical cross-entropy loss function and the Adam optimizer with default hyperparameters for 30 epochs. The batch size was set to 64."
    model_card.add_section("Training Information", training_info)

    # add datasets section
    datasets = "The model was trained on three publicly available datasets: RAVDESS, CREMA-D, and SAVEE. These datasets contain a total of 9,362 speech samples across 7 emotional categories."
    model_card.add_section("Datasets", datasets)

    model_information = "The model is a convolutional neural network (CNN) for speech emotion recognition. It consists of two convolutional layers with max pooling and batch normalization, followed by three fully connected layers with dropout and a softmax output layer. The model is compiled with the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric. "
    model_card.add_section("Model Information", model_information)

    model = load_model("data/ser_v_30ep.h5")
    dot_img_file = "data/model.png"
    plot_model(model, to_file=dot_img_file, show_shapes=True)
    model_card.add_chart("Model architecture", dot_img_file)

    x_test = np.load("data/x_test_ex.npy")
    y_test = np.load("data/y_test.npy")
    target_names = np.load("data/classes.npy")
    y_pred = model.predict(x_test, batch_size=64, verbose=1)

    report = classification_report(
        np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=target_names, output_dict=True
    )
    plot_classification_report(report, target_names, "ser/data/classification_report.png")
    model_card.add_chart("Classification Report", "data/classification_report.png")

    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    plot_confusion_matrix(matrix, target_names, "ser/data/confusion_matrix.png")
    model_card.add_chart("Confusion Matrix", "data/confusion_matrix.png")

    # Get the Markdown string for the model card
    model_card.generate_md("ser_model_card.md")
