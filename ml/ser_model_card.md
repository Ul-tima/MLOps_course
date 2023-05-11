# Speech Emotion Recognition Model Card

## Model Overview

The model is a Keras deep learning model trained to predict the emotion of a given speech sample.
## Intended Use

This model is intended to be used for classifying the emotion in speech samples in real-time. It can be integrated into a mobile or web application for user feedback, emotion recognition in customer service interactions, or for educational or therapeutic purposes.
## Limitations

This model may not perform well on speech samples from speakers with different age, gender, or language backgrounds than those represented in the training data. It may also struggle with recognizing emotions that are not well-represented in the training data.
## Datasets

The model was trained on three publicly available datasets: RAVDESS, CREMA-D, and SAVEE. These datasets contain a total of 9,362 speech samples across 7 emotional categories.
## Training Information

The model was trained on a dataset of speech samples labeled with the corresponding emotion. The dataset was splited into 80% training, 10% validation, and 10% testing sets. The model was trained using the categorical cross-entropy loss function and the Adam optimizer with default hyperparameters for 30 epochs. The batch size was set to 64.
## Model Information

The model is a convolutional neural network (CNN) for speech emotion recognition. It consists of two convolutional layers with max pooling and batch normalization, followed by three fully connected layers with dropout and a softmax output layer. The model is compiled with the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric.
## Model architecture

![Model architecture](data/model.png)

## Classification Report

![Classification Report](data/classification_report.png)

## Confusion Matrix

![Confusion Matrix](data/confusion_matrix.png)
