import numpy as np
from matplotlib import pyplot as plt


def plot_classification_report(report, target_names, filename=None):
    # extract precision, recall, f1-score for each emotion
    precision = []
    recall = []
    f1_score = []
    for emotion in target_names:
        precision.append(report[emotion]["precision"])
        recall.append(report[emotion]["recall"])
        f1_score.append(report[emotion]["f1-score"])

    # Plot the recall, precision, and F1-score as bar plot
    plt.figure(figsize=(10, 6))
    x = range(len(target_names))
    plt.bar(x, recall, width=0.25, color="tab:blue", align="center", label="Recall")
    plt.bar([i + 0.25 for i in x], precision, width=0.25, color="tab:green", align="center", label="Precision")
    plt.bar([i + 0.5 for i in x], f1_score, width=0.25, color="tab:red", align="center", label="F1-score")
    plt.xticks(x, target_names)
    plt.ylim([0, 1])
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Classification Report")
    plt.savefig(filename)


def plot_confusion_matrix(matrix, target_names, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(filename)
