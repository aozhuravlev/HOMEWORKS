import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score


def precision_recall_cm(model, X, y):

    model_cm = confusion_matrix(X, y, labels=model.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=model_cm, display_labels=model.classes_
    )
    disp.plot()
    plt.show()

    print(
        f"{'Precision:':15} {precision_score(X, y):.2%}\n"
        f"{'Recall:':15} {recall_score(X, y):.2%}"
    )


def get_best_threshold(y_val, probs):
    values = []

    for threshold in np.arange(0.05, 1, 0.001):
        probs_val_classes = (probs[:, 0] < threshold).astype(int)
        precision = round(precision_score(y_val, probs_val_classes), 4)
        recall = round(recall_score(y_val, probs_val_classes), 4)
        if precision >= 0.15:
            values.append((threshold, precision, recall))

    if values:
        best_comb_by_recall = sorted(values, key=lambda x: x[2], reverse=True)[0]
        best_threshold, precision, max_recall = (
            best_comb_by_recall[0],
            best_comb_by_recall[1],
            best_comb_by_recall[2],
        )

    return [best_threshold, precision, max_recall]
