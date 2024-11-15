import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.feature_selection import RFE



def log_and_print(func):

    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)

        with open("results.log", "a") as f:
            time = str(datetime.datetime.now())
            delimiter = f"\n{'=' * 40}\n"
            reason = input("Please, enter the reason for change: ")
            f.write(f"{delimiter}{time}\n{reason}{delimiter}")

            for key, value in res.items():
                f.write(f"{key:25}: {value:.2%}\n")
                print(f"{key:25}: {value:.2%}")

    return wrapper


@log_and_print
def get_roc_auc_score(models, splits):

    _, X_test, _, y_test = splits

    data = {
        type(model).__name__: roc_auc_score(y_test, model.predict(X_test))
        for model in models
    }
    return data


def tune_model(model, params, splits):

    X_train, _, y_train, _ = splits

    search = GridSearchCV(model, params, scoring="roc_auc", cv=5, n_jobs=-1, verbose=2)
    search.fit(X_train, y_train)
    print(
        f"{'The best parameters for':20} {type(model).__name__}:\n{search.best_params_}\n"
    )
    return search.best_estimator_


def plot_roc_curve(model, splits):

    _, X_test, _, y_test = splits
    # Предсказания вероятностей для положительного класса
    y_proba = model.predict_proba(X_test)[:, 1]

    # Построение ROC-кривой
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_feature_importances(models, X):
    feature_names = X.columns

    for model in models:
        plt.figure(figsize=(6, 4))

        if hasattr(model, "coef_"):
            # Для линейных моделей
            importances = np.abs(model.coef_.flatten())
        elif hasattr(model, "feature_importances_"):
            # Для деревьев и ансамблей
            importances = model.feature_importances_
        else:
            continue

        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(
            range(len(indices)),
            feature_names[indices],
            rotation=36,
            ha="right",
            fontsize=6,
        )
        plt.title(f"Feature Importance - {type(model).__name__}")
        plt.tight_layout()
        plt.show()
