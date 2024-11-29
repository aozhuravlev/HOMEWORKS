from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from hdbscan import HDBSCAN


def print_and_log(func):
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)

        b_params, b_score, eps, min_sv = res
        string = f"Best params: {b_params}\nBest score: {b_score:.2f}"
        print(string)
        # reason = input("Please, enter the reason for change: ")
        delimiter = f"\n{'=' * 40}\n"
        with open("log.txt", "a") as f:
            f.write(
                f"\n{delimiter}eps: {eps}\nmin_samples: {min_sv}{delimiter}{string}"
            )

        return res

    return wrapper


@print_and_log
def get_best_params_DBSCAN(X, eps_values, min_samples_values):
    best_score = -1
    best_params = {}

    for eps in eps_values:
        for min_samples in min_samples_values:
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)

            labels = clusterer.fit_predict(X)

            # Игнорируем конфигурации без кластеров
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_params = {"eps": eps, "min_samples": min_samples}

    return best_params, best_score, best_params["eps"], best_params["min_samples"]


def plot_best_params_DBSCAN(X, best_params):
    clusterer = DBSCAN(**best_params)
    labels = clusterer.fit_predict(X)

    plt.figure(figsize=(6, 3))
    sns.scatterplot(data=pd.DataFrame(X), x=0, y=1, hue=labels, palette="viridis", s=50)


def get_best_params_HDBSCAN(X, min_samples_values):
    best_score = -1
    best_params = {}

    for min_samples in min_samples_values:
        clusterer = HDBSCAN(min_samples=min_samples)

        labels = clusterer.fit_predict(X)

        # Игнорируем конфигурации без кластеров
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_params = {"min_samples": min_samples}

    print(f"Best params: {best_params}\nBest score: {best_score:.2f}")
    return best_params, best_score, labels


def plot_best_params_HDBSCAN(X, best_params):
    clusterer = HDBSCAN(**best_params)
    labels = clusterer.fit_predict(X)

    plt.figure(figsize=(6, 3))
    sns.scatterplot(data=pd.DataFrame(X), x=0, y=1, hue=labels, palette="viridis", s=50)
    plt.show()

    def plot_cluster_persistence(clusterer):
        # Метрики устойчивости
        cluster_persistence = clusterer.cluster_persistence_
        print(f"Устойчивость кластеров: {cluster_persistence}")

        # Визуализация устойчивости
        plt.figure(figsize=(6, 3))
        plt.bar(range(len(cluster_persistence)), cluster_persistence, color="skyblue")
        plt.xlabel("Кластер")
        plt.ylabel("Устойчивость")
        plt.title("Устойчивость кластеров")
        plt.show()

    plot_cluster_persistence(clusterer)
