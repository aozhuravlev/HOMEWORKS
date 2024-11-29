import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans


def show_correlation_heatmap(df):

    plt.figure(figsize=(10, 8))  # Увеличим размер графика для лучшей читаемости

    # Построим heatmap с параметрами
    ax = sns.heatmap(
        df.corr(), cmap="vlag", annot=True, fmt=".2f", annot_kws={"size": 8}
    )

    # Уменьшение шрифта меток на осях и поворот под углом
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    # Фильтруем текст корреляции по значению
    for t in ax.texts:
        if abs(float(t.get_text())) < 0.3:
            t.set_text("")  # Оставляем только значения с модулем >= 0.3

    plt.tight_layout()  # Автоматическая подгонка элементов для избежания наложения
    plt.show();


def show_pair_plot(df):

    # Заменяем все значения inf на NaN
    df = df.replace([float("inf"), -float("inf")], pd.NA)

    # Строим pairplot
    sns.pairplot(df, hue="target", palette="coolwarm")
    plt.show();


def show_box_plot(df, features):
    for feat in features:
        plt.figure(figsize=(10, 2))  # Уменьшаем размер графика
        sns.boxplot(data=df[feat], orient="h")  # Горизонтальная ориентация
        plt.title(feat)
        plt.tight_layout()
        plt.show();


from sklearn.metrics import silhouette_score


def get_best_k(X, feat_name):
    inertia = []
    k_values = range(2, 15)
    best_k, best_score = None, -1
    for k in k_values:
        k_means = KMeans(n_clusters=k)
        k_means = k_means.fit(X)
        inertia.append(k_means.inertia_)
        clusters = k_means.predict(X)
        score = np.round(silhouette_score(X=X, labels=clusters), 2)
        if score > best_score:
            best_score = score
            best_k = k
    print(f"Best score for {feat_name:6}: {best_score}, k = {best_k}")
    # plt.figure(figsize=(6, 3))
    # plt.plot(k_values, inertia, "-o")
    # plt.xlabel("Number of Clusters (k)")
    # plt.ylabel("Inertia")
    # plt.title(f"{feat_name}")
    # plt.show();