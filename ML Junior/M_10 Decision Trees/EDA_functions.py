import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def show_correlation_heatmap(df):

    plt.figure(figsize=(10, 8))  # Увеличим размер графика для лучшей читаемости

    # Построим heatmap с параметрами
    ax = sns.heatmap(
        df.corr(), cmap="vlag", annot=True, fmt=".2f", annot_kws={"size": 6}
    )

    # Уменьшение шрифта меток на осях и поворот под углом
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    # Фильтруем текст корреляции по значению
    for t in ax.texts:
        if abs(float(t.get_text())) < 0.3:
            t.set_text("")  # Оставляем только значения с модулем >= 0.3

    plt.tight_layout()  # Автоматическая подгонка элементов для избежания наложения
    plt.show()


def show_pair_plot(df):

    # Заменяем все значения inf на NaN
    df = df.replace([float("inf"), -float("inf")], pd.NA)

    # Строим pairplot
    sns.pairplot(df, hue="target", palette="coolwarm")
    plt.show()
