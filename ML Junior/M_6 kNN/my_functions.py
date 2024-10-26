import math
import seaborn as sns
from matplotlib import pyplot as plt
import inspect
import feat_eng_functions
from sqlalchemy import create_engine
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


class Model:

    def get_model_values(df_):

        def categorize(df_):
            df = df_.copy()
            categories = [
                i for i in df.columns if df[i].nunique() <= 4 and i != "target"
            ]
            data_to_encode = df[categories]
            ohe = OneHotEncoder(sparse_output=False)
            ohe_encoded = ohe.fit_transform(data_to_encode)
            df_categorical = pd.DataFrame(
                ohe_encoded, columns=ohe.get_feature_names_out()
            )
            df = pd.concat(
                [df.reset_index(drop=True), df_categorical.reset_index(drop=True)],
                axis=1,
                ignore_index=False,
            )
            return df.drop(columns=categories)

        df = df_.copy()
        df = categorize(df)

        X = df.drop(columns=["target"])
        y = df.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=137
        )

        knn = KNeighborsClassifier(n_neighbors=9)
        knn.fit(X_train, y_train)

        y_train_predict = knn.predict(X_train)
        y_test_predict = knn.predict(X_test)

        knn_cm = confusion_matrix(y_test, y_test_predict, labels=knn.classes_)
        knn_precision_train = round(precision_score(y_train, y_train_predict), 4)
        knn_precision_test = round(precision_score(y_test, y_test_predict), 4)
        knn_recall_train = round(recall_score(y_train, y_train_predict), 4)
        knn_recall_test = round(recall_score(y_test, y_test_predict), 4)
        knn_f1_train = round(f1_score(y_train, y_train_predict), 4)
        knn_f1_test = round(f1_score(y_test, y_test_predict), 4)
        print(
            f"precision:\n\ttrain: {knn_precision_train:.2%},\ttest: {knn_precision_test:.2%}\n"
            f"recall:\n\ttrain: {knn_recall_train:.2%},\ttest: {knn_recall_test:.2%}\n"
            f"f1:\n\ttrain: {knn_f1_train:.2%},\ttest: {knn_f1_test:.2%}\n\n"
            f"Confusion matrix on test:\n"
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=knn_cm, display_labels=knn.classes_
        )
        disp.plot()
        plt.show()


class knn_HW_functions:

    def get_my_df(from_text):

        def get_queries(txt):
            cols = [i for i in txt.split() if i.isupper()]
            cols1 = ", ".join(cols[:-2])
            with open("data/sql_queries.txt", "r") as file:
                queries = (
                    file.read()
                    .replace("\n", " ")
                    .format(cols1, cols[-2], cols[-1])
                    .split(";")
                )
            return queries[0], queries[1]

        connection_string = "postgresql://postgres:Kbyerc@localhost:5432/module_6"
        engine = create_engine(connection_string)
        query_personal, query_loan = get_queries(from_text)

        df_personal = pd.read_sql(query_personal, con=engine)
        df_loan = pd.read_sql(query_loan, con=engine)

        df = pd.merge(left=df_personal, right=df_loan, on="id_client", how="left")
        return df

    def engineer_features(df_):

        def get_function_list():
            functions_list = [
                name
                for name, _ in inspect.getmembers(
                    feat_eng_functions, inspect.isfunction
                )
            ]
            return functions_list

        df = df_.copy()
        for function_name in get_function_list():
            df[function_name] = df.apply(
                getattr(feat_eng_functions, function_name), axis=1
            )
            cols_to_drop = [
                "agreement_rk",
                "dependants",
                "loan_num_total",
                "loan_num_closed",
                "socstatus_work_fl",
                "socstatus_pens_fl",
            ]

        return df.drop(columns=cols_to_drop)

    def multiple_scatter_plot(df, target):
        cols = [
            i
            for i in df.drop(columns=target).columns
            if df[i].dtype in ["float64", "int64"]
        ]
        n_cols = 3
        n_rows = math.ceil(len(cols) / n_cols)

        # Создадим сетку графиков
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 5))

        # Построим графики
        for i, col in enumerate(cols):
            row = i // n_cols
            col_idx = i % n_cols
            plot_name = f"{target} dependency on {col}"
            sns.scatterplot(x=target, y=col, data=df, ax=axes[row, col_idx])
            axes[row, col_idx].set_title(plot_name)
            axes[row, col_idx].tick_params(axis="x", rotation=60)  # Поворот меток оси x

        # Отключим пустые оси, если есть
        for j in range(i + 1, n_rows * n_cols):
            fig.delaxes(axes.flatten()[j])

        plt.tight_layout()
        plt.show()

    def show_corr_matrix(df):
        # Матрица корреляций для числовых признаков
        corr_matrix = df.corr()

        # Визуализация матрицы корреляций
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.show()
