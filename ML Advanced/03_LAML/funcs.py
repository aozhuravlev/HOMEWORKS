import pandas as pd
import numpy as np

import pygame


from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer


TARGET_NAME = "SalePrice"


def get_num_features(df: pd.DataFrame) -> list:
    return [
        i
        for i in df.select_dtypes(include=["int64", "float64"]).columns
        if i != TARGET_NAME
    ]


def get_cat_features(df: pd.DataFrame) -> list:
    return [i for i in df.select_dtypes(include=["object"]).columns if i != TARGET_NAME]


def create_features(input_df: pd.DataFrame) -> pd.DataFrame:
    def create_aggregated_features(input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Создаёт 5 агрегированных фичей для датасета House Prices.

        Параметры:
        df (pd.DataFrame): Исходный DataFrame.

        Возвращает:
        pd.DataFrame: DataFrame с добавленными агрегированными фичами.
        """
        df = input_df.copy()
        # 1. Средняя площадь 1-го этажа по районам (Neighborhood)
        df["Neighborhood_1stFlrSF_Mean"] = df.groupby("Neighborhood")[
            "1stFlrSF"
        ].transform("mean")

        # 2. Общая площадь подвала по типу дома (MSSubClass)
        df["MSSubClass_TotalBsmtSF_Sum"] = df.groupby("MSSubClass")[
            "TotalBsmtSF"
        ].transform("sum")

        # 3. Среднее количество ванных комнат по году постройки (YearBuilt)
        df["YearBuilt_FullBath_Mean"] = df.groupby("YearBuilt")["FullBath"].transform(
            "mean"
        )

        # 4. Средняя цена за квадратный фут жилой площади по районам (Neighborhood)
        df["Neighborhood_PricePerSqFt"] = df["SalePrice"] / df["GrLivArea"]
        df["Neighborhood_PricePerSqFt_Mean"] = df.groupby("Neighborhood")[
            "Neighborhood_PricePerSqFt"
        ].transform("mean")

        # 5. Средний размер гаража по типу гаража (GarageType)
        df["GarageType_GarageArea_Mean"] = df.groupby("GarageType")[
            "GarageArea"
        ].transform("mean")

        return df.drop("Neighborhood_PricePerSqFt", axis=1)

    def create_transformed_features(input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Создаёт 5 трансформационных фичей для датасета House Prices.

        Параметры:
        df (pd.DataFrame): Исходный DataFrame.

        Возвращает:
        pd.DataFrame: DataFrame с добавленными трансформационными фичами.
        """

        df = input_df.copy()
        # 1. Взаимодействие: общее качество * общее состояние (OverallQual * OverallCond)
        df["OverallQual_x_OverallCond"] = df["OverallQual"] * df["OverallCond"]

        # 2. Разность: общее качество - общее состояние (OverallQual - OverallCond)
        df["OverallQual_minus_OverallCond"] = df["OverallQual"] - df["OverallCond"]

        # 3. Полиномиальная фича: квадрат общей площади подвала (TotalBsmtSF)
        df["TotalBsmtSF_Squared"] = df["TotalBsmtSF"] ** 2

        # 4. Взаимодействие: общая площадь * качество кухни (GrLivArea * KitchenQual)
        # Кодируем KitchenQual численно (например, Ex=5, Gd=4, TA=3, Fa=2, Po=1)
        kitchen_qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
        df["KitchenQual_Encoded"] = df["KitchenQual"].map(kitchen_qual_map)
        df["GrLivArea_KitchenQual"] = df["GrLivArea"] * df["KitchenQual_Encoded"]

        # 5. Отношение общей жилой площади к общей площади участка (GrLivArea / LotArea)
        df["GrLivArea_to_LotArea"] = df["GrLivArea"] / df["LotArea"]

        # Заменяем бесконечные значения на NaN (если LotArea = 0)
        df["GrLivArea_to_LotArea"].replace([np.inf, -np.inf], np.nan, inplace=True)

        return df.drop(["TotalBsmtSF"], axis=1)

    df = input_df.copy()
    df_aggregated = create_aggregated_features(df)
    df_transformed = create_transformed_features(df_aggregated)

    return df_transformed


def get_mae(automl, oof_pred, train_data, test_data):
    test_pred = automl.predict(test_data)

    print(
        f"MAE on train: {mean_absolute_error(train_data[TARGET_NAME].values, oof_pred.data[:, 0])}"
    )
    print(
        f"MAE on test: {mean_absolute_error(test_data[TARGET_NAME].values, test_pred.data[:, 0])}"
    )


def nan_processing(input_df):
    df = input_df.copy()
    knn = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(knn.fit_transform(df), columns=df.columns)
    return df


def playsound() -> None:
        pygame.init()
        pygame.mixer.music.load("../../../DIPLOMA/media/finished_.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

def target_log1p(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    return df.assign(SalePrice=np.log1p(df.SalePrice))