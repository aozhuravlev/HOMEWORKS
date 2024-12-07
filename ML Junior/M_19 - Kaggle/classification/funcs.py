import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd


def preprocess(input_df):
    df_prep = input_df.copy()
    # df_prep["native-country"] = (df_prep["native-country"] == "United-States").astype(
    #     int
    # )
    df_prep["gender"] = (df_prep["gender"] == "Male").astype(int)
    df_prep["marital-status"] = df_prep["marital-status"].apply(
        lambda x: 1 if x.startswith("Married") else (0 if x.startswith("Never") else 2)
    )

    # df_prep["workclass"] = df_prep["workclass"].apply(
    #     lambda x: (
    #         0
    #         if x == "Private"
    #         else (1 if x.startswith("Self") else (2 if "gov" in x else 3))
    #     )
    # )

    df_prep["education"] = df_prep["education"].map(
        {
            "HS-grad": 0,
            "Some-college": 0,
            "Bachelors": 0,
            "Masters": 0,
            "Assoc-voc": 0,
            "11th": 1,
            "Assoc-acdm": 0,
            "10th": 1,
            "7th-8th": 1,
            "Prof-school": 2,
            "9th": 1,
            "12th": 1,
            "Doctorate": 0,
            "5th-6th": 3,
            "1st-4th": 3,
            "Preschool": 3,
        }
    )

    df_prep["occupation"] = df_prep["occupation"].map(
        {
            "Prof-specialty": 0,
            "Craft-repair": 0,
            "Exec-managerial": 1,
            "Adm-clerical": 1,
            "Sales": 1,
            "Other-service": 0,
            "Machine-op-inspct": 0,
            "Transport-moving": 0,
            "Handlers-cleaners": 2,
            "Farming-fishing": 0,
            "Tech-support": 0,
            "Protective-serv": 3,
            "Priv-house-serv": 0,
            "Armed-Forces": 3,
            "?": 2,
        }
    )


    df_prep["relationship"] = df_prep["relationship"].map(
        {
            "Husband": 0,
            "Not-in-family": 1,
            "Wife": 2,
            "Own-child": 1,
            "Unmarried": 1,
            "Other-relative": 1,
        }
    )

    # df_prep["race"] = df_prep["race"].apply(
    #     lambda x: (
    #         0
    #         if x == "White"
    #         else (1 if x == "Black" else (2 if x == "Asian-Pac-Islander" else 3))
    #     )
    # )

    return df_prep
