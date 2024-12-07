import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd


def get_cv_rmse(model, X, y):
    rmse = (
        cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error").mean()
        * -1
    )
    # print(f"CV RMSE: {rmse}")
    mape = rmse / y.mean()
    info = {"CV RMSE": rmse, "MAPE": mape}
    for k, v in info.items():
        print(f"{k:10}: {v:.2f}")


def get_rmse_score(models, X, y):
    for model in models:
        model.fit(X, y)
        rmse = (
            cross_val_score(
                model, X, y, cv=5, scoring="neg_root_mean_squared_error"
            ).mean()
            * -1
        )
        print(f"RMSE score for {type(model).__name__}: {rmse:.2f}")


def get_r2_score(models, splits):
    X_train, y_train, X_test, y_test = splits
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # y_pred = np.expm1(model.predict(X_test))
        r2 = r2_score(y_test, y_pred)
        print(f"R2 score for {type(model).__name__}: {r2:.2%}")


def show_box_plot(df):
    for feat in df.columns:
        plt.figure(figsize=(10, 1.5))  # Уменьшаем размер графика
        sns.boxplot(data=df[feat], orient="h")  # Горизонтальная ориентация
        plt.title(feat)
        # plt.tight_layout()
        plt.show()


def show_pair_plot(df):
    # Заменяем все значения inf на NaN
    df = df.replace([float("inf"), -float("inf")], pd.NA)

    # Строим pairplot
    sns.pairplot(df, hue="target", palette="coolwarm")
    plt.show()


def tune_model(model, params, X_train, y_train):
    search = GridSearchCV(
        model, params, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1, verbose=2
    )
    search.fit(X_train, y_train)
    print(
        f"{'The best parameters for':20} {model.__class__.__name__}:\n{search.best_params_}\n"
    )
    return search.best_estimator_


def engineer_features(input_df):
    X = input_df.copy()

    X["CWR"] = (X["C"] / X["W"]) 
    X["SPWR"] = X["SP"] + 0.0001 / X["W"]
    X["C^"] = X["C"] ** 2
    X["SP^"] = X["SP"] ** 2
    X["BFS^"] = X["BFS"] ** 2
    X["FA^"] = X["FA"] ** 2
    # X["Ag_Af"] = X["Ag"] / X["Af"]
    # X["NormC"] = X["C"] / X["t"]
    # X["NormFA"] = X["FA"] / X["t"]
    # X["NormW"] = X["W"] / X["t"]


    cols_to_drop = [
        "C",
        "W",
        # "t",
        # "SP",
        # "Af",
        # "Ag",
        "BFS",
        "FA",
    ]

    return X.drop(columns=cols_to_drop)

    # X = X.drop(['target'], axis=1)

    # X["Total"] = (
    #     X["C"]
    #     + X["BFS"]
    #     + X["FA"]
    #     + X["W"]
    #     + X["SP"]
    #     + X["Ag"]
    #     + X["Af"]
    # )
    # X["CementRatio"] = X["Cement"] / X["TotalComponents"]

    # X["BFSFAR"] = (X["BFS"] + .0001) / (X["FA"] + .0001)
    # X['SPCR'] = X['SP'] / X['C']
    # X['AfAg'] = X['Af'] / X['Ag']
    # X['BFS+'] = (X['BFS'] > 0).astype(int)
    # X['FA+'] = (X['FA'] > 0).astype(int)
    # X['SP+'] = (X['SP'] > 0).astype(int)

    # X['LogAge'] = np.log1p(X['Age'])
    # X['RootAge'] = np.sqrt(X['Age'])
    # X["CatAge"] = pd.cut(
    #     X["Age"],
    #     bins=[0, 3, 7, 14, 28, 62, 90, 110, 185, 280, 370],
    #     labels=[1, 2, 3, 4, 5, 6, 5, 4, 5, 5],
    #     ordered=False,
    # ).astype(int)

    # X["CementWaterInteraction"] = X["Cement"] * X["Water"]
    # X["CoarseFineInteraction"] = X["Ag"] * X["Af"]
    # X["AgeCementInteraction"] = X["CatAge"] * X["Cement"]
    # # X['AgeFlyAshInteraction'] = X['Age'] * X['FlyAsh']
    # X["AgeWaterInteraction"] = X["CatAge"] * X["Water"]
    # X["AgeSuperplasticizerInteraction"] = X["CatAge"] * X["Superplasticizer"]
    # X['AgeAgInteraction'] = X['Age'] * X['Ag']
    # X['AgeAfInteraction'] = X['Age'] * X['Af']
    # X['AgeBlastFurnaceSlagInteraction'] = X['Age'] * X['BlastFurnaceSlag']

    # maxabs = MaxAbsScaler()
    # cols_to_scale = [
    #     "Cement^2",
    #     "Af",
    #     "Ag",
    # "AgeCementInteraction",
    # "AgeFlyAshInteraction",
    # "AgeWaterInteraction",
    # "AgeSuperplasticizerInteraction",
    # "AgeAgInteraction",
    # "AgeAfInteraction",
    # "AgeBlastFurnaceSlagInteraction",
    # ]
    # X[cols_to_scale] = maxabs.fit_transform(X[cols_to_scale])

    # X["Log_BlastFurnaceSlag"] = np.log1p(X["BlastFurnaceSlag"])
    # X["Log_FlyAsh"] = np.log1p(X["FlyAsh"])
    # X["Log_Superplasticizer"] = np.log1p(X["Superplasticizer"])

    # X["NormalizedCement"] = X["C"] / X["t"]
    # X["NormalizedFlyAsh"] = X["FA"] / X["t"]
    # X["NormalizedWater"] = X["W"] / X["t"]

    # total = X["C"] + X["BFS"] + X["FA"] + X["W"] + X["SP"] + X["Ag"] + X["Af"]
    # X["CR"] = X["C"] / total
    # X["BFSR"] = X["BFS"] / X["Total"]

    # X["BFSFAR"] = (X["BFS"] + .0001) / (X["FA"] + .0001)
    # X['SPCR'] = X['SP'] + .0001 / X['W']
    # X["AfAg_tot"] = (X["Af"] + X["Ag"]) / total
    # X["R"] = k1 * ((X["C"] + k5 * X["BFS"] + k6 * X["FA"]) / (X["W"] + k2 * X["SP"])) / (1 + k3 * X["Ag"] + k4 * X["Af"]) * ((X["t"] / X["t"] + alpha) * np.log1p(X["t"]))
    # X["R"] = ((X["C"] + X["BFS"] + X["FA"]) / (X["W"] + X["SP"])) * ((1 + X["Ag"] + X["Af"])**-1)
    # X["t"] = np.log1p(X["t"]) * .7
