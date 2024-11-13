def fill_with_mode(input_df):
    df = input_df.copy()
    for i in df.columns:
        df[i] = df[i].fillna(df[i].mode()[0])

    return df


def show_corr_matrix(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.show()


def fit_models(models, splits):
    X_train, y_train, _, _ = splits
    fit_models = [model.fit(X_train, y_train) for model in models]
    return fit_models

def log_and_print(data):
    import datetime

    with open("results.log", "a") as f:
        time = str(datetime.datetime.now())
        delimiter = f"\n{'=' *30}\n"
        reason = input("Please, enter the reason for change:")
        f.write(f"{delimiter}{time}\n{reason}{delimiter}")
        for key, value in data.items():
            f.write(f"{key:20}: {value:.2%}\n")
            print(f"{key:20}: {value:.2%}")


def get_f1_score(model, split_list):
    from sklearn.metrics import f1_score

    _, _, X_test, y_test = split_list
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average="macro")



def evaluate_models(models, splits):
    data = {type(model).__name__: get_f1_score(model, splits) for model in models}
    log_and_print(data)


def tune_model(model, params, splits):
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import f1_score

    f1_scorer = make_scorer(f1_score)
    X_train, y_train, _, _ = splits

    grid = GridSearchCV(model, params, scoring=f1_scorer, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(
        f"{'The best parameters for':20} {type(model).__name__}:\n{grid.best_params_}\n"
    )
    return grid.best_estimator_


def get_svm_poly(splits):
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.preprocessing import StandardScaler

    X_train, y_train, _, _ = splits
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=3)),
            ("scaler", StandardScaler()),
            (
                "linear",
                LinearSVC(C=1, class_weight="balanced", loss="hinge", random_state=137),
            ),
        ]
    )

    return model.fit(X_train, y_train)


def get_kernel_trick(splits):
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    X_train, y_train, _, _ = splits
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="poly",
                    degree=5,
                    coef0=1,
                    C=1,
                    class_weight="balanced",
                    random_state=137,
                ),
            ),
        ]
    )

    return model.fit(X_train, y_train)
