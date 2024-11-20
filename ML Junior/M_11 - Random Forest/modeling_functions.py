from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import r2_score




def tune_model(model, params, X_train, y_train):
    search = GridSearchCV(model, params, scoring="r2", cv=5, n_jobs=-1, verbose=2)
    search.fit(X_train, y_train)
    print(
        f"{'The best parameters for':20} {model.__class__.__name__}:\n{search.best_params_}\n"
    )
    return search.best_estimator_


def get_r2_score(models, splits):
    X_train, y_train, X_test, y_test = splits
    for model in models:
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        # y_pred = np.expm1(model.predict(X_test))
        r2 = r2_score(y_test, y_pred)
        print(f"R2 score for {type(model).__name__}: {r2:.2%}")    

def get_mae_score(models, splits):
    X_train, y_train, X_test, y_test = splits
    for model in models:
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        # y_pred = np.expm1(model.predict(X_test))
        mae = mean_absolute_error(y_test, y_pred)
        print(f"MAE score for {type(model).__name__}: {mae:.2f}")

def plot_feature_importances(model, X):
    importances = model.feature_importances_
    features = X.columns

    # Построим график
    plt.figure(figsize=(10, 6))
    pd.Series(importances, index=features).nlargest(10).plot(kind="barh")
    plt.title("Feature Importance")
    plt.show()
