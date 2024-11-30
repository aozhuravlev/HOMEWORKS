import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


def print_and_log(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        method, score = result
        string = f"ROC-AUC with {method:26}: {score:.2%}"
        print(string)
        with open("results.txt", "a") as f:
            f.write(f"{string}\n")
        return result

    return wrapper


@print_and_log
def get_roc_auc_cb(X_train, y_train, X_test, y_test, method):
    cb = CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.1,
        l2_leaf_reg=3,
        bagging_temperature=0.8,
        random_strength=0.2,
        logging_level="Silent",
        random_seed=137,
        eval_metric="AUC",
    )
    with open("results.txt", "a") as f:
        f.write(f"\n{type(cb).__name__}\n")
    cb.fit(X_train, y_train)
    y_pred = cb.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred)
    return method, score


@print_and_log
def get_roc_auc_special(model, X_train, y_train, X_test, y_test, method):
    with open("results.txt", "a") as f:
        f.write(f"\n{type(model).__name__}\n")
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred)
    return method, score


@print_and_log
def get_roc_auc_xgb(X_train, y_train, X_test, y_test, method):
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=137,
    )
    with open("results.txt", "a") as f:
        f.write(f"\n{type(xgb).__name__}\n")
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred)
    return method, score
