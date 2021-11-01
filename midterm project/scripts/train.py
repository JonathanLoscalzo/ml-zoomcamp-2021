#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from toolz import partial
import pickle
import optuna

from xgboost import XGBClassifier


def rename_columns(path: str = "../data/raw/heart.csv"):  # set columns as constants
    NUMERICAL_COLUMNS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    CATEGORICAL_COLUMNS = [
        "FastingBS",
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope",
    ]
    TARGET_COLUMN = "HeartDisease"
    data = pd.read_csv(path)
    data = data[CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + [TARGET_COLUMN]]
    data.columns = [
        "fasting_bs",
        "sex",
        "chest_pain_type",
        "resting_ecg",
        "exercise_angina",
        "st_slope",
        "age",
        "resting_bp",
        "cholesterol",
        "max_heart_rate",
        "old_peak",
        "heart_disease",
    ]
    data.to_csv("../data/heart.csv", index=False)
    return data


def reconstruct(transformed_data, preprocessor):
    return pd.DataFrame(
        transformed_data,
        columns=preprocessor.transformers_[0][1].get_feature_names_out().tolist()
        + NUMERICAL_COLUMNS,
    )


def preprocess(X, preprocessor, fit=False):
    if fit:
        preprocessor.fit(X)
    return reconstruct(preprocessor.transform(X), preprocessor), preprocessor


CATEGORICAL_COLUMNS = [
    "fasting_bs",
    "sex",
    "chest_pain_type",
    "resting_ecg",
    "exercise_angina",
    "st_slope",
]
NUMERICAL_COLUMNS = ["cholesterol", "age", "resting_bp", "max_heart_rate", "old_peak"]
TARGET_COLUMN = "heart_disease"
SEED = 42


def main(
    PREPROCESSOR_FILE="../data/models/preprocessor.bin",
    LR_FILE="../data/models/lr.bin",
    RF_FILE="../data/models/rf.bin",
    XGB_FILE="../data/models/xgb.bin",
):

    # get data
    data = rename_columns()

    # split test-train-val sets
    X = data[CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS]
    y = data[TARGET_COLUMN]
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train, y_full_train, test_size=0.25
    )

    # define preprocessing step
    onehot_preprocessor = OneHotEncoder(handle_unknown="ignore")
    cholesterol_imputer = Pipeline(
        steps=[("imputer", SimpleImputer(missing_values=0, strategy="mean"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", onehot_preprocessor, CATEGORICAL_COLUMNS),
            ("cholesterol_processor", cholesterol_imputer, ["cholesterol"]),
        ],
        remainder="passthrough",
    )

    # preprocess for training
    X_train, preprocessor = preprocess(X_train, preprocessor, fit=True)
    X_val, _preprocessor = preprocess(X_val, preprocessor)

    lr_best_params = evaluate_lr(X_train, y_train, X_val, y_val)
    rf_best_params = evaluate_rf(X_train, y_train, X_val, y_val)
    xgb_best_params = evaluate_xgb(X_train, y_train, X_val, y_val)

    # refit preprocessor
    X_full_train, preprocessor = preprocess(X_full_train, preprocessor, fit=True)
    X_test, _preprocessor = preprocess(X_test, preprocessor)

    pickle.dump(preprocessor, open(PREPROCESSOR_FILE, "wb"))

    ## TODO: put these lines on a foreach
    # LR retrain
    model = LogisticRegression(random_state=SEED, **lr_best_params)
    model.fit(X_full_train, y_full_train)
    print("LR params", lr_best_params)
    print(
        "LR Accuracy (train, test):",
        accuracy_score(y_full_train, model.predict(X_full_train)),
        accuracy_score(y_test, model.predict(X_test)),
    )
    pickle.dump(model, open(LR_FILE, "wb"))

    # RF retraing
    model = RandomForestClassifier(random_state=SEED, **rf_best_params)
    model.fit(X_full_train, y_full_train)
    print("RF params", rf_best_params)
    print(
        "RF Accuracy (train, test):",
        accuracy_score(y_full_train, model.predict(X_full_train)),
        accuracy_score(y_test, model.predict(X_test)),
    )
    pickle.dump(model, open(RF_FILE, "wb"))

    # XGB retrain
    model = XGBClassifier(
        **xgb_best_params, **{"eval_metric": accuracy_score, "verbosity": 0}
    )
    model.fit(
        X_full_train,
        y_full_train,
        eval_set=[(X_full_train, y_full_train), (X_test, y_test)],
    )
    print("XGB params", xgb_best_params)
    print(
        "XGBClassifier Accuracy (train, test):",
        accuracy_score(y_full_train, model.predict(X_full_train)),
        accuracy_score(y_test, model.predict(X_test)),
    )

    pickle.dump(model, open(XGB_FILE, "wb"))


def evaluate_lr(X_train, y_train, X_val, y_val):
    def objective(trial, X_train, y_train, X_val, y_val):
        C = trial.suggest_loguniform("C", 1e-7, 10.0)
        solver = trial.suggest_categorical("solver", ("lbfgs", "saga"))

        clf = LogisticRegression(C=C, solver=solver)

        clf.fit(X_train, y_train)

        return accuracy_score(y_val, clf.predict(X_val))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        partial(objective, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val),
        n_trials=10,
    )

    return study.best_params


def evaluate_rf(X_train, y_train, X_val, y_val):
    def objective_rf(trial, X_train, y_train, X_val, y_val):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 150),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
        }

        model = RandomForestClassifier(random_state=SEED, **params)

        model.fit(X_train, y_train)

        return accuracy_score(y_val, model.predict(X_val))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        partial(
            objective_rf, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val
        ),
        n_trials=20,
    )

    rf_best_params = study.best_params
    return rf_best_params


def evaluate_xgb(X_train, y_train, X_val, y_val):
    def objective_xgb(trial, X_train, y_train, X_val, y_val):

        param = {
            "eval_metric": accuracy_score,
            "early_stopping_rounds": 20,
            "verbosity": 0,
            # use exact for small dataset.
            "tree_method": "exact",
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 9, step=2),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        }

        model = XGBClassifier(**param)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
        )

        return accuracy_score(y_val, model.predict(X_val))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study_xgb = optuna.create_study(direction="maximize", sampler=sampler)
    study_xgb.optimize(
        partial(
            objective_xgb, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val
        ),
        n_trials=20,
    )
    xgb_best_params = study_xgb.best_params
    return xgb_best_params


if __name__ == "__main__":
    main()
