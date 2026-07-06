import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def get_scale_pos_weight(y):
    neg = np.sum(y == 0)
    pos = np.sum(y == 1)
    return neg / max(pos, 1)


def train_logistic_regression(X_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    scale_pos_weight = get_scale_pos_weight(y_train)

    model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
    )

    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    scale_pos_weight = get_scale_pos_weight(y_train)

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train):
    scale_pos_weight = get_scale_pos_weight(y_train)

    model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="PRAUC",
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=False,
    )

    model.fit(X_train, y_train)
    return model


def train_isolation_forest(X_train, y_train):
    """
    Train Isolation Forest mostly on normal transactions.
    Output will be anomaly score later.
    """
    X_normal = X_train[y_train == 0]

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_normal)
    return model


def get_model_proba(model, X):
    """
    Return positive class probability.
    """
    return model.predict_proba(X)[:, 1]


def get_isolation_score(model, X):
    """
    Convert IsolationForest score to anomaly probability-like score.
    Higher = more anomalous.
    """
    raw_score = -model.decision_function(X)

    min_score = raw_score.min()
    max_score = raw_score.max()

    if max_score == min_score:
        return np.zeros_like(raw_score)

    return (raw_score - min_score) / (max_score - min_score)