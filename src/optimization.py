import optuna
import numpy as np

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score


def get_scale_pos_weight(y):
    neg = np.sum(y == 0)
    pos = np.sum(y == 1)
    return neg / max(pos, 1)


def optimize_lightgbm(
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_trials=10,
    random_state=42,
):
    """
    Optimize LightGBM hyperparameters using Optuna.
    Objective: maximize PR-AUC on validation set.

    The search space is centered around the current strong baseline.
    """
    scale_pos_weight = get_scale_pos_weight(y_train)

    def objective(trial):
        params = {
            "objective": "binary",
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": -1,
            "boosting_type": "gbdt",
            "scale_pos_weight": scale_pos_weight,

            # Search around the current working baseline
            "n_estimators": trial.suggest_int("n_estimators", 600, 1400, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.06),

            "num_leaves": trial.suggest_int("num_leaves", 48, 128),
            "max_depth": trial.suggest_int("max_depth", 6, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 160),

            "subsample": trial.suggest_float("subsample", 0.75, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),

            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        }

        model = LGBMClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="average_precision",
            callbacks=[
                early_stopping(stopping_rounds=80, verbose=False),
                log_evaluation(period=0),
            ],
        )

        y_proba = model.predict_proba(X_valid)[:, 1]
        return average_precision_score(y_valid, y_proba)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params

    final_params = {
        "objective": "binary",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
        "boosting_type": "gbdt",
        "scale_pos_weight": scale_pos_weight,
        **best_params,
    }

    best_model = LGBMClassifier(**final_params)

    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=[
            early_stopping(stopping_rounds=80, verbose=False),
            log_evaluation(period=0),
        ],
    )

    return best_model, study