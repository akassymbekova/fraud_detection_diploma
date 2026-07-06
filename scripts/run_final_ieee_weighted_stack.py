import itertools
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

from config import (
    IEEE_TRANSACTION_PATH,
    IEEE_IDENTITY_PATH,
    TARGET_COL,
    TIME_COL_IEEE,
    TRAIN_RATIO,
    VALID_RATIO,
    OUTPUT_DIR,
    RANDOM_STATE,
)

from src.data_loader import load_ieee
from src.splitting import time_train_valid_test_split
from src.features import prepare_feature_engineering, add_velocity_features_searchsorted
from src.preprocessing import prepare_basic_features
from src.metrics import evaluate_binary_classifier, results_to_dataframe


def get_scale_pos_weight(y):
    neg = np.sum(y == 0)
    pos = np.sum(y == 1)
    return neg / max(pos, 1)


def make_recency_weights(n_old, n_recent, recent_weight=3.0):
    return np.concatenate([
        np.ones(n_old),
        np.ones(n_recent) * recent_weight,
    ])


def normalize_0_1(x):
    x = np.asarray(x)
    min_x = x.min()
    max_x = x.max()
    if max_x == min_x:
        return np.zeros_like(x)
    return (x - min_x) / (max_x - min_x)


def add_pair_frequency_features_train_test(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    pair_cols = [
        ("card1", "addr1"),
        ("card1", "P_emaildomain"),
        ("card1", "R_emaildomain"),
        ("card1", "DeviceInfo"),
        ("card1", "ProductCD"),
        ("card2", "addr1"),
        ("addr1", "P_emaildomain"),
        ("ProductCD", "P_emaildomain"),
    ]

    for col1, col2 in pair_cols:
        if col1 not in train_df.columns or col2 not in train_df.columns:
            continue

        feature_name = f"{col1}_{col2}_pair_freq"

        train_pair = train_df[col1].astype(str) + "_" + train_df[col2].astype(str)
        test_pair = test_df[col1].astype(str) + "_" + test_df[col2].astype(str)

        freq_map = train_pair.value_counts(dropna=False).to_dict()

        train_df[feature_name] = train_pair.map(freq_map).fillna(0)
        test_df[feature_name] = test_pair.map(freq_map).fillna(0)

    return train_df, test_df


def add_pair_amount_aggregates_train_test(train_df, test_df, amount_col="TransactionAmt"):
    train_df = train_df.copy()
    test_df = test_df.copy()

    pair_cols = [
        ("card1", "addr1"),
        ("card1", "ProductCD"),
        ("card2", "addr1"),
        ("addr1", "ProductCD"),
    ]

    for col1, col2 in pair_cols:
        if col1 not in train_df.columns or col2 not in train_df.columns:
            continue

        pair_name = f"{col1}_{col2}"

        train_key = train_df[col1].astype(str) + "_" + train_df[col2].astype(str)
        test_key = test_df[col1].astype(str) + "_" + test_df[col2].astype(str)

        temp_col = f"{pair_name}_key_temp"

        train_df[temp_col] = train_key
        test_df[temp_col] = test_key

        grouped = train_df.groupby(temp_col)[amount_col].agg(
            ["count", "mean", "std", "max"]
        ).fillna(0)

        for stat in ["count", "mean", "std", "max"]:
            feature_name = f"{pair_name}_{amount_col}_{stat}"
            train_df[feature_name] = train_key.map(grouped[stat]).fillna(0)
            test_df[feature_name] = test_key.map(grouped[stat]).fillna(0)

        mean_feature = f"{pair_name}_{amount_col}_mean"
        ratio_feature = f"{pair_name}_{amount_col}_to_mean_ratio"

        train_df[ratio_feature] = train_df[amount_col] / (train_df[mean_feature] + 1e-6)
        test_df[ratio_feature] = test_df[amount_col] / (test_df[mean_feature] + 1e-6)

        train_df = train_df.drop(columns=[temp_col])
        test_df = test_df.drop(columns=[temp_col])

    return train_df, test_df


def apply_extra_features(train_df, test_df):
    train_df, test_df = add_pair_frequency_features_train_test(train_df, test_df)

    train_df, test_df = add_pair_amount_aggregates_train_test(
        train_df,
        test_df,
        amount_col="TransactionAmt",
    )

    train_df, test_df, _unused = add_velocity_features_searchsorted(
        train_df,
        test_df,
        test_df.copy(),
        group_col="card2",
        time_col=TIME_COL_IEEE,
        amount_col="TransactionAmt",
        windows_hours=[1, 6, 24],
    )

    train_df, test_df, _unused = add_velocity_features_searchsorted(
        train_df,
        test_df,
        test_df.copy(),
        group_col="addr1",
        time_col=TIME_COL_IEEE,
        amount_col="TransactionAmt",
        windows_hours=[1, 6, 24],
    )

    return train_df, test_df


def build_models(y_train, seed_offset=0):
    scale_pos_weight = get_scale_pos_weight(y_train)

    return {
        "lgbm_a": LGBMClassifier(
            objective="binary",
            n_estimators=1400,
            learning_rate=0.022,
            num_leaves=112,
            min_child_samples=75,
            subsample=0.84,
            subsample_freq=1,
            colsample_bytree=0.84,
            reg_alpha=0.05,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=404 + seed_offset,
            n_jobs=-1,
            verbosity=-1,
        ),
        "lgbm_b": LGBMClassifier(
            objective="binary",
            n_estimators=1300,
            learning_rate=0.023,
            num_leaves=96,
            min_child_samples=70,
            subsample=0.88,
            subsample_freq=1,
            colsample_bytree=0.88,
            reg_alpha=0.08,
            reg_lambda=1.2,
            scale_pos_weight=scale_pos_weight,
            random_state=101 + seed_offset,
            n_jobs=-1,
            verbosity=-1,
        ),
        "lgbm_c": LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.024,
            num_leaves=96,
            min_child_samples=85,
            subsample=0.86,
            subsample_freq=1,
            colsample_bytree=0.86,
            reg_alpha=0.10,
            reg_lambda=1.5,
            scale_pos_weight=scale_pos_weight,
            random_state=303 + seed_offset,
            n_jobs=-1,
            verbosity=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=900,
            learning_rate=0.025,
            max_depth=6,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE + seed_offset,
            n_jobs=-1,
        ),
        "cat": CatBoostClassifier(
            iterations=900,
            learning_rate=0.025,
            depth=6,
            loss_function="Logloss",
            eval_metric="PRAUC",
            scale_pos_weight=scale_pos_weight,
            random_seed=RANDOM_STATE + seed_offset,
            verbose=False,
        ),
    }


def fit_predict_models(models, X_train, y_train, X_pred, sample_weight=None):
    preds = {}

    for name, model in models.items():
        print(f"Training {name}...")

        if sample_weight is not None and name.startswith("lgbm"):
            model.fit(X_train, y_train, sample_weight=sample_weight)
        elif sample_weight is not None and name.startswith("xgb"):
            model.fit(X_train, y_train, sample_weight=sample_weight)
        elif sample_weight is not None and name.startswith("cat"):
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)

        preds[name] = model.predict_proba(X_pred)[:, 1]

    return preds


def search_blend_weights(preds_valid, y_valid):
    """
    Search weights on inner validation only.
    Coarse grid, not test.
    """
    model_names = list(preds_valid.keys())

    best_score = -1
    best_weights = None

    # We deliberately keep grid small to avoid overfitting inner_valid.
    grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]

    for weights in itertools.product(grid, repeat=len(model_names)):
        weights = np.array(weights, dtype=float)

        if weights.sum() == 0:
            continue

        weights = weights / weights.sum()

        # Encourage LightGBM-dominant ensembles, since XGB/Cat are weaker.
        lgbm_weight_sum = sum(
            w for w, name in zip(weights, model_names)
            if name.startswith("lgbm")
        )

        if lgbm_weight_sum < 0.70:
            continue

        blend = np.zeros_like(next(iter(preds_valid.values())))

        for w, name in zip(weights, model_names):
            blend += w * normalize_0_1(preds_valid[name])

        score = average_precision_score(y_valid, blend)

        if score > best_score:
            best_score = score
            best_weights = dict(zip(model_names, weights))

    return best_score, best_weights


def apply_weights(preds, weights):
    blend = np.zeros_like(next(iter(preds.values())))

    for name, weight in weights.items():
        blend += weight * normalize_0_1(preds[name])

    return blend


def main():
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    print("Loading IEEE-CIS dataset...")
    df = load_ieee(IEEE_TRANSACTION_PATH, IEEE_IDENTITY_PATH)

    print("Creating original split...")
    train_df, valid_df, test_df = time_train_valid_test_split(
        df,
        time_col=TIME_COL_IEEE,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
    )

    final_train_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
    final_train_df = final_train_df.sort_values(TIME_COL_IEEE).reset_index(drop=True)

    print(f"Development data train+valid: {final_train_df.shape}")
    print(f"Future test: {test_df.shape}")

    inner_split_idx = int(len(final_train_df) * 0.80)

    inner_train_df = final_train_df.iloc[:inner_split_idx].copy()
    inner_valid_df = final_train_df.iloc[inner_split_idx:].copy()

    print(f"Inner train: {inner_train_df.shape}")
    print(f"Inner valid: {inner_valid_df.shape}")

    print("\n=== INNER VALIDATION PHASE ===")
    inner_train_fe, inner_valid_fe, _unused, _ = prepare_feature_engineering(
        inner_train_df,
        inner_valid_df.copy(),
        inner_valid_df.copy(),
    )

    inner_train_fe, inner_valid_fe = apply_extra_features(inner_train_fe, inner_valid_fe)

    X_inner_train, y_inner_train, X_inner_valid, y_inner_valid, _x_unused, _y_unused, _ = prepare_basic_features(
        inner_train_fe,
        inner_valid_fe,
        inner_valid_fe,
        target_col=TARGET_COL,
    )

    n_inner_old = int(len(X_inner_train) * 0.75)
    n_inner_recent = len(X_inner_train) - n_inner_old
    inner_weights = make_recency_weights(n_inner_old, n_inner_recent, recent_weight=3.0)

    inner_models = build_models(y_inner_train, seed_offset=0)

    inner_preds_valid = fit_predict_models(
        inner_models,
        X_inner_train,
        y_inner_train,
        X_inner_valid,
        sample_weight=inner_weights,
    )

    print("Searching blend weights on inner validation...")
    best_inner_score, best_weights = search_blend_weights(
        inner_preds_valid,
        y_inner_valid,
    )

    print("\nBest inner validation PR-AUC:", best_inner_score)
    print("Best weights:")
    for k, v in best_weights.items():
        print(f"{k}: {v:.4f}")

    print("\n=== FINAL TRAINING PHASE ===")
    final_train_fe, final_test_fe, _unused, _ = prepare_feature_engineering(
        final_train_df,
        test_df.copy(),
        test_df.copy(),
    )

    final_train_fe, final_test_fe = apply_extra_features(final_train_fe, final_test_fe)

    X_final_train, y_final_train, X_final_test, y_final_test, _x_unused, _y_unused, _ = prepare_basic_features(
        final_train_fe,
        final_test_fe,
        final_test_fe,
        target_col=TARGET_COL,
    )

    n_old = len(train_df)
    n_recent = len(valid_df)
    final_weights = make_recency_weights(n_old, n_recent, recent_weight=3.0)

    final_models = build_models(y_final_train, seed_offset=0)

    final_preds_test = fit_predict_models(
        final_models,
        X_final_train,
        y_final_train,
        X_final_test,
        sample_weight=final_weights,
    )

    final_weighted_blend = apply_weights(final_preds_test, best_weights)

    predictions = pd.DataFrame({
        "y_true": y_final_test.values,
        "Weighted_inner_valid_blend": final_weighted_blend,
    })

    for name, pred in final_preds_test.items():
        predictions[name] = pred

    results = {}

    results["Weighted_inner_valid_blend"] = evaluate_binary_classifier(
        y_final_test,
        final_weighted_blend,
        threshold=0.5,
    )

    for name, pred in final_preds_test.items():
        results[name] = evaluate_binary_classifier(
            y_final_test,
            pred,
            threshold=0.5,
        )

    results_df = results_to_dataframe(results)

    print("\nWeighted stacking/blending final test results:")
    print(results_df.to_string())

    print("\nFinal test PR-AUC weighted blend:", average_precision_score(y_final_test, final_weighted_blend))

    results_df.to_csv(
        OUTPUT_DIR / "reports" / "weighted_inner_valid_blend_test_results.csv"
    )

    predictions.to_csv(
        OUTPUT_DIR / "predictions" / "weighted_inner_valid_blend_test_predictions.csv",
        index=False,
    )

    weights_df = pd.DataFrame([
        {"model": k, "weight": v}
        for k, v in best_weights.items()
    ])

    weights_df.to_csv(
        OUTPUT_DIR / "reports" / "weighted_inner_valid_blend_weights.csv",
        index=False,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()