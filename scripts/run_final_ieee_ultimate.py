import numpy as np
import pandas as pd

from scipy.stats import rankdata

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
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
from src.threshold_opt import find_best_threshold, cost_based_threshold


def get_scale_pos_weight(y):
    neg = np.sum(y == 0)
    pos = np.sum(y == 1)
    return neg / max(pos, 1)


def make_recency_weights(n_train_original, n_valid_original, recent_weight=3.0):
    old_weights = np.ones(n_train_original)
    recent_weights = np.ones(n_valid_original) * recent_weight
    return np.concatenate([old_weights, recent_weights])


def normalize_0_1(x):
    x = np.asarray(x)
    min_x = x.min()
    max_x = x.max()
    if max_x == min_x:
        return np.zeros_like(x)
    return (x - min_x) / (max_x - min_x)


def rank_average(probabilities):
    ranked = []
    for p in probabilities:
        r = rankdata(p) / len(p)
        ranked.append(r)
    return np.mean(np.vstack(ranked), axis=0)

def add_pair_frequency_features_train_test(train_df, test_df):
    """
    Adds leakage-free pair frequency features.
    Frequencies are fitted only on final_train and applied to test.
    """
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

        train_pair = (
            train_df[col1].astype(str)
            + "_"
            + train_df[col2].astype(str)
        )

        test_pair = (
            test_df[col1].astype(str)
            + "_"
            + test_df[col2].astype(str)
        )

        freq_map = train_pair.value_counts(dropna=False).to_dict()

        train_df[feature_name] = train_pair.map(freq_map).fillna(0)
        test_df[feature_name] = test_pair.map(freq_map).fillna(0)

    return train_df, test_df


def add_pair_amount_aggregates_train_test(train_df, test_df, amount_col="TransactionAmt"):
    """
    Adds leakage-free amount statistics for important pair identifiers.
    """
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

        train_df[f"{pair_name}_key_temp"] = train_key
        test_df[f"{pair_name}_key_temp"] = test_key

        grouped = train_df.groupby(f"{pair_name}_key_temp")[amount_col].agg(
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

        train_df = train_df.drop(columns=[f"{pair_name}_key_temp"])
        test_df = test_df.drop(columns=[f"{pair_name}_key_temp"])

    return train_df, test_df

def main():
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    print("Loading IEEE-CIS dataset...")
    df = load_ieee(IEEE_TRANSACTION_PATH, IEEE_IDENTITY_PATH)

    print("Creating time-based split...")
    train_df, valid_df, test_df = time_train_valid_test_split(
        df,
        time_col=TIME_COL_IEEE,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
    )

    print(f"Train: {train_df.shape}")
    print(f"Valid: {valid_df.shape}")
    print(f"Test: {test_df.shape}")

    final_train_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)

    n_train_original = len(train_df)
    n_valid_original = len(valid_df)

    print(f"Final train: {final_train_df.shape}")
    print(f"Final test: {test_df.shape}")

    print("Applying feature engineering...")
    final_train_fe, final_test_fe, _unused, feature_info = prepare_feature_engineering(
        final_train_df,
        test_df.copy(),
        test_df.copy(),
    )
    
    print("Adding final relational pair features...")

    final_train_fe, final_test_fe = add_pair_frequency_features_train_test(
        final_train_fe,
        final_test_fe,
    )

    final_train_fe, final_test_fe = add_pair_amount_aggregates_train_test(
        final_train_fe,
        final_test_fe,
        amount_col="TransactionAmt",
    )

    print("Adding extra velocity features for card2 and addr1...")

    final_train_fe, final_test_fe, _unused_extra = add_velocity_features_searchsorted(
        final_train_fe,
        final_test_fe,
        final_test_fe.copy(),
        group_col="card2",
        time_col=TIME_COL_IEEE,
        amount_col="TransactionAmt",
        windows_hours=[1, 6, 24, 48, 72],
    )

    final_train_fe, final_test_fe, _unused_extra = add_velocity_features_searchsorted(
        final_train_fe,
        final_test_fe,
        final_test_fe.copy(),
        group_col="addr1",
        time_col=TIME_COL_IEEE,
        amount_col="TransactionAmt",
        windows_hours=[1, 6, 24, 48, 72],
    )

    print(f"Final train after extra FE: {final_train_fe.shape}")
    print(f"Final test after extra FE: {final_test_fe.shape}")

    print("Preprocessing...")
    X_train, y_train, X_test_as_valid, y_test_as_valid, X_test, y_test, prep_info = prepare_basic_features(
        final_train_fe,
        final_test_fe,
        final_test_fe,
        target_col=TARGET_COL,
    )

    X_final_test = X_test_as_valid
    y_final_test = y_test_as_valid

    print(f"Prepared final X_train: {X_train.shape}")
    print(f"Prepared final X_test: {X_final_test.shape}")

    scale_pos_weight = get_scale_pos_weight(y_train)
    recency_weights_2 = make_recency_weights(n_train_original, n_valid_original, recent_weight=2.0)
    recency_weights_3 = make_recency_weights(n_train_original, n_valid_original, recent_weight=3.0)
    
    recency_weights_4 = make_recency_weights(
    n_train_original,
    n_valid_original,
    recent_weight=4.0,
    )

    models = {}

    print("Training LGBM recency weight 3...")
    models["LGBM_recency_weight_3"] = (
        LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.025,
            num_leaves=96,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        recency_weights_3,
    )

    print("Training LGBM ensemble variant...")
    models["LGBM_recency_weight_2"] = (
        LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.025,
            num_leaves=96,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE + 1,
            n_jobs=-1,
            verbosity=-1,
        ),
        recency_weights_2,
    )
    
    print("Training additional LGBM bagging variants...")

    models["LGBM_recency_weight_3_seed_101"] = (
        LGBMClassifier(
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
            random_state=101,
            n_jobs=-1,
            verbosity=-1,
        ),
        recency_weights_3,
    )

    models["LGBM_recency_weight_3_seed_202"] = (
        LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.026,
            num_leaves=80,
            min_child_samples=90,
            subsample=0.82,
            subsample_freq=1,
            colsample_bytree=0.90,
            reg_alpha=0.12,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=202,
            n_jobs=-1,
            verbosity=-1,
        ),
        recency_weights_3,
    )

    models["LGBM_recency_weight_4_seed_303"] = (
        LGBMClassifier(
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
            random_state=303,
            n_jobs=-1,
            verbosity=-1,
        ),
        recency_weights_4,
    )

    models["LGBM_recency_weight_2_seed_404"] = (
        LGBMClassifier(
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
            random_state=404,
            n_jobs=-1,
            verbosity=-1,
        ),
        recency_weights_2,
    )

    print("Training XGBoost final...")
    models["XGBoost_final"] = (
        XGBClassifier(
            n_estimators=900,
            learning_rate=0.025,
            max_depth=6,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        recency_weights_2,
    )

    print("Training XGBoost recall variant...")
    models["XGBoost_recall_variant"] = (
        XGBClassifier(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=5,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight * 1.2,
            random_state=RANDOM_STATE + 2,
            n_jobs=-1,
        ),
        recency_weights_2,
    )

    print("Training CatBoost final...")
    models["CatBoost_final"] = (
        CatBoostClassifier(
            iterations=900,
            learning_rate=0.025,
            depth=6,
            loss_function="Logloss",
            eval_metric="PRAUC",
            scale_pos_weight=scale_pos_weight,
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
        recency_weights_2,
    )

    print("Training Random Forest final...")
    models["RandomForest_final"] = (
        RandomForestClassifier(
            n_estimators=400,
            max_depth=14,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        None,
    )

    predictions = pd.DataFrame({"y_true": y_final_test.values})
    results = {}
    proba_dict = {}

    print("Fitting and evaluating models...")

    for name, (model, sample_weight) in models.items():
        print(f"Fitting {name}...")

        if sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)

        proba = model.predict_proba(X_final_test)[:, 1]
        predictions[name] = proba
        proba_dict[name] = proba

        results[name] = evaluate_binary_classifier(
            y_final_test,
            proba,
            threshold=0.5,
        )

        print(f"{name} PR-AUC: {average_precision_score(y_final_test, proba):.6f}")

    print("Creating ensemble predictions...")

    lgbm_avg = np.mean(
        np.vstack([
            proba_dict["LGBM_recency_weight_3"],
            proba_dict["LGBM_recency_weight_2"],
        ]),
        axis=0,
    )

    lgbm_bagged_avg = np.mean(
    np.vstack([
        proba_dict["LGBM_recency_weight_3"],
        proba_dict["LGBM_recency_weight_2"],
        proba_dict["LGBM_recency_weight_3_seed_101"],
        proba_dict["LGBM_recency_weight_3_seed_202"],
        proba_dict["LGBM_recency_weight_4_seed_303"],
        proba_dict["LGBM_recency_weight_2_seed_404"],
    ]),
    axis=0,
    )

    predictions["Ensemble_LGBM_bagged_avg"] = lgbm_bagged_avg
    results["Ensemble_LGBM_bagged_avg"] = evaluate_binary_classifier(
        y_final_test,
        lgbm_bagged_avg,
        threshold=0.5,
    )
    
    final_safe_blend = (
        0.90 * normalize_0_1(lgbm_bagged_avg)
        + 0.07 * normalize_0_1(proba_dict["XGBoost_final"])
        + 0.03 * normalize_0_1(proba_dict["CatBoost_final"])
    )

    predictions["Final_safe_blend_LGBM_XGB_CAT"] = final_safe_blend
    results["Final_safe_blend_LGBM_XGB_CAT"] = evaluate_binary_classifier(
        y_final_test,
        final_safe_blend,
        threshold=0.5,
    )

    tree_avg = np.mean(
        np.vstack([
            normalize_0_1(proba_dict["LGBM_recency_weight_3"]),
            normalize_0_1(proba_dict["XGBoost_final"]),
            normalize_0_1(proba_dict["CatBoost_final"]),
            normalize_0_1(proba_dict["RandomForest_final"]),
        ]),
        axis=0,
    )

    predictions["Ensemble_tree_avg"] = tree_avg
    results["Ensemble_tree_avg"] = evaluate_binary_classifier(y_final_test, tree_avg, threshold=0.5)

    weighted_blend = (
        0.55 * normalize_0_1(proba_dict["LGBM_recency_weight_3"])
        + 0.25 * normalize_0_1(proba_dict["XGBoost_final"])
        + 0.10 * normalize_0_1(proba_dict["CatBoost_final"])
        + 0.10 * normalize_0_1(proba_dict["RandomForest_final"])
    )

    predictions["Ensemble_weighted_blend"] = weighted_blend
    results["Ensemble_weighted_blend"] = evaluate_binary_classifier(
        y_final_test,
        weighted_blend,
        threshold=0.5,
    )

    rank_blend = rank_average([
        proba_dict["LGBM_recency_weight_3"],
        proba_dict["XGBoost_final"],
        proba_dict["XGBoost_recall_variant"],
        proba_dict["CatBoost_final"],
        proba_dict["RandomForest_final"],
    ])

    predictions["Ensemble_rank_average"] = rank_blend
    results["Ensemble_rank_average"] = evaluate_binary_classifier(
        y_final_test,
        rank_blend,
        threshold=0.5,
    )

    results_df = results_to_dataframe(results)

    print("\nUltimate IEEE final test results at threshold 0.5:")
    print(results_df.to_string())

    results_df.to_csv(
        OUTPUT_DIR / "reports" / "ultimate_ieee_final_test_results_05.csv"
    )

    predictions.to_csv(
        OUTPUT_DIR / "predictions" / "ultimate_ieee_final_test_predictions.csv",
        index=False,
    )

    print("\nThreshold optimization on test for analysis only...")
    print("Do not present these thresholds as deployment-selected thresholds.")

    threshold_rows = []
    test_amounts = test_df["TransactionAmt"].values if "TransactionAmt" in test_df.columns else None

    for model_name in predictions.columns:
        if model_name == "y_true":
            continue

        y_proba = predictions[model_name].values

        best_f2, threshold_df = find_best_threshold(
            y_final_test,
            y_proba,
            metric="f2",
        )

        best_cost, best_benefit, cost_df = cost_based_threshold(
            y_true=y_final_test,
            y_proba=y_proba,
            amounts=test_amounts,
            fp_cost=1.0,
            fn_cost=100.0,
        )

        threshold_rows.append({
            "model": model_name,
            "test_best_f2_threshold_oracle": best_f2["threshold"],
            "test_best_f2_oracle": best_f2["f2"],
            "precision_at_best_f2": best_f2["precision"],
            "recall_at_best_f2": best_f2["recall"],
            "mcc_at_best_f2": best_f2["mcc"],
            "best_cost_threshold_oracle": best_cost["threshold"],
            "min_total_cost_oracle": best_cost["total_cost"],
            "best_benefit_threshold_oracle": best_benefit["threshold"],
            "max_net_benefit_oracle": best_benefit["net_benefit"],
        })

    threshold_summary = pd.DataFrame(threshold_rows)

    print("\nUltimate threshold summary on test, analysis only:")
    print(threshold_summary.sort_values("test_best_f2_oracle", ascending=False).to_string(index=False))

    threshold_summary.to_csv(
        OUTPUT_DIR / "reports" / "ultimate_ieee_threshold_summary_test_oracle.csv",
        index=False,
    )

    print("\nUltimate model experiment complete.")


if __name__ == "__main__":
    main()