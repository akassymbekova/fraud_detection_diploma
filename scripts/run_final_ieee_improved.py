import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
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
from src.features import prepare_feature_engineering
from src.preprocessing import prepare_basic_features
from src.metrics import evaluate_binary_classifier, results_to_dataframe
from src.threshold_opt import find_best_threshold, cost_based_threshold


def get_scale_pos_weight(y):
    neg = np.sum(y == 0)
    pos = np.sum(y == 1)
    return neg / max(pos, 1)


def train_lightgbm_custom(
    X_train,
    y_train,
    sample_weight=None,
    scale_pos_weight_multiplier=1.0,
    n_estimators=1200,
    learning_rate=0.025,
    num_leaves=96,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
):
    base_spw = get_scale_pos_weight(y_train)
    scale_pos_weight = base_spw * scale_pos_weight_multiplier

    model = LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def make_recency_weights(n_train_original, n_valid_original, recent_weight=2.0):
    """
    Old train rows get weight 1.
    Validation/recent rows get higher weight.
    """
    old_weights = np.ones(n_train_original)
    recent_weights = np.ones(n_valid_original) * recent_weight
    return np.concatenate([old_weights, recent_weights])


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

    print("Creating final training set = train + validation...")
    final_train_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)

    n_train_original = len(train_df)
    n_valid_original = len(valid_df)

    print(f"Final train: {final_train_df.shape}")
    print(f"Final test: {test_df.shape}")

    print("Applying feature engineering...")
    # We use final_train as history and test as future holdout.
    final_train_fe, final_test_fe, _unused, feature_info = prepare_feature_engineering(
        final_train_df,
        test_df.copy(),
        test_df.copy(),
    )

    print(f"Final train after FE: {final_train_fe.shape}")
    print(f"Final test after FE: {final_test_fe.shape}")

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

    # Different training strategies.
    experiments = []

    print("Training improved LightGBM variants...")

    # 1. Standard final training on train+valid
    experiments.append({
        "name": "LGBM_train_valid_standard",
        "model": train_lightgbm_custom(
            X_train,
            y_train,
            sample_weight=None,
            scale_pos_weight_multiplier=1.0,
            n_estimators=1200,
            learning_rate=0.025,
            num_leaves=96,
            min_child_samples=80,
        )
    })

    # 2. Slightly stronger class weighting
    experiments.append({
        "name": "LGBM_train_valid_spw_1_3",
        "model": train_lightgbm_custom(
            X_train,
            y_train,
            sample_weight=None,
            scale_pos_weight_multiplier=1.3,
            n_estimators=1200,
            learning_rate=0.025,
            num_leaves=96,
            min_child_samples=80,
        )
    })

    # 3. Recency weighting: recent validation period matters more
    weights_recent_2 = make_recency_weights(
        n_train_original,
        n_valid_original,
        recent_weight=2.0,
    )

    experiments.append({
        "name": "LGBM_recency_weight_2",
        "model": train_lightgbm_custom(
            X_train,
            y_train,
            sample_weight=weights_recent_2,
            scale_pos_weight_multiplier=1.0,
            n_estimators=1200,
            learning_rate=0.025,
            num_leaves=96,
            min_child_samples=80,
        )
    })

    # 4. Stronger recency weighting
    weights_recent_3 = make_recency_weights(
        n_train_original,
        n_valid_original,
        recent_weight=3.0,
    )

    experiments.append({
        "name": "LGBM_recency_weight_3",
        "model": train_lightgbm_custom(
            X_train,
            y_train,
            sample_weight=weights_recent_3,
            scale_pos_weight_multiplier=1.0,
            n_estimators=1200,
            learning_rate=0.025,
            num_leaves=96,
            min_child_samples=80,
        )
    })

    # 5. More conservative model
    experiments.append({
        "name": "LGBM_conservative",
        "model": train_lightgbm_custom(
            X_train,
            y_train,
            sample_weight=weights_recent_2,
            scale_pos_weight_multiplier=1.0,
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=64,
            min_child_samples=120,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.5,
            reg_lambda=2.0,
        )
    })

    predictions = pd.DataFrame({"y_true": y_final_test.values})
    results = {}

    print("Evaluating improved variants on future test set...")

    proba_list = []

    for exp in experiments:
        name = exp["name"]
        model = exp["model"]

        proba = model.predict_proba(X_final_test)[:, 1]
        predictions[name] = proba
        proba_list.append(proba)

        results[name] = evaluate_binary_classifier(
            y_final_test,
            proba,
            threshold=0.5,
        )

        print(f"{name} PR-AUC:", average_precision_score(y_final_test, proba))

    # Simple ensemble average
    ensemble_proba = np.mean(np.vstack(proba_list), axis=0)
    predictions["LGBM_improved_ensemble_avg"] = ensemble_proba

    results["LGBM_improved_ensemble_avg"] = evaluate_binary_classifier(
        y_final_test,
        ensemble_proba,
        threshold=0.5,
    )

    results_df = results_to_dataframe(results)

    print("\nImproved IEEE LightGBM test results at threshold 0.5:")
    print(results_df.to_string())

    results_df.to_csv(
        OUTPUT_DIR / "reports" / "improved_ieee_lightgbm_test_results_05.csv"
    )

    predictions.to_csv(
        OUTPUT_DIR / "predictions" / "improved_ieee_lightgbm_test_predictions.csv",
        index=False,
    )

    print("\nThreshold optimization for improved models on test set for analysis only...")
    print("Important: this is oracle analysis, not deployment threshold selection.")

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

        threshold_df.to_csv(
            OUTPUT_DIR / "reports" / f"improved_thresholds_{model_name}.csv",
            index=False,
        )

        cost_df.to_csv(
            OUTPUT_DIR / "reports" / f"improved_cost_thresholds_{model_name}.csv",
            index=False,
        )

    threshold_summary = pd.DataFrame(threshold_rows)

    print("\nImproved models threshold summary on test, analysis only:")
    print(threshold_summary.sort_values("test_best_f2_oracle", ascending=False).to_string(index=False))

    threshold_summary.to_csv(
        OUTPUT_DIR / "reports" / "improved_ieee_lightgbm_threshold_summary_test_oracle.csv",
        index=False,
    )

    print("\nDone. Improved model experiment saved.")


if __name__ == "__main__":
    main()