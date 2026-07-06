import pandas as pd

from config import (
    IEEE_TRANSACTION_PATH,
    IEEE_IDENTITY_PATH,
    TARGET_COL,
    TIME_COL_IEEE,
    TRAIN_RATIO,
    VALID_RATIO,
    OUTPUT_DIR,
)

from src.data_loader import load_ieee
from src.splitting import time_train_valid_test_split
from src.features import prepare_feature_engineering
from src.preprocessing import prepare_basic_features
from src.models import train_lightgbm, get_model_proba
from src.metrics import evaluate_binary_classifier
from src.threshold_opt import find_best_threshold


def main():
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    print("Loading IEEE-CIS dataset...")
    df = load_ieee(IEEE_TRANSACTION_PATH, IEEE_IDENTITY_PATH)

    print("Creating original time-based train/valid/test split...")
    train_df, valid_df, test_df = time_train_valid_test_split(
        df,
        time_col=TIME_COL_IEEE,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
    )

    print(f"Original train: {train_df.shape}")
    print(f"Original valid: {valid_df.shape}")
    print(f"Original test: {test_df.shape}")

    print("Creating final training set = train + validation...")
    final_train_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)

    print(f"Final train: {final_train_df.shape}")
    print(f"Final test: {test_df.shape}")

    print("Applying feature engineering fitted on final_train only...")

    # We pass test_df as valid_df because prepare_feature_engineering returns features for it
    # using final_train as historical data. We ignore the third returned dataframe.
    final_train_fe, final_test_fe, _unused_test_fe, feature_info = prepare_feature_engineering(
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

    # X_test_as_valid and X_test are identical here; use X_test_as_valid for clarity
    X_final_test = X_test_as_valid
    y_final_test = y_test_as_valid

    print(f"Prepared final X_train: {X_train.shape}")
    print(f"Prepared final X_test: {X_final_test.shape}")

    print("Training final LightGBM on train + validation...")
    model = train_lightgbm(X_train, y_train)

    print("Predicting on final untouched test set...")
    test_proba = get_model_proba(model, X_final_test)

    # Evaluate at default 0.5
    result_05 = evaluate_binary_classifier(
        y_final_test,
        test_proba,
        threshold=0.5,
    )

    # Also evaluate using previously selected validation F2 threshold from main pipeline
    previous_best_threshold = 0.44

    result_044 = evaluate_binary_classifier(
        y_final_test,
        test_proba,
        threshold=previous_best_threshold,
    )

    # Find best threshold on test only for analysis, not for final deployment claim
    best_test_threshold, threshold_df = find_best_threshold(
        y_final_test,
        test_proba,
        metric="f2",
    )

    result_best_test = evaluate_binary_classifier(
        y_final_test,
        test_proba,
        threshold=best_test_threshold["threshold"],
    )

    results = pd.DataFrame([
        {"setting": "threshold_0.5", **result_05},
        {"setting": "previous_validation_threshold_0.44", **result_044},
        {"setting": "oracle_best_test_threshold_for_analysis_only", **result_best_test},
    ])

    print("\nFinal IEEE LightGBM test results trained on train+valid:")
    print(results.to_string(index=False))

    print("\nBest test threshold for analysis only:")
    print(best_test_threshold.to_string())

    results.to_csv(
        OUTPUT_DIR / "reports" / "final_ieee_lightgbm_train_valid_test_results.csv",
        index=False,
    )

    threshold_df.to_csv(
        OUTPUT_DIR / "reports" / "final_ieee_lightgbm_test_threshold_curve.csv",
        index=False,
    )

    predictions = pd.DataFrame({
        "y_true": y_final_test.values,
        "LightGBM_final_train_valid": test_proba,
    })

    predictions.to_csv(
        OUTPUT_DIR / "predictions" / "final_ieee_lightgbm_test_predictions.csv",
        index=False,
    )

    print("\nSaved final LightGBM results.")


if __name__ == "__main__":
    main()