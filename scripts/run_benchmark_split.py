import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    IEEE_TRANSACTION_PATH,
    IEEE_IDENTITY_PATH,
    TARGET_COL,
    OUTPUT_DIR,
)

from src.data_loader import load_ieee
from src.preprocessing import prepare_basic_features
from src.features import prepare_feature_engineering
from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_lightgbm,
    train_xgboost,
    train_catboost,
    get_model_proba,
)
from src.metrics import evaluate_binary_classifier, results_to_dataframe
from src.threshold_opt import find_best_threshold


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)

    print("Loading IEEE-CIS dataset...")
    df = load_ieee(IEEE_TRANSACTION_PATH, IEEE_IDENTITY_PATH)

    print("Creating stratified benchmark split...")

    train_valid_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[TARGET_COL],
        random_state=42,
    )

    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=0.25,
        stratify=train_valid_df[TARGET_COL],
        random_state=42,
    )

    print(f"Train: {train_df.shape}")
    print(f"Valid: {valid_df.shape}")
    print(f"Test: {test_df.shape}")

    print("Applying feature engineering...")
    train_df, valid_df, test_df, _ = prepare_feature_engineering(
        train_df,
        valid_df,
        test_df,
    )

    print("Preprocessing...")
    X_train, y_train, X_valid, y_valid, X_test, y_test, _ = prepare_basic_features(
        train_df,
        valid_df,
        test_df,
        target_col=TARGET_COL,
    )

    models = {}

    print("Training Logistic Regression...")
    models["Logistic Regression"] = train_logistic_regression(X_train, y_train)

    print("Training Random Forest...")
    models["Random Forest"] = train_random_forest(X_train, y_train)

    print("Training LightGBM...")
    models["LightGBM"] = train_lightgbm(X_train, y_train)

    print("Training XGBoost...")
    models["XGBoost"] = train_xgboost(X_train, y_train)

    print("Training CatBoost...")
    models["CatBoost"] = train_catboost(X_train, y_train)

    valid_results = {}
    test_results_05 = {}
    test_results_valid_threshold = {}

    print("Evaluating models...")

    for name, model in models.items():
        valid_proba = get_model_proba(model, X_valid)
        test_proba = get_model_proba(model, X_test)

        valid_results[name] = evaluate_binary_classifier(
            y_valid,
            valid_proba,
            threshold=0.5,
        )

        test_results_05[name] = evaluate_binary_classifier(
            y_test,
            test_proba,
            threshold=0.5,
        )

        best_f2, _ = find_best_threshold(y_valid, valid_proba, metric="f2")
        best_threshold = best_f2["threshold"]

        test_results_valid_threshold[name] = evaluate_binary_classifier(
            y_test,
            test_proba,
            threshold=best_threshold,
        )
        test_results_valid_threshold[name]["validation_selected_threshold"] = best_threshold

    valid_df_results = results_to_dataframe(valid_results)
    test_05_df = results_to_dataframe(test_results_05)
    test_threshold_df = results_to_dataframe(test_results_valid_threshold)

    print("\nBenchmark validation results:")
    print(valid_df_results)

    print("\nBenchmark test results at threshold 0.5:")
    print(test_05_df)

    print("\nBenchmark test results using validation-selected F2 threshold:")
    print(test_threshold_df)

    valid_df_results.to_csv(
        OUTPUT_DIR / "reports" / "benchmark_stratified_validation_results.csv"
    )
    test_05_df.to_csv(
        OUTPUT_DIR / "reports" / "benchmark_stratified_test_threshold_05.csv"
    )
    test_threshold_df.to_csv(
        OUTPUT_DIR / "reports" / "benchmark_stratified_test_validation_threshold.csv"
    )


if __name__ == "__main__":
    main()