import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from config import CREDITCARD_PATH, OUTPUT_DIR, RANDOM_STATE
from src.splitting import time_train_valid_test_split
from src.metrics import evaluate_binary_classifier, results_to_dataframe
from src.threshold_opt import find_best_threshold, cost_based_threshold
from src.stacking import (
    build_meta_features,
    train_logistic_meta_model,
    predict_meta_model,
)


def get_scale_pos_weight(y):
    neg = np.sum(y == 0)
    pos = np.sum(y == 1)
    return neg / max(pos, 1)


def add_creditcard_features(df):
    df = df.copy()

    if "Amount" in df.columns:
        df["Amount_log"] = np.log1p(df["Amount"])

    if "Time" in df.columns:
        df["hour"] = (df["Time"] // 3600) % 24
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["time_since_previous_transaction"] = df["Time"].diff().fillna(0)

    return df


def prepare_creditcard_data(train_df, valid_df, test_df):
    train_df = add_creditcard_features(train_df)
    valid_df = add_creditcard_features(valid_df)
    test_df = add_creditcard_features(test_df)

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"].astype(int)

    X_valid = valid_df.drop(columns=["Class"])
    y_valid = valid_df["Class"].astype(int)

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"].astype(int)

    # Fill if any unexpected missing values appear
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_valid = X_valid.fillna(medians)
    X_test = X_test.fillna(medians)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def train_models(X_train, y_train):
    scale_pos_weight = get_scale_pos_weight(y_train)

    models = {}

    models["Logistic Regression"] = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    models["LightGBM"] = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    models["XGBoost"] = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

    print("Training Isolation Forest...")
    iso_model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso_model.fit(X_train[y_train == 0])

    return models, iso_model


def get_isolation_score(model, X):
    raw_score = -model.decision_function(X)
    min_score = raw_score.min()
    max_score = raw_score.max()

    if max_score == min_score:
        return np.zeros_like(raw_score)

    return (raw_score - min_score) / (max_score - min_score)


def main():
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    print("Loading Credit Card Fraud dataset...")
    df = pd.read_csv(CREDITCARD_PATH)

    print(f"Dataset shape: {df.shape}")
    print("Fraud rate:")
    print(df["Class"].value_counts(normalize=True))

    print("Creating time-based train/valid/test split by Time...")
    train_df, valid_df, test_df = time_train_valid_test_split(
        df,
        time_col="Time",
        train_ratio=0.60,
        valid_ratio=0.20,
    )

    print(f"Train: {train_df.shape}")
    print(f"Valid: {valid_df.shape}")
    print(f"Test: {test_df.shape}")

    print("Preparing features...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_creditcard_data(
        train_df,
        valid_df,
        test_df,
    )

    print(f"Prepared X_train: {X_train.shape}")
    print(f"Prepared X_valid: {X_valid.shape}")
    print(f"Prepared X_test: {X_test.shape}")

    models, iso_model = train_models(X_train, y_train)

    predictions_valid = pd.DataFrame({"y_true": y_valid.values})
    predictions_test = pd.DataFrame({"y_true": y_test.values})

    results_valid = {}
    results_test = {}

    print("Evaluating base models...")
    for name, model in models.items():
        valid_proba = model.predict_proba(X_valid)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        predictions_valid[name] = valid_proba
        predictions_test[name] = test_proba

        results_valid[name] = evaluate_binary_classifier(y_valid, valid_proba, threshold=0.5)
        results_test[name] = evaluate_binary_classifier(y_test, test_proba, threshold=0.5)

    iso_valid = get_isolation_score(iso_model, X_valid)
    iso_test = get_isolation_score(iso_model, X_test)

    predictions_valid["Isolation Forest"] = iso_valid
    predictions_test["Isolation Forest"] = iso_test

    results_valid["Isolation Forest"] = evaluate_binary_classifier(y_valid, iso_valid, threshold=0.5)
    results_test["Isolation Forest"] = evaluate_binary_classifier(y_test, iso_test, threshold=0.5)

    print("Training validation-based stacking benchmark...")
    X_meta_valid, y_meta_valid = build_meta_features(predictions_valid, target_col="y_true")
    meta_model = train_logistic_meta_model(X_meta_valid, y_meta_valid)

    stack_valid = predict_meta_model(meta_model, predictions_valid, target_col="y_true")
    stack_test = predict_meta_model(meta_model, predictions_test, target_col="y_true")

    predictions_valid["Stacking Logistic Meta"] = stack_valid
    predictions_test["Stacking Logistic Meta"] = stack_test

    results_valid["Stacking Logistic Meta"] = evaluate_binary_classifier(
        y_valid,
        stack_valid,
        threshold=0.5,
    )
    results_test["Stacking Logistic Meta"] = evaluate_binary_classifier(
        y_test,
        stack_test,
        threshold=0.5,
    )

    valid_results_df = results_to_dataframe(results_valid)
    test_results_df = results_to_dataframe(results_test)

    print("\nCredit Card validation results at threshold 0.5:")
    print(valid_results_df)

    print("\nCredit Card test results at threshold 0.5:")
    print(test_results_df)

    valid_results_df.to_csv(
        OUTPUT_DIR / "reports" / "creditcard_validation_results_threshold_05.csv"
    )
    test_results_df.to_csv(
        OUTPUT_DIR / "reports" / "creditcard_test_results_threshold_05.csv"
    )

    predictions_valid.to_csv(
        OUTPUT_DIR / "predictions" / "creditcard_validation_predictions.csv",
        index=False,
    )
    predictions_test.to_csv(
        OUTPUT_DIR / "predictions" / "creditcard_test_predictions.csv",
        index=False,
    )

    print("\nThreshold optimization on Credit Card validation set...")

    threshold_reports = []

    valid_amounts = valid_df["Amount"].values if "Amount" in valid_df.columns else None

    for model_name in predictions_valid.columns:
        if model_name == "y_true":
            continue

        y_proba = predictions_valid[model_name].values

        best_f2, threshold_df = find_best_threshold(y_valid, y_proba, metric="f2")

        best_cost, best_benefit, cost_df = cost_based_threshold(
            y_true=y_valid,
            y_proba=y_proba,
            amounts=valid_amounts,
            fp_cost=1.0,
            fn_cost=100.0,
        )

        threshold_df.to_csv(
            OUTPUT_DIR / "reports" / f"creditcard_thresholds_{model_name.replace(' ', '_')}.csv",
            index=False,
        )

        cost_df.to_csv(
            OUTPUT_DIR / "reports" / f"creditcard_cost_thresholds_{model_name.replace(' ', '_')}.csv",
            index=False,
        )

        threshold_reports.append({
            "model": model_name,
            "best_f2_threshold": best_f2["threshold"],
            "best_f2": best_f2["f2"],
            "precision_at_best_f2": best_f2["precision"],
            "recall_at_best_f2": best_f2["recall"],
            "mcc_at_best_f2": best_f2["mcc"],
            "best_cost_threshold": best_cost["threshold"],
            "min_total_cost": best_cost["total_cost"],
            "best_benefit_threshold": best_benefit["threshold"],
            "max_net_benefit": best_benefit["net_benefit"],
        })

    threshold_summary = pd.DataFrame(threshold_reports)

    print("\nCredit Card threshold optimization summary:")
    print(threshold_summary.sort_values("best_f2", ascending=False))

    threshold_summary.to_csv(
        OUTPUT_DIR / "reports" / "creditcard_threshold_optimization_summary.csv",
        index=False,
    )

    print("\nCredit Card test results using validation-selected F2 thresholds...")

    test_threshold_results = {}

    for _, row in threshold_summary.iterrows():
        model_name = row["model"]

        if model_name not in predictions_test.columns:
            continue

        threshold = row["best_f2_threshold"]
        y_proba_test = predictions_test[model_name].values

        test_threshold_results[model_name] = evaluate_binary_classifier(
            y_test,
            y_proba_test,
            threshold=threshold,
        )
        test_threshold_results[model_name]["validation_selected_threshold"] = threshold

    test_threshold_df = results_to_dataframe(test_threshold_results)

    print(test_threshold_df)

    test_threshold_df.to_csv(
        OUTPUT_DIR / "reports" / "creditcard_test_results_validation_f2_thresholds.csv"
    )

    print("\nCredit Card benchmark complete.")


if __name__ == "__main__":
    main()