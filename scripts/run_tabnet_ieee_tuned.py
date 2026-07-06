import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import average_precision_score
from pytorch_tabnet.metrics import Metric
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
from src.features import (
    prepare_feature_engineering,
    add_velocity_features_searchsorted,
)
from src.preprocessing import prepare_basic_features
from src.metrics import evaluate_binary_classifier, results_to_dataframe

class PRAUCMetric(Metric):
    def __init__(self):
        self._name = "prauc"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return average_precision_score(y_true, y_score[:, 1])

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


def apply_extra_features(train_df, valid_df, test_df):
    train_df, valid_df = add_pair_frequency_features_train_test(train_df, valid_df)
    train_df, test_df = add_pair_frequency_features_train_test(train_df, test_df)

    train_df, valid_df = add_pair_amount_aggregates_train_test(train_df, valid_df)
    train_df, test_df = add_pair_amount_aggregates_train_test(train_df, test_df)

    train_df, valid_df, test_df = add_velocity_features_searchsorted(
        train_df,
        valid_df,
        test_df,
        group_col="card2",
        time_col=TIME_COL_IEEE,
        amount_col="TransactionAmt",
        windows_hours=[1, 6, 24],
    )

    train_df, valid_df, test_df = add_velocity_features_searchsorted(
        train_df,
        valid_df,
        test_df,
        group_col="addr1",
        time_col=TIME_COL_IEEE,
        amount_col="TransactionAmt",
        windows_hours=[1, 6, 24],
    )

    return train_df, valid_df, test_df


def scale_for_tabnet(X_train, X_valid, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled.astype(np.float32),
        X_valid_scaled.astype(np.float32),
        X_test_scaled.astype(np.float32),
    )


def main():
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

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

    print("Applying leakage-aware feature engineering...")
    train_df, valid_df, test_df, feature_info = prepare_feature_engineering(
        train_df,
        valid_df,
        test_df,
    )

    print("Adding extra pair and velocity features...")
    train_df, valid_df, test_df = apply_extra_features(
        train_df,
        valid_df,
        test_df,
    )

    print(f"Train after FE: {train_df.shape}")
    print(f"Valid after FE: {valid_df.shape}")
    print(f"Test after FE: {test_df.shape}")

    print("Preparing numeric features...")
    X_train, y_train, X_valid, y_valid, X_test, y_test, prep_info = prepare_basic_features(
        train_df,
        valid_df,
        test_df,
        target_col=TARGET_COL,
    )

    print(f"Prepared X_train: {X_train.shape}")
    print(f"Prepared X_valid: {X_valid.shape}")
    print(f"Prepared X_test: {X_test.shape}")

    print("Scaling features for TabNet...")
    X_train_np, X_valid_np, X_test_np = scale_for_tabnet(
        X_train,
        X_valid,
        X_test,
    )

    y_train_np = y_train.values.astype(np.int64)
    y_valid_np = y_valid.values.astype(np.int64)
    y_test_np = y_test.values.astype(np.int64)

    print("Training tuned TabNet configurations...")

    tabnet_configs = [
        {
            "name": "TabNet_w5_small",
            "params": {
                "n_d": 32,
                "n_a": 32,
                "n_steps": 3,
                "gamma": 1.3,
                "lambda_sparse": 1e-4,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "mask_type": "sparsemax",
                "class_weight": {0: 1.0, 1: 5.0},
            },
        },
        {
            "name": "TabNet_w8_medium",
            "params": {
                "n_d": 48,
                "n_a": 48,
                "n_steps": 3,
                "gamma": 1.3,
                "lambda_sparse": 1e-4,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "mask_type": "sparsemax",
                "class_weight": {0: 1.0, 1: 8.0},
            },
        },
        {
            "name": "TabNet_w10_medium",
            "params": {
                "n_d": 48,
                "n_a": 48,
                "n_steps": 3,
                "gamma": 1.4,
                "lambda_sparse": 1e-4,
                "lr": 8e-4,
                "weight_decay": 2e-4,
                "mask_type": "entmax",
                "class_weight": {0: 1.0, 1: 10.0},
            },
        },
    ]
    
    all_results = {}
    all_predictions_valid = {}
    all_predictions_test = {}

    best_model_name = None
    best_valid_pr_auc = -1
    best_valid_proba = None
    best_test_proba = None

    for cfg in tabnet_configs:
        name = cfg["name"]
        p = cfg["params"]

        print(f"\n=== Training {name} ===")

        clf = TabNetClassifier(
            n_d=p["n_d"],
            n_a=p["n_a"],
            n_steps=p["n_steps"],
            gamma=p["gamma"],
            lambda_sparse=p["lambda_sparse"],
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(
                lr=p["lr"],
                weight_decay=p["weight_decay"],
            ),
            scheduler_params={
                "step_size": 12,
                "gamma": 0.85,
            },
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type=p["mask_type"],
            seed=RANDOM_STATE,
            verbose=10,
            device_name="cuda" if torch.cuda.is_available() else "cpu",
        )

        clf.fit(
            X_train=X_train_np,
            y_train=y_train_np,
            eval_set=[(X_valid_np, y_valid_np)],
            eval_name=["valid"],
            eval_metric=[PRAUCMetric],
            max_epochs=90,
            patience=15,
            batch_size=8192,
            virtual_batch_size=512,
            num_workers=0,
            drop_last=False,
            weights=p["class_weight"],
        )

        valid_proba = clf.predict_proba(X_valid_np)[:, 1]
        test_proba = clf.predict_proba(X_test_np)[:, 1]

        valid_pr_auc = average_precision_score(y_valid, valid_proba)
        test_pr_auc = average_precision_score(y_test, test_proba)
        valid_roc_auc = roc_auc_score(y_valid, valid_proba)
        test_roc_auc = roc_auc_score(y_test, test_proba)

        print(f"{name} validation PR-AUC: {valid_pr_auc:.6f}")
        print(f"{name} validation ROC-AUC: {valid_roc_auc:.6f}")
        print(f"{name} test PR-AUC: {test_pr_auc:.6f}")
        print(f"{name} test ROC-AUC: {test_roc_auc:.6f}")

        all_predictions_valid[name] = valid_proba
        all_predictions_test[name] = test_proba

        all_results[f"{name}_valid"] = evaluate_binary_classifier(
            y_valid,
            valid_proba,
            threshold=0.5,
        )
        all_results[f"{name}_test"] = evaluate_binary_classifier(
            y_test,
            test_proba,
            threshold=0.5,
        )

        if valid_pr_auc > best_valid_pr_auc:
            best_valid_pr_auc = valid_pr_auc
            best_model_name = name
            best_valid_proba = valid_proba
            best_test_proba = test_proba

    print("\nBest TabNet selected by validation PR-AUC:")
    print(best_model_name)
    print("Best validation PR-AUC:", best_valid_pr_auc)
    print("Best selected test PR-AUC:", average_precision_score(y_test, best_test_proba))
    print("Best selected test ROC-AUC:", roc_auc_score(y_test, best_test_proba))

    results_df = results_to_dataframe(all_results)

    print("\nTuned TabNet results:")
    print(results_df.to_string())

    results_df.to_csv(
        OUTPUT_DIR / "reports" / "tabnet_tuned_ieee_results.csv"
    )

    pred_valid_df = pd.DataFrame({"y_true": y_valid.values})
    pred_test_df = pd.DataFrame({"y_true": y_test.values})

    for model_name, proba in all_predictions_valid.items():
        pred_valid_df[model_name] = proba

    for model_name, proba in all_predictions_test.items():
        pred_test_df[model_name] = proba

    pred_valid_df.to_csv(
        OUTPUT_DIR / "predictions" / "tabnet_tuned_validation_predictions.csv",
        index=False,
    )

    pred_test_df.to_csv(
        OUTPUT_DIR / "predictions" / "tabnet_tuned_test_predictions.csv",
        index=False,
    )

    summary_rows = []

    for model_name in all_predictions_valid.keys():
        summary_rows.append({
            "model": model_name,
            "valid_pr_auc": average_precision_score(y_valid, all_predictions_valid[model_name]),
            "valid_roc_auc": roc_auc_score(y_valid, all_predictions_valid[model_name]),
            "test_pr_auc": average_precision_score(y_test, all_predictions_test[model_name]),
            "test_roc_auc": roc_auc_score(y_test, all_predictions_test[model_name]),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("valid_pr_auc", ascending=False)

    print("\nTabNet tuning summary:")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(
        OUTPUT_DIR / "reports" / "tabnet_tuning_summary.csv",
        index=False,
    )

    print("\nSaved tuned TabNet benchmark results.")


if __name__ == "__main__":
    main()