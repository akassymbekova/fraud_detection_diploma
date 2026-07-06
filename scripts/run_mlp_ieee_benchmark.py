import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

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
from src.features import (
    prepare_feature_engineering,
    add_velocity_features_searchsorted,
)
from src.preprocessing import prepare_basic_features
from src.metrics import evaluate_binary_classifier, results_to_dataframe


class FraudMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.BatchNorm1d(384),
            nn.SiLU(),
            nn.Dropout(0.25),

            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Dropout(0.20),

            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.SiLU(),
            nn.Dropout(0.15),

            nn.Linear(96, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
    
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=targets.device),
            torch.tensor(1 - self.alpha, device=targets.device),
        )

        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()


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


def scale_data(X_train, X_valid, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_valid_scaled = scaler.transform(X_valid).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return X_train_scaled, X_valid_scaled, X_test_scaled


def predict_proba(model, X_np, device, batch_size=8192):
    model.eval()

    dataset = TensorDataset(torch.tensor(X_np, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits).cpu().numpy()
            probs.append(prob)

    return np.concatenate(probs)


def train_mlp(
    X_train,
    y_train,
    X_valid,
    y_valid,
    max_epochs=60,
    batch_size=4096,
    patience=10,
    pos_weight_cap=5.0,
    lr=7e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    y_train_np = y_train.values.astype(np.float32)
    y_valid_np = y_valid.values.astype(np.float32)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = FraudMLP(input_dim=X_train.shape[1]).to(device)

    neg = np.sum(y_train_np == 0)
    pos = np.sum(y_train_np == 1)
    raw_pos_weight = neg / max(pos, 1)
    pos_weight_value = min(raw_pos_weight, pos_weight_cap)

    print(f"Raw pos_weight: {raw_pos_weight:.3f}")
    print(f"Used capped pos_weight: {pos_weight_value:.3f}")

    criterion = FocalLoss(
        alpha=0.35,
        gamma=2.0,
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32).to(device),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=2e-4,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=12,
        gamma=0.75,
    )

    best_valid_pr_auc = -1
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

            optimizer.step()

            total_loss += loss.item() * len(xb)

        scheduler.step()

        valid_proba = predict_proba(model, X_valid, device)
        valid_pr_auc = average_precision_score(y_valid_np, valid_proba)
        valid_roc_auc = roc_auc_score(y_valid_np, valid_proba)

        avg_loss = total_loss / len(train_dataset)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"loss={avg_loss:.5f} | "
            f"valid PR-AUC={valid_pr_auc:.6f} | "
            f"valid ROC-AUC={valid_roc_auc:.6f}"
        )

        if valid_pr_auc > best_valid_pr_auc:
            best_valid_pr_auc = valid_pr_auc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)

    return model, device, best_valid_pr_auc

def main():
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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
    train_df, valid_df, test_df, _ = prepare_feature_engineering(
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

    print("Preparing features...")
    X_train, y_train, X_valid, y_valid, X_test, y_test, _ = prepare_basic_features(
        train_df,
        valid_df,
        test_df,
        target_col=TARGET_COL,
    )

    print(f"Prepared X_train: {X_train.shape}")
    print(f"Prepared X_valid: {X_valid.shape}")
    print(f"Prepared X_test: {X_test.shape}")

    print("Scaling features...")
    X_train_np, X_valid_np, X_test_np = scale_data(
        X_train,
        X_valid,
        X_test,
    )

    print("Training Focal Loss MLP variants...")

    mlp_configs = [
        {"name": "MLP_focal_cap5", "pos_weight_cap": 5.0, "lr": 7e-4},
        {"name": "MLP_focal_cap8", "pos_weight_cap": 8.0, "lr": 5e-4},
        {"name": "MLP_focal_cap12", "pos_weight_cap": 12.0, "lr": 5e-4},
    ]

    valid_predictions = {}
    test_predictions = {}
    results = {}

    for cfg in mlp_configs:
        print(f"\n=== Training {cfg['name']} ===")

        model, device, best_valid_pr_auc = train_mlp(
            X_train_np,
            y_train,
            X_valid_np,
            y_valid,
            max_epochs=70,
            batch_size=4096,
            patience=12,
            pos_weight_cap=cfg["pos_weight_cap"],
            lr=cfg["lr"],
        )

        valid_proba = predict_proba(model, X_valid_np, device)
        test_proba = predict_proba(model, X_test_np, device)

        valid_predictions[cfg["name"]] = valid_proba
        test_predictions[cfg["name"]] = test_proba

        results[f"{cfg['name']}_valid"] = evaluate_binary_classifier(
            y_valid,
            valid_proba,
            threshold=0.5,
        )

        results[f"{cfg['name']}_test"] = evaluate_binary_classifier(
            y_test,
            test_proba,
            threshold=0.5,
        )

        print(f"{cfg['name']} valid PR-AUC:", average_precision_score(y_valid, valid_proba))
        print(f"{cfg['name']} test PR-AUC:", average_precision_score(y_test, test_proba))

    valid_ensemble = np.mean(
        np.column_stack([valid_predictions[name] for name in valid_predictions.keys()]),
        axis=1,
    )

    test_ensemble = np.mean(
        np.column_stack([test_predictions[name] for name in test_predictions.keys()]),
        axis=1,
    )

    valid_predictions["MLP_focal_ensemble_avg"] = valid_ensemble
    test_predictions["MLP_focal_ensemble_avg"] = test_ensemble

    results["MLP_focal_ensemble_avg_valid"] = evaluate_binary_classifier(
        y_valid,
        valid_ensemble,
        threshold=0.5,
    )

    results["MLP_focal_ensemble_avg_test"] = evaluate_binary_classifier(
        y_test,
        test_ensemble,
        threshold=0.5,
    )

    best_model_name = max(
        valid_predictions.keys(),
        key=lambda name: average_precision_score(y_valid, valid_predictions[name]),
    )

    valid_proba = valid_predictions[best_model_name]
    test_proba = test_predictions[best_model_name]

    print("\nBest Focal MLP selected by validation PR-AUC:", best_model_name)
    print("Best validation PR-AUC:", average_precision_score(y_valid, valid_proba))
    print("Best selected test PR-AUC:", average_precision_score(y_test, test_proba))
    print("Best selected test ROC-AUC:", roc_auc_score(y_test, test_proba))

    results_df = results_to_dataframe(results)

    print("\nFocal MLP results:")
    print(results_df.to_string())

    summary_rows = []
    for name in valid_predictions.keys():
        summary_rows.append({
            "model": name,
            "valid_pr_auc": average_precision_score(y_valid, valid_predictions[name]),
            "valid_roc_auc": roc_auc_score(y_valid, valid_predictions[name]),
            "test_pr_auc": average_precision_score(y_test, test_predictions[name]),
            "test_roc_auc": roc_auc_score(y_test, test_predictions[name]),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "valid_pr_auc",
        ascending=False,
    )

    print("\nFocal MLP tuning summary:")
    print(summary_df.to_string(index=False))

    results_df.to_csv(
        OUTPUT_DIR / "reports" / "mlp_focal_ieee_results.csv"
    )

    summary_df.to_csv(
        OUTPUT_DIR / "reports" / "mlp_focal_tuning_summary.csv",
        index=False,
    )

    pred_valid_df = pd.DataFrame({"y_true": y_valid.values})
    pred_test_df = pd.DataFrame({"y_true": y_test.values})

    for name, proba in valid_predictions.items():
        pred_valid_df[name] = proba

    for name, proba in test_predictions.items():
        pred_test_df[name] = proba

    pred_valid_df.to_csv(
        OUTPUT_DIR / "predictions" / "mlp_focal_ieee_validation_predictions.csv",
        index=False,
    )

    pred_test_df.to_csv(
        OUTPUT_DIR / "predictions" / "mlp_focal_ieee_test_predictions.csv",
        index=False,
    )

    print("\nSaved Focal MLP benchmark results.")


if __name__ == "__main__":
    main()
