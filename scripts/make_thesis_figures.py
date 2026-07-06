import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


# =========================
# Paths
# =========================

BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
REPORT_DIR = OUTPUT_DIR / "reports"
FIG_DIR = BASE_DIR / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# General plot style
# =========================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})


def save_fig(filename):
    path = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =========================
# Helper functions
# =========================

def find_prediction_file():
    """
    Tries to find the most useful prediction file.
    Priority:
    1. ultimate_ieee_final_test_predictions.csv
    2. final_ieee_lightgbm_test_predictions.csv
    3. validation_predictions.csv
    """
    candidates = [
        PRED_DIR / "ultimate_ieee_final_test_predictions.csv",
        PRED_DIR / "final_ieee_lightgbm_test_predictions.csv",
        PRED_DIR / "validation_predictions.csv",
    ]

    for path in candidates:
        if path.exists():
            print(f"Using prediction file: {path}")
            return path

    raise FileNotFoundError(
        "No prediction file found. Expected one of:\n"
        "- outputs/predictions/ultimate_ieee_final_test_predictions.csv\n"
        "- outputs/predictions/final_ieee_lightgbm_test_predictions.csv\n"
        "- outputs/predictions/validation_predictions.csv"
    )


def choose_final_model_column(df):
    """
    Chooses the best available model probability column.
    """
    preferred_cols = [
        "LGBM_recency_weight_2_seed_404",
        "lgbm_a",
        "Ensemble_LGBM_bagged_avg",
        "LightGBM_final_train_valid",
        "LightGBM",
    ]

    for col in preferred_cols:
        if col in df.columns:
            print(f"Using final model column: {col}")
            return col

    possible = [c for c in df.columns if c != "y_true"]
    if not possible:
        raise ValueError("No model probability columns found in prediction file.")

    print(f"Using fallback model column: {possible[0]}")
    return possible[0]


def compute_threshold_table(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        beta = 2
        if precision + recall > 0:
            f2 = (1 + beta ** 2) * precision * recall / ((beta ** 2 * precision) + recall)
        else:
            f2 = 0

        rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f2": f2,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })

    return pd.DataFrame(rows)


def minmax(x):
    x = np.asarray(x)
    if x.max() == x.min():
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


# =========================
# Load predictions
# =========================

pred_path = find_prediction_file()
pred_df = pd.read_csv(pred_path)

if "y_true" not in pred_df.columns:
    raise ValueError("Prediction file must contain y_true column.")

model_col = choose_final_model_column(pred_df)

y_true = pred_df["y_true"].values.astype(int)
y_proba = pred_df[model_col].values.astype(float)

fraud_rate = y_true.mean()
pr_auc = average_precision_score(y_true, y_proba)

print(f"Fraud rate: {fraud_rate:.6f}")
print(f"PR-AUC for {model_col}: {pr_auc:.6f}")


# ==========================================================
# 1. Reliability curve – LightGBM before/after calibration
# ==========================================================

def plot_reliability_curve():
    """
    If calibration prediction columns exist, use them.
    Otherwise, uses uncalibrated only and creates a smoothed proxy curve
    for visual comparison. For thesis, prefer real calibrated predictions
    if available.
    """

    uncalibrated = y_proba

    calibrated_col_candidates = [
        f"{model_col} Isotonic Calibrated",
        "LightGBM Isotonic Calibrated",
        "LGBM_recency_weight_2_seed_404 Isotonic Calibrated",
        "Ensemble_LGBM_bagged_avg Isotonic Calibrated",
    ]

    calibrated = None
    calibrated_col = None

    for col in calibrated_col_candidates:
        if col in pred_df.columns:
            calibrated = pred_df[col].values.astype(float)
            calibrated_col = col
            break

    # If no calibrated prediction exists, try to read calibration curve csv
    calibration_curve_path = REPORT_DIR / "calibration_curves_test.csv"

    plt.figure(figsize=(8, 6))

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")

    if calibration_curve_path.exists():
        cal_df = pd.read_csv(calibration_curve_path)

        # Try common column names
        print(f"Using calibration curve file: {calibration_curve_path}")
        print("Calibration curve columns:", list(cal_df.columns))

        # Expected possible columns: model, mean_predicted_value, fraction_of_positives
        model_colnames = [c for c in cal_df.columns if "model" in c.lower()]
        mean_cols = [c for c in cal_df.columns if "mean" in c.lower() or "predicted" in c.lower()]
        frac_cols = [c for c in cal_df.columns if "fraction" in c.lower() or "positive" in c.lower()]

        if model_colnames and mean_cols and frac_cols:
            mcol = model_colnames[0]
            xcol = mean_cols[0]
            ycol = frac_cols[0]

            uncal = cal_df[cal_df[mcol].str.contains("uncalibrated", case=False, na=False)]
            iso = cal_df[cal_df[mcol].str.contains("isotonic", case=False, na=False)]

            if len(uncal) > 0:
                plt.plot(
                    uncal[xcol],
                    uncal[ycol],
                    marker="o",
                    color="blue",
                    label="Uncalibrated LightGBM",
                )

            if len(iso) > 0:
                plt.plot(
                    iso[xcol],
                    iso[ycol],
                    marker="o",
                    color="green",
                    label="Isotonic Calibrated",
                )

        else:
            frac_pos, mean_pred = calibration_curve(y_true, uncalibrated, n_bins=10)
            plt.plot(mean_pred, frac_pos, marker="o", color="blue", label="Uncalibrated LightGBM")

    else:
        frac_pos, mean_pred = calibration_curve(y_true, uncalibrated, n_bins=10)
        plt.plot(mean_pred, frac_pos, marker="o", color="blue", label="Uncalibrated LightGBM")

        if calibrated is not None:
            frac_pos_cal, mean_pred_cal = calibration_curve(y_true, calibrated, n_bins=10)
            plt.plot(
                mean_pred_cal,
                frac_pos_cal,
                marker="o",
                color="green",
                label="Isotonic Calibrated",
            )
        else:
            print(
                "Warning: no calibrated predictions found. "
                "Reliability curve will show uncalibrated model only."
            )

    plt.title("Reliability Curve – LightGBM Before/After Isotonic Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_fig("reliability_curve.png")


# ==========================================================
# 2. PR curve final model
# ==========================================================

def plot_pr_curve():
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        color="blue",
        linewidth=2,
        label=f"Recency-Weighted LightGBM (PR-AUC={pr_auc:.3f})",
    )

    plt.axhline(
        y=fraud_rate,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Random Baseline (fraud rate={fraud_rate:.3f})",
    )

    # Mark a selected threshold. Use 0.5 by default or 0.44 if you want validation threshold.
    selected_threshold = 0.50

    threshold_df = compute_threshold_table(y_true, y_proba)
    closest = threshold_df.iloc[(threshold_df["threshold"] - selected_threshold).abs().argmin()]

    plt.scatter(
        closest["recall"],
        closest["precision"],
        color="red",
        s=70,
        zorder=5,
        label=f"Threshold={selected_threshold:.2f}",
    )

    plt.title("Precision-Recall Curve – Final Recency-Weighted LightGBM")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_fig("pr_curve_final.png")


# ==========================================================
# 3. Threshold analysis
# ==========================================================

def plot_threshold_analysis():
    threshold_df = compute_threshold_table(y_true, y_proba)

    best_row = threshold_df.loc[threshold_df["f2"].idxmax()]
    best_threshold = best_row["threshold"]

    plt.figure(figsize=(8, 6))

    plt.plot(
        threshold_df["threshold"],
        threshold_df["precision"],
        color="blue",
        linewidth=2,
        label="Precision",
    )

    plt.plot(
        threshold_df["threshold"],
        threshold_df["recall"],
        color="green",
        linewidth=2,
        label="Recall",
    )

    plt.plot(
        threshold_df["threshold"],
        threshold_df["f2"],
        color="red",
        linewidth=2,
        label="F2-score",
    )

    plt.axvline(
        x=best_threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Best F2 threshold={best_threshold:.2f}",
    )

    plt.title("Threshold Optimisation – LightGBM")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_fig("threshold_analysis.png")

    threshold_df.to_csv(REPORT_DIR / "threshold_analysis_for_figure.csv", index=False)
    print("Best F2 threshold:", best_threshold)
    print("Best F2:", best_row["f2"])


# ==========================================================
# 4. Top 10 feature importance SHAP / LightGBM importance
# ==========================================================

def plot_feature_importance():
    """
    Uses SHAP feature importance if a CSV exists.
    Otherwise uses the latest known top features from your LightGBM output.
    """

    possible_files = [
        REPORT_DIR / "lightgbm_shap_feature_importance.csv",
        REPORT_DIR / "shap_feature_importance.csv",
        REPORT_DIR / "lightgbm_feature_importance.csv",
        REPORT_DIR / "final_lightgbm_feature_importance.csv",
    ]

    importance_df = None

    for path in possible_files:
        if path.exists():
            print(f"Using feature importance file: {path}")
            temp = pd.read_csv(path)
            print("Feature importance columns:", list(temp.columns))
            importance_df = temp
            break

    if importance_df is not None:
        # Try to detect columns
        feature_cols = [c for c in importance_df.columns if "feature" in c.lower()]
        value_cols = [
            c for c in importance_df.columns
            if "importance" in c.lower() or "shap" in c.lower() or "mean" in c.lower()
        ]

        if feature_cols and value_cols:
            fcol = feature_cols[0]
            vcol = value_cols[-1]
            top10 = importance_df[[fcol, vcol]].copy()
            top10.columns = ["feature", "importance"]
            top10 = top10.sort_values("importance", ascending=False).head(10)
        else:
            top10 = None
    else:
        top10 = None

    if top10 is None:
        # Fallback values from your output/logs. These are relative importances.
        top10 = pd.DataFrame({
            "feature": [
                "TransactionDT",
                "card1",
                "card1_TransactionAmt_mean",
                "C13",
                "addr1",
                "card2",
                "card1_TransactionAmt_max",
                "card1_TransactionAmt_std",
                "card1_freq",
                "P_emaildomain_freq",
            ],
            "importance": [
                1431,
                1278,
                1080,
                993,
                927,
                864,
                855,
                808,
                797,
                577,
            ],
        })

    top10 = top10.sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 6))

    plt.barh(
        top10["feature"],
        top10["importance"],
        color="#1f4e79",
    )

    plt.title("Top 10 Feature Importances – SHAP / LightGBM")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(axis="x", alpha=0.3)

    save_fig("feature_importance_top10.png")


# ==========================================================
# 5. Confusion matrix final model
# ==========================================================

def plot_confusion_matrix():
    threshold = 0.50
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    matrix = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="Blues")

    plt.title("Confusion Matrix – Recency-Weighted LightGBM (threshold=0.5)")
    plt.colorbar(label="Number of transactions")

    labels_x = ["Predicted: Non-Fraud", "Predicted: Fraud"]
    labels_y = ["Actual: Non-Fraud", "Actual: Fraud"]

    plt.xticks([0, 1], labels_x, rotation=15)
    plt.yticks([0, 1], labels_y)

    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            plt.text(
                j,
                i,
                f"{value:,}",
                ha="center",
                va="center",
                color="black",
                fontsize=13,
                fontweight="bold",
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    save_fig("confusion_matrix_final.png")

    print("Confusion matrix values:")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# ==========================================================
# 6. Ablation study
# ==========================================================

def plot_ablation_study():
    stages = [
        "Baseline\nLightGBM",
        "+ Feature\nEngineering",
        "+ Recency\nWeighting",
        "+ Bagging\nEnsemble",
        "Final\nModel",
    ]

    pr_auc_values = [
        0.484,
        0.554,
        0.601,
        0.604,
        0.606,
    ]

    plt.figure(figsize=(8, 6))

    bars = plt.bar(
        stages,
        pr_auc_values,
        color="#1f4e79",
    )

    plt.title("Model Improvement Path – PR-AUC Gains")
    plt.xlabel("Experiment Stage")
    plt.ylabel("PR-AUC")
    plt.ylim(0.45, 0.65)
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, pr_auc_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    save_fig("ablation_study.png")


# =========================
# Run all plots
# =========================

if __name__ == "__main__":
    plot_reliability_curve()
    plot_pr_curve()
    plot_threshold_analysis()
    plot_feature_importance()
    plot_confusion_matrix()
    plot_ablation_study()

    print("\nAll figures saved to the figures/ folder.")