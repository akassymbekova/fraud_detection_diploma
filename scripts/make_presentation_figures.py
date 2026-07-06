from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_model_results_table():
    results = pd.DataFrame([
        {
            "Model": "Baseline LightGBM",
            "PR-AUC": 0.484,
            "ROC-AUC": 0.881,
            "Precision": 0.337,
            "Recall": 0.544,
            "F2": np.nan,
            "MCC": np.nan,
        },
        {
            "Model": "TabNet",
            "PR-AUC": 0.253,
            "ROC-AUC": 0.796,
            "Precision": 0.656,
            "Recall": 0.125,
            "F2": 0.149,
            "MCC": 0.277,
        },
        {
            "Model": "CatBoost",
            "PR-AUC": 0.508,
            "ROC-AUC": 0.915,
            "Precision": 0.208,
            "Recall": 0.759,
            "F2": 0.496,
            "MCC": 0.360,
        },
        {
            "Model": "XGBoost",
            "PR-AUC": 0.549,
            "ROC-AUC": 0.921,
            "Precision": 0.311,
            "Recall": 0.684,
            "F2": 0.552,
            "MCC": 0.434,
        },
        {
            "Model": "Bagged LGBM Ensemble",
            "PR-AUC": 0.604,
            "ROC-AUC": 0.930,
            "Precision": 0.637,
            "Recall": 0.554,
            "F2": 0.569,
            "MCC": 0.581,
        },
        {
            "Model": "Best Recency LGBM",
            "PR-AUC": 0.606,
            "ROC-AUC": 0.928,
            "Precision": 0.660,
            "Recall": 0.537,
            "F2": 0.558,
            "MCC": 0.582,
        },
    ])

    results.to_csv(OUTPUT_DIR / "final_model_results_table.csv", index=False)
    return results


def plot_pr_auc_comparison(results):
    df = results.sort_values("PR-AUC", ascending=True)

    plt.figure(figsize=(10, 5.5))
    plt.barh(df["Model"], df["PR-AUC"])
    plt.title("Model Comparison by PR-AUC", fontsize=16, weight="bold")
    plt.xlabel("PR-AUC")
    plt.xlim(0, 0.70)

    for index, value in enumerate(df["PR-AUC"]):
        plt.text(value + 0.01, index, f"{value:.3f}", va="center", fontsize=10)

    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_model_comparison_pr_auc.png", dpi=300)
    plt.close()


def plot_roc_auc_comparison(results):
    df = results.sort_values("ROC-AUC", ascending=True)

    plt.figure(figsize=(10, 5.5))
    plt.barh(df["Model"], df["ROC-AUC"])
    plt.title("Model Comparison by ROC-AUC", fontsize=16, weight="bold")
    plt.xlabel("ROC-AUC")
    plt.xlim(0.70, 1.00)

    for index, value in enumerate(df["ROC-AUC"]):
        plt.text(value + 0.005, index, f"{value:.3f}", va="center", fontsize=10)

    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_model_comparison_roc_auc.png", dpi=300)
    plt.close()


def plot_precision_recall_tradeoff(results):
    plt.figure(figsize=(7, 5.5))

    for _, row in results.iterrows():
        plt.scatter(row["Recall"], row["Precision"], s=90)
        plt.text(
            row["Recall"] + 0.008,
            row["Precision"] + 0.008,
            row["Model"],
            fontsize=8,
        )

    plt.title("Precision–Recall Trade-off", fontsize=16, weight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0.05, 0.82)
    plt.ylim(0.05, 0.75)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_precision_recall_tradeoff.png", dpi=300)
    plt.close()


def plot_improvement_path():
    stages = [
        "Strict time-based\nbaseline",
        "Train+valid\nretraining",
        "Relational +\nvelocity features",
        "Bagged LGBM\nensemble",
        "Best recency\nLGBM",
    ]

    pr_auc = [0.484, 0.554, 0.601, 0.604, 0.606]

    plt.figure(figsize=(10, 5.5))
    plt.plot(stages, pr_auc, marker="o", linewidth=2)

    plt.title("Model Improvement Path", fontsize=16, weight="bold")
    plt.ylabel("PR-AUC")
    plt.ylim(0.45, 0.63)

    for i, value in enumerate(pr_auc):
        plt.text(i, value + 0.004, f"{value:.3f}", ha="center", fontsize=10)

    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_model_improvement_path.png", dpi=300)
    plt.close()


def plot_final_model_metrics():
    metrics = {
        "PR-AUC": 0.606,
        "ROC-AUC": 0.928,
        "Precision": 0.660,
        "Recall": 0.537,
        "F2-score": 0.558,
        "MCC": 0.582,
    }

    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 5.5))
    plt.barh(names, values)
    plt.title("Best Recency-Weighted LightGBM: Test Metrics", fontsize=15, weight="bold")
    plt.xlim(0, 1)

    for i, value in enumerate(values):
        plt.text(value + 0.015, i, f"{value:.3f}", va="center", fontsize=10)

    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_final_lgbm_metrics.png", dpi=300)
    plt.close()


def plot_tabnet_vs_lightgbm():
    models = ["TabNet", "Best Recency LGBM"]
    pr_auc = [0.253, 0.606]
    roc_auc = [0.796, 0.928]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(7, 5.5))
    plt.bar(x - width / 2, pr_auc, width, label="PR-AUC")
    plt.bar(x + width / 2, roc_auc, width, label="ROC-AUC")

    plt.title("Deep Learning Comparison: TabNet vs LightGBM", fontsize=14, weight="bold")
    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    for i, value in enumerate(pr_auc):
        plt.text(i - width / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=10)

    for i, value in enumerate(roc_auc):
        plt.text(i + width / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_tabnet_vs_lightgbm.png", dpi=300)
    plt.close()


def create_slide_style_summary_image(results):
    """
    Creates one image that looks like a ready-made slide:
    title + metrics table + two charts.
    """
    fig = plt.figure(figsize=(16, 9))

    fig.suptitle(
        "Machine Learning Models",
        fontsize=30,
        weight="bold",
        x=0.06,
        ha="left",
    )

    fig.text(
        0.06,
        0.88,
        "Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM, TabNet, Bagging Ensemble",
        fontsize=13,
        weight="bold",
    )

    # Table area
    ax_table = fig.add_axes([0.05, 0.48, 0.52, 0.34])
    ax_table.axis("off")

    table_df = results.copy()
    table_df = table_df[["Model", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F2", "MCC"]]
    table_df = table_df.round(3)
    table_df = table_df.fillna("—")

    table = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.05, 1.45)

    ax_table.set_title("Table: Final model comparison", fontsize=13, weight="bold", pad=10)

    # PR-AUC bar chart
    ax_bar = fig.add_axes([0.62, 0.50, 0.33, 0.32])
    chart_df = results.sort_values("PR-AUC", ascending=True)
    ax_bar.barh(chart_df["Model"], chart_df["PR-AUC"])
    ax_bar.set_title("PR-AUC comparison", fontsize=12, weight="bold")
    ax_bar.set_xlim(0, 0.70)
    ax_bar.grid(axis="x", alpha=0.25)

    # Improvement line chart
    ax_line = fig.add_axes([0.08, 0.10, 0.42, 0.27])
    stages = [
        "Baseline",
        "Retrain",
        "+ FE",
        "Bagging",
        "Best LGBM",
    ]
    pr_auc_path = [0.484, 0.554, 0.601, 0.604, 0.606]
    ax_line.plot(stages, pr_auc_path, marker="o", linewidth=2)
    ax_line.set_title("Improvement path", fontsize=12, weight="bold")
    ax_line.set_ylabel("PR-AUC")
    ax_line.set_ylim(0.45, 0.63)
    ax_line.grid(axis="y", alpha=0.25)

    for i, value in enumerate(pr_auc_path):
        ax_line.text(i, value + 0.004, f"{value:.3f}", ha="center", fontsize=9)

    # Precision-recall scatter
    ax_scatter = fig.add_axes([0.58, 0.10, 0.35, 0.27])
    ax_scatter.scatter(results["Recall"], results["Precision"], s=70)

    for _, row in results.iterrows():
        ax_scatter.text(
            row["Recall"] + 0.007,
            row["Precision"] + 0.007,
            row["Model"],
            fontsize=7,
        )

    ax_scatter.set_title("Precision–Recall trade-off", fontsize=12, weight="bold")
    ax_scatter.set_xlabel("Recall")
    ax_scatter.set_ylabel("Precision")
    ax_scatter.set_xlim(0.05, 0.82)
    ax_scatter.set_ylim(0.05, 0.75)
    ax_scatter.grid(alpha=0.25)

    fig.text(
        0.06,
        0.035,
        "Final result: PR-AUC improved from 0.484 to 0.606 on the future holdout test set.",
        fontsize=14,
        weight="bold",
    )

    plt.savefig(OUTPUT_DIR / "07_slide_machine_learning_models.png", dpi=300)
    plt.close()


def main():
    results = save_model_results_table()

    plot_pr_auc_comparison(results)
    plot_roc_auc_comparison(results)
    plot_precision_recall_tradeoff(results)
    plot_improvement_path()
    plot_final_model_metrics()
    plot_tabnet_vs_lightgbm()
    create_slide_style_summary_image(results)

    print("Done. Figures saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()