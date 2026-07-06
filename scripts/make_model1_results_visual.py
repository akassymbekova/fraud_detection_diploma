import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def make_metric_bar_chart():
    metrics = ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F2", "MCC"]
    values = [0.606, 0.928, 0.660, 0.537, 0.558, 0.582]

    plt.figure(figsize=(9, 5.2))
    bars = plt.bar(metrics, values)

    plt.title("Recency-weighted LightGBM – Final Test Metrics", fontsize=15, fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    output_path = FIG_DIR / "model1_recency_lightgbm_metrics.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def make_confusion_matrix():
    # Values from lgbm_a result
    tn = 112922
    fp = 1122
    fn = 1883
    tp = 2181

    matrix = np.array([
        [tn, fp],
        [fn, tp],
    ])

    plt.figure(figsize=(7, 5.8))
    plt.imshow(matrix, cmap="Blues")

    plt.title("Confusion Matrix – Recency-weighted LightGBM", fontsize=15, fontweight="bold")
    plt.colorbar(label="Number of transactions")

    plt.xticks([0, 1], ["Predicted\nNon-Fraud", "Predicted\nFraud"])
    plt.yticks([0, 1], ["Actual\nNon-Fraud", "Actual\nFraud"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{matrix[i, j]:,}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="black",
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()
    output_path = FIG_DIR / "model1_recency_lightgbm_confusion_matrix.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def make_baseline_comparison():
    models = ["Baseline\nLightGBM", "Recency-weighted\nLightGBM"]
    pr_auc = [0.484, 0.606]
    roc_auc = [0.881, 0.928]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5.2))

    bars1 = plt.bar(x - width / 2, pr_auc, width, label="PR-AUC")
    bars2 = plt.bar(x + width / 2, roc_auc, width, label="ROC-AUC")

    plt.title("Baseline vs Recency-weighted LightGBM", fontsize=15, fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(x, models)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    output_path = FIG_DIR / "model1_baseline_comparison.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    make_metric_bar_chart()
    make_confusion_matrix()
    make_baseline_comparison()