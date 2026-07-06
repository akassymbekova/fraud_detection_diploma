import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def add_box(ax, x, y, w, h, text, color, fontsize=10):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.5,
        edgecolor="#333333",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="black",
        wrap=True,
    )

def add_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=18,
        linewidth=1.8,
        color="#333333",
    )
    ax.add_patch(arrow)

def main():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title
    ax.text(
        8,
        8.6,
        "Proposed Hybrid ML/DL Pipeline for Fraud Detection",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )

    # Main horizontal pipeline boxes
    y = 6.7
    w = 2.1
    h = 0.9
    gap = 0.35

    boxes = [
        (
            "Data Sources\nIEEE-CIS\nCredit Card\nSynthetic Kazakh",
            "#dbeafe",
        ),
        (
            "Leakage-Aware\nPreprocessing\nTime-based split\nEncoding / Imputation",
            "#dcfce7",
        ),
        (
            "Feature\nEngineering\nBehavioural\nVelocity\nRelational",
            "#bbf7d0",
        ),
        (
            "Model Training\nML + DL\nComparison",
            "#fde68a",
        ),
        (
            "Evaluation\nFuture Test Set\nPR-AUC / F2 / MCC",
            "#fecaca",
        ),
        (
            "Explainability\nCalibration\nMonitoring\nSHAP / PSI / ECE",
            "#e9d5ff",
        ),
        (
            "Dashboard\nRisk Score\nDecision Support",
            "#e5e7eb",
        ),
    ]

    start_x = 0.5
    centers = []

    for i, (text, color) in enumerate(boxes):
        x = start_x + i * (w + gap)
        add_box(ax, x, y, w, h, text, color, fontsize=9)
        centers.append((x + w / 2, y + h / 2))

        if i > 0:
            prev_x = start_x + (i - 1) * (w + gap)
            add_arrow(ax, prev_x + w, y + h / 2, x, y + h / 2)

    # Model training split: ML and DL tracks
    model_x = start_x + 3 * (w + gap)
    model_center_x = model_x + w / 2

    add_arrow(ax, model_center_x, y, model_center_x - 2.0, 5.1)
    add_arrow(ax, model_center_x, y, model_center_x + 2.0, 5.1)

    # ML branch
    add_box(
        ax,
        model_center_x - 4.0,
        4.0,
        3.5,
        1.1,
        "Machine Learning Models\nLogistic Regression, RF\nXGBoost, CatBoost\nRecency-weighted LightGBM",
        "#fed7aa",
        fontsize=9,
    )

    add_box(
        ax,
        model_center_x - 4.0,
        2.6,
        3.5,
        0.9,
        "Final ML Model\nRecency-weighted LightGBM\nPR-AUC = 0.606",
        "#fdba74",
        fontsize=10,
    )

    add_arrow(ax, model_center_x - 2.25, 4.0, model_center_x - 2.25, 3.5)

    # DL branch
    add_box(
        ax,
        model_center_x + 0.5,
        4.0,
        3.5,
        1.1,
        "Deep Learning Models\nTabNet\nMLP + BatchNorm + Dropout\nFocal Loss tested",
        "#ddd6fe",
        fontsize=9,
    )

    add_box(
        ax,
        model_center_x + 0.5,
        2.6,
        3.5,
        0.9,
        "DL Benchmark Result\nBest MLP PR-AUC = 0.384\nDL underperformed LightGBM",
        "#c4b5fd",
        fontsize=9,
    )

    add_arrow(ax, model_center_x + 2.25, 4.0, model_center_x + 2.25, 3.5)

    # Final selection box
    add_box(
        ax,
        5.4,
        1.0,
        5.2,
        1.0,
        "Final Selected Approach: Hybrid LightGBM-based Fraud Risk Scoring System\nLeakage-aware features + recency weighting + bagging + calibration + SHAP",
        "#bfdbfe",
        fontsize=10,
    )

    add_arrow(ax, model_center_x - 2.25, 2.6, 7.2, 2.0)
    add_arrow(ax, model_center_x + 2.25, 2.6, 8.8, 2.0)

    # Footer metrics
    ax.text(
        8,
        0.45,
        "Final results: PR-AUC = 0.606 | ROC-AUC = 0.928 | F2 = 0.558 | MCC = 0.582",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#1f2937",
    )

    output_path = FIG_DIR / "hybrid_ml_dl_pipeline_flowchart.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()