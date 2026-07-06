import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, text, facecolor, edgecolor="#4b5563", fontsize=8):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        linespacing=1.15,
    )


def add_arrow(ax, x1, y1, x2, y2, curved=False):
    connectionstyle = "arc3,rad=0.15" if curved else "arc3,rad=0.0"
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.2,
        color="#374151",
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        0.4,
        8.45,
        "Work Architecture",
        fontsize=30,
        fontweight="bold",
        ha="left",
        va="center",
        color="#111827",
    )

    blue = "#e0f2fe"
    green = "#dcfce7"
    yellow = "#fef9c3"
    orange = "#ffedd5"
    red = "#fee2e2"
    purple = "#ede9fe"
    gray = "#f3f4f6"

    # Data sources
    add_box(
        ax, 0.8, 6.25, 2.2, 0.8,
        "IEEE-CIS\nFraud Dataset\n590k records",
        blue,
        fontsize=8,
    )

    add_box(
        ax, 0.8, 4.95, 2.2, 0.8,
        "Credit Card\nBenchmark\n284k records",
        blue,
        fontsize=8,
    )

    add_box(
        ax, 0.8, 3.65, 2.2, 0.8,
        "Synthetic Kazakh\nTransactions\nAdditional data",
        blue,
        fontsize=8,
    )

    # Preprocessing and feature engineering
    add_box(
        ax, 3.9, 5.65, 2.5, 1.05,
        "Leakage-Aware\nPreprocessing\n\nTime-based split\nEncoding, imputation",
        green,
        fontsize=7.8,
    )

    add_box(
        ax, 3.9, 3.95, 2.5, 1.1,
        "Feature Engineering\n\nBehavioural\nVelocity\nRelational features",
        green,
        fontsize=7.8,
    )

    # ML/DL
    add_box(
        ax, 7.4, 5.8, 2.7, 1.05,
        "Machine Learning\nModels\n\nLR, RF, XGBoost\nCatBoost, LightGBM",
        yellow,
        fontsize=7.8,
    )

    add_box(
        ax, 7.4, 3.75, 2.7, 1.05,
        "Deep Learning\nBaselines\n\nTabNet\nMLP",
        purple,
        fontsize=7.8,
    )

    # Selected models
    add_box(
        ax, 11.0, 5.45, 2.6, 1.05,
        "Final Model\n\nRecency-weighted\nLightGBM\nPR-AUC = 0.606",
        orange,
        fontsize=7.8,
    )

    add_box(
        ax, 11.0, 3.75, 2.6, 1.05,
        "Bagged Ensemble\n\n6 LightGBM models\nPR-AUC = 0.604",
        orange,
        fontsize=7.8,
    )

    # Outputs
    add_box(
        ax, 14.5, 6.2, 2.4, 0.85,
        "Future Test\nEvaluation\n\nPR-AUC, F2, MCC",
        red,
        fontsize=7.5,
    )

    add_box(
        ax, 14.5, 4.75, 2.4, 0.85,
        "Calibration\n\nIsotonic\nECE 0.093 → 0.007",
        red,
        fontsize=7.5,
    )

    add_box(
        ax, 14.5, 3.3, 2.4, 0.85,
        "Explainability\n& Monitoring\n\nSHAP + PSI",
        red,
        fontsize=7.5,
    )

    # Final system
    add_box(
        ax, 5.5, 1.0, 7.2, 1.0,
        "Final Fraud Risk Scoring System\n\nFraud probability • Risk level • Analyst support • Dashboard prototype",
        gray,
        fontsize=8.5,
    )

    # Arrows: data to preprocessing / features
    add_arrow(ax, 3.0, 6.65, 3.9, 6.25)
    add_arrow(ax, 3.0, 5.35, 3.9, 5.95)
    add_arrow(ax, 3.0, 4.05, 3.9, 4.5)

    # Preprocessing to feature engineering
    add_arrow(ax, 5.15, 5.65, 5.15, 5.05)

    # Feature engineering to models
    add_arrow(ax, 6.4, 4.55, 7.4, 6.25, curved=True)
    add_arrow(ax, 6.4, 4.45, 7.4, 4.25)

    # Models to selected models
    add_arrow(ax, 10.1, 6.25, 11.0, 5.95)
    add_arrow(ax, 10.1, 4.25, 11.0, 4.25)

    # Selected models to evaluation outputs
    add_arrow(ax, 13.6, 5.95, 14.5, 6.62)
    add_arrow(ax, 13.6, 5.85, 14.5, 5.18)
    add_arrow(ax, 13.6, 4.25, 14.5, 3.72)

    # Outputs to final system
    add_arrow(ax, 15.7, 3.3, 11.3, 2.0, curved=True)
    add_arrow(ax, 12.3, 3.75, 9.5, 2.0, curved=True)
    add_arrow(ax, 12.3, 5.45, 8.6, 2.0, curved=True)

    # Section labels
    ax.text(
        7.5,
        7.15,
        "Model comparison",
        fontsize=10,
        fontweight="bold",
        color="#374151",
    )

    ax.text(
        11.0,
        6.85,
        "Selected approach",
        fontsize=10,
        fontweight="bold",
        color="#374151",
    )

    ax.text(
        14.5,
        7.35,
        "Validation outputs",
        fontsize=10,
        fontweight="bold",
        color="#374151",
    )

    output_path = FIG_DIR / "work_architecture_fraud_detection.png"
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()