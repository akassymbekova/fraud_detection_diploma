import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, text, color, fontsize=9, edgecolor="#374151"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.3,
        edgecolor=edgecolor,
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
        color="#111827",
        linespacing=1.15,
    )


def add_arrow(ax, x1, y1, x2, y2, curved=False):
    connectionstyle = "arc3,rad=0.12" if curved else "arc3,rad=0.0"
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.3,
        color="#374151",
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(
        0.4,
        7.45,
        "Model 2: LightGBM Bagging Ensemble",
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="center",
        color="#111827",
    )

    # Colors
    input_color = "#e0f2fe"
    model_color = "#fef9c3"
    avg_color = "#ffedd5"
    output_color = "#dcfce7"
    note_color = "#f3f4f6"

    # Input box
    add_box(
        ax,
        0.7,
        3.35,
        2.5,
        1.3,
        "Engineered\nTransaction Features\n\nBehavioural\nVelocity\nRelational",
        input_color,
        fontsize=9,
    )

    # Six LightGBM boxes
    model_x = 4.3
    model_w = 2.2
    model_h = 0.55
    y_positions = [5.9, 5.15, 4.4, 3.65, 2.9, 2.15]

    model_labels = [
        "LightGBM 1\nseed = 42",
        "LightGBM 2\nseed = 101",
        "LightGBM 3\nseed = 202",
        "LightGBM 4\nseed = 303",
        "LightGBM 5\nseed = 404",
        "LightGBM 6\nseed = 505",
    ]

    for y, label in zip(y_positions, model_labels):
        add_box(
            ax,
            model_x,
            y,
            model_w,
            model_h,
            label,
            model_color,
            fontsize=8,
        )
        add_arrow(ax, 3.2, 4.0, model_x, y + model_h / 2, curved=True)

    # Average probability box
    add_box(
        ax,
        7.6,
        3.35,
        2.3,
        1.3,
        "Average\nPredicted\nProbabilities\n\nP_final = 1/6 Σ P_k",
        avg_color,
        fontsize=9,
    )

    # Arrows from models to average
    for y in y_positions:
        add_arrow(ax, model_x + model_w, y + model_h / 2, 7.6, 4.0, curved=True)

    # Output box
    add_box(
        ax,
        11.0,
        3.35,
        2.4,
        1.3,
        "Stable Fraud\nRisk Score\n\nFraud probability\n0 to 1",
        output_color,
        fontsize=9,
    )

    add_arrow(ax, 9.9, 4.0, 11.0, 4.0)

    # Bottom explanation boxes
    add_box(
        ax,
        1.0,
        0.8,
        3.4,
        0.85,
        "Different random seeds\nreduce dependence on\none model initialization",
        note_color,
        fontsize=8.5,
    )

    add_box(
        ax,
        5.2,
        0.8,
        3.4,
        0.85,
        "Different recency weights\nand hyperparameters\nincrease robustness",
        note_color,
        fontsize=8.5,
    )

    add_box(
        ax,
        9.4,
        0.8,
        3.4,
        0.85,
        "Averaging probabilities\nreduces variance and\nstabilizes predictions",
        note_color,
        fontsize=8.5,
    )

    # Small labels
    ax.text(
        4.25,
        6.75,
        "Bagging: six LightGBM variants",
        fontsize=11,
        fontweight="bold",
        color="#374151",
    )

    ax.text(
        7.65,
        5.05,
        "Ensemble aggregation",
        fontsize=11,
        fontweight="bold",
        color="#374151",
    )

    ax.text(
        11.0,
        5.05,
        "Output",
        fontsize=11,
        fontweight="bold",
        color="#374151",
    )

    # Results note
    ax.text(
        7.0,
        6.95,
        "Final test performance: PR-AUC = 0.604 | ROC-AUC = 0.930 | F2 = 0.569 | MCC = 0.581",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#111827",
        fontweight="bold",
    )

    output_path = FIG_DIR / "lightgbm_bagging_ensemble_diagram.png"
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()