import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, text, color, fontsize=9, edge="#374151"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.3,
        edgecolor=edge,
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
        linewidth=1.35,
        color="#374151",
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(
        0.4,
        7.45,
        "Hybrid XGBoost + LightGBM Model Architecture",
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="center",
        color="#111827",
    )

    # Colors
    feature_color = "#e0f2fe"
    vector_color = "#dcfce7"
    weight_color = "#fef9c3"
    lgbm_color = "#ede9fe"
    xgb_color = "#fce7f3"
    blend_color = "#ffedd5"
    output_color = "#dbeafe"
    decision_color = "#fee2e2"
    note_color = "#f3f4f6"

    # Feature blocks
    add_box(
        ax,
        0.5,
        5.8,
        2.4,
        0.9,
        "Behavioural\nFeatures\n\ncard freq\namount stats",
        feature_color,
        fontsize=8.5,
    )

    add_box(
        ax,
        3.25,
        5.8,
        2.4,
        0.9,
        "Velocity\nFeatures\n\n1h / 6h / 24h\ncounts & sums",
        feature_color,
        fontsize=8.5,
    )

    add_box(
        ax,
        6.0,
        5.8,
        2.4,
        0.9,
        "Relational\nFeatures\n\ncard+addr\ncard+product",
        feature_color,
        fontsize=8.5,
    )

    # Feature vector
    add_box(
        ax,
        3.0,
        4.6,
        3.0,
        0.75,
        "Feature Vector  xᵢ\nEngineered fraud features",
        vector_color,
        fontsize=9,
    )

    add_arrow(ax, 1.7, 5.8, 3.3, 5.35, curved=True)
    add_arrow(ax, 4.45, 5.8, 4.5, 5.35)
    add_arrow(ax, 7.2, 5.8, 5.6, 5.35, curved=True)

    # Recency weighting
    add_box(
        ax,
        3.0,
        3.55,
        3.0,
        0.7,
        "Recency Weighting\nwᵢ = 1 old,  wᵢ = α recent",
        weight_color,
        fontsize=9,
    )
    add_arrow(ax, 4.5, 4.6, 4.5, 4.25)

    # Split into two models
    add_box(
        ax,
        1.7,
        2.1,
        2.7,
        0.85,
        "LightGBM\nGradient Boosting\nTree 1 + ... + Tree N",
        lgbm_color,
        fontsize=8.5,
    )

    add_box(
        ax,
        5.2,
        2.1,
        2.7,
        0.85,
        "XGBoost\nGradient Boosting\nTree 1 + ... + Tree N",
        xgb_color,
        fontsize=8.5,
    )

    add_arrow(ax, 4.5, 3.55, 3.05, 2.95, curved=True)
    add_arrow(ax, 4.5, 3.55, 6.55, 2.95, curved=True)

    # Probability outputs
    add_box(
        ax,
        1.7,
        1.1,
        2.7,
        0.55,
        "P_LGB = fraud probability",
        lgbm_color,
        fontsize=8.5,
    )

    add_box(
        ax,
        5.2,
        1.1,
        2.7,
        0.55,
        "P_XGB = fraud probability",
        xgb_color,
        fontsize=8.5,
    )

    add_arrow(ax, 3.05, 2.1, 3.05, 1.65)
    add_arrow(ax, 6.55, 2.1, 6.55, 1.65)

    # Weighted blending
    add_box(
        ax,
        9.0,
        1.55,
        2.8,
        1.0,
        "Weighted Blending\n\nP_final = 0.53·P_XGB\n+ 0.47·P_LGB",
        blend_color,
        fontsize=9,
    )

    add_arrow(ax, 4.4, 1.38, 9.0, 2.05, curved=True)
    add_arrow(ax, 7.9, 1.38, 9.0, 2.05, curved=True)

    # Final fraud probability
    add_box(
        ax,
        12.3,
        1.75,
        2.0,
        0.75,
        "Fraud Probability\nP_final ∈ [0, 1]",
        output_color,
        fontsize=9,
    )

    add_arrow(ax, 11.8, 2.05, 12.3, 2.1)

    # Decision
    add_box(
        ax,
        12.3,
        0.55,
        2.0,
        0.75,
        "Threshold Decision\nFraud / Non-Fraud",
        decision_color,
        fontsize=9,
    )

    add_arrow(ax, 13.3, 1.75, 13.3, 1.3)

    # Formula note
    ax.text(
        10.6,
        4.85,
        r"$P_{\mathrm{final}} = 0.53 \cdot P_{\mathrm{XGB}} + 0.47 \cdot P_{\mathrm{LGB}}$",
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="center",
        color="#111827",
    )

    ax.text(
        10.6,
        4.35,
        "Both models use the same engineered feature vector and recency-weighted training.",
        fontsize=10,
        ha="center",
        va="center",
        color="#374151",
    )

    # Bottom note
    add_box(
        ax,
        0.7,
        0.25,
        10.8,
        0.55,
        "Hybrid idea: combine LightGBM efficiency with XGBoost robustness through weighted probability blending.",
        note_color,
        fontsize=9,
    )

    output_path = FIG_DIR / "hybrid_xgboost_lightgbm_model_architecture.png"
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()