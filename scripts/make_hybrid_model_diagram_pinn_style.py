import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, text, fc="#F2F2F2", ec="#666666", fs=12, bold=False):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.04",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.4,
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        color="#111111",
        linespacing=1.25,
    )


def add_arrow(ax, x1, y1, x2, y2, rad=0.0):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=17,
        linewidth=1.5,
        color="#444444",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(
        0.7,
        8.35,
        "How the Hybrid XGBoost + LightGBM Model Works",
        fontsize=27,
        fontweight="bold",
        ha="left",
        va="center",
        color="#111111",
    )

    ax.text(
        0.72,
        7.88,
        "Model input, internal prediction flow, weighted blending and evaluation metrics",
        fontsize=14,
        ha="left",
        va="center",
        color="#444444",
    )

    # Grey academic palette
    input_fc = "#F2F2F2"
    feature_fc = "#E8E8E8"
    weight_fc = "#F4EFE8"
    model_fc = "#EDEDED"
    probability_fc = "#F7F7F7"
    blend_fc = "#E6E6E6"
    output_fc = "#F3F3F3"
    metrics_fc = "#EFEFEF"

    # ===== TOP INPUT BLOCKS =====
    add_box(
        ax,
        0.8,
        6.35,
        3.25,
        1.2,
        "Training Data\n\nX_train: features\ny_train: fraud label",
        input_fc,
        ec="#777777",
        fs=12,
        bold=True,
    )

    add_box(
        ax,
        4.55,
        6.35,
        3.25,
        1.2,
        "Engineered Features\n\nBehavioural\nVelocity\nRelational",
        feature_fc,
        ec="#777777",
        fs=12,
        bold=True,
    )

    add_box(
        ax,
        8.3,
        6.35,
        3.25,
        1.2,
        "Training Weights\n\nclass imbalance weight\nrecency weight",
        weight_fc,
        ec="#777777",
        fs=12,
        bold=True,
    )

    # Model input
    add_box(
        ax,
        4.25,
        5.0,
        4.0,
        0.85,
        "Model Input\nxᵢ + yᵢ + wᵢ",
        blend_fc,
        ec="#666666",
        fs=14,
        bold=True,
    )

    add_arrow(ax, 2.42, 6.35, 5.05, 5.85, rad=0.12)
    add_arrow(ax, 6.18, 6.35, 6.18, 5.85)
    add_arrow(ax, 9.93, 6.35, 7.35, 5.85, rad=-0.12)

    # ===== MODEL BRANCHES =====
    add_box(
        ax,
        1.15,
        3.45,
        3.8,
        1.05,
        "LightGBM Branch\n\nn_estimators\nlearning_rate\nnum_leaves\nscale_pos_weight",
        model_fc,
        ec="#666666",
        fs=11,
        bold=True,
    )

    add_box(
        ax,
        7.55,
        3.45,
        3.8,
        1.05,
        "XGBoost Branch\n\nn_estimators\nmax_depth\nlearning_rate\nscale_pos_weight",
        model_fc,
        ec="#666666",
        fs=11,
        bold=True,
    )

    add_arrow(ax, 6.25, 5.0, 3.05, 4.5, rad=0.16)
    add_arrow(ax, 6.25, 5.0, 9.45, 4.5, rad=-0.16)

    # Probability outputs
    add_box(
        ax,
        1.45,
        2.15,
        3.2,
        0.75,
        "Output 1\nP_LGB = fraud probability",
        probability_fc,
        ec="#777777",
        fs=12,
        bold=True,
    )

    add_box(
        ax,
        7.85,
        2.15,
        3.2,
        0.75,
        "Output 2\nP_XGB = fraud probability",
        probability_fc,
        ec="#777777",
        fs=12,
        bold=True,
    )

    add_arrow(ax, 3.05, 3.45, 3.05, 2.9)
    add_arrow(ax, 9.45, 3.45, 9.45, 2.9)

    # Weighted blend
    add_box(
        ax,
        4.55,
        1.05,
        4.3,
        0.9,
        "Weighted Probability Blend\nP_final = 0.53·P_XGB + 0.47·P_LGB",
        blend_fc,
        ec="#555555",
        fs=12.5,
        bold=True,
    )

    add_arrow(ax, 4.65, 2.4, 5.45, 1.95, rad=-0.1)
    add_arrow(ax, 7.85, 2.4, 7.95, 1.95, rad=0.1)

    # Final output and decision
    add_box(
        ax,
        9.85,
        1.05,
        2.2,
        0.9,
        "Final Output\nfraud score 0–1",
        output_fc,
        ec="#666666",
        fs=12,
        bold=True,
    )

    add_arrow(ax, 8.85, 1.5, 9.85, 1.5)

    add_box(
        ax,
        12.7,
        1.05,
        2.3,
        0.9,
        "Threshold Decision\nFraud / Non-Fraud",
        output_fc,
        ec="#666666",
        fs=12,
        bold=True,
    )

    add_arrow(ax, 12.05, 1.5, 12.7, 1.5)

    # ===== METRICS BLOCK =====
    add_box(
        ax,
        12.25,
        5.0,
        3.15,
        2.25,
        "Evaluation Metrics\n\nPR-AUC\nROC-AUC\nPrecision / Recall\nF2-score\nMCC",
        metrics_fc,
        ec="#666666",
        fs=12,
        bold=True,
    )

    add_arrow(ax, 13.85, 1.95, 13.85, 5.0)

    output_path = FIG_DIR / "hybrid_model_how_it_works_grey.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()