import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, text, facecolor="#F5F5F5", fontsize=8.5, lw=1.2):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor="black",
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
        color="black",
        linespacing=1.15,
    )


def add_arrow(ax, x1, y1, x2, y2, rad=0.0):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=13,
        linewidth=1.2,
        color="black",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # ================= TITLE =================
    ax.text(
        0.5,
        8.35,
        "Model Architecture",
        fontsize=30,
        fontweight="bold",
        ha="left",
        va="center",
    )

    # ================= LEFT: ARCHITECTURE FRAME =================
    left_x, left_y, left_w, left_h = 0.55, 0.75, 7.2, 6.85

    outer = FancyBboxPatch(
        (left_x, left_y),
        left_w,
        left_h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(outer)

    # Feature blocks
    add_box(
        ax,
        0.85,
        6.55,
        1.85,
        0.58,
        "Behavioural\nFeatures",
        "#E6F2FF",
        fontsize=8,
    )
    add_box(
        ax,
        3.18,
        6.55,
        1.85,
        0.58,
        "Velocity\nFeatures",
        "#E6F2FF",
        fontsize=8,
    )
    add_box(
        ax,
        5.50,
        6.55,
        1.85,
        0.58,
        "Relational\nFeatures",
        "#E6F2FF",
        fontsize=8,
    )

    # Feature vector
    add_box(
        ax,
        2.35,
        5.55,
        3.55,
        0.62,
        "Feature Vector xᵢ\nbehavioural + velocity + relational",
        "#EAFBE7",
        fontsize=8,
    )

    add_arrow(ax, 1.78, 6.55, 3.10, 6.17, rad=0.08)
    add_arrow(ax, 4.10, 6.55, 4.10, 6.17)
    add_arrow(ax, 6.42, 6.55, 5.10, 6.17, rad=-0.08)

    # Recency weighting
    add_box(
        ax,
        2.35,
        4.60,
        3.55,
        0.62,
        "Recency Weighting\nwᵢ = 1 old,   wᵢ = α recent",
        "#FFF7CC",
        fontsize=8,
    )
    add_arrow(ax, 4.12, 5.55, 4.12, 5.22)

    # Two model branches
    add_box(
        ax,
        1.00,
        3.45,
        2.45,
        0.72,
        "LightGBM\nGradient Boosting Trees",
        "#EDEBFF",
        fontsize=8,
    )
    add_box(
        ax,
        4.80,
        3.45,
        2.45,
        0.72,
        "XGBoost\nGradient Boosting Trees",
        "#FFE8F2",
        fontsize=8,
    )

    add_arrow(ax, 4.12, 4.60, 2.25, 4.17, rad=0.12)
    add_arrow(ax, 4.12, 4.60, 6.02, 4.17, rad=-0.12)

    # Tree mini blocks for LightGBM
    tree_y = 2.62
    x_start = 0.95
    for i, label in enumerate(["T1", "T2", "T3", "TN"]):
        add_box(
            ax,
            x_start + i * 0.62,
            tree_y,
            0.45,
            0.40,
            label,
            "#F5F3FF",
            fontsize=7,
            lw=1,
        )

    # Tree mini blocks for XGBoost
    x_start = 4.75
    for i, label in enumerate(["T1", "T2", "T3", "TN"]):
        add_box(
            ax,
            x_start + i * 0.62,
            tree_y,
            0.45,
            0.40,
            label,
            "#FFF1F6",
            fontsize=7,
            lw=1,
        )

    add_arrow(ax, 2.22, 3.45, 2.22, 3.02)
    add_arrow(ax, 6.02, 3.45, 6.02, 3.02)

    # Probability outputs
    add_box(
        ax,
        1.00,
        1.90,
        2.45,
        0.55,
        "P_LGB\nfraud probability",
        "#EDEBFF",
        fontsize=8,
    )
    add_box(
        ax,
        4.80,
        1.90,
        2.45,
        0.55,
        "P_XGB\nfraud probability",
        "#FFE8F2",
        fontsize=8,
    )

    add_arrow(ax, 2.22, 2.62, 2.22, 2.45)
    add_arrow(ax, 6.02, 2.62, 6.02, 2.45)

    # Weighted blend
    add_box(
        ax,
        2.55,
        1.05,
        3.20,
        0.58,
        "Weighted Blending\nP_final = 0.53·P_XGB + 0.47·P_LGB",
        "#FFEBCD",
        fontsize=7.5,
    )

    add_arrow(ax, 2.22, 1.90, 3.35, 1.63, rad=0.06)
    add_arrow(ax, 6.02, 1.90, 4.95, 1.63, rad=-0.06)

    # Final output
    add_box(
        ax,
        6.10,
        1.05,
        1.25,
        0.58,
        "Fraud\nScore\n0–1",
        "#DDF3FF",
        fontsize=7.5,
    )
    add_arrow(ax, 5.75, 1.34, 6.10, 1.34)

    add_box(
        ax,
        6.10,
        0.42,
        1.25,
        0.40,
        "Fraud /\nNon-Fraud",
        "#FFE1E1",
        fontsize=7,
    )
    add_arrow(ax, 6.72, 1.05, 6.72, 0.82)

    ax.text(
        4.15,
        0.18,
        "Hybrid XGBoost + LightGBM architecture",
        fontsize=12.5,
        ha="center",
        va="center",
    )

    # ================= RIGHT: TABLE =================
    ax.text(
        8.25,
        7.92,
        "Table: Model architecture",
        fontsize=17,
        fontstyle="italic",
        ha="left",
        va="center",
    )

    col_labels = ["Component", "Input / Output", "Function"]
    table_data = [
        ["Feature vector", "xᵢ", "Combines behavioural,\nvelocity and relational features"],
        ["Recency weighting", "xᵢ, wᵢ", "Gives higher importance\nto recent transactions"],
        ["LightGBM", "xᵢ, wᵢ → P_LGB", "Fast gradient boosting\nfor large tabular data"],
        ["XGBoost", "xᵢ, wᵢ → P_XGB", "Robust tree-based\nclassification model"],
        ["Weighted blend", "P_LGB, P_XGB → P_final", "Combines two fraud\nprobability scores"],
        ["Threshold", "P_final → class", "Final Fraud / Non-Fraud\ndecision"],
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="left",
        colLoc="center",
        bbox=[8.25, 1.05, 7.25, 6.55],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)

        if row == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(fontweight="bold", ha="center", va="center")
        else:
            cell.set_facecolor("white")
            cell.set_text_props(ha="left", va="center")

    # Save output
    output_path = FIG_DIR / "hybrid_xgb_lgbm_model_architecture_clean.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()