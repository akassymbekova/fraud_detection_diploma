import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#f5f5f5", ec="black", fs=8, lw=1.2):
    rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        linespacing=1.15,
    )


def arrow(ax, x1, y1, x2, y2, lw=1.2):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=lw,
        color="black",
    )
    ax.add_patch(arr)


def main():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title
    ax.text(
        0.4,
        8.45,
        "Model Architecture",
        fontsize=28,
        fontweight="bold",
        ha="left",
        va="center",
    )

    # Big left frame
    ax.add_patch(Rectangle((0.4, 0.7), 7.4, 6.9, fill=False, edgecolor="black", linewidth=1.5))

    # Top input features
    box(ax, 0.8, 6.55, 1.8, 0.55, "Behavioural\nFeatures", "#e8f4ff", fs=8)
    box(ax, 3.0, 6.55, 1.8, 0.55, "Velocity\nFeatures", "#e8f4ff", fs=8)
    box(ax, 5.2, 6.55, 1.8, 0.55, "Relational\nFeatures", "#e8f4ff", fs=8)

    box(ax, 2.4, 5.65, 3.0, 0.55, "Feature Vector  xᵢ\n467 model-ready features", "#e9fbe8", fs=8)

    arrow(ax, 1.7, 6.55, 3.0, 6.2)
    arrow(ax, 3.9, 6.55, 3.9, 6.2)
    arrow(ax, 6.1, 6.55, 4.8, 6.2)

    # Recency
    box(ax, 2.4, 4.85, 3.0, 0.55, "Recency Weighting\nwᵢ = 1 old,  wᵢ = α recent", "#fff7cc", fs=8)
    arrow(ax, 3.9, 5.65, 3.9, 5.4)

    # Two models
    box(ax, 1.0, 3.75, 2.5, 0.8, "LightGBM\nGradient Boosting Trees", "#eeeeff", fs=8)
    box(ax, 4.3, 3.75, 2.5, 0.8, "XGBoost\nGradient Boosting Trees", "#ffeef8", fs=8)

    arrow(ax, 3.9, 4.85, 2.25, 4.55)
    arrow(ax, 3.9, 4.85, 5.55, 4.55)

    # Tree blocks under models
    for i, x in enumerate([0.8, 1.35, 1.9, 2.45]):
        box(ax, x, 2.85, 0.45, 0.45, f"T{i+1}", "#f7f7ff", fs=7)

    for i, x in enumerate([4.1, 4.65, 5.2, 5.75]):
        box(ax, x, 2.85, 0.45, 0.45, f"T{i+1}", "#fff5fa", fs=7)

    arrow(ax, 2.25, 3.75, 2.25, 3.3)
    arrow(ax, 5.55, 3.75, 5.55, 3.3)

    box(ax, 1.0, 2.05, 2.5, 0.55, "P_LGB\nfraud probability", "#eeeeff", fs=8)
    box(ax, 4.3, 2.05, 2.5, 0.55, "P_XGB\nfraud probability", "#ffeef8", fs=8)

    arrow(ax, 2.25, 2.85, 2.25, 2.6)
    arrow(ax, 5.55, 2.85, 5.55, 2.6)

    # Blend
    box(
        ax,
        2.45,
        1.15,
        2.9,
        0.55,
        "Weighted Blending\nP_final = 0.53·P_XGB + 0.47·P_LGB",
        "#ffe8cc",
        fs=7.5,
    )
    arrow(ax, 2.25, 2.05, 3.3, 1.7)
    arrow(ax, 5.55, 2.05, 4.5, 1.7)

    box(ax, 5.9, 1.15, 1.4, 0.55, "Fraud Score\n0–1", "#ddf4ff", fs=8)
    arrow(ax, 5.35, 1.42, 5.9, 1.42)

    box(ax, 5.9, 0.75, 1.4, 0.3, "Fraud / Non-Fraud", "#ffe2e2", fs=7)
    arrow(ax, 6.6, 1.15, 6.6, 1.05)

    # Label under architecture
    ax.text(
        4.1,
        0.35,
        "Hybrid XGBoost + LightGBM architecture",
        fontsize=13,
        ha="center",
        va="center",
    )

    # Right table title
    ax.text(
        8.2,
        7.95,
        "Table: Model architecture",
        fontsize=16,
        fontstyle="italic",
        ha="left",
        va="center",
    )

    # Table data
    headers = ["Component", "Input / Output", "Role"]
    rows = [
        ["Input features", "xᵢ", "Behavioural, velocity,\nrelational features"],
        ["Recency weighting", "xᵢ, wᵢ", "Gives higher weight\nto recent transactions"],
        ["LightGBM", "xᵢ, wᵢ → P_LGB", "Fast gradient boosting\nfor tabular data"],
        ["XGBoost", "xᵢ, wᵢ → P_XGB", "Robust gradient boosting\nclassifier"],
        ["Weighted blend", "P_XGB, P_LGB → P_final", "Combines model\nprobabilities"],
        ["Threshold", "P_final → class", "Fraud / Non-Fraud\ndecision"],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        colLoc="center",
        bbox=[8.2, 1.1, 7.3, 6.4],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold", ha="center")
        else:
            cell.set_facecolor("white")

    # Save
    output_path = FIG_DIR / "hybrid_xgb_lgbm_model_architecture_like_example.png"
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()