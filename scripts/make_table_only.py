from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.DataFrame([
        ["Baseline LightGBM", "0.484", "0.881", "0.337", "0.544", "—", "—"],
        ["TabNet", "0.253", "0.796", "0.656", "0.125", "0.149", "0.277"],
        ["CatBoost", "0.508", "0.915", "0.208", "0.759", "0.496", "0.360"],
        ["XGBoost", "0.549", "0.921", "0.311", "0.684", "0.552", "0.434"],
        ["Bagged LGBM Ensemble", "0.604", "0.930", "0.637", "0.554", "0.569", "0.581"],
        ["Best Recency LGBM", "0.606", "0.928", "0.660", "0.537", "0.558", "0.582"],
    ], columns=[
        "Model",
        "PR-AUC",
        "ROC-AUC",
        "Precision",
        "Recall",
        "F2",
        "MCC",
    ])

    # Save CSV too
    df.to_csv(OUTPUT_DIR / "table_model_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    # Header style
    for col in range(len(df.columns)):
        cell = table[(0, col)]
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#EAEAEA")

    # Bold final best row
    best_row_index = len(df)  # because header is row 0
    for col in range(len(df.columns)):
        table[(best_row_index, col)].set_text_props(weight="bold")

    plt.title(
        "Table: Model Performance on Future Holdout Test Set",
        fontsize=16,
        weight="bold",
        pad=18,
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "table_model_results.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Saved:")
    print(OUTPUT_DIR / "table_model_results.png")
    print(OUTPUT_DIR / "table_model_results.csv")


if __name__ == "__main__":
    main()