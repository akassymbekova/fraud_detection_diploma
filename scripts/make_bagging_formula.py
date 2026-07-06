import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

formula = r"$P_{\mathrm{final}} = \frac{1}{6}\,(P_1 + P_2 + P_3 + P_4 + P_5 + P_6)$"

fig, ax = plt.subplots(figsize=(9, 1.6))
ax.axis("off")

ax.text(
    0.5,
    0.5,
    formula,
    fontsize=30,
    ha="center",
    va="center",
    color="black",
)

output_path = FIG_DIR / "bagging_ensemble_formula.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
plt.close()

print(f"Saved: {output_path}")