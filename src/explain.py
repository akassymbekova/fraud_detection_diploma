import shap
import pandas as pd
import matplotlib.pyplot as plt


def save_lightgbm_feature_importance(model, X_train, output_path, top_n=30):
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(output_path, index=False)

    return importance_df.head(top_n)


def save_shap_summary_plot(model, X_sample, output_path):
    """
    Save SHAP summary plot for tree-based LightGBM model.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_shap_bar_plot(model, X_sample, output_path):
    """
    Save SHAP bar plot.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_local_shap_waterfall(model, X_sample, row_index, output_path):
    """
    Save local SHAP waterfall plot for one transaction.
    """
    explainer = shap.TreeExplainer(model)

    row = X_sample.iloc[[row_index]]
    shap_values = explainer(row)

    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()