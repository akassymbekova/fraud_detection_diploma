import pandas as pd

from src.metrics import evaluate_binary_classifier, results_to_dataframe
from src.threshold_opt import find_best_threshold


def evaluate_predictions_file(predictions_path, threshold_summary_path=None):
    """
    Evaluate saved prediction probabilities on test set.
    """
    predictions = pd.read_csv(predictions_path)

    y_true = predictions["y_true"].astype(int)

    results = {}

    for model_name in predictions.columns:
        if model_name == "y_true":
            continue

        y_proba = predictions[model_name].values
        results[model_name] = evaluate_binary_classifier(
            y_true,
            y_proba,
            threshold=0.5,
        )

    return results_to_dataframe(results)