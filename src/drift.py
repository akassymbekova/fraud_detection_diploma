import numpy as np
import pandas as pd

from src.metrics import evaluate_binary_classifier


def calculate_psi(expected, actual, n_bins=10):
    """
    Population Stability Index.
    expected: baseline distribution, e.g. validation predictions
    actual: comparison distribution, e.g. test slice predictions
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    breakpoints = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) <= 2:
        return np.nan

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def create_time_slices(df, time_col, n_slices=4):
    """
    Assign each row to a chronological slice.
    """
    df = df.copy()
    df = df.sort_values(time_col).reset_index(drop=True)

    df["time_slice"] = pd.qcut(
        df.index,
        q=n_slices,
        labels=[f"slice_{i+1}" for i in range(n_slices)]
    )

    return df


def evaluate_temporal_slices(
    test_df,
    predictions_test,
    y_test,
    time_col="TransactionDT",
    model_names=None,
    threshold=0.5,
    n_slices=4,
):
    """
    Evaluate model performance across chronological test slices.
    """
    test_with_slices = create_time_slices(test_df, time_col=time_col, n_slices=n_slices)

    results = []

    if model_names is None:
        model_names = [c for c in predictions_test.columns if c != "y_true"]

    for slice_name in test_with_slices["time_slice"].unique():
        slice_idx = test_with_slices.index[test_with_slices["time_slice"] == slice_name].values

        for model_name in model_names:
            if model_name not in predictions_test.columns:
                continue

            y_slice = y_test.iloc[slice_idx] if hasattr(y_test, "iloc") else y_test[slice_idx]
            proba_slice = predictions_test[model_name].iloc[slice_idx].values

            metrics = evaluate_binary_classifier(
                y_slice,
                proba_slice,
                threshold=threshold,
            )

            metrics["time_slice"] = slice_name
            metrics["model"] = model_name
            metrics["fraud_rate"] = np.mean(y_slice)
            metrics["mean_prediction"] = np.mean(proba_slice)
            metrics["n_rows"] = len(slice_idx)

            results.append(metrics)

    return pd.DataFrame(results)


def psi_by_test_slice(
    predictions_valid,
    predictions_test,
    test_df,
    time_col="TransactionDT",
    model_names=None,
    n_slices=4,
):
    """
    Calculate PSI between validation predictions and each test time slice.
    """
    test_with_slices = create_time_slices(test_df, time_col=time_col, n_slices=n_slices)

    rows = []

    if model_names is None:
        model_names = [c for c in predictions_test.columns if c != "y_true"]

    for model_name in model_names:
        if model_name not in predictions_valid.columns or model_name not in predictions_test.columns:
            continue

        expected = predictions_valid[model_name].values

        for slice_name in test_with_slices["time_slice"].unique():
            slice_idx = test_with_slices.index[test_with_slices["time_slice"] == slice_name].values
            actual = predictions_test[model_name].iloc[slice_idx].values

            psi = calculate_psi(expected, actual, n_bins=10)

            rows.append({
                "model": model_name,
                "time_slice": slice_name,
                "psi": psi,
            })

    return pd.DataFrame(rows)