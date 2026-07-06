import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def expected_calibration_error(y_true, y_proba, n_bins=10):
    """
    Expected Calibration Error for binary classification.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_proba, bins) - 1

    ece = 0.0

    for i in range(n_bins):
        mask = bin_ids == i

        if not np.any(mask):
            continue

        bin_confidence = np.mean(y_proba[mask])
        bin_accuracy = np.mean(y_true[mask])
        bin_weight = np.mean(mask)

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def fit_platt_scaler(y_valid, proba_valid):
    """
    Fit Platt scaling using validation predictions.
    """
    scaler = LogisticRegression(max_iter=1000)
    scaler.fit(proba_valid.reshape(-1, 1), y_valid)
    return scaler


def apply_platt_scaler(scaler, proba):
    return scaler.predict_proba(proba.reshape(-1, 1))[:, 1]


def fit_isotonic_scaler(y_valid, proba_valid):
    """
    Fit isotonic calibration using validation predictions.
    """
    scaler = IsotonicRegression(out_of_bounds="clip")
    scaler.fit(proba_valid, y_valid)
    return scaler


def apply_isotonic_scaler(scaler, proba):
    return scaler.predict(proba)


def calibration_report(y_true, y_proba, model_name):
    return {
        "model": model_name,
        "brier_score": brier_score_loss(y_true, y_proba),
        "ece_10_bins": expected_calibration_error(y_true, y_proba, n_bins=10),
    }


def get_calibration_curve_dataframe(y_true, y_proba, model_name, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_proba,
        n_bins=n_bins,
        strategy="quantile",
    )

    return pd.DataFrame({
        "model": model_name,
        "mean_predicted_probability": prob_pred,
        "fraction_of_positives": prob_true,
    })