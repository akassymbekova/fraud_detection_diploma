import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    confusion_matrix,
)


def threshold_metrics(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })

    return pd.DataFrame(rows)


def find_best_threshold(y_true, y_proba, metric="f2"):
    df = threshold_metrics(y_true, y_proba)

    if metric not in df.columns:
        raise ValueError(f"Metric {metric} not found.")

    best_row = df.loc[df[metric].idxmax()]
    return best_row, df


def cost_based_threshold(
    y_true,
    y_proba,
    amounts=None,
    fp_cost=1.0,
    fn_cost=100.0,
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    if amounts is not None:
        amounts = np.array(amounts)

    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        tp_mask = (y_true == 1) & (y_pred == 1)

        fp_count = fp_mask.sum()
        fn_count = fn_mask.sum()
        tp_count = tp_mask.sum()

        fp_total_cost = fp_count * fp_cost

        if amounts is not None:
            fn_total_cost = amounts[fn_mask].sum()
            prevented_fraud_amount = amounts[tp_mask].sum()
        else:
            fn_total_cost = fn_count * fn_cost
            prevented_fraud_amount = tp_count * fn_cost

        total_cost = fp_total_cost + fn_total_cost
        net_benefit = prevented_fraud_amount - fp_total_cost

        rows.append({
            "threshold": threshold,
            "fp_count": fp_count,
            "fn_count": fn_count,
            "tp_count": tp_count,
            "fp_cost": fp_total_cost,
            "fn_cost": fn_total_cost,
            "total_cost": total_cost,
            "prevented_fraud_amount": prevented_fraud_amount,
            "net_benefit": net_benefit,
        })

    result = pd.DataFrame(rows)
    best_by_cost = result.loc[result["total_cost"].idxmin()]
    best_by_benefit = result.loc[result["net_benefit"].idxmax()]

    return best_by_cost, best_by_benefit, result