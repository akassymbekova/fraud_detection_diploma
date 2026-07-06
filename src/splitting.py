import pandas as pd


def time_train_valid_test_split(
    df: pd.DataFrame,
    time_col: str,
    train_ratio: float = 0.60,
    valid_ratio: float = 0.20,
):
    """
    Chronological train/validation/test split.

    Fraud detection is time-dependent, so random split may overestimate performance.
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train = df_sorted.iloc[:train_end].copy()
    valid = df_sorted.iloc[train_end:valid_end].copy()
    test = df_sorted.iloc[valid_end:].copy()

    return train, valid, test