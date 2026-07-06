import numpy as np
import pandas as pd


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    return df


def add_time_since_previous_transaction(
    df: pd.DataFrame,
    group_col: str = "card1",
    time_col: str = "TransactionDT",
) -> pd.DataFrame:
    df = df.copy()

    if group_col not in df.columns or time_col not in df.columns:
        return df

    df = df.sort_values(time_col)

    new_col = f"time_since_prev_{group_col}"
    df[new_col] = df.groupby(group_col)[time_col].diff()

    median_gap = df[new_col].median()

    if pd.isna(median_gap):
        median_gap = 0

    df[new_col] = df[new_col].fillna(median_gap)

    return df


def fit_frequency_maps(train_df: pd.DataFrame, cols: list):
    """
    Fit frequency encoding maps on train data only.
    """
    freq_maps = {}

    for col in cols:
        if col in train_df.columns:
            freq_maps[col] = train_df[col].value_counts(dropna=False).to_dict()

    return freq_maps


def apply_frequency_encoding(df: pd.DataFrame, freq_maps: dict) -> pd.DataFrame:
    """
    Apply frequency encoding using train-based maps.
    Unknown categories get 0.
    """
    df = df.copy()

    for col, mapping in freq_maps.items():
        if col in df.columns:
            df[f"{col}_freq"] = df[col].map(mapping).fillna(0)

    return df


def fit_aggregation_maps(
    train_df: pd.DataFrame,
    group_cols: list,
    amount_col: str = "TransactionAmt",
):
    """
    Fit aggregation maps on train data only.
    """
    agg_maps = {}

    for col in group_cols:
        if col not in train_df.columns or amount_col not in train_df.columns:
            continue

        grouped = train_df.groupby(col)[amount_col].agg(["count", "mean", "std", "median", "max"])
        grouped = grouped.fillna(0)

        agg_maps[col] = grouped

    return agg_maps


def apply_aggregation_features(
    df: pd.DataFrame,
    agg_maps: dict,
    amount_col: str = "TransactionAmt",
) -> pd.DataFrame:
    """
    Apply train-based aggregation features.
    """
    df = df.copy()

    for col, grouped in agg_maps.items():
        if col not in df.columns:
            continue

        for stat in grouped.columns:
            feature_name = f"{col}_{amount_col}_{stat}"
            df[feature_name] = df[col].map(grouped[stat]).fillna(0)

        mean_feature = f"{col}_{amount_col}_mean"
        ratio_feature = f"{col}_{amount_col}_to_mean_ratio"

        if amount_col in df.columns and mean_feature in df.columns:
            df[ratio_feature] = df[amount_col] / (df[mean_feature] + 1e-6)

    return df


def fit_nunique_maps(train_df: pd.DataFrame, pairs: list):
    """
    Fit number of unique related values on train data only.
    Example: card1 -> number of unique P_emaildomain.
    """
    maps = {}

    for group_col, related_col in pairs:
        if group_col in train_df.columns and related_col in train_df.columns:
            maps[(group_col, related_col)] = (
                train_df.groupby(group_col)[related_col].nunique(dropna=True).to_dict()
            )

    return maps


def apply_nunique_features(df: pd.DataFrame, nunique_maps: dict) -> pd.DataFrame:
    df = df.copy()

    for (group_col, related_col), mapping in nunique_maps.items():
        if group_col in df.columns:
            feature_name = f"{group_col}_unique_{related_col}"
            df[feature_name] = df[group_col].map(mapping).fillna(0)

    return df

def _compute_velocity_for_combined(
    combined_df: pd.DataFrame,
    current_mask_col: str,
    group_col: str = "card1",
    time_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
    windows_hours: list = [1, 6, 24],
) -> pd.DataFrame:
    """
    Compute velocity features for rows marked as current_mask_col == True.

    For each transaction, uses only previous transactions within the same group.
    This avoids future leakage.
    """
    df = combined_df.copy()
    df = df.sort_values([group_col, time_col]).reset_index(drop=True)

    windows_sec = [w * 3600 for w in windows_hours]

    for w in windows_hours:
        df[f"{group_col}_tx_count_last_{w}h"] = 0
        df[f"{group_col}_amt_sum_last_{w}h"] = 0.0

    result_rows = []

    for _, group in df.groupby(group_col, sort=False):
        times = group[time_col].values
        amounts = group[amount_col].values
        current_flags = group[current_mask_col].values
        group_index = group.index.values

        cumsum_amounts = np.concatenate([[0.0], np.cumsum(amounts)])

        for i in range(len(group)):
            if not current_flags[i]:
                continue

            t = times[i]

            row_result = {"_row_id": group.loc[group_index[i], "_row_id"]}

            for w, w_sec in zip(windows_hours, windows_sec):
                left = np.searchsorted(times, t - w_sec, side="left")
                right = i  # exclude current transaction

                count = right - left
                amount_sum = cumsum_amounts[right] - cumsum_amounts[left]

                row_result[f"{group_col}_tx_count_last_{w}h"] = count
                row_result[f"{group_col}_amt_sum_last_{w}h"] = amount_sum

            result_rows.append(row_result)

    result = pd.DataFrame(result_rows)
    return result


def add_velocity_features_searchsorted(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "card1",
    time_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
    windows_hours: list = [1, 6, 24],
):
    """
    Leakage-aware velocity features.

    Train:
        uses previous train transactions only.

    Validation:
        uses train history + previous validation transactions.

    Test:
        uses train + validation history + previous test transactions.

    This simulates real deployment: model can use past events, not future events.
    """
    train_df = train_df.copy().reset_index(drop=True)
    valid_df = valid_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    train_df["_row_id"] = np.arange(len(train_df))
    valid_df["_row_id"] = np.arange(len(valid_df))
    test_df["_row_id"] = np.arange(len(test_df))

    def _apply_velocity(history_df, current_df):
        history = history_df.copy()
        current = current_df.copy()

        history["_is_current"] = False
        current["_is_current"] = True

        combined = pd.concat([history, current], axis=0, ignore_index=True)

        velocity = _compute_velocity_for_combined(
            combined_df=combined,
            current_mask_col="_is_current",
            group_col=group_col,
            time_col=time_col,
            amount_col=amount_col,
            windows_hours=windows_hours,
        )

        current = current.merge(velocity, on="_row_id", how="left")

        velocity_cols = [
            f"{group_col}_tx_count_last_{w}h" for w in windows_hours
        ] + [
            f"{group_col}_amt_sum_last_{w}h" for w in windows_hours
        ]

        for col in velocity_cols:
            current[col] = current[col].fillna(0)

        current = current.drop(columns=["_is_current"], errors="ignore")
        return current

    # Train uses only previous train rows
    empty_history = pd.DataFrame(columns=train_df.columns)
    train_out = _apply_velocity(empty_history, train_df)

    # Valid uses train + previous valid rows
    valid_out = _apply_velocity(train_df, valid_df)

    # Test uses train + valid + previous test rows
    history_test = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
    test_out = _apply_velocity(history_test, test_df)

    train_out = train_out.drop(columns=["_row_id"], errors="ignore")
    valid_out = valid_out.drop(columns=["_row_id"], errors="ignore")
    test_out = test_out.drop(columns=["_row_id"], errors="ignore")

    return train_out, valid_out, test_out

def prepare_feature_engineering(train_df, valid_df, test_df):
    """
    Leakage-aware feature engineering.

    Fit encoding/aggregation maps only on train.
    Apply the same mappings to valid/test.
    """
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()

    # 1. Basic amount features
    train_df = add_amount_features(train_df)
    valid_df = add_amount_features(valid_df)
    test_df = add_amount_features(test_df)

    # 2. Time since previous card transaction
    train_df = add_time_since_previous_transaction(train_df, "card1", "TransactionDT")
    valid_df = add_time_since_previous_transaction(valid_df, "card1", "TransactionDT")
    test_df = add_time_since_previous_transaction(test_df, "card1", "TransactionDT")

    # 3. Frequency encoding
    freq_cols = [
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "addr1",
        "addr2",
    ]

    freq_maps = fit_frequency_maps(train_df, freq_cols)

    train_df = apply_frequency_encoding(train_df, freq_maps)
    valid_df = apply_frequency_encoding(valid_df, freq_maps)
    test_df = apply_frequency_encoding(test_df, freq_maps)

    # 4. Aggregation features
    agg_cols = ["card1", "card2", "card3", "card5", "addr1", "ProductCD"]

    agg_maps = fit_aggregation_maps(train_df, agg_cols, amount_col="TransactionAmt")

    train_df = apply_aggregation_features(train_df, agg_maps, amount_col="TransactionAmt")
    valid_df = apply_aggregation_features(valid_df, agg_maps, amount_col="TransactionAmt")
    test_df = apply_aggregation_features(test_df, agg_maps, amount_col="TransactionAmt")

    # 5. Graph-inspired relational features
    nunique_pairs = [
        ("card1", "P_emaildomain"),
        ("card1", "R_emaildomain"),
        ("card1", "DeviceInfo"),
        ("card1", "addr1"),
        ("DeviceInfo", "card1"),
        ("P_emaildomain", "card1"),
        ("addr1", "card1"),
    ]

    nunique_maps = fit_nunique_maps(train_df, nunique_pairs)

    train_df = apply_nunique_features(train_df, nunique_maps)
    valid_df = apply_nunique_features(valid_df, nunique_maps)
    test_df = apply_nunique_features(test_df, nunique_maps)
    
    # 6. Velocity features
    train_df, valid_df, test_df = add_velocity_features_searchsorted(
        train_df,
        valid_df,
        test_df,
        group_col="card1",
        time_col="TransactionDT",
        amount_col="TransactionAmt",
        windows_hours=[1, 6, 24],
    )

    return train_df, valid_df, test_df, {
        "freq_maps": freq_maps,
        "agg_maps": agg_maps,
        "nunique_maps": nunique_maps,
    }