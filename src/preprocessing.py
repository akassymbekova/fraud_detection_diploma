import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def identify_columns_to_drop(train_df: pd.DataFrame, missing_threshold: float = 0.80):
    """
    Identify columns with too many missing values using train data only.
    """
    missing_ratio = train_df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    return cols_to_drop


def split_features_target(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def basic_time_features(df: pd.DataFrame, time_col: str = "TransactionDT"):
    """
    Create basic time features from TransactionDT.
    IEEE-CIS TransactionDT is a relative timestamp in seconds.
    """
    df = df.copy()

    seconds_in_day = 24 * 60 * 60

    df["hour"] = (df[time_col] // 3600) % 24
    df["day"] = df[time_col] // seconds_in_day
    df["weekday"] = df["day"] % 7

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def prepare_basic_features(train_df, valid_df, test_df, target_col="isFraud"):
    """
    Basic leakage-aware preprocessing:
    - drop high-missing columns based on train only
    - create simple time features
    - separate X/y
    - encode categorical columns with OrdinalEncoder fitted on train only
    - fill missing values
    """
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()

    # Time features
    if "TransactionDT" in train_df.columns:
        train_df = basic_time_features(train_df, "TransactionDT")
        valid_df = basic_time_features(valid_df, "TransactionDT")
        test_df = basic_time_features(test_df, "TransactionDT")

    # Drop high-missing columns based only on train
    cols_to_drop = identify_columns_to_drop(train_df.drop(columns=[target_col]), missing_threshold=0.80)

    train_df = train_df.drop(columns=cols_to_drop, errors="ignore")
    valid_df = valid_df.drop(columns=cols_to_drop, errors="ignore")
    test_df = test_df.drop(columns=cols_to_drop, errors="ignore")

    X_train, y_train = split_features_target(train_df, target_col)
    X_valid, y_valid = split_features_target(valid_df, target_col)
    X_test, y_test = split_features_target(test_df, target_col)

    # Drop ID column from features but keep it separately later if needed
    for col in ["TransactionID"]:
        if col in X_train.columns:
            X_train = X_train.drop(columns=[col])
            X_valid = X_valid.drop(columns=[col])
            X_test = X_test.drop(columns=[col])

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Fill numeric missing values using train medians
    medians = X_train[num_cols].median(numeric_only=True)
    X_train[num_cols] = X_train[num_cols].fillna(medians)
    X_valid[num_cols] = X_valid[num_cols].fillna(medians)
    X_test[num_cols] = X_test[num_cols].fillna(medians)

    # Fill categorical missing values
    X_train[cat_cols] = X_train[cat_cols].fillna("missing")
    X_valid[cat_cols] = X_valid[cat_cols].fillna("missing")
    X_test[cat_cols] = X_test[cat_cols].fillna("missing")

    # Ordinal encoding fitted on train only
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
        X_valid[cat_cols] = encoder.transform(X_valid[cat_cols])
        X_test[cat_cols] = encoder.transform(X_test[cat_cols])
    else:
        encoder = None

    return X_train, y_train, X_valid, y_valid, X_test, y_test, {
        "cols_to_drop": cols_to_drop,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "encoder": encoder,
        "medians": medians,
    }