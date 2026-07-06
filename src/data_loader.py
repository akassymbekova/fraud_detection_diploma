import pandas as pd


def load_ieee(transaction_path, identity_path):
    """
    Load and merge IEEE-CIS transaction and identity data.
    """
    trans = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)

    df = trans.merge(identity, on="TransactionID", how="left")
    return df


def load_creditcard(path):
    """
    Load Credit Card Fraud Detection dataset.
    """
    return pd.read_csv(path)