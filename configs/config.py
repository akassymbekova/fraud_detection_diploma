from pathlib import Path

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

IEEE_TRANSACTION_PATH = DATA_RAW_DIR / "train_transaction.csv"
IEEE_IDENTITY_PATH = DATA_RAW_DIR / "train_identity.csv"
CREDITCARD_PATH = DATA_RAW_DIR / "creditcard.csv"

TARGET_COL = "isFraud"
TIME_COL_IEEE = "TransactionDT"
TIME_COL_CC = "Time"

TRAIN_RATIO = 0.60
VALID_RATIO = 0.20
TEST_RATIO = 0.20