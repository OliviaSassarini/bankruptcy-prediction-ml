"""
data_prep.py - LOAD, CLEAN AND SPLIT THE DATA
=============================================
What this file does:
    1. load_raw():  reads dataset/data.csv (the Kaggle file) and strips the
       stray spaces from the column names (e.g. " Current Ratio").
    2. clean():     removes duplicate companies, drops columns that never
       change (e.g. "Net Income Flag" is always 1, so it tells us nothing),
       and fills any missing values with the column median.
    3. split():     splits the data 80% training / 20% testing. The split is
       "stratified", so both parts keep the same ~3% bankruptcy rate.
    4. load_and_prepare(): runs steps 1-2, saves dataset/clean_data.csv
       and prints the dataset size and bankruptcy rate.

"""
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts import config


def load_raw(path=config.RAW_DATA) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it from https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction "
            "and save the CSV as dataset/data.csv"
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: duplicates, constant columns, missing values."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        print(f"Dropping constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    missing = int(df.isna().sum().sum())
    if missing:
        print(f"Filling {missing} missing values with column medians")
        df = df.fillna(df.median(numeric_only=True))
    return df


def split(df: pd.DataFrame):
    """Stratified split so the (rare) bankrupt class appears in both sets."""
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )


def load_and_prepare():
    df = clean(load_raw())
    df.to_csv(config.CLEAN_DATA, index=False)
    print(f"Clean dataset: {df.shape[0]:,} companies x {df.shape[1] - 1} ratios")
    print(f"Bankruptcy rate: {df[config.TARGET].mean():.2%}")
    return df
