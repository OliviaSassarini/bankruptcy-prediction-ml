"""
main.py - THE WHOLE PROJECT
===============================


What it does, in order:
    1. Loads and cleans the Kaggle data          (scripts/data_prep.py)
    2. Makes the exploratory charts              (scripts/eda.py)
    3. Splits into training and test sets        (scripts/data_prep.py)
    4. Trains and compares 3 models              (scripts/train.py)
"""
import pandas as pd

from scripts import config
from scripts.data_prep import load_and_prepare, split
from scripts.eda import run_eda
from scripts.train import run_training


def main():
    df = load_and_prepare()
    ratio_table = run_eda(df)

    X_train, X_test, y_train, y_test = split(df)
    results, predictions, scorecard, intercept, threshold = run_training(
        X_train, X_test, y_train, y_test
    )

    
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(config.EXCEL_OUTPUT) as xl:
        results.to_excel(xl, sheet_name="Model Comparison", index=False)
        ratio_table.to_excel(xl, sheet_name="Ratios by Outcome")
        predictions.sort_values("predicted_probability", ascending=False).to_excel(
            xl, sheet_name="Test Predictions", index=False
        )
        scorecard.assign(intercept=intercept).to_excel(xl, sheet_name="Scorecard Coefs", index=False)
    print(f"\nExcel results saved to {config.EXCEL_OUTPUT}")


if __name__ == "__main__":
    main()
