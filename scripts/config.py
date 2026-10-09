"""
config.py - PROJECT SETTINGS
============================
What this file does:
    Keeps every setting for the project in one place, so the other files
    never hard-code anything. If you want to change something, change it here.

    1. Paths: where the Kaggle CSV lives, and where charts, the saved model
       and the Excel file are written.
    2. Modelling settings: the target column ("Bankrupt?"), the random seed
       (so results are repeatable) and the test-set size (20%).
    3. Business costs: missing a bankruptcy costs 10, a false alarm costs 1.
       These decide the probability cut-off used to flag risky companies.
    4. Key ratios: 9 accounting ratios used in the charts and the
       simple scorecard model.

"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data
RAW_DATA = ROOT / "dataset" / "data.csv"            # Kaggle file goes here
CLEAN_DATA = ROOT / "dataset" / "clean_data.csv"     # written by data_prep.py

# Outputs
FIGURES_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"
MODELS_DIR = ROOT / "models"
EXCEL_OUTPUT = REPORT_DIR / "bankruptcy_risk_results.xlsx"

TARGET = "Bankrupt?"
RANDOM_STATE = 42
TEST_SIZE = 0.2


COST_MISSED_BANKRUPTCY = 10
COST_FALSE_ALARM = 1


KEY_RATIOS = [
    "Debt ratio %",
    "Current Ratio",
    "Quick Ratio",
    "Net Income to Total Assets",
    "Working Capital to Total Assets",
    "Retained Earnings to Total Assets",
    "Cash Flow to Total Assets",
    "Total Asset Turnover",
    "Equity to Liability",
]
