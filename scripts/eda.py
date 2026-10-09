"""
eda.py - EXPLORATORY DATA ANALYSIS (CHARTS)
===========================================
What this file does:

    1. class_balance():      bar chart of healthy vs bankrupt companies,
                             showing how rare bankruptcy is (~3%).
    2. key_ratio_boxplots(): compares the 9 key accounting ratios for healthy
                             vs bankrupt firms (e.g. bankrupt firms carry
                             more debt and earn less on their assets).
    3. top_correlations():   the 15 ratios most linked to bankruptcy.
    4. summary_table():      median of each key ratio by outcome - the
                             "accountant's view", also exported to Excel.

Run by: main.py.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts import config

HEALTHY, BANKRUPT = "#4C78A8", "#E45756"


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def class_balance(df):
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Healthy", "Bankrupt"], counts.values, color=[HEALTHY, BANKRUPT])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Class balance: bankruptcy is rare")
    ax.set_ylabel("Companies")
    ax.margins(y=0.2)
    _save(fig, "01_class_balance.png")


def key_ratio_boxplots(df):
    ratios = [r for r in config.KEY_RATIOS if r in df.columns]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ax, ratio in zip(axes.flat, ratios):
        # Clip extreme outliers so the boxes are readable
        lo, hi = df[ratio].quantile([0.01, 0.99])
        data = [df.loc[df[config.TARGET] == k, ratio].clip(lo, hi) for k in (0, 1)]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        ax.set_xticks([1, 2], ["Healthy", "Bankrupt"])
        for patch, color in zip(bp["boxes"], [HEALTHY, BANKRUPT]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(ratio, fontsize=10)
    for ax in list(axes.flat)[len(ratios):]:
        ax.axis("off")
    fig.suptitle("Key accounting ratios: healthy vs bankrupt companies", fontsize=14)
    _save(fig, "02_key_ratios_boxplots.png")


def top_correlations(df, n=15):
    corr = df.corr(numeric_only=True)[config.TARGET].drop(config.TARGET)
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color=[BANKRUPT if v > 0 else HEALTHY for v in top.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"Top {n} ratios correlated with bankruptcy")
    ax.set_xlabel("Correlation with Bankrupt? (red = higher ratio, higher risk)")
    _save(fig, "03_top_correlations.png")
    return top


def summary_table(df) -> pd.DataFrame:
    """Median of each key ratio by outcome - the 'accountant's view'."""
    ratios = [r for r in config.KEY_RATIOS if r in df.columns]
    table = df.groupby(config.TARGET)[ratios].median().T
    table.columns = ["Healthy (median)", "Bankrupt (median)"]
    return table


def run_eda(df):
    class_balance(df)
    key_ratio_boxplots(df)
    top_correlations(df)
    table = summary_table(df)
    print("\nMedian key ratios by outcome:\n", table.round(3))
    return table
