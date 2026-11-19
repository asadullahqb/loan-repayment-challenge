"""Reusable EDA helpers."""
import pandas as pd


def missingness_report(df):
    """Return a sorted DataFrame of columns that contain missing values."""
    missing = df.isna().sum()
    missing = missing[missing > 0]
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"n_missing": missing, "pct_missing": pct})
    return summary.sort_values("pct_missing", ascending=False)


def segment_default_rate(df, col):
    """Return default rate (size + mean) grouped by a categorical column."""
    g = df.groupby(col, observed=True)["defaulted"].agg(["size", "mean"]).round(3)
    g.columns = ["n", "default_rate"]
    return g.sort_values("default_rate", ascending=False)


def fraud_indicator_sparsity(underwriting):
    """Return the positive (`true`) rate of each fraud-indicator column."""
    indicator_cols = [c for c in underwriting.columns if "indicator" in c]
    pos = underwriting[indicator_cols].apply(
        lambda s: (s.astype(str).str.lower() == "true").mean()
    ).round(4)
    return pos.sort_values(ascending=False)
