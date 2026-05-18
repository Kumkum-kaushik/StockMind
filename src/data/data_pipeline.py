"""
Feature engineering pipeline for the Inventory Forecasting System.

Reads  : data/raw/inventory_data.csv
Writes : data/processed/features.csv

Features built:
  - Lag features        : demand from 1, 7, 14, 28 days ago
  - Rolling statistics  : mean, std, min, max over 7 / 14 / 28-day windows
  - Calendar effects    : day-of-month, week-of-year, month-start/end flags, quarter-start/end
  - Inventory features  : stock coverage ratio, lagged stockout flag
  - Demand momentum     : short vs long rolling mean ratio (trend signal)

IMPORTANT: all lag/rolling ops are computed per (sku_id, warehouse) group
           so values never bleed across products or locations.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH       = Path("data/raw/inventory_data.csv")
PROCESSED_PATH = Path("data/processed/features.csv")


# ── Helpers ──────────────────────────────────────────────────────────────────

def lag_features(g: pd.DataFrame) -> pd.DataFrame:
    """Add demand lag columns for 1, 7, 14, 28 days."""
    for lag in [1, 7, 14, 28]:
        g[f"demand_lag_{lag}d"] = g["demand_units"].shift(lag)
    return g


def rolling_features(g: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean, std, min, max over 7 / 14 / 28-day windows."""
    demand = g["demand_units"]
    for window in [7, 14, 28]:
        roll = demand.shift(1).rolling(window)   # shift(1) avoids using today's value
        g[f"demand_roll_mean_{window}d"] = roll.mean().round(2)
        g[f"demand_roll_std_{window}d"]  = roll.std().round(2)
        g[f"demand_roll_min_{window}d"]  = roll.min()
        g[f"demand_roll_max_{window}d"]  = roll.max()

    # Rolling stockout rate (fraction of days with stockout in last 28 days)
    g["stockout_rate_28d"] = (
        g["stockout_flag"].shift(1).rolling(28).mean().round(3)
    )
    return g


def demand_momentum(g: pd.DataFrame) -> pd.DataFrame:
    """Ratio of 7-day rolling mean to 28-day rolling mean — captures trend direction."""
    short = g["demand_roll_mean_7d"]
    long_ = g["demand_roll_mean_28d"]
    g["demand_momentum"] = (short / long_.replace(0, np.nan)).round(3)
    return g


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extended calendar signals beyond what the raw data already has."""
    dt = df["date"]
    df["day_of_month"]       = dt.dt.day
    df["week_of_year"]       = dt.dt.isocalendar().week.astype(int)
    df["is_month_start"]     = dt.dt.is_month_start.astype(int)
    df["is_month_end"]       = dt.dt.is_month_end.astype(int)
    df["is_quarter_start"]   = dt.dt.is_quarter_start.astype(int)
    df["is_quarter_end"]     = dt.dt.is_quarter_end.astype(int)

    # Proximity to year-end (Black Friday / Christmas season) — peaks at Dec 31
    df["days_to_year_end"]   = (
        pd.to_datetime(dt.dt.year.astype(str) + "-12-31") - dt
    ).dt.days
    return df


def inventory_features(g: pd.DataFrame) -> pd.DataFrame:
    """Stock coverage and lagged inventory signals."""
    # How many days of stock remain relative to 7-day average demand
    avg_7d = g["demand_roll_mean_7d"]
    g["stock_cover_7d"] = (g["inventory_level"] / avg_7d.replace(0, np.nan)).round(2)

    # Lagged inventory level (yesterday's stock)
    g["inventory_lag_1d"] = g["inventory_level"].shift(1)

    # Lagged stockout flag
    g["stockout_lag_1d"]  = g["stockout_flag"].shift(1)
    return g


# ── Pipeline ─────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku_id", "warehouse", "date"]).reset_index(drop=True)

    # Per-group lag + rolling features (never leaks across SKU/warehouse)
    df = (
        df.groupby(["sku_id", "warehouse"], group_keys=False)
          .apply(lag_features)
          .groupby(["sku_id", "warehouse"], group_keys=False)
          .apply(rolling_features)
          .groupby(["sku_id", "warehouse"], group_keys=False)
          .apply(demand_momentum)
          .groupby(["sku_id", "warehouse"], group_keys=False)
          .apply(inventory_features)
    )

    # Calendar features (no grouping needed)
    df = calendar_features(df)

    # Drop rows where lag/rolling windows are incomplete (first 28 days per group)
    df = df.dropna(subset=["demand_lag_28d", "demand_roll_mean_28d"]).reset_index(drop=True)

    return df


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading raw data...")
    raw = pd.read_csv(RAW_PATH, parse_dates=["date"])
    print(f"  Raw shape : {raw.shape[0]:,} rows x {raw.shape[1]} columns")

    print("\nBuilding features...")
    features = build_features(raw)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_PATH, index=False)

    # ── Inspection ───────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("FEATURE DATAFRAME SUMMARY")
    print(f"{'='*60}")
    print(f"Shape          : {features.shape[0]:,} rows x {features.shape[1]} columns")
    print(f"Date range     : {features['date'].min().date()} to {features['date'].max().date()}")
    print(f"Rows dropped   : {raw.shape[0] - features.shape[0]:,}  (first 28 days per group, warm-up period)")

    print(f"\n--- All columns ({features.shape[1]}) ---")
    col_groups = {
        "Identity"       : ["date","sku_id","product_name","category","warehouse"],
        "Product info"   : ["unit_cost","shelf_life_days","lead_time_days","reorder_point","reorder_qty"],
        "Core targets"   : ["demand_units","inventory_level","stockout_units","stockout_flag","days_of_stock"],
        "Lag features"   : [c for c in features.columns if "lag" in c and "roll" not in c],
        "Rolling stats"  : [c for c in features.columns if "roll" in c],
        "Momentum"       : ["demand_momentum","stockout_rate_28d"],
        "Inventory feats": ["stock_cover_7d"],
        "Calendar (raw)" : ["day_of_week","month","quarter","is_weekend","year"],
        "Calendar (new)" : ["day_of_month","week_of_year","is_month_start","is_month_end",
                            "is_quarter_start","is_quarter_end","days_to_year_end"],
        "Promo"          : ["promotion"],
    }
    for group, cols in col_groups.items():
        present = [c for c in cols if c in features.columns]
        print(f"  {group:<18}: {present}")

    print(f"\n--- Sample rows (first 5) ---")
    sample_cols = [
        "date","sku_id","warehouse","demand_units",
        "demand_lag_1d","demand_lag_7d",
        "demand_roll_mean_7d","demand_roll_std_7d",
        "demand_momentum","stock_cover_7d","promotion",
        "is_weekend","days_to_year_end"
    ]
    print(features[sample_cols].head(5).to_string(index=False))

    print(f"\n--- Numeric feature stats ---")
    lag_roll_cols = [c for c in features.columns if "lag" in c or "roll" in c or c in
                     ["demand_momentum","stock_cover_7d","stockout_rate_28d","days_to_year_end"]]
    print(features[lag_roll_cols].describe().round(2).to_string())

    print(f"\n--- Missing values per column ---")
    missing = features.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  None — all features are complete after warm-up drop.")
    else:
        print(missing.to_string())

    print(f"\nSaved to: {PROCESSED_PATH}")
