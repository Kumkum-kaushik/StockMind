"""
Synthetic data generator for the AI-Powered Explainable Inventory Forecasting System.

Produces a realistic multi-SKU time-series dataset with:
  - Trend, seasonality, and noise baked into demand
  - Promotions and stockout events
  - Supplier lead times and reorder points
  - Product metadata (category, unit cost, shelf life)
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

# ── Configuration ────────────────────────────────────────────────────────────

START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"

PRODUCTS = {
    "SKU-001": {"name": "Wireless Headphones",  "category": "Electronics",  "unit_cost": 45.00, "shelf_life_days": None, "base_demand": 30},
    "SKU-002": {"name": "USB-C Hub",             "category": "Electronics",  "unit_cost": 18.50, "shelf_life_days": None, "base_demand": 50},
    "SKU-003": {"name": "Ergonomic Mouse",        "category": "Electronics",  "unit_cost": 22.00, "shelf_life_days": None, "base_demand": 40},
    "SKU-004": {"name": "Vitamin C Supplements",  "category": "Health",       "unit_cost":  8.00, "shelf_life_days": 365,  "base_demand": 80},
    "SKU-005": {"name": "Protein Powder 1kg",     "category": "Health",       "unit_cost": 25.00, "shelf_life_days": 540,  "base_demand": 60},
    "SKU-006": {"name": "Yoga Mat",               "category": "Sports",       "unit_cost": 15.00, "shelf_life_days": None, "base_demand": 25},
    "SKU-007": {"name": "Water Bottle 1L",        "category": "Sports",       "unit_cost":  9.00, "shelf_life_days": None, "base_demand": 70},
    "SKU-008": {"name": "Notebook A5",            "category": "Stationery",   "unit_cost":  3.50, "shelf_life_days": None, "base_demand": 90},
    "SKU-009": {"name": "Ballpoint Pen Pack",     "category": "Stationery",   "unit_cost":  2.00, "shelf_life_days": None, "base_demand": 120},
    "SKU-010": {"name": "Instant Coffee 200g",    "category": "Groceries",    "unit_cost": 12.00, "shelf_life_days": 730,  "base_demand": 55},
}

WAREHOUSES = ["WH-North", "WH-South", "WH-East"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def seasonal_factor(dates: pd.DatetimeIndex, category: str) -> np.ndarray:
    """Return a multiplicative seasonal factor per day."""
    month = dates.month
    dow   = dates.dayofweek  # 0=Mon … 6=Sun

    # Base sinusoidal annual cycle (peaks in Dec)
    annual = 1 + 0.25 * np.sin(2 * np.pi * (month - 3) / 12)

    # Category tweaks
    if category == "Electronics":
        # Strong Q4 (Nov/Dec) spike
        annual = np.where(month >= 11, annual * 1.4, annual)
    elif category == "Health":
        # Jan resolution bump
        annual = np.where(month == 1, annual * 1.3, annual)
    elif category == "Sports":
        # Summer peak
        annual = np.where((month >= 6) & (month <= 8), annual * 1.3, annual)

    # Weekend uplift for groceries/stationery
    weekend_boost = np.where(dow >= 5, 1.15, 1.0)

    return annual * weekend_boost


def simulate_demand(dates, base_demand, category, trend_pct=0.15):
    n = len(dates)
    trend   = 1 + trend_pct * np.arange(n) / n
    season  = seasonal_factor(dates, category)
    noise   = rng.normal(loc=1.0, scale=0.12, size=n)
    raw     = base_demand * trend * season * noise
    return np.clip(np.round(raw).astype(int), 0, None)


def add_promotions(df: pd.DataFrame) -> pd.DataFrame:
    """Randomly inject ~6 promotional periods per SKU per year; demand ×1.5."""
    promo_flags = np.zeros(len(df), dtype=int)
    for sku in df["sku_id"].unique():
        mask = df["sku_id"] == sku
        idx  = df.index[mask]
        n_promos = int(len(idx) / 60)   # roughly every 2 months
        for _ in range(n_promos):
            start = rng.integers(0, len(idx) - 7)
            duration = rng.integers(3, 8)
            promo_flags[idx[start: start + duration]] = 1
    df["promotion"] = promo_flags
    df["demand_units"] = np.where(
        df["promotion"] == 1,
        np.round(df["demand_units"] * rng.uniform(1.3, 1.7, len(df))).astype(int),
        df["demand_units"],
    )
    return df


def simulate_inventory(demand: np.ndarray, reorder_point: int, reorder_qty: int, lead_time: int):
    """Simple (s, Q) inventory policy simulation."""
    n          = len(demand)
    stock      = np.zeros(n, dtype=int)
    stockout   = np.zeros(n, dtype=int)
    on_order   = 0
    arrival_in = 0

    stock[0] = reorder_qty  # start with one full order

    for t in range(1, n):
        # Receive order
        if arrival_in == 1:
            stock[t] = stock[t - 1] + on_order
            on_order  = 0
            arrival_in = 0
        else:
            stock[t] = stock[t - 1]
            if arrival_in > 0:
                arrival_in -= 1

        # Fulfill demand
        fulfilled  = min(stock[t], demand[t])
        stockout[t] = demand[t] - fulfilled
        stock[t]   -= fulfilled

        # Reorder trigger
        if stock[t] <= reorder_point and on_order == 0:
            on_order   = reorder_qty
            arrival_in = lead_time

    return stock, stockout


# ── Main generator ───────────────────────────────────────────────────────────

def generate(output_path: Path) -> pd.DataFrame:
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    rows  = []

    for sku, meta in PRODUCTS.items():
        for wh in WAREHOUSES:
            # Warehouse-level demand scalar
            wh_scale = {"WH-North": 1.0, "WH-South": 0.75, "WH-East": 0.55}[wh]
            base      = int(meta["base_demand"] * wh_scale)

            demand = simulate_demand(dates, base, meta["category"])

            lead_time    = int(rng.integers(3, 14))
            reorder_pt   = int(base * lead_time * 1.5)
            reorder_qty  = int(base * 30)

            stock, stockout = simulate_inventory(demand, reorder_pt, reorder_qty, lead_time)

            for i, d in enumerate(dates):
                rows.append({
                    "date":              d,
                    "sku_id":            sku,
                    "product_name":      meta["name"],
                    "category":          meta["category"],
                    "warehouse":         wh,
                    "unit_cost":         meta["unit_cost"],
                    "shelf_life_days":   meta["shelf_life_days"],
                    "lead_time_days":    lead_time,
                    "reorder_point":     reorder_pt,
                    "reorder_qty":       reorder_qty,
                    "demand_units":      int(demand[i]),
                    "inventory_level":   int(stock[i]),
                    "stockout_units":    int(stockout[i]),
                    "promotion":         0,  # filled by add_promotions
                    "day_of_week":       d.dayofweek,
                    "month":             d.month,
                    "quarter":           d.quarter,
                    "is_weekend":        int(d.dayofweek >= 5),
                    "year":              d.year,
                })

    df = pd.DataFrame(rows)
    df = add_promotions(df)

    # Derived columns
    df["days_of_stock"] = np.where(
        df["demand_units"] > 0,
        (df["inventory_level"] / df["demand_units"]).round(1),
        np.nan,
    )
    df["stockout_flag"] = (df["stockout_units"] > 0).astype(int)

    df = df.sort_values(["sku_id", "warehouse", "date"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    out = Path("data/raw/inventory_data.csv")
    print("Generating synthetic inventory dataset …")
    df = generate(out)

    print(f"\nSaved to: {out}")
    print(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Date range : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"SKUs       : {df['sku_id'].nunique()}  |  Warehouses: {df['warehouse'].nunique()}")
    print(f"Total demand  : {df['demand_units'].sum():,} units")
    print(f"Total stockout: {df['stockout_units'].sum():,} units  "
          f"({df['stockout_flag'].mean():.1%} of days)")
    print(f"\nColumn list:\n{list(df.columns)}")
    print(f"\nSample (5 rows):\n{df.head().to_string()}")
    print(f"\nNumeric summary:\n{df.describe().to_string()}")
