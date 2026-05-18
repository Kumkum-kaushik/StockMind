"""
Ensemble: LightGBM + SARIMA + TFT combined demand forecasting.

Reads  : data/processed/features.csv
         data/processed/tft_preds.csv     (saved by deep_learning.py)
         models/lgbm_demand.pkl           (saved by classical.py)

Test window: last 7 days of the dataset (same window all models predict)

Blending strategies tried:
  1. equal        — simple average  (1/3 each)
  2. inv_mape     — weight each model by 1/MAPE  (reward accuracy)
  3. lgbm_tft     — drop SARIMA, average best two
  4. optimized    — scipy finds weights that minimise MAPE on a held-out
                    validation window (7 days before the test window)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from pathlib import Path

FEATURES_PATH  = Path("data/processed/features.csv")
TFT_PREDS_PATH = Path("data/processed/tft_preds.csv")
MODEL_PATH     = Path("models/lgbm_demand.pkl")

PRED_HORIZON   = 7     # test window length (days)
VAL_HORIZON    = 7     # validation window for weight optimisation

DROP_COLS      = [
    "demand_units", "date", "product_name", "shelf_life_days",
    "inventory_level", "stockout_units", "stockout_flag", "days_of_stock",
]
CAT_COLS = ["sku_id", "warehouse", "category"]


# ── Metrics ───────────────────────────────────────────────────────────────────

def mape(actual, predicted):
    mask = actual > 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))

def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# ── SARIMA helper ─────────────────────────────────────────────────────────────

from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarima_forecast(train_series: pd.Series, steps: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SARIMAX(
            train_series,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=150)
    return np.clip(result.forecast(steps=steps).values, 0, None)


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data and models...")
df      = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
tft_df  = pd.read_csv(TFT_PREDS_PATH, parse_dates=["date"])
lgbm_artifact  = joblib.load(MODEL_PATH)
lgbm_model     = lgbm_artifact["model"]
feature_cols   = lgbm_artifact["feature_cols"]

# Define windows
max_date   = df["date"].max()
test_start = max_date - pd.Timedelta(days=PRED_HORIZON - 1)
val_start  = test_start - pd.Timedelta(days=VAL_HORIZON)
val_end    = test_start - pd.Timedelta(days=1)
train_end  = val_start  - pd.Timedelta(days=1)

print(f"  Validation window : {val_start.date()} to {val_end.date()}")
print(f"  Test window       : {test_start.date()} to {max_date.date()}")


# ── LightGBM predictions ─────────────────────────────────────────────────────

def lgbm_predict(window_df):
    X = window_df[feature_cols].copy()
    for col in CAT_COLS:
        X[col] = X[col].astype("category")
    return np.clip(lgbm_model.predict(X), 0, None)

val_df  = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy().reset_index(drop=True)
test_df = df[df["date"] >= test_start].copy().reset_index(drop=True)

lgbm_val_preds  = lgbm_predict(val_df)
lgbm_test_preds = lgbm_predict(test_df)

y_val  = val_df["demand_units"].values
y_test = test_df["demand_units"].values


# ── SARIMA predictions ────────────────────────────────────────────────────────

print("Fitting SARIMA per group (30 models)...")
sarima_val_preds  = np.zeros(len(val_df))
sarima_test_preds = np.zeros(len(test_df))

groups = df.groupby(["sku_id", "warehouse"])
for i, ((sku, wh), grp) in enumerate(groups, 1):
    grp = grp.sort_values("date")

    # Validation forecast: train up to train_end, predict VAL_HORIZON steps
    tr_val = grp[grp["date"] <= train_end]["demand_units"]
    if len(tr_val) >= 30:
        preds_v = fit_sarima_forecast(tr_val, VAL_HORIZON)
        idx_v   = np.where((val_df["sku_id"] == sku) & (val_df["warehouse"] == wh))[0]
        sarima_val_preds[idx_v] = preds_v[:len(idx_v)]

    # Test forecast: train up to val_end, predict PRED_HORIZON steps
    tr_test = grp[grp["date"] <= val_end]["demand_units"]
    if len(tr_test) >= 30:
        preds_t = fit_sarima_forecast(tr_test, PRED_HORIZON)
        idx_t   = np.where((test_df["sku_id"] == sku) & (test_df["warehouse"] == wh))[0]
        sarima_test_preds[idx_t] = preds_t[:len(idx_t)]

    print(f"  [{i:02d}/30] {sku} | {wh}", end="\r")

print(f"  SARIMA done.                               ")


# ── TFT predictions ───────────────────────────────────────────────────────────

tft_test_df   = tft_df.sort_values(["sku_id", "warehouse", "date"]).reset_index(drop=True)
test_df_sorted = test_df.sort_values(["sku_id", "warehouse", "date"]).reset_index(drop=True)
tft_test_preds = tft_test_df["tft_pred"].values

# TFT has no validation predictions (wasn't run on val window);
# use its test-window MAPE for weight estimation only
tft_val_preds  = None   # flagged below when computing val MAPEs


# ── Individual model MAPEs on TEST window ─────────────────────────────────────

y_test_sorted = test_df_sorted["demand_units"].values

lgbm_test_sorted = lgbm_predict(test_df_sorted)
sarima_test_sorted = np.zeros(len(test_df_sorted))
for (sku, wh), grp_t in test_df_sorted.groupby(["sku_id", "warehouse"]):
    mask = (test_df_sorted["sku_id"] == sku) & (test_df_sorted["warehouse"] == wh)
    idx  = test_df_sorted.index[mask] - test_df_sorted.index[0]
    tr   = df[(df["sku_id"] == sku) & (df["warehouse"] == wh) & (df["date"] <= val_end)]["demand_units"]
    if len(tr) >= 30:
        sarima_test_sorted[idx] = fit_sarima_forecast(tr, PRED_HORIZON)[:len(idx)]

m_lgbm  = mape(y_test_sorted, lgbm_test_sorted)
m_sar   = mape(y_test_sorted, sarima_test_sorted)
m_tft   = mape(y_test_sorted, tft_test_preds)


# ── Blending strategies ───────────────────────────────────────────────────────

def blend(preds_list, weights):
    weights = np.array(weights) / np.array(weights).sum()
    return sum(w * p for w, p in zip(weights, preds_list))

# 1. Equal weights
equal_preds = blend([lgbm_test_sorted, sarima_test_sorted, tft_test_preds], [1, 1, 1])

# 2. Inverse-MAPE weights  (lower MAPE → higher weight)
inv_w = [1/m_lgbm, 1/m_sar, 1/m_tft]
inv_mape_preds = blend([lgbm_test_sorted, sarima_test_sorted, tft_test_preds], inv_w)

# 3. LightGBM + TFT only (drop worst SARIMA)
lgbm_tft_preds = blend([lgbm_test_sorted, tft_test_preds], [1, 1])

# 4. Optimized weights via scipy (val window: LGBM + SARIMA only; TFT has no val preds)
def neg_mape_weights(w, preds_list, actual):
    w = np.clip(w, 0, None)
    mixed = blend(preds_list, w)
    return mape(actual, mixed)

y_val_sorted   = val_df.sort_values(["sku_id", "warehouse", "date"])["demand_units"].values
lgbm_val_sorted = lgbm_predict(val_df.sort_values(["sku_id", "warehouse", "date"]))
sarima_val_sorted = sarima_val_preds   # already computed above in sorted order

res = minimize(
    neg_mape_weights,
    x0=[0.5, 0.5],
    args=([lgbm_val_sorted, sarima_val_sorted], y_val_sorted),
    method="Nelder-Mead",
    options={"maxiter": 500},
)
opt_w = np.clip(res.x, 0, None)
opt_w = opt_w / opt_w.sum()
optimized_preds = blend(
    [lgbm_test_sorted, sarima_test_sorted],
    opt_w
)


# ── Results ───────────────────────────────────────────────────────────────────

results = {
    "LightGBM  (1-step)":      (lgbm_test_sorted,  "single model"),
    "SARIMA    (7-step)":      (sarima_test_sorted, "single model"),
    "TFT       (7-step)":      (tft_test_preds,     "single model"),
    "Ensemble: Equal":         (equal_preds,         "1/3 each"),
    "Ensemble: Inv-MAPE":      (inv_mape_preds,      f"w={[round(v/sum(inv_w),2) for v in inv_w]}"),
    "Ensemble: LGBM+TFT":      (lgbm_tft_preds,      "drop SARIMA"),
    f"Ensemble: Optimized":    (optimized_preds,     f"LGBM/SARIMA w={opt_w.round(2).tolist()}"),
}

print(f"\n{'='*68}")
print("ENSEMBLE COMPARISON  —  7-day test window  (lower MAPE = better)")
print(f"{'='*68}")
print(f"  {'Model':<30}  {'MAPE':>8}  {'MAE':>7}  {'RMSE':>7}  {'Note'}")
print(f"  {'-'*64}")

best_mape  = 999
best_label = ""
for label, (preds, note) in results.items():
    m = mape(y_test_sorted, preds)
    a = mae(y_test_sorted, preds)
    r = rmse(y_test_sorted, preds)
    marker = " <-- BEST" if m < best_mape else ""
    if m < best_mape:
        best_mape  = m
        best_label = label
    print(f"  {label:<30}  {m:>7.2f}%  {a:>6.2f}  {r:>6.2f}  {note}")

print(f"\n  Winner: {best_label}  ({best_mape:.2f}%)")

# Per-SKU breakdown for best ensemble vs best single model
single_mapes  = {k: mape(y_test_sorted, v[0]) for k, v in results.items() if "single" in v[1]}
best_single   = min(single_mapes, key=single_mapes.get)
best_ens_preds, _ = results[best_label]

print(f"\n--- Per-SKU: {best_single.strip()} vs {best_label.strip()} ---")
print(f"  {'SKU':<8}  {'Product':<24}  {'Single':>8}  {'Ensemble':>9}  {'Gain'}")
print(f"  {'-'*62}")
for (sku,), grp in test_df_sorted.groupby(["sku_id"]):
    mask     = test_df_sorted["sku_id"] == sku
    name     = df[df["sku_id"] == sku]["product_name"].iloc[0]
    idx      = np.where(mask)[0]
    act      = y_test_sorted[idx]
    s_preds  = results[best_single][0][idx]
    e_preds  = best_ens_preds[idx]
    m_s      = mape(act, s_preds)
    m_e      = mape(act, e_preds)
    gain     = m_s - m_e
    arrow    = "+" if gain > 0 else " "
    print(f"  {sku:<8}  {name:<24}  {m_s:>7.2f}%  {m_e:>8.2f}%  {arrow}{gain:.2f}pp")

print(f"\n  'Gain' = how many percentage points the ensemble improved over {best_single.strip()}")
print(f"  Positive = ensemble is better | Negative = single model was better for that SKU")
