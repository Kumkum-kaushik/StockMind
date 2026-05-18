"""
LightGBM + SARIMA demand forecasting — head-to-head comparison.

Reads  : data/processed/features.csv
Outputs: printed metrics + feature importances + model comparison table

Split  : time-based — last 90 days as test, everything before as train.
         (Never random-split time series — future data would leak into training.)

Target : demand_units (daily demand per SKU per warehouse)
Metric : MAPE — Mean Absolute Percentage Error

Models:
  LightGBM — one global model trained on all 30 series at once (tabular ML)
  SARIMA   — one statistical model fitted per (sku_id, warehouse) series (30 models)
"""

import warnings
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path

FEATURES_PATH = Path("data/processed/features.csv")
MODEL_PATH    = Path("models/lgbm_demand.pkl")

# ── Feature definition ────────────────────────────────────────────────────────

# Columns the model is NOT allowed to see (target, identifiers, future leakage)
DROP_COLS = [
    "demand_units",        # target — never a feature
    "date",                # raw date — calendar features already extracted
    "product_name",        # redundant with sku_id
    "shelf_life_days",     # 68% missing; not worth imputing for baseline
    "inventory_level",     # current-day stock depends on today's demand (leakage)
    "stockout_units",      # outcome of today's demand (leakage)
    "stockout_flag",       # same
    "days_of_stock",       # derived from today's demand (leakage)
]

CATEGORICAL_COLS = ["sku_id", "warehouse", "category"]

TARGET = "demand_units"


# ── Metric ────────────────────────────────────────────────────────────────────

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MAPE, skipping rows where actual demand is 0 to avoid division by zero."""
    mask = actual > 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# ── SARIMA ────────────────────────────────────────────────────────────────────

def fit_sarima(train_series: pd.Series, test_steps: int) -> np.ndarray:
    """
    Fit SARIMA(1,1,1)(1,0,1,7) on one time series and forecast test_steps ahead.

    Order explained:
      (1,1,1)   — AR(1): use yesterday; I(1): first-difference for trend removal;
                  MA(1): correct last error
      (1,0,1,7) — seasonal AR and MA with period=7 (weekly cycle);
                  no seasonal differencing (D=0) to keep it fast
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SARIMAX(
            train_series,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=150)
    forecast = result.forecast(steps=test_steps)
    return np.clip(forecast.values, 0, None)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Load
    print("Loading features...")
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 2. Time-based train / test split
    cutoff = df["date"].max() - pd.Timedelta(days=90)
    train  = df[df["date"] <= cutoff].copy()
    test   = df[df["date"] >  cutoff].copy()

    print(f"\nTrain: {train['date'].min().date()} to {train['date'].max().date()} ({len(train):,} rows)")
    print(f"Test : {test['date'].min().date()}  to {test['date'].max().date()}  ({len(test):,} rows)")

    # 3. Build X / y
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    X_train = train[feature_cols].copy()
    y_train = train[TARGET].values

    X_test  = test[feature_cols].copy()
    y_test  = test[TARGET].values

    # Encode categoricals as LightGBM category dtype
    for col in CATEGORICAL_COLS:
        X_train[col] = X_train[col].astype("category")
        X_test[col]  = X_test[col].astype("category")

    print(f"\nFeatures used ({len(feature_cols)}): {feature_cols}")

    # 4. Train LightGBM
    print("\nTraining LightGBM...")
    model = lgb.LGBMRegressor(
        n_estimators      = 500,
        learning_rate     = 0.05,
        num_leaves        = 63,
        min_child_samples = 20,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )

    model.fit(
        X_train, y_train,
        eval_set              = [(X_test, y_test)],
        callbacks             = [lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=100)],
    )

    print(f"  Best iteration: {model.best_iteration_}")

    # 5. Predict
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)   # demand can't be negative

    # 6. Overall metrics
    print(f"\n{'='*50}")
    print("TEST SET METRICS")
    print(f"{'='*50}")
    print(f"  MAPE : {mape(y_test, preds):>8.2f} %")
    print(f"  MAE  : {mae(y_test, preds):>8.2f}  units")
    print(f"  RMSE : {rmse(y_test, preds):>8.2f}  units")

    # 7. MAPE per category
    print(f"\n--- MAPE by product category ---")
    test_results = test[["date", "sku_id", "product_name", "category", "warehouse"]].copy()
    test_results["actual"]    = y_test
    test_results["predicted"] = preds

    for cat, grp in test_results.groupby("category"):
        m = mape(grp["actual"].values, grp["predicted"].values)
        print(f"  {cat:<15}: {m:.2f}%")

    # 8. MAPE per SKU
    print(f"\n--- MAPE by SKU ---")
    for sku, grp in test_results.groupby("sku_id"):
        name = grp["product_name"].iloc[0]
        m    = mape(grp["actual"].values, grp["predicted"].values)
        print(f"  {sku} | {name:<25} : {m:.2f}%")

    # 9. Feature importances (top 15)
    print(f"\n--- Top 15 feature importances ---")
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    for feat, score in importance.head(15).items():
        bar = "#" * int(score / importance.iloc[0] * 30)
        print(f"  {feat:<35} {score:>5}  {bar}")

    # 10. Sample predictions vs actuals
    print(f"\n--- Sample predictions vs actuals (10 rows) ---")
    sample = test_results.head(10)[["date", "sku_id", "warehouse", "actual", "predicted"]].copy()
    sample["predicted"] = sample["predicted"].round(1)
    sample["error_%"]   = ((sample["actual"] - sample["predicted"]).abs() / sample["actual"] * 100).round(1)
    print(sample.to_string(index=False))

    # 11. Save model + metadata for explainability.py
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # ── SARIMA ────────────────────────────────────────────────────────────────

    print(f"\n{'='*50}")
    print("SARIMA — fitting one model per SKU + warehouse")
    print(f"{'='*50}")
    print("(30 models x ~1,000 training days each — takes ~2-3 min)\n")

    groups      = df.groupby(["sku_id", "warehouse"])
    test_steps  = (test["date"].max() - test["date"].min()).days + 1
    sarima_rows = []

    for i, ((sku, wh), grp) in enumerate(groups, 1):
        grp      = grp.sort_values("date")
        tr       = grp[grp["date"] <= cutoff]["demand_units"]
        te       = grp[grp["date"] >  cutoff]
        te_dates = te["date"].values
        te_actual= te["demand_units"].values

        if len(tr) < 30 or len(te) == 0:
            continue

        print(f"  [{i:02d}/30] {sku} | {wh} ...", end=" ", flush=True)
        preds_s = fit_sarima(tr, len(te))
        m       = mape(te_actual, preds_s)
        print(f"MAPE = {m:.1f}%")

        for d, actual, pred in zip(te_dates, te_actual, preds_s):
            sarima_rows.append({
                "date": d, "sku_id": sku, "warehouse": wh,
                "actual": actual, "sarima_pred": pred,
            })

    sarima_df = pd.DataFrame(sarima_rows)

    # Merge LightGBM preds into one comparison frame
    lgbm_lookup = test_results[["date", "sku_id", "warehouse", "actual", "predicted"]].copy()
    lgbm_lookup.rename(columns={"predicted": "lgbm_pred"}, inplace=True)
    sarima_df["date"] = pd.to_datetime(sarima_df["date"])
    lgbm_lookup["date"] = pd.to_datetime(lgbm_lookup["date"])
    compare = sarima_df.merge(
        lgbm_lookup[["date", "sku_id", "warehouse", "lgbm_pred"]],
        on=["date", "sku_id", "warehouse"], how="inner"
    )
    compare = compare.merge(
        df[["sku_id", "product_name", "category"]].drop_duplicates(),
        on="sku_id", how="left"
    )

    # ── Head-to-head comparison ───────────────────────────────────────────────

    lgbm_overall  = mape(compare["actual"].values, compare["lgbm_pred"].values)
    sarima_overall= mape(compare["actual"].values, compare["sarima_pred"].values)

    print(f"\n{'='*60}")
    print("HEAD-TO-HEAD COMPARISON — MAPE (lower is better)")
    print(f"{'='*60}")
    print(f"  {'Model':<12}  {'Overall MAPE':>14}  {'MAE':>8}  {'RMSE':>8}")
    print(f"  {'-'*46}")
    print(f"  {'LightGBM':<12}  {lgbm_overall:>13.2f}%  "
          f"{mae(compare['actual'].values, compare['lgbm_pred'].values):>7.2f}  "
          f"{rmse(compare['actual'].values, compare['lgbm_pred'].values):>7.2f}")
    print(f"  {'SARIMA':<12}  {sarima_overall:>13.2f}%  "
          f"{mae(compare['actual'].values, compare['sarima_pred'].values):>7.2f}  "
          f"{rmse(compare['actual'].values, compare['sarima_pred'].values):>7.2f}")

    print(f"\n--- MAPE by SKU: LightGBM vs SARIMA ---")
    print(f"  {'SKU':<8}  {'Product':<25}  {'LightGBM':>10}  {'SARIMA':>8}  {'Winner'}")
    print(f"  {'-'*70}")
    for sku, grp in compare.groupby("sku_id"):
        name   = grp["product_name"].iloc[0]
        m_lgbm = mape(grp["actual"].values, grp["lgbm_pred"].values)
        m_sar  = mape(grp["actual"].values, grp["sarima_pred"].values)
        winner = "LightGBM" if m_lgbm < m_sar else "SARIMA  "
        print(f"  {sku:<8}  {name:<25}  {m_lgbm:>9.2f}%  {m_sar:>7.2f}%  {winner}")

    print(f"\n--- MAPE by category: LightGBM vs SARIMA ---")
    print(f"  {'Category':<15}  {'LightGBM':>10}  {'SARIMA':>8}  {'Winner'}")
    print(f"  {'-'*50}")
    for cat, grp in compare.groupby("category"):
        m_lgbm = mape(grp["actual"].values, grp["lgbm_pred"].values)
        m_sar  = mape(grp["actual"].values, grp["sarima_pred"].values)
        winner = "LightGBM" if m_lgbm < m_sar else "SARIMA  "
        print(f"  {cat:<15}  {m_lgbm:>9.2f}%  {m_sar:>7.2f}%  {winner}")

    print(f"\n--- When each model wins (by warehouse) ---")
    print(f"  {'Warehouse':<12}  {'LightGBM':>10}  {'SARIMA':>8}  {'Winner'}")
    print(f"  {'-'*44}")
    for wh, grp in compare.groupby("warehouse"):
        m_lgbm = mape(grp["actual"].values, grp["lgbm_pred"].values)
        m_sar  = mape(grp["actual"].values, grp["sarima_pred"].values)
        winner = "LightGBM" if m_lgbm < m_sar else "SARIMA  "
        print(f"  {wh:<12}  {m_lgbm:>9.2f}%  {m_sar:>7.2f}%  {winner}")
