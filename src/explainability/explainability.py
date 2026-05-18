"""
SHAP explainability for the demand forecasting model.

Reads  : data/processed/features.csv
         models/lgbm_demand.pkl
Outputs: reports/figures/shap_global_importance.png   (which features matter most, globally)
         reports/figures/shap_waterfall.png            (why the model made ONE specific prediction)
         printed SHAP summary in terminal
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves to file instead of opening a window
import matplotlib.pyplot as plt
import shap
from pathlib import Path

FEATURES_PATH  = Path("data/processed/features.csv")
MODEL_PATH     = Path("models/lgbm_demand.pkl")
FIGURES_DIR    = Path("reports/figures")

DROP_COLS = [
    "demand_units", "date", "product_name", "shelf_life_days",
    "inventory_level", "stockout_units", "stockout_flag", "days_of_stock",
]
CATEGORICAL_COLS = ["sku_id", "warehouse", "category"]

SHAP_SAMPLE_SIZE = 500   # rows used for global plot — full set is slow


# ── Load model + data ─────────────────────────────────────────────────────────

print("Loading model and features...")
artifact     = joblib.load(MODEL_PATH)
model        = artifact["model"]
feature_cols = artifact["feature_cols"]

df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
cutoff = df["date"].max() - pd.Timedelta(days=90)
test   = df[df["date"] > cutoff].copy()

X_test = test[feature_cols].copy()
y_test = test["demand_units"].values

for col in CATEGORICAL_COLS:
    X_test[col] = X_test[col].astype("category")

print(f"  Test rows : {len(X_test):,}")
print(f"  Features  : {len(feature_cols)}")


# ── SHAP explainer ────────────────────────────────────────────────────────────

print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(model)

# Sample for global plot (faster; 500 rows is representative)
rng         = np.random.default_rng(42)
sample_idx  = rng.choice(len(X_test), size=min(SHAP_SAMPLE_SIZE, len(X_test)), replace=False)
X_sample    = X_test.iloc[sample_idx].reset_index(drop=True)

shap_values = explainer.shap_values(X_sample)   # shape: (500, n_features)
expected_value = explainer.expected_value        # model's average prediction (baseline)

print(f"  SHAP values shape : {shap_values.shape}")
print(f"  Baseline (expected value) : {expected_value:.2f} units  "
      f"(model predicts this when it knows nothing specific)")


# ── Global feature importance ─────────────────────────────────────────────────
# Mean absolute SHAP value per feature = average impact on prediction

print("\n--- Global SHAP feature importance (top 20) ---")
mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=feature_cols
).sort_values(ascending=False)

for feat, val in mean_abs_shap.head(20).items():
    bar = "#" * int(val / mean_abs_shap.iloc[0] * 35)
    print(f"  {feat:<35}  {val:>5.2f}  {bar}")


# ── Plot 1: Global SHAP summary (beeswarm) ────────────────────────────────────
# Each dot = one row in the sample. Red = high feature value, blue = low.
# Position on x-axis = how much that feature pushed the prediction up or down.

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("\nSaving global SHAP summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_cols,
    max_display=20,
    show=False,
    plot_size=None,
)
plt.title("Global SHAP Feature Importance\n(each dot = one prediction; red = high value, blue = low)",
          fontsize=11, pad=12)
plt.tight_layout()
global_path = FIGURES_DIR / "shap_global_importance.png"
plt.savefig(global_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {global_path}")


# ── Pick an interesting row for the waterfall chart ───────────────────────────
# We want a promotion day so the chart shows a clear "promotion pushed demand up" story.

promo_rows = test[test["promotion"] == 1].index
if len(promo_rows) == 0:
    chosen_pos = 0   # fallback: first row
else:
    # Pick the promotion row with the highest actual demand (most dramatic story)
    chosen_global_idx = test.loc[promo_rows, "demand_units"].idxmax()
    chosen_pos = test.index.get_loc(chosen_global_idx)

X_single    = X_test.iloc[[chosen_pos]]
y_actual    = y_test[chosen_pos]
y_predicted = float(np.clip(model.predict(X_single)[0], 0, None))
row_meta    = test.iloc[chosen_pos]

print(f"\nWaterfall chart row:")
print(f"  Date       : {row_meta['date'].date()}")
print(f"  SKU        : {row_meta['sku_id']} — {row_meta['product_name']}")
print(f"  Warehouse  : {row_meta['warehouse']}")
print(f"  Promotion  : {'YES' if row_meta['promotion'] else 'NO'}")
print(f"  Actual demand    : {y_actual:.0f} units")
print(f"  Predicted demand : {y_predicted:.1f} units")


# ── Plot 2: Local waterfall chart ─────────────────────────────────────────────
# Starts at the baseline (average prediction), then adds/subtracts each feature's
# contribution until it reaches THIS prediction. Shows exactly WHY the model
# predicted what it did for this one row.

shap_single = explainer.shap_values(X_single)   # shape: (1, n_features)

# Build shap.Explanation object for the waterfall API
explanation = shap.Explanation(
    values        = shap_single[0],
    base_values   = expected_value,
    data          = X_single.iloc[0].values,
    feature_names = feature_cols,
)

print("\nSaving waterfall chart...")
plt.figure(figsize=(10, 8))
shap.plots.waterfall(explanation, max_display=15, show=False)
plt.title(
    f"Why did the model predict {y_predicted:.0f} units?\n"
    f"{row_meta['product_name']} | {row_meta['warehouse']} | "
    f"{row_meta['date'].date()} | Actual: {y_actual:.0f}",
    fontsize=10, pad=12
)
plt.tight_layout()
waterfall_path = FIGURES_DIR / "shap_waterfall.png"
plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {waterfall_path}")


# ── Terminal summary ──────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print("SHAP EXPLAINABILITY SUMMARY")
print(f"{'='*55}")
print(f"  Baseline prediction (no context) : {expected_value:.1f} units")
print(f"  Chosen row actual demand         : {y_actual:.0f} units")
print(f"  Chosen row model prediction      : {y_predicted:.1f} units")

print(f"\n  Top 5 features driving THIS prediction:")
top_shap = pd.Series(shap_single[0], index=feature_cols).reindex(
    pd.Series(shap_single[0], index=feature_cols).abs().sort_values(ascending=False).index
).head(5)
for feat, val in top_shap.items():
    direction = "UP  (+)" if val > 0 else "DOWN (-)"
    feat_val  = X_single[feat].values[0]
    print(f"    {feat:<35}  pushed {direction}  by {abs(val):.2f}  [value={feat_val}]")

print(f"\n  Output files:")
print(f"    {global_path}")
print(f"    {waterfall_path}")
