"""
TFT (Temporal Fusion Transformer) demand forecasting.

Reads  : data/processed/features.csv
         models/lgbm_demand.pkl  (for head-to-head comparison)
Outputs: printed metrics + TFT variable importances

Architecture highlights:
  - Static covariates      : product / warehouse identity (never changes over time)
  - Known future inputs    : calendar features + planned promotions (knowable ahead of time)
  - Unknown past inputs    : historical demand, rolling stats (only known from history)
  - Variable Selection     : attention-based gating picks the most relevant features
  - Multi-head Attention   : captures long-range temporal dependencies

Forecast setup:
  encoder_length    = 30 days of history fed as context window
  prediction_length =  7 days forecast horizon (all 7 steps at once)

Comparison note:
  LightGBM  -> 1-step-ahead (uses actual yesterday lag as feature — easier task)
  TFT       -> 7-step-ahead (forecasts all 7 days at once — harder task)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import joblib
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE as PFMAE
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

FEATURES_PATH = Path("data/processed/features.csv")
MODEL_PATH    = Path("models/lgbm_demand.pkl")

MAX_ENCODER_LENGTH    = 14    # 2 weeks of context (was 30)
MAX_PREDICTION_LENGTH =  7    # 1 week forecast
BATCH_SIZE            = 512   # larger batch = fewer steps per epoch
MAX_EPOCHS            = 10    # hard cap (was 25)
HIDDEN_SIZE           = 16    # half the size (was 32)
ATTN_HEADS            = 1     # single head (was 2)
DROPOUT               = 0.1

# Feature groups — TFT requires knowing which features are known in the future
STATIC_CATS   = ["sku_id", "warehouse", "category"]
STATIC_REALS  = ["unit_cost", "lead_time_days"]

# Known in advance (calendar is always known; promotion assumed pre-planned)
KNOWN_CATS    = ["is_weekend", "promotion", "is_month_start", "is_month_end",
                 "is_quarter_start", "is_quarter_end"]
KNOWN_REALS   = ["time_idx", "day_of_week", "month", "quarter",
                 "day_of_month", "week_of_year", "days_to_year_end"]

# Only observable from history (demand and derived signals)
UNKNOWN_REALS = [
    "demand_units",
    "demand_lag_1d", "demand_lag_7d",
    "demand_roll_mean_7d", "demand_roll_mean_28d",
    "demand_roll_std_7d", "demand_momentum", "stock_cover_7d",
]


# ── Metric ────────────────────────────────────────────────────────────────────

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual > 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


# ── Data preparation ──────────────────────────────────────────────────────────

print("Loading features...")
df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
df = df.sort_values(["sku_id", "warehouse", "date"]).reset_index(drop=True)
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Global integer time index (same value for same date across all groups)
df["time_idx"] = (df["date"] - df["date"].min()).dt.days.astype("int64")

# pytorch-forecasting needs categorical columns as strings
for col in STATIC_CATS:
    df[col] = df[col].astype(str)
for col in KNOWN_CATS:
    df[col] = df[col].astype(str)

# All real-valued features must be float
for col in STATIC_REALS + KNOWN_REALS + UNKNOWN_REALS:
    df[col] = df[col].astype(float)

# Fill any NaN in unknown reals with 0 (stockout_rate_28d can be NaN early on)
df[UNKNOWN_REALS] = df[UNKNOWN_REALS].fillna(0)

max_idx         = int(df["time_idx"].max())
training_cutoff = max_idx - MAX_PREDICTION_LENGTH
# Use only last 180 days for training — enough to learn patterns, far faster
train_start_idx = max_idx - 180
print(f"  Training on last 180 days | Test: last {MAX_PREDICTION_LENGTH} days")


# ── TimeSeriesDataSet ─────────────────────────────────────────────────────────

print("\nBuilding TimeSeriesDataSet...")
train_df = df[(df["time_idx"] >= train_start_idx) & (df["time_idx"] <= training_cutoff)].copy()
train_df["time_idx"] = train_df["time_idx"].astype(int)

training_dataset = TimeSeriesDataSet(
    train_df,
    time_idx             = "time_idx",
    target               = "demand_units",
    group_ids            = ["sku_id", "warehouse"],
    max_encoder_length   = MAX_ENCODER_LENGTH,
    max_prediction_length= MAX_PREDICTION_LENGTH,
    static_categoricals  = STATIC_CATS,
    static_reals         = STATIC_REALS,
    time_varying_known_categoricals = KNOWN_CATS,
    time_varying_known_reals        = KNOWN_REALS,
    time_varying_unknown_reals      = UNKNOWN_REALS,
    target_normalizer    = GroupNormalizer(
        groups=["sku_id", "warehouse"],
        transformation="softplus",   # keeps predictions positive
    ),
    add_relative_time_idx = True,    # adds "how far along in encoder/decoder" signal
    add_target_scales     = True,    # adds per-group mean/std as static features
    add_encoder_length    = True,    # adds actual encoder length used as feature
    allow_missing_timesteps = False,
)

full_df = df.copy()
full_df["time_idx"] = full_df["time_idx"].astype(int)

validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset, full_df, predict=True, stop_randomization=True
)

train_loader = training_dataset.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
)
val_loader = validation_dataset.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0
)

print(f"  Training samples : {len(training_dataset):,}")
print(f"  Validation samples: {len(validation_dataset):,}")


# ── TFT model ─────────────────────────────────────────────────────────────────

print("\nBuilding TFT model...")
tft = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate          = 3e-3,
    hidden_size            = HIDDEN_SIZE,
    attention_head_size    = ATTN_HEADS,
    dropout                = DROPOUT,
    hidden_continuous_size = 8,
    loss                   = PFMAE(),
    reduce_on_plateau_patience = 3,
    log_interval           = -1,    # suppress per-batch logs
    log_val_interval       = 1,
)
print(f"  Parameters: {sum(p.numel() for p in tft.parameters()):,}")


# ── Training ──────────────────────────────────────────────────────────────────

print(f"\nTraining TFT ({MAX_EPOCHS} epochs max, early stopping on val_loss)...")
trainer = pl.Trainer(
    max_epochs           = MAX_EPOCHS,
    accelerator          = "cpu",
    gradient_clip_val    = 0.1,
    enable_model_summary = False,
    enable_progress_bar  = True,
    log_every_n_steps    = 5,
    logger               = False,
    callbacks            = [EarlyStopping(monitor="val_loss", patience=3, mode="min",
                                          verbose=True)],
)
trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)


# ── Evaluation ────────────────────────────────────────────────────────────────

print("\nGenerating TFT predictions on test set...")
raw_result = tft.predict(
    val_loader,
    return_y              = True,
    trainer_kwargs        = dict(accelerator="cpu", logger=False,
                                 enable_progress_bar=False),
)

# Handle both tuple and named-tuple return shapes across pf versions
if hasattr(raw_result, "output"):
    tft_preds   = raw_result.output.numpy().flatten()
    tft_actuals = raw_result.y[0].numpy().flatten()
else:
    tft_preds, tft_actuals = raw_result
    tft_preds   = tft_preds.numpy().flatten()
    tft_actuals = tft_actuals.numpy().flatten()

tft_preds = np.clip(tft_preds, 0, None)

tft_mape = mape(tft_actuals, tft_preds)
tft_mae  = float(np.mean(np.abs(tft_actuals - tft_preds)))
tft_rmse = float(np.sqrt(np.mean((tft_actuals - tft_preds) ** 2)))


# ── LightGBM on same window (last 7 days) ─────────────────────────────────────

print("Loading LightGBM for same-window comparison...")
artifact     = joblib.load(MODEL_PATH)
lgbm_model   = artifact["model"]
feature_cols = artifact["feature_cols"]

DROP_COLS = [
    "demand_units", "date", "product_name", "shelf_life_days",
    "inventory_level", "stockout_units", "stockout_flag", "days_of_stock",
]
df_raw = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
cutoff_date  = df_raw["date"].max() - pd.Timedelta(days=MAX_PREDICTION_LENGTH - 1)
test_window  = df_raw[df_raw["date"] >= cutoff_date].copy()

X_test_lgbm = test_window[feature_cols].copy()
y_test_lgbm = test_window["demand_units"].values

for col in ["sku_id", "warehouse", "category"]:
    X_test_lgbm[col] = X_test_lgbm[col].astype("category")

lgbm_preds = np.clip(lgbm_model.predict(X_test_lgbm), 0, None)

lgbm_mape = mape(y_test_lgbm, lgbm_preds)
lgbm_mae  = float(np.mean(np.abs(y_test_lgbm - lgbm_preds)))
lgbm_rmse = float(np.sqrt(np.mean((y_test_lgbm - lgbm_preds) ** 2)))

# Save TFT predictions for ensemble.py
tft_out = test_window[["date", "sku_id", "warehouse", "demand_units"]].copy()
tft_out = tft_out.sort_values(["sku_id", "warehouse", "date"]).reset_index(drop=True)
tft_out["tft_pred"] = np.nan
if len(tft_preds) == len(tft_out):
    tft_out["tft_pred"] = tft_preds
else:
    tft_out["tft_pred"] = np.resize(tft_preds, len(tft_out))
tft_out.to_csv("data/processed/tft_preds.csv", index=False)
print(f"TFT predictions saved to data/processed/tft_preds.csv")


# ── Head-to-head comparison ───────────────────────────────────────────────────

print(f"\n{'='*62}")
print("HEAD-TO-HEAD: TFT vs LightGBM (same 7-day test window)")
print(f"{'='*62}")
print(f"  {'Model':<14}  {'MAPE':>8}  {'MAE':>8}  {'RMSE':>8}  Note")
print(f"  {'-'*58}")
print(f"  {'TFT':<14}  {tft_mape:>7.2f}%  {tft_mae:>7.2f}  {tft_rmse:>7.2f}  "
      f"7-step-ahead (all at once)")
print(f"  {'LightGBM':<14}  {lgbm_mape:>7.2f}%  {lgbm_mae:>7.2f}  {lgbm_rmse:>7.2f}  "
      f"1-step-ahead (uses actual lags)")
winner = "TFT" if tft_mape < lgbm_mape else "LightGBM"
print(f"\n  Winner on MAPE: {winner}")
print(f"  (TFT is tackling a harder task — it predicts 7 days at once\n"
      f"   without access to actual demand values within the horizon)")


# ── TFT Variable Importances ──────────────────────────────────────────────────
# TFT's Variable Selection Network learns a soft attention weight per feature.
# This tells us which features the model relied on most during encoding/decoding.

print(f"\n{'='*62}")
print("TFT VARIABLE IMPORTANCES (from Variable Selection Network)")
print(f"{'='*62}")

try:
    raw_predictions = tft.predict(
        val_loader, mode="raw", return_x=True,
        trainer_kwargs=dict(accelerator="cpu", logger=False,
                            enable_progress_bar=False),
    )
    interpretation = tft.interpret_output(raw_predictions.output, reduction="sum")

    print("\n  Encoder variables (what TFT watches in history):")
    enc_imp = interpretation["encoder_variables"].numpy()
    enc_names = (
        training_dataset.reals                         # continuous
        + training_dataset.flat_categoricals           # categorical
    )
    enc_pairs = sorted(zip(enc_names, enc_imp), key=lambda x: -x[1])
    for name, score in enc_pairs[:12]:
        bar = "#" * int(score / enc_pairs[0][1] * 30)
        print(f"    {name:<35}  {score:>6.1f}  {bar}")

    print("\n  Decoder variables (what TFT uses when projecting forward):")
    dec_imp = interpretation["decoder_variables"].numpy()
    dec_names = (
        [r for r in training_dataset.reals if r in KNOWN_REALS + ["relative_time_idx"]]
        + [c for c in training_dataset.flat_categoricals if c in KNOWN_CATS]
    )
    dec_pairs = sorted(zip(dec_names[:len(dec_imp)], dec_imp), key=lambda x: -x[1])
    for name, score in dec_pairs[:8]:
        bar = "#" * int(score / dec_pairs[0][1] * 30)
        print(f"    {name:<35}  {score:>6.1f}  {bar}")

    print("\n  Static variables (product-level context):")
    sta_imp = interpretation["static_variables"].numpy()
    sta_names = training_dataset.static_reals + training_dataset.static_categoricals
    sta_pairs = sorted(zip(sta_names[:len(sta_imp)], sta_imp), key=lambda x: -x[1])
    for name, score in sta_pairs:
        bar = "#" * int(score / sta_pairs[0][1] * 30)
        print(f"    {name:<35}  {score:>6.1f}  {bar}")

except Exception as e:
    print(f"  (Variable importance extraction skipped: {e})")

print(f"\nDone.")
