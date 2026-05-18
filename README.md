# 📦 Stock Mind — Explainable AI-Based Smart Demand Forecasting System

A production-ready inventory demand forecasting system that combines classical statistics, machine learning, and deep learning — with full explainability powered by SHAP.

---

## 🎯 What It Does

Most forecasting systems tell you **what** demand will be. This system also tells you **why** — which features drove each prediction, how confident the model is, and where each model wins or loses.

| Capability | Detail |
|---|---|
| **Forecasting** | Daily demand per SKU × Warehouse |
| **Explainability** | SHAP global importance + per-prediction waterfall |
| **Models** | LightGBM · SARIMA · TFT · Ensemble |
| **Dashboard** | 5-page interactive Streamlit app |
| **Data** | 10 SKUs × 3 Warehouses × 3 Years (32,880 rows) |

---

## 🏗️ Architecture

```
Raw Data → Feature Engineering → Model Training → Explainability → Dashboard
   ↓               ↓                    ↓                ↓              ↓
generate_data   data_pipeline     classical.py      explainability   app.py
   .py              .py           deep_learning.py      .py
                               ensemble.py
```

### Model Results

| Model | Type | MAPE | Horizon |
|---|---|---|---|
| LightGBM | Gradient Boosting | 10.46% | 1-step-ahead |
| SARIMA | Statistical | 24.39% | 7-step-ahead |
| TFT | Transformer (Deep Learning) | 11.16% | 7-step-ahead |
| **Ensemble (LGBM + TFT)** | Blended | **10.57%** | 7-step-ahead |

---

## 📁 Project Structure

```
Stock_Mind/
├── dashboard/
│   └── app.py                  # Streamlit dashboard (5 pages)
├── data/
│   ├── raw/
│   │   └── inventory_data.csv  # Synthetic dataset
│   └── processed/
│       ├── features.csv        # Engineered features
│       └── tft_preds.csv       # TFT test predictions
├── models/
│   └── lgbm_demand.pkl         # Trained LightGBM model
├── reports/figures/
│   ├── shap_global_importance.png
│   └── shap_waterfall.png
├── scripts/
│   └── generate_data.py        # Synthetic data generator
├── src/
│   ├── data/
│   │   └── data_pipeline.py    # Feature engineering pipeline
│   ├── models/
│   │   ├── classical.py        # LightGBM + SARIMA
│   │   ├── deep_learning.py    # TFT (Temporal Fusion Transformer)
│   │   └── ensemble.py         # Model blending
│   └── explainability/
│       └── explainability.py   # SHAP analysis + charts
├── requirements.txt
└── runtime.txt
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/kumkum-kaushik/StockMind.git
cd StockMind
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔄 Reproducing the Full Pipeline

Run these scripts in order to regenerate everything from scratch:

```bash
# Step 1 — Generate synthetic data
python scripts/generate_data.py

# Step 2 — Engineer features
python src/data/data_pipeline.py

# Step 3 — Train LightGBM + SARIMA
python src/models/classical.py

# Step 4 — Train TFT (deep learning)
python src/models/deep_learning.py

# Step 5 — Blend all three models
python src/models/ensemble.py

# Step 6 — Generate SHAP explainability charts
python src/explainability/explainability.py
```

> **Note:** Step 4 requires additional packages.
> Install with: `pip install pytorch-forecasting lightning`

---

## 📊 Dashboard Pages

| Page | What you will find |
|---|---|
| 🏠 **Overview** | KPI cards, demand by category, monthly heatmap |
| 🔍 **Data Explorer** | Per-SKU demand, promotions, inventory level charts |
| 📊 **Model Performance** | MAPE comparison, per-SKU accuracy heatmap |
| 🔮 **Forecast** | Actual vs predicted, error distribution, all-model overlay |
| 💡 **Explainability** | SHAP global importance, waterfall chart, feature scores |

---

## 🧠 Feature Engineering

28 new features engineered from raw daily demand:

- **Lag features** — demand 1, 7, 14, 28 days ago
- **Rolling statistics** — mean, std, min, max over 7/14/28-day windows
- **Demand momentum** — short-term vs long-term average ratio
- **Calendar effects** — day of week, month, quarter, proximity to year-end
- **Inventory signals** — stock coverage days, lagged stockout flag

---

## 📈 Dataset

Synthetic but realistic — 3 years of daily inventory data with:

- **Trend** — 15% growth over 3 years
- **Seasonality** — Electronics spikes Q4, Health spikes Jan, Sports spikes summer
- **Promotions** — random bursts boosting demand 30–70%
- **Inventory simulation** — reorder policy with variable supplier lead times
- **Stockouts** — 0.2% of days have unmet demand

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, numpy |
| ML model | LightGBM |
| Statistical model | statsmodels (SARIMA) |
| Deep learning | PyTorch Forecasting (TFT) |
| Explainability | SHAP |
| Visualisation | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## 📄 License

MIT License — free to use, modify, and distribute.
