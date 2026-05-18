"""
Stock Mind — AI-Powered Explainable Inventory Forecasting
Streamlit dashboard  |  run from project root:  streamlit run dashboard/app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock Mind",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE           = Path(__file__).parent.parent
FEATURES_PATH  = BASE / "data/processed/features.csv"
TFT_PREDS_PATH = BASE / "data/processed/tft_preds.csv"
MODEL_PATH     = BASE / "models/lgbm_demand.pkl"
SHAP_GLOBAL    = BASE / "reports/figures/shap_global_importance.png"
SHAP_WATERFALL = BASE / "reports/figures/shap_waterfall.png"

DROP_COLS = [
    "demand_units", "date", "product_name", "shelf_life_days",
    "inventory_level", "stockout_units", "stockout_flag", "days_of_stock",
]
CAT_COLS = ["sku_id", "warehouse", "category"]

PALETTE = px.colors.qualitative.Plotly


# ── Data loaders (cached) ─────────────────────────────────────────────────────

@st.cache_data
def load_features():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    return df

@st.cache_resource
def load_model():
    art = joblib.load(MODEL_PATH)
    return art["model"], art["feature_cols"]

@st.cache_data
def load_tft_preds():
    return pd.read_csv(TFT_PREDS_PATH, parse_dates=["date"])

@st.cache_data
def get_lgbm_test_preds(cutoff_days=90):
    df               = load_features()
    model, feat_cols = load_model()
    cutoff           = df["date"].max() - pd.Timedelta(days=cutoff_days)
    test             = df[df["date"] > cutoff].copy()
    X = test[feat_cols].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    test = test.copy()
    test["lgbm_pred"] = np.clip(model.predict(X), 0, None)
    test["error_pct"] = (
        (test["demand_units"] - test["lgbm_pred"]).abs()
        / test["demand_units"].replace(0, np.nan) * 100
    )
    return test

def mape(actual, predicted):
    mask = actual > 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("📦 Stock Mind")
st.sidebar.caption("AI-Powered Inventory Forecasting")
st.sidebar.markdown("---")

PAGES = {
    "🏠  Overview":          "overview",
    "🔍  Data Explorer":     "explorer",
    "📊  Model Performance": "performance",
    "🔮  Forecast":          "forecast",
    "💡  Explainability":    "explain",
}
page = st.sidebar.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")
selected = PAGES[page]

df    = load_features()
preds = get_lgbm_test_preds()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if selected == "overview":
    st.title("📦 Stock Mind — Inventory Forecasting System")
    st.caption("AI-powered demand forecasting with full explainability across 10 SKUs × 3 warehouses")
    st.markdown("---")

    # KPI row
    total_demand   = int(df["demand_units"].sum())
    total_stockout = df["stockout_flag"].mean() * 100
    active_skus    = df["sku_id"].nunique()
    model_mape     = mape(preds["demand_units"].values, preds["lgbm_pred"].values)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Units Demanded",  f"{total_demand:,}")
    c2.metric("Active SKUs",           f"{active_skus}  ×  3 warehouses")
    c3.metric("Stockout Rate",         f"{total_stockout:.1f}% of days")
    c4.metric("LightGBM MAPE",        f"{model_mape:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Daily Demand — All SKUs")
        daily = (
            df.groupby(["date", "category"])["demand_units"]
            .sum().reset_index()
        )
        fig = px.area(
            daily, x="date", y="demand_units", color="category",
            labels={"demand_units": "Units", "date": "Date", "category": "Category"},
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(height=320, margin=dict(t=10, b=10), legend_title="Category")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Demand by Category")
        cat_total = df.groupby("category")["demand_units"].sum().reset_index()
        fig2 = px.pie(
            cat_total, names="category", values="demand_units",
            color_discrete_sequence=PALETTE, hole=0.4,
        )
        fig2.update_layout(height=320, margin=dict(t=10, b=10), showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Monthly Demand Heatmap (Units by SKU)")
        heat = (
            df.groupby([df["date"].dt.to_period("M").astype(str), "sku_id"])
            ["demand_units"].sum().reset_index()
        )
        heat.columns = ["month", "sku_id", "demand_units"]
        pivot = heat.pivot(index="sku_id", columns="month", values="demand_units").fillna(0)
        fig3 = px.imshow(
            pivot, aspect="auto", color_continuous_scale="Blues",
            labels=dict(color="Units"),
        )
        fig3.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Top 10 Products by Total Demand")
        top = (
            df.groupby(["sku_id", "product_name"])["demand_units"]
            .sum().reset_index().sort_values("demand_units", ascending=True)
        )
        fig4 = px.bar(
            top, x="demand_units", y="product_name", orientation="h",
            labels={"demand_units": "Total Units", "product_name": ""},
            color="demand_units", color_continuous_scale="Blues",
        )
        fig4.update_layout(height=300, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif selected == "explorer":
    st.title("🔍 Data Explorer")
    st.caption("Explore historical demand, inventory levels, and promotions per product")
    st.markdown("---")

    # Filters
    f1, f2, f3 = st.columns(3)
    skus = sorted(df["sku_id"].unique())
    warehouses = sorted(df["warehouse"].unique())

    sel_sku = f1.selectbox("SKU", skus, index=0)
    sel_wh  = f2.selectbox("Warehouse", warehouses, index=0)
    date_min, date_max = df["date"].min().date(), df["date"].max().date()
    date_range = f3.date_input(
        "Date range",
        value=(pd.Timestamp("2024-01-01").date(), date_max),
        min_value=date_min, max_value=date_max,
    )
    if len(date_range) == 2:
        d_start, d_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        d_start, d_end = pd.Timestamp("2024-01-01"), pd.Timestamp(date_max)

    view = df[
        (df["sku_id"] == sel_sku) &
        (df["warehouse"] == sel_wh) &
        (df["date"] >= d_start) &
        (df["date"] <= d_end)
    ].copy()

    prod_name = view["product_name"].iloc[0] if len(view) else sel_sku
    st.subheader(f"{prod_name}  ·  {sel_wh}")

    # KPI strip
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Daily Demand",   f"{view['demand_units'].mean():.1f} units")
    m2.metric("Total Demand",       f"{view['demand_units'].sum():,} units")
    m3.metric("Promo Days",         f"{view['promotion'].sum()} days  ({view['promotion'].mean()*100:.0f}%)")
    m4.metric("Stockout Days",      f"{view['stockout_flag'].sum()} days  ({view['stockout_flag'].mean()*100:.1f}%)")

    st.markdown("---")

    # Demand + rolling mean + promotions
    st.subheader("Demand over Time")
    view["rolling_7d"] = view["demand_units"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    # Promotion shading
    promo_periods = view[view["promotion"] == 1]["date"]
    for d in promo_periods:
        fig.add_vrect(
            x0=d - pd.Timedelta(hours=12),
            x1=d + pd.Timedelta(hours=12),
            fillcolor="gold", opacity=0.25, line_width=0,
        )
    fig.add_trace(go.Scatter(
        x=view["date"], y=view["demand_units"],
        name="Actual Demand", mode="lines",
        line=dict(color="#1f77b4", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=view["date"], y=view["rolling_7d"],
        name="7-day Rolling Mean", mode="lines",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
    ))
    fig.update_layout(
        height=320, margin=dict(t=10, b=10),
        legend=dict(orientation="h", y=1.05),
        xaxis_title="Date", yaxis_title="Units",
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Inventory Level")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=view["date"], y=view["inventory_level"],
            fill="tozeroy", name="Stock on Hand",
            line=dict(color="#2ca02c"),
        ))
        fig2.add_hline(
            y=view["reorder_point"].iloc[0], line_dash="dash",
            line_color="red", annotation_text="Reorder Point",
        )
        fig2.update_layout(height=280, margin=dict(t=10, b=10),
                           xaxis_title="Date", yaxis_title="Units")
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.subheader("Demand Distribution")
        fig3 = px.histogram(
            view, x="demand_units", nbins=30,
            color_discrete_sequence=["#1f77b4"],
            labels={"demand_units": "Daily Demand (units)"},
        )
        promo_mean  = view[view["promotion"] == 1]["demand_units"].mean()
        normal_mean = view[view["promotion"] == 0]["demand_units"].mean()
        fig3.add_vline(x=normal_mean, line_dash="solid",  line_color="#ff7f0e",
                       annotation_text=f"Normal avg: {normal_mean:.0f}")
        fig3.add_vline(x=promo_mean,  line_dash="dash",   line_color="gold",
                       annotation_text=f"Promo avg: {promo_mean:.0f}")
        fig3.update_layout(height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Raw data table"):
        st.dataframe(
            view[["date","demand_units","inventory_level","stockout_units",
                  "promotion","days_of_stock"]].reset_index(drop=True),
            use_container_width=True, height=280,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif selected == "performance":
    st.title("📊 Model Performance")
    st.caption("Head-to-head MAPE comparison across all models on the same 7-day test window")
    st.markdown("---")

    # Pre-computed results (from our earlier runs)
    model_results = pd.DataFrame([
        {"Model": "LightGBM",         "MAPE": 10.58, "MAE": 4.84,  "RMSE": 6.98,  "Type": "ML",          "Horizon": "1-step"},
        {"Model": "SARIMA",           "MAPE": 13.58, "MAE": 6.19,  "RMSE": 10.27, "Type": "Statistical", "Horizon": "7-step"},
        {"Model": "TFT",              "MAPE": 11.16, "MAE": 4.96,  "RMSE": 7.09,  "Type": "Deep Learning","Horizon": "7-step"},
        {"Model": "Ensemble (L+T)",   "MAPE": 10.57, "MAE": 4.79,  "RMSE": 6.84,  "Type": "Ensemble",    "Horizon": "7-step"},
    ])

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    for col, (_, row) in zip([c1, c2, c3, c4], model_results.iterrows()):
        col.metric(
            row["Model"],
            f"{row['MAPE']:.2f}% MAPE",
            delta=f"MAE {row['MAE']:.1f} units",
            delta_color="off",
        )

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("MAPE by Model")
        colors = {"ML": "#1f77b4", "Statistical": "#d62728",
                  "Deep Learning": "#9467bd", "Ensemble": "#2ca02c"}
        fig = px.bar(
            model_results.sort_values("MAPE"),
            x="MAPE", y="Model", orientation="h",
            color="Type", color_discrete_map=colors,
            text="MAPE", labels={"MAPE": "MAPE (%)"},
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(height=280, margin=dict(t=10, b=10),
                          xaxis_range=[0, 16], showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Error Metrics Comparison")
        metrics_long = model_results.melt(
            id_vars="Model", value_vars=["MAPE", "MAE", "RMSE"],
            var_name="Metric", value_name="Value",
        )
        fig2 = px.bar(
            metrics_long, x="Model", y="Value", color="Metric",
            barmode="group",
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
        )
        fig2.update_layout(height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Per-SKU MAPE — LightGBM (90-day test window)")

    sku_mapes = []
    for (sku, wh), grp in preds.groupby(["sku_id", "warehouse"]):
        name = grp["product_name"].iloc[0]
        m    = mape(grp["demand_units"].values, grp["lgbm_pred"].values)
        sku_mapes.append({"SKU": sku, "Warehouse": wh, "Product": name, "MAPE": round(m, 2)})
    sku_df = pd.DataFrame(sku_mapes)

    pivot = sku_df.pivot(index="Product", columns="Warehouse", values="MAPE")
    fig3  = px.imshow(
        pivot, text_auto=".1f", aspect="auto",
        color_continuous_scale="RdYlGn_r",
        labels=dict(color="MAPE %"),
    )
    fig3.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.caption("Darker red = harder to forecast  |  Darker green = more accurate")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════

elif selected == "forecast":
    st.title("🔮 Forecast")
    st.caption("LightGBM predictions vs actual demand on the 90-day test window")
    st.markdown("---")

    skus = sorted(preds["sku_id"].unique())
    whs  = sorted(preds["warehouse"].unique())

    fc1, fc2 = st.columns(2)
    sel_sku = fc1.selectbox("Select SKU", skus)
    sel_wh  = fc2.selectbox("Select Warehouse", whs)

    view = preds[
        (preds["sku_id"] == sel_sku) &
        (preds["warehouse"] == sel_wh)
    ].sort_values("date").copy()

    prod_name = view["product_name"].iloc[0]
    st.subheader(f"{prod_name}  ·  {sel_wh}")

    m1, m2, m3, m4 = st.columns(4)
    vm = mape(view["demand_units"].values, view["lgbm_pred"].values)
    m1.metric("MAPE",            f"{vm:.2f}%")
    m2.metric("Avg Error",       f"{(view['demand_units'] - view['lgbm_pred']).abs().mean():.1f} units")
    m3.metric("Avg Actual",      f"{view['demand_units'].mean():.1f} units/day")
    m4.metric("Avg Predicted",   f"{view['lgbm_pred'].mean():.1f} units/day")

    st.markdown("---")

    # Main forecast chart
    fig = go.Figure()

    # Promotion shading
    for d in view[view["promotion"] == 1]["date"]:
        fig.add_vrect(
            x0=d - pd.Timedelta(hours=12),
            x1=d + pd.Timedelta(hours=12),
            fillcolor="gold", opacity=0.2, line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=view["date"], y=view["demand_units"],
        name="Actual", mode="lines",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=view["date"], y=view["lgbm_pred"],
        name="LightGBM Forecast", mode="lines",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))
    fig.update_layout(
        height=360, margin=dict(t=10, b=10),
        legend=dict(orientation="h", y=1.05),
        xaxis_title="Date", yaxis_title="Units",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    col_e, col_h = st.columns(2)

    with col_e:
        st.subheader("Prediction Error Over Time")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=view["date"], y=view["demand_units"] - view["lgbm_pred"],
            name="Error (Actual − Predicted)",
            marker_color=np.where(
                (view["demand_units"] - view["lgbm_pred"]) >= 0, "#2ca02c", "#d62728"
            ),
        ))
        fig2.add_hline(y=0, line_color="black", line_width=1)
        fig2.update_layout(height=260, margin=dict(t=10, b=10),
                           xaxis_title="Date", yaxis_title="Error (units)")
        st.plotly_chart(fig2, use_container_width=True)

    with col_h:
        st.subheader("Error Distribution")
        errors = view["demand_units"] - view["lgbm_pred"]
        fig3 = px.histogram(
            x=errors, nbins=25,
            color_discrete_sequence=["#1f77b4"],
            labels={"x": "Error (Actual − Predicted)"},
        )
        fig3.add_vline(x=0, line_dash="dash", line_color="red")
        fig3.add_vline(x=errors.mean(), line_dash="dot", line_color="orange",
                       annotation_text=f"Mean: {errors.mean():.1f}")
        fig3.update_layout(height=260, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    # TFT predictions overlay (last 7 days)
    tft = load_tft_preds()
    tft_view = tft[
        (tft["sku_id"] == sel_sku) &
        (tft["warehouse"] == sel_wh)
    ].sort_values("date")

    if len(tft_view):
        st.subheader("Last 7 Days — All Models")
        last7 = view[view["date"].isin(tft_view["date"])].copy()
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=last7["date"], y=last7["demand_units"],
            name="Actual", mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
        ))
        fig4.add_trace(go.Scatter(
            x=last7["date"], y=last7["lgbm_pred"],
            name="LightGBM", mode="lines+markers",
            line=dict(color="#ff7f0e", dash="dash"),
            marker=dict(size=7),
        ))
        fig4.add_trace(go.Scatter(
            x=tft_view["date"], y=tft_view["tft_pred"],
            name="TFT", mode="lines+markers",
            line=dict(color="#9467bd", dash="dot"),
            marker=dict(size=7),
        ))
        fig4.update_layout(
            height=280, margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=1.05),
            xaxis_title="Date", yaxis_title="Units",
        )
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

elif selected == "explain":
    st.title("💡 Explainability")
    st.caption("SHAP values reveal exactly why the model made each prediction")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["Global Feature Importance", "Single Prediction Waterfall", "LightGBM Feature Scores"]
    )

    with tab1:
        st.subheader("What drives demand forecasts globally?")
        st.markdown("""
        Each dot = one prediction from the test set.
        - **Position on x-axis** = how much that feature pushed the forecast **up** (+) or **down** (−)
        - **Colour** = feature value: 🔴 high value → 🔵 low value
        """)
        if SHAP_GLOBAL.exists():
            st.image(str(SHAP_GLOBAL), use_container_width=True)
        else:
            st.warning("Run `src/explainability/explainability.py` first to generate this chart.")

        st.markdown("---")
        st.markdown("""
        **Key takeaways:**
        - `demand_roll_mean_7d` is the strongest signal — the last week's average demand dominates
        - `promotion` has a large, consistent positive push (+3.4 units on average)
        - `day_of_week` matters — weekdays vs weekends drive ~2.5 units of swing
        - `days_to_year_end` captures the holiday season spike
        """)

    with tab2:
        st.subheader("Why did the model predict this specific number?")
        st.markdown("""
        This **waterfall chart** shows a single promotion-day prediction.
        It starts from the **baseline** (what the model predicts knowing nothing specific),
        then each bar shows one feature **pushing the prediction up or down**
        until it reaches the final forecast.
        """)
        if SHAP_WATERFALL.exists():
            st.image(str(SHAP_WATERFALL), use_container_width=True)
        else:
            st.warning("Run `src/explainability/explainability.py` first to generate this chart.")

        st.markdown("""
        **How to read it:**
        - Start at `E[f(x)]` (baseline ≈ average prediction across all data)
        - Each red bar = feature pushing prediction **higher**
        - Each blue bar = feature pushing prediction **lower**
        - End at `f(x)` = the final prediction for this specific row
        """)

    with tab3:
        st.subheader("LightGBM Feature Importances (split-based)")
        model, feat_cols = load_model()
        importance = pd.Series(
            model.feature_importances_,
            index=feat_cols,
        ).sort_values(ascending=False).head(20).reset_index()
        importance.columns = ["Feature", "Importance"]

        fig = px.bar(
            importance.sort_values("Importance"),
            x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            labels={"Importance": "Split Count", "Feature": ""},
        )
        fig.update_layout(
            height=520, margin=dict(t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Split count = how many times the model used this feature to split a decision tree node. "
            "Higher = more relied upon."
        )
