"""Interactive dashboard for aggregate Rossmann study results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"

st.set_page_config(
    page_title="Rossmann Demand & Inventory Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_artifacts():
    required = [
        "dashboard_model_metrics.csv",
        "validation_metrics.csv",
        "dashboard_inventory_summary.csv",
        "test_error_by_weekday.csv",
        "test_error_by_promo.csv",
        "dashboard_experiment.json",
    ]
    absent = [name for name in required if not (RESULTS / name).is_file()]
    if absent:
        raise FileNotFoundError("Missing aggregate dashboard artifact(s): " + ", ".join(absent))
    return (
        pd.read_csv(RESULTS / "dashboard_model_metrics.csv"),
        pd.read_csv(RESULTS / "validation_metrics.csv"),
        pd.read_csv(RESULTS / "dashboard_inventory_summary.csv"),
        pd.read_csv(RESULTS / "test_error_by_weekday.csv"),
        pd.read_csv(RESULTS / "test_error_by_promo.csv"),
        json.loads((RESULTS / "dashboard_experiment.json").read_text()),
    )


st.title("Deep Demand Forecasting & Inventory Intelligence")
st.caption("Rossmann Store Sales · reproducible 50-store CPU experiment")

try:
    metrics_df, validation_metrics, inventory, weekday_errors, promo_errors, experiment = load_artifacts()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Could not load the aggregate experiment artifacts: {exc}")
    st.info("Run prepare_data.py, forecast.py, inventory.py, error_analysis.py, and publish_dashboard_data.py from the repository root first.")
    st.stop()

best_model = str(validation_metrics.iloc[0]["model"])
date_info = experiment["split_dates"]

with st.sidebar:
    st.header("Experiment scope")
    st.metric("Sampled stores", experiment["sample_scope"].split()[0])
    st.metric("Test predictions", f"{experiment['test_rows_sequence']:,}")
    st.write(f"**Validation:** {date_info['validation_start']} to before {date_info['test_start']}")
    st.write(f"**Test:** {date_info['test_start']} to {date_info['last_date']}")
    st.caption("Rolling one-day-ahead; prior observed sales may inform later dates. Model fitting is train-only.")
    st.warning("Results cover 50 sampled stores only, not all 1,115. Sales is a monetary value, not SKU unit demand.")

overview, errors, inventory_tab, methodology = st.tabs(["Model comparison", "Error analysis", "Inventory proxy", "Methodology"])

with overview:
    st.subheader("Chronological held-out model comparison")
    st.caption("The inventory demonstration uses the model selected by validation WAPE. Test scores are reported separately.")
    selected_row = metrics_df.loc[metrics_df.model == best_model].iloc[0]
    first, second, third, fourth = st.columns(4)
    first.metric("Validation-selected model", best_model)
    second.metric("Test WAPE", f"{selected_row.WAPE:.2%}")
    third.metric("Test MAE", f"{selected_row.MAE:,.1f}")
    fourth.metric("Test RMSE", f"{selected_row.RMSE:,.1f}")
    left, right = st.columns([1, 1.15])
    with left:
        chart_data = metrics_df.set_index("model")[["WAPE"]].sort_values("WAPE")
        st.bar_chart(chart_data, y="WAPE", height=360)
    with right:
        display = metrics_df[["model", "n", "RMSE", "MAE", "WAPE"]].copy()
        display["WAPE"] = display.WAPE.map(lambda value: f"{value:.2%}")
        st.dataframe(display, hide_index=True, width="stretch")
    st.download_button("Download aggregate model metrics", metrics_df.to_csv(index=False), "model_metrics.csv", "text/csv")

with errors:
    st.subheader(f"Aggregate test errors: {best_model}")
    st.caption("Error slices summarize the test period; they do not expose individual dates or stores.")
    left, right = st.columns(2)
    with left:
        st.markdown("**By weekday**")
        wk = weekday_errors.set_index("DayOfWeek")[["WAPE"]]
        st.bar_chart(wk, y="WAPE")
        st.dataframe(weekday_errors, hide_index=True, width="stretch")
        st.caption("Sunday WAPE has a small total-sales denominator because many stores are closed; interpret it alongside MAE/RMSE.")
    with right:
        st.markdown("**By promotion flag**")
        promo = promo_errors.copy()
        promo["WAPE"] = promo.WAPE.map(lambda value: f"{value:.2%}")
        st.dataframe(promo, hide_index=True, width="stretch")
        st.caption("Promotion groups are observational; their error difference does not estimate promotion causality.")

with inventory_tab:
    st.subheader("Aggregate reorder-point proxy")
    st.warning("Rossmann has no SKU quantities, unit costs, or store-specific lead times. These sales-value proxies are a formula demonstration, not purchasing instructions.")
    service = st.selectbox("Target service-level assumption", [90, 95, 99], index=1)
    row = inventory.loc[inventory.service_level_pct == service].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Stores aggregated", f"{int(row.stores_aggregated)}")
    b.metric("Median reorder-point proxy", f"{row.median_reorder_point_proxy:,.0f}")
    c.metric("90th percentile proxy", f"{row.p90_reorder_point_proxy:,.0f}")
    st.dataframe(inventory, hide_index=True, width="stretch")
    st.download_button("Download aggregate inventory summary", inventory.to_csv(index=False), "inventory_summary.csv", "text/csv")
    st.caption("Formula: 3-day forecast demand + z × pooled validation residual SD × √3. Assumes independent daily errors and a constant three-day lead time.")

with methodology:
    st.subheader("Method and limits")
    st.markdown(
        """
        - Fixed random sample of 50 stores; sample IDs and seed 42 are recorded in the repository.
        - Chronological 42-day validation and test blocks. Validation WAPE selects the inventory demonstration model.
        - Baselines: previous day, same weekday one week earlier, and the previous seven calendar-day mean.
        - Comparisons: Linear Regression, 1D CNN, vanilla RNN, LSTM, and GRU.
        - Exact calendar-date lag joins preserve absent dates; no row-based lag crosses missing history.
        - `Customers` and same-day target `Sales` are excluded as predictors.

        The dashboard publishes aggregate metrics and group summaries only. It does not include raw Kaggle rows, per-date predictions, per-store tables, uploads, or retraining. To reproduce training locally, obtain the files from Kaggle under the competition's terms and follow the repository README.
        """
    )
    st.json(experiment)
    st.download_button("Download public experiment metadata", json.dumps(experiment, indent=2), "experiment.json", "application/json")

st.divider()
st.caption("Educational forecasting study · see README and PROGRESS for full reproducibility details.")
