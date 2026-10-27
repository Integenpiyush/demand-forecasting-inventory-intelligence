"""Streamlit dashboard for point forecasts, quantile calibration, and inventory proxies."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
sys.path.insert(0, str(ROOT))
from src.inventory import contiguous_lead_time_errors, empirical_safety_stock  # noqa: E402

st.set_page_config(page_title="Rossmann Demand & Inventory Intelligence", page_icon="📈", layout="wide")


@st.cache_data
def load_artifacts():
    required = [
        "dashboard_model_metrics.csv", "validation_metrics.csv", "test_error_by_weekday.csv",
        "test_error_by_promo.csv", "dashboard_experiment.json",
        "dashboard_transformer_quantile_predictions.csv", "dashboard_transformer_quantile_metrics.csv",
        "dashboard_transformer_calibration_slices.csv",
    ]
    absent = [name for name in required if not (RESULTS / name).is_file()]
    if absent:
        raise FileNotFoundError("Missing dashboard artifacts: " + ", ".join(absent))
    quantiles = pd.read_csv(RESULTS / "dashboard_transformer_quantile_predictions.csv", parse_dates=["Date"])
    return (
        pd.read_csv(RESULTS / "dashboard_model_metrics.csv"),
        pd.read_csv(RESULTS / "validation_metrics.csv"),
        pd.read_csv(RESULTS / "test_error_by_weekday.csv"),
        pd.read_csv(RESULTS / "test_error_by_promo.csv"),
        json.loads((RESULTS / "dashboard_experiment.json").read_text()), quantiles,
        pd.read_csv(RESULTS / "dashboard_transformer_quantile_metrics.csv"),
        pd.read_csv(RESULTS / "dashboard_transformer_calibration_slices.csv"),
    )


st.title("Deep Demand Forecasting & Inventory Intelligence")
st.caption("Rossmann Store Sales · fixed 50-store sample · CPU-trained experiments")

try:
    model_metrics, validation_metrics, weekday_errors, promo_errors, experiment, quantiles, qmetrics, calibration_slices = load_artifacts()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Could not load the experiment artifacts: {exc}")
    st.info("Run prepare_data.py, forecast.py, probabilistic.py, inventory.py, error_analysis.py, and publish_dashboard_data.py in that order.")
    st.stop()

selected_model = str(validation_metrics.sort_values("WAPE").iloc[0].model)
date_info = experiment["split_dates"]
validation_q = quantiles[quantiles.split == "validation"].copy()
test_q = quantiles[quantiles.split == "test"].copy()

with st.sidebar:
    st.header("Experiment scope")
    st.metric("Sampled stores", "50")
    st.metric("Test targets", f"{len(test_q):,}")
    st.write(f"**Validation:** {date_info['validation_start']} to before {date_info['test_start']}")
    st.write(f"**Test:** {date_info['test_start']} to {date_info['last_date']}")
    st.caption("Rolling one-day-ahead evaluation; prior observed sales can enter later inputs. Test targets never fit model parameters.")
    st.warning("Results cover this 50-store sample only. Sales is a monetary measure, not item units.")

overview, errors, probabilistic, store_analysis, inventory_tab, methodology = st.tabs(
    ["Model comparison", "Error analysis", "Probabilistic forecasts", "Store analysis", "Inventory scenarios", "Methodology"]
)

with overview:
    st.subheader("Chronological model comparison")
    st.caption(f"Selected on validation WAPE: {selected_model}. Final test metrics are reported separately.")
    display = model_metrics.sort_values("WAPE").copy()
    if selected_model == "Transformer":
        selected_row = display.loc[display.model == "Transformer"].iloc[0]
    else:
        selected_row = display.loc[display.model == selected_model].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Validation-selected model", selected_model)
    b.metric("Test WAPE", f"{selected_row.WAPE:.2%}")
    c.metric("Test MAE", f"{selected_row.MAE:,.1f}")
    d.metric("Test RMSE", f"{selected_row.RMSE:,.1f}")
    left, right = st.columns([1, 1.2])
    with left:
        st.bar_chart(display.set_index("model")[["WAPE"]].sort_values("WAPE"), y="WAPE", height=360)
    with right:
        display["WAPE"] = display.WAPE.map(lambda x: f"{x:.2%}")
        st.dataframe(display[["model", "n", "RMSE", "MAE", "WAPE"]], hide_index=True, width="stretch")
    st.markdown("**Validation metrics**")
    val_view = validation_metrics.sort_values("WAPE").copy()
    val_view["WAPE"] = val_view.WAPE.map(lambda x: f"{x:.2%}")
    st.dataframe(val_view[["model", "n", "RMSE", "MAE", "WAPE"]], hide_index=True, width="stretch")
    st.download_button("Download model metrics", model_metrics.to_csv(index=False), "model_metrics.csv", "text/csv")

with errors:
    st.subheader(f"Held-out test errors: {selected_model}")
    left, right = st.columns(2)
    with left:
        st.markdown("**By weekday**")
        st.bar_chart(weekday_errors.set_index("DayOfWeek")[["WAPE"]], y="WAPE")
        st.dataframe(weekday_errors, hide_index=True, width="stretch")
        st.caption("Sunday WAPE may be sensitive to the aggregate sales denominator; review MAE and RMSE too.")
    with right:
        st.markdown("**By promotion flag**")
        promo_view = promo_errors.copy()
        promo_view["WAPE"] = promo_view.WAPE.map(lambda value: f"{value:.2%}")
        st.dataframe(promo_view, hide_index=True, width="stretch")
        st.caption("These groups are observational and do not measure a causal promotion effect.")

with probabilistic:
    st.subheader("P10, P50, and P90 predictive quantiles")
    st.caption("These are predictive quantiles, not confidence intervals. P10–P90 is the model's central 80% predictive range.")
    st.dataframe(qmetrics, hide_index=True, width="stretch")
    qm_test = qmetrics[qmetrics.split == "test"]
    if not qm_test.empty:
        row = qm_test.iloc[0]
        x, y, z = st.columns(3)
        x.metric("P10–P90 empirical coverage", f"{row.p10_p90_coverage:.1%}")
        y.metric("P10 pinball loss", f"{row.pinball_p10:,.1f}")
        z.metric("P90 pinball loss", f"{row.pinball_p90:,.1f}")
    st.caption("Coverage is the observed fraction of actuals between P10 and P90. The nonnegative-sales constraint creates a zero-demand mass, so P10 coverage can exceed its nominal 10% even after validation calibration. Calibration tables include promotion, weekday, and store slices where sample size is adequate.")
    st.markdown("**Test calibration by segment**")
    segment = st.selectbox("Calibration slice", ["promo", "weekday", "store"])
    segment_view = calibration_slices[(calibration_slices.split == "test") & (calibration_slices.group == segment)].copy()
    segment_view = segment_view[["value", "n", "p10_p90_coverage", "p10_coverage", "p50_coverage", "p90_coverage", "pinball_mean", "WAPE"]]
    st.dataframe(segment_view, hide_index=True, width="stretch")

with store_analysis:
    st.subheader("Store-level test forecasts")
    available = sorted(test_q.Store.unique().astype(int).tolist())
    requested_store = st.number_input("Store ID", min_value=1, max_value=1115, value=int(available[0]), step=1)
    if int(requested_store) not in available:
        st.warning("This store is not supported by the fixed 50-store experiment. No forecast is available; the model's reserved unknown-store embedding is for safe inference handling, not a trained forecast for a new store.")
    else:
        series = test_q[test_q.Store == int(requested_store)].sort_values("Date")
        chart = alt.Chart(series).encode(x=alt.X("Date:T", title="Date"))
        band = chart.mark_area(opacity=0.22, color="#4787d1").encode(y=alt.Y("P10:Q", title="Sales value"), y2="P90:Q")
        lines = chart.transform_fold(["target", "P50"], as_=["series", "value"]).mark_line().encode(
            y="value:Q", color=alt.Color("series:N", title="Series", scale=alt.Scale(domain=["target", "P50"], range=["#242a34", "#1677c8"]))
        )
        st.altair_chart(band + lines, width="stretch")
        st.dataframe(series[["Date", "target", "P10", "P50", "P90", "Promo"]], hide_index=True, width="stretch")
        st.download_button("Download this store's test forecast", series.to_csv(index=False), f"store_{int(requested_store)}_forecasts.csv", "text/csv")

with inventory_tab:
    st.subheader("Configurable empirical lead-time scenarios")
    st.warning("These are sales-value planning proxies, not physical stock quantities or observed stockout probabilities. The dataset has no SKU inventory, units, stockout labels, costs, or store-specific lead times.")
    lead_days = st.slider("Assumed supplier lead time (calendar days)", min_value=1, max_value=14, value=3)
    residual_windows = contiguous_lead_time_errors(validation_q, lead_days)
    st.caption(f"Safety stock uses empirical {lead_days}-day sums of validation actual-minus-P50 errors; {len(residual_windows):,} complete store windows. No normality assumption is imposed.")
    store_ids = sorted(test_q.Store.unique().astype(int).tolist())
    inventory_rows = []
    for service in (0.90, 0.95, 0.99):
        safety, coverage = empirical_safety_stock(residual_windows, service)
        per_store = test_q.groupby("Store").P50.mean() * lead_days
        reorder = per_store + safety
        inventory_rows.append({
            "target_service_level": f"{service:.0%}", "assumed_lead_time_days": lead_days,
            "median_lead_time_demand_proxy": float(per_store.median()), "safety_stock_proxy": safety,
            "median_reorder_point_proxy": float(reorder.median()), "p90_reorder_point_proxy": float(reorder.quantile(.9)),
            "validation_buffer_coverage": coverage, "estimated_validation_exceedance": 1.0 - coverage,
            "residual_windows": len(residual_windows),
        })
    inventory_view = pd.DataFrame(inventory_rows)
    st.dataframe(inventory_view, hide_index=True, width="stretch")
    st.caption("Validation buffer coverage estimates how often historical lead-time forecast error was within the chosen buffer. It is not measured stockout risk; actual stockouts require inventory availability and stockout labels.")
    st.download_button("Download inventory scenarios", inventory_view.to_csv(index=False), "inventory_scenarios.csv", "text/csv")

with methodology:
    st.subheader("Methods and limitations")
    st.markdown(
        """
        - Fixed 50-store sample (seed 42), 14-day sequence history, chronological 42-day validation and test windows.
        - Baselines: lag-1, seasonal lag-7, previous-seven-day mean; Linear Regression; CNN, vanilla RNN, LSTM, GRU; and attention-based Transformer.
        - Transformer inputs use historical sales/calendar/promotion/store features, a learned store embedding, and explicit learned positional vectors. Feature scaling/imputation statistics are fit on training rows only.
        - The Transformer predicts ordered nonnegative P10/P50/P90 quantiles using pinball loss. The median is evaluated with WAPE, MAE, and RMSE; calibration uses pinball loss and empirical coverage.
        - `Customers` and same-day target `Sales` are not predictors. Evaluation is rolling one-day-ahead; model parameters are not refit using validation or test targets.
        - Inventory scenarios use empirical validation lead-time forecast-error sums. A service-level target is an assumption; the empirical coverage is a calibration diagnostic. Neither is an observed stockout rate.
        - Rossmann Sales is monetary sales value. Unit-level replenishment requires SKU demand, inventory, stockout, lead-time, and cost data.
        """
    )
    st.json(experiment)

st.divider()
st.caption("Educational forecasting study · see README and PROGRESS for reproduction commands and measured results.")
