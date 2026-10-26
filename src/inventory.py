"""Create illustrative reorder-point outputs from the selected sample's forecasts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"


def main() -> None:
    validation = pd.read_csv(RESULTS / "validation_predictions.csv", parse_dates=["Date"])
    predictions = pd.read_csv(RESULTS / "test_predictions.csv", parse_dates=["Date"])
    validation_metrics = pd.read_csv(RESULTS / "validation_metrics.csv")
    selected_model = str(validation_metrics.iloc[0]["model"])
    if selected_model not in predictions:
        selected_model = "LinearRegression"
    # Pooled validation residual standard deviation is the demand-variability proxy.
    residual_sd = float((validation.target - validation[selected_model]).std(ddof=1))
    service_z = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}
    lead_days = 3
    by_store = predictions.groupby("Store").agg(
        average_daily_forecast=(selected_model, "mean"),
        observed_test_days=(selected_model, "size"),
        average_daily_actual=("target", "mean"),
    ).reset_index()
    by_store["lead_time_demand_proxy"] = by_store.average_daily_forecast * lead_days
    for service, z in service_z.items():
        col = f"safety_stock_proxy_{int(service * 100)}pct"
        by_store[col] = z * residual_sd * np.sqrt(lead_days)
        by_store[f"reorder_point_proxy_{int(service * 100)}pct"] = by_store.lead_time_demand_proxy + by_store[col]
    by_store.to_csv(RESULTS / "inventory_reorder_points.csv", index=False)
    info = {
        "forecast_model": selected_model,
        "lead_time_days": lead_days,
        "pooled_validation_residual_sd": residual_sd,
        "service_level_z_values": {str(k): v for k, v in service_z.items()},
        "interpretation": "Illustrative sales-value demand proxy, not physical inventory units. Uses pooled validation residual variation, a constant 3-day lead time, normal-demand z values, and independent daily errors; no stockouts, costs, pack sizes, or store-specific lead-time data are modeled.",
    }
    (RESULTS / "inventory_assumptions.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Model used: {selected_model}; stores with test predictions: {len(by_store)}")
    print(f"Validation residual SD proxy: {residual_sd:.2f}; lead time: {lead_days} days")
    print(f"Saved illustrative reorder points to {RESULTS.relative_to(ROOT)}/inventory_reorder_points.csv")


if __name__ == "__main__":
    main()
