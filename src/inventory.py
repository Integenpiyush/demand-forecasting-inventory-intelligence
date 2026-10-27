"""Create illustrative reorder-point outputs from the selected sample's forecasts."""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"


def contiguous_lead_time_errors(predictions: pd.DataFrame, lead_days: int, actual_col: str = "target",
                                forecast_col: str = "P50") -> np.ndarray:
    """Aggregate actual-minus-forecast error over complete consecutive store-day windows."""
    if lead_days < 1:
        raise ValueError("lead_days must be at least 1")
    data = predictions[["Store", "Date", actual_col, forecast_col]].copy()
    data["Date"] = pd.to_datetime(data.Date)
    data["error"] = data[actual_col] - data[forecast_col]
    totals = []
    for _, group in data.sort_values("Date").groupby("Store"):
        group = group.sort_values("Date").reset_index(drop=True)
        consecutive = group.Date.diff().dt.days.eq(1)
        block = consecutive.ne(True).cumsum()
        for _, run in group.groupby(block):
            if len(run) >= lead_days:
                totals.extend(run.error.rolling(lead_days).sum().dropna().tolist())
    return np.asarray(totals, dtype=float)


def empirical_safety_stock(lead_time_errors: np.ndarray, service_level: float) -> tuple[float, float]:
    """Return nonnegative empirical residual quantile and observed exceedance coverage."""
    if not 0 < service_level < 1:
        raise ValueError("service_level must be between 0 and 1")
    values = np.asarray(lead_time_errors, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No complete lead-time residual windows are available")
    stock = max(0.0, float(np.quantile(values, service_level)))
    coverage = float(np.mean(values <= stock))
    return stock, coverage


def main(lead_days: int = 3) -> None:
    validation = pd.read_csv(RESULTS / "validation_predictions.csv", parse_dates=["Date"])
    predictions = pd.read_csv(RESULTS / "test_predictions.csv", parse_dates=["Date"])
    if lead_days < 1:
        raise ValueError("lead_days must be at least 1")
    quantile_path = RESULTS / "transformer_quantile_predictions.csv"
    quantile_metrics_path = RESULTS / "transformer_quantile_metrics.csv"
    if quantile_path.exists() and quantile_metrics_path.exists():
        quantiles = pd.read_csv(quantile_path, parse_dates=["Date"])
        validation = quantiles[quantiles.split == "validation"].copy()
        predictions = quantiles[quantiles.split == "test"].copy()
        selected_model = "Transformer P50"
        lead_errors = contiguous_lead_time_errors(validation, lead_days)
        stock_by_service = {p: empirical_safety_stock(lead_errors, p) for p in (0.90, 0.95, 0.99)}
        uncertainty_method = "Empirical quantiles of contiguous validation actual-minus-P50 errors summed over the configured lead time; no normality assumption."
    else:
        validation_metrics = pd.read_csv(RESULTS / "validation_metrics.csv")
        selected_model = str(validation_metrics.iloc[0]["model"])
        if selected_model not in predictions:
            selected_model = "LinearRegression"
        residual_sd = float((validation.target - validation[selected_model]).std(ddof=1))
        service_z = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}
        stock_by_service = {p: (z * residual_sd * np.sqrt(lead_days), float("nan")) for p, z in service_z.items()}
        uncertainty_method = "Legacy normal-error approximation using pooled validation residual SD; used only when Transformer quantile artifacts are absent."
    by_store = predictions.groupby("Store").agg(
        average_daily_forecast=("P50" if "P50" in predictions else selected_model, "mean"),
        observed_test_days=("P50" if "P50" in predictions else selected_model, "size"),
        average_daily_actual=("target", "mean"),
    ).reset_index()
    by_store["lead_time_demand_proxy"] = by_store.average_daily_forecast * lead_days
    for service, (stock_value, coverage) in stock_by_service.items():
        col = f"safety_stock_proxy_{int(service * 100)}pct"
        by_store[col] = stock_value
        by_store[f"reorder_point_proxy_{int(service * 100)}pct"] = by_store.lead_time_demand_proxy + by_store[col]
        by_store[f"empirical_coverage_{int(service * 100)}pct"] = coverage
    by_store.to_csv(RESULTS / "inventory_reorder_points.csv", index=False)
    info = {
        "forecast_model": selected_model,
        "lead_time_days": lead_days,
        "uncertainty_method": uncertainty_method,
        "lead_time_residual_windows": int(len(lead_errors)) if "lead_errors" in locals() else None,
        "interpretation": "Illustrative sales-value planning proxy, not physical inventory units. Empirical validation residual exceedance estimates forecast-buffer coverage only; there are no actual stockout labels, inventory balances, costs, pack sizes, or store-specific lead-time data.",
    }
    (RESULTS / "inventory_assumptions.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Model used: {selected_model}; stores with test predictions: {len(by_store)}")
    print(f"Lead time: {lead_days} days; uncertainty method: {uncertainty_method}")
    print(f"Saved illustrative reorder points to {RESULTS.relative_to(ROOT)}/inventory_reorder_points.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-days", type=int, default=3, help="assumed calendar-day supplier lead time (default: 3)")
    main(parser.parse_args().lead_days)
