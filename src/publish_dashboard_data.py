"""Build small aggregate dashboard artifacts; never export row-level competition data."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"


def validate_quantile_dashboard_data(frame: pd.DataFrame) -> None:
    """Fail early if dashboard forecast artifact schema/order/scope is invalid."""
    required = {"Store", "Date", "target", "P10", "P50", "P90", "split", "Promo"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dashboard quantile data missing columns: {sorted(missing)}")
    if not {"validation", "test"}.issubset(set(frame.split.unique())):
        raise ValueError("Dashboard quantile data must include validation and test rows")
    if frame[["P10", "P50", "P90"]].isna().any().any():
        raise ValueError("Dashboard quantile forecasts contain missing values")
    if (frame.P10 < 0).any() or (frame.P10 > frame.P50).any() or (frame.P50 > frame.P90).any():
        raise ValueError("Dashboard quantile forecasts must be nonnegative and ordered")
    if frame.Store.nunique() > 50:
        raise ValueError("Dashboard forecast artifact exceeds the 50-store experiment scope")


def main() -> None:
    model_metrics = pd.read_csv(RESULTS / "model_metrics.csv")
    validation_metrics = pd.read_csv(RESULTS / "validation_metrics.csv")
    store_inventory = pd.read_csv(RESULTS / "inventory_reorder_points.csv")
    selected_model = str(validation_metrics.iloc[0].model)
    model_metrics.to_csv(RESULTS / "dashboard_model_metrics.csv", index=False)
    # Quantile records power the requested store-level plots and configurable,
    # empirical lead-time scenarios. These cover only the fixed 50-store sample.
    quantile_path = RESULTS / "transformer_quantile_predictions.csv"
    if quantile_path.is_file():
        quantile_predictions = pd.read_csv(quantile_path)
        quantile_predictions = quantile_predictions[quantile_predictions.split.isin(["validation", "test"])].copy()
        validate_quantile_dashboard_data(quantile_predictions)
        quantile_predictions[["Store", "Date", "target", "Promo", "split", "P10", "P50", "P90"]].to_csv(
            RESULTS / "dashboard_transformer_quantile_predictions.csv", index=False
        )
        pd.read_csv(RESULTS / "transformer_quantile_metrics.csv").to_csv(
            RESULTS / "dashboard_transformer_quantile_metrics.csv", index=False
        )
        slices_path = RESULTS / "transformer_calibration_slices.csv"
        if slices_path.is_file():
            pd.read_csv(slices_path).to_csv(RESULTS / "dashboard_transformer_calibration_slices.csv", index=False)
    summary = []
    for service in (90, 95, 99):
        stock = store_inventory[f"safety_stock_proxy_{service}pct"]
        reorder = store_inventory[f"reorder_point_proxy_{service}pct"]
        summary.append({
            "service_level_pct": service,
            "stores_aggregated": int(store_inventory.Store.nunique()),
            "average_safety_stock_proxy": float(stock.mean()),
            "median_reorder_point_proxy": float(reorder.median()),
            "p90_reorder_point_proxy": float(reorder.quantile(0.90)),
        })
    pd.DataFrame(summary).to_csv(RESULTS / "dashboard_inventory_summary.csv", index=False)
    manifest = json.loads((RESULTS / "experiment.json").read_text())
    public_meta = {key: value for key, value in manifest.items() if key != "store_ids"}
    public_meta["validation_selected_model"] = selected_model
    (RESULTS / "dashboard_experiment.json").write_text(json.dumps(public_meta, indent=2) + "\n")
    print("Wrote dashboard metrics, selected-store quantile forecasts, and inventory summaries.")
    print(f"Selected model: {selected_model}; aggregated stores: {store_inventory.Store.nunique()}")


if __name__ == "__main__":
    main()
