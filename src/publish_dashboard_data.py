"""Build small aggregate dashboard artifacts; never export row-level competition data."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"


def main() -> None:
    model_metrics = pd.read_csv(RESULTS / "model_metrics.csv")
    validation_metrics = pd.read_csv(RESULTS / "validation_metrics.csv")
    store_inventory = pd.read_csv(RESULTS / "inventory_reorder_points.csv")
    selected_model = str(validation_metrics.iloc[0].model)
    model_metrics.to_csv(RESULTS / "dashboard_model_metrics.csv", index=False)
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
    print("Wrote aggregate dashboard artifacts; per-date and per-store records were not exported by this script.")
    print(f"Selected model: {selected_model}; aggregated stores: {store_inventory.Store.nunique()}")


if __name__ == "__main__":
    main()
