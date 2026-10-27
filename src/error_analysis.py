"""Summarize held-out errors for the validation-selected model."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"


def main() -> None:
    predictions = pd.read_csv(RESULTS / "test_predictions.csv", parse_dates=["Date"])
    quantile_path = RESULTS / "transformer_quantile_predictions.csv"
    if quantile_path.is_file():
        quantiles = pd.read_csv(quantile_path, parse_dates=["Date"])
        transformer_test = quantiles.loc[quantiles.split == "test", ["Store", "Date", "P50"]].rename(columns={"P50": "Transformer"})
        predictions = predictions.merge(transformer_test, on=["Store", "Date"], how="left", validate="one_to_one")
    frame = pd.read_csv(ROOT / "data" / "raw" / "train.csv", parse_dates=["Date"], dtype={"StateHoliday": "string"})
    selected = str(pd.read_csv(RESULTS / "validation_metrics.csv").sort_values("WAPE").iloc[0]["model"])
    ids = json.loads((ROOT / "config" / "store_sample.json").read_text())["store_ids"]
    test = predictions.merge(frame[["Store", "Date", "DayOfWeek", "Promo"]], on=["Store", "Date"], how="left", validate="one_to_one")
    test["error"] = test[selected] - test.target
    test["absolute_error"] = test.error.abs()
    test["absolute_percentage_error"] = test.absolute_error / test.target.replace(0, np.nan)
    def summarize(group):
        return pd.Series({"n": len(group), "MAE": group.absolute_error.mean(), "RMSE": np.sqrt(np.mean(group.error ** 2)), "WAPE": group.absolute_error.sum() / max(group.target.abs().sum(), 1e-12), "mean_signed_error": group.error.mean()})
    test.groupby("DayOfWeek").apply(summarize, include_groups=False).to_csv(RESULTS / "test_error_by_weekday.csv")
    test.groupby("Promo").apply(summarize, include_groups=False).to_csv(RESULTS / "test_error_by_promo.csv")
    test.groupby("Store").apply(summarize, include_groups=False).to_csv(RESULTS / "test_error_by_store.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS / "model_metrics.csv").sort_values("WAPE")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(metrics.model, metrics.WAPE, color="steelblue")
    ax.set(title="Held-out WAPE by model (50-store sample)", ylabel="WAPE (lower is better)", xlabel="Model")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(FIGURES / "test_model_wape.png", dpi=150); plt.close(fig)
    print(f"Selected for error slices using validation WAPE: {selected}")
    print(f"Test rows and stores: {len(test):,}, {test.Store.nunique()} of {len(ids)} selected stores")
    print("Saved test error breakdowns by weekday, promotion, and store.")


if __name__ == "__main__":
    main()
