"""Validate, select the fixed sample, and save a joined modeling table."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SAMPLE_PATH = ROOT / "config" / "store_sample.json"


def prepare() -> pd.DataFrame:
    ids = json.loads(SAMPLE_PATH.read_text())["store_ids"]
    sales_path, store_path = RAW / "train.csv", RAW / "store.csv"
    if not sales_path.is_file() or not store_path.is_file():
        raise FileNotFoundError("Expected data/raw/train.csv and data/raw/store.csv; download them from the Kaggle competition first.")
    sales = pd.read_csv(sales_path, parse_dates=["Date"], dtype={"StateHoliday": "string"})
    stores = pd.read_csv(store_path)
    required_sales = {"Store", "Date", "Sales", "Open", "Promo", "StateHoliday", "SchoolHoliday"}
    required_store = {"Store", "StoreType", "Assortment", "CompetitionDistance", "Promo2"}
    if missing := required_sales.difference(sales.columns):
        raise ValueError(f"train.csv is missing expected columns: {sorted(missing)}")
    if missing := required_store.difference(stores.columns):
        raise ValueError(f"store.csv is missing expected columns: {sorted(missing)}")
    if sales.duplicated(["Store", "Date"]).any() or stores.Store.duplicated().any():
        raise ValueError("Expected unique (Store, Date) training keys and unique store metadata keys")
    if not set(ids).issubset(set(sales.Store)) or not set(ids).issubset(set(stores.Store)):
        raise ValueError("The saved 50-store sample IDs are not all present in both source files")
    sample = sales.loc[sales.Store.isin(ids)].merge(stores, on="Store", how="left", validate="many_to_one")
    if sample.Store.nunique() != len(ids) or len(sample) != int(sales.Store.isin(ids).sum()):
        raise AssertionError("Sample selection or metadata join changed the expected observation count")
    # This is a type cleanup only: recorded closed days and zero sales remain valid observations.
    sample["StateHoliday"] = sample.StateHoliday.fillna("0").astype("string")
    sample = sample.sort_values(["Store", "Date"]).reset_index(drop=True)
    return sample


def main() -> None:
    sample = prepare()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    output = PROCESSED / "sample_sales_with_store.csv"
    sample.to_csv(output, index=False)
    manifest = {
        "source_files": ["data/raw/train.csv", "data/raw/store.csv"],
        "output_file": "data/processed/sample_sales_with_store.csv",
        "rows": len(sample),
        "stores": int(sample.Store.nunique()),
        "date_min": str(sample.Date.min().date()),
        "date_max": str(sample.Date.max().date()),
        "missing_values": {key: int(value) for key, value in sample.isna().sum().items() if value},
        "notes": "Joined by Store after key validation; closed rows and zero Sales retained; absent dates are not generated; missing metadata values are preserved for train-only model imputation.",
    }
    (PROCESSED / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(sample):,} rows for {sample.Store.nunique()} stores to {output.relative_to(ROOT)}")
    print(f"Date range: {manifest['date_min']} through {manifest['date_max']}")
    print(f"Missing values retained: {manifest['missing_values']}")


if __name__ == "__main__":
    main()
