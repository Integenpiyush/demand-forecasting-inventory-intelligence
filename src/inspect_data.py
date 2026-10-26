import json
from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data" / "raw"

train = pd.read_csv(
    data_dir / "train.csv",
    parse_dates=["Date"],
    dtype={"StateHoliday": "string"},
)
stores = pd.read_csv(data_dir / "store.csv")

for name, frame in [("train", train), ("store", stores)]:
    print(f"\n{name}: {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    print("Columns:", list(frame.columns))
    print("Data types:\n", frame.dtypes)
    print("Missing values:\n", frame.isna().sum())

print("\nTraining date range:", train["Date"].min().date(), "to", train["Date"].max().date())
print("Distinct training dates:", train["Date"].nunique())
print("Duplicate (Store, Date) keys:", train.duplicated(["Store", "Date"]).sum())
print("Duplicate Store IDs in metadata:", stores["Store"].duplicated().sum())

records_per_store = train.groupby("Store").size()
print("Stores in train:", train["Store"].nunique())
print("Records per store (min / median / max):", records_per_store.min(), records_per_store.median(), records_per_store.max())
print("Stores with fewer rows than the distinct-date count:", (records_per_store < train["Date"].nunique()).sum())

train_store_ids = set(train["Store"].unique())
metadata_store_ids = set(stores["Store"].unique())
print("Train stores missing metadata:", sorted(train_store_ids - metadata_store_ids))
print("Metadata stores absent from train:", sorted(metadata_store_ids - train_store_ids))

promo_columns = ["Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]
print("\nMissing Promo2 fields by Promo2 participation:\n", stores.groupby("Promo2")[promo_columns].agg(lambda column: column.isna().sum()))
competition_missingness = stores.assign(distance_missing=stores["CompetitionDistance"].isna()).groupby("distance_missing").agg(
    stores=("Store", "size"),
    missing_open_month=("CompetitionOpenSinceMonth", lambda column: column.isna().sum()),
    missing_open_year=("CompetitionOpenSinceYear", lambda column: column.isna().sum()),
)
print("\nCompetition metadata missingness by distance availability:\n", competition_missingness)
print("\nOpen status vs zero sales:\n", pd.crosstab(train["Open"], train["Sales"].eq(0), rownames=["Open"], colnames=["Sales == 0"]))
open_zero_sales_customers = train.loc[(train["Open"] == 1) & (train["Sales"] == 0), "Customers"]
print("Open + zero-sales rows with zero customers:", open_zero_sales_customers.eq(0).sum(), "of", len(open_zero_sales_customers))
print("Open + zero-sales rows with customers > 0:\n", train.loc[(train["Open"] == 1) & (train["Sales"] == 0) & (train["Customers"] > 0), ["Store", "Date", "Customers", "Promo", "StateHoliday", "SchoolHoliday"]].to_string(index=False))

store_coverage = train.groupby("Store")["Date"].agg(first_date="min", last_date="max", observed_dates="nunique")
store_coverage["span_days"] = (store_coverage["last_date"] - store_coverage["first_date"]).dt.days + 1
store_coverage["missing_dates_within_span"] = store_coverage["span_days"] - store_coverage["observed_dates"]
global_start = train["Date"].min()
global_end = train["Date"].max()
incomplete_stores = store_coverage.loc[store_coverage["observed_dates"] < train["Date"].nunique()]
print("\nIncomplete stores whose observed dates span the full dataset range:", ((incomplete_stores["first_date"] == global_start) & (incomplete_stores["last_date"] == global_end)).sum())
print("Incomplete stores with missing dates inside their own first-to-last span:", (incomplete_stores["missing_dates_within_span"] > 0).sum())

manifest_path = project_root / "config" / "store_sample.json"
selected_ids = json.loads(manifest_path.read_text(encoding="utf-8"))["store_ids"]
sample_coverage = store_coverage.loc[selected_ids]
print("Saved 50-store sample: records per store (min / median / max):", sample_coverage["observed_dates"].min(), sample_coverage["observed_dates"].median(), sample_coverage["observed_dates"].max())
print("Sample stores with shorter histories:", (sample_coverage["observed_dates"] < train["Date"].nunique()).sum())
print("Incomplete sample-store histories:\n", sample_coverage.loc[sample_coverage["observed_dates"] < train["Date"].nunique()].to_string())
for store_id in sample_coverage.index[sample_coverage["observed_dates"] < train["Date"].nunique()]:
    observed_dates = train.loc[train["Store"] == store_id, "Date"]
    missing_dates = pd.Series(pd.date_range(global_start, global_end).difference(observed_dates))
    gap_groups = missing_dates.diff().dt.days.ne(1).cumsum()
    largest_gap = missing_dates.groupby(gap_groups).agg(["min", "max", "size"]).sort_values("size", ascending=False).iloc[0]
    print(f"Store {store_id}: longest missing run {largest_gap['size']} dates, {largest_gap['min'].date()} to {largest_gap['max'].date()}")
sample_open_zero_sales = train.loc[train["Store"].isin(selected_ids) & (train["Open"] == 1) & (train["Sales"] == 0)]
print("Open + zero-sales rows in 50-store sample:", len(sample_open_zero_sales))
print("Sample open + zero-sales row details:\n", sample_open_zero_sales[["Store", "Date", "Customers", "Promo", "StateHoliday", "SchoolHoliday"]].to_string(index=False))

sample_train = train.loc[train["Store"].isin(selected_ids)]
print("\nSales summary in the 50-store sample:\n", sample_train["Sales"].describe(percentiles=[0.5, 0.9, 0.99]))
print("\nCustomer-count summary in the 50-store sample:\n", sample_train["Customers"].describe(percentiles=[0.5, 0.9, 0.99]))
print("\nSales summary by Open status in the 50-store sample:\n", sample_train.groupby("Open")["Sales"].describe())
