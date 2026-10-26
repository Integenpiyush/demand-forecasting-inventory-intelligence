import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
SAMPLE_PATH = PROJECT_ROOT / "config" / "store_sample.json"

processed = PROJECT_ROOT / "data" / "processed" / "sample_sales_with_store.csv"
if processed.is_file():
    sample = pd.read_csv(processed, parse_dates=["Date"], dtype={"StateHoliday": "string"})
else:
    with SAMPLE_PATH.open(encoding="utf-8") as file:
        selected_store_ids = json.load(file)["store_ids"]
    sales = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["Date"], dtype={"StateHoliday": "string"})
    stores = pd.read_csv(DATA_DIR / "store.csv")
    sample = sales.loc[sales["Store"].isin(selected_store_ids)].merge(
        stores, on="Store", how="left", validate="many_to_one"
    )
open_days = sample.loc[sample["Open"] == 1].copy()
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

# The store gap is visible here as fewer observed stores on some dates.
daily = sample.groupby("Date").agg(
    mean_sales=("Sales", "mean"),
    observed_stores=("Store", "nunique"),
)
fig, (sales_ax, count_ax) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
sales_ax.plot(daily.index, daily["mean_sales"], color="navy", linewidth=1)
sales_ax.set(title="Daily mean sales across the saved 50-store sample", ylabel="Mean sales")
count_ax.plot(daily.index, daily["observed_stores"], color="darkorange", linewidth=1)
count_ax.set(xlabel="Date", ylabel="Stores observed")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "daily_sales_and_coverage.png", dpi=150)
plt.close(fig)

# Restricting to open days makes the weekday means easier to interpret.
weekday = open_days.groupby("DayOfWeek")["Sales"].mean().reindex(range(1, 8))
weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=weekday_names, y=weekday.to_numpy(), color="steelblue", ax=ax)
ax.set(title="Mean sales by weekday (open days only)", xlabel="Weekday", ylabel="Mean sales")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "weekday_sales.png", dpi=150)
plt.close(fig)

# Promotions are observational in this dataset; this plot does not establish causation.
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=open_days, x="Promo", y="Sales", showfliers=False, ax=ax)
ax.set(title="Sales by promotion flag (open days only)", xlabel="Promo", ylabel="Sales")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "promotion_sales.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(open_days["Sales"], bins=60, ax=ax)
ax.set(title="Sales distribution (open days only)", xlabel="Sales", ylabel="Number of observations")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "sales_distribution.png", dpi=150)
plt.close(fig)

store_means = open_days.groupby("Store")["Sales"].mean()
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(store_means, bins=20, ax=ax)
ax.set(title="Variation in mean open-day sales across sampled stores", xlabel="Store mean sales", ylabel="Number of stores")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "store_mean_sales.png", dpi=150)
plt.close(fig)

print(f"Rows plotted: {len(sample):,} across {sample['Store'].nunique()} saved sample stores")
print(f"Open-day rows: {len(open_days):,}")
print("Saved figures:")
for figure_path in sorted(FIGURE_DIR.glob("*.png")):
    print(f"- {figure_path.relative_to(PROJECT_ROOT)}")
