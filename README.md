# Deep Demand Forecasting & Inventory Intelligence

This repository implements a CPU-friendly, reproducible Rossmann Store Sales forecasting study. The initial experiment is limited to a persisted random sample of 50 stores. Metrics from this sample must not be presented as results for all 1,115 stores.

## Current project status

The project contains a data audit, exploratory plots, date-exact lag features, a rolling one-day-ahead forecasting experiment, classical baselines, Linear Regression, four PyTorch sequence architectures (1D CNN, vanilla RNN, LSTM, GRU), evaluation outputs, and an illustrative inventory calculation. See [`PROGRESS.md`](PROGRESS.md) for completed steps and measured findings. Results are created locally when the experiment finishes; no score is promised in advance.

The MIT license in this repository covers the project code only. Kaggle competition data retains its own terms; raw source files, row-level forecasts, per-store outputs, and EDA plots are excluded from Git. The dashboard uses aggregate experiment summaries.

## Setup

Use Python 3.12 and the project-local virtual environment. On macOS, from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The venv isolates project libraries from the system Python. Raw Kaggle data is intentionally excluded from version control.

## Dataset

Obtain the competition files from [Kaggle Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data) after accepting the [competition rules](https://www.kaggle.com/competitions/rossmann-store-sales/rules). Keep the raw files and row-level outputs local; the Git ignore rules exclude them. Place the extracted files exactly here:

```text
data/raw/train.csv
data/raw/store.csv
```

`train.csv` has 1,017,209 daily store observations and nine columns: `Store` (store key), `DayOfWeek` (1–7), `Date`, `Sales` (target), `Customers` (not used as a predictor because future customers are unknown), `Open`, `Promo`, `StateHoliday`, and `SchoolHoliday`.

`store.csv` has 1,115 rows and ten store-level columns: `Store` (join key), `StoreType`, `Assortment`, `CompetitionDistance`, `CompetitionOpenSinceMonth`, `CompetitionOpenSinceYear`, `Promo2`, `Promo2SinceWeek`, `Promo2SinceYear`, and `PromoInterval`. Competition-opening fields are not currently used; `CompetitionDistance` is median-imputed inside each training-fold preprocessing pipeline. Missing Promo2 calendar fields are retained as source metadata and not used as model inputs.

To inspect the dataset and its missingness:

```sh
python src/inspect_data.py
```

The sample ID list is in `config/store_sample.json`; `src/select_stores.py` creates it once with seed 42 and reuses it on later calls. Do not delete that file if you need to reproduce the same sample.

## Run the project

```sh
python src/prepare_data.py
python src/inspect_data.py
python src/eda.py
python src/forecast.py
python src/inventory.py
python src/error_analysis.py
python src/publish_dashboard_data.py
```

## Interactive dashboard

The dashboard browses the checked-in experiment outputs. It does not upload user data or retrain models. To run it locally in an isolated environment, use:

```sh
python3.12 -m venv .venv-dashboard
source .venv-dashboard/bin/activate
python -m pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

The dashboard environment uses pandas 2.2 because the pinned Streamlit release requires pandas below 3; keep it separate from the training `.venv`, which uses pandas 3.1. To publish on [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), choose the GitHub repository, branch, and `dashboard/app.py` as the entry point. The project’s root `.streamlit/config.toml` supplies the theme. Keep `data/raw/` private and excluded from Git; the dashboard uses saved result artifacts and does not need the Kaggle source files.

Live dashboard: [Deep Demand Forecasting & Inventory Intelligence](https://demand-forecasting-inventory-intelligence-bk8hrfzrqmicjhie9njz.streamlit.app/). Public results cover the 50-store experiment only; they do not establish performance across all 1,115 stores.

`forecast.py` uses the 50 saved stores, exact date lookups for prior-day/prior-week/14-day history, and chronological blocks of 42 validation days and 42 test days. The prediction task is rolling one-day-ahead: prior observed sales can be used for later test dates, but test targets do not fit model parameters. Date gaps remain missing; any target without the required lag or complete 14-day history is omitted from the corresponding model. The tabular model and deep models can therefore have different row counts, recorded in `reports/results/experiment.json`.

Linear Regression fits preprocessing (median imputation, standardization, categorical one-hot encoding, including store ID as a category) on the training block only. Its predictors are known calendar/promotion/store covariates and sales lags; `Customers` and current-day `Sales` are excluded. The deep models consume the previous 14 calendar days of sales and target-date/store covariates, normalize target/history using training data, select the epoch with lowest validation MSE, and use early stopping. Training is CPU-only and seeded. Architecture and training settings are teaching baselines, not a hyperparameter search.

## Outputs

- `reports/figures/`: descriptive plots for sample coverage, weekday sales, promotion association, sales distribution, and store-level variability.
- `reports/results/model_metrics.csv`: RMSE, MAE, and WAPE on the held-out test block.
- `reports/results/test_predictions.csv`: actual and predicted sales for evaluated test rows.
- `reports/results/validation_predictions.csv` and `validation_metrics.csv`: local validation forecasts and scores, used to select the inventory demonstration model without consulting test scores.
- `reports/results/test_error_by_weekday.csv`, `test_error_by_promo.csv`, and `test_error_by_store.csv`: local held-out error slices for the validation-selected model.
- `reports/results/experiment.json`: fixed sample, split dates, row counts, and protocol metadata.
- `models/`: fitted Linear Regression pipeline, neural model state dictionaries, and the deep-model preprocessing/scaling metadata.
- `reports/results/inventory_reorder_points.csv`: per-store illustrative lead-time demand, safety-stock, and reorder-point proxies.
- `reports/results/inventory_assumptions.json`: inventory formula assumptions.
- `reports/results/dashboard_*.csv` and `dashboard_experiment.json`: aggregate-only outputs used by the public dashboard; row-level actuals, predictions, and per-store inventory outputs are gitignored.
- `docs/deployment.md`: GitHub authentication and Streamlit Community Cloud deployment steps.

WAPE is `sum(abs(actual - forecast)) / sum(abs(actual))`. RMSE emphasizes large misses; MAE is average absolute error. Lower is better for all three. Compare models only with their row counts and evaluation scope in view.

## Inventory interpretation

The inventory module uses the model with the lowest validation WAPE and that model's pooled validation residual standard deviation for uncertainty. It demonstrates `reorder point = lead-time demand + z × residual SD × sqrt(lead time)` for 90%, 95%, and 99% service-level z values and an assumed three-day lead time. Rossmann `Sales` is a monetary sales value, not item-level unit demand. These outputs are therefore illustrative sales-value proxies, not executable replenishment recommendations. The calculation assumes constant lead time, approximately normal independent daily errors, and does not model stockouts, costs, pack sizes, or item assortment.

## Validation and limitations

Run the project checks with:

```sh
pytest
```

The sample is a random store sample, not stratified by store type (the saved sample has 30 type-A, 1 type-B, 6 type-C, and 13 type-D stores). Eight sampled stores have a common missing interval from 2014-07-01 to 2014-12-31. The model preserves that absence instead of converting it to zero demand. Test results describe this sample and this rolling daily-update protocol only. No causal effect is inferred from the promotion plot. No claim is made that the 50-store model generalizes to the full set of stores.

## Project structure

```text
config/       saved sample configuration
data/raw/     local Kaggle files (gitignored)
data/processed/  generated intermediates (gitignored)
models/       fitted estimator and neural weights
reports/      plots, metrics, predictions, and inventory artifacts
dashboard/    aggregate results viewer and separate runtime dependencies
.streamlit/   dashboard theme configuration
src/          audit, sample selection, EDA, forecasting, inventory code
tests/        lightweight correctness checks
```
