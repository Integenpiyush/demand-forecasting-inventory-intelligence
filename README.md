# Deep Demand Forecasting & Inventory Intelligence

This repository implements a CPU-friendly, reproducible Rossmann Store Sales forecasting study. The initial experiment is limited to a persisted random sample of 50 stores. Metrics from this sample must not be presented as results for all 1,115 stores.

## Current project status

The project contains a data audit, exploratory plots, date-exact lag features, a rolling one-day-ahead forecasting experiment, classical baselines, Linear Regression, CNN/RNN/LSTM/GRU models, and a PyTorch Transformer with learned store embeddings and P10/P50/P90 predictive quantiles. It includes chronological evaluation, empirical quantile calibration, a Streamlit dashboard, and sales-value inventory planning proxies. See [`PROGRESS.md`](PROGRESS.md) for measured findings; do not generalize the 50-store sample to all stores.

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
python src/probabilistic.py
python src/inventory.py --lead-days 3
python src/error_analysis.py
python src/publish_dashboard_data.py
```

## Interactive dashboard

The dashboard browses checked-in experiment outputs, including sampled-store Transformer quantiles and inventory scenarios. It does not upload user data or retrain models. To run it locally in an isolated environment, use:

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
- `reports/results/model_metrics.csv`: RMSE, MAE, and WAPE on the held-out test block, including Transformer median forecasts after `probabilistic.py` runs.
- `reports/results/test_predictions.csv`: actual and predicted sales for evaluated test rows.
- `reports/results/validation_predictions.csv` and `validation_metrics.csv`: local validation forecasts and scores, used to select the inventory demonstration model without consulting test scores.
- `reports/results/test_error_by_weekday.csv`, `test_error_by_promo.csv`, and `test_error_by_store.csv`: local held-out error slices for the validation-selected model.
- `reports/results/experiment.json`: fixed sample, split dates, row counts, and protocol metadata.
- `models/`: fitted Linear Regression pipeline, neural model state dictionaries, and the deep-model preprocessing/scaling metadata.
- `reports/results/inventory_reorder_points.csv`: per-store illustrative lead-time demand, safety-stock, and reorder-point proxies.
- `reports/results/inventory_assumptions.json`: inventory formula assumptions.
- `reports/results/transformer_quantile_predictions.csv`: validation and test P10/P50/P90 for each supported sampled-store forecast date.
- `reports/results/transformer_quantile_metrics.csv`: validation/test median WAPE, MAE, RMSE, quantile pinball losses, and empirical coverage.
- `reports/results/transformer_calibration_slices.csv`: quantile calibration and median-error slices by store, promotion, and weekday where sample size permits.
- `models/transformer_quantile.pt`: trained Transformer weights, store-to-embedding mapping, and training-only normalization metadata.
- `reports/results/dashboard_*.csv` and `dashboard_experiment.json`: public dashboard outputs. The Transformer dashboard forecast artifact contains only the fixed sample's validation/test dates, Store IDs, actual Sales, promotions, and P10/P50/P90 predictions; raw Kaggle source files remain local and gitignored.
- `docs/deployment.md`: GitHub authentication and Streamlit Community Cloud deployment steps.

WAPE is `sum(abs(actual - forecast)) / sum(abs(actual))`. RMSE emphasizes large misses; MAE is average absolute error. Lower is better for all three. Compare models only with their row counts and evaluation scope in view.

## Transformer quantile model

`src/probabilistic.py` implements a two-layer, four-head PyTorch Transformer encoder over the previous 14 exact calendar days. Each token contains historical sales, weekday, month, promotion, open status, school holiday, Promo2 flag, and competition distance. A learned positional vector preserves order. Target-date known features and a learned Store embedding are joined to the encoded history. Store index 0 is reserved for unseen IDs; a new store maps safely to that index but has no learned store-specific signal and is not considered supported by this experiment.

`Customers` and same-day actual `Sales` are excluded. Imputation and scaling statistics use training rows only. The model minimizes mean pinball loss for q=0.1, 0.5, and 0.9. It predicts an unconstrained standardized median with positive gaps on either side, then inverse-scales, clips negative Sales to zero, and enforces order. This avoids incorrectly forcing standardized outputs to be positive. Additive quantile offsets are calibrated from validation residual quantiles only; test data is not used for calibration. The median is evaluated with WAPE, MAE, and RMSE; quantiles are evaluated with pinball loss and empirical coverage. P10/P50/P90 are predictive quantiles, not confidence intervals. Since Sales cannot be negative and a nontrivial fraction of targets are zero, empirical P10 coverage cannot reach 10% in every segment; this is reported rather than hidden.

Run the existing deterministic CNN/RNN/LSTM/GRU/baseline experiment first, then run `python src/probabilistic.py --history-days 14`. Change the argument to evaluate another history length; 14 days is the default and initial comparable configuration. Transformer rows are added to the existing validation and test model metric tables, retaining prior model rows. Validation WAPE selects the model; the 42-day final test block is used only for final reporting. Training uses CPU, seed 42, AdamW, gradient clipping, up to 40 epochs, and validation pinball early stopping. Training time and epoch count are recorded in `experiment.json`.

Latest fixed-seed run: the Transformer trained for 34 epochs in 133.8 seconds on this Mac CPU. It was selected by validation WAPE (7.87%). Test median results were WAPE **8.57%**, MAE **535.30**, and RMSE **893.60** over 2,100 targets. The GRU comparison was 8.85% test WAPE; this is a small observed advantage on this one sampled split, not a claim of general superiority. Transformer test P10–P90 coverage was 81.6% against a nominal central 80%; P90 coverage was 90.1%. P10 coverage was 22.6%, partly constrained by nonnegative predictions and zero-sales observations. See [`transformer_quantile_metrics.csv`](reports/results/transformer_quantile_metrics.csv) and [`model_metrics.csv`](reports/results/model_metrics.csv) for complete metrics.

## Inventory interpretation

When Transformer quantile artifacts exist, `src/inventory.py` and the dashboard estimate safety stock from empirical quantiles of contiguous validation actual-minus-P50 errors summed over the selected lead time. This captures the observed multi-day residual distribution without imposing a normal distribution. The default lead time is three calendar days; the dashboard allows 1–14 days and the script accepts `--lead-days`. For 90%, 95%, and 99% service-level assumptions, the safety buffer is the corresponding empirical validation-error quantile; reorder point proxy is expected lead-time Sales forecast plus that buffer. The dashboard also reports historical validation buffer coverage and exceedance frequency as diagnostics. They are **not observed stockout probabilities** because no stockout labels or inventory-on-hand data exist. For the 3-day, 95% scenario, the median reorder-point proxy was approximately 19,450 Sales-value units across 50 stores. Rossmann `Sales` is monetary sales value, not item-level unit demand, so these are planning proxies, not physical reorder quantities.

## Testing and compute

Run the lightweight fixture end-to-end smoke test and correctness checks with:

```sh
.venv/bin/pytest -q
```

This checks date splitting/lags, Transformer forward and optimizer steps, unknown-store embedding behavior, quantile ordering, pinball loss, and lead-time residual calculations. Full experiment commands are listed under “Run the project”; `forecast.py` trains the existing models and `probabilistic.py` trains the Transformer on the fixed sample. Both are CPU workloads; Transformer runtime is reported in the experiment metadata.

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
