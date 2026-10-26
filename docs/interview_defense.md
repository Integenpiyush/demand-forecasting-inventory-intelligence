# Interview defense: Rossmann forecasting project

## Short project pitch

“I built a reproducible daily sales forecasting comparison on a fixed 50-store sample of the Rossmann competition data. I audited missingness first, used calendar-date joins for lag features so missing dates could not corrupt the history, then compared simple lag baselines and Linear Regression with CNN, RNN, LSTM, and GRU models on chronological validation and test windows. I report RMSE, MAE, and WAPE with the exact sample and forecast protocol, then demonstrate a clearly labeled reorder-point proxy. The experiment is a CPU-sized study and does not claim performance for all 1,115 stores or item-level inventory.”

## Interview-sized answers

- **What did it achieve?** In the final run, GRU had the lowest validation WAPE (0.08794) and scored test RMSE 920.55, MAE 552.94, and WAPE 0.08854 on 2,100 daily predictions covering all 50 sampled stores. These are Rossmann sales-value metrics for the stated sample and protocol, not all-store or item-unit results.

- **Why 50 stores?** The project constraint is CPU capacity. A saved list and fixed seed make the sample reproducible; it is not evidence of full-population performance.
- **Why chronological splits?** Random splitting can let future dates inform a past forecast. The latest 42 days are held out for testing, preceded by 42 validation days.
- **Why exact-date lags?** A row shift means “previous observed row,” which can be months earlier when dates are absent. Exact date-key joins preserve missing history.
- **What does rolling one-day-ahead mean?** Each day's forecast can use observations already available from earlier days, including earlier test days. Model parameters are fit only on training data; this is not a fixed-origin 42-day forecast.
- **Why not use Customers?** It is not known at the time a future sales forecast is made and would create target leakage in this setup.
- **Why compare simple baselines?** A complex network is useful only if it improves on inexpensive, interpretable forecasts under the same test protocol.
- **What does WAPE measure?** Total absolute error divided by total actual sales, giving an aggregate scale-normalized error. It can obscure weak results for low-volume stores, so I also report RMSE and MAE.
- **What are the inventory results?** They are a method demonstration in sales-value units, using pooled validation residual variability and an assumed lead time. Rossmann has no SKU-level units, lead times, or costs here, so I would not call them purchase quantities.
- **What is the main limitation?** A random 50-store sample with uneven store-type representation and a shared six-month observation gap in eight stores. Results do not establish population-wide performance.

## Evidence policy for resume claims

Use only the metrics and coverage written by an actual completed run in `reports/results/model_metrics.csv` and `experiment.json`. A defensible resume bullet should state the 50-store sample, test protocol, evaluated model, and measured metric. Do not claim full-store coverage, causal promotion lift, or reduced stockout cost: this project does not measure those outcomes.

**Evidence-backed example:** “Built a reproducible 50-store daily sales forecasting experiment comparing lag baselines, Linear Regression, CNN, RNN, LSTM, and GRU; the validation-selected GRU achieved 8.85% test WAPE on a 2,100-row chronological rolling one-day-ahead test set, with date-exact lags and an illustrative reorder-point analysis.”
