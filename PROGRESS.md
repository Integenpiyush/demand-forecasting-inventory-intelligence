# Project progress

## Completed
- Phase 0 audit: the repository had no existing notebooks, source code, README, requirements file, or experiment reports to reuse.
- Created the project directory structure.
- Checked the system Python environment: Python 3.12.4 and pip 24.0.
- Created an isolated virtual environment at `.venv/` using Python 3.12.4.
- Added ignore rules for the virtual environment, local datasets, Python cache, and macOS metadata.
- Downloaded the Kaggle Rossmann `train.csv` and `store.csv` files into `data/raw/`.
- Verified `train.csv`: 1,017,209 data rows and 9 columns.
- Verified `store.csv`: 1,115 data rows and 10 columns.
- Checked `.venv/` before installing the first project library; it initially contained pip only.
- Confirmed there was no local record of the original 50 selected store IDs, random seed, chronological split, model settings, or measured metrics.
- Installed pandas 3.1.0 and its NumPy dependency in `.venv/`.
- Added and ran `src/inspect_data.py` to load both CSVs and report shape, schema, inferred dtypes, and missing-value counts.
- Resolved pandas' mixed-type warning by explicitly reading `StateHoliday` as a string category.
- Initial missingness check: `train.csv` has no missing values; `store.csv` has 3 missing `CompetitionDistance`, 354 missing competition-open month/year values, and 544 missing Promo2 start/interval values.
- Added and ran duplicate, date/store coverage, metadata key-matching, missingness-pattern, and zero-sales checks in `src/inspect_data.py`.
- Found 0 duplicate `(Store, Date)` keys and 0 duplicate store metadata IDs; train and metadata contain the same 1,115 stores.
- Training dates span 2013-01-01 through 2015-07-31 (942 distinct dates); 181 stores have fewer than 942 rows (minimum 758).
- All 544 missing Promo2 fields belong to stores with `Promo2 == 0`; participating stores (`Promo2 == 1`) have those fields populated.
- Of 354 missing competition-open month/year values, 3 overlap with missing competition distance; 351 have known distance but unknown opening month/year.
- Found 172,817 closed/zero-sales rows and 54 open/zero-sales rows; 52 of the latter have zero customers. These cases need interpretation before any filtering.
- Inspected open/zero-sales rows: the 2 full-dataset cases with positive customers are stores 948 and 1100; neither is in the saved 50-store sample. The sample has 4 open/zero-sales rows, all with zero customers.
- Inspected incomplete histories in the selected sample: all 8 have 758 observed dates and omit the same contiguous 184-date interval, 2014-07-01 through 2014-12-31; these dates are absent, not recorded as closed days.
- Summarized the target and customers in the 50-store sample: 45,628 observations; median sales 5,763, mean sales 6,032.82, and maximum sales 33,668. The open-store median sales is 6,554.5; closed rows have zero sales.
- Confirmed the saved sample contains four open/zero-sales rows, all with zero customers; these remain unfiltered pending domain interpretation.
- Added `src/select_stores.py` and `config/store_sample.json` to create and persist a new uniform random sample of 50 stores using seed 42; reruns reuse the saved IDs.
- Sample composition by store type is `a: 30`, `b: 1`, `c: 6`, `d: 13`; the random sample is not balanced by store type.

## Completed implementation
- Data audit and five EDA plots are complete. The joined processed sample contains 45,628 observations and 50 stores (2013-01-01 through 2015-07-31); absent dates and metadata gaps are preserved.
- Implemented exact-calendar lags, 14-day sequence windows, and chronological 42-day validation/test blocks. Dates: validation 2015-05-09 through 2015-06-19; test 2015-06-20 through 2015-07-31.
- Compared three lag baselines, Linear Regression, CNN, RNN, LSTM, and GRU. Final test set: 2,100 rows from all 50 selected stores. GRU test RMSE=920.55, MAE=552.94, WAPE=0.08854. This is only the selected sample under the rolling one-day-ahead protocol.
- Saved fitted linear and neural artifacts under `models/`, plus test/validation predictions and experiment metadata under `reports/results/`.
- Generated test error slices. GRU WAPE was 0.0812 without promotion and 0.0956 with promotion; these are descriptive associations, not a causal effect. Sunday WAPE was 0.448 with a small actual-sales denominator.
- Generated illustrative three-day reorder-point proxies at 90%, 95%, and 99% service levels. The chosen model was selected by validation WAPE; pooled validation residual SD was 813.94. These are sales-value proxies, not item-unit recommendations.
- Added project README and interview defense notes. Verification passed: Python compilation and 4 pytest checks.
- Added an interactive Streamlit dashboard with model comparison, grouped error summaries, inventory proxy summaries, methodology, and CSV downloads. It publishes aggregate outputs only; row-level and per-store files remain gitignored.
- Added a separate dashboard dependency set (Streamlit 1.51.0 with pandas 2.2.3) so it does not alter the model-training environment (pandas 3.1.0).

## Dataset notes
- Raw files are local in `data/raw/` and excluded from Git by `.gitignore`.
- The experiment is explicitly 50-store only; its measured results do not establish performance across all 1,115 stores.
- Forecast protocol: rolling one-day-ahead over chronological validation and test blocks; earlier observed target values may enter later-day lag inputs, while model fitting is train-only.
- Sample is store-type imbalanced, and eight sampled stores share a six-month missing observation interval; this experiment does not establish performance across all 1,115 stores.
