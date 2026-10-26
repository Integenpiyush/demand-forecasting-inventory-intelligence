"""Train reproducible 50-store, rolling one-day-ahead forecasting experiments."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "reports" / "results"
MODEL_DIR = ROOT / "models"
SAMPLE = ROOT / "config" / "store_sample.json"
SEED = 42
HISTORY = 14
TEST_DAYS = 42
VAL_DAYS = 42


def exact_calendar_lag(frame: pd.DataFrame, source: pd.DataFrame, lag_days: int, column: str) -> pd.DataFrame:
    """Attach Sales only when the exact prior calendar date exists for the same store."""
    history = source[["Store", "Date", "Sales"]].rename(columns={"Date": "history_date", "Sales": column}).copy()
    history["Date"] = history["history_date"] + pd.Timedelta(days=lag_days)
    return frame.merge(history[["Store", "Date", column]], on=["Store", "Date"], how="left", validate="one_to_one")


def chronological_split_dates(dates: list[pd.Timestamp], test_days: int, validation_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (validation_start, test_start) from sorted unique dates."""
    if len(dates) <= test_days + validation_days:
        raise ValueError("Need more dates than the validation and test blocks combined")
    return pd.Timestamp(dates[-(test_days + validation_days)]), pd.Timestamp(dates[-test_days])


def build_frame() -> pd.DataFrame:
    """Join metadata and create lags by exact calendar date, never by row order."""
    processed = ROOT / "data" / "processed" / "sample_sales_with_store.csv"
    if not processed.is_file():
        raise FileNotFoundError("Run `.venv/bin/python src/prepare_data.py` to validate and prepare the 50-store sample first.")
    frame = pd.read_csv(processed, parse_dates=["Date"], dtype={"StateHoliday": "string"})
    frame = frame.sort_values(["Store", "Date"]).reset_index(drop=True)
    source = frame[["Store", "Date", "Sales"]].copy()
    for lag in (1, 7):
        frame = exact_calendar_lag(frame, source, lag, f"lag_{lag}")
    # Previous seven calendar days; requiring all seven observed prevents crossing missing-data gaps.
    for lag in range(1, 8):
        frame = exact_calendar_lag(frame, source, lag, f"previous_{lag}")
    frame["mean_prev7"] = frame[[f"previous_{i}" for i in range(1, 8)]].mean(axis=1)
    frame.loc[frame[[f"previous_{i}" for i in range(1, 8)]].isna().any(axis=1), "mean_prev7"] = np.nan
    # Sequence is ordered oldest to newest, and each point is matched to Date - offset.
    for lag in range(1, HISTORY + 1):
        frame = exact_calendar_lag(frame, source, lag, f"seq_{lag}")
    frame["weekday"] = frame.Date.dt.day_of_week
    frame["month"] = frame.Date.dt.month
    frame["year"] = frame.Date.dt.year
    frame["StateHoliday"] = frame.StateHoliday.fillna("0").astype(str)
    frame["StoreType"] = frame.StoreType.astype(str)
    frame["Assortment"] = frame.Assortment.astype(str)
    frame["Promo2"] = frame.Promo2.fillna(0).astype(int)
    frame["target"] = frame.Sales.astype(float)
    # Deep models use fully observed calendar sequences. Tabular models require their configured lags.
    seq_cols = [f"seq_{i}" for i in range(HISTORY, 0, -1)]
    frame["sequence_ok"] = frame[seq_cols].notna().all(axis=1)
    frame["split"] = "train"
    dates = list(pd.to_datetime(sorted(frame.Date.unique())))
    val_start, test_start = chronological_split_dates(dates, TEST_DAYS, VAL_DAYS)
    frame.loc[frame.Date >= val_start, "split"] = "validation"
    frame.loc[frame.Date >= test_start, "split"] = "test"
    frame.attrs["split_dates"] = {"validation_start": str(pd.Timestamp(val_start).date()), "test_start": str(pd.Timestamp(test_start).date()), "last_date": str(pd.Timestamp(dates[-1]).date())}
    return frame


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(np.abs(err))), "WAPE": float(np.sum(np.abs(err)) / max(np.sum(np.abs(y)), 1e-12))}


class SequenceNet(nn.Module):
    def __init__(self, kind: str, exog_dim: int):
        super().__init__()
        if kind == "CNN":
            self.temporal = nn.Sequential(nn.Conv1d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv1d(8, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
            temporal_dim = 8
        else:
            layer = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
            self.temporal = layer(input_size=1, hidden_size=16, batch_first=True)
            temporal_dim = 16
        self.head = nn.Sequential(nn.Linear(temporal_dim + exog_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, sequence: torch.Tensor, exog: torch.Tensor) -> torch.Tensor:
        if isinstance(self.temporal, nn.Sequential):
            encoded = self.temporal(sequence.transpose(1, 2))
        else:
            output = self.temporal(sequence)
            encoded = output[1][0] if isinstance(self.temporal, nn.LSTM) else output[1]
            encoded = encoded[-1]
        return self.head(torch.cat([encoded, exog], dim=1)).squeeze(1)


def run() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(2)
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_frame()
    # Fixed daily update protocol: each test date may use observations from earlier dates, including prior test days.
    # Test targets themselves are never used to fit model parameters.
    tabular = ["lag_1", "lag_7", "mean_prev7", "weekday", "month", "year", "Promo", "Open", "SchoolHoliday", "Promo2", "CompetitionDistance", "DayOfWeek", "StateHoliday", "StoreType", "Assortment", "Store"]
    numeric = ["lag_1", "lag_7", "mean_prev7", "weekday", "month", "year", "Promo", "Open", "SchoolHoliday", "Promo2", "CompetitionDistance", "DayOfWeek"]
    categorical = ["StateHoliday", "StoreType", "Assortment", "Store"]
    fit = frame[(frame.split == "train") & frame[["lag_1", "lag_7", "mean_prev7"]].notna().all(axis=1)]
    val = frame[(frame.split == "validation") & frame[["lag_1", "lag_7", "mean_prev7"]].notna().all(axis=1)]
    test = frame[(frame.split == "test") & frame[["lag_1", "lag_7", "mean_prev7"]].notna().all(axis=1)].copy()
    prep = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])
    linear = Pipeline([("prep", prep), ("model", LinearRegression())])
    linear.fit(fit[tabular], fit.target)
    joblib.dump(linear, MODEL_DIR / "linear_regression.joblib")
    test["LinearRegression"] = np.maximum(0, linear.predict(test[tabular]))
    val = val.copy()
    val["LinearRegression"] = np.maximum(0, linear.predict(val[tabular]))
    test["Naive_lag1"] = test.lag_1
    test["Seasonal_naive_7d"] = test.lag_7
    test["Mean_previous_7d"] = test.mean_prev7
    val["Naive_lag1"] = val.lag_1
    val["Seasonal_naive_7d"] = val.lag_7
    val["Mean_previous_7d"] = val.mean_prev7
    scores = []
    validation_scores = []
    for name in ["Naive_lag1", "Seasonal_naive_7d", "Mean_previous_7d", "LinearRegression"]:
        scores.append({"model": name, "n": len(test), **metrics(test.target.to_numpy(), test[name].to_numpy())})
        validation_scores.append({"model": name, "n": len(val), **metrics(val.target.to_numpy(), val[name].to_numpy())})

    # Neural models get 14 past calendar-day sales plus target-date/store covariates.
    seq_cols = [f"seq_{i}" for i in range(HISTORY, 0, -1)]
    exog_num = ["weekday", "month", "Promo", "Open", "SchoolHoliday", "Promo2", "CompetitionDistance"]
    deep_train = frame[(frame.split == "train") & frame.sequence_ok].copy()
    deep_val = frame[(frame.split == "validation") & frame.sequence_ok].copy()
    deep_test = frame[(frame.split == "test") & frame.sequence_ok].copy()
    deep_pre = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), exog_num), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])
    ex_train = deep_pre.fit_transform(deep_train[exog_num + categorical]).astype(np.float32)
    ex_val = deep_pre.transform(deep_val[exog_num + categorical]).astype(np.float32)
    ex_test = deep_pre.transform(deep_test[exog_num + categorical]).astype(np.float32)
    sales_mean = float(deep_train.target.mean()); sales_std = float(deep_train.target.std()) or 1.0
    joblib.dump(deep_pre, MODEL_DIR / "sequence_covariate_preprocessor.joblib")
    (MODEL_DIR / "sequence_scaling.json").write_text(json.dumps({"history_days": HISTORY, "sales_mean_train": sales_mean, "sales_std_train": sales_std, "exog_numeric": exog_num, "exog_categorical": categorical}, indent=2) + "\n")
    def arrays(part):
        xseq = ((part[seq_cols].to_numpy(dtype=np.float32) - sales_mean) / sales_std)[:, :, None]
        y = (part.target.to_numpy(dtype=np.float32) - sales_mean) / sales_std
        return torch.from_numpy(xseq), None, torch.from_numpy(y)
    xtr, _, ytr = arrays(deep_train); xva, _, yva = arrays(deep_val); xte, _, yte = arrays(deep_test)
    etr, eva, ete = torch.from_numpy(ex_train), torch.from_numpy(ex_val), torch.from_numpy(ex_test)
    loader = DataLoader(TensorDataset(xtr, etr, ytr), batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    for kind in ("CNN", "RNN", "LSTM", "GRU"):
        model = SequenceNet(kind, ex_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        loss_fn = nn.MSELoss()
        best_val = float("inf"); best_state = None; stale = 0; epochs_done = 0
        for epoch in range(12):
            model.train()
            for xb, eb, yb in loader:
                optimizer.zero_grad(); loss = loss_fn(model(xb, eb), yb); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                vloss = float(loss_fn(model(xva, eva), yva))
            epochs_done = epoch + 1
            if vloss < best_val - 1e-5:
                best_val = vloss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; stale = 0
            else:
                stale += 1
                if stale >= 3: break
        model.load_state_dict(best_state); model.eval()
        torch.save({"model_kind": kind, "state_dict": best_state, "history_days": HISTORY, "exog_dim": ex_train.shape[1], "seed": SEED, "best_validation_mse_normalized": best_val, "epochs_trained": epochs_done}, MODEL_DIR / f"{kind.lower()}_best.pt")
        with torch.no_grad():
            val_pred = np.maximum(0, model(xva, eva).numpy() * sales_std + sales_mean)
            pred = np.maximum(0, model(xte, ete).numpy() * sales_std + sales_mean)
        val.loc[deep_val.index, kind] = val_pred
        test.loc[deep_test.index, kind] = pred
        scores.append({"model": kind, "n": len(deep_test), "validation_epochs": epochs_done, **metrics(deep_test.target.to_numpy(), pred)})
        validation_scores.append({"model": kind, "n": len(deep_val), "validation_epochs": epochs_done, **metrics(deep_val.target.to_numpy(), val_pred)})
        print(f"{kind} completed: {epochs_done} epochs, validation MSE={best_val:.4f}, test rows={len(deep_test):,}")

    # Write predictions and aggregate scores; omitted rows have a documented missing-history reason.
    test[["Store", "Date", "target", "Naive_lag1", "Seasonal_naive_7d", "Mean_previous_7d", "LinearRegression", "CNN", "RNN", "LSTM", "GRU"]].to_csv(RESULTS / "test_predictions.csv", index=False)
    val[["Store", "Date", "target", "Naive_lag1", "Seasonal_naive_7d", "Mean_previous_7d", "LinearRegression", "CNN", "RNN", "LSTM", "GRU"]].to_csv(RESULTS / "validation_predictions.csv", index=False)
    pd.DataFrame(scores).sort_values("WAPE").to_csv(RESULTS / "model_metrics.csv", index=False)
    pd.DataFrame(validation_scores).sort_values("WAPE").to_csv(RESULTS / "validation_metrics.csv", index=False)
    meta = {"seed": SEED, "store_ids": json.loads(SAMPLE.read_text())["store_ids"], "history_days": HISTORY, "validation_days": VAL_DAYS, "test_days": TEST_DAYS, "split_dates": frame.attrs["split_dates"], "train_rows_tabular": len(fit), "validation_rows_tabular": len(val), "test_rows_tabular": len(test), "train_rows_sequence": len(deep_train), "validation_rows_sequence": len(deep_val), "test_rows_sequence": len(deep_test), "protocol": "rolling one-day-ahead; observed prior-day sales may be used for later test dates; fit is train-only; date-exact lags; missing history omitted", "sample_scope": "50 sampled stores only; no claim of generalization to all 1,115 stores"}
    (RESULTS / "experiment.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(pd.DataFrame(scores).sort_values("WAPE").to_string(index=False))
    print(f"Saved experiment artifacts under {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    run()
