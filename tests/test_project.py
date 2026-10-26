import numpy as np
import pandas as pd

from src.forecast import SequenceNet, chronological_split_dates, exact_calendar_lag, metrics


def test_metrics_match_hand_calculation():
    result = metrics(np.array([10.0, 20.0]), np.array([12.0, 17.0]))
    assert result["MAE"] == 2.5
    assert np.isclose(result["RMSE"], np.sqrt(6.5))
    assert np.isclose(result["WAPE"], 5 / 30)


def test_sequence_model_shapes():
    import torch

    sequence = torch.zeros((4, 14, 1))
    covariates = torch.zeros((4, 6))
    for kind in ("CNN", "RNN", "LSTM", "GRU"):
        model = SequenceNet(kind, exog_dim=6)
        assert model(sequence, covariates).shape == (4,)


def test_date_lag_does_not_jump_over_missing_calendar_day():
    observed = pd.DataFrame({"Store": [1, 1], "Date": pd.to_datetime(["2020-01-01", "2020-01-03"]), "Sales": [10.0, 30.0]})
    targets = observed[["Store", "Date"]].copy()
    result = exact_calendar_lag(targets, observed, 1, "lag_1")
    assert np.isnan(result.loc[result.Date == "2020-01-03", "lag_1"].iloc[0])


def test_chronological_cutoffs_use_requested_calendar_blocks():
    dates = list(pd.date_range("2020-01-01", periods=126))
    validation_start, test_start = chronological_split_dates(dates, test_days=42, validation_days=42)
    assert validation_start == dates[42]
    assert test_start == dates[84]
