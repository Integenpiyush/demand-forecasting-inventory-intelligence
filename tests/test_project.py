import numpy as np
import pandas as pd

from src.forecast import SequenceNet, chronological_split_dates, exact_calendar_lag, metrics
from src.inventory import contiguous_lead_time_errors, empirical_safety_stock
from src.probabilistic import StoreTransformer, _feature_matrix, calibrate_quantiles, encode_store_id, pinball_loss, restore_sales_quantiles
from src.probabilistic import HIST_NUMERIC, TARGET_NUMERIC, _scale
from src.publish_dashboard_data import validate_quantile_dashboard_data


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


def test_transformer_quantiles_are_nonnegative_ordered_and_unknown_store_is_safe():
    import torch

    model = StoreTransformer(n_stores=3, history=14)
    history = torch.zeros((2, 14, 8))
    target = torch.zeros((2, 7))
    # 0 is the explicit unknown-store embedding index.
    pred = model(history, target, torch.tensor([1, 0]))
    assert pred.shape == (2, 3)
    assert torch.all(pred[:, 0] <= pred[:, 1])
    assert torch.all(pred[:, 1] <= pred[:, 2])
    restored = restore_sales_quantiles(pred.detach().numpy(), mean=100.0, std=10.0)
    assert np.all(restored >= 0)
    assert np.all(restored[:, 0] <= restored[:, 1])
    assert np.all(restored[:, 1] <= restored[:, 2])


def test_pinball_loss_hand_calculation():
    import torch

    pred = torch.tensor([[1.0, 2.0, 3.0]])
    actual = torch.tensor([2.0])
    # Zero for the median; 0.1 for each outer quantile, averaged over 3.
    assert np.isclose(pinball_loss(pred, actual).item(), (0.1 + 0.0 + 0.1) / 3)


def test_quantile_calibration_uses_validation_residual_offsets_and_keeps_order():
    actual = np.array([8.0, 10.0, 12.0, 14.0, 16.0])
    val_pred = np.tile(np.array([[10.0, 12.0, 14.0]]), (5, 1))
    calibrated, offsets = calibrate_quantiles(actual, val_pred, val_pred[:2])
    assert offsets.shape == (3,)
    assert np.all(calibrated >= 0)
    assert np.all(calibrated[:, 0] <= calibrated[:, 1])
    assert np.all(calibrated[:, 1] <= calibrated[:, 2])


def test_empirical_inventory_buffer_uses_complete_consecutive_windows():
    rows = pd.DataFrame({
        "Store": [1, 1, 1, 1],
        "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"]),
        "target": [12.0, 14.0, 100.0, 104.0],
        "P50": [10.0, 10.0, 90.0, 90.0],
    })
    errors = contiguous_lead_time_errors(rows, 2)
    assert errors.tolist() == [6.0, 24.0]
    safety, coverage = empirical_safety_stock(errors, 0.5)
    assert safety == 15.0
    assert coverage == 0.5


def test_transformer_fixture_end_to_end_smoke():
    import torch

    dates = pd.date_range("2020-01-01", periods=18)
    fixture = pd.DataFrame({
        "Store": 7, "Date": dates, "Sales": np.linspace(100, 270, len(dates)),
        "Promo": 0, "Open": 1, "SchoolHoliday": 0, "Promo2": 0,
        "CompetitionDistance": 500.0, "target": np.linspace(100, 270, len(dates)),
        "weekday": dates.day_of_week, "split": "train",
    })
    history, target_features, stores, kept = _feature_matrix(fixture)
    assert history.shape == (4, 14, 8)
    model = StoreTransformer(n_stores=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(history[:2])
    context = torch.from_numpy(target_features[:2])
    store_index = torch.ones(2, dtype=torch.long)
    target = torch.tensor([0.1, 0.2])
    optimizer.zero_grad()
    loss = pinball_loss(model(x, context, store_index), target)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert len(stores) == len(kept) == 4
    short_history, _, _, _ = _feature_matrix(fixture, history_days=7)
    assert short_history.shape == (11, 7, 8)
    assert encode_store_id(7, {7: 1}) == 1
    assert encode_store_id(999, {7: 1}) == 0


def test_known_feature_lists_avoid_customers_and_same_day_target():
    assert "Customers" not in HIST_NUMERIC + TARGET_NUMERIC
    assert "Sales" not in TARGET_NUMERIC


def test_preprocessing_uses_training_statistics_only():
    train = np.array([[1.0, 10.0], [3.0, 20.0]])
    validation = np.array([[1000.0, 2000.0]])
    scaled_train, scaled_validation = _scale(train, validation)
    assert np.allclose(scaled_train.mean(axis=0), [0.0, 0.0])
    assert scaled_validation[0, 0] > 100


def test_dashboard_quantile_artifact_contract_and_rejects_crossing_intervals():
    frame = pd.DataFrame({
        "Store": [1, 1], "Date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "target": [5.0, 7.0], "P10": [1.0, 2.0], "P50": [5.0, 6.0],
        "P90": [10.0, 11.0], "split": ["validation", "test"], "Promo": [0, 1],
    })
    validate_quantile_dashboard_data(frame)
    frame.loc[0, "P10"] = 6.0
    with np.testing.assert_raises(ValueError):
        validate_quantile_dashboard_data(frame)
