"""Transformer quantile forecasting and calibration for the sampled stores."""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecast import HISTORY, ROOT, SEED, build_frame, metrics

RESULTS = ROOT / "reports" / "results"
MODEL_DIR = ROOT / "models"
QUANTILES = (0.1, 0.5, 0.9)
HIST_NUMERIC = ["sales", "weekday", "month", "Promo", "Open", "SchoolHoliday", "Promo2", "CompetitionDistance"]
TARGET_NUMERIC = ["weekday", "month", "Promo", "Open", "SchoolHoliday", "Promo2", "CompetitionDistance"]


class StoreTransformer(nn.Module):
    """Encode an ordered 14-day sequence with attention and a learned store ID."""

    def __init__(self, n_stores: int, history: int = HISTORY, input_dim: int = 8,
                 target_dim: int = 7, d_model: int = 32, nhead: int = 4,
                 layers: int = 2, embedding_dim: int = 8):
        super().__init__()
        self.history = history
        self.store_embedding = nn.Embedding(n_stores + 1, embedding_dim, padding_idx=0)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, history, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(d_model + target_dim + embedding_dim, 64), nn.GELU(), nn.Linear(64, 3)
        )

    def forward(self, history: torch.Tensor, target_features: torch.Tensor,
                store_index: torch.Tensor) -> torch.Tensor:
        embedded = self.store_embedding(store_index)
        tokens = self.input_projection(history) + self.position
        summary = self.encoder(tokens).mean(dim=1)
        raw = self.head(torch.cat([summary, target_features, embedded], dim=1))
        # In standardized target space values may be negative. Use an unconstrained
        # median plus positive gaps; nonnegative clipping happens after inverse scaling.
        p50 = raw[:, 0]
        p10 = p50 - torch.nn.functional.softplus(raw[:, 1])
        p90 = p50 + torch.nn.functional.softplus(raw[:, 2])
        return torch.stack((p10, p50, p90), dim=1)


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor,
                 quantiles: tuple[float, ...] = QUANTILES) -> torch.Tensor:
    """Mean pinball loss across observations and requested quantiles."""
    q = torch.as_tensor(quantiles, dtype=prediction.dtype, device=prediction.device).view(1, -1)
    error = target.view(-1, 1) - prediction
    return torch.maximum(q * error, (q - 1) * error).mean()


def encode_store_id(store_id: int, store_to_index: dict[int, int]) -> int:
    """Map an observed store to its learned embedding index, or 0 if unseen."""
    return store_to_index.get(int(store_id), 0)


def _feature_matrix(frame: pd.DataFrame, history_days: int = HISTORY) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build exact-calendar historical feature windows for each target row."""
    ordered = frame.sort_values(["Store", "Date"]).copy()
    data = ordered.set_index(["Store", "Date"], drop=False)
    hist_rows, target_rows, store_ids, keep = [], [], [], []
    for ix, row in ordered.iterrows():
        dated = []
        for days_back in range(history_days, 0, -1):
            key = (row.Store, row.Date - pd.Timedelta(days=days_back))
            if key not in data.index:
                dated = []
                break
            dated.append(data.loc[key])
        if len(dated) != history_days:
            continue
        hist_rows.append(np.asarray([[float(r.Sales), float(r.Date.day_of_week), float(r.Date.month),
                                      float(r.Promo), float(r.Open), float(r.SchoolHoliday),
                                      float(r.Promo2), float(r.CompetitionDistance)] for r in dated], dtype=np.float32))
        target_rows.append([float(row.Date.day_of_week), float(row.Date.month), float(row.Promo),
                            float(row.Open), float(row.SchoolHoliday), float(row.Promo2),
                            float(row.CompetitionDistance)])
        store_ids.append(int(row.Store)); keep.append(ix)
    if not keep:
        raise ValueError("No targets have complete exact-calendar history")
    subset = ordered.loc[keep].copy()
    return np.stack(hist_rows), np.asarray(target_rows, dtype=np.float32), np.asarray(store_ids), subset.index.astype(int).tolist()


def _scale(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    flat = train.reshape(-1, train.shape[-1])
    mean = np.nanmean(flat, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.nanstd(flat, axis=0)
    std[std == 0] = 1.0
    std = np.where(np.isfinite(std), std, 1.0)
    scaled = []
    for x in (train, *others):
        imputed = np.where(np.isfinite(x), x, mean)
        scaled.append(((imputed - mean) / std).astype(np.float32))
    return tuple(scaled)


def _quantile_metrics(actual: np.ndarray, quantile_pred: np.ndarray) -> dict[str, float]:
    qloss = []
    for index, q in enumerate(QUANTILES):
        err = actual - quantile_pred[:, index]
        qloss.append(np.maximum(q * err, (q - 1) * err).mean())
    return {
        "pinball_p10": float(qloss[0]), "pinball_p50": float(qloss[1]), "pinball_p90": float(qloss[2]),
        "pinball_mean": float(np.mean(qloss)),
        "p10_p90_coverage": float(np.mean((actual >= quantile_pred[:, 0]) & (actual <= quantile_pred[:, 2]))),
        "p10_coverage": float(np.mean(actual <= quantile_pred[:, 0])),
        "p50_coverage": float(np.mean(actual <= quantile_pred[:, 1])),
        "p90_coverage": float(np.mean(actual <= quantile_pred[:, 2])),
    }


def restore_sales_quantiles(prediction_scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Undo target scaling, then clip sales to zero while preserving quantile order."""
    prediction = np.asarray(prediction_scaled, dtype=float) * std + mean
    prediction = np.maximum(prediction, 0.0)
    prediction[:, 1] = np.maximum(prediction[:, 1], prediction[:, 0])
    prediction[:, 2] = np.maximum(prediction[:, 2], prediction[:, 1])
    return prediction


def calibrate_quantiles(validation_actual: np.ndarray, validation_prediction: np.ndarray,
                        prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply validation-only additive residual quantiles; return forecasts and offsets."""
    actual = np.asarray(validation_actual, dtype=float)
    val_pred = np.asarray(validation_prediction, dtype=float)
    offsets = np.asarray([np.quantile(actual - val_pred[:, i], q) for i, q in enumerate(QUANTILES)])
    calibrated = np.maximum(np.asarray(prediction, dtype=float) + offsets[None, :], 0.0)
    calibrated[:, 1] = np.maximum(calibrated[:, 1], calibrated[:, 0])
    calibrated[:, 2] = np.maximum(calibrated[:, 2], calibrated[:, 1])
    return calibrated, offsets


def train_and_evaluate(history_days: int = HISTORY) -> None:
    if history_days < 1:
        raise ValueError("history_days must be at least 1")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(2)
    RESULTS.mkdir(parents=True, exist_ok=True); MODEL_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_frame()
    history, target_features, store_ids, indices = _feature_matrix(frame, history_days=history_days)
    meta = frame.loc[indices, ["Store", "Date", "target", "Promo", "weekday", "split"]].reset_index(drop=True)
    train_mask = meta.split.eq("train").to_numpy()
    val_mask = meta.split.eq("validation").to_numpy()
    test_mask = meta.split.eq("test").to_numpy()
    # Scale every continuous input using training rows only.
    hist_train, hist_val, hist_test = _scale(history[train_mask], history[val_mask], history[test_mask])
    target_train, target_val, target_test = _scale(target_features[train_mask], target_features[val_mask], target_features[test_mask])
    y = meta.target.to_numpy(dtype=np.float32)
    y_mean, y_std = float(y[train_mask].mean()), float(y[train_mask].std() or 1.0)
    y_scaled = (y - y_mean) / y_std
    # Embedding IDs are learned from training stores; 0 is reserved for unknown IDs.
    known_stores = sorted(set(store_ids[train_mask].tolist()))
    store_map = {store: i + 1 for i, store in enumerate(known_stores)}
    store_index = np.asarray([encode_store_id(s, store_map) for s in store_ids], dtype=np.int64)
    all_hist = np.empty_like(history); all_target = np.empty_like(target_features)
    all_hist[train_mask], all_hist[val_mask], all_hist[test_mask] = hist_train, hist_val, hist_test
    all_target[train_mask], all_target[val_mask], all_target[test_mask] = target_train, target_val, target_test
    tensors = [torch.from_numpy(all_hist), torch.from_numpy(all_target), torch.from_numpy(store_index), torch.from_numpy(y_scaled)]
    train_ids = np.flatnonzero(train_mask); val_ids = np.flatnonzero(val_mask); test_ids = np.flatnonzero(test_mask)
    loader = DataLoader(TensorDataset(*(t[train_ids] for t in tensors)), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    model = StoreTransformer(len(known_stores), history=history_days)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_loss, best_state, stale, epochs_done = float("inf"), None, 0, 0
    started = time.perf_counter()
    for epoch in range(40):
        model.train()
        for xb, eb, sb, yb in loader:
            optimizer.zero_grad(); loss = pinball_loss(model(xb, eb, sb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = pinball_loss(model(*(t[val_ids] for t in tensors[:3])), tensors[3][val_ids]).item()
        epochs_done = epoch + 1
        if vl < best_loss - 1e-5:
            best_loss = vl; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 5: break
    training_seconds = time.perf_counter() - started
    assert best_state is not None
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred = restore_sales_quantiles(model(*(t for t in tensors[:3])).numpy(), y_mean, y_std)
    output = meta[["Store", "Date", "target", "Promo", "weekday", "split"]].copy()
    output[["P10_raw", "P50_raw", "P90_raw"]] = pred
    validation_rows = output.split.eq("validation").to_numpy()
    validation_actual = output.loc[validation_rows, "target"].to_numpy()
    validation_raw = pred[validation_rows]
    calibrated, calibration_offsets = calibrate_quantiles(validation_actual, validation_raw, pred)
    output[["P10", "P50", "P90"]] = calibrated
    output.to_csv(RESULTS / "transformer_quantile_predictions.csv", index=False)
    rows, slices = [], []
    for split, mask in (("validation", val_mask), ("test", test_mask)):
        part = output.loc[mask].copy(); actual = part.target.to_numpy(); qp = part[["P10", "P50", "P90"]].to_numpy()
        med = metrics(actual, qp[:, 1]); rows.append({"split": split, "model": "Transformer", "n": len(part), **med, **_quantile_metrics(actual, qp)})
        for group_name, group_col in (("promo", "Promo"), ("weekday", "weekday"), ("store", "Store")):
            for value, group in part.groupby(group_col):
                if len(group) < (20 if group_name == "store" else 30):
                    continue
                ga = group.target.to_numpy(); gq = group[["P10", "P50", "P90"]].to_numpy()
                slices.append({"split": split, "group": group_name, "value": value, "n": len(group), **_quantile_metrics(ga, gq), **metrics(ga, gq[:, 1])})
    pd.DataFrame(rows).to_csv(RESULTS / "transformer_quantile_metrics.csv", index=False)
    pd.DataFrame(slices).to_csv(RESULTS / "transformer_calibration_slices.csv", index=False)
    # Add the Transformer medians to the existing all-model comparison, retaining original models.
    existing = pd.read_csv(RESULTS / "model_metrics.csv")
    test_metric = pd.DataFrame([r for r in rows if r["split"] == "test"])[["model", "n", "RMSE", "MAE", "WAPE"]]
    combined = pd.concat([existing.loc[existing.model != "Transformer", test_metric.columns], test_metric], ignore_index=True)
    combined.sort_values("WAPE").to_csv(RESULTS / "model_metrics.csv", index=False)
    existing_val = pd.read_csv(RESULTS / "validation_metrics.csv")
    val_metric = pd.DataFrame([r for r in rows if r["split"] == "validation"])[["model", "n", "RMSE", "MAE", "WAPE"]]
    pd.concat([existing_val.loc[existing_val.model != "Transformer", val_metric.columns], val_metric], ignore_index=True).sort_values("WAPE").to_csv(RESULTS / "validation_metrics.csv", index=False)
    torch.save({"state_dict": best_state, "history_days": history_days, "store_to_index": store_map,
                "unknown_store_index": 0, "seed": SEED, "training_seconds": training_seconds,
                "best_validation_pinball_scaled": best_loss, "epochs_trained": epochs_done,
                "validation_quantile_calibration_offsets_sales": calibration_offsets.tolist(),
                "hist_mean": history[train_mask].reshape(-1, 8).mean(0).tolist(),
                "hist_std": history[train_mask].reshape(-1, 8).std(0).tolist(),
                "target_feature_mean": target_features[train_mask].mean(0).tolist(),
                "target_feature_std": target_features[train_mask].std(0).tolist(), "sales_mean": y_mean, "sales_std": y_std},
               MODEL_DIR / "transformer_quantile.pt")
    manifest_path = RESULTS / "experiment.json"
    manifest = json.loads(manifest_path.read_text()); manifest["transformer_training_seconds"] = training_seconds
    manifest["transformer_history_days"] = history_days
    manifest["transformer_epochs_trained"] = epochs_done
    manifest["transformer_unknown_store_policy"] = "reserved embedding index 0; training samples use known store IDs"
    manifest["transformer_quantile_calibration"] = "Per-quantile additive offsets fit on validation residual quantiles; final test is not used for calibration. Calibrated outputs are clipped to nonnegative sales and sorted."
    manifest["transformer_quantile_calibration_offsets_sales"] = calibration_offsets.tolist()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Transformer trained {epochs_done} epochs in {training_seconds:.1f}s; validation pinball={best_loss:.4f}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-days", type=int, default=HISTORY, help="historical sequence length (default: 14)")
    train_and_evaluate(parser.parse_args().history_days)
