import numpy as np
import pandas as pd

from app.schemas import TrainRequest
from app.services.training_service import prepare_sequences


def make_frame(rows=100):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h"),
        "sensor": np.arange(rows, dtype=float),
        "target": np.arange(rows, dtype=float) * 2,
    })


def make_request(**overrides):
    values = {
        "date_column": "timestamp",
        "target_column": "target",
        "feature_columns": ["sensor"],
        "lookback": 8,
        "epochs": 4,
        "train_ratio": 0.7,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
    }
    values.update(overrides)
    return TrainRequest(**values)


def test_sequences_are_chronological_and_scalers_fit_training_rows_only():
    prepared = prepare_sequences(make_frame(), make_request())

    assert prepared["X_train"].shape == (62, 8, 1)
    assert prepared["X_validation"].shape == (15, 8, 1)
    assert prepared["X_test"].shape == (15, 8, 1)
    assert prepared["feature_scaler"].data_max_[0] == 69
    assert prepared["target_scaler"].data_max_[0] == 138
    assert prepared["test_dates"][0].startswith("2026-01-04 13:00")
    assert np.all(np.diff(prepared["test_actual"]) > 0)


def test_forward_fill_imputes_leading_missing_values_without_future_scaler_leakage():
    frame = make_frame()
    frame.loc[0, "sensor"] = np.nan
    frame.loc[72, "sensor"] = np.nan

    prepared = prepare_sequences(frame, make_request())

    assert np.isfinite(prepared["X_train"]).all()
    assert np.isfinite(prepared["X_validation"]).all()
    assert prepared["feature_scaler"].data_max_[0] == 69


def test_interpolation_is_limited_to_training_segment():
    frame = make_frame()
    frame.loc[20, "sensor"] = np.nan
    frame.loc[75, "sensor"] = np.nan

    prepared = prepare_sequences(frame, make_request(missing_strategy="interpolate"))

    assert np.isfinite(prepared["X_train"]).all()
    assert np.isfinite(prepared["X_validation"]).all()
    assert prepared["feature_scaler"].data_max_[0] == 69
