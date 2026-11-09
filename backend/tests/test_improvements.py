import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import TrainRequest
from app.services.data_service import explore_dataset
from app.services.training_service import forecast_future, prepare_sequences


def seasonal_frame(rows=240, period=12):
    index = np.arange(rows)
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h"),
        "value": 50 + 10 * np.sin(2 * np.pi * index / period) + index * 0.05,
        "other": np.cos(index / 7),
    })


def make_request(**overrides):
    values = {
        "date_column": "timestamp",
        "target_column": "value",
        "feature_columns": ["value"],
        "lookback": 12,
        "epochs": 2,
        "train_ratio": 0.7,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
    }
    values.update(overrides)
    return TrainRequest(**values)


class FakeModel:
    """Predicts a constant scaled value so the recursive loop is deterministic."""

    def __call__(self, window, training=False):
        assert window.shape[0] == 1
        return np.array([[0.5]], dtype=np.float32)


def test_eda_reports_histogram_boxplot_acf_and_seasonal_decomposition():
    eda = explore_dataset(seasonal_frame(), "value")

    assert sum(bin_["count"] for bin_ in eda["histograms"]["value"]) == 240
    box = eda["boxplots"]["value"]
    assert box["q1"] <= box["median"] <= box["q3"]
    assert eda["acf"]["values"][0] == 1.0
    assert eda["acf"]["confidence"] > 0
    assert eda["period"] == 12
    assert eda["decomposition"]["period"] == 12
    point = eda["decomposition"]["points"][50]
    assert point["observed"] is not None and point["seasonal"] is not None


def test_eda_boxplot_counts_outliers():
    frame = seasonal_frame()
    frame.loc[10, "value"] = 10_000

    box = explore_dataset(frame, "value")["boxplots"]["value"]

    assert box["outlier_count"] == 1
    assert box["outliers"] == [10_000.0]


def test_eda_manual_period_and_date_column_selection():
    frame = seasonal_frame()
    frame["other_date"] = pd.date_range("2025-01-01", periods=len(frame), freq="D")

    eda = explore_dataset(frame, "value", date_column="other_date", period=24)

    assert eda["series_date_column"] == "other_date"
    assert set(eda["date_columns"]) == {"timestamp", "other_date"}
    assert eda["decomposition"]["period"] == 24


def test_eda_keeps_previous_keys_and_rejects_unknown_date_column():
    with TestClient(app) as client:
        content = seasonal_frame().to_csv(index=False)
        dataset_id = client.post(
            "/api/v1/datasets", files={"file": ("s.csv", content, "text/csv")}
        ).json()["id"]
        ok = client.get(f"/api/v1/datasets/{dataset_id}/eda", params={"column": "value"})
        bad = client.get(
            f"/api/v1/datasets/{dataset_id}/eda", params={"column": "value", "date_column": "value"}
        )

    assert ok.status_code == 200
    assert {"statistics", "correlation", "series", "series_date_column"} <= set(ok.json())
    assert bad.status_code == 422


def test_naive_baseline_uses_previous_observed_value_and_split_dates_are_ordered():
    frame = seasonal_frame()
    prepared = prepare_sequences(frame, make_request())

    expected_first_naive = frame["value"].iloc[
        int(len(frame) * 0.85) - 1
    ]
    assert prepared["test_naive"][0] == expected_first_naive
    dates = prepared["split_dates"]
    assert dates["start"] < dates["train_end"] < dates["validation_start"]
    assert dates["validation_end"] < dates["test_start"] <= dates["end"]
    series_dates = [item["date"] for item in prepared["full_series"]]
    assert dates["train_end"] in series_dates and dates["test_start"] in series_dates


def test_forecast_is_recursive_dated_and_in_original_units():
    request = make_request()
    data = prepare_sequences(seasonal_frame(), request)

    forecast, note = forecast_future(FakeModel(), data, request, 5)

    assert len(forecast) == 5
    low, high = data["target_scaler"].data_min_[0], data["target_scaler"].data_max_[0]
    assert all(abs(item["predicted"] - (low + 0.5 * (high - low))) < 1e-6 for item in forecast)
    assert forecast[0]["date"] == str(data["dates_all"][-1] + pd.Timedelta(hours=1))
    assert forecast[-1]["date"] == str(data["dates_all"][-1] + pd.Timedelta(hours=5))
    assert note


def test_forecast_repeats_other_predictors_cyclically_instead_of_freezing_them():
    request = make_request(feature_columns=["value", "other"], lookback=6)
    data = prepare_sequences(seasonal_frame(), request)
    seen = []

    class SpyModel(FakeModel):
        def __call__(self, window, training=False):
            seen.append(np.array(window[0]))
            return super().__call__(window, training)

    forecast_future(SpyModel(), data, request, 4)

    cycle = np.array(data["last_window"])
    # Step k+1 sees, in its newest row, the "other" value from k rows into the original window.
    for step in range(1, 4):
        assert np.isclose(seen[step][-1][1], cycle[step - 1][1], atol=1e-6)
    assert len({round(float(row[-1][1]), 6) for row in seen[1:]}) > 1


def test_forecast_requires_target_among_predictors_and_zero_steps_is_empty():
    request = make_request(feature_columns=["other"])
    data = prepare_sequences(seasonal_frame(), request)

    forecast, note = forecast_future(FakeModel(), data, request, 5)
    empty, empty_note = forecast_future(FakeModel(), data, make_request(), 0)

    assert forecast == [] and "target" in note
    assert empty == [] and empty_note is None


def wait_for(client, job_id, statuses, timeout=120):
    deadline = time.monotonic() + timeout
    job = None
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.1)
    raise AssertionError(job)


def upload_series(client):
    content = seasonal_frame(120).to_csv(index=False)
    return client.post("/api/v1/datasets", files={"file": ("s.csv", content, "text/csv")}).json()["id"]


def train_payload(**overrides):
    payload = {
        "date_column": "timestamp",
        "target_column": "value",
        "feature_columns": ["value", "other"],
        "lookback": 6,
        "epochs": 1,
        "batch_size": 8,
        "units": 4,
        "dropout": 0,
        "patience": 1,
        "train_ratio": 0.6,
        "validation_ratio": 0.2,
        "test_ratio": 0.2,
    }
    payload.update(overrides)
    return payload


def test_training_results_include_baseline_series_split_and_forecast():
    with TestClient(app) as client:
        dataset_id = upload_series(client)
        job_id = client.post(
            f"/api/v1/datasets/{dataset_id}/train", json=train_payload(forecast_steps=4)
        ).json()["job_id"]
        wait_for(client, job_id, {"completed", "failed"})
        results = client.get(f"/api/v1/jobs/{job_id}/results").json()
        csv_text = client.get(f"/api/v1/jobs/{job_id}/predictions.csv").text

    assert set(results["baseline_metrics"]) == {"mae", "mse", "rmse", "r2"}
    assert len(results["forecast"]) == 4
    assert results["forecast_note"]
    assert all("naive" in row for row in results["predictions"])
    assert len(results["series"]) >= len(results["predictions"])
    assert {"train_end", "test_start"} <= set(results["split_dates"])
    assert csv_text.splitlines()[0] == "date,actual,predicted,naive"


def test_training_without_forecast_steps_returns_empty_forecast():
    with TestClient(app) as client:
        dataset_id = upload_series(client)
        job_id = client.post(f"/api/v1/datasets/{dataset_id}/train", json=train_payload()).json()["job_id"]
        wait_for(client, job_id, {"completed", "failed"})
        results = client.get(f"/api/v1/jobs/{job_id}/results").json()

    assert results["forecast"] == [] and results["forecast_note"] is None


def test_cancel_stops_training_and_hides_results():
    with TestClient(app) as client:
        dataset_id = upload_series(client)
        job_id = client.post(
            f"/api/v1/datasets/{dataset_id}/train", json=train_payload(epochs=300, patience=50)
        ).json()["job_id"]
        cancelled = client.post(f"/api/v1/jobs/{job_id}/cancel")
        job = wait_for(client, job_id, {"cancelled", "completed", "failed"})
        results = client.get(f"/api/v1/jobs/{job_id}/results")
        missing = client.post("/api/v1/jobs/unknown/cancel")

    assert cancelled.status_code == 200
    assert job["status"] == "cancelled"
    assert results.status_code == 409
    assert missing.status_code == 404


def test_cancel_after_completion_is_rejected():
    with TestClient(app) as client:
        dataset_id = upload_series(client)
        job_id = client.post(f"/api/v1/datasets/{dataset_id}/train", json=train_payload()).json()["job_id"]
        wait_for(client, job_id, {"completed", "failed"})
        response = client.post(f"/api/v1/jobs/{job_id}/cancel")

    assert response.status_code == 409
