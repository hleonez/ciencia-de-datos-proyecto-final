import time

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


def test_upload_eda_and_delete_dataset():
    content = (
        "timestamp,value\n"
        "2026-01-01 08:00,1.5\n"
        "2026-01-01 09:00,2.0\n"
        "2026-01-01 10:00,2.5\n"
    )
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/v1/datasets",
            files={"file": ("sample.csv", content, "text/csv")},
        )
        assert uploaded.status_code == 201
        dataset = uploaded.json()
        assert dataset["row_count"] == 3
        assert dataset["validation_status"] == "valid"
        assert len(dataset["preview"]) == 3

        eda = client.get(
            f"/api/v1/datasets/{dataset['id']}/eda",
            params={"column": "value"},
        )
        assert eda.status_code == 200
        assert eda.json()["statistics"]["value"]["mean"] == 2.0
        assert len(eda.json()["series"]) == 3

        deleted = client.delete(f"/api/v1/datasets/{dataset['id']}")
        assert deleted.status_code == 204
        assert client.get(f"/api/v1/datasets/{dataset['id']}").status_code == 404


def test_upload_rejects_unsupported_file_extensions():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/datasets",
            files={"file": ("sample.txt", "not a dataset", "text/plain")},
        )

    assert response.status_code == 415


def test_training_job_completes_and_returns_real_predictions():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=60, freq="h"),
        "sensor": range(60),
        "target": [value * 2 for value in range(60)],
    })
    with TestClient(app) as client:
        uploaded = client.post(
            "/api/v1/datasets",
            files={"file": ("series.csv", frame.to_csv(index=False), "text/csv")},
        )
        assert uploaded.status_code == 201
        dataset_id = uploaded.json()["id"]
        started = client.post(
            f"/api/v1/datasets/{dataset_id}/train",
            json={
                "date_column": "timestamp",
                "target_column": "target",
                "feature_columns": ["sensor"],
                "lookback": 4,
                "epochs": 1,
                "batch_size": 8,
                "units": 4,
                "dropout": 0,
                "patience": 1,
                "train_ratio": 0.6,
                "validation_ratio": 0.2,
                "test_ratio": 0.2,
            },
        )
        assert started.status_code == 202
        job_id = started.json()["job_id"]

        deadline = time.monotonic() + 120
        job = None
        while time.monotonic() < deadline:
            response = client.get(f"/api/v1/jobs/{job_id}")
            assert response.status_code == 200
            job = response.json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.1)

        assert job is not None and job["status"] == "completed", job
        results = client.get(f"/api/v1/jobs/{job_id}/results")
        assert results.status_code == 200
        assert set(results.json()["metrics"]) == {"mae", "mse", "rmse", "r2"}
        assert len(results.json()["predictions"]) == 12
        csv_response = client.get(f"/api/v1/jobs/{job_id}/predictions.csv")
        assert csv_response.status_code == 200
        assert "date,actual,predicted" in csv_response.text
