import pandas as pd
import pytest

from app.services.data_service import DatasetValidationError, dataset_summary, explore_dataset, inspect_columns, read_dataset


def test_inspect_columns_detects_dates_numeric_and_text():
    frame = pd.DataFrame({
        "time": ["2026-01-01 08:00", "2026-01-01 09:00"],
        "value": [1.5, 2.5],
        "label": ["north", "south"],
    })

    columns = {item["name"]: item["data_type"] for item in inspect_columns(frame)}

    assert columns == {"time": "date", "value": "numeric", "label": "text"}


def test_summary_and_eda_return_quality_and_descriptive_statistics():
    frame = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=3, freq="h"),
        "value": [1.0, None, 3.0],
        "other": [2.0, 4.0, 6.0],
    })

    summary = dataset_summary(frame)
    eda = explore_dataset(frame, "other")

    assert summary["row_count"] == 3
    assert summary["null_count"] == 1
    assert len(summary["preview"]) == 3
    assert eda["statistics"]["other"]["mean"] == 4
    assert eda["statistics"]["value"]["null_count"] == 1
    assert len(eda["series"]) == 3


def test_eda_includes_numeric_values_stored_as_text():
    frame = pd.DataFrame({
        "time": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "measurement": ["1.5", "2.5", "3.5"],
    })

    eda = explore_dataset(frame, "measurement")

    assert eda["statistics"]["measurement"]["mean"] == 2.5
    assert len(eda["series"]) == 3


def test_csv_headerless_data_is_rejected(tmp_path):
    path = tmp_path / "headerless.csv"
    path.write_text(
        "2026-01-01 08:00,10,20\n2026-01-01 09:00,11,22\n2026-01-01 10:00,12,24\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="encabezados"):
        read_dataset(path)


def test_column_metadata_reports_values_with_inconsistent_numeric_types():
    frame = pd.DataFrame({
        "measurement": ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "bad"],
    })

    column = inspect_columns(frame)[0]

    assert column["data_type"] == "numeric"
    assert column["invalid_count"] == 1
