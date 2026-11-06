import csv
from pathlib import Path

from zipfile import BadZipFile

import numpy as np
import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException


class DatasetValidationError(ValueError):
    pass


def read_dataset(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                sample = source.read(65536)
            try:
                has_header = csv.Sniffer().has_header(sample)
            except csv.Error as exc:
                raise DatasetValidationError(
                    "No se pudo identificar una fila de encabezados en el CSV."
                ) from exc
            if not has_header:
                raise DatasetValidationError(
                    "No se detectó una fila de encabezados. Añade los nombres de las columnas en la primera fila."
                )
            frame = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="error")
        else:
            frame = pd.read_excel(path, engine="openpyxl")
    except DatasetValidationError:
        raise
    except (
        UnicodeDecodeError,
        pd.errors.ParserError,
        ValueError,
        OSError,
        BadZipFile,
        InvalidFileException,
    ) as exc:
        raise DatasetValidationError(
            "No se pudo leer el archivo. Verifica que sea un CSV UTF-8 o un libro XLSX válido."
        ) from exc
    if frame.empty or len(frame.columns) == 0:
        raise DatasetValidationError("El archivo debe incluir encabezados y al menos un registro.")
    frame.columns = frame.columns.map(lambda column: str(column).strip())
    if any(not str(column).strip() or str(column).startswith("Unnamed:") for column in frame.columns):
        raise DatasetValidationError("Todos los encabezados deben tener un nombre.")
    if frame.columns.duplicated().any():
        raise DatasetValidationError("Los nombres de las columnas deben ser únicos.")
    if len(frame.columns) > 200:
        raise DatasetValidationError("El archivo supera el límite de 200 columnas.")
    return frame


def inspect_columns(frame: pd.DataFrame) -> list[dict]:
    columns: list[dict] = []
    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        date_ratio = 0.0
        parsed_dates = None
        if not pd.api.types.is_numeric_dtype(series) and len(non_null):
            parsed_dates = pd.to_datetime(non_null, errors="coerce", format="mixed")
            date_ratio = float(parsed_dates.notna().mean())
        is_date = date_ratio >= 0.8
        numeric_ratio = 0.0
        parsed_numeric = None
        if not pd.api.types.is_numeric_dtype(series) and len(non_null):
            parsed_numeric = pd.to_numeric(non_null, errors="coerce")
            numeric_ratio = float(parsed_numeric.notna().mean())
        is_numeric = pd.api.types.is_numeric_dtype(series) or numeric_ratio >= 0.8
        if is_date:
            data_type = "date"
            invalid_count = int(parsed_dates.isna().sum())
        elif is_numeric:
            data_type = "numeric"
            invalid_count = int(parsed_numeric.isna().sum()) if parsed_numeric is not None else 0
        else:
            data_type = "text"
            invalid_count = 0
        columns.append({
            "name": str(column),
            "data_type": data_type,
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "invalid_count": invalid_count,
        })
    return columns


def dataset_summary(frame: pd.DataFrame) -> dict:
    columns = inspect_columns(frame)
    numeric_columns = [item["name"] for item in columns if item["data_type"] == "numeric"]
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": columns,
        "null_count": int(frame.isna().sum().sum()),
        "duplicate_count": int(frame.duplicated().sum()),
        "numeric_columns": numeric_columns,
        "date_columns": [
            item["name"] for item in columns if item["data_type"] == "date"
        ],
        "preview": frame.head(10).replace({np.nan: None}).to_dict(orient="records"),
    }


def explore_dataset(frame: pd.DataFrame, selected_column: str | None = None) -> dict:
    columns = inspect_columns(frame)
    numeric_columns = [item["name"] for item in columns if item["data_type"] == "numeric"]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    stats = {}
    for column in numeric.columns:
        values = numeric[column]
        stats[str(column)] = {
            "mean": _finite(values.mean()),
            "median": _finite(values.median()),
            "std": _finite(values.std()),
            "min": _finite(values.min()),
            "max": _finite(values.max()),
            "null_count": int(values.isna().sum()),
        }
    correlation = numeric.corr().replace({np.nan: None}).to_dict()

    date_candidates = [item["name"] for item in columns if item["data_type"] == "date"]
    series = []
    if selected_column and selected_column in numeric.columns and date_candidates:
        date_column = date_candidates[0]
        dates = pd.to_datetime(frame[date_column], errors="coerce", format="mixed")
        chart_frame = pd.DataFrame({"date": dates, "value": numeric[selected_column]})
        chart_frame = chart_frame.dropna().sort_values("date")
        if len(chart_frame) > 500:
            positions = np.linspace(0, len(chart_frame) - 1, 500, dtype=int)
            chart_frame = chart_frame.iloc[positions]
        series = [
            {"date": date.isoformat(), "value": float(value)}
            for date, value in zip(chart_frame["date"], chart_frame["value"])
        ]
    return {
        "statistics": stats,
        "correlation": correlation,
        "duplicate_count": int(frame.duplicated().sum()),
        "null_count": int(frame.isna().sum().sum()),
        "series": series,
        "series_date_column": date_candidates[0] if date_candidates else None,
    }


def _finite(value):
    if pd.isna(value):
        return None
    return float(value)
