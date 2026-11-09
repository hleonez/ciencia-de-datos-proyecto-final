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


def _histogram(values: pd.Series, max_bins: int = 20) -> list[dict]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return []
    bins = max(1, min(max_bins, len(np.unique(clean))))
    counts, edges = np.histogram(clean, bins=bins)
    return [
        {"start": float(edges[i]), "end": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]


def _boxplot(values: pd.Series) -> dict | None:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return None
    q1, median, q3 = (float(v) for v in np.percentile(clean, [25, 50, 75]))
    iqr = q3 - q1
    low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = clean[(clean >= low_fence) & (clean <= high_fence)]
    outliers = clean[(clean < low_fence) | (clean > high_fence)]
    return {
        "min": float(clean.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(clean.max()),
        "whisker_low": float(inside.min()),
        "whisker_high": float(inside.max()),
        "outlier_count": int(len(outliers)),
        "outliers": [float(v) for v in outliers[:50]],
    }


def _acf(values: np.ndarray, max_lag: int) -> list[float]:
    centered = values - values.mean()
    denominator = float((centered ** 2).sum())
    if denominator == 0 or len(values) < 3:
        return []
    max_lag = min(max_lag, len(values) - 1)
    return [
        float((centered[: len(values) - lag] * centered[lag:]).sum() / denominator)
        for lag in range(max_lag + 1)
    ]


def _detect_period(acf: list[float]) -> int | None:
    """Pick the strongest local autocorrelation peak above 0.2 as the seasonal period."""
    best_lag, best_value = None, 0.2
    for lag in range(2, len(acf) - 1):
        if acf[lag] > acf[lag - 1] and acf[lag] >= acf[lag + 1] and acf[lag] > best_value:
            best_lag, best_value = lag, acf[lag]
    return best_lag


def _decompose(values: np.ndarray, period: int) -> dict[str, list[float | None]] | None:
    """Classical additive decomposition: centered moving-average trend plus mean seasonal profile."""
    if period < 2 or len(values) < 2 * period:
        return None
    series = pd.Series(values)
    trend = series.rolling(period, center=True, min_periods=period).mean()
    detrended = series - trend
    positions = np.arange(len(values)) % period
    profile = detrended.groupby(positions).mean()
    profile = profile - profile.mean()
    seasonal = pd.Series(profile.reindex(positions).to_numpy())
    residual = series - trend - seasonal
    clean = lambda s: [None if pd.isna(v) else float(v) for v in s]  # noqa: E731
    return {
        "observed": clean(series),
        "trend": clean(trend),
        "seasonal": clean(seasonal),
        "residual": clean(residual),
    }


def explore_dataset(
    frame: pd.DataFrame,
    selected_column: str | None = None,
    date_column: str | None = None,
    period: int | None = None,
) -> dict:
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
    if date_column and date_column not in date_candidates:
        raise DatasetValidationError("La columna de fecha seleccionada no es válida.")
    active_date = date_column or (date_candidates[0] if date_candidates else None)
    series = []
    acf: list[float] = []
    detected_period = None
    decomposition = None
    observations = 0
    if selected_column and selected_column in numeric.columns and active_date:
        dates = pd.to_datetime(frame[active_date], errors="coerce", format="mixed")
        chart_frame = pd.DataFrame({"date": dates, "value": numeric[selected_column]})
        chart_frame = chart_frame.dropna().sort_values("date")
        values = chart_frame["value"].to_numpy(dtype=float)
        observations = len(values)
        acf = _acf(values, min(60, len(values) // 2)) if len(values) >= 6 else []
        detected_period = period if period and period >= 2 else _detect_period(acf)
        components = _decompose(values, detected_period) if detected_period else None
        positions = (
            np.linspace(0, len(chart_frame) - 1, 500, dtype=int)
            if len(chart_frame) > 500
            else np.arange(len(chart_frame))
        )
        chart_dates = chart_frame["date"].iloc[positions]
        series = [
            {"date": date.isoformat(), "value": float(value)}
            for date, value in zip(chart_dates, chart_frame["value"].iloc[positions])
        ]
        if components is not None:
            decomposition = {
                "period": int(detected_period),
                "points": [
                    {
                        "date": date.isoformat(),
                        **{key: components[key][position] for key in components},
                    }
                    for date, position in zip(chart_dates, positions)
                ],
            }
    return {
        "statistics": stats,
        "correlation": correlation,
        "duplicate_count": int(frame.duplicated().sum()),
        "null_count": int(frame.isna().sum().sum()),
        "series": series,
        "series_date_column": active_date,
        "date_columns": date_candidates,
        "histograms": {name: _histogram(numeric[name]) for name in numeric.columns},
        "boxplots": {name: _boxplot(numeric[name]) for name in numeric.columns},
        "acf": {
            "values": acf,
            "confidence": float(1.96 / np.sqrt(observations)) if observations else None,
        },
        "period": int(detected_period) if detected_period else None,
        "decomposition": decomposition,
    }


def _finite(value):
    if pd.isna(value):
        return None
    return float(value)
