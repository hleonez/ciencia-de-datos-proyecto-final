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
        # A column with no values at all (e.g. an empty "Message" column) is not usable as numeric.
        is_numeric = len(non_null) > 0 and (pd.api.types.is_numeric_dtype(series) or numeric_ratio >= 0.8)
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


def apply_filters(frame: pd.DataFrame, filters: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Keep rows matching every filter. match is "exact" or "prefix" (value is the leading word)."""
    selected = frame
    normalized: list[tuple[str, str, str]] = []
    for item in filters:
        if len(item) == 2:
            column, value = item  # type: ignore[misc]
            match = "exact"
        else:
            column, value, match = item
        normalized.append((column, value, match))
    for column, value, match in normalized:
        if column not in selected.columns:
            raise DatasetValidationError(f"No existe la columna de filtro «{column}».")
        text = selected[column].astype("string").fillna("")
        if match == "prefix":
            selected = selected[text.str.startswith(str(value))]
        else:
            selected = selected[text == str(value)]
    if filters and selected.empty:
        raise DatasetValidationError("Esa combinación de filtros no deja ninguna fila.")
    return selected


def category_options(frame: pd.DataFrame, limit: int = 200) -> list[dict]:
    """Text columns that can split one table into separate time series, with their values."""
    categories = []
    for item in inspect_columns(frame):
        if item["data_type"] != "text" or item["unique_count"] < 2:
            continue
        counts = frame[item["name"]].astype("string").fillna("").value_counts()
        counts = counts[counts.index != ""]
        choices = [
            {"value": str(value), "count": int(count), "match": "exact"}
            for value, count in counts.head(limit).items()
        ]
        # Labels such as "Annual Average 2012" each occur once. The first word
        # ("Annual", "Summer") is the grouping that still leaves enough rows.
        if not counts.empty and int(counts.max()) < 12:
            words = frame[item["name"]].astype("string").str.extract(r"^([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)", expand=False)
            grouped = frame.groupby(words.fillna(""), dropna=False).size()
            prefixes = [
                {"value": str(word), "count": int(count), "match": "prefix"}
                for word, count in grouped.items()
                if word and int(count) >= 12
            ]
            choices = prefixes + choices
        categories.append({
            "name": item["name"],
            "unique_count": int(len(counts)),
            "truncated": int(len(counts)) > limit,
            "options": choices,
        })
    return categories


def describe_series(
    frame: pd.DataFrame,
    date_column: str | None = None,
    filters: list[tuple[str, str, str]] | None = None,
) -> dict:
    """Say whether the (filtered) table is one series per date and how long a lookback can be."""
    selected = apply_filters(frame, filters or [])
    columns = inspect_columns(selected)
    date_names = [item["name"] for item in columns if item["data_type"] == "date"]
    active = date_column if date_column in date_names else (date_names[0] if date_names else None)
    unique_dates = 0
    if active:
        unique_dates = int(pd.to_datetime(selected[active], errors="coerce", format="mixed").nunique(dropna=True))
    rows = int(len(selected))
    single_series = rows > 0 and unique_dates > 0 and unique_dates >= rows * 0.9
    # Training needs a lookback of at least 2, ten extra rows, and a training slice longer than the lookback.
    max_lookback = min(int(rows * 0.7) - 1, rows - 10)
    trainable = bool(single_series and max_lookback >= 2)
    suggested = max(2, min(max_lookback, max(2, rows // 6))) if trainable else None
    scope = ""
    if filters:
        described = []
        for item in filters:
            column, value = item[0], item[1]
            match = item[2] if len(item) > 2 else "exact"
            described.append(f"{column} empieza por {value}" if match == "prefix" else f"{column} = {value}")
        scope = "Filtro: " + "; ".join(described) + ". "
    if not active:
        story = scope + "No hay una columna de fecha, así que no se puede armar una serie temporal."
    elif not single_series:
        story = (
            f"{scope}Hay {rows} filas y solo {unique_dates} fechas distintas. "
            "Varias observaciones comparten la misma fecha (otro indicador, otra zona u otra categoría). "
            "Elige un valor en las columnas de texto hasta que quede una fila por fecha."
        )
    elif not trainable:
        story = (
            f"{scope}Ya es una sola serie, pero solo tiene {rows} observaciones. "
            "Hacen falta al menos 12 en una sola serie para entrenar y evaluar sin usar el futuro."
        )
    else:
        short = " La serie es corta, así que el resultado orienta sobre la tendencia; no es un pronóstico fino." if rows < 30 else ""
        story = (
            f"{scope}Una sola serie de {rows} observaciones ordenadas por {active}. "
            f"El modelo mirará las {suggested} observaciones anteriores para predecir la siguiente, "
            "y se evaluará solo en las últimas fechas, que no entran al entrenamiento."
            f"{short}"
        )
    return {
        "row_count": rows,
        "unique_dates": unique_dates,
        "date_column": active,
        "single_series": single_series,
        "trainable": trainable,
        "suggested_lookback": suggested,
        "minimum_rows": 12,
        "story": story,
        "categories": category_options(selected),
    }


def explore_dataset(
    frame: pd.DataFrame,
    selected_column: str | None = None,
    date_column: str | None = None,
    period: int | None = None,
    filters: list[tuple[str, str, str]] | None = None,
) -> dict:
    frame = apply_filters(frame, filters or [])
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
