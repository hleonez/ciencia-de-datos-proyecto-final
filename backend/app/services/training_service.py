from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from app.schemas import TrainRequest
from app.services.data_service import DatasetValidationError

MAX_FULL_SERIES_POINTS = 1000


class TrainingCancelled(Exception):
    """Raised when a training job is stopped by the user."""


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mse = mean_squared_error(actual, predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
    }


def _full_series(prepared: pd.DataFrame, request: TrainRequest, train_end: int, validation_end: int) -> list[dict]:
    """Full target series for charting; train/validation are thinned, test is kept whole."""
    total = len(prepared)
    keep = set(range(validation_end, total))
    stride = max(1, int(np.ceil(validation_end / MAX_FULL_SERIES_POINTS)))
    keep.update(range(0, validation_end, stride))
    keep.update({train_end - 1, train_end, validation_end - 1, validation_end})
    keep = {index for index in keep if 0 <= index < total}
    dates = prepared[request.date_column].astype(str).tolist()
    values = prepared[request.target_column].to_numpy(dtype=float)
    return [{"date": dates[index], "actual": float(values[index])} for index in sorted(keep)]


def forecast_future(model, data: dict, request: TrainRequest, steps: int) -> tuple[list[dict], str | None]:
    """Recursive multi-step forecast. Other predictors are held at their last observed value."""
    if steps <= 0:
        return [], None
    if request.target_column not in request.feature_columns:
        return [], "El pronóstico requiere incluir el target entre las variables predictoras."
    dates = data["dates_all"]
    deltas = dates.to_series().diff().dropna()
    step = deltas.median() if len(deltas) else pd.NaT
    if pd.isna(step) or step <= pd.Timedelta(0):
        return [], "No se pudo inferir el intervalo temporal para pronosticar."

    target_index = request.feature_columns.index(request.target_column)
    window = np.array(data["last_window"], dtype=np.float32)
    scaled_predictions = []
    for _ in range(steps):
        output = model(window[np.newaxis, ...], training=False)
        value = float(np.asarray(output).reshape(-1)[0])
        scaled_predictions.append(value)
        next_row = window[-1].copy()
        next_row[target_index] = value
        window = np.vstack([window[1:], next_row])

    actual = data["target_scaler"].inverse_transform(
        np.asarray(scaled_predictions, dtype=float).reshape(-1, 1)
    ).ravel()
    last_date = dates[-1]
    forecast = [
        {"date": str(last_date + step * (index + 1)), "predicted": float(value)}
        for index, value in enumerate(actual)
    ]
    note = (
        "Pronóstico recursivo: las demás variables predictoras se mantienen en su último valor observado "
        "y el intervalo temporal se estima con la mediana de las diferencias de fecha."
        if len(request.feature_columns) > 1
        else "Pronóstico recursivo: el intervalo temporal se estima con la mediana de las diferencias de fecha."
    )
    return forecast, note


def _impute_and_split(frame: pd.DataFrame, request: TrainRequest):
    # The target may also be a predictor (its own history); keep each column once.
    numeric_columns = list(dict.fromkeys([*request.feature_columns, request.target_column]))
    selected = frame[[request.date_column, *numeric_columns]].copy()
    selected[request.date_column] = pd.to_datetime(
        selected[request.date_column], errors="coerce", format="mixed"
    )
    selected = selected.dropna(subset=[request.date_column]).sort_values(request.date_column)
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    if request.missing_strategy == "drop":
        selected = selected.dropna(subset=numeric_columns)
    if len(selected) < request.lookback + 10:
        raise DatasetValidationError(
            f"Se requieren al menos {request.lookback + 10} filas válidas para el lookback seleccionado."
        )

    train_end = int(len(selected) * request.train_ratio)
    validation_end = int(len(selected) * (request.train_ratio + request.validation_ratio))
    if train_end <= request.lookback or validation_end <= train_end or validation_end >= len(selected):
        raise DatasetValidationError("Las proporciones y el lookback no dejan suficientes filas por segmento.")

    train = selected.iloc[:train_end].copy()
    validation = selected.iloc[train_end:validation_end].copy()
    test = selected.iloc[validation_end:].copy()
    if request.missing_strategy == "forward_fill":
        train = train.ffill()
        train_medians = train[numeric_columns].median()
        if train_medians.isna().any():
            raise DatasetValidationError("No se pueden imputar columnas sin valores numéricos en entrenamiento.")
        train[numeric_columns] = train[numeric_columns].fillna(train_medians)
        seed = train.iloc[-1:][numeric_columns]
        evaluation = pd.concat([seed, validation, test]).ffill().iloc[1:]
        evaluation[numeric_columns] = evaluation[numeric_columns].fillna(train_medians)
        validation = evaluation.iloc[:len(validation)].copy()
        test = evaluation.iloc[len(validation):].copy()
    elif request.missing_strategy == "interpolate":
        train[numeric_columns] = train[numeric_columns].interpolate(
            method="linear", limit_direction="both"
        )
        medians = train[numeric_columns].median()
        if medians.isna().any():
            raise DatasetValidationError("No se pueden imputar columnas sin valores numéricos en entrenamiento.")
        evaluation = pd.concat([train.iloc[-1:][numeric_columns], validation, test]).ffill().iloc[1:]
        evaluation[numeric_columns] = evaluation[numeric_columns].fillna(medians)
        validation = evaluation.iloc[:len(validation)].copy()
        test = evaluation.iloc[len(validation):].copy()
    else:
        train = train.dropna(subset=numeric_columns)
        validation = validation.dropna(subset=numeric_columns)
        test = test.dropna(subset=numeric_columns)

    if train.empty or validation.empty or test.empty:
        raise DatasetValidationError("El tratamiento de valores faltantes dejó vacío un segmento.")
    prepared = pd.concat([train, validation, test], ignore_index=True)
    return prepared, len(train), len(train) + len(validation)


def prepare_sequences(frame: pd.DataFrame, request: TrainRequest) -> dict:
    prepared, train_end, validation_end = _impute_and_split(frame, request)
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    train_features = prepared.iloc[:train_end][request.feature_columns]
    train_target = prepared.iloc[:train_end][[request.target_column]]
    feature_scaler.fit(train_features)
    target_scaler.fit(train_target)
    features = feature_scaler.transform(prepared[request.feature_columns])
    target = target_scaler.transform(prepared[[request.target_column]]).ravel()

    sequences, labels, label_indices = [], [], []
    for index in range(request.lookback, len(prepared)):
        sequences.append(features[index - request.lookback:index])
        labels.append(target[index])
        label_indices.append(index)
    sequences = np.asarray(sequences, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    label_indices = np.asarray(label_indices)
    train_mask = label_indices < train_end
    validation_mask = (label_indices >= train_end) & (label_indices < validation_end)
    test_mask = label_indices >= validation_end
    splits = {
        "X_train": sequences[train_mask],
        "y_train": labels[train_mask],
        "X_validation": sequences[validation_mask],
        "y_validation": labels[validation_mask],
        "X_test": sequences[test_mask],
        "y_test": labels[test_mask],
        "test_dates": prepared.iloc[label_indices[test_mask]][request.date_column].astype(str).tolist(),
        "test_actual": target_scaler.inverse_transform(labels[test_mask].reshape(-1, 1)).ravel(),
        # Naive baseline: each test value is predicted with the previous observed target value.
        "test_naive": prepared[request.target_column].to_numpy(dtype=float)[label_indices[test_mask] - 1],
        "full_series": _full_series(prepared, request, train_end, validation_end),
        "dates_all": pd.DatetimeIndex(prepared[request.date_column]),
        "last_window": features[-request.lookback:],
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "train_end": train_end,
        "validation_end": validation_end,
        "split_dates": {
            "start": str(prepared[request.date_column].iloc[0]),
            "train_end": str(prepared[request.date_column].iloc[train_end - 1]),
            "validation_start": str(prepared[request.date_column].iloc[train_end]),
            "validation_end": str(prepared[request.date_column].iloc[validation_end - 1]),
            "test_start": str(prepared[request.date_column].iloc[validation_end]),
            "end": str(prepared[request.date_column].iloc[-1]),
        },
    }
    if any(len(splits[key]) == 0 for key in ("X_train", "X_validation", "X_test")):
        raise DatasetValidationError("Cada segmento debe contener al menos una secuencia.")
    return splits


def train_model(
    frame: pd.DataFrame,
    request: TrainRequest,
    progress: Callable[[dict], None],
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow no está instalado. Instala las dependencias de backend/requirements.txt."
        ) from exc

    data = prepare_sequences(frame, request)
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(request.lookback, len(request.feature_columns))),
        tf.keras.layers.LSTM(request.units),
        tf.keras.layers.Dropout(request.dropout),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.train_losses = []
            self.validation_losses = []

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if should_stop is not None and should_stop():
                self.model.stop_training = True
            self.train_losses.append(float(logs.get("loss", 0)))
            self.validation_losses.append(float(logs.get("val_loss", 0)))
            progress({
                "current_epoch": epoch + 1,
                "total_epochs": request.epochs,
                "train_loss": float(logs.get("loss", 0)),
                "validation_loss": float(logs.get("val_loss", 0)),
                "loss_history": list(self.train_losses),
                "validation_loss_history": list(self.validation_losses),
            })

    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_validation"], data["y_validation"]),
        epochs=request.epochs,
        batch_size=request.batch_size,
        shuffle=False,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=request.patience, restore_best_weights=True
            ),
            ProgressCallback(),
        ],
        verbose=0,
    )
    if should_stop is not None and should_stop():
        raise TrainingCancelled("El entrenamiento fue cancelado.")
    predicted_scaled = model.predict(data["X_test"], verbose=0)
    predicted = data["target_scaler"].inverse_transform(predicted_scaled).ravel()
    actual = data["test_actual"]
    naive = data["test_naive"]
    metrics = _metrics(actual, predicted)
    baseline_metrics = _metrics(actual, naive)
    forecast, forecast_note = forecast_future(model, data, request, request.forecast_steps)
    losses = {
        "train": [float(value) for value in history.history.get("loss", [])],
        "validation": [float(value) for value in history.history.get("val_loss", [])],
    }
    predictions = [
        {"date": date, "actual": float(real), "predicted": float(pred), "naive": float(base)}
        for date, real, pred, base in zip(data["test_dates"], actual, predicted, naive)
    ]
    return {
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "losses": losses,
        "predictions": predictions,
        "series": data["full_series"],
        "split_dates": data["split_dates"],
        "forecast": forecast,
        "forecast_note": forecast_note,
        "summary": (
            f"Modelo LSTM entrenado con {len(data['X_train'])} secuencias de entrenamiento. "
            f"Evaluado sobre {len(predictions)} observaciones cronológicas de prueba."
        ),
    }
