from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from app.schemas import TrainRequest
from app.services.data_service import DatasetValidationError


def _impute_and_split(frame: pd.DataFrame, request: TrainRequest):
    selected = frame[[request.date_column, *request.feature_columns, request.target_column]].copy()
    selected[request.date_column] = pd.to_datetime(
        selected[request.date_column], errors="coerce", format="mixed"
    )
    selected = selected.dropna(subset=[request.date_column]).sort_values(request.date_column)
    for column in [*request.feature_columns, request.target_column]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    if request.missing_strategy == "drop":
        selected = selected.dropna(subset=[*request.feature_columns, request.target_column])
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
    numeric_columns = [*request.feature_columns, request.target_column]
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
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "train_end": train_end,
        "validation_end": validation_end,
    }
    if any(len(splits[key]) == 0 for key in ("X_train", "X_validation", "X_test")):
        raise DatasetValidationError("Cada segmento debe contener al menos una secuencia.")
    return splits


def train_model(
    frame: pd.DataFrame,
    request: TrainRequest,
    progress: Callable[[dict], None],
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
    predicted_scaled = model.predict(data["X_test"], verbose=0)
    predicted = data["target_scaler"].inverse_transform(predicted_scaled).ravel()
    actual = data["test_actual"]
    mse = mean_squared_error(actual, predicted)
    metrics = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
    }
    losses = {
        "train": [float(value) for value in history.history.get("loss", [])],
        "validation": [float(value) for value in history.history.get("val_loss", [])],
    }
    predictions = [
        {"date": date, "actual": float(real), "predicted": float(pred)}
        for date, real, pred in zip(data["test_dates"], actual, predicted)
    ]
    return {
        "metrics": metrics,
        "losses": losses,
        "predictions": predictions,
        "summary": (
            f"Modelo LSTM entrenado con {len(data['X_train'])} secuencias de entrenamiento. "
            f"Evaluado sobre {len(predictions)} observaciones cronológicas de prueba."
        ),
    }
