from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TrainRequest(BaseModel):
    date_column: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    feature_columns: list[str] = Field(min_length=1)
    lookback: int = Field(default=24, ge=2, le=500)
    epochs: int = Field(default=50, ge=1, le=300)
    batch_size: int = Field(default=32, ge=1, le=512)
    units: int = Field(default=64, ge=4, le=512)
    dropout: float = Field(default=0.2, ge=0, le=0.8)
    train_ratio: float = Field(default=0.7, gt=0.5, lt=0.9)
    validation_ratio: float = Field(default=0.15, gt=0, lt=0.4)
    test_ratio: float = Field(default=0.15, gt=0, lt=0.4)
    missing_strategy: Literal["drop", "forward_fill", "interpolate"] = "forward_fill"
    patience: int = Field(default=8, ge=1, le=50)
    forecast_steps: int = Field(default=0, ge=0, le=200)

    @model_validator(mode="after")
    def validate_ratios_and_columns(self) -> "TrainRequest":
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1) > 0.001:
            raise ValueError("Las proporciones de entrenamiento, validación y prueba deben sumar 1.")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("No se pueden repetir variables predictoras.")
        if self.date_column == self.target_column or self.date_column in self.feature_columns:
            raise ValueError("La columna temporal no puede ser target ni variable predictora.")
        return self
