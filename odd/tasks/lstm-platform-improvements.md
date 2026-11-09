# LSTM platform improvements (phases A-E)

Objective: make the app show a trustworthy LSTM training/results/forecast/EDA experience.
Branch: feat/lstm-platform-improvements. Delivery: single-pr. Test runner: `backend/.venv/Scripts/python -m pytest` and `npm run build` in `frontend/`.
Engram mirror: pending (project not registered in Engram store).

## Specs

- S1: "Incluir el target como predictor por defecto" (target is a default predictor), the split label must follow the real ratios, and the EDA date column must be selectable. (Phase A)
- S2: Compare the LSTM against a naive baseline ("mañana igual que hoy"), show the full series with train/validation/test zones marked, and show a residuals chart. (Phase B)
- S3: Forecast N future steps ("pronosticar N pasos") recursively; API returns the forecast and the chart shows it. Assumption: other predictors are held at their last observed value; forecast needs the target among the predictors. (Phase C)
- S4: EDA adds histograms, boxplot statistics, trend/seasonality decomposition and autocorrelation (ACF). (Phase D)
- S5: Sample dataset button, cancel a running training, and an in-session history of runs to compare. (Phase E)
- S6: Every phase keeps existing behavior (`/datasets`, `/eda`, `/train`, `/jobs`, `predictions.csv` header still contains `date,actual,predicted`) and has pytest coverage.

## Tasks

- T1 (S1,S6) backend+frontend fixes - route: inline - status: pending
- T2 (S2,S3,S6) training service: baseline, full series, forecast - route: inline - status: pending
- T3 (S4,S6) EDA service additions - route: inline - status: pending
- T4 (S5,S6) cancel endpoint and job state - route: inline - status: pending
- T5 (S1-S5) frontend: types, api, new charts and sections - route: inline - status: pending

## Log

- L1 (2026-11-08): "bueno vamos a realizar esas mejoras" (plan A-E: corregir, evaluación honesta, pronóstico, EDA ampliado, pulido)
