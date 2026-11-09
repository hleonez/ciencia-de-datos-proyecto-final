# Plataforma de análisis y predicción con redes LSTM

Aplicación académica para cargar datos temporales, explorarlos y entrenar bajo demanda un modelo LSTM de regresión. El cálculo y el entrenamiento se ejecutan en el backend Python; la interfaz solo presenta los datos y resultados recibidos por la API.

## Arquitectura

```text
frontend/                 React + Vite + TypeScript + Tailwind + Recharts
  src/
    components/           Navegación, carga, tarjetas y gráficos reutilizables
    services/             Cliente HTTP tipado para la API
    types/                Contratos de datos
backend/
  app/
    api/                  Rutas y esquemas HTTP
    services/             Carga/validación, EDA, preprocesamiento y entrenamiento
    core/                 Configuración y almacenamiento temporal
    main.py               Aplicación FastAPI
  tests/                  Pruebas del procesamiento sin requerir entrenamiento
```

### Flujo de datos

1. El usuario carga CSV/XLSX. Python valida formato, encabezados, tamaño y contenido, conserva el archivo temporalmente y devuelve metadatos y una vista previa.
2. EDA calcula estadísticas descriptivas, calidad, duplicados, correlaciones y series para las columnas numéricas.
3. El usuario elige fecha, target, features y parámetros; no se inicia ningún entrenamiento al cargar el archivo.
4. El backend parsea y ordena la fecha, resuelve valores faltantes según la estrategia elegida y divide cronológicamente las filas.
5. MinMaxScaler se ajusta exclusivamente con las filas de entrenamiento. Las secuencias de validación y prueba pueden usar observaciones históricas previas como contexto, pero sus etiquetas permanecen dentro de sus respectivos tramos.
6. Un trabajo en segundo plano entrena TensorFlow/Keras con Early Stopping. El cliente consulta su estado y pérdidas mediante polling.
7. El backend desnormaliza las predicciones y calcula MAE, MSE, RMSE y R² en unidades originales. La interfaz presenta resultados y permite descargar el CSV.

### API REST (prefijo `/api/v1`)

| Método | Ruta | Propósito |
|---|---|---|
| `POST` | `/datasets` | Validar y cargar CSV/XLSX (límite configurable) |
| `GET` | `/datasets/{dataset_id}` | Metadatos, columnas, calidad y vista previa |
| `GET` | `/datasets/{dataset_id}/eda` | Estadísticas, correlaciones, serie, histogramas, boxplots, ACF y descomposición. Parámetros opcionales: `column`, `date_column`, `period`, `filters` (JSON) |
| `POST` | `/datasets/{dataset_id}/series` | Dice si los filtros dejan una sola serie por fecha, cuántas filas quedan y qué lookback cabe |
| `DELETE` | `/datasets/{dataset_id}` | Eliminar archivo y metadatos temporales |
| `POST` | `/datasets/{dataset_id}/train` | Iniciar entrenamiento asíncrono (`forecast_steps` opcional, 0–200) |
| `GET` | `/jobs/{job_id}` | Estado y progreso del entrenamiento |
| `POST` | `/jobs/{job_id}/cancel` | Cancelar un entrenamiento en cola o en curso (se detiene al terminar la época actual) |
| `GET` | `/jobs/{job_id}/results` | Métricas de la LSTM y del modelo ingenuo, predicciones, serie completa, tramos, pronóstico futuro y pérdidas |
| `GET` | `/jobs/{job_id}/predictions.csv` | Descargar predicciones (`date,actual,predicted,naive`) |

### Evaluación y pronóstico

- **Modelo ingenuo:** cada valor de prueba se predice con el valor observado anterior. Los resultados comparan MAE, RMSE y R² de ambos y explican si la LSTM aporta algo.
- **Pronóstico futuro:** recursivo, paso a paso. Requiere que el target esté entre las variables predictoras. Como el futuro de las demás variables es desconocido, estas repiten cíclicamente su última ventana observada; el intervalo temporal se estima con la mediana de las diferencias de fecha.
- **Reproducibilidad:** el entrenamiento no fija semilla, así que dos ejecuciones con la misma configuración pueden dar métricas distintas. El historial de la sesión permite compararlas.

Los errores de entrada usan respuestas HTTP 4xx con mensajes accionables. Las ejecuciones fallidas se reportan como estado `failed`, sin resultados de éxito sustitutos.

## Decisiones y límites técnicos

- La API usa almacenamiento temporal y estado de trabajos en memoria; está pensada para desarrollo/demostración con una instancia de backend. Un despliegue multiinstancia requeriría almacenamiento compartido y una cola de trabajos.
- CSV se lee con Pandas y XLSX con `openpyxl`. TensorFlow puede requerir una combinación compatible de Python y sistema operativo; instalar las dependencias del backend en un entorno virtual es recomendable.
- Se exige una columna temporal válida y al menos una variable numérica seleccionada. No se suponen nombres de columnas.
- Los datos cargados y trabajos se eliminan por solicitud o por vencimiento; no se guardan en una base permanente.

## Ejecución local

Backend, desde `backend/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --reload-dir app
```

`--reload-dir app` es importante: sin él, `uvicorn` también vigila `.venv`, se reinicia cuando TensorFlow genera archivos al importarse y borra de la memoria los datasets y trabajos en curso. Si editas código con un entrenamiento activo, el servidor espera a que termine antes de recargar.

La documentación interactiva está en `http://localhost:8000/docs`. El límite de carga y el tiempo de vida temporal se pueden configurar con `MAX_UPLOAD_SIZE_MB` y `DATASET_TTL_HOURS`.

Frontend, desde `frontend/`:

```powershell
npm install
npm run dev
```

La URL del backend se configura con `VITE_API_URL` (por defecto `http://localhost:8000/api/v1`).

## Pruebas

Desde `backend/` (con el entorno virtual activo), ejecutar `pytest`. Las pruebas cubren validación, división cronológica, prevención de fuga en escalado, API y un entrenamiento real de una época con TensorFlow sobre datos de prueba generados dentro de la suite.
