import csv
import io
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.core.config import MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB
from app.core.storage import Dataset, dataset_store
from app.schemas import TrainRequest
from app.services.data_service import (
    DatasetValidationError,
    dataset_summary,
    explore_dataset,
    read_dataset,
)
from app.services.job_service import job_manager


router = APIRouter()


def _require_dataset(dataset_id: str) -> Dataset:
    dataset = dataset_store.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="No se encontró el conjunto de datos o ya venció.")
    return dataset


@router.post("/datasets", status_code=201)
async def upload_dataset(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise HTTPException(status_code=415, detail="Formato no compatible. Carga un archivo CSV o XLSX.")
    dataset_store.cleanup_expired()
    dataset_id, path = dataset_store.create_path(suffix)
    size = 0
    try:
        with path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"El archivo supera el límite de {MAX_UPLOAD_SIZE_MB} MB.",
                    )
                target.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="El archivo está vacío.")
        frame = read_dataset(path)
        summary = dataset_summary(frame)
        dataset = Dataset(
            id=dataset_id,
            filename=filename,
            path=path,
            size_bytes=size,
            row_count=summary["row_count"],
            column_count=summary["column_count"],
            columns=summary["columns"],
            null_count=summary["null_count"],
            duplicate_count=summary["duplicate_count"],
            created_at=pd.Timestamp.now().timestamp(),
        )
        dataset_store.add(dataset)
    except DatasetValidationError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="No se pudo guardar temporalmente el archivo.") from exc
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "size_bytes": dataset.size_bytes,
        "validation_status": "valid",
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "columns": dataset.columns,
        "null_count": dataset.null_count,
        "duplicate_count": dataset.duplicate_count,
        "numeric_columns": summary["numeric_columns"],
        "date_columns": summary["date_columns"],
        "preview": summary["preview"],
    }


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    dataset = _require_dataset(dataset_id)
    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "size_bytes": dataset.size_bytes,
        "validation_status": "valid",
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "columns": dataset.columns,
        "null_count": dataset.null_count,
        "duplicate_count": dataset.duplicate_count,
    }


@router.get("/datasets/{dataset_id}/eda")
def get_eda(dataset_id: str, column: str | None = Query(default=None)):
    dataset = _require_dataset(dataset_id)
    try:
        return explore_dataset(read_dataset(dataset.path), column)
    except DatasetValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str):
    if not dataset_store.delete(dataset_id):
        raise HTTPException(status_code=404, detail="No se encontró el conjunto de datos.")
    return None


@router.post("/datasets/{dataset_id}/train", status_code=202)
def start_training(dataset_id: str, request: TrainRequest):
    dataset = _require_dataset(dataset_id)
    available = {column["name"]: column["data_type"] for column in dataset.columns}
    if request.date_column not in available or available[request.date_column] != "date":
        raise HTTPException(status_code=422, detail="Selecciona una columna temporal válida.")
    if request.target_column not in available or available[request.target_column] != "numeric":
        raise HTTPException(status_code=422, detail="El target debe ser una columna numérica.")
    if any(available.get(column) != "numeric" for column in request.feature_columns):
        raise HTTPException(status_code=422, detail="Todas las variables predictoras deben ser numéricas.")
    job_id = job_manager.start(dataset, request)
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No se encontró el trabajo de entrenamiento.")
    return job


@router.get("/jobs/{job_id}/results")
def get_job_results(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No se encontró el trabajo de entrenamiento.")
    if job["status"] == "failed":
        raise HTTPException(status_code=422, detail=job["error"] or "El entrenamiento falló.")
    results = job_manager.results(job_id)
    if results is None:
        raise HTTPException(status_code=409, detail="Los resultados estarán disponibles al terminar el entrenamiento.")
    return results


@router.get("/jobs/{job_id}/predictions.csv")
def download_predictions(job_id: str):
    results = job_manager.results(job_id)
    if results is None:
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No se encontró el trabajo de entrenamiento.")
        raise HTTPException(status_code=409, detail="Las predicciones aún no están disponibles.")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["date", "actual", "predicted"])
    writer.writeheader()
    writer.writerows(results["predictions"])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="predicciones-{job_id[:8]}.csv"'},
    )
