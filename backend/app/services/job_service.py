import threading
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.storage import Dataset
from app.schemas import TrainRequest
from app.services.data_service import read_dataset
from app.services.training_service import TrainingCancelled, train_model


logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lstm-training")

    def start(self, dataset: Dataset, request: TrainRequest) -> str:
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "dataset_id": dataset.id,
            "status": "queued",
            "current_epoch": 0,
            "total_epochs": request.epochs,
            "train_loss": None,
            "validation_loss": None,
            "loss_history": [],
            "validation_loss_history": [],
            "error": None,
            "results": None,
            "cancel_requested": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, dataset, request)
        return job_id

    def _run(self, job_id: str, dataset: Dataset, request: TrainRequest) -> None:
        with self._lock:
            if self._jobs[job_id]["status"] == "cancelled":
                return
            self._jobs[job_id]["status"] = "running"
        try:
            frame = read_dataset(dataset.path)

            def update_progress(values: dict) -> None:
                with self._lock:
                    job = self._jobs[job_id]
                    job.update(values)

            def should_stop() -> bool:
                with self._lock:
                    return bool(self._jobs[job_id]["cancel_requested"])

            results = train_model(frame, request, update_progress, should_stop)
        except TrainingCancelled:
            with self._lock:
                self._jobs[job_id]["status"] = "cancelled"
            return
        except (ValueError, RuntimeError) as exc:
            with self._lock:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = str(exc)
            return
        except (OSError, pd.errors.ParserError) as exc:
            with self._lock:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = f"No se pudo leer el archivo para entrenar: {exc}"
            return
        except Exception:
            logger.exception("Unexpected training failure for job %s", job_id)
            with self._lock:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = (
                    "Ocurrió un error inesperado durante el entrenamiento. "
                    "Consulta los registros del servidor."
                )
            return
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "completed"
            job["results"] = results

    def cancel(self, job_id: str) -> dict | None:
        """Request cancellation. Queued jobs are cancelled at once; running jobs stop after the current epoch."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job["status"] in {"queued", "running"}:
                job["cancel_requested"] = True
                if job["status"] == "queued":
                    job["status"] = "cancelled"
            return {key: value for key, value in job.items() if key != "results"}

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {key: value for key, value in job.items() if key != "results"}

    def results(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] != "completed":
                return None
            return job["results"]


job_manager = JobManager()
