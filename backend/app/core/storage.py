import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.config import DATASET_TTL_HOURS, STORAGE_DIR


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    id: str
    filename: str
    path: Path
    size_bytes: int
    row_count: int
    column_count: int
    columns: list[dict]
    null_count: int
    duplicate_count: int
    created_at: float


class DatasetStore:
    def __init__(self) -> None:
        self._datasets: dict[str, Dataset] = {}
        self._lock = threading.RLock()
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    def add(self, dataset: Dataset) -> None:
        with self._lock:
            self._datasets[dataset.id] = dataset

    def get(self, dataset_id: str) -> Dataset | None:
        with self._lock:
            return self._datasets.get(dataset_id)

    def delete(self, dataset_id: str) -> bool:
        with self._lock:
            dataset = self._datasets.pop(dataset_id, None)
            if dataset is None:
                return False
            dataset.path.unlink(missing_ok=True)
            return True

    def cleanup_expired(self) -> None:
        cutoff = time.time() - DATASET_TTL_HOURS * 3600
        with self._lock:
            expired = [
                dataset_id
                for dataset_id, dataset in self._datasets.items()
                if dataset.created_at < cutoff
            ]
            for dataset_id in expired:
                dataset = self._datasets.pop(dataset_id)
                dataset.path.unlink(missing_ok=True)
            for path in STORAGE_DIR.iterdir():
                if path.suffix.lower() not in {".csv", ".xlsx"}:
                    continue
                try:
                    if path.is_file() and path.stat().st_mtime < cutoff:
                        path.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Could not remove expired dataset file %s", path.name)

    def create_path(self, suffix: str) -> tuple[str, Path]:
        dataset_id = uuid.uuid4().hex
        return dataset_id, STORAGE_DIR / f"{dataset_id}{suffix}"


dataset_store = DatasetStore()
