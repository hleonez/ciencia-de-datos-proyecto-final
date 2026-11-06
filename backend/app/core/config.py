import os
from pathlib import Path


MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
DATASET_TTL_HOURS = int(os.getenv("DATASET_TTL_HOURS", "24"))
STORAGE_DIR = Path(os.getenv("DATA_STORAGE_DIR", Path(__file__).resolve().parents[2] / "data"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
