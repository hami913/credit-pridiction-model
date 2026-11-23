"""Backend paths and serving limits; independent of the process working directory."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env", override=False)
_configured_root = Path(os.getenv("CREDIT_ARTIFACT_DIR") or BACKEND_DIR)
ROOT = (_configured_root if _configured_root.is_absolute() else BACKEND_DIR / _configured_root).resolve()
RAW_FEATURES = json.loads((BACKEND_DIR / "config/input_features.json").read_text(encoding="utf-8"))
TARGET = "default.payment.next.month"
MODEL_FILENAME = "credit_default_model_v1.joblib"
REQUIRED_SKLEARN_VERSION = "1.6.1"
MAX_BATCH_ROWS = 50_000
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
INTEGER_FEATURES = {"SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"}


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
