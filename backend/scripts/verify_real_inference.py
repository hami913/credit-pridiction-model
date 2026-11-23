"""Run the original inference verification through the central service.

From backend/: python -m scripts.verify_real_inference [--csv PATH]
This command does not fit a model or calculate independent test performance.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.prediction_service import PredictionService


ROOT = Path(__file__).resolve().parents[1]


def verify(csv_path, output_path=None):
    model_path = ROOT / "models/credit_default_model_v1.joblib"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    service = PredictionService(ROOT)
    service.load()
    data = pd.read_csv(csv_path)
    if data.empty:
        raise ValueError("Dataset must contain at least one customer.")
    prepared = service.prepare(data)
    probability = service.predict_probabilities(data)
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Model returned invalid probabilities.")
    predicted = (probability >= service.threshold).astype(int)
    report = {
        "verification_type": "inference_only",
        "model_version": str(service.model_version),
        "model_sha256": model_hash,
        "dataset_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "total_records": int(len(data)),
        "input_features": int(prepared.shape[1]),
        "threshold": float(service.threshold),
        "average_default_probability": float(probability.mean()),
        "minimum_probability": float(probability.min()),
        "maximum_probability": float(probability.max()),
        "predicted_default_count": int(predicted.sum()),
        "predicted_nondefault_count": int((predicted == 0).sum()),
        "independent_test_performance": "not_evaluated",
        "training_performed": False,
    }
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != model_hash:
        raise RuntimeError("Model artifact changed during read-only verification.")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data/processed/UCI_Credit_Card.csv")
    parser.add_argument("--output", type=Path, help="New report path; existing files are protected")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or ROOT / "reports" / f"real_data_inference_{stamp}.json"
    report = verify(args.csv, output)
    print(json.dumps(report, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
