"""Independent regression against predictions from the pristine uploaded source.

The fixture was generated before integrating the original inference code. It
contains all 30,000 restored UCI rows, not fitted or invented model metrics.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.prediction_service import PredictionService


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def original_baseline():
    metadata = json.loads((FIXTURES / "original_baseline.json").read_text())
    with np.load(FIXTURES / "original_baseline.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    return metadata, arrays


@pytest.fixture(scope="module")
def integrated_service():
    service = PredictionService(ROOT)
    service.load()
    return service


def test_original_model_and_source_bytes_preserved(original_baseline):
    metadata, _ = original_baseline
    assert digest(ROOT / "models/credit_default_model_v1.joblib") == metadata["model_sha256"]
    for relative_path, expected in metadata["source_module_sha256"].items():
        assert digest(ROOT / relative_path) == expected, relative_path
    assert digest(ROOT / "config/input_features.json") == metadata["raw_features_sha256"]


def test_baseline_and_dataset_integrity(original_baseline):
    metadata, saved = original_baseline
    assert digest(FIXTURES / "original_baseline.npz") == metadata["baseline_sha256"]
    assert digest(ROOT / metadata["dataset_file"]) == metadata["dataset_sha256"]
    assert metadata["training_performed"] is False
    assert len(saved["customer_ids"]) == metadata["rows"] == 30000


def test_all_30000_predictions_match_pristine_original(original_baseline, integrated_service):
    metadata, saved = original_baseline
    raw = pd.read_csv(ROOT / metadata["dataset_file"])
    np.testing.assert_array_equal(raw["ID"], saved["customer_ids"])

    prepared = integrated_service.prepare(raw)
    assert prepared.shape == (30000, 38)
    assert prepared.columns.tolist() == metadata["feature_order"]
    np.testing.assert_array_equal(prepared.columns, saved["feature_names"])
    np.testing.assert_allclose(
        prepared.to_numpy(dtype=float), saved["engineered_features"], rtol=0, atol=1e-12
    )
    transformed = integrated_service.pipeline.named_steps["preprocessor"].transform(prepared)
    assert transformed.shape == (30000, 41)

    probability = integrated_service.predict_probabilities(raw)
    np.testing.assert_allclose(
        probability, saved["default_probability"], rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(
        (probability >= metadata["threshold"]).astype(int), saved["prediction"]
    )
    assert integrated_service.threshold == metadata["threshold"] == 0.5
