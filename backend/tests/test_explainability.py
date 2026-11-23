from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.explainability import explain_default_prediction
from src.preprocessing import prepare_model_input


@pytest.fixture(scope="module")
def sample_pipeline():
    # Exercise the shipped model. No synthetic fitting or artifact writes.
    root = Path(__file__).resolve().parents[1]
    artifact = joblib.load(root / "models/credit_default_model_v1.joblib")
    raw = pd.read_csv(root / "data/processed/UCI_Credit_Card.csv", nrows=1)
    prepared = prepare_model_input(raw, artifact["features"])
    return artifact["pipeline"], prepared


def test_returns_explanations(sample_pipeline):
    model, X = sample_pipeline
    result = explain_default_prediction(model, X)

    assert len(result) == 10
    assert "shap_value" in result.columns
    assert np.isfinite(result["shap_value"]).all()


def test_top_n(sample_pipeline):
    model, X = sample_pipeline
    result = explain_default_prediction(model, X, top_n=1)

    assert len(result) == 1


def test_rejects_batch(sample_pipeline):
    model, X = sample_pipeline

    with pytest.raises(ValueError):
        explain_default_prediction(
            model, pd.concat([X, X])
        )


def test_invalid_top_n(sample_pipeline):
    model, X = sample_pipeline

    with pytest.raises(ValueError):
        explain_default_prediction(model, X, top_n=0)


def test_missing_pipeline_steps(sample_pipeline):
    _, X = sample_pipeline

    with pytest.raises(ValueError):
        explain_default_prediction(object(), X)
