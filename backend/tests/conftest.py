"""Shared saved-artifact fixtures. Nothing here trains or rewrites a model."""
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def backend_root():
    return BACKEND


@pytest.fixture(scope="session")
def saved_artifact():
    return joblib.load(BACKEND / "models" / "credit_default_model_v1.joblib")


@pytest.fixture
def raw_customer():
    """The original API's synthetic functional example, not an evaluation set."""
    row = {
        "LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2,
        "MARRIAGE": 1, "AGE": 30,
    }
    row.update({f"PAY_{month}": 0 for month in [0, 2, 3, 4, 5, 6]})
    row.update({f"BILL_AMT{month}": 20000 for month in range(1, 7)})
    row.update({f"PAY_AMT{month}": 5000 for month in range(1, 7)})
    return row


@pytest.fixture(scope="session")
def app_client():
    from app.main import create_app

    with TestClient(create_app(BACKEND)) as client:
        yield client


@pytest.fixture(scope="session")
def labeled_uci_sample():
    """Actual labels from restored UCI rows; this is a functionality fixture only."""
    return pd.read_csv(BACKEND / "data" / "processed" / "UCI_Credit_Card.csv").head(120)
