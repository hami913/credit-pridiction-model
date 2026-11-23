# Complete delivered folder structure

```text
Integrated_Credit_Default_System/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .streamlit/
│   └── config.toml
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── prediction.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── prediction_service.py
│   │   ├── __init__.py
│   │   └── main.py
│   ├── config/
│   │   └── input_features.json
│   ├── data/
│   │   ├── processed/
│   │   │   └── UCI_Credit_Card.csv
│   │   ├── raw/
│   │   │   └── UCI_Credit_Card.xls
│   │   ├── dataset_provenance.json
│   │   └── README.md
│   ├── models/
│   │   ├── credit_default_model_v1.joblib
│   │   └── model_metadata.json
│   ├── notebooks/
│   │   ├── 01_Credit_Default_Inference.ipynb
│   │   ├── Credit_default_pridiction_model_final_project.ipynb
│   │   └── README.md
│   ├── reports/
│   │   ├── verification/
│   │   │   ├── backend_pytest.txt
│   │   │   ├── backend_pytest.xml
│   │   │   ├── frontend_pytest.txt
│   │   │   ├── frontend_pytest.xml
│   │   │   ├── inference_diagnostics.json
│   │   │   ├── notebook_execution.json
│   │   │   ├── script_checks.json
│   │   │   └── verification_summary.json
│   │   └── credit_risk_regulatory_report.json
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── audit_encoder_categories.py
│   │   ├── import_historical_mlflow.py
│   │   ├── log_inference_mlflow.py
│   │   ├── README.md
│   │   ├── train_and_track_mlflow.py
│   │   └── verify_real_inference.py
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── evaluate.py
│   │   ├── explainability.py
│   │   ├── fairness.py
│   │   ├── feature_engineering.py
│   │   ├── monitoring.py
│   │   ├── preprocessing.py
│   │   ├── reporting.py
│   │   ├── risk_scoring.py
│   │   └── train.py
│   ├── tests/
│   │   ├── fixtures/
│   │   │   ├── original_baseline.json
│   │   │   └── original_baseline.npz
│   │   ├── conftest.py
│   │   ├── test_api.py
│   │   ├── test_architecture.py
│   │   ├── test_category_mapping.py
│   │   ├── test_evaluate.py
│   │   ├── test_explainability.py
│   │   ├── test_fairness.py
│   │   ├── test_features.py
│   │   ├── test_model.py
│   │   ├── test_monitoring.py
│   │   ├── test_original_parity.py
│   │   ├── test_reporting.py
│   │   ├── test_risk_scoring.py
│   │   ├── test_service.py
│   │   └── test_train.py
│   ├── .env.example
│   ├── pytest.ini
│   ├── requirements-dev.txt
│   ├── requirements-research.txt
│   └── requirements.txt
├── docs/
│   ├── ARCHITECTURE.md
│   ├── FILE_MAPPING.csv
│   ├── FILE_MAPPING.json
│   ├── FILE_MAPPING.md
│   ├── FOLDER_STRUCTURE.md
│   ├── REMOVED_FILES.md
│   ├── SOURCE_INTEGRITY.json
│   └── TEST_RESULTS.md
├── frontend/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── analytics.py
│   │   ├── batch.py
│   │   ├── common.py
│   │   └── ui.py
│   ├── pages/
│   │   ├── 1_Batch_Assessment.py
│   │   ├── 2_Model_Information.py
│   │   └── 3_Analytics.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── api.py
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_api_client.py
│   │   ├── test_live_integration.py
│   │   └── test_streamlit.py
│   ├── .env.example
│   ├── __init__.py
│   ├── app.py
│   ├── pytest.ini
│   └── requirements.txt
├── scripts/
│   └── verify_project.py
├── .gitignore
├── PACKAGE_MANIFEST.json
├── README.md
├── requirements-dev.txt
└── requirements.txt
```

The ZIP contains this single top-level project directory. Required artifacts, data, tests and evidence are included.
